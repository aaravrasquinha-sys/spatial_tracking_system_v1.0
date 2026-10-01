"""
GTSAM batch pose-graph backend. Primary backend on the target machine
(GTSAM is already installed there). Falls back automatically to
backend_native if gtsam cannot be imported, so development/CI on a
machine without GTSAM still works -- see posegraph.py.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core.types import Link
from pyslam.core import lie

try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False


def _permute_info_rho_phi_to_phi_rho(info: np.ndarray) -> np.ndarray:
    """Reorder a 6x6 information matrix from this codebase's [rho(3),
    phi(3)] (translation-first) convention to GTSAM's Pose3 tangent
    convention, [phi(3), rho(3)] (rotation-first).

    Phase 0's backend_gtsam.py passed the [rho,phi]-ordered information
    matrix straight into gtsam.noiseModel.Gaussian.Information() and
    argued in a comment that a full 6x6 matrix sidesteps any ordering
    question. It doesn't: Gaussian.Information(M) tells GTSAM "the
    information matrix of MY 6-vector is M", and GTSAM's own 6-vector for
    a Pose3 BetweenFactor is [phi,rho], not [rho,phi]. Passing M unpermuted
    tells GTSAM the WRONG information is attached to each component --
    harmless while M is a scaled identity (isotropic), silently
    mis-weighting the optimisation the moment a link gets an anisotropic
    covariance (which WP-B2 introduces). This performs the permutation
    P such that P @ M @ P.T is M re-expressed in [phi,rho] order, via a
    single block swap (both diagonal blocks AND the off-diagonal
    coupling block, transposed correctly) -- NOT just swapping the two
    diagonal 3x3 blocks, which would silently drop the rho-phi coupling
    terms whenever they're nonzero.
    """
    P = np.zeros((6, 6))
    P[0:3, 3:6] = np.eye(3)   # new phi block <- old phi block (was rows/cols 3:6)
    P[3:6, 0:3] = np.eye(3)   # new rho block <- old rho block (was rows/cols 0:3)
    return P @ info @ P.T


def _T_to_pose3(T: np.ndarray):
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(T[0, 3], T[1, 3], T[2, 3])
    return gtsam.Pose3(R, t)


def _pose3_to_T(p) -> np.ndarray:
    M = p.matrix()
    return np.array(M, dtype=np.float64)


def _info_to_noise(info: np.ndarray, robust_delta: Optional[float] = None,
                    kernel: str = "huber", dcs_xi: float = 6.0):
    """info is in this codebase's [rho(3), phi(3)] order; GTSAM's Pose3
    tangent is [phi(3), rho(3)]. Permute before handing it to GTSAM --
    see _permute_info_rho_phi_to_phi_rho's docstring for why the earlier
    "pass the full matrix, ordering doesn't matter" argument was wrong.

    WP-T0: kernel selects Huber (default) or DCS, mirroring
    backend_native.py's cfg.loop_robust_kernel -- previously this
    backend always used Huber regardless of that config field, so
    cfg.loop_robust_kernel="dcs" (validated only via the native backend
    in WP-P4's gate) silently had no effect on the GTSAM backend this
    project's own target hardware actually runs."""
    info_gtsam_order = _permute_info_rho_phi_to_phi_rho(info)
    noise = gtsam.noiseModel.Gaussian.Information(info_gtsam_order)
    if robust_delta is not None:
        if kernel == "dcs":
            robust = gtsam.noiseModel.mEstimator.DCS(dcs_xi)
        else:
            robust = gtsam.noiseModel.mEstimator.Huber(robust_delta)
        noise = gtsam.noiseModel.Robust(robust, noise)
    return noise


class GtsamBackend:
    def __init__(self, huber_delta: float = 1.0, robust_kernel: str = "huber", dcs_xi: float = 6.0):
        if not GTSAM_AVAILABLE:
            raise RuntimeError("gtsam is not importable")
        self.huber_delta = huber_delta
        self.robust_kernel = robust_kernel  # WP-T0: "huber" | "dcs", parity with backend_native.py
        self.dcs_xi = dcs_xi
        self._poses: dict[int, np.ndarray] = {}
        self._links: list[Link] = []

    def add_node(self, id: int, pose: np.ndarray) -> None:
        self._poses[id] = pose.copy()

    def add_link(self, link: Link) -> None:
        self._links.append(link)

    def optimize(self, fixed: list[int]) -> dict[int, np.ndarray]:
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        node_ids = list(self._poses.keys())

        for nid in node_ids:
            values.insert(nid, _T_to_pose3(self._poses[nid]))

        if not fixed:
            raise ValueError("optimize() requires at least one fixed (gauge) node")
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
        for nid in fixed:
            graph.add(gtsam.PriorFactorPose3(nid, _T_to_pose3(self._poses[nid]), prior_noise))

        for link in self._links:
            if link.a not in self._poses or link.b not in self._poses:
                continue
            # WP-T0 (GTSAM-parity fix): backend_native.py already
            # robustifies 'proximity' links the same as 'loop' (both are
            # geometry/appearance-triggered candidates verified through
            # the same GeometricVerifier, unlike 'odom'/'bridge' which
            # are direct or deliberately-weak-by-construction measurements
            # -- see that module's own comment). This backend previously
            # only Huber'd 'loop', silently leaving proximity links
            # un-robustified on whichever machine actually has GTSAM
            # installed (this project's target hardware).
            # WP-LIVE: local_reloc gets the same robust-kernel parity
            # native backend now has -- see backend_native.py's comment.
            robust_delta = self.huber_delta if link.kind in ("loop", "proximity", "local_reloc") else None
            noise = _info_to_noise(link.info, robust_delta, kernel=self.robust_kernel, dcs_xi=self.dcs_xi)
            # se3_log convention is [rho, phi]; se3_exp/se3_log both use
            # this order consistently, and _T_to_pose3/pose3 use full 4x4
            # matrices directly, so no ordering translation is needed here
            # -- we pass link.T_ab as a full transform, not as a tangent
            # vector, sidestepping the [rho,phi] vs [phi,rho] ambiguity.
            graph.add(gtsam.BetweenFactorPose3(link.a, link.b, _T_to_pose3(link.T_ab), noise))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(200)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        result = optimizer.optimize()

        out = {}
        for nid in node_ids:
            out[nid] = _pose3_to_T(result.atPose3(nid))
        self._poses = {i: out[i].copy() for i in node_ids}
        return out

    def get_poses(self) -> dict[int, np.ndarray]:
        return {i: p.copy() for i, p in self._poses.items()}

    def set_poses(self, poses: dict[int, np.ndarray]) -> None:
        self._poses = {i: T.copy() for i, T in poses.items()}

    def pop_link(self) -> None:
        if self._links:
            self._links.pop()
