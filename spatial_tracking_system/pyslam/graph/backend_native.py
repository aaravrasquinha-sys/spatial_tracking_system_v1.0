"""
Native pose-graph backend. No external optimizer dependency: always
available, fully inspectable, and used as the GTSAM cross-check.

Parameterization: each unknown pose is represented as base_pose @
exp(xi), xi in R^6, and the whole graph is optimised in one
scipy.optimize.least_squares call over the concatenated tangent-space
perturbation vector. This is the standard manifold-NLLS approach
(equivalent in spirit to what g2o/GTSAM/Ceres do per linearisation);
scipy's internal Levenberg-Marquardt iterations handle the nonlinearity,
and se3_exp is a valid global retraction so large perturbations are not
a correctness problem, only a conditioning one for extreme cases.

F5 fix (was: dense finite-difference Jacobians, O(n^2)-ish and the
dominant runtime cost -- see SYSTEM_SUMMARY.md section 5/6): every
factor residual depends on exactly 2 of the N unknown poses (12 of the
6N columns), so a dense Jacobian is >99% structural zero once the graph
has more than a couple dozen nodes, and scipy was both re-deriving those
zeros via finite differences every iteration AND handing them to a dense
linear solve. We fix this by telling `least_squares` the sparsity
pattern via `jac_sparsity` for large graphs: scipy's finite-difference
estimator then uses graph colouring to batch non-conflicting columns
into far fewer function evaluations (this computes the SAME
finite-difference values the dense path would, not an approximation of
them -- the win is fewer redundant evaluations, not lower precision).

This is deliberately gated by size (`_SPARSE_THRESHOLD_VARS` below), not
applied unconditionally, based on measurement rather than the a priori
assumption that sparsity always wins:
  - On corridor_v2 (the fixture F5 was originally diagnosed against,
    89 keyframes / up to 534 vars), unconditional sparse cuts wall time
    from 949 to 414 ms/frame with drift/ATE numbers matching the dense
    path to full float precision -- the actual target case, a clear win.
  - On the smaller end-to-end fixtures (~25-45 keyframes, <=270 vars --
    square6dof/room_orbit/aliasing_rooms/static_60s and the selftest
    loop-closure gate all fall in this range), unconditional sparse was
    measured to be 2-3x SLOWER per optimize() call than the original
    dense path, not faster. `jac_sparsity` forces scipy onto the
    iterative `lsmr` trust-region solver (scipy disallows combining
    sparse Jacobians with the dense `exact` solver), and these graphs'
    heuristic, widely-varying info-matrix weights (inlier-count-scaled,
    see odometry.py) make for exactly the ill-conditioned regime where
    an iterative Krylov solver needs many more inner iterations than a
    direct dense SVD solve -- a real, data-dependent effect, confirmed
    by direct instrumentation, not something a bigger sparsity win at
    that scale would paper over.
The threshold below sits between these two measured regimes so every
currently-existing small fixture keeps the fast, well-tested dense path
unchanged, and only corridor_v2-scale graphs (whose loop-closure events
were observed to fire late, i.e. only once the graph is already near
its final size, not while still small) take the sparse path. Full
methodology and numbers in `WP_B_F5_Findings.md`.

A fully analytic (closed-form) SE(3) Jacobian would remove the
remaining FD-evaluation cost on the sparse path too, but was judged
higher-risk to hand-derive correctly (the SE(3) right-Jacobian has a
non-trivial rho/phi coupling block) for an incremental win once the
dominant O(n^2) cost is already gone at the scale that mattered; left
as a possible future refinement, not attempted here.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import cholesky
from scipy.sparse import lil_matrix

from pyslam.core.types import Link
from pyslam.core import lie

# See module docstring for the measurement behind this number: below it,
# a direct dense solve beats sparse+lsmr on every currently-existing
# small fixture; at/above it, corridor_v2-scale graphs win big from
# sparsity. Expressed in unknown POSES (not vars = 6x this) since that's
# the natural unit fixture sizes are described in.
_SPARSE_THRESHOLD_POSES = 60


class NativeBackend:
    def __init__(self, huber_delta: float = 1.0, robust_kernel: str = "huber", dcs_xi: float = 6.0):
        self.huber_delta = huber_delta
        self.robust_kernel = robust_kernel  # "huber" (default, unchanged) | "dcs" (WP-P4)
        self.dcs_xi = dcs_xi
        self._poses: dict[int, np.ndarray] = {}
        self._links: list[Link] = []

    def add_node(self, id: int, pose: np.ndarray) -> None:
        self._poses[id] = pose.copy()

    def add_link(self, link: Link) -> None:
        self._links.append(link)

    def _factor_residual(self, T_a: np.ndarray, T_b: np.ndarray, link: Link) -> np.ndarray:
        T_ab_pred = lie.se3_inverse(T_a) @ T_b
        err = lie.se3_log(lie.se3_inverse(link.T_ab) @ T_ab_pred)  # (6,)
        L = cholesky(link.info, lower=False)  # L^T L = info
        w = L @ err
        # WP-T0 (GTSAM-parity fix): proximity links (WP-P4) are
        # geometry-triggered, not appearance-verified, and go through
        # the exact same GeometricVerifier as loop closures -- they
        # deserve the same robustness treatment, not the bare
        # (un-robustified) treatment "everything that isn't kind=='loop'"
        # used to give them. 'odom' and 'bridge' stay un-robustified
        # deliberately: odom is a direct measurement (robustifying it
        # would let the optimiser silently discount real motion), and
        # 'bridge' is already given a near-zero information weight at
        # construction (see Config.bridge_link_info_scale) rather than a
        # robust kernel -- the two are different ways of saying
        # "trust this hardly at all" and shouldn't be stacked.
        # WP-LIVE: local_reloc links (loop/local_reloc.py) get the same
        # robust-kernel treatment as loop/proximity links -- despite
        # passing hard inlier/ratio/RMS gates before acceptance, a
        # local_reloc is still a single verified match, not a mutually-
        # confirmed one the way the frozen-map relocalizer's cross-
        # candidate consensus requires; robustifying it costs nothing
        # when it's right and protects the graph if it's ever wrong.
        if link.kind in ("loop", "proximity", "local_reloc"):
            if self.robust_kernel == "dcs":
                # WP-P4: Dynamic Covariance Scaling (Agarwal et al.,
                # "Robust Map Optimization using Dynamic Covariance
                # Scaling", 2013), extended to a 6-DoF factor with
                # dcs_xi = the factor's DOF (6) -- the expected
                # chi-square value for a genuinely correct factor,
                # so a factor sitting at its expected residual is
                # untouched (s=1) and only factors with residual well
                # beyond what a CORRECT loop closure would ever produce
                # get scaled down. Unlike the static Huber kernel this
                # replaces when selected, DCS's scale factor asymptotes
                # toward 0 (not toward a constant linear slope) as
                # chi2->inf, closer to actually "switching off" a
                # confidently-wrong loop closure rather than merely
                # down-weighting its pull. Computed fresh from THIS
                # call's residual, which is the CURRENT estimate during
                # optimisation -- least_squares calls this function at
                # every trial iterate, so this naturally behaves as an
                # iteratively-reweighted scheme with no separate outer
                # loop needed.
                chi2 = float(w @ w)
                s = min(1.0, 2.0 * self.dcs_xi / (self.dcs_xi + chi2)) if chi2 > 0 else 1.0
                w = w * np.sqrt(s)
            else:
                n = np.linalg.norm(w)
                if n > self.huber_delta and n > 1e-12:
                    w = w * np.sqrt(self.huber_delta / n)
        return w

    def optimize(self, fixed: list[int]) -> dict[int, np.ndarray]:
        node_ids = list(self._poses.keys())
        unknown_ids = [i for i in node_ids if i not in fixed]
        if len(unknown_ids) == 0:
            return {i: self._poses[i].copy() for i in node_ids}
        idx_of = {nid: k for k, nid in enumerate(unknown_ids)}
        base = {i: self._poses[i].copy() for i in node_ids}
        n_vars = 6 * len(unknown_ids)

        # Same membership test resfun's per-call loop used to make
        # ("link.a not in poses"), hoisted out since it doesn't depend on
        # x: poses always has exactly node_ids as keys (unpack fills
        # every node_id, fixed or not), so this is a one-time filter.
        node_id_set = set(node_ids)
        active_links = [l for l in self._links if l.a in node_id_set and l.b in node_id_set]

        def unpack(x: np.ndarray) -> dict[int, np.ndarray]:
            poses = {}
            for i in node_ids:
                if i in idx_of:
                    xi = x[6 * idx_of[i]: 6 * idx_of[i] + 6]
                    poses[i] = base[i] @ lie.se3_exp(xi)
                else:
                    poses[i] = base[i]
            return poses

        def resfun(x: np.ndarray) -> np.ndarray:
            poses = unpack(x)
            if not active_links:
                return np.zeros(1)
            out = [self._factor_residual(poses[l.a], poses[l.b], l) for l in active_links]
            return np.concatenate(out)

        x0 = np.zeros(n_vars)

        # F5 fix: sparsity pattern for the FD Jacobian, mirroring resfun's
        # row layout exactly (row-block i <-> active_links[i], 6 rows;
        # nonzero columns are only the 6-wide blocks belonging to
        # whichever of link.a/link.b are unknown -- a fixed endpoint
        # contributes no columns since its perturbation is not a
        # variable). This is a *structural* pattern (which entries CAN be
        # nonzero), not a numeric one -- scipy still finite-differences
        # the actual values, just far fewer of them.
        #
        # Only built/used above _SPARSE_THRESHOLD_POSES -- see module
        # docstring. Below threshold we pass jac_sparsity=None, which is
        # scipy's original dense-FD + dense-exact-solve behaviour,
        # exactly as before this fix, on the graphs where that measurably
        # wins.
        use_sparse = len(unknown_ids) >= _SPARSE_THRESHOLD_POSES
        if use_sparse and active_links:
            sparsity = lil_matrix((6 * len(active_links), n_vars), dtype=np.int8)
            for row, link in enumerate(active_links):
                r0 = 6 * row
                if link.a in idx_of:
                    c0 = 6 * idx_of[link.a]
                    sparsity[r0:r0 + 6, c0:c0 + 6] = 1
                if link.b in idx_of:
                    c0 = 6 * idx_of[link.b]
                    sparsity[r0:r0 + 6, c0:c0 + 6] = 1
            sparsity = sparsity.tocsr()
        else:
            sparsity = None  # dense path: either below threshold, or the
            # degenerate no-active-links case where there's nothing to
            # sparsify anyway.

        # method='lm' (MINPACK) requires #residuals >= #variables, which
        # fails early in a run (e.g. right after the very first loop
        # closure, when few links exist yet relative to free poses) --
        # hit exactly this on the corridor-loop fixture. 'trf' has no such
        # restriction and handles the rank-deficient/underdetermined case
        # gracefully (falls back toward the initial guess for unconstrained
        # directions), so it's the safer default here.
        result = least_squares(resfun, x0, method="trf", xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=3000, jac_sparsity=sparsity)
        final = unpack(result.x)
        self._poses = final
        return {i: final[i].copy() for i in node_ids}

    def get_poses(self) -> dict[int, np.ndarray]:
        return {i: p.copy() for i, p in self._poses.items()}

    def set_poses(self, poses: dict[int, np.ndarray]) -> None:
        self._poses = {i: T.copy() for i, T in poses.items()}

    def pop_link(self) -> None:
        if self._links:
            self._links.pop()
