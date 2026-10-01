"""
WP-LIVE graph: incremental (iSAM2) pose-graph backend.

Replaces the two-tier online/finalize split (pipeline.py's _accept_link
doing a bounded WM-only optimize() during the run, then a separate
full-graph Pipeline.finalize() batch solve at shutdown) with ONE
incremental solve that is always current. This is required by "map
should be made in real time only" -- there is no shutdown-time batch
step left to lean on, and it also sidesteps WP-M's still-unresolved
finalize() regression (backend_native.py's warm-started `x0` behaviour)
by construction: iSAM2 IS a principled incremental solver, not a
batch solver being asked to behave like one.

Reuses backend_gtsam.py's already-validated helpers (_T_to_pose3,
_pose3_to_T, _info_to_noise, and critically
_permute_info_rho_phi_to_phi_rho -- the same rho/phi ordering bug class
this project's culture explicitly guards against) rather than
reimplementing them, so a fix to that permutation only has one place to
live.

Honest status, same discipline as every other WP-* findings note in
this repo: GTSAM is not importable in the sandbox this was developed
in (no hardware, no GTSAM package -- see the repo's own README_ORIN.md
for why GTSAM is built from source on the target Orin only). Every
function here is straightforward composition of already-oracle-tested
primitives (backend_gtsam.py's helpers, already used in the existing,
passing GTSAM backend) but this module's OWN incremental-update
behavior -- in particular, that repeated partial updates converge to
the same answer a batch solve would -- has NOT been run end-to-end on
this machine. Treat GTSAM_AVAILABLE=False everywhere in this sandbox as
expected, not as a signal this code is broken; run
tests/gates/test_g_isam2.py (added alongside this module, SKIPPED here
for the same reason) on the target hardware before trusting it.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core.types import Link
from pyslam.graph.backend_gtsam import (
    GTSAM_AVAILABLE, _T_to_pose3, _pose3_to_T, _info_to_noise,
)

if GTSAM_AVAILABLE:
    import gtsam


class Isam2Backend:
    """Same Protocol as NativeBackend/GtsamBackend (add_node, add_link,
    optimize, get_poses, set_poses, pop_link) PLUS one addition,
    update_incremental(), which the live backend PROCESS calls once per
    keyframe instead of optimize(). optimize() itself is kept (calls
    update_incremental() on whatever is new, then returns the current
    estimate) purely so this backend is a drop-in replacement anywhere
    a PoseGraph.optimize() call already exists (lockstep mode, gates
    written against the existing interface).
    """

    def __init__(self, huber_delta: float = 1.0, robust_kernel: str = "huber", dcs_xi: float = 6.0,
                 relinearize_threshold: float = 0.1, relinearize_skip: int = 1):
        if not GTSAM_AVAILABLE:
            raise RuntimeError("gtsam is not importable")
        self.huber_delta = huber_delta
        self.robust_kernel = robust_kernel
        self.dcs_xi = dcs_xi
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(relinearize_threshold)
        params.relinearizeSkip = relinearize_skip
        self._isam = gtsam.ISAM2(params)
        self._poses: dict[int, np.ndarray] = {}       # current best estimate, all nodes ever added
        self._links: list[Link] = []
        self._added_node_ids: set[int] = set()         # already inserted into iSAM2's Values at least once
        self._added_link_count = 0                     # how many of self._links are already in the factor graph
        self._fixed: set[int] = set()

    # -- Protocol-compatible surface --------------------------------
    def add_node(self, id: int, pose: np.ndarray) -> None:
        self._poses[id] = pose.copy()

    def add_link(self, link: Link) -> None:
        self._links.append(link)

    def pop_link(self) -> None:
        if self._links:
            self._links.pop()
            # NOTE: iSAM2 cannot un-add a factor already incorporated
            # into an update() call -- unlike the batch backends' pop_link
            # (which just removes it before the NEXT optimize() rebuilds
            # the whole graph from self._links), a link already pushed
            # through update_incremental() stays in iSAM2's factor graph.
            # The live backend process must therefore call pop_link()
            # (e.g. the post-loop-closure NEES rollback, pipeline.py's
            # _try_loop_closure) BEFORE calling update_incremental() for
            # that link, never after -- see RUNBOOK_LIVE.md's own note
            # on this ordering requirement, and
            # tests/gates/test_g_isam2.py::check_rollback_before_not_after
            # (GTSAM-gated, unvalidated in this sandbox) for the intended
            # regression test.
            if self._added_link_count > len(self._links):
                self._added_link_count = len(self._links)

    def get_poses(self) -> dict[int, np.ndarray]:
        return {i: p.copy() for i, p in self._poses.items()}

    def set_poses(self, poses: dict[int, np.ndarray]) -> None:
        # Only meaningful for nodes not yet pushed into iSAM2 (e.g. a
        # restored snapshot before the first update_incremental() call);
        # once a node is in iSAM2's Bayes tree its estimate is owned by
        # the incremental solver, not overwritable from outside without
        # a real iSAM2 "fixLag"/reset operation this module doesn't
        # implement (out of WP-LIVE's approved scope).
        for i, T in poses.items():
            if i not in self._added_node_ids:
                self._poses[i] = T.copy()

    def optimize(self, fixed: Optional[list[int]] = None) -> dict[int, np.ndarray]:
        """Protocol-compatibility shim: push any not-yet-incorporated
        nodes/links through one incremental update, then return the
        current estimate. Prefer calling update_incremental() directly
        from the live backend process (per keyframe) -- this exists so
        lockstep mode and any gate written against the shared
        PoseGraph.optimize() interface work unmodified."""
        if fixed:
            self._fixed.update(fixed)
        self.update_incremental()
        return self.get_poses()

    # -- The real incremental entry point -----------------------------
    def update_incremental(self) -> dict[int, np.ndarray]:
        """Pushes every node not yet in iSAM2's Values and every link
        not yet in its factor graph through ONE gtsam.ISAM2.update()
        call, then reads back calculateEstimate() for every node this
        backend has ever seen. Call once per keyframe (the backend
        process's natural cadence) -- NOT once per frame; iSAM2's own
        cost scales with the size of the update batch, not with total
        graph size, so this stays cheap even as the map grows, which is
        exactly the property a real-time-only mapping run needs that
        the old finalize()-at-shutdown design didn't have to provide.
        """
        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        new_node_ids = [nid for nid in self._poses if nid not in self._added_node_ids]
        for nid in new_node_ids:
            new_values.insert(nid, _T_to_pose3(self._poses[nid]))
            self._added_node_ids.add(nid)

        # gauge: a prior on every "fixed" node not yet anchored. Unlike
        # the batch backends (which re-fix the SAME node set every
        # optimize() call), iSAM2 only needs ONE prior ever, on the
        # very first node -- adding a second prior on an already-well-
        # constrained later node would over-weight the graph. Track
        # which fixed ids already have a prior in self._added_node_ids'
        # bookkeeping implicitly (a node only gets a prior the update
        # it's first inserted).
        for nid in self._fixed:
            if nid in new_node_ids:
                prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
                new_factors.add(gtsam.PriorFactorPose3(nid, _T_to_pose3(self._poses[nid]), prior_noise))

        new_links = self._links[self._added_link_count:]
        for link in new_links:
            if link.a not in self._poses or link.b not in self._poses:
                continue
            robust_delta = self.huber_delta if link.kind in ("loop", "proximity", "local_reloc") else None
            noise = _info_to_noise(link.info, robust_delta, kernel=self.robust_kernel, dcs_xi=self.dcs_xi)
            new_factors.add(gtsam.BetweenFactorPose3(link.a, link.b, _T_to_pose3(link.T_ab), noise))
        self._added_link_count = len(self._links)

        if new_values.size() > 0 or new_factors.size() > 0:
            self._isam.update(new_factors, new_values)
            estimate = self._isam.calculateEstimate()
            for nid in self._added_node_ids:
                self._poses[nid] = _pose3_to_T(estimate.atPose3(nid))
        return self.get_poses()

    def marginal_covariance(self, node_id: int) -> Optional[np.ndarray]:
        """6x6 marginal covariance for one node, in this codebase's
        [rho,phi] order (inverse-permuted back from GTSAM's [phi,rho] --
        see backend_gtsam.py's _permute_info_rho_phi_to_phi_rho, whose
        permutation matrix is its own inverse since it's a pure block
        swap). Used by M2's calibration fusion (planning notes section
        C2) to get an honest, always-current covariance instead of a
        one-off Hessian computed outside the graph. None if the node
        isn't in the current Bayes tree yet."""
        if node_id not in self._added_node_ids:
            return None
        from pyslam.graph.backend_gtsam import _permute_info_rho_phi_to_phi_rho
        cov_gtsam_order = self._isam.marginalCovariance(node_id)
        # the permutation matrix P satisfies P @ P.T = I (pure swap), so
        # applying it again inverts it: cov_rho_phi = P @ cov_phi_rho @ P.T
        return _permute_info_rho_phi_to_phi_rho(np.asarray(cov_gtsam_order))
