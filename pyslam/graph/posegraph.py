from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core.types import Link
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.graph.backend_native import NativeBackend
from pyslam.graph import backend_gtsam
from pyslam.graph import backend_isam2

log = get_logger("graph.posegraph")


def make_backend(cfg: Config, prefer: str = "auto"):
    """prefer: 'auto' | 'gtsam' | 'native' | 'isam2' (WP-LIVE).

    'isam2' is requested explicitly by the live backend process (see
    pyslam/live/backend_process.py) -- it is NOT part of 'auto', since
    unlike gtsam-vs-native (numerically interchangeable batch solvers),
    isam2 changes the CALL PATTERN a caller must use
    (update_incremental() per keyframe, not a shutdown-time optimize())
    and should only be selected by a caller that actually uses it that
    way. lockstep mode and every existing gate keep using 'auto' and
    get the original gtsam/native batch behaviour unchanged.
    """
    if prefer == "isam2":
        if not backend_gtsam.GTSAM_AVAILABLE:
            raise RuntimeError("isam2 backend requested but gtsam is not importable")
        log.info("Using incremental (iSAM2) pose-graph backend.")
        return backend_isam2.Isam2Backend(huber_delta=cfg.loop_huber_delta,
                                           robust_kernel=cfg.loop_robust_kernel, dcs_xi=cfg.dcs_xi)
    if prefer in ("auto", "gtsam"):
        if backend_gtsam.GTSAM_AVAILABLE:
            try:
                log.info("Using GTSAM pose-graph backend.")
                # WP-T0: robust_kernel/dcs_xi now threaded through to the
                # GTSAM backend too -- see backend_gtsam.py's _info_to_noise
                # docstring for why this was previously native-backend-only.
                return backend_gtsam.GtsamBackend(huber_delta=cfg.loop_huber_delta,
                                                   robust_kernel=cfg.loop_robust_kernel,
                                                   dcs_xi=cfg.dcs_xi)
            except Exception as e:
                log.warning(f"GTSAM backend failed to initialise ({e}); falling back to native.")
        elif prefer == "gtsam":
            raise RuntimeError("gtsam requested but not importable")
    log.info("Using native pose-graph backend.")
    return NativeBackend(huber_delta=cfg.loop_huber_delta,
                          robust_kernel=cfg.loop_robust_kernel, dcs_xi=cfg.dcs_xi)


class PoseGraph:
    def __init__(self, cfg: Config, prefer_backend: str = "auto"):
        self.cfg = cfg
        self.backend = make_backend(cfg, prefer_backend)
        self.node_ids: list[int] = []
        self.links: list[Link] = []

    def add_node(self, node_id: int, pose: np.ndarray) -> None:
        self.backend.add_node(node_id, pose)
        self.node_ids.append(node_id)

    def add_link(self, link: Link) -> None:
        self.backend.add_link(link)
        self.links.append(link)

    def optimize(self, fixed: Optional[list[int]] = None) -> dict[int, np.ndarray]:
        if fixed is None:
            fixed = [self.node_ids[0]] if self.node_ids else []
        return self.backend.optimize(fixed)

    def update_incremental(self) -> dict[int, np.ndarray]:
        """WP-LIVE: passthrough to Isam2Backend.update_incremental() for
        the live backend process's per-keyframe cadence. Raises
        AttributeError if the current backend isn't isam2 -- deliberate
        (a caller using this must have asked for prefer='isam2'
        explicitly; there is no silent fallback to a batch re-solve
        here, since that would defeat the point of calling this instead
        of optimize())."""
        return self.backend.update_incremental()

    def marginal_covariance(self, node_id: int):
        """WP-LIVE: None on any backend that doesn't support it (batch
        backends never did); Isam2Backend overrides with a real answer."""
        fn = getattr(self.backend, "marginal_covariance", None)
        return fn(node_id) if fn is not None else None

    def get_poses(self) -> dict[int, np.ndarray]:
        return self.backend.get_poses()

    def snapshot(self) -> dict[int, np.ndarray]:
        return self.backend.get_poses()

    def restore(self, snapshot: dict[int, np.ndarray]) -> None:
        self.backend.set_poses(snapshot)

    def remove_last_link(self) -> None:
        """Used by the post-optimisation loop-acceptance check (WP-B4):
        an optimisation that made things worse gets its triggering link
        popped from both the backend and the bookkeeping list here."""
        self.backend.pop_link()
        if self.links:
            self.links.pop()
