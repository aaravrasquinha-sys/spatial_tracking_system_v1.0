"""
WP-LIVE N1: live QA stream ("mapping_status"), built from
pyslam.live.tracker_stage.TrackerQaSnapshot entries as they're
produced (real time, one call per frame) so an operator sees a bad
capture WHILE walking, not after locking. Deliberately simple
aggregation (running counts/means) rather than a database -- this is
meant to be cheap enough to update every frame and cheap enough to
serialize to JSON for a debug viewer or WebSocket message every few
frames, mirroring the system plan's health/mapping_status message
conventions (1-5 Hz, always-sent even when nothing changed).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class MappingStatusAggregator:
    n_frames: int = 0
    n_keyframes: int = 0
    n_lost_frames: int = 0
    n_icp_fallback_used: int = 0
    max_speed_mps: float = 0.0
    max_rot_rate_dps: float = 0.0
    n_speed_over_limit: int = 0       # frames above the protocol's ~0.5 m/s guidance
    n_rot_over_limit: int = 0         # frames above the protocol's ~30 deg/s guidance
    fused_depth_valid_frac_last: Optional[float] = None
    _t_start: float = field(default_factory=time.time)

    SPEED_LIMIT_MPS: float = 0.5
    ROT_LIMIT_DPS: float = 30.0

    def add_frame(self, snap) -> None:
        self.n_frames += 1
        if snap.is_keyframe:
            self.n_keyframes += 1
        if snap.odom_status == "LOST":
            self.n_lost_frames += 1
        if snap.used_icp_fallback:
            self.n_icp_fallback_used += 1
        if snap.speed_mps is not None:
            self.max_speed_mps = max(self.max_speed_mps, snap.speed_mps)
            if snap.speed_mps > self.SPEED_LIMIT_MPS:
                self.n_speed_over_limit += 1
        if snap.rot_rate_dps is not None:
            self.max_rot_rate_dps = max(self.max_rot_rate_dps, snap.rot_rate_dps)
            if snap.rot_rate_dps > self.ROT_LIMIT_DPS:
                self.n_rot_over_limit += 1
        if snap.fused_depth_valid_frac is not None:
            self.fused_depth_valid_frac_last = snap.fused_depth_valid_frac

    def snapshot(self, layers=None) -> dict:
        """layers: an optional pyslam.mapping.layers.LayerGrid, folded
        in as coverage stats if given -- kept optional so this module
        has no hard dependency on the dense/layers pipeline being wired
        up yet (e.g. Phase-1 tracker-only testing)."""
        out = {
            "type": "mapping_status",
            "t": time.time(),
            "elapsed_s": time.time() - self._t_start,
            "n_frames": self.n_frames,
            "n_keyframes": self.n_keyframes,
            "n_lost_frames": self.n_lost_frames,
            "n_icp_fallback_used": self.n_icp_fallback_used,
            "max_speed_mps": self.max_speed_mps,
            "max_rot_rate_dps": self.max_rot_rate_dps,
            "n_speed_over_limit": self.n_speed_over_limit,
            "n_rot_over_limit": self.n_rot_over_limit,
            "fused_depth_valid_frac_last": self.fused_depth_valid_frac_last,
        }
        if layers is not None:
            out["coverage"] = layers.coverage_stats()
        return out
