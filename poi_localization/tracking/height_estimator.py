"""
Section 7: "height ≈ vertical extent from floor-projected footpoint to
the highest reliable keypoint, transformed the same way as the depth
measurement. Smooth it with a simple exponential moving average."

Split into a stateless per-frame estimate (estimate_instantaneous_height)
and a tiny stateful smoother (HeightEMA) so the instantaneous estimate is
independently testable.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from poi_localization import geometry
from poi_localization.config import HeightConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_perception.contracts import Intrinsics

# Priority order of "top of head"-ish keypoints, highest-quality first.
_TOP_KEYPOINT_PRIORITY = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def _patch_median_depth_m(depth_raw: np.ndarray, u: float, v: float, intr: Intrinsics, radius_px: int) -> Optional[float]:
    h, w = depth_raw.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(0, ui - radius_px), min(w, ui + radius_px + 1)
    v0, v1 = max(0, vi - radius_px), min(h, vi + radius_px + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    patch = depth_raw[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * intr.depth_scale


def estimate_instantaneous_height(
    depth_raw: Optional[np.ndarray],
    keypoints: Dict[str, Tuple[float, float, float]],
    intr: Intrinsics,
    floor_transform: FloorFrameTransform,
    cfg: HeightConfig,
    fallback_depth_m: Optional[float] = None,
) -> Optional[float]:
    """Returns a single frame's height estimate in meters, or None if no
    usable top keypoint / depth is available this frame. `fallback_depth_m`
    is typically the same frame's torso median depth (Section 7:
    "transformed the same way as the depth measurement") -- used when the
    top keypoint's own local depth patch has no valid data (common: hair/
    thin structure absorbs IR return more than it does for the torso)."""
    top_px = None
    top_name = None
    for name in _TOP_KEYPOINT_PRIORITY:
        if name not in keypoints:
            continue
        x, y, conf = keypoints[name]
        if conf >= cfg.keypoint_conf_thresh:
            top_px = (x, y)
            top_name = name
            break
    if top_px is None:
        return None

    depth_m = None
    if depth_raw is not None:
        depth_m = _patch_median_depth_m(depth_raw, top_px[0], top_px[1], intr, cfg.top_keypoint_patch_radius_px)
    if depth_m is None:
        depth_m = fallback_depth_m
    if depth_m is None:
        return None

    point_cam = geometry.deproject_pixel(top_px[0], top_px[1], depth_m, intr)
    point_floor = floor_transform.apply_point(point_cam)
    # The keypoint itself (nose/eye/ear) sits below the actual crown of
    # the head -- see HeightConfig.top_keypoint_head_offset_m. Falls back
    # to 0 for an unrecognized name rather than raising, so a config with
    # a trimmed-down offset table still degrades to the old (biased)
    # behavior instead of crashing.
    head_offset_m = cfg.top_keypoint_head_offset_m.get(top_name, 0.0)
    height_m = float(point_floor[2]) + head_offset_m

    if not (cfg.min_height_m <= height_m <= cfg.max_height_m):
        return None
    return height_m


class HeightEMA:
    """Per-track exponential moving average, per Section 7: "it doesn't
    need its own Kalman filter." """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, sample: Optional[float]) -> Optional[float]:
        if sample is None:
            return self.value
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1 - self.alpha) * self.value
        return self.value
