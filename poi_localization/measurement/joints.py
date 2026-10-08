"""
Optional "Part B": lift M3's 2D COCO-17 keypoints to 3D joints in the same
floor/room frame the tracker already uses.

Why this exists: M3 (yolo11n-pose) finds 17 keypoints per person in the
IMAGE; M4 used to keep only the shoulders and head points (for torso depth
and height) and drop the rest. The 3D viewer needs every joint in room
coordinates to draw a moving skeleton.

How a joint is built (same maths as height_estimator / depth_measurement):
  keypoint pixel -> median depth of a small patch -> (undistort) -> deproject
  to camera space -> FloorFrameTransform.apply_point -> room (x, y, z).

Thin limbs are the hard part: a wrist is a few pixels wide, so its depth patch
often lands on the wall behind the person or on a hole (depth == 0). Every
joint depth is therefore compared with the person's torso depth; when it is
missing or off by more than cfg.max_depth_dev_m the joint is placed at torso
depth instead (its left/right and up/down position is still correct, only
its forward/back position becomes approximate) and its confidence is halved.

Output: list of 17 entries in COCO_KEYPOINT_NAMES order, each
[x, y, z, conf] (metres, room frame, mm precision) or None. Returns None for
the whole person if too few joints are usable or the person is too far away.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from poi_localization import geometry
from poi_localization.config import JointsConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_perception.contracts import COCO_KEYPOINT_NAMES, Intrinsics

# Shoulders + hips: the trunk joints used to build a reference depth when the
# caller has no torso depth for this person.
_TRUNK = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _patch_median_m(depth_raw: np.ndarray, u: float, v: float, radius: int, depth_scale: float) -> Optional[float]:
    h, w = depth_raw.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(0, ui - radius), min(w, ui + radius + 1)
    v0, v1 = max(0, vi - radius), min(h, vi + radius + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    patch = depth_raw[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


def lift_joints(
    depth_raw: Optional[np.ndarray],
    keypoints: Dict[str, Tuple[float, float, float]],
    intr: Intrinsics,
    floor_transform: FloorFrameTransform,
    cfg: JointsConfig,
    ref_depth_m: Optional[float] = None,
    distortion_coeffs: Optional[Tuple] = None,
) -> Optional[List[Optional[List[float]]]]:
    """See module docstring. `ref_depth_m` is the person's torso depth (the
    value M4 already computes for the height fallback); if None it is derived
    from the trunk keypoints."""
    if depth_raw is None or not keypoints:
        return None
    h, w = depth_raw.shape[:2]
    radius = max(0, int(cfg.patch_radius_px))

    # 1) which keypoints are trustworthy enough to lift at all
    cand: Dict[str, Tuple[float, float, float]] = {}
    for name in COCO_KEYPOINT_NAMES:
        kp = keypoints.get(name)
        if kp is None:
            continue
        x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
            continue
        if c < cfg.conf_thresh or not (0.0 <= x < w and 0.0 <= y < h):
            continue
        cand[name] = (x, y, c)
    if len(cand) < cfg.min_joints:
        return None

    # 2) raw depth for every candidate
    raw: Dict[str, Optional[float]] = {n: _patch_median_m(depth_raw, x, y, radius, intr.depth_scale) for n, (x, y, _) in cand.items()}

    # 3) reference (torso) depth
    ref = ref_depth_m if (ref_depth_m is not None and ref_depth_m > 0) else None
    if ref is None:
        trunk = [d for n, d in raw.items() if n in _TRUNK and d is not None]
        if not trunk:
            trunk = [d for d in raw.values() if d is not None]
        if not trunk:
            return None
        ref = float(np.median(trunk))
    if ref > cfg.max_range_m or ref < cfg.min_range_m:
        return None

    # 4) build 3D joints
    out: List[Optional[List[float]]] = []
    n_ok = 0
    for name in COCO_KEYPOINT_NAMES:
        if name not in cand:
            out.append(None)
            continue
        x, y, c = cand[name]
        d = raw[name]
        if d is None or abs(d - ref) > cfg.max_depth_dev_m:
            d = ref
            c *= 0.5
        d += cfg.depth_offset_m  # keypoint is on the body surface; the joint centre is slightly deeper
        u, v = geometry.undistort_pixel(x, y, intr, distortion_coeffs)
        p_cam = geometry.deproject_pixel(u, v, d, intr)
        p = floor_transform.apply_point(p_cam)
        if not np.all(np.isfinite(p)):
            out.append(None)
            continue
        out.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3), round(float(min(c, 1.0)), 2)])
        n_ok += 1
    if n_ok < cfg.min_joints:
        return None
    return out
