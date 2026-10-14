"""Own, stricter 3D joint lifting for the identity layer (R1 spec section 7.2).

Unlike measurement/joints.py::lift_joints (viewer-oriented, falls back to torso depth), a joint here is
"measured" ONLY if its own depth patch is valid, it belongs to the person, and the keypoint confidence is
>= kp_conf_min. "Belongs to the person" = within max_dev_xy_m of the track's floor position (room frame; invariant to
camera pitch -- a tilted camera sees feet ~0.4 m farther in camera-z than the torso, so a pure |d - d_torso| test
wrongly rejects ankles), falling back to |d - d_torso| <= max_dev_m when no floor position is supplied. Joints that would have fallen back are simply absent -- they are never
used for lengths. Each joint carries its uncertainty and viewing ray.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from poi_localization.identity.room_geometry import RoomGeometry, depth_sigma_m, lateral_sigma_m


@dataclass
class Joint:
    p: np.ndarray            # (3,) room-frame point
    sigma_z: float           # depth noise (m), along the viewing ray
    sigma_lat: float         # lateral noise (m), perpendicular to the viewing ray
    ray: np.ndarray          # unit vector camera -> point (room frame)


def lift_measured_joints(geom: RoomGeometry, depth_m: np.ndarray,
                         keypoints: Dict[str, Tuple[float, float, float]], d_torso: float,
                         kp_conf_min: float = 0.5, kp_pixel_sigma_px: float = 3.0,
                         sigma_disparity_px: float = 0.1, bias_frac: float = 0.008,
                         stereo_fx_px: float = 390.0, max_dev_m: float = 0.35,
                         patch: int = 7, min_valid_px: int = 8,
                         torso_xy: Optional[Tuple[float, float]] = None,
                         max_dev_xy_m: float = 0.5) -> Dict[str, Joint]:
    h, w = depth_m.shape
    half = patch // 2
    out: Dict[str, Joint] = {}
    for name, kp in keypoints.items():
        if kp is None or kp[2] < kp_conf_min:
            continue
        u, v = int(round(kp[0])), int(round(kp[1]))
        if not (half <= u < w - half and half <= v < h - half):
            continue
        win = depth_m[v - half:v + half + 1, u - half:u + half + 1]
        valid = win[np.isfinite(win) & (win > 0)]
        if valid.size < min_valid_px:
            continue
        d = float(np.median(valid))
        p = d * geom.A[v, u].astype(np.float64) + geom.t
        if torso_xy is not None:
            if np.hypot(p[0] - torso_xy[0], p[1] - torso_xy[1]) > max_dev_xy_m:
                continue
        elif abs(d - d_torso) > max_dev_m:
            continue
        ray = p - geom.t
        ray = ray / np.linalg.norm(ray)
        out[name] = Joint(p=p, sigma_z=depth_sigma_m(d, stereo_fx_px, geom.intr.baseline, sigma_disparity_px, bias_frac),
                          sigma_lat=lateral_sigma_m(d, kp_pixel_sigma_px, geom.intr.fx), ray=ray)
    return out
