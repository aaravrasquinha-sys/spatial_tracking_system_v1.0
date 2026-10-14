"""G4 band visibility and G5 viewpoint (R1 spec sections 6.4 and 6.5). Pure numpy."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from poi_localization.identity.room_geometry import RoomGeometry

VISIBLE = "visible"
OUT_OF_VIEW = "out_of_view"
STATIC_OCCLUDED = "static_occluded"
DYNAMIC_OCCLUDED = "dynamic_occluded"

BBox = Tuple[float, float, float, float]


def band_mids(edges: Sequence[float]) -> List[float]:
    return [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]


def band_visibility(geom: RoomGeometry, ref_depth: Optional[np.ndarray], x: float, y: float, stature: float,
                    edges: Sequence[float], others: Sequence[Tuple[BBox, float]] = (),
                    max_range_m: float = 6.0, static_margin_m: float = 0.15,
                    dynamic_margin_m: float = 0.2) -> List[str]:
    """Per body band: is it visible to the camera for a person standing at floor point (x, y)?

    Each band is represented by a point on the body axis (band mid-height), pulled 0.12 m toward the camera.
    - out_of_view: projects outside the image, behind the camera, or beyond max_range_m
    - static_occluded: the empty-room reference depth at that pixel is > static_margin in FRONT of the point
      (desk, pillar, chair captured in the map)
    - dynamic_occluded: another person's bbox covers the pixel and that person is nearer
    """
    C = geom.t
    mids = band_mids(edges)
    base = np.array([[x, y, m * stature] for m in mids])
    to_c = C - base
    P = base + 0.12 * to_c / np.linalg.norm(to_c, axis=1, keepdims=True)
    u, v, z = geom.project(P)
    states: List[str] = []
    for i in range(len(mids)):
        if not (z[i] > 0 and z[i] <= max_range_m and geom.in_image(u[i:i + 1], v[i:i + 1], z[i:i + 1]).all()):
            states.append(OUT_OF_VIEW)
            continue
        ui, vi = int(round(u[i])), int(round(v[i]))
        if ref_depth is not None and ref_depth.shape == (geom.height, geom.width):
            r = ref_depth[vi, ui]
            if np.isfinite(r) and r < z[i] - static_margin_m:
                states.append(STATIC_OCCLUDED)
                continue
        dyn = False
        for (bb, od) in others:
            if bb[0] <= u[i] <= bb[2] and bb[1] <= v[i] <= bb[3] and od < z[i] - dynamic_margin_m:
                dyn = True
                break
        states.append(DYNAMIC_OCCLUDED if dyn else VISIBLE)
    return states


def face_visible(keypoints: Dict[str, Tuple[float, float, float]], conf_min: float = 0.5) -> bool:
    """Nose and at least one eye confidently detected => the front of the head faces the camera."""
    def ok(name):
        k = keypoints.get(name)
        return k is not None and k[2] >= conf_min
    return ok("nose") and (ok("left_eye") or ok("right_eye"))


def viewpoint(left_shoulder: Optional[np.ndarray], right_shoulder: Optional[np.ndarray], cam_xy: np.ndarray,
              person_xy: np.ndarray, face_seen: bool, velocity_xy: Optional[np.ndarray] = None,
              min_speed_mps: float = 0.4) -> Optional[str]:
    """'front' | 'back' | 'side' | None (unknown).

    Shoulder vector s = R - L on the floor plane; body-forward normal f = rot90(s) = (-s_y, s_x) (a person
    facing +x has their left on +y). Sign: (i) face seen -> front faces the camera; else (ii) walking
    direction when speed > min_speed; else (iii) unknown. theta = angle(f, direction to camera):
    front if |theta| < 45 deg, back if > 135 deg, else side.
    """
    if left_shoulder is None or right_shoulder is None:
        return None
    s = np.asarray(right_shoulder[:2], dtype=float) - np.asarray(left_shoulder[:2], dtype=float)
    n = np.linalg.norm(s)
    if n < 1e-6:
        return None
    f = np.array([-s[1], s[0]]) / n
    u_cam = np.asarray(cam_xy, dtype=float) - np.asarray(person_xy, dtype=float)
    m = np.linalg.norm(u_cam)
    if m < 1e-6:
        return None
    u_cam = u_cam / m
    if face_seen:
        if f @ u_cam < 0:
            f = -f
    elif velocity_xy is not None and np.linalg.norm(velocity_xy) > min_speed_mps:
        if f @ np.asarray(velocity_xy, dtype=float) < 0:
            f = -f
    else:
        return None
    c = float(f @ u_cam)
    return "front" if c > np.cos(np.pi / 4) else ("back" if c < -np.cos(np.pi / 4) else "side")
