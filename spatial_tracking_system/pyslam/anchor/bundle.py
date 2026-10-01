"""
Write the ANCHOR BUNDLE (a directory NEXT TO, never inside, module 1's map bundle -- M1's bundle stays
untouched/read-only). Layout:

  anchors/<name>/
    calibration.cam0.json          only if ACCEPTED  (exact shape poi_localization/frames/room_frame.py loads)
    calibration.cam0.REJECTED.json otherwise (a name M4 can never load by accident)
    room_frame.json                T_room_map, floor fit + quadrants, walls (inward normals), yaw source
    walkable.json                  STS schema + room-frame stats  (M4 phase_b.walkable_grid_path)
    viewer/points.ply              decimated map cloud IN THE ROOM FRAME (M5 discovers <map_bundle_dir>/viewer/points.ply)
    cam0_reference_depth.npz/.png  median depth + stable mask (watchdog reference)
    static_capture.npz             the raw processed capture (re-solve / audit without the camera)
    report.json                    every gate with its number, config used, ambiguity table
    overlay.png depth_residual.png frustum_coverage.png top_down.png     visual sign-off
"""
from __future__ import annotations
import json
import os
import time
from typing import Optional
import numpy as np

from pyslam.core.log import get_logger
from pyslam.anchor import geo
from pyslam.anchor import render as R
from pyslam.anchor.walkable import walkable_to_json

log = get_logger("anchor.bundle")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def build_calibration_json(result, cam_id: str = "cam0", map_capture_profile_sha: Optional[str] = None) -> dict:
    cap = result.capture
    i = cap.intr
    checks_compact = {c.name: (None if c.value is None or not np.isfinite(c.value) else float(c.value)) for c in result.checks}
    return {
        "cam_id": cam_id, "map_id": result.map_id, "serial": cap.serial,
        "intrinsics": {"width": int(i["width"]), "height": int(i["height"]), "fx": float(i["fx"]), "fy": float(i["fy"]),
                       "cx": float(i["cx"]), "cy": float(i["cy"]), "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0],
                       "depth_scale": float(i["depth_scale"]), "baseline": float(i.get("baseline", 0.05))},
        "T_room_cam": result.T_room_cam.tolist(),
        "sigma": {"trans_m": float(result.sigma["trans_m"]), "rot_deg": float(result.sigma["rot_deg"])},
        "sigma_components": result.sigma["components"],
        "method": result.method,
        "accepted": bool(result.accepted),
        "checks": checks_compact,
        "gravity_up_cam": None if result.up_imu is None else [float(x) for x in result.up_imu],
        "capture_profile_sha": cap.capture_profile_sha, "map_capture_profile_sha": map_capture_profile_sha,
        "reference_depth": f"{cam_id}_reference_depth.npz",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def top_down_png(walkable: dict, walls: list, path: str, T_room_cam: Optional[np.ndarray] = None,
                 vis: Optional[np.ndarray] = None, unv: Optional[np.ndarray] = None, px_per_m: int = 100):
    """Top-down preview (y up): walkable grid, wall ids (for marker offsets / hints), and -- once calibrated --
    the camera, its heading and which floor cells it sees."""
    import cv2
    grid = walkable["grid"]
    r = walkable["resolution"]
    ox, oy = walkable["origin"]
    H, W = grid.shape
    scale = max(1, int(round(px_per_m * r)))
    col = np.full((H, W, 3), 25, np.uint8)
    col[grid] = (70, 70, 70)
    if vis is not None:
        col[vis] = (60, 150, 60)
    if unv is not None:
        col[unv] = (150, 140, 50)
    img = np.kron(col, np.ones((scale, scale, 1), np.uint8))[::-1].copy()

    def to_px(x, y):
        return int((x - ox) / r * scale), int((H * r - (y - oy)) / r * scale)

    for wl in walls:
        n = np.array(wl["normal_in_room"][:2])
        tang = np.array([-n[1], n[0]])
        p0 = n * wl["offset_room"]
        a_ = p0 - tang * wl["length_m"] / 2
        b_ = p0 + tang * wl["length_m"] / 2
        cv2.line(img, to_px(*a_), to_px(*b_), (80, 200, 255), 1)
        m = (a_ + b_) / 2 + n * 0.15
        cv2.putText(img, f"w{wl['id']}", to_px(*m), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1)
    # 1 m grid ticks so a human can read x,y off the picture for --hint
    for gx in range(int(np.ceil(ox)), int(ox + W * r) + 1):
        cv2.putText(img, str(gx), to_px(gx, oy + 0.02), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    for gy in range(int(np.ceil(oy)), int(oy + H * r) + 1):
        cv2.putText(img, str(gy), to_px(ox + 0.02, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    if T_room_cam is not None:
        cam = T_room_cam[:3, 3]
        cv2.circle(img, to_px(cam[0], cam[1]), 5, (255, 80, 80), -1)
        fwd = T_room_cam[:3, :3] @ np.array([0, 0, 1.0])
        tip = cam[:2] + 0.8 * fwd[:2] / max(np.linalg.norm(fwd[:2]), 1e-6)
        cv2.arrowedLine(img, to_px(cam[0], cam[1]), to_px(tip[0], tip[1]), (255, 80, 80), 2)
        cv2.putText(img, "green=visible olive=in view/no depth grey=dead red=camera", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def write_map_products(out_dir: str, room_frame, pts_room: np.ndarray, colors, walkable: dict) -> dict:
    """Everything derivable from the map alone (no camera needed): usable by M5 before any calibration exists."""
    os.makedirs(os.path.join(out_dir, "viewer"), exist_ok=True)
    with open(os.path.join(out_dir, "room_frame.json"), "w") as f:
        json.dump(room_frame.to_dict(), f, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "walkable.json"), "w") as f:
        json.dump(walkable_to_json(walkable), f)
    n = write_viewer_points(os.path.join(out_dir, "viewer", "points.ply"), pts_room, colors)
    top_down_png(walkable, room_frame.walls, os.path.join(out_dir, "top_down.png"))
    return {"viewer_points": n}


def write_viewer_points(path: str, pts_room: np.ndarray, colors: Optional[np.ndarray], max_points: int = 1_500_000):
    from pyslam.mapping.export import write_ply
    p, c = geo.voxel_downsample(pts_room, 0.03, colors)
    if p.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(p.shape[0], max_points, replace=False)
        p = p[sel]
        c = None if c is None else c[sel]
    if c is None:
        c = np.full((p.shape[0], 3), 200, np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_ply(path, p.astype(np.float32), c)
    return int(p.shape[0])


def write_anchor_bundle(result, out_dir: str, cam_id: str = "cam0", map_capture_profile_sha: Optional[str] = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "viewer"), exist_ok=True)
    cal = build_calibration_json(result, cam_id, map_capture_profile_sha)
    name = f"calibration.{cam_id}.json" if result.accepted else f"calibration.{cam_id}.REJECTED.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(cal, f, indent=2, default=_json_default)
    n_view = write_map_products(out_dir, result.room_frame, result.pts_room, result.colors, result.walkable)["viewer_points"]
    cap = result.capture
    stable = result.images["stable_mask"]
    np.savez_compressed(os.path.join(out_dir, f"{cam_id}_reference_depth.npz"), depth_med=cap.depth_med,
                        stable=stable, depth_mad=cap.depth_mad, intr=np.array(json.dumps(cap.intr)),
                        up_cam=np.zeros(3) if result.up_imu is None else result.up_imu)
    import cv2
    d16 = np.nan_to_num(cap.depth_med * 1000.0, nan=0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, f"{cam_id}_reference_depth.png"), d16)
    cap.save(os.path.join(out_dir, "static_capture.npz"))
    R.save_png(os.path.join(out_dir, "overlay.png"), result.images["overlay"])
    R.save_png(os.path.join(out_dir, "depth_residual.png"), result.images["residual"])
    top_down_png(result.walkable, result.room_frame.walls, os.path.join(out_dir, "top_down.png"), result.T_room_cam,
                 result.frustum["visible_grid"], result.frustum["unverified_grid"])
    fc = np.zeros(result.walkable["grid"].shape + (3,), np.uint8)
    fc[result.walkable["grid"]] = (70, 70, 70)
    fc[result.frustum["visible_grid"]] = (60, 180, 60)
    fc[result.frustum["unverified_grid"]] = (180, 170, 60)
    R.save_png(os.path.join(out_dir, "frustum_coverage.png"), fc[::-1])
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(result.report, f, indent=2, default=_json_default)
    log.info(f"Anchor bundle written to {out_dir} ({name}; viewer/points.ply has {n_view} points)")
    return {"calibration": os.path.join(out_dir, name), "accepted": result.accepted, "out_dir": out_dir}
