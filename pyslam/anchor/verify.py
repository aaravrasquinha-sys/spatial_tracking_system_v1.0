"""
Stage 6: acceptance gates. Every row is a named number with a named threshold; nothing is written as an
accepted calibration unless every REQUIRED row passes (informational rows never gate).

Also: frustum coverage (what fraction of the walkable floor this camera can actually see -- M5 draws the rest
as dead zones) and the wall-referenced marker check (closest thing to what M4 needs: pixel -> room coordinates
compared with tape measurements from map walls).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from pyslam.core import lie
from pyslam.anchor import geo
from pyslam.anchor.config import AnchorConfig


@dataclass
class Check:
    name: str
    value: Optional[float]
    threshold: str
    passed: bool
    required: bool = True
    note: str = ""

    def to_dict(self):
        return {"name": self.name, "value": None if self.value is None or not np.isfinite(self.value) else float(self.value),
                "threshold": self.threshold, "pass": bool(self.passed), "required": self.required, "note": self.note}


def _le(name, v, thr, unit_note="", required=True, fmt="{:.4g}"):
    ok = v is not None and np.isfinite(v) and v <= thr
    return Check(name, v, "<= " + fmt.format(thr), bool(ok), required, unit_note)


def _ge(name, v, thr, unit_note="", required=True, fmt="{:.4g}"):
    ok = v is not None and np.isfinite(v) and v >= thr
    return Check(name, v, ">= " + fmt.format(thr), bool(ok), required, unit_note)


def run_checks(T: np.ndarray, icp_res, physics: dict, up_imu: Optional[np.ndarray], imu_ok: bool,
               ambiguity: dict, repeat: dict, depth_res: dict, sigma: dict, cfg: AnchorConfig,
               profile_match: Optional[bool], room_warnings: List[str]) -> List[Check]:
    ch: List[Check] = []
    ch.append(_le("icp_inlier_rmse_m", icp_res.rmse_m, cfg.gate_icp_rmse_m))
    ch.append(_ge("icp_fitness", icp_res.fitness, cfg.gate_fitness_min, f"share of query points within {cfg.fitness_dist_m*100:.0f} cm of the map"))
    ch.append(_ge("map_coverage_in_view", icp_res.coverage, cfg.gate_coverage_min,
                  f"share of query points with ANY map within {cfg.coverage_dist_m*100:.0f} cm (low => holes in the map here, e.g. dropped keyframes)"))
    # --- IMU / floor cross-checks (independent of the map) ---
    up_pose = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])       # room Z expressed in the camera frame
    if up_imu is not None and imu_ok:
        ch.append(_le("roll_pitch_vs_imu_deg", geo.angle_between_deg(up_pose, up_imu), cfg.gate_imu_tilt_deg,
                      "unconstrained ICP result vs resting accelerometer"))
    else:
        ch.append(Check("roll_pitch_vs_imu_deg", None, f"<= {cfg.gate_imu_tilt_deg}", not cfg.require_imu, cfg.require_imu,
                        "IMU unavailable/unusable -- check skipped" + (" (require_imu=True => FAIL)" if cfg.require_imu else "")))
    fl = physics["floor"]
    ch.append(_le("height_vs_floor_fit_m", abs(float(T[2, 3]) - fl["height_m"]), cfg.gate_floor_height_m,
                  "camera height above the MAP floor vs height above the camera's OWN floor fit"))
    ch.append(_le("tilt_vs_floor_fit_deg", geo.angle_between_deg(up_pose, fl["up_floor_cam"]), cfg.gate_floor_tilt_deg))
    # --- global uniqueness ---
    ch.append(Check("global_uniqueness", ambiguity.get("second_fitness_ratio"),
                    f"no distinct pose with >= {cfg.ambiguity_icp_fitness_ratio:.2f}x best fitness (unless tie-broken/hinted)",
                    not ambiguity["ambiguous"], True, ambiguity.get("note", "")))
    # --- observability & uncertainty ---
    ch.append(_ge("observability_min_eig", icp_res.obs_min_eig, cfg.gate_min_observability,
                  f"weakest direction: {icp_res.obs_worst}", fmt="{:.4g}"))
    ch.append(_le("sigma_trans_m", sigma["trans_m"], cfg.gate_sigma_trans_m, "combined ICP+repeat+map-floor"))
    ch.append(_le("sigma_rot_deg", sigma["rot_deg"], cfg.gate_sigma_rot_deg, "0.3 deg ~ 2.6 cm at 5 m"))
    # --- repeatability ---
    if repeat.get("n", 0) >= 2:
        ch.append(_le("repeat_spread_trans_m", repeat["trans_m"], cfg.gate_repeat_trans_m, repeat.get("kind", "")))
        ch.append(_le("repeat_spread_rot_deg", repeat["rot_deg"], cfg.gate_repeat_rot_deg, repeat.get("kind", "")))
    else:
        ch.append(Check("repeat_spread", None, "n >= 2 subsets", False, True, "no repeat solutions available"))
    # --- depth residual image ---
    ch.append(_ge("depth_residual_frac_within_tol", depth_res.get("frac_within_tol"), cfg.gate_depth_resid_frac,
                  f"share of compared pixels with |live - rendered| <= {cfg.gate_depth_resid_tol_m*100:.0f} cm "
                  f"(render coverage {depth_res.get('render_coverage', 0):.0%})"))
    # --- provenance ---
    if profile_match is None:
        ch.append(Check("capture_profile_match", None, "same hash as the map's", True, False,
                        "capture or map has no profile hash recorded -- cannot verify depth pipelines match"))
    else:
        ch.append(Check("capture_profile_match", 1.0 if profile_match else 0.0, "same hash as the map's", bool(profile_match), True,
                        "" if profile_match else "capture used a different depth pipeline than mapping"))
    ch.append(Check("room_frame_warnings", float(len(room_warnings)), "informational", True, False, "; ".join(room_warnings)))
    return ch


def frustum_coverage(walkable: dict, T_room_cam: np.ndarray, intr: dict, live_depth: Optional[np.ndarray],
                     max_range: float = 8.0) -> dict:
    """Which walkable-floor cells can this camera see? A cell is 'visible' if it projects inside the image,
    within range, and nothing measurably closer occludes it (checked against the live reference depth where
    available); 'unverified' if it's in the frustum but the reference depth has no reading there."""
    grid = walkable["grid"]
    r = walkable["resolution"]
    ox, oy = walkable["origin"]
    rows, cols = np.nonzero(grid)
    X = np.stack([ox + (cols + 0.5) * r, oy + (rows + 0.5) * r, np.zeros(rows.size)], axis=1)
    R, t = T_room_cam[:3, :3], T_room_cam[:3, 3]
    pc = (X - t) @ R
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr["fx"] * pc[:, 0] / z + intr["cx"]
        v = intr["fy"] * pc[:, 1] / z + intr["cy"]
    H, W = intr["height"], intr["width"]
    inside = (z > 0.3) & (z < max_range) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    vis = np.zeros(rows.size, bool)
    unver = np.zeros(rows.size, bool)
    if live_depth is not None:
        ui, vi = np.rint(u[inside]).astype(int), np.rint(v[inside]).astype(int)
        d = live_depth[vi, ui]
        idx = np.nonzero(inside)[0]
        has = np.isfinite(d)
        vis[idx[has]] = d[has] >= z[idx[has]] - 0.15
        unver[idx[~has]] = True
    else:
        vis = inside
    vis_grid = np.zeros_like(grid)
    unv_grid = np.zeros_like(grid)
    vis_grid[rows[vis], cols[vis]] = True
    unv_grid[rows[unver], cols[unver]] = True
    n = max(int(grid.sum()), 1)
    return {"frac_visible": float(vis.sum() / n), "frac_in_frustum_unverified": float(unver.sum() / n),
            "frac_dead": float(1.0 - (vis.sum() + unver.sum()) / n),
            "visible_grid": vis_grid, "unverified_grid": unv_grid}


def check_markers(markers: List[dict], T_room_cam: np.ndarray, intr: dict, walls: List[dict],
                  tol_m: float) -> dict:
    """markers: [{"name":..., "pixel":[u,v], and EITHER "room_xy":[x,y] OR "wall_offsets":[{"wall":id,"d":m},{"wall":id,"d":m}]}]
    Room position from wall offsets: intersect the two lines  n_i . p = offset_i + d_i  (inward normals, so
    a point d metres INSIDE wall i satisfies n_i.p = offset_i + d_i). Predicted position: ray through the
    pixel intersected with the floor z = 0 using T_room_cam. Returns per-marker error and the max."""
    wall_by_id = {w["id"]: w for w in walls}
    R, t = T_room_cam[:3, :3], T_room_cam[:3, 3]
    out = []
    for m in markers:
        u, v = m["pixel"]
        ray_cam = np.array([(u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"], 1.0])
        ray = R @ ray_cam
        if ray[2] >= -1e-6:
            out.append({"name": m.get("name"), "error_m": None, "note": "pixel ray does not hit the floor"})
            continue
        s = -t[2] / ray[2]
        pred = (t + s * ray)[:2]
        if "room_xy" in m:
            truth = np.asarray(m["room_xy"], float)
        else:
            a, b = m["wall_offsets"]
            wa, wb = wall_by_id[a["wall"]], wall_by_id[b["wall"]]
            A = np.array([wa["normal_in_room"][:2], wb["normal_in_room"][:2]], float)
            rhs = np.array([wa["offset_room"] + a["d"], wb["offset_room"] + b["d"]], float)
            if abs(np.linalg.det(A)) < 0.2:
                out.append({"name": m.get("name"), "error_m": None, "note": "the two walls are (nearly) parallel"})
                continue
            truth = np.linalg.solve(A, rhs)
        err = float(np.linalg.norm(pred - truth))
        out.append({"name": m.get("name"), "predicted_xy": pred.tolist(), "measured_xy": truth.tolist(), "error_m": err,
                    "range_m": float(np.linalg.norm(pred - t[:2]))})
    errs = [o["error_m"] for o in out if o.get("error_m") is not None]
    return {"markers": out, "max_error_m": max(errs) if errs else None,
            "pass": bool(errs) and max(errs) <= tol_m, "tol_m": tol_m}
