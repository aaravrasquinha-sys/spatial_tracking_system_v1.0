"""
Orchestrator: map bundle + static capture(s) -> T_room_cam with checks. Pure numpy/scipy (no hardware).

    result = calibrate(load_map_bundle(map_dir), [StaticCapture.load(p) ...], cfg)
    if result.accepted: write_anchor_bundle(...)

Design rules enforced here (see SYSTEM_SUMMARY_ANCHOR.md):
  * never guess: an ambiguous global pose, a degenerate view, or a failed cross-check REJECTS the calibration
  * the answer is chosen by ICP fitness/RMSE of the top-K DISTINCT search peaks (not by the coarse search score)
  * sigma = sqrt(ICP^2 + repeat^2 + map-accuracy-floor^2) -- what M4 propagates into every position covariance
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
import numpy as np

from pyslam.core import lie
from pyslam.core.log import get_logger
from pyslam.anchor import geo
from pyslam.anchor.config import AnchorConfig
from pyslam.anchor.capture import StaticCapture
from pyslam.anchor.query_cloud import capture_to_cloud, stable_mask
from pyslam.anchor.room_frame import derive_room_frame, RoomFrame
from pyslam.anchor.physics_prior import physics_prior
from pyslam.anchor.global_search import (build_map_raster, fine_config, query_slice, csm_search, distinct_peaks,
                                         refine_local, Candidate)
from pyslam.anchor.icp import MapTarget, icp, colour_correlation, IcpResult
from pyslam.anchor.walkable import build_walkable
from pyslam.anchor import render as R
from pyslam.anchor import verify as V

log = get_logger("anchor.calibrate")


@dataclass
class CalibrationResult:
    accepted: bool
    T_room_cam: np.ndarray
    sigma: dict
    checks: List[V.Check]
    report: dict
    room_frame: RoomFrame
    pts_room: np.ndarray
    colors: Optional[np.ndarray]
    walkable: dict
    capture: StaticCapture
    images: dict
    frustum: dict
    method: str
    up_imu: Optional[np.ndarray]
    map_id: str


def pool_captures(caps: List[StaticCapture]) -> StaticCapture:
    if len(caps) == 1:
        return caps[0]
    base = caps[0]
    for c in caps[1:]:
        if c.depth_med.shape != base.depth_med.shape:
            raise ValueError("captures have different resolutions -- cannot pool")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(np.stack([c.depth_med for c in caps]), axis=0).astype(np.float32)
    mad = np.nanmax(np.stack([np.nan_to_num(c.depth_mad, nan=1e3) for c in caps]), axis=0).astype(np.float32)
    vf = np.mean(np.stack([c.valid_frac for c in caps]), axis=0).astype(np.float32)
    ups = [c.up_cam for c in caps if c.up_cam is not None]
    up = None
    if ups:
        up = np.mean(ups, axis=0); up /= np.linalg.norm(up)
    infos = [c.imu_info for c in caps]
    ok = all(i.get("ok", False) for i in infos) and len(ups) == len(caps)
    spread = max((geo.angle_between_deg(u, up) for u in ups), default=0.0) if up is not None else None
    return StaticCapture(depth_med=med, depth_mad=mad, valid_frac=vf, rgb_med=base.rgb_med,
                         sub_med=np.concatenate([c.sub_med for c in caps]), intr=base.intr, up_cam=up,
                         imu_info={"ok": ok, "pooled_from": len(caps), "up_spread_deg": spread},
                         n_frames=sum(c.n_frames for c in caps), serial=base.serial,
                         capture_profile_sha=base.capture_profile_sha, created=base.created,
                         source="pooled:" + ",".join(sorted({c.source for c in caps})))


def _T_room_cam_from_seed(c: Candidate, h: float, R_L_cam: np.ndarray) -> np.ndarray:
    T_room_L = lie.make_T(geo.rot_z(c.yaw), np.array([c.x, c.y, h]))
    return T_room_L @ lie.make_T(R_L_cam, np.zeros(3))


def _spread(Tref: np.ndarray, Ts: List[np.ndarray]) -> Tuple[float, float]:
    """max deviation from the mean of a set of poses, (metres, degrees), in Tref's tangent space."""
    if len(Ts) < 2:
        return 0.0, 0.0
    xi = np.array([lie.se3_log(lie.se3_inverse(Tref) @ T) for T in Ts])
    d = xi - xi.mean(axis=0)
    return float(np.max(np.linalg.norm(d[:, :3], axis=1))), float(np.degrees(np.max(np.linalg.norm(d[:, 3:], axis=1))))


def calibrate(map_bundle: dict, captures: List[StaticCapture], cfg: Optional[AnchorConfig] = None,
              hint: Optional[Tuple[float, float, float, float]] = None, room_frame: Optional[RoomFrame] = None,
              cam_id: str = "cam0", seed: int = 0, t_true_for_logging: Optional[np.ndarray] = None) -> CalibrationResult:
    cfg = cfg or AnchorConfig()
    t0 = time.time()
    pts_map, colors = map_bundle["pts_map"], map_bundle.get("colors")
    warnings: List[str] = []

    # ---- Stage 0: clean map, derive room frame, walkable ----
    if pts_map.shape[0] <= 4_000_000:
        keep = geo.statistical_outlier_mask(pts_map, cfg.sor_k, cfg.sor_std)
        pts_map = pts_map[keep]
        colors = None if colors is None else colors[keep]
        log.info(f"outlier removal kept {int(keep.sum())} / {keep.size} map points")
    rf = room_frame or derive_room_frame(pts_map, cfg, seed)
    Tr = rf.T_room_map
    pts_room = pts_map @ Tr[:3, :3].T + Tr[:3, 3]
    walk = build_walkable(pts_room, cfg)

    # ---- Stage 1/2: capture -> cloud -> physics prior ----
    cap = pool_captures(captures)
    cloud = capture_to_cloud(cap, cfg)
    if cloud["pts"].shape[0] < 3000:
        raise RuntimeError(f"only {cloud['pts'].shape[0]} stable depth points survived filtering -- capture is too sparse "
                           f"(dark/IR-absorbing scene, shimmering surfaces, or the camera moved during capture)")
    imu_ok = bool(cap.imu_info.get("ok", False))
    pp = physics_prior(cloud["pts"], cap.up_cam, imu_ok, cfg, seed)
    warnings += pp["warnings"]
    R_L = pp["R_L_cam"]
    h = pp["height_m"]

    # ---- Stage 3: global search ----
    pts_L = cloud["pts"] @ R_L.T
    q_xy, q_w = query_slice(pts_L, h, cfg, cloud["weights"])
    raster = build_map_raster(pts_room, cfg)
    fras = build_map_raster(pts_room, fine_config(cfg))
    cands = csm_search(raster, q_xy, q_w, cfg, hint)
    peaks = distinct_peaks(cands, cfg)
    if not peaks:
        raise RuntimeError("global search produced no candidates (query has no structure inside the map extent)")
    csm_ratio = (peaks[0].score / peaks[1].score) if len(peaks) > 1 and peaks[1].score > 0 else float("inf")

    # ---- Stage 4: ICP from every distinct peak, decide by ICP quality ----
    rows = []
    targets = []
    for pk in peaks:
        rp = refine_local(fras, q_xy, q_w, pk)
        T_seed = _T_room_cam_from_seed(rp, h, R_L)
        tgt = None
        for c_center, c_tgt in targets:                      # reuse a crop when the seeds are within 0.5 m of each other
            if np.linalg.norm(c_center - T_seed[:3, 3]) < 0.5:
                tgt = c_tgt
                break
        if tgt is None:
            try:
                tgt = MapTarget.crop(pts_room, colors, T_seed[:3, 3], cfg.range_max_m + cfg.icp_crop_margin_m + 0.5,
                                     cfg.icp_normal_k)
            except ValueError as e:
                log.warning(f"peak at ({pk.x:.2f},{pk.y:.2f}): {e}")
                continue
            targets.append((T_seed[:3, 3].copy(), tgt))
        res = icp(cloud["pts"], cloud["weights"], cloud["sigma"], tgt, T_seed, cfg)
        rows.append({"peak": pk, "refined": rp, "res": res, "target": tgt})
    if not rows:
        raise RuntimeError("no search peak had enough surrounding map to run ICP")
    rows.sort(key=lambda r: (-round(r["res"].fitness, 3), r["res"].rmse_m))
    best = rows[0]
    distinct = None
    for r in rows[1:]:
        dt, dr = geo.pose_delta(best["res"].T, r["res"].T)
        if dt > cfg.csm_distinct_xy_m or dr > cfg.csm_distinct_yaw_deg:
            distinct = r
            break
    ambiguity = {"csm_score_ratio": float(csm_ratio), "ambiguous": False, "second_fitness_ratio": None, "tiebreak": None,
                 "n_candidates_icp": len(rows),
                 "candidates": [{"x": float(r["res"].T[0, 3]), "y": float(r["res"].T[1, 3]), "fitness": float(r["res"].fitness),
                                 "rmse_m": float(r["res"].rmse_m)} for r in rows]}
    # every DISTINCT alternative pose (>= 0.3 m or >= 8 deg from the best and from each other)
    alts = []
    for r in rows[1:]:
        if geo.pose_delta(best["res"].T, r["res"].T)[0] > cfg.csm_distinct_xy_m or \
                geo.pose_delta(best["res"].T, r["res"].T)[1] > cfg.csm_distinct_yaw_deg:
            if all(geo.pose_delta(a_["res"].T, r["res"].T)[0] > 0.05 or geo.pose_delta(a_["res"].T, r["res"].T)[1] > 1.0 for a_ in alts):
                alts.append(r)
    if alts:
        ratio = alts[0]["res"].fitness / max(best["res"].fitness, 1e-9)
        ambiguity["second_fitness_ratio"] = float(ratio)
        tied = [r for r in alts if r["res"].fitness / max(best["res"].fitness, 1e-9) >= cfg.ambiguity_icp_fitness_ratio
                and r["res"].fitness >= 0.3 and r["res"].rmse_m <= 2 * cfg.gate_icp_rmse_m]
        if tied:
            group = [best] + tied                          # ALL geometrically tied explanations, not just two of them
            cc = [colour_correlation(g["res"].T, cloud["pts"], cloud["colors"], g["target"], cfg) for g in group]
            ambiguity["colour_corr"] = [None if c is None else float(c) for c in cc]
            resolved = False
            if all(c is not None for c in cc):
                order = np.argsort(cc)[::-1]
                if cc[order[0]] - cc[order[1]] >= cfg.colour_tiebreak_margin:
                    winner = group[int(order[0])]
                    best = winner
                    ambiguity["tiebreak"] = "colour"
                    ambiguity["note"] = (f"{len(group)} geometrically tied poses; colour correlation {cc[order[0]]:.2f} vs next "
                                         f"{cc[order[1]]:.2f} broke the tie")
                    resolved = True
            if not resolved:
                ambiguity["ambiguous"] = True
                ambiguity["note"] = (f"{len(tied)} other distinct pose(s) explain the view almost as well as the best (fitness ratio "
                                     f"{ratio:.2f}, nearest {geo.pose_delta(best['res'].T, tied[0]['res'].T)[0]:.2f} m away); colour could not "
                                     f"break the tie. Pass a rough position/heading hint (read x,y off top_down.png) or mount the camera "
                                     f"where it sees something asymmetric.")
    res: IcpResult = best["res"]
    T = res.T

    # ---- repeatability: sub-medians (and, if given, independent captures) ----
    tgt = best["target"]
    Ts_sub = []
    for i in range(cap.sub_med.shape[0]):
        ci = capture_to_cloud(cap, cfg, depth=cap.sub_med[i], stride=cfg.query_stride * 2)
        if ci["pts"].shape[0] < 1000:
            continue
        ri = icp(ci["pts"], ci["weights"], ci["sigma"], tgt, T, cfg, scales=(0.04, 0.025), iters=(15, 20))
        Ts_sub.append(ri.T)
    rep_sub = _spread(T, Ts_sub)
    Ts_cap = []
    if len(captures) >= 2:
        for c in captures:
            ci = capture_to_cloud(c, cfg, stride=cfg.query_stride * 2)
            if ci["pts"].shape[0] < 1000:
                continue
            ri = icp(ci["pts"], ci["weights"], ci["sigma"], tgt, T, cfg, scales=(0.04, 0.025), iters=(15, 20))
            Ts_cap.append(ri.T)
    rep_cap = _spread(T, Ts_cap)
    repeat = {"n": len(Ts_sub) + len(Ts_cap), "trans_m": max(rep_sub[0], rep_cap[0]), "rot_deg": max(rep_sub[1], rep_cap[1]),
              "kind": (f"{len(Ts_sub)} sub-medians of one capture (share systematic errors)" +
                       (f" + {len(Ts_cap)} independent captures" if Ts_cap else " -- record 3-5 separate captures for a true session-to-session number")),
              "sub_median": {"trans_m": rep_sub[0], "rot_deg": rep_sub[1]},
              "independent_captures": {"n": len(Ts_cap), "trans_m": rep_cap[0], "rot_deg": rep_cap[1]}}

    # ---- uncertainty ----
    s_icp_t, s_icp_r = res.sigma_trans_m(), res.sigma_rot_deg()
    rep_t, rep_r = repeat["trans_m"], repeat["rot_deg"]
    sigma = {"trans_m": float(np.sqrt(s_icp_t ** 2 + rep_t ** 2 + rf.map_trans_sigma_m ** 2)),
             "rot_deg": float(np.sqrt(s_icp_r ** 2 + rep_r ** 2 + rf.map_rot_sigma_deg ** 2)),
             "components": {"icp_trans_m": s_icp_t, "icp_rot_deg": s_icp_r, "repeat_trans_m": rep_t, "repeat_rot_deg": rep_r,
                            "map_floor_trans_m": rf.map_trans_sigma_m, "map_floor_rot_deg": rf.map_rot_sigma_deg}}

    # ---- depth residual image + overlay + frustum coverage ----
    rendered = R.render_depth(tgt.pts, T, cap.intr, max_range=cfg.range_max_m + 1.0)
    stable = stable_mask(cap.depth_med, cap.depth_mad, cap.valid_frac, cfg)
    dres = R.depth_residual(cap.depth_med, rendered, stable, cfg.gate_depth_resid_tol_m)
    images = {"overlay": R.overlay_edges(cap.rgb_med, rendered, np.where(stable, cap.depth_med, np.nan)),
              "residual": R.colorize_residual(dres["residual_img"]), "rendered_depth": rendered,
              "stable_mask": stable}
    frustum = V.frustum_coverage(walk, T, cap.intr, np.where(stable, cap.depth_med, np.nan))

    # ---- provenance ----
    prof_map = map_bundle.get("capture_profile_sha")
    prof_cap = cap.capture_profile_sha
    profile_match = None if (prof_map is None or prof_cap is None) else (prof_map == prof_cap)
    checks = V.run_checks(T, res, pp, cap.up_cam, imu_ok, ambiguity, repeat, dres, sigma, cfg, profile_match, rf.warnings)
    accepted = all(c.passed for c in checks if c.required) and res.converged
    method = "imu_floor_prior+correlative_search+robust_icp"
    if ambiguity.get("tiebreak"):
        method += "+colour_tiebreak"
    if hint is not None:
        method += "+operator_hint"
    report = {
        "accepted": accepted, "elapsed_s": time.time() - t0, "map_id": map_bundle["map_id"],
        "n_query_points": int(cloud["pts"].shape[0]), "up_source": pp["up_source"],
        "physics": {"height_m": pp["height_m"], "floor_rms_m": pp["floor"]["rms_m"],
                    "imu_vs_floor_tilt_deg": pp["imu_vs_floor_tilt_deg"], "imu_info": cap.imu_info},
        "ambiguity": ambiguity, "repeatability": repeat, "sigma": sigma,
        "icp": {"rmse_m": res.rmse_m, "fitness": res.fitness, "coverage": res.coverage, "n_inliers": res.n_inliers,
                "obs_min_eig": res.obs_min_eig, "obs_worst_direction": res.obs_worst, "chi2_scale": res.chi2_scale,
                "stages": res.stage_log, "converged": res.converged},
        "depth_residual": {k: v for k, v in dres.items() if k != "residual_img"},
        "frustum": {k: v for k, v in frustum.items() if not k.endswith("_grid")},
        "walkable_stats": walk["stats"], "warnings": warnings + rf.warnings,
        "checks": [c.to_dict() for c in checks], "config": cfg.to_dict(),
    }
    log.info(f"Calibration {'ACCEPTED' if accepted else 'REJECTED'} in {time.time()-t0:.1f}s: "
             f"sigma={sigma['trans_m']*1000:.1f} mm / {sigma['rot_deg']:.2f} deg, fitness={res.fitness:.2f}, rmse={res.rmse_m*1000:.1f} mm")
    for c in checks:
        if c.required and not c.passed:
            log.warning(f"  FAILED gate {c.name}: value={c.value} needs {c.threshold}. {c.note}")
    return CalibrationResult(accepted=accepted, T_room_cam=T, sigma=sigma, checks=checks, report=report, room_frame=rf,
                             pts_room=pts_room, colors=colors, walkable=walk, capture=cap, images=images,
                             frustum=frustum, method=method, up_imu=cap.up_cam, map_id=map_bundle["map_id"])
