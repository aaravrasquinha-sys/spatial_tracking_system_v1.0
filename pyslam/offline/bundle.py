"""
Turn an offline-mapper result into the Module 1 map bundle that Module 2 reads.

    <out>/dense/points.ply        the fused map, in W_cam0 (= the FIRST camera frame's optical frame:
                                  x right, y down, z forward, not gravity-aligned), coloured points
    <out>/capture_profile.json    the capture profile the recording was made with
    <out>/session_summary.json    QA numbers + plain-language warnings (copied into manifest.json at lock)
    <out>/offline_map/            the mesh, loop list and report (optional extras, NOT part of the map_id)

Only dense/points.ply and capture_profile.json decide map_id (contracts/map_bundle.md), so the extras can be
added or removed without changing it. `sts map-lock` then writes manifest.json exactly as for a live map.
"""
from __future__ import annotations
import json
import os
import shutil
from typing import Optional

import numpy as np

from pyslam.core import lie
from pyslam.core.log import get_logger
from pyslam.mapping.dense_filter import remove_isolated_voxels
from pyslam.mapping.export import write_ply

log = get_logger("offline.bundle")


def _warnings(report: dict, n_points: int) -> list:
    w = []
    ev = report.get("evidence", {})
    lp = report.get("loops", {})
    tr = report.get("trajectory", {})
    if n_points < 50_000:
        w.append(f"only {n_points} map points -- the map is sparse (bag too short, or tracking lost for long stretches)")
    lost, frames = ev.get("frames_lost", 0), max(ev.get("frames", 1), 1)
    if lost / frames > 0.10:
        w.append(f"tracking was LOST on {100 * lost / frames:.0f}% of frames; those parts are bridged, not measured")
    if lp.get("loops_final", 0) == 0:
        w.append("no loop closures were found: drift is NOT corrected (fine for a one-way walk; for a "
                 "revisited area record a second pass over the same place)")
    gap = tr.get("start_to_end_gap_m")
    if gap is not None and lp.get("loops_final", 0) > 0 and gap > 0.75:
        w.append(f"start-to-end gap is {gap:.2f} m after loop closing -- check the map for doubled walls")
    return w


def write_bundle(run_dir: str, out_dir: str, capture_profile_src: Optional[str], bag_path: Optional[str] = None,
                 min_neighbors: int = 2, voxel_m: float = 0.02) -> dict:
    off = os.path.join(run_dir, "offline")
    cloud = os.path.join(off, "map_cloud.ply")
    if not os.path.exists(cloud):
        raise FileNotFoundError(f"{cloud} missing -- the offline mapper did not produce a fused cloud "
                                f"(was the recording passed with --bag, and did any frame get fused?)")
    from pyslam.anchor.map_io import read_ply
    pts, rgb = read_ply(cloud)
    if rgb is None:
        rgb = np.full((pts.shape[0], 3), 160, dtype=np.uint8)

    # W_cam0: express everything in the first frame's camera frame (normally the offline world already is)
    from pyslam.offline.rundir import read_tum
    _, Ts = read_tum(os.path.join(off, "trajectory_final_frames.tum"))
    T0 = Ts[0]
    shift = float(np.linalg.norm(T0[:3, 3]) + np.linalg.norm(lie.se3_log(T0)[3:]))
    if shift > 1e-6:
        Ti = lie.se3_inverse(T0)
        pts = (Ti[:3, :3] @ pts.T).T + Ti[:3, 3]
    n_raw = pts.shape[0]
    p2, c2 = remove_isolated_voxels(pts, rgb, voxel_m, min_neighbors)
    if p2.shape[0] >= 0.4 * max(n_raw, 1):       # a real surface keeps most points; if not, the cloud is just sparse
        pts, rgb = p2, c2
    pts = pts.astype(np.float32)
    if pts.shape[0] == 0:
        raise ValueError("the fused map is empty -- no map bundle written (nothing for Module 2 to use)")
    log.info(f"bundle: {n_raw} points, {n_raw - pts.shape[0]} isolated removed, rebased={shift > 1e-6}")

    os.makedirs(os.path.join(out_dir, "dense"), exist_ok=True)
    ply = os.path.join(out_dir, "dense", "points.ply")
    if os.path.exists(ply):
        os.chmod(ply, 0o644)
    write_ply(ply, pts, rgb)

    from pyslam.live.capture_profile import CaptureProfile
    prof = CaptureProfile.load(capture_profile_src) if capture_profile_src and os.path.exists(capture_profile_src) \
        else CaptureProfile()
    prof.save(os.path.join(out_dir, "capture_profile.json"))

    extra = os.path.join(out_dir, "offline_map")
    os.makedirs(extra, exist_ok=True)
    for name in ("map_mesh.ply", "loops.csv", "report.json", "trajectory_final_frames.tum"):
        src = os.path.join(off, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(extra, name))

    with open(os.path.join(off, "report.json")) as f:
        report = json.load(f)
    summary = {"mode": "offline", "builder": "pyslam_offline", "source_bag": bag_path, "run_dir": run_dir,
               "n_points": int(pts.shape[0]), "frame": "W_cam0",
               "qa": {"frames": report["evidence"].get("frames"), "keyframes": report["evidence"].get("keyframes"),
                      "frames_lost": report["evidence"].get("frames_lost"),
                      "loops_final": report.get("loops", {}).get("loops_final"),
                      "loop_time_gap_s": report.get("loops", {}).get("loop_time_gap_s"),
                      "start_to_end_gap_m": report.get("trajectory", {}).get("start_to_end_gap_m"),
                      "path_length_m": report.get("trajectory", {}).get("path_length_m")},
               "warnings": _warnings(report, int(pts.shape[0]))}
    with open(os.path.join(out_dir, "session_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
