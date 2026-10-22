"""
One command from a recording to a finished map.

    python run_map.py ~/bags/room2.db3

  1. tracking: replays the whole recording through the pipeline, no frame dropped, with the offline
     settings (pyslam/offline/profile.py);
  2. offline mapping (run_offline.py): loop closing over ALL keyframes, robust pose graph, lost-gap
     repair, then every tracked frame fused into one surface.

Everything lands in runs/run_<time>_bag/ ; the finished map is in runs/run_<time>_bag/offline/
(map_mesh.ply, map_cloud.ply, trajectory_final_frames.tum, report.json).

    python run_map.py BAG [--max-frames N] [--profile offline|default]
                          [--config-override KEY=VALUE ...]      (tracking settings)
                          [--stride 2] [--voxel 0.02] [--depth-trunc 3.0] [--no-open3d]
                          [--min-gap-s 20] [--top-k 5] [--min-inliers 25]   (offline mapper)
                          [--tracking-only]      stop after step 1
                          [--map-only RUN_DIR]   skip step 1, redo step 2 on an existing run folder

Accepts .bag / .db3 (RealSense recordings) and the legacy .npz bags.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyslam.core.config import add_config_args
from pyslam.core.log import get_logger
from pyslam.offline.profile import offline_config

log = get_logger("run_map")


def _open_bag(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".bag", ".db3"):
        from pyslam.sensors.realsense import RealSenseSource
        return RealSenseSource(playback_path=path)
    if ext == ".npz":
        from pyslam.sensors.bagfile import BagReader
        return BagReader(path)
    raise ValueError(f"Unrecognised recording extension '{ext}' for {path} (expected .bag, .db3 or .npz)")


def track(bag_path: str, cfg, max_frames, backend: str = "auto") -> str:
    from pyslam.pipeline import Pipeline
    from pyslam.tools.run_outputs import finalize_and_export, summarize_telemetry

    run_dir = os.path.join("runs", f"run_{int(time.time())}_bag")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run directory: {run_dir}")
    source = _open_bag(bag_path)
    cfg.save(os.path.join(run_dir, "config.json"))
    R_body_cam = source.imu_to_color_rotation() if hasattr(source, "imu_to_color_rotation") else None
    pipeline = Pipeline(cfg, vocab_path=None, backend_prefer=backend, R_body_cam=R_body_cam,
                        imagery_cache_dir=os.path.join(run_dir, "imagery_cache"))
    result = None
    try:
        result = pipeline.run(source, max_frames=max_frames, verbose=True)
    finally:
        try:
            source.close()
        except Exception as e:  # noqa: BLE001
            log.warning(f"source.close() raised during shutdown: {e}")
        if result is not None:
            with open(os.path.join(run_dir, "telemetry.json"), "w") as f:
                json.dump(result.telemetry, f, indent=2)
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump({"bag": bag_path, "n_frames": result.n_frames, "n_keyframes": result.n_keyframes,
                           "n_loop_closures": len(result.loop_events), "status_log": result.status_log,
                           "timing": summarize_telemetry(result.telemetry, cfg.wm_budget_ms)}, f, indent=2)
            finalize_and_export(run_dir, pipeline, result, source.intrinsics(),
                                summary_path=os.path.join(run_dir, "summary.json"))
            log.info(f"Tracking done. frames={result.n_frames} keyframes={result.n_keyframes}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_path", nargs="?", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto")
    ap.add_argument("--profile", choices=["offline", "default"], default="offline",
                    help="'offline' (default): the settings of the best room2 run; 'default': plain Config()")
    add_config_args(ap)
    ap.add_argument("--tracking-only", action="store_true")
    ap.add_argument("--map-only", metavar="RUN_DIR", default=None)
    for name, typ, default in (("--stride", int, 2), ("--voxel", float, 0.02), ("--depth-trunc", float, 3.0),
                               ("--min-gap-s", float, 20.0), ("--top-k", int, 5), ("--min-inliers", int, 25)):
        ap.add_argument(name, type=typ, default=default)
    ap.add_argument("--no-open3d", action="store_true")
    args = ap.parse_args()

    if args.map_only:
        run_dir = args.map_only
        if args.bag_path is None:
            sys.exit("--map-only needs the recording too:  python run_map.py BAG --map-only RUN_DIR")
    else:
        if args.bag_path is None:
            ap.error("the recording is required")
        try:
            cfg = offline_config(args.config_override, args.profile)
        except ValueError as e:
            sys.exit(f"error: {e}")
        t0 = time.time()
        run_dir = track(args.bag_path, cfg, args.max_frames, args.backend)
        log.info(f"tracking took {time.time() - t0:.0f} s")
    if args.tracking_only:
        print(f"tracking finished: {run_dir}")
        return

    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_offline.py"), run_dir,
           "--bag", args.bag_path, "--stride", str(args.stride), "--voxel", str(args.voxel),
           "--depth-trunc", str(args.depth_trunc), "--min-gap-s", str(args.min_gap_s),
           "--top-k", str(args.top_k), "--min-inliers", str(args.min_inliers)]
    if args.no_open3d:
        cmd.append("--no-open3d")
    log.info("offline mapping: " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"offline mapping failed (exit {rc}); the tracking results are still in {run_dir}")
    print(f"\nFINISHED. map: {os.path.join(run_dir, 'offline')}")


if __name__ == "__main__":
    main()
