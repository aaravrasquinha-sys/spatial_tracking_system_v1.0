"""
Replay a recorded bag through the pipeline.

    python3 run_bag.py data/bags/room_loop.bag             # native RealSense .bag
    python3 run_bag.py data/bags/old_recording.npz          # legacy Phase-0 .npz format
    python3 run_bag.py data/bags/room_loop.bag --max-frames 200

Dispatches on extension: .bag files (from run_slam.py --record, or
recorded directly with Intel's realsense-viewer / rs-record) replay via
RealSenseSource(playback_path=...) with real_time playback disabled, so
every frame from the file is fed to the pipeline with none dropped
regardless of how slow processing is. .npz files use the legacy
dependency-free BagReader kept for synthetic-pipeline bags that
predate the native-recording fix (see the Phase 1 plan's finding F7).
"""
from __future__ import annotations
import argparse
import json
import os
import time
import sys

sys.path.insert(0, ".")

from pyslam.core.config import Config, add_config_args, config_from_args
from pyslam.core.log import get_logger
from pyslam.pipeline import Pipeline
from pyslam.tools.run_outputs import finalize_and_export, summarize_telemetry

log = get_logger("run_bag")


def _open_bag(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bag":
        from pyslam.sensors.realsense import RealSenseSource
        return RealSenseSource(playback_path=path)
    elif ext == ".npz":
        from pyslam.sensors.bagfile import BagReader
        return BagReader(path)
    else:
        raise ValueError(f"Unrecognised bag extension '{ext}' for {path} "
                          f"(expected .bag [native RealSense] or .npz [legacy])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_path", type=str)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto")
    add_config_args(ap)  # WP-K2: --config-override KEY=VALUE (repeatable)
    ap.add_argument("--keep-imagery-cache", action="store_true",
                     help="WP-K3: keep <run_dir>/imagery_cache/ instead of deleting it after map.ply is written")
    args = ap.parse_args()

    # WP-K2: parse --config-override FIRST -- a typo must fail before a camera is
    # opened or a run directory is created, not after the D435i has started streaming.
    cfg = config_from_args(args)

    run_dir = os.path.join("runs", f"run_{int(time.time())}_bag")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run directory: {run_dir}")

    source = _open_bag(args.bag_path)
    log.info(f"Opened {args.bag_path}")

    cfg.save(os.path.join(run_dir, "config.json"))
    if args.config_override:
        log.info(f"Config overrides in effect: {args.config_override}")

    # WP-T3: only RealSenseSource (native .bag playback) can report a real
    # accel<-color extrinsic; the legacy .npz BagReader has no IMU/extrinsics
    # concept, so this falls back to identity (with the same known-wrong
    # caveat as everywhere else) rather than erroring.
    R_body_cam = source.imu_to_color_rotation() if hasattr(source, "imu_to_color_rotation") else None
    pipeline = Pipeline(cfg, vocab_path=None, backend_prefer=args.backend, R_body_cam=R_body_cam,
                        imagery_cache_dir=os.path.join(run_dir, "imagery_cache"))  # WP-K3
    result = None
    try:
        result = pipeline.run(source, max_frames=args.max_frames, verbose=True)
    finally:
        try:
            source.close()
        except Exception as e:
            log.warning(f"source.close() raised during shutdown: {e}")

        if result is not None:
            with open(os.path.join(run_dir, "telemetry.json"), "w") as f:
                json.dump(result.telemetry, f, indent=2)
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump({
                    "bag": args.bag_path,
                    "n_frames": result.n_frames,
                    "n_keyframes": result.n_keyframes,
                    "n_loop_closures": len(result.loop_events),
                    "status_log": result.status_log,
                    "config_overrides": args.config_override,
                    "timing": summarize_telemetry(result.telemetry, cfg.wm_budget_ms),  # WP-K5
                }, f, indent=2)

            # WP-K1/K3: finalize -> map -> trajectory via the shared implementation
            # (pyslam/tools/run_outputs.py) -- map.ply now reflects the FINAL poses
            # and includes keyframes that were evicted to LTM during the run.
            intr = source.intrinsics()
            out = finalize_and_export(run_dir, pipeline, result, intr,
                                       summary_path=os.path.join(run_dir, "summary.json"),
                                       keep_imagery_cache=args.keep_imagery_cache)

            log.info(f"Done. frames={result.n_frames} keyframes={result.n_keyframes} "
                     f"loop_closures={len(result.loop_events)}")
        log.info(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
