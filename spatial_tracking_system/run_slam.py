"""
Run the pipeline live on a connected D435i.

    python3 run_slam.py --realsense --imu
    python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
    python3 run_slam.py --realsense --imu --max-frames 300

Every run writes a run_<timestamp>/ directory under runs/ with the exact
config used, telemetry, the trajectory, and (on request) a native
librealsense .bag recording -- see tools/bundle.py to package one of
these for sharing.

Recording note (Phase 1 fix, see the Phase 1 plan's finding F7):
Phase 0 recorded by wrapping this process's own frame iterator, so a
recording only ever captured frames after our pipeline (which can spend
up to several seconds per frame) had consumed them -- badly subsampled
and desynced from IMU. This version passes --record straight into
librealsense's own recorder (RealSenseSource(record_path=...)), which
writes every stream at full hardware rate in the C++ SDK layer,
independent of how fast Python drains frames. Replay it with run_bag.py.
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

log = get_logger("run_slam")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realsense", action="store_true", required=True)
    ap.add_argument("--imu", action="store_true", help="enable IMU streams (recorded, not yet fused)")
    ap.add_argument("--record", type=str, default=None,
                     help="record every stream to this .bag via librealsense's native recorder")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto")
    ap.add_argument("--imu-mode", choices=["callback", "synced"], default="callback",
                     help="WP-J3 (Orin port): 'callback' (default) opens the motion "
                          "sensor directly on its own SDK thread, independent of "
                          "capture cadence -- more robust against IMU sample drops "
                          "on a slow/loaded frontend. 'synced' is the original "
                          "Phase-0 behaviour (motion samples only drained from "
                          "wait_for_frames() framesets). Falls back to 'synced' "
                          "automatically (with a logged warning) if 'callback' fails "
                          "to start -- this flag is for when you want that fallback "
                          "forced, e.g. to compare the two on your specific rig.")
    add_config_args(ap)  # WP-K2: --config-override KEY=VALUE (repeatable)
    ap.add_argument("--keep-imagery-cache", action="store_true",
                     help="WP-K3: keep <run_dir>/imagery_cache/ (lossless PNGs of every keyframe "
                          "evicted to LTM, ~1MB each) instead of deleting it after map.ply is written")
    args = ap.parse_args()

    # WP-K2: parse --config-override FIRST -- a typo must fail before a camera is
    # opened or a run directory is created, not after the D435i has started streaming.
    cfg = config_from_args(args)

    run_dir = os.path.join("runs", f"run_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run directory: {run_dir}")

    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)

    from pyslam.sensors.realsense import RealSenseSource
    source = RealSenseSource(enable_imu=args.imu, record_path=args.record,
                              imu_capture_mode=args.imu_mode)

    cfg.save(os.path.join(run_dir, "config.json"))
    if args.config_override:
        log.info(f"Config overrides in effect: {args.config_override}")

    # WP-T3: real accel<-color extrinsic, queried from the device
    # (identity + a logged warning if that query fails or --imu wasn't
    # passed) -- see RealSenseSource.imu_to_color_rotation's docstring.
    R_body_cam = source.imu_to_color_rotation()
    # WP-RELOC: LTM now lands inside the run directory instead of a
    # randomly-named tempfile.mkstemp() path in /tmp (the previous
    # default -- see memory/ltm_store.py and memory/memory.py's own
    # constructor comment on why that happened). This is what makes a
    # run's descriptors findable at all after the fact: a relocalization
    # map pack (finalize_and_export, below) reads directly from the live
    # Pipeline while it still has this run's Memory resident, so this
    # path is mostly relevant to pyslam.tools.build_reloc_map's
    # best-effort retrofit of OLDER runs that predate this change.
    pipeline = Pipeline(cfg, vocab_path=None, backend_prefer=args.backend, R_body_cam=R_body_cam,
                        ltm_path=os.path.join(run_dir, "ltm.sqlite3"),
                        imagery_cache_dir=os.path.join(run_dir, "imagery_cache"))  # WP-K3

    # Everything from here down is wrapped so that Ctrl-C (or any
    # exception once frames have started flowing) still leaves a valid
    # run directory: summary.json, telemetry.json and map.ply for
    # whatever was captured before the interruption, per finding F7.
    # Pipeline.run() itself catches KeyboardInterrupt internally and
    # returns the partial result rather than raising, so `result` is
    # always defined here as long as source.intrinsics() succeeded.
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
                    "n_frames": result.n_frames,
                    "n_keyframes": result.n_keyframes,
                    "n_loop_closures": len(result.loop_events),
                    "status_log": result.status_log,
                    "config_overrides": args.config_override,
                    "timing": summarize_telemetry(result.telemetry, cfg.wm_budget_ms),  # WP-K5
                }, f, indent=2)

            # WP-K1/K3: finalize -> map -> trajectory, in THAT order, via the one
            # shared implementation (see pyslam/tools/run_outputs.py). Previously
            # map.ply was written BEFORE finalize() and so never saw the closing
            # optimisation; evicted keyframes were also silently missing from it.
            intr = source.intrinsics()
            out = finalize_and_export(run_dir, pipeline, result, intr,
                                       summary_path=os.path.join(run_dir, "summary.json"),
                                       keep_imagery_cache=args.keep_imagery_cache)

            log.info(f"Done. frames={result.n_frames} keyframes={result.n_keyframes} "
                     f"loop_closures={len(result.loop_events)}")
        log.info(f"Run directory: {run_dir}")
        if args.record:
            log.info(f"Recording: {args.record}")


if __name__ == "__main__":
    main()
