from __future__ import annotations
import argparse
import json
import os
import sys
import time
import multiprocessing

from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.live.capture_profile import CaptureProfile

log = get_logger("run_live_map")


class RealSenseSourceFactory:
    def __init__(self, realsense=False, bag=None, serial=None, record=None, imu_mode="callback"):
        self.realsense = realsense
        self.bag = bag
        self.serial = serial
        self.record = record
        self.imu_mode = imu_mode

    def __call__(self):
        from pyslam.sensors.realsense import RealSenseSource
        if self.bag:
            return RealSenseSource(enable_imu=True, playback_path=self.bag)
        elif self.realsense:
            return RealSenseSource(
                enable_imu=True,
                serial=self.serial,
                record_path=self.record,
                imu_capture_mode=self.imu_mode,
            )
        else:
            raise SystemExit("one of --realsense or --bag is required")


def _build_source_factory(args):
    if not args.realsense and not args.bag:
        raise SystemExit("one of --realsense or --bag is required")
    return RealSenseSourceFactory(
        realsense=args.realsense,
        bag=args.bag,
        serial=args.serial,
        record=args.record,
        imu_mode=args.imu_mode,
    )


def _probe_intrinsics(source_factory):
    src = source_factory()
    try:
        intr = src.intrinsics()
    finally:
        try:
            src.close()
        except Exception:
            pass
    return intr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--bag", type=str, default=None)
    ap.add_argument("--serial", type=str, default=None)
    ap.add_argument("--record", type=str, default=None, help="also record a bag file")
    ap.add_argument("--imu-mode", type=str, default="callback", choices=["callback", "polling"])
    ap.add_argument("--out", type=str, required=True, help="map bundle output directory")
    ap.add_argument("--capture-profile", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs",
                                             "capture_profile.mapping.json"))
    ap.add_argument("--mode", type=str, default="live", choices=["live", "lockstep"],
                     help="'live' = real multi-process architecture (default); "
                          "'lockstep' = single-process, deterministic")
    ap.add_argument("--mp-start-method", type=str, default="fork", choices=["spawn", "fork", "forkserver"],
                         help="multiprocessing start method")
    ap.add_argument("--max-frames", type=int, default=None, help="lockstep mode frame limit")
    
    from pyslam.core.config import add_config_args
    add_config_args(ap)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    capture_profile = CaptureProfile.load(args.capture_profile) if os.path.exists(args.capture_profile) \
        else CaptureProfile()
    capture_profile.save(os.path.join(args.out, "capture_profile.json"))

    cfg = Config()
    if args.config_override:
        from pyslam.core.config import apply_overrides
        cfg = apply_overrides(cfg, args.config_override)

    source_factory = _build_source_factory(args)

    # For live mode with fork, bypass parent probing to prevent librealsense USB deadlock
    if args.mode == "live" and args.mp_start_method == "fork":
        from pyslam.sensors.realsense import Intrinsics
        intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480, depth_scale=0.001, baseline=0.05)
        log.info("Bypassed parent intrinsics probe for clean fork start.")
    elif args.mode == "lockstep":
        from pyslam.live.lockstep import run_lockstep
        source = source_factory()
        intr = source.intrinsics()
        log.info(f"Initialized source and probed intrinsics: {intr}")
        
        result = run_lockstep(cfg, source, max_frames=args.max_frames)
        log.info(f"Lockstep run complete: {result.pipeline_result.n_keyframes} keyframes, "
                 f"{result.dense.fuser.n_voxels()} dense voxels.")
        _lock_and_write(args.out, cfg, capture_profile, result.pipeline, result.pipeline_result,
                        result.dense, result.qa)
        return
    else:
        intr = _probe_intrinsics(source_factory)
        log.info(f"Probed intrinsics: {intr}")

    from pyslam.live.orchestrator import start_live_mapping
    session = start_live_mapping(cfg, source_factory, intr,
                                 out_ply_path=os.path.join(args.out, "dense", "points.ply"),
                                 mp_start_method=args.mp_start_method)
    os.makedirs(os.path.join(args.out, "dense"), exist_ok=True)
    log.info("Live mapping running. Ctrl-C to stop and lock the map.")
    try:
        while True:
            time.sleep(1.0)
            qa = session.poll_qa()
            if qa is not None:
                log.info(f"QA: frames={qa.get('n_frames', 0)} keyframes={qa.get('n_keyframes', 0)} "
                         f"lost={qa.get('n_lost_frames', 0)} "
                         f"icp_fallback={qa.get('n_icp_fallback_recoveries', qa.get('n_icp_fallback', 0))} "
                         f"max_speed={qa.get('max_speed_mps', 0.0):.2f}m/s")
    except KeyboardInterrupt:
        log.warning("Interrupted -- stopping processes and locking whatever was captured.")
    finally:
        session.stop(timeout=20.0)

    summary_data = {
        "note": "WP-LIVE multi-process session -- see dense/points.ply and this run's log for QA history."
    }
    with open(os.path.join(args.out, "session_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
        
    log.info(f"Session stopped. Dense export at {args.out}/dense/points.ply.")


if __name__ == "__main__":
    main()
