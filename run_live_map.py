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


def _qa_warnings(qa: dict) -> list:
    """Plain-language problems found in the QA numbers (goes into session_summary.json)."""
    w = []
    if qa.get("n_lost_frames", 0):
        w.append(f"{qa['n_lost_frames']} frames were LOST: each one joins the trajectory with a weak guess. "
                 f"Walk slower and avoid blank walls.")
    if qa.get("n_rot_over_limit", 0):
        w.append(f"{qa['n_rot_over_limit']} frames turned faster than 30 deg/s (max {qa.get('max_rot_rate_dps', 0):.0f}).")
    if qa.get("n_speed_over_limit", 0):
        w.append(f"{qa['n_speed_over_limit']} frames moved faster than 0.5 m/s (max {qa.get('max_speed_mps', 0):.2f}).")
    if qa.get("n_frames_dropped", 0):
        w.append(f"{qa['n_frames_dropped']} frames never reached the backend (queue full): the machine could not keep up.")
    if qa.get("n_points_dropped", 0):
        w.append(f"{qa['n_points_dropped']} keyframe point clouds never reached the dense map (queue full).")
    return w


def _write_session_summary(out_dir: str, mode: str, qa: dict, extra: dict = None) -> dict:
    """session_summary.json is what `sts map-lock` copies into manifest.json as qa_snapshot_at_lock.
    It is NOT part of the map_id hash (only dense/points.ply + capture_profile.json are)."""
    path = os.path.join(out_dir, "session_summary.json")
    summ = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                summ = json.load(f)
        except Exception:
            summ = {}
    summ.update({"mode": mode, "qa": qa or {}, "warnings": _qa_warnings(qa or {})})
    summ.update(extra or {})
    with open(path, "w") as f:
        json.dump(summ, f, indent=2, default=float)
    return summ


def _lock_and_write(out_dir, cfg, capture_profile, pipeline, pipeline_result, dense, qa, intr=None):
    """End of a --mode lockstep run (this function was called but never defined, so lockstep crashed).
    1. pipeline.finalize + map.ply + trajectory + reloc_map (pyslam.tools.run_outputs, the one shared
       place that enforces 'finalize BEFORE exporting');
    2. push the FINAL poses to the dense map and write dense/points.ply -- the file M2 and the map_id use;
    3. write session_summary.json.
    Locking (manifest + read-only) is still `sts map-lock`, so a bad run can be redone in the same folder."""
    from pyslam.mapping.export import write_ply
    from pyslam.live.lockstep import dense_frame_poses
    from pyslam.tools.run_outputs import finalize_and_export

    os.makedirs(os.path.join(out_dir, "dense"), exist_ok=True)
    summary_path = os.path.join(out_dir, "session_summary.json")
    out = finalize_and_export(out_dir, pipeline, pipeline_result, intr, summary_path=summary_path) \
        if intr is not None else {"finalized": False}

    poses = getattr(pipeline_result, "final_poses", None) or pipeline.graph.get_poses()
    n_re = dense.on_pose_update(dense_frame_poses(poses, pipeline_result.frame_records,
                                                  list(dense._stored.keys()), cfg))
    pts, colors = dense.export_ply_points(min_neighbors=int(getattr(cfg, "dense_export_min_neighbors", 0)))
    ply = os.path.join(out_dir, "dense", "points.ply")
    if pts.shape[0] > 0:
        write_ply(ply, pts, colors)
    else:
        log.warning("Dense map is EMPTY -- no dense/points.ply written.")
    log.info(f"Lockstep bundle written to {out_dir}: finalized={out.get('finalized')}, "
             f"{pts.shape[0]} dense points ({n_re} keyframes re-integrated at final poses).")
    return _write_session_summary(out_dir, "lockstep", qa.snapshot(),
                                  {"finalized": bool(out.get("finalized")), "dense_points": int(pts.shape[0])})


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

        try:
            result = run_lockstep(cfg, source, max_frames=args.max_frames,
                                  R_body_cam=getattr(source, "R_body_cam", None))
        finally:
            try:
                source.close()
            except Exception:
                pass
        log.info(f"Lockstep run complete: {result.pipeline_result.n_keyframes} keyframes, "
                 f"{result.dense.fuser.n_voxels()} dense voxels.")
        _lock_and_write(args.out, cfg, capture_profile, result.pipeline, result.pipeline_result,
                        result.dense, result.qa, intr=intr)
        return
    else:
        intr = _probe_intrinsics(source_factory)
        log.info(f"Probed intrinsics: {intr}")

    from pyslam.live.orchestrator import start_live_mapping
    session = start_live_mapping(cfg, source_factory, intr, capture_profile=capture_profile,
                                 out_ply_path=os.path.join(args.out, "dense", "points.ply"),
                                 mp_start_method=args.mp_start_method)
    os.makedirs(os.path.join(args.out, "dense"), exist_ok=True)
    log.info("Live mapping running. Ctrl-C to stop and save the map (then run `sts map-lock`).")
    last_qa = {}
    try:
        while True:
            time.sleep(1.0)
            qa = session.poll_qa()
            if qa is not None:
                last_qa = qa
                log.info(f"QA: frames={qa.get('n_frames', 0)} keyframes={qa.get('n_keyframes', 0)} "
                         f"lost={qa.get('n_lost_frames', 0)} "
                         f"icp_fallback={qa.get('n_icp_fallback_used', 0)} "
                         f"fps={qa.get('n_frames', 0) / max(qa.get('elapsed_s', 0.0), 1e-3):.1f} "
                         f"max_speed={qa.get('max_speed_mps', 0.0):.2f}m/s "
                         f"max_turn={qa.get('max_rot_rate_dps', 0.0):.0f}deg/s "
                         f"dropped={qa.get('n_frames_dropped', 0)}f/{qa.get('n_points_dropped', 0)}pc")
    except KeyboardInterrupt:
        log.warning("Interrupted -- stopping processes and locking whatever was captured.")
    finally:
        # staged stop: tracker -> backend (drains queue + full-graph finalize) -> dense (exports PLY)
        session.stop(timeout=20.0, backend_timeout=150.0)

    final_qa = session.poll_qa()          # the tracker pushes one last snapshot as it exits
    if final_qa:
        last_qa = final_qa
    summ = _write_session_summary(
        args.out, "live", last_qa,
        {"note": "WP-LIVE multi-process session. Next: `python -m sts map-lock` to write the manifest "
                 "and make the bundle read-only."})
    for w in summ.get("warnings", []):
        log.warning(f"QA: {w}")
    log.info(f"Session stopped. Dense export at {args.out}/dense/points.ply. "
             f"Run `python -m sts map-lock` to lock it.")


if __name__ == "__main__":
    main()
