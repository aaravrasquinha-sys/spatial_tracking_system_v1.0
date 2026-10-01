"""
Module 2 entry point: static-camera anchor.

  # 0. (no camera) inspect what the map gives us: room frame, walkable grid, top_down.png with wall ids + axis ticks
  python3 run_anchor.py prepare --map maps/site_001 --out anchors/site_001

  # 1. record the static capture with the camera SEATED IN ITS FINAL MOUNT (room empty, camera untouched)
  python3 run_anchor.py capture --realsense --out captures/cam0_a.npz
  python3 run_anchor.py capture --realsense --out captures/cam0_b.npz      # 3-5 separate sessions => true repeatability

  # 2. solve (no camera needed; runs anywhere)
  python3 run_anchor.py solve --map maps/site_001 --capture captures/cam0_*.npz --out anchors/site_001
  #    if the room is symmetric / the solve reports ambiguity, tap-equivalent hint (room frame: x, y, radius m, camera heading deg):
  python3 run_anchor.py solve ... --hint 1.0,3.4,1.0,-25

  # 3. tape-measure check (pixel -> room coordinates vs measurements from map walls)
  python3 run_anchor.py markers --calib anchors/site_001/calibration.cam0.json --room anchors/site_001/room_frame.json --markers markers.json

  # 4. one-shot watchdog check against the stored calibration (camera must be free)
  python3 run_anchor.py check --realsense --anchor anchors/site_001 --map maps/site_001

  # convenience: capture + solve in one go
  python3 run_anchor.py calibrate --realsense --map maps/site_001 --out anchors/site_001 --sessions 3
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np

from pyslam.core.log import get_logger
from pyslam.anchor.config import AnchorConfig, apply_overrides

log = get_logger("run_anchor")


def _factory(args):
    if args.bag:
        def f():
            from pyslam.sensors.realsense import RealSenseSource
            return RealSenseSource(enable_imu=True, playback_path=args.bag)
    elif args.realsense:
        def f():
            from pyslam.sensors.realsense import RealSenseSource
            return RealSenseSource(enable_imu=True, serial=args.serial, imu_capture_mode=args.imu_mode)
    else:
        raise SystemExit("one of --realsense or --bag is required")
    return f


def _cfg(args) -> AnchorConfig:
    cfg = AnchorConfig()
    if getattr(args, "up_prior_map", None):
        cfg = apply_overrides(cfg, [f"up_prior_map={args.up_prior_map}"])
    return apply_overrides(cfg, getattr(args, "set", None))


def _add_cam_args(ap):
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--bag", type=str, default=None, help="use a recorded .bag instead of the live camera (no warm-up)")
    ap.add_argument("--serial", type=str, default=None)
    ap.add_argument("--imu-mode", type=str, default="callback", choices=["callback", "synced"])
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--capture-profile", type=str, default=None,
                    help="capture profile JSON (default: configs/capture_profile.mapping.json, same as run_live_map.py)")
    ap.add_argument("--warmup-s", type=float, default=None)


_DEFAULT_PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "capture_profile.mapping.json")


def _profile_sha(path=None):
    """Hash of the capture profile USED FOR THIS CAPTURE -- the same default file run_live_map.py uses, or
    --capture-profile. NOT read from the map (that would make the profile-match gate trivially true).
    NOTE: this records the profile's identity; RealSenseSource (as of this repo) does not itself push the
    profile to the device, so 'match' means 'same profile file declared', not 'verified on the sensor' -- see
    RUNBOOK_ANCHOR.md, 'Capture profile'."""
    from pyslam.live.capture_profile import CaptureProfile
    p = path or _DEFAULT_PROFILE
    return (CaptureProfile.load(p) if os.path.exists(p) else CaptureProfile()).content_hash()


def _do_capture(args, cfg, out_path, map_dir=None):
    from pyslam.anchor.capture import capture_static
    cap = capture_static(_factory(args), cfg, n_frames=args.frames, warmup_s=args.warmup_s,
                         capture_profile_sha=_profile_sha(getattr(args, "capture_profile", None)), serial=args.serial, is_playback=bool(args.bag))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cap.save(out_path)
    ii = cap.imu_info
    log.info(f"Saved {out_path}: {cap.n_frames} frames, IMU ok={ii.get('ok')} "
             f"({ii.get('reason', 'static, |g|=%.2f' % ii.get('accel_norm_mps2', 0))})")
    return cap


def _parse_hint(s):
    if not s:
        return None
    x, y, r, yaw = [float(v) for v in s.split(",")]
    return (x, y, r, np.radians(yaw))


def _do_solve(args, cfg, caps):
    from pyslam.anchor.map_io import load_map_bundle
    from pyslam.anchor.calibrate import calibrate
    from pyslam.anchor.bundle import write_anchor_bundle
    mb = load_map_bundle(args.map)
    res = calibrate(mb, caps, cfg, hint=_parse_hint(getattr(args, "hint", None)), cam_id=args.cam_id)
    info = write_anchor_bundle(res, args.out, args.cam_id, mb.get("capture_profile_sha"))
    print(json.dumps({"accepted": res.accepted, "sigma": {k: v for k, v in res.sigma.items() if k != "components"},
                      "calibration": info["calibration"]}, indent=2))
    if not res.accepted:
        print("\nREJECTED -- failed gates:")
        for c in res.checks:
            if c.required and not c.passed:
                print(f"  - {c.name}: value={c.value} needs {c.threshold}  {c.note}")
    return 0 if res.accepted else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="map-only products: room frame, walkable grid, viewer points, top_down.png")
    p.add_argument("--map", required=True); p.add_argument("--out", required=True)
    p.add_argument("--up-prior-map", type=str, default=None, help="rough 'up' in the MAP frame, 'x,y,z' (default 0,-1,0)")
    p.add_argument("--set", nargs="*", default=[], help="AnchorConfig overrides KEY=VALUE")

    p = sub.add_parser("capture", help="record a static capture (camera in its mount)")
    _add_cam_args(p); p.add_argument("--out", required=True); p.add_argument("--map", default=None, help="map dir (records its capture-profile hash)")
    p.add_argument("--set", nargs="*", default=[])

    p = sub.add_parser("solve", help="map + capture(s) -> calibration (no camera needed)")
    p.add_argument("--map", required=True); p.add_argument("--capture", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--hint", type=str, default=None, help="x,y,radius_m,heading_deg in the ROOM frame (see top_down.png from `prepare`)")
    p.add_argument("--cam-id", default="cam0"); p.add_argument("--up-prior-map", type=str, default=None)
    p.add_argument("--set", nargs="*", default=[])

    p = sub.add_parser("calibrate", help="capture (N sessions) + solve")
    _add_cam_args(p); p.add_argument("--map", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sessions", type=int, default=3); p.add_argument("--hint", type=str, default=None)
    p.add_argument("--cam-id", default="cam0"); p.add_argument("--up-prior-map", type=str, default=None)
    p.add_argument("--set", nargs="*", default=[])

    p = sub.add_parser("markers", help="tape-measure verification of the calibration")
    p.add_argument("--calib", required=True); p.add_argument("--room", required=True); p.add_argument("--markers", required=True)
    p.add_argument("--tol-m", type=float, default=None)

    p = sub.add_parser("check", help="one-shot watchdog check against a stored calibration")
    _add_cam_args(p); p.add_argument("--anchor", required=True); p.add_argument("--map", required=True)
    p.add_argument("--cam-id", default="cam0"); p.add_argument("--set", nargs="*", default=[])

    args = ap.parse_args()
    try:
        return _run(args)
    except Exception as e:                                   # known, operator-actionable failures get a clean message
        from pyslam.anchor.physics_prior import PhysicsPriorError
        from pyslam.anchor.room_frame import RoomFrameError
        if isinstance(e, (PhysicsPriorError, RoomFrameError, FileNotFoundError, ValueError)):
            print(f"\nERROR ({type(e).__name__}): {e}", file=sys.stderr)
            return 3
        raise


def _run(args):

    if args.cmd == "prepare":
        from pyslam.anchor.map_io import load_map_bundle
        from pyslam.anchor.room_frame import derive_room_frame
        from pyslam.anchor.walkable import build_walkable
        from pyslam.anchor.bundle import write_map_products
        from pyslam.anchor import geo
        cfg = _cfg(args)
        mb = load_map_bundle(args.map)
        keep = geo.statistical_outlier_mask(mb["pts_map"], cfg.sor_k, cfg.sor_std)
        pts, col = mb["pts_map"][keep], None if mb["colors"] is None else mb["colors"][keep]
        rf = derive_room_frame(pts, cfg)
        T = rf.T_room_map
        pr = pts @ T[:3, :3].T + T[:3, 3]
        walk = build_walkable(pr, cfg)
        write_map_products(args.out, rf, pr, col, walk)
        for w in rf.warnings:
            log.warning(w)
        print(f"map_id={mb['map_id']}  walkable={walk['stats']['walkable_area_m2']:.1f} m^2  "
              f"floor rms={rf.floor['rms_m']*1000:.1f} mm  -> {args.out}/top_down.png")
        return 0

    if args.cmd == "capture":
        _do_capture(args, _cfg(args), args.out, args.map)
        return 0

    if args.cmd == "solve":
        from pyslam.anchor.capture import StaticCapture
        paths = sorted({q for pat in args.capture for q in (glob.glob(pat) or [pat])})
        return _do_solve(args, _cfg(args), [StaticCapture.load(p) for p in paths])

    if args.cmd == "calibrate":
        cfg = _cfg(args)
        caps = []
        for i in range(args.sessions):
            if i > 0 and not args.bag:
                input(f"Session {i+1}/{args.sessions}: leave the camera in its mount; press Enter when the room is still...")
            caps.append(_do_capture(args, cfg, os.path.join(args.out, "captures", f"{args.cam_id}_{i}.npz"), args.map))
            args.warmup_s = 5.0 if not args.bag else args.warmup_s     # already warm after the first session
        return _do_solve(args, cfg, caps)

    if args.cmd == "markers":
        from pyslam.anchor.verify import check_markers
        cal = json.load(open(args.calib))
        room = json.load(open(args.room))
        mk = json.load(open(args.markers))
        cfg = AnchorConfig()
        rep = check_markers(mk["markers"] if isinstance(mk, dict) else mk, np.array(cal["T_room_cam"]),
                            cal["intrinsics"], room["walls"], args.tol_m or cfg.gate_marker_err_m)
        print(json.dumps(rep, indent=2))
        return 0 if rep["pass"] else 2

    if args.cmd == "check":
        from pyslam.anchor.capture import capture_static
        from pyslam.anchor.watchdog import AnchorWatchdog
        from pyslam.anchor.map_io import load_map_bundle
        cfg = _cfg(args)
        wd = AnchorWatchdog(os.path.join(args.anchor, f"calibration.{args.cam_id}.json"),
                            os.path.join(args.anchor, f"{args.cam_id}_reference_depth.npz"), cfg)
        cap = capture_static(_factory(args), cfg, n_frames=args.frames or 60, warmup_s=args.warmup_s if args.warmup_s is not None else 3.0,
                             is_playback=bool(args.bag))
        print("tilt_deg:", wd.check_tilt(cap.up_cam))
        print("moved_frac:", wd.check_depth(cap.depth_med))
        mb = load_map_bundle(args.map)
        from pyslam.anchor.room_frame import RoomFrame
        rfj = json.load(open(os.path.join(args.anchor, "room_frame.json")))
        Tr = np.array(rfj["T_room_map"])
        pts = mb["pts_map"] @ Tr[:3, :3].T + Tr[:3, 3]
        rc = wd.recheck_pose(cap, pts, mb["colors"])
        rc.pop("T_measured", None)
        print(json.dumps({"state": wd.state.__dict__, "recheck": rc}, indent=2, default=float))
        return 0 if wd.health_field() == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
