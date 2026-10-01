"""python -m sts <command>  -- the operator interface. Run from anywhere; paths resolve from the repo root.

  init | doctor | configs | check | slots | provenance | boundaries | contracts
  map | map-lock | anchor <verb> | accept | engine | run | replay | test | retention
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from sts import __version__
from sts.paths import REPO_ROOT
from sts.site import SiteConfig, SiteError, default_site_path


# ------------------------------------------------------------------ helpers
def _load_site(args) -> SiteConfig:
    path = Path(args.site) if getattr(args, "site", None) else default_site_path()
    return SiteConfig.load(path)


def _site_path(args) -> Path:
    return Path(args.site) if getattr(args, "site", None) else default_site_path()


def _run_command(site: SiteConfig, cmd, cam_key: str = "default", mode: str = "") -> int:
    from sts.camera_lock import CameraBusy, CameraLock
    print(f"$ {cmd.printable()}\n  ({cmd.description})", flush=True)
    lock = None
    if cmd.needs_camera:
        try:
            lock = CameraLock(site.data_root() / "locks", cam_key, mode or cmd.description.split()[0]).acquire()
        except CameraBusy as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 4
    try:
        env = {**os.environ, **cmd.env}
        return subprocess.call(cmd.argv, cwd=str(cmd.cwd), env=env)
    finally:
        if lock:
            lock.release()


def _registry():
    from sts.registry import default_registry
    return default_registry()


# ------------------------------------------------------------------ commands
def cmd_init(args) -> int:
    p = _site_path(args)
    if p.exists() and not args.force:
        print(f"{p} already exists (use --force to overwrite)")
        return 1
    site = SiteConfig()
    site.site = args.name or "site_001"
    site.save(p)
    for d in (site.map_dir().parent, site.anchor_base().parent, site.captures_dir(), site.recordings_dir(),
              site.logs_dir(), site.data_root() / "models" / "engines", site.data_root() / "locks"):
        d.mkdir(parents=True, exist_ok=True)
    print(f"wrote {p}\ncreated data directories under {site.data_root()}\nNext: python -m sts doctor")
    return 0


def cmd_doctor(args) -> int:
    from sts.doctor import Doctor
    try:
        site = _load_site(args)
    except SiteError as e:
        site = None
        print(f"[FAIL] site.json: {e}")
    d = Doctor(site, cam_id=args.cam, camera=args.camera)
    d.run()
    print(d.render())
    return 0 if d.ok else 1


def cmd_configs(args) -> int:
    from sts.configgen import write_configs
    site = _load_site(args)
    cam = site.camera(args.cam)
    g = write_configs(site, cam.cam_id, phase=args.phase)
    print(json.dumps(g.as_dict(), indent=2))
    return 0


def cmd_check(args) -> int:
    from sts.configgen import write_configs
    from sts.consistency import check_phase_b
    site = _load_site(args)
    cam = site.camera(args.cam)
    from pyslam.anchor.map_io import resolve_map_id
    mid = None
    if (site.map_dir() / "dense" / "points.ply").exists():
        mid = resolve_map_id(str(site.map_dir()))["map_id"]
    g = write_configs(site, cam.cam_id, phase="B", map_id=mid)
    rep = check_phase_b(site, cam.cam_id, g.m4_path, g.present_path)
    print(rep.render())
    return 0 if rep.ok else 1


def cmd_slots(args) -> int:
    from sts.slots import SLOTS
    reg = _registry()
    site = None
    try:
        site = _load_site(args)
    except SiteError:
        pass
    for name, spec in SLOTS.items():
        chosen = (site.slots.get(name) if site else None) or spec.default
        print(f"{name:24s} [{spec.kind:7s}] chosen={chosen:22s} built={reg.names(name)} reserved={list(spec.reserved)}")
    return 0


def cmd_provenance(args) -> int:
    from sts.provenance import by_module, compare
    d = compare()
    print(json.dumps({"summary": by_module(d), **d}, indent=2))
    return 0


def cmd_boundaries(args) -> int:
    from sts.boundaries import check
    v = check()
    for f, pkg, imp in v:
        print(f"FORBIDDEN: {f} ({pkg}) imports {imp}")
    print("import boundaries: OK" if not v else f"{len(v)} violation(s)")
    return 0 if not v else 1


def cmd_contracts(args) -> int:
    from sts.contracts import ALL, validate_file
    bad = 0
    target = Path(args.file) if args.file else None
    if target:
        errs = validate_file(target, args.schema)
        print("OK" if not errs else "\n".join(errs))
        return 0 if not errs else 1
    print("schemas:", ", ".join(ALL))
    return bad


def cmd_map(args) -> int:
    site = _load_site(args)
    reg = _registry()
    builder = reg.create("map_builder", site.slots.get("map_builder"))
    cmd = builder.build(site, args.cam, extra_args=args.extra, record=not args.no_record)
    return _run_command(site, cmd, mode="map")


def cmd_map_lock(args) -> int:
    site = _load_site(args)
    fin = _registry().create("map_finalizer", site.slots.get("map_finalizer"))
    info = fin.finalize(site, read_only=not args.writable)
    print(json.dumps(info, indent=2))
    site.map.map_id = info["map_id"]
    site.save(_site_path(args))
    print(f"site.json map.map_id = {info['map_id']}")
    return 0


def cmd_anchor(args) -> int:
    site = _load_site(args)
    solver = _registry().create("anchor_solver", site.slots.get("anchor_solver"))
    cmd = solver.command(args.verb, site, args.cam, bag=args.bag, frames=args.frames, warmup_s=args.warmup_s,
                         hint=args.hint, capture_paths=args.capture, up_prior_map=args.up_prior_map,
                         markers=args.markers, sessions=args.sessions)
    rc = _run_command(site, cmd, mode=f"anchor {args.verb}")
    if rc == 0 and args.verb in ("solve", "calibrate"):
        print("\naccepted. Next: `python -m sts accept` (records map_id in site.json), then the tape check "
              "(`sts anchor markers --markers markers.json`).")
    return rc


def cmd_accept(args) -> int:
    from sts.configgen import write_configs
    from sts.consistency import check_phase_b
    from pyslam.anchor.map_io import resolve_map_id
    site = _load_site(args)
    cam = site.camera(args.cam)
    info = resolve_map_id(str(site.map_dir()))
    site.map.map_id = info["map_id"]
    g = write_configs(site, cam.cam_id, phase="B", map_id=info["map_id"])
    rep = check_phase_b(site, cam.cam_id, g.m4_path, g.present_path)
    print(rep.render())
    if rep.ok:
        site.save(_site_path(args))
        print(f"\nsite.json updated: map.map_id = {info['map_id']}")
    return 0 if rep.ok else 1


def cmd_engine(args) -> int:
    site = _load_site(args)
    cam = site.camera(args.cam)
    eng = site.engine_path()
    eng.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "poi_perception.models.export_engine", "--weights", args.weights or site.perception.weights_path,
            "--out", str(eng), "--width", str(cam.width), "--height", str(cam.height), "--precision", args.precision]
    from sts.adapters.offline import Command
    return _run_command(site, Command(argv, f"export TensorRT engine ON THIS DEVICE -> {eng}"))


def _synthetic_parts(site, cam_id, scenario: str, frames: int):
    from poi_perception.capture.synthetic_source import SyntheticSource
    from poi_perception.config import M3Config
    from poi_perception.inference import mock_infer
    sc = {"single_loop": mock_infer.scenario_single_loop, "two_crossing": mock_infer.scenario_two_crossing,
          "exit_reenter": mock_infer.scenario_exit_reenter, "sitting": mock_infer.scenario_sitting,
          "partial_occlusion": mock_infer.scenario_partial_occlusion, "near_mirror": mock_infer.scenario_near_mirror}
    if scenario not in sc:
        raise SiteError(f"unknown scenario {scenario!r}; one of {sorted(sc)}")
    cam = site.camera(cam_id)
    src = SyntheticSource(M3Config().camera.__class__(cam_id=cam.cam_id, width=cam.width, height=cam.height, fps=cam.fps,
                                                      expected_fx=cam.expected_fx, expected_fy=cam.expected_fy,
                                                      expected_cx=cam.expected_cx, expected_cy=cam.expected_cy), n_frames=frames)
    be = mock_infer.MockPoseBackend(sc[scenario](frame_w=cam.width, frame_h=cam.height, n_frames=frames))
    return src, be


def cmd_run(args) -> int:
    from sts.consistency import ConsistencyError
    from sts.runtime import build_chain, plan_run, run_live
    site = _load_site(args)
    try:
        plan = plan_run(site, args.cam, phase=args.phase)
    except ConsistencyError as e:
        print(e, file=sys.stderr)
        print("\nREFUSING TO START: fix the failures above (or run --phase A for an unregistered, provisional floor frame).",
              file=sys.stderr)
        return 2
    if plan.report is not None:
        print(plan.report.render())
    print(f"run manifest: {plan.manifest_path}")
    source = backend = None
    if args.source == "synthetic":
        source, backend = _synthetic_parts(site, plan.cam_id, args.mock_scenario, args.frames)
    from sts.camera_lock import CameraBusy, CameraLock
    lock = None
    if args.source == "realsense" and not args.playback:
        try:
            lock = CameraLock(site.data_root() / "locks", "default", "run").acquire()
        except CameraBusy as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 4
    try:
        chain = build_chain(plan, source_kind=args.source, playback=args.playback, serve=True, source=source, backend=backend)
        print(f"serving on http://{site.present.host}:{site.present.port}/  (phase {plan.phase}, calib={chain.calib.state()})")
        return run_live(chain)
    finally:
        if lock:
            lock.release()


def cmd_replay(args) -> int:
    from sts.replay import compare, replay_bag, summarize
    from sts.runtime import plan_run
    site = _load_site(args)
    plan = plan_run(site, args.cam, phase=args.phase, write_run_manifest=False)
    log_path = replay_bag(plan, args.bag, max_frames=args.max_frames)
    cur = summarize(log_path)
    print(json.dumps({"log": str(log_path), "summary": cur}, indent=2))
    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps(cur, indent=2))
        print(f"baseline saved -> {args.save_baseline}")
    if args.compare:
        res = compare(json.loads(Path(args.compare).read_text()), cur)
        print(json.dumps(res, indent=2))
        return 1 if res["regressed"] else 0
    return 0


def cmd_test(args) -> int:
    suites = {
        "unit": [sys.executable, "-m", "pytest", "tests/unit", "-q"],
        "integration": [sys.executable, "-m", "pytest", "tests/integration", "-q"],
        "selftest": [sys.executable, "-m", "pyslam.selftest"],
        "g_live": [sys.executable, "-m", "tests.gates.test_g_live"],
        "g_anchor": [sys.executable, "-m", "tests.gates.test_g_anchor"],
    }
    chosen = args.suite or ["unit", "integration"]
    if "all" in chosen:
        chosen = list(suites)
    rc = 0
    for s in chosen:
        print(f"\n=== {s} ===", flush=True)
        r = subprocess.call(suites[s], cwd=str(REPO_ROOT))
        rc = rc or r
    return rc


def cmd_retention(args) -> int:
    from sts.retention import prune
    site = _load_site(args)
    items = prune(site, dry_run=not args.apply)
    for why, p in items:
        print(f"{'DELETED' if args.apply else 'would delete'}  {p}  ({why})")
    print(f"{len(items)} file(s)")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=None, help="site.json path (default: configs/site.json)")
    ap.add_argument("--version", action="version", version=f"sts {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cam(p):
        p.add_argument("--cam", default=None, help="cam_id (default: the only enabled camera)")

    p = sub.add_parser("init", help="write a default configs/site.json + data dirs"); p.add_argument("--name"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_init)
    p = sub.add_parser("doctor", help="preflight checks"); cam(p); p.add_argument("--camera", action="store_true", help="also probe the D435i (takes the camera lock)"); p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("configs", help="generate the per-camera M3/M4/M5 configs"); cam(p); p.add_argument("--phase", choices=["A", "B"]); p.set_defaults(fn=cmd_configs)
    p = sub.add_parser("check", help="Phase-B consistency gate (map == calibration == M4 == M5)"); cam(p); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("slots", help="list slots, chosen + built implementations"); p.set_defaults(fn=cmd_slots)
    p = sub.add_parser("provenance", help="legacy files changed since the merge"); p.set_defaults(fn=cmd_provenance)
    p = sub.add_parser("boundaries", help="import-boundary check"); p.set_defaults(fn=cmd_boundaries)
    p = sub.add_parser("contracts", help="validate a file against a contract schema"); p.add_argument("--file"); p.add_argument("--schema"); p.set_defaults(fn=cmd_contracts)

    p = sub.add_parser("map", help="M1: live mapping (records a bag by default)"); cam(p); p.add_argument("--no-record", action="store_true")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra args for run_live_map.py after `--`"); p.set_defaults(fn=cmd_map)
    p = sub.add_parser("map-lock", help="M1: write manifest.json + map_id, make the bundle read-only"); p.add_argument("--writable", action="store_true"); p.set_defaults(fn=cmd_map_lock)
    p = sub.add_parser("anchor", help="M2: prepare|capture|solve|calibrate|markers|check"); cam(p)
    p.add_argument("verb", choices=["prepare", "capture", "solve", "calibrate", "markers", "check"])
    p.add_argument("--bag"); p.add_argument("--frames", type=int); p.add_argument("--warmup-s", type=float, dest="warmup_s")
    p.add_argument("--hint"); p.add_argument("--capture", nargs="+"); p.add_argument("--up-prior-map", dest="up_prior_map")
    p.add_argument("--markers"); p.add_argument("--sessions", type=int); p.set_defaults(fn=cmd_anchor)
    p = sub.add_parser("accept", help="after an accepted anchor: record map_id in site.json and run the gate"); cam(p); p.set_defaults(fn=cmd_accept)
    p = sub.add_parser("engine", help="M3: build the TensorRT engine on this Orin"); cam(p); p.add_argument("--weights"); p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"]); p.set_defaults(fn=cmd_engine)
    p = sub.add_parser("run", help="live chain: capture -> M3 -> M4 -> M5 (+watchdog)"); cam(p)
    p.add_argument("--phase", choices=["A", "B"]); p.add_argument("--source", choices=["realsense", "synthetic"], default="realsense")
    p.add_argument("--playback", help="replay a .bag through the live chain (no camera)")
    p.add_argument("--mock-scenario", default="two_crossing"); p.add_argument("--frames", type=int, default=600); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("replay", help="server-less run over a bag; summarise / compare a baseline"); cam(p)
    p.add_argument("--bag", required=True); p.add_argument("--phase", choices=["A", "B"]); p.add_argument("--max-frames", type=int)
    p.add_argument("--save-baseline"); p.add_argument("--compare"); p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("test", help="run test suites"); p.add_argument("suite", nargs="*", choices=["unit", "integration", "selftest", "g_live", "g_anchor", "all"]); p.set_defaults(fn=cmd_test)
    p = sub.add_parser("retention", help="prune logs/bags older than site.retention"); p.add_argument("--apply", action="store_true"); p.set_defaults(fn=cmd_retention)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except SiteError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
