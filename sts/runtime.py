"""The live chain: capture -> M3 -> M4 -> M5, supervised.

This mirrors scripts/run_present.py::_build_live (which stays untouched) and adds exactly
what the plan's Module 6 asks for, without changing any module:

  * the consistency gate runs BEFORE anything opens the camera (Phase B);
  * the real map_id is handed to LiveSource (the stock script passes None);
  * the calibration watchdog is fed from M3's existing `on_detections` hook
    (composed as: watchdog.observe -> M4.process) and its verdict drives
    health.calib through the additive PresentServer.calib_provider hook;
  * `on_suspect` policy: "suppress" publishes EMPTY tracks while calibration is suspect
    (rejecting beats being confidently wrong); "flag" keeps publishing and lets health say so;
  * if the capture/inference thread dies the process EXITS NON-ZERO so systemd restarts it
    (stock run_present would keep serving a frozen stream forever);
  * a soak monitor records memory/CPU/temperature/drop counters and bounds M4's
    otherwise-unbounded `total_events` list.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

from sts.calib_health import CalibState
from sts.configgen import GeneratedConfigs, anchor_overrides, write_configs
from sts.consistency import ConsistencyReport, check_phase_b
from sts.manifest import write_manifest
from sts.registry import SlotRegistry, default_registry
from sts.site import SiteConfig, SiteError
from sts.watchdog_runtime import WatchdogSupervisor


def _log():
    from pyslam.core.log import get_logger
    return get_logger("sts.runtime")


# --------------------------------------------------------------------------- planning
@dataclass
class RunPlan:
    site: SiteConfig
    cam_id: str
    phase: str
    gen: GeneratedConfigs
    map_id: Optional[str]
    report: Optional[ConsistencyReport]
    manifest_path: Optional[Path] = None


def plan_run(site: SiteConfig, cam_id: Optional[str] = None, phase: Optional[str] = None,
             write_run_manifest: bool = True) -> RunPlan:
    """Resolve ids, generate configs, run the consistency gate. Raises ConsistencyError if a
    Phase-B run is inconsistent. Never touches the camera."""
    cam = site.camera(cam_id)
    phase = phase or site.localization.phase
    map_id = None
    if phase == "B":
        ply = site.map_dir() / "dense" / "points.ply"
        if ply.exists():
            from pyslam.anchor.map_io import resolve_map_id
            map_id = resolve_map_id(str(site.map_dir()))["map_id"]
    gen = write_configs(site, cam.cam_id, phase=phase, map_id=map_id)
    report = None
    if phase == "B":
        report = check_phase_b(site, cam.cam_id, gen.m4_path, gen.present_path)
        report.raise_if_blocked()
    plan = RunPlan(site, cam.cam_id, phase, gen, map_id, report)
    if write_run_manifest:
        plan.manifest_path = write_manifest(site, cam.cam_id, phase, gen, map_id)
    return plan


# --------------------------------------------------------------------------- watchdog assembly
def build_anchor_watchdog(site: SiteConfig, cam_id: str):
    from pyslam.anchor.config import AnchorConfig, apply_overrides
    from pyslam.anchor.watchdog import AnchorWatchdog
    ov = [f"{k}={v}" for k, v in anchor_overrides(site, cam_id).items()]
    cfg = apply_overrides(AnchorConfig(), ov) if ov else AnchorConfig()
    wd = AnchorWatchdog(str(site.calibration_path(cam_id)), str(site.reference_depth_path(cam_id)), cfg)
    return wd, cfg


def make_map_loader(site: SiteConfig, cam_id: str, anchor_cfg) -> Callable[[], tuple]:
    """Lazy loader for the ICP recheck: the full map in the ROOM frame, CROPPED to the camera's
    working volume. Loaded on first use (seconds; the Orin has 8 GB shared with TensorRT), and
    the un-cropped arrays are dropped immediately."""
    def load():
        from pyslam.anchor.map_io import load_map_bundle
        mb = load_map_bundle(str(site.map_dir()))
        rfj = json.loads((site.anchor_dir(cam_id) / "room_frame.json").read_text())
        Tr = np.array(rfj["T_room_map"], float)
        pts = mb["pts_map"] @ Tr[:3, :3].T + Tr[:3, 3]
        cols = mb["colors"]
        cal = json.loads(site.calibration_path(cam_id).read_text())
        cam_xyz = np.array(cal["T_room_cam"], float)[:3, 3]
        r = anchor_cfg.range_max_m + anchor_cfg.icp_crop_margin_m + 1.0
        keep = np.linalg.norm(pts - cam_xyz[None], axis=1) <= r
        out = (pts[keep].copy(), None if cols is None else cols[keep].copy())
        del mb, pts, cols
        return out
    return load


# --------------------------------------------------------------------------- events log
class EventsLog:
    """Append-only JSONL of calibration-state transitions and runtime events, so a track log
    can later be filtered to periods when calibration was trusted."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()
        self.path = path

    def write(self, kind: str, **kw):
        with self._lock:
            self._f.write(json.dumps({"t": time.time(), "kind": kind, **kw}) + "\n")

    def close(self):
        with self._lock:
            self._f.close()


# --------------------------------------------------------------------------- soak monitor
class SoakMonitor(threading.Thread):
    """Every `period_s`: rss/cpu/temperature/thermal-throttle/drop counters -> soak.jsonl.
    This is what the one-hour soak test reads ("flat memory")."""

    def __init__(self, path: Path, chain: "Chain", period_s: float = 10.0):
        super().__init__(name="sts-soak", daemon=True)
        self.path, self.chain, self.period_s = path, chain, period_s
        self._stop_evt = threading.Event()
        path.parent.mkdir(parents=True, exist_ok=True)

    def stop(self):
        self._stop_evt.set()

    def sample(self) -> dict:
        import glob
        import psutil
        proc = psutil.Process()
        s = {"t": time.time(), "rss_mb": round(proc.memory_info().rss / 1e6, 1),
             "cpu_pct": proc.cpu_percent(interval=None), "threads": proc.num_threads(),
             "frames": self.chain.frames_seen, "calib": self.chain.calib.state()}
        temps = []
        for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                temps.append(int(Path(z).read_text()) / 1000.0)
            except (OSError, ValueError):
                pass
        s["temp_c_max"] = max(temps) if temps else None
        d = self.chain.daemon
        if d is not None:
            s["dropped_capture"] = d._cap_queue.dropped_count
            s["dropped_infer"] = d._infer_queue.dropped_count
        sup = self.chain.supervisor
        if sup is not None:
            s["watchdog"] = dict(sup.stats)
        return s

    def run(self):
        while not self._stop_evt.wait(self.period_s):
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(self.sample()) + "\n")
                # M4Pipeline.total_events grows without bound (noted in M5's README); it's a public
                # list that only the demos read, so bound it from outside instead of editing M4.
                ev = self.chain.m4.total_events
                if len(ev) > 2000:
                    del ev[:-1000]
            except Exception as e:                      # monitoring must never take the run down
                _log().warning(f"soak monitor: {e}")


# --------------------------------------------------------------------------- the chain
class Chain:
    def __init__(self):
        self.plan: Optional[RunPlan] = None
        self.calib = CalibState("missing")
        self.supervisor: Optional[WatchdogSupervisor] = None
        self.m3_pipeline = None
        self.m4 = None
        self.live_source = None
        self.server = None
        self.daemon = None
        self.events: Optional[EventsLog] = None
        self.soak: Optional[SoakMonitor] = None
        self.frames_seen = 0
        self.last_n_tracks = 0
        self.frames_suppressed = 0
        self.writers: List[Any] = []
        self._exit_code = 0

    # -- callbacks ---------------------------------------------------------------
    def on_detections(self, frame, dets):
        """M3's on_detections hook: watchdog first (cheap, non-blocking), then M4."""
        self.frames_seen += 1
        if self.supervisor is not None:
            self.supervisor.observe(frame, self.last_n_tracks)
        return self.m4.process(frame, dets)

    def on_tracks(self, frame, records):
        """M4's on_tracks hook: apply the degraded-mode policy, then publish."""
        self.last_n_tracks = len(records)
        publish = records
        if self.supervisor is not None and not self.supervisor.should_publish_3d():
            publish = []                                   # rejecting beats being confidently wrong
            self.frames_suppressed += 1
        if self.live_source is not None:
            self.live_source.submit(frame, publish)

    def on_calib_event(self, kind: str, reason: str):
        if self.events:
            self.events.write(kind, reason=reason)
        if self.server is not None:
            from poi_present.schema import event_message
            self.server.publish_event_threadsafe(event_message(event=kind, track_id=-1, t=time.time(), detail=reason))

    # -- lifecycle --------------------------------------------------------------
    def close(self):
        if self.soak:
            self.soak.stop()
        for w in self.writers:
            try:
                w.close()
            except Exception:
                pass
        if self.events:
            self.events.close()


def build_chain(plan: RunPlan, *, source_kind: str = "realsense", playback: Optional[str] = None,
                serve: bool = True, registry: Optional[SlotRegistry] = None,
                backend=None, source=None, tilt_source=None, inline_watchdog: bool = False,
                write_logs: bool = True) -> Chain:
    """Assemble the chain. `backend`/`source` can be injected (tests, synthetic replay); otherwise they
    come from the `detector_backend` / `capture_source` slots."""
    from poi_localization.config import M4Config
    from poi_localization.io.world_track import JsonlWorldTrackWriter
    from poi_localization.runtime.m4_pipeline import M4Pipeline
    from poi_localization.tracking.gates import load_walkable_grid
    from poi_perception.config import M3Config
    from poi_perception.inference.pose_infer import PoseEstimator
    from poi_perception.io.detection2d import JsonlDetectionWriter
    from poi_perception.runtime.m3_daemon import M3Daemon, M3Pipeline, build_source
    from poi_perception.runtime.masks import empty_mask_set, load_masks

    reg = registry or default_registry()
    site, cam_id, phase = plan.site, plan.cam_id, plan.phase
    chain = Chain()
    chain.plan = plan

    m3_cfg = M3Config.load(plan.gen.m3_path)
    m4_cfg = M4Config.load(plan.gen.m4_path)
    from poi_present.config import PresentConfig
    present_cfg = PresentConfig.load(plan.gen.present_path)

    stamp = int(time.time())
    if write_logs:
        chain.events = EventsLog(site.logs_dir() / cam_id / "events.jsonl")
        chain.events.write("run_start", phase=phase, map_id=plan.map_id)

    # ---- M4 (frame provider slot) ----
    provider = reg.create("frame_provider", site.slots.get("frame_provider")).build(m4_cfg, m3_cfg.camera)
    walkable = None
    if phase == "B" and m4_cfg.phase_b.walkable_grid_path:
        walkable = load_walkable_grid(m4_cfg.phase_b.walkable_grid_path)
    m4_writer = None
    if write_logs:
        m4_writer = JsonlWorldTrackWriter(Path(m4_cfg.output.jsonl_dir) / f"{cam_id}_{stamp}.jsonl", phase=phase)
        chain.writers.append(m4_writer)
    chain.m4 = M4Pipeline(m4_cfg, provider, walkable_grid=walkable, writer=m4_writer, on_tracks=chain.on_tracks)
    T = chain.m4.floor_transform
    if m4_writer is not None:
        m4_writer.map_id = T.map_id
    if phase == "B" and plan.map_id and T.map_id and T.map_id != plan.map_id:
        raise SiteError(f"M4 loaded map_id {T.map_id} but the map bundle is {plan.map_id}")

    # ---- calibration health + watchdog (Phase B only: Phase A has no calibration to verify) ----
    chain.calib = reg.create("calib_state_provider", site.slots.get("calib_state_provider"), "missing")
    wd, acfg = None, None
    if phase == "B" and site.watchdog.enabled:
        layers = reg.create("watchdog_layers", site.slots.get("watchdog_layers", "depth+icp")).parse(
            site.slots.get("watchdog_layers", "depth+icp"))
        wd, acfg = build_anchor_watchdog(site, cam_id)
        loader = make_map_loader(site, cam_id, acfg) if "icp" in layers else None
        tilt = tilt_source if "tilt" in layers else None
        if "tilt" in layers and tilt is None:
            raise SiteError("watchdog layer 'tilt' requested but no tilt_source was supplied (the runtime capture path has "
                            "no IMU; see docs/OPEN_ITEMS.md). Remove 'tilt' from slots.watchdog_layers.")
        spec = site.watchdog
        if tilt is not None and spec.tilt_layer != "external":
            spec = type(spec)(**{**spec.__dict__, "tilt_layer": "external"})
        chain.supervisor = WatchdogSupervisor(spec, chain.calib, wd, anchor_cfg=acfg, map_loader=loader,
                                              on_event=chain.on_calib_event, tilt_source=tilt, inline=inline_watchdog)
    elif phase == "B":
        chain.calib.set("ok", "watchdog disabled in site.json")
    # Phase A stays "missing": a provisional floor frame, not a registered one (matches M5's stock behaviour)

    # ---- M5 (track_source slot) ----
    frame_name = "room" if phase == "B" else "local"
    chain.live_source = reg.create("track_source", site.slots.get("track_source")).build(cam_id, frame_name, T.map_id)
    if serve:
        from poi_present.server.app import PresentServer
        chain.server = PresentServer(present_cfg, source=chain.live_source)
        chain.server.scene.update_camera(T.R.tolist(), T.t.tolist(), T.camera_height_m, frame_name, T.map_id)
        chain.server.calib_provider = chain.calib

    # ---- M3 ----
    if source is None:
        choice = reg.create("capture_source", site.slots.get("capture_source")).choose(
            site, cam_id, playback, synthetic=(source_kind == "synthetic"))
        source = build_source(m3_cfg, choice.kind, choice.playback)
    if backend is None:
        backend = reg.create("detector_backend", site.slots.get("detector_backend")).build(m3_cfg)
    estimator = PoseEstimator(backend)
    mask_set = None
    if m3_cfg.mask.masks_path:
        mask_set = load_masks(m3_cfg.mask.masks_path).get(m3_cfg.camera.cam_id, empty_mask_set())
    m3_writer = None
    if write_logs:
        m3_writer = JsonlDetectionWriter(Path(m3_cfg.output.jsonl_dir) / f"{cam_id}_{stamp}.jsonl")
        chain.writers.append(m3_writer)
    chain.m3_pipeline = M3Pipeline(m3_cfg, mask_set=mask_set, writer=m3_writer, on_detections=chain.on_detections)
    chain.daemon = M3Daemon(m3_cfg, source, estimator, chain.m3_pipeline)
    if write_logs:
        chain.soak = SoakMonitor(site.logs_dir() / cam_id / "soak.jsonl", chain)
    return chain


# --------------------------------------------------------------------------- running
def run_live(chain: Chain) -> int:
    """Blocking. Returns a process exit code: 0 = operator stop, 3 = the capture/inference
    pipeline ended on its own (let systemd restart us)."""
    from poi_present.server.app import install_signal_handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    install_signal_handlers(loop, chain.server)
    if chain.soak:
        chain.soak.start()
    stop_requested = {"v": False}

    async def _run():
        server_task = asyncio.create_task(chain.server.serve_forever())
        daemon_task = loop.run_in_executor(None, chain.daemon.run)

        async def watch():
            await daemon_task                              # returns when capture/inference ended
            if not stop_requested["v"]:
                chain._exit_code = 3
                _log().error("capture/inference pipeline ended unexpectedly -- stopping so the service manager restarts us")
                chain.server.stop()
        watcher = asyncio.create_task(watch())
        try:
            await server_task
            stop_requested["v"] = True
        finally:
            chain.daemon.stop()
            try:
                await daemon_task
            except Exception:
                pass
            watcher.cancel()

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()
        chain.close()
    return chain._exit_code


def run_frames(chain: Chain, max_frames: Optional[int] = None) -> int:
    """Synchronous LOCKSTEP run (replay / regression / tests): capture -> infer -> M3 -> M4 on one thread,
    every frame processed, nothing dropped. Returns frames processed.

    Why not M3Daemon.run()? The live daemon deliberately uses size-1 drop-stale queues (a 200 ms-old frame
    is worse than none), so a fast source -- a bag played with set_real_time(False), or the synthetic
    source -- gets most of its frames DROPPED, and which ones depends on thread timing. That is correct
    for live and useless for regression. Lockstep is what makes `sts replay` reproducible."""
    d = chain.daemon
    n = 0
    try:
        d.pose_estimator.warm_up()
        for frame in d.source:
            d.pipeline.process(frame, d.pose_estimator.infer(frame))
            n += 1
            if max_frames is not None and n >= max_frames:
                break
    finally:
        try:
            d.source.close()
        except Exception:
            pass
        chain.close()
    return n
