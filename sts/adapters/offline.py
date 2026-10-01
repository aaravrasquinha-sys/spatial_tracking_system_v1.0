"""Offline-slot adapters: map_builder, map_finalizer, anchor_solver,
anchor_capture_filter, viewer_asset_producer.

The two that drive hardware (map_builder, anchor capture) return a `Command`; the CLI
runs it under the camera lock. Nothing here touches the camera itself.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from sts.paths import REPO_ROOT
from sts.site import SiteConfig, SiteError

MAP_CONTENT_FILES = ["dense/points.ply", "capture_profile.json"]   # THE map_id rule; see contracts/map_bundle.md


@dataclass
class Command:
    argv: List[str]
    description: str
    needs_camera: bool = False
    cwd: Path = REPO_ROOT
    env: Dict[str, str] = field(default_factory=dict)

    def printable(self) -> str:
        return " ".join(_q(a) for a in self.argv)


def _q(a: str) -> str:
    return a if a and all(c.isalnum() or c in "-_./=:,@+" for c in a) else repr(a)


def _legacy_defines(script: str, name: str) -> bool:
    """True if `def <name>` appears in a legacy top-level script (used to refuse a known-broken code path)."""
    try:
        return f"def {name}" in (REPO_ROOT / script).read_text()
    except OSError:
        return False


def _set_args(overrides: Dict[str, object]) -> List[str]:
    return [f"{k}={v}" for k, v in overrides.items()]


# ------------------------------------------------------------------ map_builder
class PyslamLiveMapBuilder:
    """Wraps run_live_map.py. `--record` is ON by default: a bag is the only way to repair a map with
    holes (re-fuse) or to rebuild it with another pipeline later (slot map_builder, reserved names)."""

    def build(self, site: SiteConfig, cam_id: Optional[str], extra_args: Optional[List[str]] = None,
              record: bool = True, bag_name: Optional[str] = None) -> Command:
        cam = site.camera(cam_id)
        out = site.map_dir()
        argv = [sys.executable, "run_live_map.py", "--realsense", "--out", str(out),
                "--capture-profile", str(site.capture_profile_path()),
                "--imu-mode", "callback"]          # 'polling' is rejected by RealSenseSource (RUNBOOK_ANCHOR Part 0)
        if cam.serial:
            argv += ["--serial", cam.serial]
        if record:
            bag = site.recordings_dir() / (bag_name or f"map_{time.strftime('%Y%m%d_%H%M%S')}.bag")
            bag.parent.mkdir(parents=True, exist_ok=True)
            argv += ["--record", str(bag)]
        extra = [a for a in (extra_args or []) if a != "--"]
        if any(a == "lockstep" for a in extra) and not _legacy_defines("run_live_map.py", "_lock_and_write"):
            raise SiteError(
                "run_live_map.py --mode lockstep calls _lock_and_write(), which is not defined anywhere in this repo "
                "(a known defect in the uploaded run_live_map.py: it would crash with NameError AFTER mapping finished, "
                "losing the in-memory result). Map with the default live mode and lock with `sts map-lock`. "
                "See docs/OPEN_ITEMS.md #K1.")
        argv += extra
        return Command(argv, f"M1 live mapping -> {out}", needs_camera=True)


# ------------------------------------------------------------------ map_finalizer
class ManifestLockFinalizer:
    """Writes manifest.json + map_id and makes the bundle read-only, using M1's own
    `lock_bundle` and the SAME content files M2's resolve_map_id hashes, so the id it produces
    is the id M2 already computes for an unlocked bundle (contracts/map_bundle.md)."""

    def finalize(self, site: SiteConfig, read_only: bool = True) -> dict:
        from pyslam.live.capture_profile import CaptureProfile
        from pyslam.mapping.lock import compute_map_id, lock_bundle
        d = site.map_dir()
        ply = d / "dense" / "points.ply"
        if not ply.exists():
            raise SiteError(f"{ply} not found -- run `sts map` first")
        manifest = d / "manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text()).get("map_id")
            now = compute_map_id(str(d), MAP_CONTENT_FILES)
            if existing == now:
                return {"map_id": existing, "manifest_path": str(manifest), "already_locked": True}
            raise SiteError(f"{manifest} exists (map_id {existing}) but the bundle now hashes to {now}: "
                            f"a locked map was edited. Re-mapping produces a NEW bundle; never edit a locked one.")
        prof_path = d / "capture_profile.json"
        sha = CaptureProfile.load(str(prof_path)).content_hash() if prof_path.exists() else None
        qa = {}
        summ = d / "session_summary.json"
        if summ.exists():
            qa = json.loads(summ.read_text())
        info = lock_bundle(str(d), MAP_CONTENT_FILES, sha, site_frame_summary={}, plane_summary=[],
                           qa_snapshot=qa, builder_repo_root=str(REPO_ROOT), make_read_only=read_only)
        info["already_locked"] = False
        return info


class NoopFinalizer:
    def finalize(self, site: SiteConfig, read_only: bool = True) -> dict:
        from pyslam.anchor.map_io import resolve_map_id
        info = resolve_map_id(str(site.map_dir()))
        return {"map_id": info["map_id"], "manifest_path": None, "already_locked": False}


# ------------------------------------------------------------------ anchor_solver
class PyslamAnchorSolver:
    """Wraps run_anchor.py. Every sub-command gets the per-camera anchor dir, the site's capture
    profile and serial, and the merged AnchorConfig overrides (camera_model + site.anchor.set)."""

    def _cam_args(self, site: SiteConfig, cam_id: str, bag: Optional[str], frames, warmup) -> List[str]:
        cam = site.camera(cam_id)
        a = ["--bag", bag] if bag else ["--realsense"]
        if cam.serial and not bag:
            a += ["--serial", cam.serial]
        a += ["--imu-mode", "callback", "--capture-profile", str(site.capture_profile_path())]
        if frames:
            a += ["--frames", str(frames)]
        if warmup is not None:
            a += ["--warmup-s", str(warmup)]
        return a

    def _overrides(self, site, cam_id) -> List[str]:
        from sts.configgen import anchor_overrides
        ov = anchor_overrides(site, cam_id)
        return (["--set"] + _set_args(ov)) if ov else []

    def command(self, verb: str, site: SiteConfig, cam_id: Optional[str] = None, *, bag: Optional[str] = None,
                frames: Optional[int] = None, warmup_s: Optional[float] = None, hint: Optional[str] = None,
                capture_paths: Optional[List[str]] = None, up_prior_map: Optional[str] = None,
                markers: Optional[str] = None, sessions: Optional[int] = None, out_capture: Optional[str] = None) -> Command:
        cam = site.camera(cam_id)
        cid = cam.cam_id
        anchor = site.anchor_dir(cid)
        py = [sys.executable, "run_anchor.py"]
        if verb == "prepare":
            argv = py + ["prepare", "--map", str(site.map_dir()), "--out", str(anchor)]
            if up_prior_map:
                argv += ["--up-prior-map", up_prior_map]
            return Command(argv + self._overrides(site, cid), f"M2 prepare (map-only, no camera) -> {anchor}")
        if verb == "capture":
            out = out_capture or str(site.captures_dir() / f"{cid}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            argv = py + ["capture"] + self._cam_args(site, cid, bag, frames, warmup_s) + \
                   ["--map", str(site.map_dir()), "--out", out]
            return Command(argv + self._overrides(site, cid), f"M2 static capture -> {out}", needs_camera=not bag)
        if verb == "solve":
            caps = capture_paths or [str(site.captures_dir() / f"{cid}_*.npz")]
            argv = py + ["solve", "--map", str(site.map_dir()), "--capture", *caps, "--out", str(anchor), "--cam-id", cid]
            if hint:
                argv += ["--hint", hint]
            if up_prior_map:
                argv += ["--up-prior-map", up_prior_map]
            return Command(argv + self._overrides(site, cid), f"M2 solve -> {anchor}")
        if verb == "calibrate":
            argv = py + ["calibrate"] + self._cam_args(site, cid, bag, frames, warmup_s) + \
                   ["--map", str(site.map_dir()), "--out", str(anchor), "--sessions", str(sessions or site.anchor.sessions),
                    "--cam-id", cid]
            if hint:
                argv += ["--hint", hint]
            if up_prior_map:
                argv += ["--up-prior-map", up_prior_map]
            return Command(argv + self._overrides(site, cid), f"M2 capture+solve -> {anchor}", needs_camera=not bag)
        if verb == "markers":
            if not markers:
                raise SiteError("--markers FILE is required")
            argv = py + ["markers", "--calib", str(site.calibration_path(cid)), "--room", str(anchor / "room_frame.json"),
                         "--markers", markers]
            return Command(argv, "M2 tape-measure check (independent of the gates)")
        if verb == "check":
            argv = py + ["check"] + self._cam_args(site, cid, bag, frames, warmup_s) + \
                   ["--anchor", str(anchor), "--map", str(site.map_dir()), "--cam-id", cid]
            return Command(argv + self._overrides(site, cid), "M2 one-shot watchdog check", needs_camera=not bag)
        raise SiteError(f"unknown anchor verb {verb!r}")


# ------------------------------------------------------------------ small slots
class NoCaptureFilter:
    """`anchor_capture_filter: none` -- the room must be EMPTY during M2 capture."""
    def mask(self, frame):
        return None


class PointsOnlyAssets:
    """M2 already writes viewer/points.ply in the room frame. No room.glb exists (see OPEN_ITEMS)."""

    def produce(self, site: SiteConfig, cam_id: str) -> List[Path]:
        v = site.anchor_dir(cam_id) / "viewer"
        return [p for p in (v / "points.ply", v / "room.glb") if p.exists()]


def register(reg) -> None:
    reg.register("map_builder", "pyslam_live", PyslamLiveMapBuilder)
    reg.register("map_finalizer", "manifest_lock", ManifestLockFinalizer)
    reg.register("map_finalizer", "noop", NoopFinalizer)
    reg.register("anchor_solver", "pyslam_anchor", PyslamAnchorSolver)
    reg.register("anchor_capture_filter", "none", NoCaptureFilter)
    reg.register("viewer_asset_producer", "points_only", PointsOnlyAssets)
