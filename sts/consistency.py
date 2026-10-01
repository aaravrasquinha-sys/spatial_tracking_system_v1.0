"""The system-wide consistency gate: refuse to go live on a registered (Phase B) run
unless the map, the calibration, the walkable grid, the M4 config and the M5 scene
all describe the SAME map.

M4 already refuses a mismatched calibration (CalibrationMapIdMismatch) if you set
`expected_map_id`. This extends that discipline to everything else, and -- unlike
running the modules by hand -- derives the expected id itself, so it can't be left
unset. Severity: "error" blocks the run; "warn" is reported but does not block.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from sts.site import SiteConfig


@dataclass
class Check:
    name: str
    ok: bool
    severity: str = "error"      # "error" | "warn"
    detail: str = ""


@dataclass
class ConsistencyReport:
    cam_id: str
    map_id: Optional[str] = None
    map_id_source: Optional[str] = None
    checks: List[Check] = field(default_factory=list)

    def add(self, name, ok, detail="", severity="error"):
        self.checks.append(Check(name, bool(ok), severity, detail))

    @property
    def errors(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "error"]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"consistency gate  cam={self.cam_id}  map_id={self.map_id} ({self.map_id_source})"]
        for c in self.checks:
            mark = "PASS" if c.ok else ("FAIL" if c.severity == "error" else "warn")
            lines.append(f"  [{mark}] {c.name}" + (f": {c.detail}" if c.detail else ""))
        lines.append("RESULT: " + ("OK" if self.ok else f"BLOCKED ({len(self.errors)} error(s))"))
        return "\n".join(lines)

    def raise_if_blocked(self):
        if not self.ok:
            raise ConsistencyError(self.render())


class ConsistencyError(RuntimeError):
    pass


def _load_json(path: Path):
    return json.loads(path.read_text())


def check_phase_b(site: SiteConfig, cam_id: str, m4_cfg_path: Optional[Path] = None,
                  present_cfg_path: Optional[Path] = None) -> ConsistencyReport:
    rep = ConsistencyReport(cam_id=cam_id)
    cam = site.camera(cam_id)
    map_dir = site.map_dir()
    anchor_dir = site.anchor_dir(cam_id)

    # ---- map bundle --------------------------------------------------------------
    ply = map_dir / "dense" / "points.ply"
    rep.add("map bundle has dense/points.ply", ply.exists(), str(ply))
    if ply.exists():
        from pyslam.anchor.map_io import resolve_map_id
        info = resolve_map_id(str(map_dir))
        rep.map_id, rep.map_id_source = info["map_id"], info["source"]
        rep.add("map bundle unchanged since locking", not info["mismatch"],
                f"manifest says {info['map_id']} but its files hash to {info['recomputed']}: dense/points.ply or "
                f"capture_profile.json was edited after locking. Re-map (or re-lock) and re-anchor." if info["mismatch"] else "")
        if site.map.map_id:
            rep.add("site.json map.map_id matches the map bundle", site.map.map_id == info["map_id"],
                    f"site.json={site.map.map_id} bundle={info['map_id']}")

    # ---- calibration -------------------------------------------------------------
    calp = site.calibration_path(cam_id)
    rejected = anchor_dir / f"calibration.{cam_id}.REJECTED.json"
    if not calp.exists():
        rep.add("accepted calibration exists", False,
                f"{calp} missing" + (" -- a REJECTED calibration is present: fix the failed gates and re-solve" if rejected.exists() else
                                     " -- run `sts anchor calibrate`"))
        return rep
    cal = _load_json(calp)
    rep.add("accepted calibration exists", True, str(calp))
    rep.add("calibration.accepted is true", cal.get("accepted") is True, f"accepted={cal.get('accepted')!r}")
    rep.add("calibration cam_id matches", cal.get("cam_id", cam_id) == cam_id, f"file says {cal.get('cam_id')!r}")
    if rep.map_id:
        rep.add("calibration map_id == map bundle map_id", cal.get("map_id") == rep.map_id,
                f"calibration={cal.get('map_id')} map={rep.map_id}: the anchor was solved against a different map")
    if site.cameras and cam.serial and cal.get("serial"):
        rep.add("calibration serial matches site.json camera serial", str(cal["serial"]) == str(cam.serial),
                f"calibration={cal['serial']} site={cam.serial}")

    T = np.array(cal.get("T_room_cam", []), float)
    okT = T.shape == (4, 4) and np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-3) and np.linalg.det(T[:3, :3]) > 0
    rep.add("T_room_cam is a proper rigid transform", okT)
    if okT:
        h = float(T[2, 3])
        rep.add("camera height above floor is plausible (0.3..4 m)", 0.3 < h < 4.0, f"z={h:.3f} m")
    sg = cal.get("sigma", {})
    rep.add("calibration carries sigma{trans_m, rot_deg}", {"trans_m", "rot_deg"} <= set(sg))
    if {"trans_m", "rot_deg"} <= set(sg):
        rep.add("sigma within M2's own acceptance (3 cm / 0.5 deg)", sg["trans_m"] <= 0.03 and sg["rot_deg"] <= 0.5,
                f"sigma={sg}", severity="warn")
    intr = cal.get("intrinsics", {})
    if intr:
        d = max(abs(intr.get("fx", 0) - cam.expected_fx), abs(intr.get("fy", 0) - cam.expected_fy),
                abs(intr.get("cx", 0) - cam.expected_cx), abs(intr.get("cy", 0) - cam.expected_cy))
        rep.add("calibration intrinsics agree with site.json expected intrinsics", d <= cam.intrinsics_tol_px,
                f"max difference {d:.2f}px (tol {cam.intrinsics_tol_px}px)", severity="warn")

    # ---- walkable / reference / viewer assets -----------------------------------------
    wp = site.walkable_path(cam_id)
    wok = False
    if wp.exists():
        try:
            w = _load_json(wp)
            wok = {"origin", "resolution", "grid"} <= set(w) and len(w["grid"]) > 0
        except Exception:
            wok = False
    rep.add("walkable.json exists with {origin, resolution, grid}", wok, str(wp))
    rp = site.reference_depth_path(cam_id)
    rep.add("reference depth exists (calibration watchdog)", rp.exists(), str(rp),
            severity="error" if site.watchdog.enabled else "warn")
    rep.add("anchor viewer/points.ply exists (room-frame cloud for M5)", (anchor_dir / "viewer" / "points.ply").exists(),
            "VR/dashboard will show tracks without a room cloud", severity="warn")

    # ---- generated module configs ---------------------------------------------------
    if m4_cfg_path is not None:
        m4 = _load_json(Path(m4_cfg_path))
        pb = m4.get("phase_b", {})
        rep.add("M4 phase is B", m4.get("phase") == "B", f"phase={m4.get('phase')!r}")
        if rep.map_id:
            rep.add("M4 expected_map_id == map_id", pb.get("expected_map_id") == rep.map_id,
                    f"M4 expects {pb.get('expected_map_id')}")
        rep.add("M4 calibration_path is this site's calibration", Path(pb.get("calibration_path", "")).resolve() == calp.resolve(),
                str(pb.get("calibration_path")))
        rep.add("M4 walkable grid is this site's walkable.json", Path(pb.get("walkable_grid_path", "")).resolve() == wp.resolve(),
                str(pb.get("walkable_grid_path")))
    if present_cfg_path is not None:
        pr = _load_json(Path(present_cfg_path))
        sc = pr.get("scene", {})
        rep.add("M5 frame is 'room'", sc.get("frame") == "room", f"frame={sc.get('frame')!r}")
        if rep.map_id:
            rep.add("M5 scene map_id == map_id", sc.get("map_id") == rep.map_id, f"scene map_id={sc.get('map_id')}")
        bd = sc.get("map_bundle_dir")
        rep.add("M5 map_bundle_dir is the ANCHOR dir, not the map dir",
                bd is not None and Path(bd).resolve() == anchor_dir.resolve() and Path(bd).resolve() != map_dir.resolve(),
                f"map_bundle_dir={bd}: maps/<site>/dense/points.ply is in M1's W_cam0 frame (y-down); "
                f"only anchors/<site>/<cam>/viewer/points.ply is in the room frame the tracks use")
    return rep
