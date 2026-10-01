"""Run manifest: ties any track log or bug report to the exact code, config and data
that produced it. Written next to the logs at the start of every `sts run`."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from sts import __version__
from sts.paths import REPO_ROOT


def _sha(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _git() -> Dict[str, Any]:
    def run(*a):
        try:
            r = subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _ver(mod: str) -> Optional[str]:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "present")
    except Exception:
        return None


def _jetson() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    p = Path("/proc/device-tree/model")
    if p.exists():
        out["model"] = p.read_bytes().rstrip(b"\0").decode(errors="replace")
    t = Path("/etc/nv_tegra_release")
    if t.exists():
        out["l4t"] = t.read_text().strip().splitlines()[0]
    return out


def build_manifest(site, cam_id: str, phase: str, gen, map_id: Optional[str], extra: Optional[dict] = None) -> dict:
    from sts.provenance import compare, by_module
    cfg_hashes = {n: _sha(Path(p)) for n, p in (("m3", gen.m3_path), ("m4", gen.m4_path), ("present", gen.present_path))}
    cal = site.calibration_path(cam_id)
    eng = site.engine_path()
    eng_manifest = eng.with_suffix(".manifest.json")
    m = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sts_version": __version__,
        "site": site.site, "cam_id": cam_id, "phase": phase, "map_id": map_id,
        "git": _git(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": {k: _ver(k) for k in ("numpy", "scipy", "cv2", "websockets", "psutil", "pyrealsense2", "tensorrt", "gtsam")},
        "jetson": _jetson(),
        "config_hashes": cfg_hashes,
        "calibration": {"path": str(cal), "sha": _sha(cal)} if cal.exists() else None,
        "engine": {"path": str(eng), "sha": _sha(eng),
                   "manifest": json.loads(eng_manifest.read_text()) if eng_manifest.exists() else None},
        "modified_legacy_files": by_module(compare()),
        "pid": os.getpid(),
    }
    if extra:
        m.update(extra)
    return m


def write_manifest(site, cam_id: str, phase: str, gen, map_id: Optional[str], extra: Optional[dict] = None) -> Path:
    d = site.logs_dir() / cam_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run_{time.strftime('%Y%m%d_%H%M%S')}.manifest.json"
    p.write_text(json.dumps(build_manifest(site, cam_id, phase, gen, map_id, extra), indent=2, default=str))
    return p
