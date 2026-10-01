"""Retention: track logs are a record of people's movements in your home.
`sts retention prune` deletes logs older than site.retention.*; `--dry-run` lists first."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

from sts.site import SiteConfig


def _old(paths, days: int, now: float) -> List[Path]:
    cutoff = now - days * 86400
    return [p for p in paths if p.is_file() and p.stat().st_mtime < cutoff]


def plan_prune(site: SiteConfig, now: float | None = None) -> List[Tuple[str, Path]]:
    now = now or time.time()
    out: List[Tuple[str, Path]] = []
    logs = site.logs_dir()
    for cam_dir in (logs.iterdir() if logs.exists() else []):
        if not cam_dir.is_dir():
            continue
        for sub, days in (("world_tracks", site.retention.track_log_days), ("detections", site.retention.track_log_days),
                          ("clips", site.retention.clip_days)):
            for p in _old((cam_dir / sub).rglob("*") if (cam_dir / sub).exists() else [], days, now):
                out.append((f"{sub} > {days}d", p))
        for p in _old(cam_dir.glob("run_*.manifest.json"), site.retention.track_log_days, now):
            out.append(("run manifest", p))
    rec = site.recordings_dir()
    for p in _old(rec.glob("*.bag") if rec.exists() else [], site.retention.bag_days, now):
        out.append((f"bag > {site.retention.bag_days}d", p))
    return out


def prune(site: SiteConfig, dry_run: bool = True, now: float | None = None) -> List[Tuple[str, Path]]:
    items = plan_prune(site, now)
    if not dry_run:
        for _, p in items:
            try:
                p.unlink()
            except OSError:
                pass
    return items
