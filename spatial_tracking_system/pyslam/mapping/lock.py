"""
WP-LIVE: "lock" a live map bundle. Since mapping is live-only (no
offline compile stage), locking is a SNAPSHOT + hash + chmod -- seconds
of work, no recomputation. This mirrors the original plan's Module 1
bundle contract (map_id, manifest, read-only directory) but drops every
single-room-specific field (T_room_grav, one floor/gravity angle,
"acceptance summary" keyed to a rectangular room) in favour of the
generalised site frame + layered products this rearchitecture produces.

Directory shape (a subset already exists per-file from other WP-LIVE
modules; this module only computes map_id and writes manifest.json +
applies the read-only lock):

  maps/site_<map_id>/
    manifest.json       map_id, builder git commit (if available), site
                         frame definition, capture_profile_sha, per-
                         plane-landmark summary, live QA snapshot at
                         lock time, scale-check result if one was run
    trajectory.json      (existing WP-T2 keyframes export, site frame added)
    dense/points.ply      Level-2 dense export at lock time
    layers/walkable.json  STS schema, read directly by M4
    layers/layers.json    fuller per-cell class/height export
    reloc_map/             (existing reloc pack, unchanged contract)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import json
import os
import stat
import subprocess
import time


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_map_id(bundle_dir: str, content_files: list[str]) -> str:
    """First 12 hex chars of a SHA-256 over the sorted content hashes of
    every file in content_files (paths relative to bundle_dir) -- same
    construction the original plan specified for map_id, kept unchanged
    since it has no single-room dependency."""
    hashes = []
    for rel in sorted(content_files):
        p = os.path.join(bundle_dir, rel)
        if os.path.exists(p):
            hashes.append(f"{rel}:{_sha256_file(p)}")
        else:
            hashes.append(f"{rel}:MISSING")
    combined = hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()
    return combined[:12]


def _git_commit(cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def write_manifest(bundle_dir: str, map_id: str, capture_profile_sha: str,
                    site_frame_summary: dict, plane_summary: list[dict],
                    qa_snapshot: dict, scale_check: Optional[dict] = None,
                    builder_repo_root: Optional[str] = None) -> str:
    manifest = {
        "map_id": map_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder_git_commit": _git_commit(builder_repo_root) if builder_repo_root else None,
        "capture_profile_sha": capture_profile_sha,
        "site_frame": site_frame_summary,
        "plane_landmarks": plane_summary,
        "qa_snapshot_at_lock": qa_snapshot,
        "scale_check": scale_check,
        "generalised": True,   # marks this as a WP-LIVE bundle, not the
                                # original single-room-plan bundle shape
    }
    path = os.path.join(bundle_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    return path


def lock_bundle(bundle_dir: str, content_files: list[str], capture_profile_sha: str,
                 site_frame_summary: dict, plane_summary: list[dict], qa_snapshot: dict,
                 scale_check: Optional[dict] = None, builder_repo_root: Optional[str] = None,
                 make_read_only: bool = True) -> dict:
    """The single call site.lock() should go through. Returns
    {"map_id":..., "manifest_path":...}. Idempotent on map_id (content-
    hashed) but NOT idempotent on disk state -- calling this twice on a
    changed bundle produces a DIFFERENT map_id and a fresh manifest
    (the old one is left as-is; this function never edits an existing
    locked bundle)."""
    map_id = compute_map_id(bundle_dir, content_files)
    manifest_path = write_manifest(bundle_dir, map_id, capture_profile_sha, site_frame_summary,
                                    plane_summary, qa_snapshot, scale_check, builder_repo_root)
    if make_read_only:
        for root, dirs, files in os.walk(bundle_dir):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                except OSError:
                    pass
    return {"map_id": map_id, "manifest_path": manifest_path}
