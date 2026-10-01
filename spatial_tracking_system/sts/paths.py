"""Repo-root discovery and the data-directory layout.

Every module in the legacy repos assumed "CWD == repo root" and used paths
like `configs/...` or `logs/...`. Here the root is discovered from this file's
location, so `sts` works from any CWD, and every generated config carries
absolute paths.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | os.PathLike | None, base: Path | None = None) -> Path | None:
    """Expand ~ and make relative paths relative to `base` (default: repo root)."""
    if path is None:
        return None
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute():
        return p
    return ((base or REPO_ROOT) / p).resolve()


def default_capture_profile() -> Path:
    return REPO_ROOT / "configs" / "capture_profile.mapping.json"
