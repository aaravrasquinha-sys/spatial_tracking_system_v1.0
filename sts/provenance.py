"""Which legacy module files differ from the versions that were merged?

Informational, NOT a test: you are meant to edit modules. This tells you exactly
which ones you have touched, so a bug report or a regression diff can name them.
Baseline: contracts/legacy_hashes.json (sha256 at merge time).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from sts.paths import REPO_ROOT


def _h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_baseline() -> Dict[str, str]:
    return json.loads((REPO_ROOT / "contracts" / "legacy_hashes.json").read_text())["files"]


def compare() -> Dict[str, List[str]]:
    base = load_baseline()
    modified, missing = [], []
    for rel, sha in base.items():
        p = REPO_ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif _h(p) != sha:
            modified.append(rel)
    known = set(base)
    added = []
    for pkg in ("pyslam", "poi_perception", "poi_localization", "poi_present"):
        for p in (REPO_ROOT / pkg).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                rel = p.relative_to(REPO_ROOT).as_posix()
                if rel not in known:
                    added.append(rel)
    return {"modified": sorted(modified), "missing": sorted(missing), "added": sorted(added)}


def by_module(diff: Dict[str, List[str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for kind in ("modified", "missing", "added"):
        for rel in diff[kind]:
            mod = rel.split("/")[0]
            out[mod] = out.get(mod, 0) + 1
    return out
