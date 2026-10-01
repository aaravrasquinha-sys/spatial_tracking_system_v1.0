"""Import-boundary checker (rule 1 of docs/ARCHITECTURE.md).

Parses every .py file of each package with `ast` and fails on any forbidden edge. This one
check is what lets you edit a module later without accidentally coupling it to another.
Only TOP-LEVEL package names are compared (`import pyslam.core.lie` -> `pyslam`).
Imports inside `if TYPE_CHECKING:` and inside string/docstrings are not imports and are ignored.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

from sts.paths import REPO_ROOT

# package -> packages it MAY import (besides itself, the stdlib and third-party libs)
ALLOWED: Dict[str, Set[str]] = {
    "pyslam":           set(),
    "poi_perception":   set(),
    "poi_localization": {"poi_perception"},
    "poi_present":      {"poi_localization", "poi_perception"},   # M5 reads M4 record types (+ Frame via M4)
    "sts":              {"pyslam", "poi_perception", "poi_localization", "poi_present"},
}
FIRST_PARTY = set(ALLOWED)


def imports_of(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def check(root: Path = REPO_ROOT) -> List[Tuple[str, str, str]]:
    """-> list of (file, package, forbidden_import)."""
    violations = []
    for pkg, allowed in ALLOWED.items():
        base = root / pkg
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            for imp in imports_of(py):
                if imp in FIRST_PARTY and imp != pkg and imp not in allowed:
                    violations.append((py.relative_to(root).as_posix(), pkg, imp))
    return sorted(violations)


def nothing_imports_sts(root: Path = REPO_ROOT) -> List[str]:
    return sorted({v[0] for v in check(root) if v[2] == "sts"})
