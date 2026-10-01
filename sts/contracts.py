"""Validate files/messages against contracts/schemas/*.schema.json.

`jsonschema` is a DEV dependency (requirements/dev.txt), deliberately not in the runtime
venv: the runtime consistency gate (sts.consistency) does its own hand checks. Tests and
`sts contracts check` use the schemas; producers and consumers are each validated against
the SAME schema, so a change on one side fails CI instead of the Orin.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from sts.paths import REPO_ROOT

SCHEMA_DIR = REPO_ROOT / "contracts" / "schemas"
ALL = ["calibration", "walkable", "room_frame", "map_manifest", "camera_model",
       "poi_tracks", "poi_event", "poi_health", "poi_scene"]


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())


def validate(instance: Any, name: str) -> List[str]:
    """-> list of error strings ([] = valid). Needs `jsonschema`."""
    import jsonschema
    v = jsonschema.Draft202012Validator(load_schema(name))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(instance)]


def validate_file(path: str | Path, name: str) -> List[str]:
    return validate(json.loads(Path(path).read_text()), name)
