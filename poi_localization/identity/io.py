"""Identity JSONL log (explainability record, R1 spec sections 12.2 and 14)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class IdentityJsonlWriter:
    """One JSON object per line. Rotates like the other M4 JSONL logs."""

    def __init__(self, path, rotate_max_bytes: int = 200 * 1024 * 1024):
        self.base_path = Path(path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.rotate_max_bytes = rotate_max_bytes
        self._part = 0
        self._f = open(self._path_for_part(0), "a")

    def _path_for_part(self, part: int) -> Path:
        if part == 0:
            return self.base_path
        return self.base_path.with_suffix(f".{part}{self.base_path.suffix}")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, sort_keys=True) + "\n")
        self._f.flush()
        if self._f.tell() >= self.rotate_max_bytes:
            self._f.close()
            self._part += 1
            self._f = open(self._path_for_part(self._part), "a")

    def close(self) -> None:
        self._f.close()


def read_identity_jsonl(path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
