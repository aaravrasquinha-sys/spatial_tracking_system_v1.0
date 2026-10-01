"""
Section 10: two output forms, deliberately different, because Phase A
has no room/map.

Phase A uses p_local/v_local (not p/v) specifically "so nobody
downstream mistakes a Phase-A log for a room-frame position." Phase B's
WorldTrackPhaseB is exactly the poi.v1 schema's tracks[] entry shape from
the full system plan (id, state, p, v, cov_xy, height, conf, src, age_s,
bbox) -- the field-name rename plus the frame swap (Section 2) is the
entire Phase A -> Phase B transition.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union


@dataclass
class WorldTrackPhaseA:
    t_capture: float
    world_track_id: int
    state: str  # "tentative" | "confirmed" | "coasting" | "lost"
    p_local: Tuple[float, float, float]
    v_local: Tuple[float, float, float]
    cov_xy: Tuple[float, float, float]
    height_m: Optional[float]
    src: str  # "depth" | "raycast" | "fused" | "predicted"
    age_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorldTrackPhaseB:
    """The poi.v1 tracks[] entry shape (full system plan Section 4.3)."""

    id: int
    state: str
    p: Tuple[float, float, float]
    v: Tuple[float, float, float]
    cov_xy: Tuple[float, float, float]
    height: Optional[float]
    conf: float
    src: str
    age_s: float
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


WorldTrack = Union[WorldTrackPhaseA, WorldTrackPhaseB]


class JsonlWorldTrackWriter:
    """One JSON line per frame: {"t_capture", "frame_id", "phase",
    "map_id", "tracks": [...]}. Every frame is logged, even empty
    (Section 10, same discipline as Module 3's Detection2D log)."""

    def __init__(self, path: str | Path, phase: str, map_id: Optional[str] = None, rotate_max_bytes: int = 200 * 1024 * 1024):
        self.base_path = Path(path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.map_id = map_id
        self.rotate_max_bytes = rotate_max_bytes
        self._part = 0
        self._current_path = self._path_for_part(0)
        self._f = open(self._current_path, "a")

    def _path_for_part(self, part: int) -> Path:
        if part == 0:
            return self.base_path
        return self.base_path.with_suffix(f".{part}{self.base_path.suffix}")

    def write_frame(self, t_capture: float, frame_id: int, tracks: List[WorldTrack]) -> None:
        line = json.dumps(
            {
                "t_capture": t_capture,
                "frame_id": frame_id,
                "phase": self.phase,
                "map_id": self.map_id,
                "tracks": [t.to_dict() for t in tracks],
            }
        )
        self._f.write(line + "\n")
        self._f.flush()
        if self._f.tell() >= self.rotate_max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        self._f.close()
        self._part += 1
        self._current_path = self._path_for_part(self._part)
        self._f = open(self._current_path, "a")

    def close(self) -> None:
        self._f.close()


def _track_from_dict(d: dict, phase: str) -> WorldTrack:
    d = dict(d)
    d["p" if phase == "B" else "p_local"] = tuple(d["p" if phase == "B" else "p_local"])
    d["v" if phase == "B" else "v_local"] = tuple(d["v" if phase == "B" else "v_local"])
    d["cov_xy"] = tuple(d["cov_xy"])
    if phase == "B":
        d["bbox"] = tuple(d["bbox"])
        return WorldTrackPhaseB(**d)
    return WorldTrackPhaseA(**d)


def read_jsonl(path: str | Path) -> Iterator[Tuple[float, int, str, Optional[str], List[WorldTrack]]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            phase = row.get("phase", "A")
            tracks = [_track_from_dict(t, phase) for t in row["tracks"]]
            yield row["t_capture"], row["frame_id"], phase, row.get("map_id"), tracks
