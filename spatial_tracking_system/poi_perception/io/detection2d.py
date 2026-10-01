"""
Section 8: Detection2D is M3's internal record, one per tracked person
per frame -- distinct from and upstream of the system-wide poi.v1 schema
(M4 produces that). track_id_2d is explicitly NOT the final cross-module
person ID (Section 11): keep it visibly namespaced so a future debugging
session never confuses a 2D ID swap with a world-track ID swap.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


@dataclass
class Detection2D:
    t_capture: float  # seconds, same clock domain as Frame.t
    frame_id: int
    cam_id: str
    track_id_2d: int  # ByteTrack ID; NOT the final cross-module person ID
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2, pixel space
    det_conf: float
    keypoints: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # name -> (x, y, conf), pixel space
    footpoint_px: Tuple[float, float] = (0.0, 0.0)
    footpoint_source: str = "bbox_fallback"  # "ankles" | "ankle_single" | "bbox_fallback"
    torso_polygon_px: List[Tuple[float, float]] = field(default_factory=list)
    torso_quality: str = "fallback"  # "full" | "partial" | "fallback"
    low_confidence: bool = False
    track_state: str = "new"  # "new" | "tracked" | "lost_buffer"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Detection2D":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        d["footpoint_px"] = tuple(d["footpoint_px"])
        d["torso_polygon_px"] = [tuple(p) for p in d["torso_polygon_px"]]
        d["keypoints"] = {k: tuple(v) for k, v in d.get("keypoints", {}).items()}
        return cls(**d)


class JsonlDetectionWriter:
    """One JSON line per frame: {"t_capture": ..., "frame_id": ...,
    "detections": [Detection2D, ...]}. Every frame is logged, even an
    empty one (Section 8) -- an absent line and an empty-list line mean
    different things (process not running vs. genuinely nobody there),
    and only the latter is unambiguous on replay."""

    def __init__(self, path: str | Path, rotate_max_bytes: int = 200 * 1024 * 1024):
        self.base_path = Path(path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.rotate_max_bytes = rotate_max_bytes
        self._part = 0
        self._current_path = self._path_for_part(self._part)
        self._f = open(self._current_path, "a")

    def _path_for_part(self, part: int) -> Path:
        if part == 0:
            return self.base_path
        return self.base_path.with_suffix(f".{part}{self.base_path.suffix}")

    def write_frame(self, t_capture: float, frame_id: int, detections: List[Detection2D]) -> None:
        line = json.dumps(
            {
                "t_capture": t_capture,
                "frame_id": frame_id,
                "detections": [d.to_dict() for d in detections],
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


def read_jsonl(path: str | Path) -> Iterator[Tuple[float, int, List[Detection2D]]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dets = [Detection2D.from_dict(d) for d in row["detections"]]
            yield row["t_capture"], row["frame_id"], dets
