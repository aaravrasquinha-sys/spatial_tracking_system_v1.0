"""
Section 7: static ROI masks for anything in the room that reliably
produces a false person (mirrors, TVs/monitors, posters with people on
them, windows facing a street). Drawn once, loaded at startup, in the
same coordinate space as detection (raw pixel space of the color
stream). A detection whose box centroid falls inside a masked region is
dropped before it ever reaches the tracker.

Section 11: masks live in a config file keyed by cam_id, matching the
calibration file's cam_id in the full plan -- see configs/masks.cam0.example.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

Point = Tuple[float, float]


@dataclass
class MaskRegion:
    name: str
    polygon: List[Point]


def _point_in_polygon(x: float, y: float, polygon: List[Point]) -> bool:
    """Standard ray-casting point-in-polygon test. Dependency-free (no
    shapely) since this is the only geometry op masks.py needs."""
    n = len(polygon)
    inside = False
    x1, y1 = polygon[-1]
    for i in range(n):
        x2, y2 = polygon[i]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


class MaskSet:
    """Masks for a single cam_id."""

    def __init__(self, regions: List[MaskRegion]):
        self.regions = regions

    def is_masked(self, bbox: Tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return any(_point_in_polygon(cx, cy, r.polygon) for r in self.regions)

    def masked_region_name(self, bbox: Tuple[float, float, float, float]) -> str | None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for r in self.regions:
            if _point_in_polygon(cx, cy, r.polygon):
                return r.name
        return None


def load_masks(path: str | Path) -> Dict[str, MaskSet]:
    """path: JSON of the form {"cam0": [{"name": ..., "polygon": [[x,y],...]}, ...], ...}"""
    raw = json.loads(Path(path).read_text())
    out: Dict[str, MaskSet] = {}
    for cam_id, regions_raw in raw.items():
        regions = [
            MaskRegion(name=r["name"], polygon=[tuple(p) for p in r["polygon"]]) for r in regions_raw
        ]
        out[cam_id] = MaskSet(regions)
    return out


def empty_mask_set() -> MaskSet:
    return MaskSet([])
