"""
Section 8: gates applied before a measurement -- or a track -- is
trusted. Three gates total, grouped here for one obvious place to look:

  - chi_square_gate_pass: the Mahalanobis/chi-square gate used by the
    track manager for every measurement update and every association
    (Section 6, Section 9) -- a thin, named wrapper around
    geometry.mahalanobis_sq so call sites read like the doc's own
    language rather than bare linear algebra.
  - height_range_gate: flags (doesn't reject a position measurement for)
    an implausible height.
  - WalkableGrid / walkable_gate: Phase B only -- explicitly does not
    exist in Phase A (Section 8: "note this explicitly in the code so
    nobody assumes it's protecting against reflections before the map
    exists").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from poi_localization.config import GateConfig, HeightConfig


def chi_square_gate_pass(distance_sq: float, cfg: GateConfig) -> bool:
    return distance_sq <= cfg.chi2_gate_2dof


def height_range_gate(height_m: Optional[float], cfg: HeightConfig) -> bool:
    """True = plausible (or unknown, which isn't itself implausible).
    Section 8: "don't let the height gate reject the position
    measurement, only flag it for the height field itself" -- so this
    function's result is informational, and callers must not use it to
    drop a position measurement."""
    if height_m is None:
        return True
    return cfg.min_height_m <= height_m <= cfg.max_height_m


@dataclass
class WalkableGrid:
    """A simple occupancy grid: origin (floor-frame x, y of cell [0,0]'s
    corner), resolution (meters/cell), and a 2D boolean array, True =
    walkable. This is our own clean schema honoring the full system
    plan's description ("5cm floor-occupancy grid, origin, resolution")
    -- Module 1 doesn't exist yet in this build, so the exact on-disk
    format it will eventually produce isn't fixed; swap load_walkable_grid
    for whatever M1 actually emits once it exists, without touching
    is_walkable's signature."""

    origin_xy: Tuple[float, float]
    resolution_m: float
    grid: np.ndarray  # (rows, cols) bool, True = walkable

    def is_walkable(self, x: float, y: float) -> bool:
        col = int((x - self.origin_xy[0]) / self.resolution_m)
        row = int((y - self.origin_xy[1]) / self.resolution_m)
        if row < 0 or col < 0 or row >= self.grid.shape[0] or col >= self.grid.shape[1]:
            return False  # outside the mapped floor at all -- can't be walkable
        return bool(self.grid[row, col])


def load_walkable_grid(path: str | Path) -> WalkableGrid:
    raw = json.loads(Path(path).read_text())
    grid = np.array(raw["grid"], dtype=bool)
    return WalkableGrid(origin_xy=tuple(raw["origin"]), resolution_m=raw["resolution"], grid=grid)


def walkable_gate(x: float, y: float, grid: Optional[WalkableGrid]) -> bool:
    """Section 8: "This gate doesn't exist in Phase A" -- grid is None
    whenever there's no walkable map loaded (always true in Phase A;
    optionally true in Phase B before Module 1 exists), and this always
    passes in that case. Callers must not treat a pass here as "verified
    against the map" unless they've confirmed grid is not None."""
    if grid is None:
        return True
    return grid.is_walkable(x, y)
