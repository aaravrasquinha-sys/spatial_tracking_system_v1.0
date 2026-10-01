"""
Shared output type for Section 4 (depth) and Section 5 (raycast)
measurements -- the Kalman filter (tracking/kalman_track.py) and the
track manager's chi-square gate consume this uniformly, regardless of
which measurement model produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Measurement:
    position_xy: Tuple[float, float]  # floor-frame meters
    cov_xy: Tuple[float, float, float]  # upper triangle (sxx, sxy, syy), m^2
    src: str  # "depth" | "raycast"
