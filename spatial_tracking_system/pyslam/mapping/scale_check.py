"""
WP-LIVE: the single tape-distance scale check -- approved as the
ONLY scale-verification mechanism (no AprilTags, no fiducials).
Nothing else in this system can detect a uniform depth-scale bias
(the map is internally self-consistent either way; only an external
ground-truth length exposes it) -- see planning notes section 6's
acceptance-metric table and section 1's "no offline compute" framing:
this runs live, in seconds, on two points the operator picks in the
running viewer.

Usage: the operator identifies two pixels (in two, possibly different,
keyframes) that correspond to a known real-world distance (a doorway
width, a wall length, a tape stretched between two marks -- anything),
enters that measured distance, and this module reports the map's own
distance between those two points plus the ratio and a pass/fail
against a tolerance. This is NOT a calibration step (it doesn't feed
back into the map) -- it's an acceptance check, exactly like the tape-
measure pairs in the original room plan, just singular and manual by
design (approved scope).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ScaleCheckResult:
    map_distance_m: float
    tape_distance_m: float
    ratio: float          # map / tape; 1.0 = perfect
    error_pct: float
    passed: bool


def point_from_pixel(depth_m: float, px: float, py: float, K: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (px - cx) / fx * depth_m
    y = (py - cy) / fy * depth_m
    return np.array([x, y, depth_m])


def check_scale(p_a_world: np.ndarray, p_b_world: np.ndarray, tape_distance_m: float,
                 tol_pct: float = 3.0) -> ScaleCheckResult:
    """p_a_world/p_b_world: the two picked points, already transformed
    into a common (e.g. site) frame by the caller -- this function is
    pure geometry, agnostic to which frame that is, so it's usable both
    within one keyframe (frame transform is identity) and across two."""
    map_d = float(np.linalg.norm(np.asarray(p_a_world) - np.asarray(p_b_world)))
    if tape_distance_m <= 1e-6:
        raise ValueError("tape_distance_m must be a positive, real measurement")
    ratio = map_d / tape_distance_m
    error_pct = abs(ratio - 1.0) * 100.0
    return ScaleCheckResult(
        map_distance_m=map_d, tape_distance_m=float(tape_distance_m),
        ratio=ratio, error_pct=error_pct, passed=error_pct <= tol_pct,
    )
