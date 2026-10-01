"""
Frozen contracts for M3.

These shapes are deliberately a small, self-contained subset of the
`pyslam.core.types.Frame` / `Intrinsics` contracts used by M1/M2 -- same
field names and semantics (device timestamp in seconds, HxWx3 uint8 RGB,
HxW uint16 raw depth, metres = depth * depth_scale) so a frame handed to
M3 by M6's future runtime daemon needs no translation layer.

M3 does NOT import pyslam (see poi_perception/README section "Module
boundary"): the full system plan is explicit that M3 never needs SLAM,
the map, or calibration, and that the boundary should stay hard now so
this package is swappable later (a different model, a second camera on
different hardware) without dragging pyslam along. Duplicating these two
small dataclasses is the price of that boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float  # metres per raw depth unit
    baseline: float = 0.05  # metres; unused by M3 directly, carried for downstream (M4)

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )


@dataclass
class Frame:
    """One color+depth bundle, depth aligned to color. M3 only reads
    frame.rgb; frame.depth passes through untouched so a future M4/M6
    integration gets a depth frame guaranteed synced to the detection
    (see the full plan's Module 3 "Capture" note)."""

    t: float  # device time, seconds, monotonic
    rgb: np.ndarray  # HxWx3 uint8
    depth: np.ndarray  # HxW uint16, raw units
    intr: Intrinsics
    frame_id: int = -1
    cam_id: str = "cam0"


# COCO-17 keypoint order, as produced by yolo11n-pose / yolov8n-pose.
COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
