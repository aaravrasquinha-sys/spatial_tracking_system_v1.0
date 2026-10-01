"""
A pose backend that produces synthetic RawDetections directly from a
scripted scenario function, bypassing pixel content entirely. This is
what makes tracking, footpoint/torso, masking, and the daemon's threading
testable deterministically without a model, an engine, or a camera.

The scenario generators below are direct implementations of Section 9's
scripted scenario set, so the same scenarios used for hand-grading on
real hardware can also run as automated regression tests here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from poi_perception.contracts import COCO_KEYPOINT_NAMES, Frame
from poi_perception.inference.pose_infer import RawDetection

ScenarioFn = Callable[[int, float], List[RawDetection]]


class MockPoseBackend:
    def __init__(self, scenario_fn: ScenarioFn):
        self.scenario_fn = scenario_fn

    def warm_up(self, n: int = 5) -> None:
        pass

    def infer(self, frame: Frame) -> List[RawDetection]:
        return self.scenario_fn(frame.frame_id, frame.t)


def _standing_person(
    cx: float,
    cy_hip: float,
    box_w: float = 60.0,
    box_h: float = 170.0,
    conf: float = 0.9,
    ankle_conf: float = 0.9,
    visible_kpts: Optional[Tuple[str, ...]] = None,
) -> RawDetection:
    """Build a plausible standing-person detection centered at (cx, cy_hip
    is the vertical position of the hip line). All keypoints are placed
    at anatomically sane offsets so footpoint/torso logic gets realistic
    geometry to work with."""
    x1, y1 = cx - box_w / 2, cy_hip - box_h * 0.55
    x2, y2 = cx + box_w / 2, cy_hip + box_h * 0.45
    shoulder_y = y1 + box_h * 0.20
    hip_y = cy_hip
    knee_y = y1 + box_h * 0.75
    ankle_y = y2 - 2.0

    all_kpts: Dict[str, Tuple[float, float, float]] = {
        "nose": (cx, y1 + box_h * 0.05, conf),
        "left_eye": (cx - 3, y1 + box_h * 0.04, conf),
        "right_eye": (cx + 3, y1 + box_h * 0.04, conf),
        "left_ear": (cx - 6, y1 + box_h * 0.05, conf),
        "right_ear": (cx + 6, y1 + box_h * 0.05, conf),
        "left_shoulder": (cx - box_w * 0.35, shoulder_y, conf),
        "right_shoulder": (cx + box_w * 0.35, shoulder_y, conf),
        "left_elbow": (cx - box_w * 0.42, shoulder_y + box_h * 0.15, conf),
        "right_elbow": (cx + box_w * 0.42, shoulder_y + box_h * 0.15, conf),
        "left_wrist": (cx - box_w * 0.45, shoulder_y + box_h * 0.28, conf),
        "right_wrist": (cx + box_w * 0.45, shoulder_y + box_h * 0.28, conf),
        "left_hip": (cx - box_w * 0.25, hip_y, conf),
        "right_hip": (cx + box_w * 0.25, hip_y, conf),
        "left_knee": (cx - box_w * 0.22, knee_y, conf),
        "right_knee": (cx + box_w * 0.22, knee_y, conf),
        "left_ankle": (cx - box_w * 0.20, ankle_y, ankle_conf),
        "right_ankle": (cx + box_w * 0.20, ankle_y, ankle_conf),
    }
    if visible_kpts is not None:
        kpts = {
            name: (val if name in visible_kpts else (0.0, 0.0, 0.0))
            for name, val in all_kpts.items()
        }
    else:
        kpts = all_kpts
    return RawDetection(bbox=(x1, y1, x2, y2), det_conf=conf, keypoints=kpts)


def scenario_single_loop(frame_w: int = 640, frame_h: int = 480, n_frames: int = 90) -> ScenarioFn:
    """One person walking a simple loop across the frame (Section 9, #1)."""

    def fn(frame_id: int, t: float) -> List[RawDetection]:
        if frame_id >= n_frames:
            return []
        phase = frame_id / max(n_frames - 1, 1)
        cx = 60 + phase * (frame_w - 120)
        cy_hip = frame_h * 0.55 + 20 * np.sin(phase * 2 * np.pi)
        return [_standing_person(cx, cy_hip)]

    return fn


def scenario_two_crossing(frame_w: int = 640, frame_h: int = 480, n_frames: int = 90) -> ScenarioFn:
    """Two people crossing paths (Section 9, #2) -- exercises ByteTrack's
    two-tier matching and Section 6's "re-check in the world frame" note
    (2D swaps at the crossing point are expected and are M4's problem)."""

    def fn(frame_id: int, t: float) -> List[RawDetection]:
        if frame_id >= n_frames:
            return []
        phase = frame_id / max(n_frames - 1, 1)
        cx_a = 60 + phase * (frame_w - 120)
        cx_b = (frame_w - 60) - phase * (frame_w - 120)
        cy = frame_h * 0.55
        return [
            _standing_person(cx_a, cy, conf=0.9),
            _standing_person(cx_b, cy, conf=0.85),
        ]

    return fn


def scenario_exit_reenter(frame_w: int = 640, frame_h: int = 480, n_frames: int = 90) -> ScenarioFn:
    """Person exits frame and re-enters (Section 9, #3). Expected: track
    ends, then a *new* 2D track ID on re-entry (Section 3 scope note --
    re-identification is out of v1 scope)."""

    def fn(frame_id: int, t: float) -> List[RawDetection]:
        cycle = n_frames // 3
        if frame_id < cycle:
            cx = 60 + (frame_id / max(cycle - 1, 1)) * (frame_w * 0.5)
            return [_standing_person(cx, frame_h * 0.55)]
        if frame_id < 2 * cycle:
            return []  # off-screen
        if frame_id < 3 * cycle:
            local = frame_id - 2 * cycle
            cx = frame_w * 0.5 + (local / max(cycle - 1, 1)) * (frame_w * 0.4)
            return [_standing_person(cx, frame_h * 0.55)]
        return []

    return fn


def scenario_sitting(frame_w: int = 640, frame_h: int = 480, n_frames: int = 90) -> ScenarioFn:
    """Person sits down (Section 9, #4): tests keypoint degradation and
    the sitting-height case. Ankles become unreliable (occluded by a
    desk/chair in a real capture) -- simulate that by dropping ankle
    confidence for the "sitting" half."""

    def fn(frame_id: int, t: float) -> List[RawDetection]:
        cx = frame_w * 0.5
        standing_half = n_frames // 2
        if frame_id < standing_half:
            return [_standing_person(cx, frame_h * 0.55)]
        # sitting: shorter box, hip lower relative to box, ankles unreliable
        return [
            _standing_person(
                cx,
                frame_h * 0.68,
                box_h=110.0,
                ankle_conf=0.2,
            )
        ]

    return fn


def scenario_partial_occlusion(
    frame_w: int = 640, frame_h: int = 480, n_frames: int = 90, occlude_frac: float = 0.3
) -> ScenarioFn:
    """Partial occlusion behind furniture for ~1-2s (Section 9, #5):
    detection confidence drops (low-confidence second pass) then
    disappears entirely for a run of frames, then resumes -- exercises
    ByteTrack's lost-buffer survival."""
    occlude_start = int(n_frames * 0.4)
    occlude_len = max(1, int(n_frames * occlude_frac))

    def fn(frame_id: int, t: float) -> List[RawDetection]:
        if frame_id >= n_frames:
            return []
        phase = frame_id / max(n_frames - 1, 1)
        cx = 60 + phase * (frame_w - 120)
        cy = frame_h * 0.55
        if occlude_start <= frame_id < occlude_start + occlude_len:
            return []  # fully occluded -- track must survive on coasting/lost_buffer
        # a short low-confidence taper right at the edges of occlusion
        edge = 6
        if occlude_start - edge <= frame_id < occlude_start or (
            occlude_start + occlude_len <= frame_id < occlude_start + occlude_len + edge
        ):
            return [_standing_person(cx, cy, conf=0.25)]
        return [_standing_person(cx, cy)]

    return fn


def scenario_near_mirror(
    frame_w: int = 640,
    frame_h: int = 480,
    n_frames: int = 60,
    mirror_bbox: Tuple[float, float, float, float] = (400, 50, 620, 300),
) -> ScenarioFn:
    """A real person standing near a mirror/TV, plus a "reflection" false
    detection whose box centroid falls inside the masked region (Section
    9, #6 / Section 7). With the ROI mask configured over mirror_bbox,
    the reflection should be dropped before it reaches the tracker."""
    mx1, my1, mx2, my2 = mirror_bbox
    mirror_cx, mirror_cy = (mx1 + mx2) / 2, (my1 + my2) / 2

    def fn(frame_id: int, t: float) -> List[RawDetection]:
        if frame_id >= n_frames:
            return []
        real = _standing_person(150, frame_h * 0.55, conf=0.9)
        # High enough confidence to clear new_track_thresh on its own --
        # the point of this scenario is to test the ROI mask (Section 7),
        # not the tracker's own "don't spawn from medium-confidence
        # detections" gate (Section 6), so the reflection must look like
        # a plausible, confident false detection.
        reflection = _standing_person(mirror_cx, mirror_cy, box_h=140.0, conf=0.85)
        return [real, reflection]

    return fn
