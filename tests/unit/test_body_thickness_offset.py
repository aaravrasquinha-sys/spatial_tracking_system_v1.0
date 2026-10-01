"""
Tests for estimate_body_thickness_offset_m: the previous fixed
body_thickness_m offset was correct only for a person facing (or
back-to) the camera, and systematically wrong by several centimeters --
in a walking-direction-correlated way the Kalman filter cannot average
out -- for a person seen from the side.
"""
import numpy as np
import pytest

from poi_localization.config import DepthMeasurementConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.depth_measurement import estimate_body_thickness_offset_m
from poi_perception.contracts import Intrinsics

INTR = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480, depth_scale=0.001, baseline=0.05)
CFG = DepthMeasurementConfig()


def test_missing_shoulders_falls_back_to_facing_constant():
    ft = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0)
    offset = estimate_body_thickness_offset_m(None, None, 3.0, INTR, ft, np.array([1.0, 0.0]), CFG)
    assert offset == CFG.body_thickness_m


def test_facing_camera_shoulder_line_perpendicular_to_view_uses_thin_offset():
    # Camera level, looking along +Z; person directly in front, shoulder
    # line horizontal in the image -> shoulder line in floor space is
    # perpendicular to the (straight-ahead) viewing direction: a facing
    # (or back-to-camera) pose.
    ft = FloorFrameTransform.from_height_and_yaw(2.3, 0.0, 0.0)
    depth_m = 3.0
    # shoulders symmetric about the principal point at the same depth
    shoulder_l_px = (INTR.cx - 30, INTR.cy)
    shoulder_r_px = (INTR.cx + 30, INTR.cy)
    viewing_dir_floor_xy = np.array([1.0, 0.0])  # straight ahead, matches yaw=0
    offset = estimate_body_thickness_offset_m(shoulder_l_px, shoulder_r_px, depth_m, INTR, ft, viewing_dir_floor_xy, CFG)
    assert offset == pytest.approx(CFG.body_thickness_m, abs=1e-3)


def test_side_profile_shoulder_line_parallel_to_view_uses_wide_offset():
    # Construct viewing_dir_floor_xy to be EXACTLY the shoulder line's own
    # floor-space direction (rather than guessing a camera pose that
    # produces it) -- this is the direct, unambiguous "side profile" case
    # the function's docstring describes: shoulder line parallel to the
    # view ray.
    from poi_localization import geometry

    ft = FloorFrameTransform.from_height_and_yaw(2.3, 0.0, 0.0)
    depth_m = 3.0
    shoulder_l_px = (INTR.cx - 30, INTR.cy)
    shoulder_r_px = (INTR.cx + 30, INTR.cy)
    p_l = geometry.deproject_pixel(shoulder_l_px[0], shoulder_l_px[1], depth_m, INTR)
    p_r = geometry.deproject_pixel(shoulder_r_px[0], shoulder_r_px[1], depth_m, INTR)
    shoulder_dir_floor = ft.apply_direction(p_r - p_l)[:2]
    viewing_dir_floor_xy = shoulder_dir_floor / np.linalg.norm(shoulder_dir_floor)

    offset = estimate_body_thickness_offset_m(shoulder_l_px, shoulder_r_px, depth_m, INTR, ft, viewing_dir_floor_xy, CFG)
    assert offset == pytest.approx(CFG.body_thickness_side_m, abs=1e-6)


def test_offset_is_between_facing_and_side_constants():
    ft = FloorFrameTransform.from_height_and_yaw(2.3, 0.0, 0.0)
    depth_m = 3.0
    shoulder_l_px = (INTR.cx - 30, INTR.cy)
    shoulder_r_px = (INTR.cx + 30, INTR.cy)
    for angle_deg in (0, 30, 60, 90):
        rad = np.radians(angle_deg)
        viewing_dir = np.array([np.cos(rad), np.sin(rad)])
        offset = estimate_body_thickness_offset_m(shoulder_l_px, shoulder_r_px, depth_m, INTR, ft, viewing_dir, CFG)
        lo, hi = sorted((CFG.body_thickness_m, CFG.body_thickness_side_m))
        assert lo - 1e-6 <= offset <= hi + 1e-6
