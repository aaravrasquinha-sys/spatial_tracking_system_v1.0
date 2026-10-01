import numpy as np
import pytest

from poi_localization.config import DepthMeasurementConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.depth_measurement import compute_depth_measurement
from poi_perception.contracts import Intrinsics

INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001, baseline=0.05)
CFG = DepthMeasurementConfig()


def _pixel_and_depth_for_floor_point(floor_transform, point_floor):
    R_inv = floor_transform.R.T
    t_inv = -R_inv @ floor_transform.t
    p_cam = R_inv @ np.array(point_floor) + t_inv
    u = INTR.fx * p_cam[0] / p_cam[2] + INTR.cx
    v = INTR.fy * p_cam[1] / p_cam[2] + INTR.cy
    return u, v, p_cam[2]


def _make_depth_image(u, v, depth_m, half_w=15, half_h=30, fill_value_m=None, shape=(480, 640)):
    depth_raw = np.zeros(shape, dtype=np.uint16)
    x0, x1 = int(round(u - half_w)), int(round(u + half_w))
    y0, y1 = int(round(v - half_h)), int(round(v + half_h))
    val = depth_m if fill_value_m is None else fill_value_m
    depth_raw[y0:y1, x0:x1] = int(round(val / INTR.depth_scale))
    polygon = [(u - half_w, v - half_h), (u + half_w, v - half_h), (u + half_w, v + half_h), (u - half_w, v + half_h)]
    return depth_raw, polygon


def test_depth_measurement_recovers_known_position_with_body_thickness_offset():
    floor_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    target_floor_point = (2.0, 0.3, 1.0)  # x, y, height -- torso "front surface"
    u, v, depth_m = _pixel_and_depth_for_floor_point(floor_transform, target_floor_point)
    depth_raw, polygon = _make_depth_image(u, v, depth_m)

    m = compute_depth_measurement(depth_raw, polygon, "full", INTR, floor_transform, CFG)
    assert m is not None
    assert m.src == "depth"

    camera_xy = floor_transform.camera_origin_floor()[:2]
    direction = np.array(target_floor_point[:2]) - camera_xy
    direction = direction / np.linalg.norm(direction)
    expected_xy = np.array(target_floor_point[:2]) + direction * CFG.body_thickness_m

    assert np.allclose(m.position_xy, expected_xy, atol=1e-3)
    # covariance should be a valid (positive semi-definite-ish) 2x2
    sxx, sxy, syy = m.cov_xy
    assert sxx > 0 and syy > 0
    assert sxx * syy - sxy * sxy >= -1e-12


def test_depth_measurement_none_when_torso_quality_fallback():
    floor_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    u, v, depth_m = _pixel_and_depth_for_floor_point(floor_transform, (2.0, 0.0, 1.0))
    depth_raw, polygon = _make_depth_image(u, v, depth_m)
    m = compute_depth_measurement(depth_raw, polygon, "fallback", INTR, floor_transform, CFG)
    assert m is None


def test_depth_measurement_none_when_too_few_valid_pixels():
    floor_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    u, v, depth_m = _pixel_and_depth_for_floor_point(floor_transform, (2.0, 0.0, 1.0))
    depth_raw = np.zeros((480, 640), dtype=np.uint16)  # nothing valid at all
    polygon = [(u - 15, v - 30), (u + 15, v - 30), (u + 15, v + 30), (u - 15, v + 30)]
    m = compute_depth_measurement(depth_raw, polygon, "full", INTR, floor_transform, CFG)
    assert m is None


def test_depth_measurement_none_outside_valid_range():
    floor_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    # far beyond max_range_m
    u, v, depth_m = _pixel_and_depth_for_floor_point(floor_transform, (10.0, 0.0, 1.0))
    depth_raw, polygon = _make_depth_image(u, v, depth_m)
    m = compute_depth_measurement(depth_raw, polygon, "full", INTR, floor_transform, CFG)
    assert m is None


def test_depth_measurement_rejects_intruding_object_outliers():
    """A chair-edge or another person's arm intruding into the polygon
    shows up as a separate depth cluster -- the MAD-based filter should
    reject it and still recover the person's own (majority) depth."""
    floor_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    target_floor_point = (2.0, 0.0, 1.0)
    u, v, depth_m = _pixel_and_depth_for_floor_point(floor_transform, target_floor_point)
    depth_raw, polygon = _make_depth_image(u, v, depth_m)

    # inject an intruding cluster much closer (e.g. 0.5m) in a corner of the polygon
    depth_raw[int(v - 28):int(v - 20), int(u - 14):int(u - 6)] = int(round(0.5 / INTR.depth_scale))

    m = compute_depth_measurement(depth_raw, polygon, "full", INTR, floor_transform, CFG)
    assert m is not None
    camera_xy = floor_transform.camera_origin_floor()[:2]
    direction = np.array(target_floor_point[:2]) - camera_xy
    direction = direction / np.linalg.norm(direction)
    expected_xy = np.array(target_floor_point[:2]) + direction * CFG.body_thickness_m
    assert np.allclose(m.position_xy, expected_xy, atol=0.01)


def test_depth_uncertainty_grows_with_distance():
    floor_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    u1, v1, d1 = _pixel_and_depth_for_floor_point(floor_transform, (1.5, 0.0, 1.0))
    depth1, poly1 = _make_depth_image(u1, v1, d1)
    m1 = compute_depth_measurement(depth1, poly1, "full", INTR, floor_transform, CFG)

    u2, v2, d2 = _pixel_and_depth_for_floor_point(floor_transform, (2.5, 0.0, 1.0))
    depth2, poly2 = _make_depth_image(u2, v2, d2)
    m2 = compute_depth_measurement(depth2, poly2, "full", INTR, floor_transform, CFG)

    assert m1 is not None and m2 is not None
    assert m2.cov_xy[0] > m1.cov_xy[0]  # further away -> larger sxx
