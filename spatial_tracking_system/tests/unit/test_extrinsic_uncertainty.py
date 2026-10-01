"""
Tests for geometry.extrinsic_position_variance and its wiring into both
measurement models. This is the fix for the system's dominant real-world
error source: previously, cov_xy modeled sensor noise only and treated
the floor-frame transform (Phase A's IMU+depth fit, or Phase B's M2
calibration) as exact. A 1deg extrinsic pitch error alone was shown (by
simulation against this codebase) to produce ~14cm median position error
while the old covariance reported ~1cm -- an overconfident-by-~10x
tracker that both fails the accuracy acceptance test AND, less
obviously, causes the chi-square gate to reject perfectly good
measurements because their real error no longer fits inside the
(wrongly tight) covariance the gate checks against.
"""
import numpy as np
import pytest

from poi_localization import geometry
from poi_localization.config import DepthMeasurementConfig, RaycastMeasurementConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.depth_measurement import compute_depth_measurement
from poi_localization.measurement.raycast_measurement import compute_raycast_measurement
from poi_perception.contracts import Intrinsics

INTR = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480, depth_scale=0.001, baseline=0.05)


def test_extrinsic_variance_zero_when_transform_exact():
    assert geometry.extrinsic_position_variance(5.0, 0.0, 0.0) == 0.0


def test_extrinsic_variance_grows_with_range():
    near = geometry.extrinsic_position_variance(1.0, np.radians(1.0), 0.01)
    far = geometry.extrinsic_position_variance(5.0, np.radians(1.0), 0.01)
    assert far > near


def test_extrinsic_variance_matches_arclength_approximation():
    # At zero translation uncertainty, sigma should be exactly range * angle (radians).
    rot = np.radians(2.0)
    var = geometry.extrinsic_position_variance(3.0, rot, 0.0)
    assert var == pytest.approx((3.0 * rot) ** 2)


def test_add_isotropic_variance_only_touches_diagonal():
    sxx, sxy, syy = geometry.add_isotropic_variance((0.01, 0.002, 0.02), 0.05)
    assert sxx == pytest.approx(0.06)
    assert syy == pytest.approx(0.07)
    assert sxy == pytest.approx(0.002)  # off-diagonal (correlation) untouched


def test_from_height_and_yaw_propagates_sigma():
    ft = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0, sigma_rot_rad=0.01, sigma_trans_m=0.02)
    assert ft.sigma_rot_rad == 0.01
    assert ft.sigma_trans_m == 0.02
    ft_default = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0)
    assert ft_default.sigma_rot_rad == 0.0
    assert ft_default.sigma_trans_m == 0.0


def _pixel_for_floor_point(ft, x, y, z=0.0):
    R_inv = ft.R.T
    t_inv = -R_inv @ ft.t
    p_cam = R_inv @ np.array([x, y, z]) + t_inv
    u = INTR.fx * p_cam[0] / p_cam[2] + INTR.cx
    v = INTR.fy * p_cam[1] / p_cam[2] + INTR.cy
    return u, v


def test_raycast_covariance_inflated_by_exact_extrinsic_amount():
    cfg = RaycastMeasurementConfig()
    exact = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0)
    uncertain = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0, sigma_rot_rad=np.radians(1.0), sigma_trans_m=0.01)
    u, v = _pixel_for_floor_point(exact, 3.0, 0.5)
    m_exact = compute_raycast_measurement((u, v), "ankles", INTR, exact, cfg)
    m_uncertain = compute_raycast_measurement((u, v), "ankles", INTR, uncertain, cfg)
    assert m_exact is not None and m_uncertain is not None
    ground_distance = float(np.linalg.norm(np.array(m_exact.position_xy) - exact.camera_origin_floor()[:2]))
    expected_extra = geometry.extrinsic_position_variance(ground_distance, uncertain.sigma_rot_rad, uncertain.sigma_trans_m)
    assert m_uncertain.cov_xy[0] == pytest.approx(m_exact.cov_xy[0] + expected_extra, rel=1e-6)
    assert m_uncertain.cov_xy[2] == pytest.approx(m_exact.cov_xy[2] + expected_extra, rel=1e-6)


def test_depth_covariance_inflated_by_extrinsic_uncertainty():
    cfg = DepthMeasurementConfig()
    exact = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0)
    uncertain = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0, sigma_rot_rad=np.radians(1.0), sigma_trans_m=0.01)

    depth_raw = np.zeros((480, 640), dtype=np.uint16)
    u, v = _pixel_for_floor_point(exact, 3.0, 0.3, z=1.25)
    depth_val_mm = 3200  # a plausible torso-range depth reading, in mm (0.001 depth_scale)
    y0, y1 = int(v) - 15, int(v) + 15
    x0, x1 = int(u) - 10, int(u) + 10
    depth_raw[y0:y1, x0:x1] = depth_val_mm
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    m_exact = compute_depth_measurement(depth_raw, poly, "full", INTR, exact, cfg)
    m_uncertain = compute_depth_measurement(depth_raw, poly, "full", INTR, uncertain, cfg)
    assert m_exact is not None and m_uncertain is not None
    assert m_uncertain.cov_xy[0] > m_exact.cov_xy[0]
    assert m_uncertain.cov_xy[2] > m_exact.cov_xy[2]


def test_1deg_extrinsic_pitch_error_dominates_sensor_noise_at_range():
    """Directly documents the magnitude that motivated this fix: at ~4m
    range, a 1deg extrinsic error contributes a LARGER position variance
    than realistic sensor noise does -- so omitting it (the old
    behavior) was not a minor rounding gap, it was ignoring the dominant
    error term."""
    sensor_only = geometry.extrinsic_position_variance(4.0, 0.0, 0.0)
    with_1deg = geometry.extrinsic_position_variance(4.0, np.radians(1.0), 0.0)
    sigma_1deg = with_1deg**0.5
    assert sensor_only == 0.0
    assert sigma_1deg > 0.06  # ~7cm at 4m from 1deg alone
