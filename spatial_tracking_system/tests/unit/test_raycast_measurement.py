import numpy as np
import pytest

from poi_localization.config import RaycastMeasurementConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.raycast_measurement import compute_raycast_measurement
from poi_perception.contracts import Intrinsics

INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001, baseline=0.05)
CFG = RaycastMeasurementConfig()


def _pixel_for_floor_point(floor_transform, x, y):
    R_inv = floor_transform.R.T
    t_inv = -R_inv @ floor_transform.t
    p_cam = R_inv @ np.array([x, y, 0.0]) + t_inv
    u = INTR.fx * p_cam[0] / p_cam[2] + INTR.cx
    v = INTR.fy * p_cam[1] / p_cam[2] + INTR.cy
    return u, v


def test_raycast_recovers_known_floor_point():
    ft = FloorFrameTransform.from_height_and_yaw(2.3, pitch_deg=25, yaw_deg=0)
    u, v = _pixel_for_floor_point(ft, 2.0, 0.4)
    m = compute_raycast_measurement((u, v), "ankles", INTR, ft, CFG)
    assert m is not None
    assert m.src == "raycast"
    assert np.allclose(m.position_xy, [2.0, 0.4], atol=1e-6)


def test_raycast_matches_across_footpoint_sources_same_geometry():
    ft = FloorFrameTransform.from_height_and_yaw(2.3, pitch_deg=25, yaw_deg=0)
    u, v = _pixel_for_floor_point(ft, 1.5, -0.2)
    m_ankles = compute_raycast_measurement((u, v), "ankles", INTR, ft, CFG)
    m_single = compute_raycast_measurement((u, v), "ankle_single", INTR, ft, CFG)
    m_bbox = compute_raycast_measurement((u, v), "bbox_fallback", INTR, ft, CFG)
    assert m_ankles is not None and m_single is not None and m_bbox is not None
    # same geometry -> same recovered position...
    assert np.allclose(m_ankles.position_xy, m_single.position_xy, atol=1e-9)
    assert np.allclose(m_ankles.position_xy, m_bbox.position_xy, atol=1e-9)
    # ...both-ankles is always the most trustworthy...
    assert m_ankles.cov_xy[0] < m_single.cov_xy[0]
    assert m_ankles.cov_xy[0] < m_bbox.cov_xy[0]
    # ...but ankle_single is no longer guaranteed to beat bbox_fallback at
    # short range: ankle_single carries a range-INDEPENDENT gait-bias
    # floor (RaycastMeasurementConfig.ankle_single_extra_sigma_m, a
    # single visible ankle is biased toward whichever foot is forward by
    # up to half a stride, not just noisier), while bbox_fallback's
    # sigma is purely pixel-noise-driven and shrinks with range. At this
    # short range (1.5m) the gait-bias floor legitimately dominates and
    # ankle_single can exceed bbox_fallback; see the dedicated test below
    # for the crossover behavior across range.


def test_ankle_single_gait_bias_floor_dominates_at_short_range_only():
    """ankle_single's uncertainty should be range-independent-dominated
    (roughly constant, from the gait-bias floor) at short range, and
    pixel-noise-dominated (growing with range^2-ish, like the other
    sources) at long range -- i.e. it should NOT simply track
    pixel_noise_ankle_single_px the way it did before the gait-bias term
    was added."""
    ft = FloorFrameTransform.from_height_and_yaw(2.3, pitch_deg=25, yaw_deg=0)
    near_u, near_v = _pixel_for_floor_point(ft, 1.0, 0.0)
    far_u, far_v = _pixel_for_floor_point(ft, 6.0, 0.0)
    m_near = compute_raycast_measurement((near_u, near_v), "ankle_single", INTR, ft, CFG)
    m_far = compute_raycast_measurement((far_u, far_v), "ankle_single", INTR, ft, CFG)
    assert m_near is not None and m_far is not None
    sigma_near = m_near.cov_xy[0] ** 0.5
    sigma_far = m_far.cov_xy[0] ** 0.5
    # near-range sigma should be close to the gait-bias floor itself
    # (0.15m), not the sub-centimeter figure pixel noise alone would give
    assert sigma_near > 0.5 * CFG.ankle_single_extra_sigma_m
    # far-range sigma must still exceed the near-range floor (pixel noise
    # + extrinsic-style growth with range continues to matter)
    assert sigma_far > sigma_near


def test_raycast_ray_pointing_up_returns_none():
    ft = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=0, yaw_deg=0)
    # image row above the principal point, with a level camera, is above
    # the horizon -- ray points up/level, must not hit the floor.
    m = compute_raycast_measurement((320.0, 100.0), "ankles", INTR, ft, CFG)
    assert m is None


def test_raycast_none_beyond_max_range():
    ft = FloorFrameTransform.from_height_and_yaw(2.3, pitch_deg=25, yaw_deg=0)
    u, v = _pixel_for_floor_point(ft, CFG.max_range_m + 5.0, 0.0)
    m = compute_raycast_measurement((u, v), "ankles", INTR, ft, CFG)
    assert m is None


def test_raycast_uncertainty_grows_as_footpoint_approaches_horizon():
    ft = FloorFrameTransform.from_height_and_yaw(2.3, pitch_deg=25, yaw_deg=0)
    u1, v1 = _pixel_for_floor_point(ft, 1.0, 0.0)
    u2, v2 = _pixel_for_floor_point(ft, 4.0, 0.0)  # further -> closer to horizon -> shallower depression angle
    m1 = compute_raycast_measurement((u1, v1), "ankles", INTR, ft, CFG)
    m2 = compute_raycast_measurement((u2, v2), "ankles", INTR, ft, CFG)
    assert m1 is not None and m2 is not None
    assert m2.cov_xy[0] > m1.cov_xy[0]


def test_raycast_bbox_fallback_rejected_near_horizon_past_ceiling():
    # True depression angle to a floor point at ground-distance x from a
    # camera at height h is atan(h/x), independent of the camera's own
    # pitch (pitch only affects where in the image the ray appears, not
    # the ray's actual 3D direction) -- so reaching a small enough angle
    # to blow past the uncertainty ceiling needs a genuinely large ground
    # distance. Loosen max_range_m for this test only; the ceiling itself
    # stays at the config default.
    cfg = RaycastMeasurementConfig(max_range_m=50.0)
    ft = FloorFrameTransform.from_height_and_yaw(2.3, pitch_deg=25, yaw_deg=0)
    u, v = _pixel_for_floor_point(ft, 20.0, 0.0)
    m_ankles = compute_raycast_measurement((u, v), "ankles", INTR, ft, cfg)
    m_bbox = compute_raycast_measurement((u, v), "bbox_fallback", INTR, ft, cfg)
    assert m_ankles is not None  # small pixel noise -> still under ceiling
    assert m_bbox is None  # large pixel noise at a shallow angle -> over ceiling, rejected
