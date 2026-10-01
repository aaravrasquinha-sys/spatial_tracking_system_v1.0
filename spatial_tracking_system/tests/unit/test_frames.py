import numpy as np

from poi_localization.frames.types import FloorFrameTransform


def test_camera_height_matches_t_z():
    t = FloorFrameTransform.from_height_and_yaw(2.3, pitch_deg=0, yaw_deg=0)
    assert abs(t.camera_height_m - 2.3) < 1e-9


def test_camera_origin_maps_to_its_own_height():
    t = FloorFrameTransform.from_height_and_yaw(1.8, pitch_deg=15, yaw_deg=40)
    p = t.apply_point(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(p, [0.0, 0.0, 1.8])


def test_rotation_is_orthonormal_proper():
    t = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=25, yaw_deg=-60)
    should_be_I = t.R @ t.R.T
    assert np.allclose(should_be_I, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(t.R) - 1.0) < 1e-9


def test_level_camera_down_direction_points_toward_floor():
    """Physical sanity check: a level (pitch=0), unrolled camera's image
    -down- direction (+Y_cam) must map to -Z in floor space (toward the
    floor), not +Z. This is exactly the sign bug caught during
    development -- kept as a regression test."""
    t = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=0, yaw_deg=0)
    down_cam = np.array([0.0, 1.0, 0.0])
    down_floor_direction = t.apply_direction(down_cam)
    assert down_floor_direction[2] < -0.9  # strongly negative Z


def test_level_camera_forward_direction_is_horizontal():
    t = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=0, yaw_deg=0)
    forward_cam = np.array([0.0, 0.0, 1.0])
    forward_floor_direction = t.apply_direction(forward_cam)
    assert abs(forward_floor_direction[2]) < 1e-9  # no vertical component
    assert forward_floor_direction[0] > 0.9  # points along +X by convention at yaw=0


def test_pitched_down_camera_forward_has_negative_z_component():
    t = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=30, yaw_deg=0)
    forward_cam = np.array([0.0, 0.0, 1.0])
    forward_floor_direction = t.apply_direction(forward_cam)
    assert forward_floor_direction[2] < 0  # tilted down -> looking toward -Z


def test_yaw_rotates_forward_direction_in_xy_plane():
    t = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=0, yaw_deg=90)
    forward_cam = np.array([0.0, 0.0, 1.0])
    forward_floor_direction = t.apply_direction(forward_cam)
    assert np.allclose(forward_floor_direction, [0.0, 1.0, 0.0], atol=1e-9)


def test_apply_direction_has_no_translation_component():
    t = FloorFrameTransform.from_height_and_yaw(5.0, pitch_deg=10, yaw_deg=10)
    d1 = t.apply_direction(np.array([1.0, 0.0, 0.0]))
    d2 = t.apply_direction(np.array([2.0, 0.0, 0.0]))
    assert np.allclose(d2, 2 * d1)  # linear, no offset
