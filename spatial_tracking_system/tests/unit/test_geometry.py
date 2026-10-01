import numpy as np
import pytest

from poi_localization import geometry
from poi_perception.contracts import Intrinsics

INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001)


def test_deproject_pixel_at_center_gives_pure_depth():
    p = geometry.deproject_pixel(INTR.cx, INTR.cy, 2.5, INTR)
    assert np.allclose(p, [0.0, 0.0, 2.5])


def test_deproject_pixel_offset_matches_pinhole_formula():
    u, v, z = 320.0 + 60.0, 240.0 - 30.0, 3.0
    p = geometry.deproject_pixel(u, v, z, INTR)
    expected_x = (u - INTR.cx) / INTR.fx * z
    expected_y = (v - INTR.cy) / INTR.fy * z
    assert np.allclose(p, [expected_x, expected_y, z])


def test_pixel_ray_direction_is_unit_length():
    d = geometry.pixel_ray_direction(400, 100, INTR)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9


def test_pixel_ray_direction_matches_deprojected_point_direction():
    u, v = 380.0, 260.0
    p = geometry.deproject_pixel(u, v, 4.0, INTR)
    d = geometry.pixel_ray_direction(u, v, INTR)
    assert np.allclose(p / np.linalg.norm(p), d)


def test_undistort_pixel_passthrough_when_no_coeffs():
    u, v = geometry.undistort_pixel(123.4, 55.6, INTR, coeffs=None)
    assert (u, v) == (123.4, 55.6)
    u2, v2 = geometry.undistort_pixel(123.4, 55.6, INTR, coeffs=(0, 0, 0, 0, 0))
    assert (u2, v2) == (123.4, 55.6)


def test_transform_point_and_direction_and_invert_roundtrip():
    rng = np.random.default_rng(0)
    # a random-ish but valid rotation via QR decomposition
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    t = np.array([1.0, -2.0, 3.0])

    p_cam = np.array([0.5, -0.3, 2.0])
    p_floor = geometry.transform_point(Q, t, p_cam)

    R_inv, t_inv = geometry.invert_rigid_transform(Q, t)
    p_cam_roundtrip = geometry.transform_point(R_inv, t_inv, p_floor)
    assert np.allclose(p_cam_roundtrip, p_cam, atol=1e-9)

    d_cam = np.array([0.0, 0.0, 1.0])
    d_floor = geometry.transform_direction(Q, d_cam)
    d_cam_roundtrip = geometry.transform_direction(R_inv, d_floor)
    assert np.allclose(d_cam_roundtrip, d_cam, atol=1e-9)


def test_intersect_ray_with_floor_straight_down():
    origin = np.array([0.0, 0.0, 2.0])
    direction = np.array([0.0, 0.0, -1.0])
    hit = geometry.intersect_ray_with_floor(origin, direction)
    assert hit is not None
    assert np.allclose(hit, [0.0, 0.0, 0.0])


def test_intersect_ray_with_floor_angled():
    origin = np.array([0.0, 0.0, 2.0])
    direction = np.array([1.0, 0.0, -1.0])
    direction = direction / np.linalg.norm(direction)
    hit = geometry.intersect_ray_with_floor(origin, direction)
    # 45 degrees down from height 2 -> travels 2m forward
    assert hit is not None
    assert np.allclose(hit, [2.0, 0.0, 0.0], atol=1e-9)


def test_intersect_ray_with_floor_pointing_up_returns_none():
    origin = np.array([0.0, 0.0, 2.0])
    direction = np.array([1.0, 0.0, 0.1])
    direction = direction / np.linalg.norm(direction)
    assert geometry.intersect_ray_with_floor(origin, direction) is None


def test_intersect_ray_with_floor_horizontal_returns_none():
    origin = np.array([0.0, 0.0, 2.0])
    direction = np.array([1.0, 0.0, 0.0])
    assert geometry.intersect_ray_with_floor(origin, direction) is None


def test_depression_angle_straight_down_is_90deg():
    d = np.array([0.0, 0.0, -1.0])
    assert abs(geometry.depression_angle(d) - np.pi / 2) < 1e-9


def test_depression_angle_horizontal_is_zero():
    d = np.array([1.0, 0.0, 0.0])
    assert abs(geometry.depression_angle(d)) < 1e-9


def test_depression_angle_45deg():
    d = np.array([1.0, 0.0, -1.0])
    d = d / np.linalg.norm(d)
    assert abs(geometry.depression_angle(d) - np.pi / 4) < 1e-9


def test_radial_tangential_cov_pure_radial_along_x():
    sxx, sxy, syy = geometry.radial_tangential_to_xy_cov(2.0, 0.5, direction_xy=np.array([1.0, 0.0]))
    assert abs(sxx - 4.0) < 1e-9
    assert abs(syy - 0.25) < 1e-9
    assert abs(sxy - 0.0) < 1e-9


def test_radial_tangential_cov_along_y():
    sxx, sxy, syy = geometry.radial_tangential_to_xy_cov(2.0, 0.5, direction_xy=np.array([0.0, 3.0]))
    assert abs(sxx - 0.25) < 1e-9  # tangential now aligns with x
    assert abs(syy - 4.0) < 1e-9
    assert abs(sxy - 0.0) < 1e-9


def test_radial_tangential_cov_is_rotation_invariant_in_trace():
    # trace (sxx + syy) should equal var_r + var_t regardless of direction
    for angle in [0.0, 0.3, 1.1, 2.7]:
        d = np.array([np.cos(angle), np.sin(angle)])
        sxx, sxy, syy = geometry.radial_tangential_to_xy_cov(3.0, 1.0, d)
        assert abs((sxx + syy) - (9.0 + 1.0)) < 1e-9


def test_radial_tangential_cov_zero_direction_falls_back_isotropic():
    sxx, sxy, syy = geometry.radial_tangential_to_xy_cov(2.0, 0.5, direction_xy=np.array([0.0, 0.0]))
    assert abs(sxx - syy) < 1e-9
    assert abs(sxy) < 1e-9


def test_mahalanobis_sq_identity_cov_is_euclidean_sq():
    delta = np.array([3.0, 4.0])
    cov = np.eye(2)
    assert abs(geometry.mahalanobis_sq(delta, cov) - 25.0) < 1e-9


def test_mahalanobis_sq_scales_with_variance():
    delta = np.array([1.0, 0.0])
    cov = np.diag([4.0, 1.0])
    # variance 4 along x -> distance^2 = 1/4
    assert abs(geometry.mahalanobis_sq(delta, cov) - 0.25) < 1e-9


def test_cov_upper_to_matrix():
    m = geometry.cov_upper_to_matrix((1.0, 0.5, 2.0))
    assert np.allclose(m, [[1.0, 0.5], [0.5, 2.0]])


def test_depth_image_to_points_basic():
    depth = np.full((40, 60), 1000, dtype=np.uint16)  # 1.0m everywhere at scale 0.001
    depth[0:5, 0:5] = 0  # invalid patch
    intr = Intrinsics(fx=50, fy=50, cx=30, cy=20, width=60, height=40, depth_scale=0.001)
    pts = geometry.depth_image_to_points(depth, 0.001, intr, stride=2, min_range_m=0.1, max_range_m=5.0)
    assert pts.shape[1] == 3
    assert np.allclose(pts[:, 2], 1.0)
    assert pts.shape[0] > 0
