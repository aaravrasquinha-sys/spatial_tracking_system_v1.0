import numpy as np
import pytest

from poi_localization.frames.local_floor_frame import (
    FloorFitError,
    estimate_floor_frame_from_points,
)
from poi_localization.frames.types import FloorFrameTransform


def _synthesize_floor_points_in_camera_space(true_transform: FloorFrameTransform, n=3000, noise_m=0.003, seed=0):
    """Generate points on the floor plane (Z=0 in floor space), in a
    forward-facing wedge the camera would actually see, transform them
    into camera space via the true transform's inverse, and add small
    noise -- exactly what a real (very clean) depth capture of a flat
    floor would look like in camera coordinates."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 4.0, size=n)  # forward distance
    y = rng.uniform(-1.5, 1.5, size=n)  # lateral spread
    points_floor = np.stack([x, y, np.zeros(n)], axis=1)

    R, t = true_transform.R, true_transform.t
    R_inv = R.T
    t_inv = -R_inv @ t
    points_cam = (R_inv @ points_floor.T).T + t_inv

    points_cam += rng.normal(scale=noise_m, size=points_cam.shape)
    return points_cam


def test_recovers_known_transform_level_camera():
    true_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    points_cam = _synthesize_floor_points_in_camera_space(true_transform)
    up_hint_cam = true_transform.R.T @ np.array([0.0, 0.0, 1.0])

    recovered, diagnostics = estimate_floor_frame_from_points(
        points_cam, up_hint_cam=up_hint_cam, ransac_dist_thresh_m=0.02, ransac_iterations=200, seed=1
    )

    assert abs(recovered.camera_height_m - 2.0) < 0.01
    assert diagnostics.gravity_alignment_deg < 1.0
    assert diagnostics.inlier_rms_m < 0.01

    # A point known in floor space should round-trip through both
    # transforms consistently (the real end-to-end correctness check,
    # not just "height looks right").
    test_point_floor = np.array([1.5, -0.4, 0.0])
    R_inv = true_transform.R.T
    t_inv = -R_inv @ true_transform.t
    test_point_cam = R_inv @ test_point_floor + t_inv
    recovered_point_floor = recovered.apply_point(test_point_cam)
    assert np.allclose(recovered_point_floor, test_point_floor, atol=0.02)


def test_recovers_known_transform_steep_pitch():
    # Phase A's floor-frame X axis is, by definition (Section 2), the
    # camera's own forward direction flattened onto the floor -- there is
    # no externally-imposed "yaw" for Phase A to recover against (unlike
    # Phase B's room frame, which genuinely can differ from camera
    # forward). So this test varies pitch/height only, at yaw=0, where
    # X_floor and camera-forward-flattened necessarily coincide -- an
    # arbitrary synthesized "yaw" would test an invariant Phase A's frame
    # doesn't (and shouldn't) satisfy; see test_yaw_is_self_anchored_not_externally_recoverable below.
    true_transform = FloorFrameTransform.from_height_and_yaw(2.6, pitch_deg=32, yaw_deg=0)
    points_cam = _synthesize_floor_points_in_camera_space(true_transform, n=4000)
    up_hint_cam = true_transform.R.T @ np.array([0.0, 0.0, 1.0])

    recovered, diagnostics = estimate_floor_frame_from_points(
        points_cam, up_hint_cam=up_hint_cam, ransac_iterations=300, seed=2
    )
    assert abs(recovered.camera_height_m - 2.6) < 0.015

    test_point_floor = np.array([2.0, 0.8, 0.0])
    R_inv = true_transform.R.T
    t_inv = -R_inv @ true_transform.t
    test_point_cam = R_inv @ test_point_floor + t_inv
    recovered_point_floor = recovered.apply_point(test_point_cam)
    assert np.allclose(recovered_point_floor, test_point_floor, atol=0.03)


def test_yaw_is_self_anchored_not_externally_recoverable():
    """Phase A's frame always defines its own X axis as camera-forward-
    flattened (Section 2) -- so a camera that happens to be physically
    yawed relative to some external reference still produces a
    self-consistent frame where the recovered X axis aligns with
    camera-forward, not with the external reference. This test checks
    that self-anchoring invariant directly (rather than expecting an
    externally-defined yaw to round-trip, which it structurally can't
    for Phase A) and confirms distance/height are still preserved even
    though absolute bearing isn't externally meaningful here."""
    true_transform = FloorFrameTransform.from_height_and_yaw(2.6, pitch_deg=32, yaw_deg=-15)
    points_cam = _synthesize_floor_points_in_camera_space(true_transform, n=4000)
    up_hint_cam = true_transform.R.T @ np.array([0.0, 0.0, 1.0])

    recovered, _ = estimate_floor_frame_from_points(
        points_cam, up_hint_cam=up_hint_cam, ransac_iterations=300, seed=2
    )
    assert abs(recovered.camera_height_m - 2.6) < 0.015

    # Invariant: recovered X axis == camera-forward flattened, expressed
    # in the recovered frame itself.
    forward_cam = np.array([0.0, 0.0, 1.0])
    forward_recovered = recovered.apply_direction(forward_cam)
    forward_flat = forward_recovered.copy()
    forward_flat[2] = 0.0
    forward_flat /= np.linalg.norm(forward_flat)
    assert np.allclose(forward_flat, [1.0, 0.0, 0.0], atol=1e-6)

    # Distance from the camera's floor projection to a known point is a
    # yaw-independent, externally meaningful quantity -- it must still
    # be recovered correctly even though bearing (X, Y individually) isn't.
    test_point_floor = np.array([2.0, 0.8, 0.0])
    R_inv = true_transform.R.T
    t_inv = -R_inv @ true_transform.t
    test_point_cam = R_inv @ test_point_floor + t_inv
    recovered_point_floor = recovered.apply_point(test_point_cam)
    true_radius = np.linalg.norm(test_point_floor[:2] - true_transform.t[:2])
    recovered_radius = np.linalg.norm(recovered_point_floor[:2] - recovered.t[:2])
    assert abs(true_radius - recovered_radius) < 0.03
    assert abs(recovered_point_floor[2]) < 0.01  # still on the floor


def test_recovers_without_gravity_hint_but_less_precisely():
    true_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    points_cam = _synthesize_floor_points_in_camera_space(true_transform)
    recovered, diagnostics = estimate_floor_frame_from_points(
        points_cam, up_hint_cam=None, ransac_iterations=300, seed=3
    )
    assert abs(recovered.camera_height_m - 2.0) < 0.03
    assert diagnostics.gravity_alignment_deg is None


def test_outliers_off_the_floor_plane_are_rejected_by_ransac():
    true_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    points_cam = _synthesize_floor_points_in_camera_space(true_transform, n=2000)
    up_hint_cam = true_transform.R.T @ np.array([0.0, 0.0, 1.0])

    # Inject a cluster of "furniture" points well off the floor plane,
    # still within the gravity-selected candidate band in some cases --
    # RANSAC (not just the gravity pre-filter) needs to reject these.
    rng = np.random.default_rng(4)
    n_outliers = 400
    furniture_floor = np.stack(
        [rng.uniform(0.5, 2.0, n_outliers), rng.uniform(-1, 1, n_outliers), rng.uniform(0.05, 0.4, n_outliers)],
        axis=1,
    )
    R_inv = true_transform.R.T
    t_inv = -R_inv @ true_transform.t
    furniture_cam = (R_inv @ furniture_floor.T).T + t_inv
    all_points = np.vstack([points_cam, furniture_cam])

    recovered, diagnostics = estimate_floor_frame_from_points(
        all_points, up_hint_cam=up_hint_cam, ransac_iterations=300, seed=5
    )
    assert abs(recovered.camera_height_m - 2.0) < 0.02


def test_too_few_points_raises():
    with pytest.raises(FloorFitError):
        estimate_floor_frame_from_points(np.zeros((5, 3)), min_candidate_points=50)


def test_gravity_disagreement_raises():
    true_transform = FloorFrameTransform.from_height_and_yaw(2.0, pitch_deg=20, yaw_deg=0)
    points_cam = _synthesize_floor_points_in_camera_space(true_transform)
    # A wildly wrong "up" hint (perpendicular-ish to the true up) should
    # cause the gravity-alignment check to fire.
    bad_hint = np.array([1.0, 0.0, 0.0])
    with pytest.raises(FloorFitError):
        estimate_floor_frame_from_points(
            points_cam, up_hint_cam=bad_hint, gravity_alignment_deg_max=15.0, seed=6
        )
