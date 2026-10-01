"""
Section 2, Phase A: "using only the D435i's own IMU (gravity direction)
and its own depth stream (RANSAC floor fit... done live, once, at
startup), define a provisional floor-referenced coordinate frame."

Split into two layers on purpose:
  - estimate_floor_frame_from_points(): pure numpy, no hardware --
    RANSAC + refine + build the frame, unit-testable against synthetic
    point clouds with a known ground-truth plane.
  - capture_calibration_data(): the one piece that touches hardware
    (guarded pyrealsense2 import), isolated so it's easy to swap for a
    fixed/canned transform in tests and synthetic demos (see
    frames.types.FloorFrameTransform.from_height_and_yaw and
    M4Config.frame.fixed_transform).

Sign convention: `up_hint_cam` is a unit vector in camera space that
points AWAY from the floor (physically "up"). A stationary MEMS
accelerometer's raw reading (specific force = normal-force reaction,
not gravity itself) points in exactly this direction, so
capture_calibration_data() can hand the normalized mean accel reading
straight through as up_hint_cam with no sign flip -- verified against
the D435i's accel convention in this module's tests where practical,
but flagged here explicitly since a flipped sign would silently invert
the whole floor frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from poi_localization import geometry
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.log import get_logger
from poi_perception.contracts import Intrinsics

log = get_logger("frames.local_floor_frame")


@dataclass
class FloorFitDiagnostics:
    n_candidate_points: int
    n_inliers: int
    inlier_rms_m: float
    gravity_alignment_deg: Optional[float]
    camera_height_m: float


class FloorFitError(RuntimeError):
    """Raised when the floor fit fails badly enough that proceeding
    would silently produce a wrong frame -- e.g. too few candidate
    points, or (if a gravity hint was given) the fitted plane disagrees
    with gravity by more than the configured tolerance."""


def _ransac_plane(points: np.ndarray, dist_thresh: float, iterations: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (best_normal, inlier_mask). Normal is unit-length but
    unoriented (sign fixed later using up_hint_cam)."""
    n = points.shape[0]
    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1

    idx = rng.integers(0, n, size=(iterations, 3))
    for i in range(iterations):
        i0, i1, i2 = idx[i]
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue
        p0, p1, p2 = points[i0], points[i1], points[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue  # degenerate (collinear) triplet
        normal = normal / norm_len
        d = -normal @ p0
        residual = points @ normal + d
        inliers = np.abs(residual) < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    return best_inliers, best_count


def _refine_plane_least_squares(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through `points` via SVD. Returns (unit_normal, centroid)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]  # smallest-singular-value direction = plane normal
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


def estimate_floor_frame_from_points(
    points_cam: np.ndarray,
    up_hint_cam: Optional[np.ndarray] = None,
    ransac_dist_thresh_m: float = 0.02,
    ransac_iterations: int = 500,
    gravity_alignment_deg_max: float = 15.0,
    candidate_percentile: float = 80.0,
    min_candidate_points: int = 50,
    seed: int = 0,
    frame_name: str = "local",
    min_sigma_rot_deg: float = 0.3,
    min_sigma_trans_m: float = 0.01,
) -> Tuple[FloorFrameTransform, FloorFitDiagnostics]:
    """Fit the floor plane in camera-space points and build the
    corresponding FloorFrameTransform (Section 2). Raises FloorFitError
    if the result can't be trusted (too few points, or a gravity hint
    that strongly disagrees with the fitted plane)."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.shape[0] < min_candidate_points:
        raise FloorFitError(
            f"Only {points_cam.shape[0]} depth points available; need at least "
            f"{min_candidate_points}. Point the camera somewhere with more visible "
            f"floor (Section 14 risk: tilt the camera down during Phase A setup)."
        )

    if up_hint_cam is not None:
        up_hint_cam = np.asarray(up_hint_cam, dtype=np.float64)
        up_hint_cam = up_hint_cam / np.linalg.norm(up_hint_cam)
        down_hint = -up_hint_cam
        # M1's pattern, at M4's smaller live scale (Section 2): candidate
        # floor points are the ones furthest in the "down" direction --
        # the lowest-height cluster -- not the whole scene.
        projection = points_cam @ down_hint
        threshold = np.percentile(projection, candidate_percentile)
        candidate = points_cam[projection >= threshold]
        if candidate.shape[0] < min_candidate_points:
            log.warning(
                f"Only {candidate.shape[0]} points in the gravity-selected floor "
                f"candidate cluster; falling back to the full point cloud for RANSAC."
            )
            candidate = points_cam
    else:
        log.warning("No gravity/up hint available -- RANSAC over the full point "
                    "cloud with no floor-ward seed. Verify the result visually.")
        candidate = points_cam

    rng = np.random.default_rng(seed)
    inlier_mask, n_inliers = _ransac_plane(candidate, ransac_dist_thresh_m, ransac_iterations, rng)
    if n_inliers < min_candidate_points:
        raise FloorFitError(
            f"RANSAC found only {n_inliers} inliers (need {min_candidate_points}). "
            f"The candidate cluster may not actually be the floor."
        )

    normal, centroid = _refine_plane_least_squares(candidate[inlier_mask])
    residuals = candidate[inlier_mask] @ normal - normal @ centroid
    inlier_rms = float(np.sqrt(np.mean(residuals**2)))

    gravity_alignment_deg = None
    if up_hint_cam is not None:
        # Orient normal to agree with the hint, THEN measure disagreement
        # -- otherwise a valid plane found with the "wrong" SVD sign
        # would spuriously read as a near-180 degree disagreement.
        if normal @ up_hint_cam < 0:
            normal = -normal
        cos_angle = float(np.clip(normal @ up_hint_cam, -1.0, 1.0))
        gravity_alignment_deg = float(np.degrees(np.arccos(cos_angle)))
        if gravity_alignment_deg > gravity_alignment_deg_max:
            raise FloorFitError(
                f"Fitted floor normal disagrees with the IMU gravity direction by "
                f"{gravity_alignment_deg:.1f} degrees (max allowed: "
                f"{gravity_alignment_deg_max}). Likely a bad RANSAC fit (not enough "
                f"real floor in view) rather than a sensor problem -- retry with the "
                f"camera tilted to see more floor."
            )
    else:
        # No hint to check sign against -- assume the camera is above the
        # floor, so the normal should point back toward the camera origin.
        if normal @ (-centroid) < 0:
            normal = -normal

    z_floor_in_cam = normal
    forward_cam = np.array([0.0, 0.0, 1.0])
    forward_flat = forward_cam - (forward_cam @ z_floor_in_cam) * z_floor_in_cam
    flat_norm = np.linalg.norm(forward_flat)
    if flat_norm < 1e-6:
        raise FloorFitError(
            "Camera is looking almost exactly along the floor normal (straight up or "
            "down) -- forward direction doesn't project meaningfully onto the floor "
            "plane, so the X axis is undefined. This shouldn't happen for a normally "
            "mounted camera."
        )
    x_floor_in_cam = forward_flat / flat_norm
    y_floor_in_cam = np.cross(z_floor_in_cam, x_floor_in_cam)

    R = np.stack([x_floor_in_cam, y_floor_in_cam, z_floor_in_cam], axis=0)
    height_m = float(-z_floor_in_cam @ centroid)
    if height_m <= 0:
        raise FloorFitError(
            f"Computed a non-positive camera height ({height_m:.3f} m) above the "
            f"fitted floor plane -- the plane fit is almost certainly wrong."
        )
    t = np.array([0.0, 0.0, height_m])

    # Extrinsic uncertainty for this transform (geometry.py's
    # extrinsic_position_variance consumes this downstream). Phase A has
    # no independent check on yaw at all (the IMU only gives gravity,
    # i.e. roll/pitch; yaw comes from "camera forward", unverified), and
    # the gravity-alignment check only bounds roll/pitch disagreement
    # with the accelerometer -- it says nothing about how well the
    # RANSAC+SVD plane fit itself is oriented if the visible floor patch
    # was small or nearly all at one range (a common Phase-A situation:
    # camera on a desk, most floor points at a similar depth). So the
    # rotation sigma is the larger of (a) a floor representing "Phase A
    # is never better than this even on a good fit" and (b) the measured
    # gravity disagreement, when available -- NOT gravity disagreement
    # alone, which can read near-zero even when yaw is poorly
    # constrained. Translation sigma uses the plane fit's own inlier RMS
    # (a direct measure of how flat/well-fit the floor patch was) against
    # the same kind of floor.
    sigma_rot_deg = min_sigma_rot_deg
    if gravity_alignment_deg is not None:
        sigma_rot_deg = max(sigma_rot_deg, gravity_alignment_deg)
    sigma_trans = max(min_sigma_trans_m, inlier_rms)

    transform = FloorFrameTransform(
        R=R, t=t, frame_name=frame_name,
        sigma_rot_rad=float(np.radians(sigma_rot_deg)),
        sigma_trans_m=float(sigma_trans),
    )
    diagnostics = FloorFitDiagnostics(
        n_candidate_points=candidate.shape[0],
        n_inliers=n_inliers,
        inlier_rms_m=inlier_rms,
        gravity_alignment_deg=gravity_alignment_deg,
        camera_height_m=height_m,
    )
    log.info(
        f"Local floor frame estimated: height={height_m:.3f}m, "
        f"inliers={n_inliers}/{candidate.shape[0]}, rms={inlier_rms*1000:.1f}mm"
        + (f", gravity_alignment={gravity_alignment_deg:.1f}deg" if gravity_alignment_deg is not None else "")
    )
    return transform, diagnostics


@dataclass
class CalibrationCaptureData:
    points_cam: np.ndarray  # (N, 3)
    up_hint_cam: Optional[np.ndarray]  # unit vector, or None if IMU capture failed
    intr: Intrinsics


def capture_calibration_data(cam_cfg, seconds: float = 2.5, depth_stride: int = 4) -> CalibrationCaptureData:
    """Hardware-touching step: opens its OWN short-lived RealSense
    pipeline (accel + depth), separate from the main capture source used
    for the continuous run, captures `seconds` of stationary accel
    samples plus one depth frame, then closes cleanly. Do this BEFORE
    constructing the main poi_perception.capture.realsense_source.RealSenseSource
    for the same physical device -- opening two pipelines on one device
    at once is unreliable across librealsense versions.

    The camera MUST be held still (in its final mount, or wherever
    Phase A validation is happening) for the full capture window --
    moving it invalidates both the accel average and, less obviously,
    can blur/corrupt the depth frame used for the floor fit.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise RuntimeError(
            "pyrealsense2 is not importable -- Phase A calibration capture needs "
            "real hardware. For tests/synthetic demos, use "
            "FloorFrameTransform.from_height_and_yaw() via M4Config.frame.fixed_transform instead."
        ) from e

    pipeline = rs.pipeline()
    cfg = rs.config()
    if cam_cfg.serial:
        cfg.enable_device(cam_cfg.serial)
    cfg.enable_stream(rs.stream.depth, cam_cfg.width, cam_cfg.height, rs.format.z16, cam_cfg.fps)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)

    profile = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    dintr = depth_stream.get_intrinsics()
    intr = Intrinsics(
        fx=dintr.fx, fy=dintr.fy, cx=dintr.ppx, cy=dintr.ppy,
        width=dintr.width, height=dintr.height, depth_scale=depth_scale,
    )

    accel_samples = []
    depth_raw = None
    log.info(f"Capturing {seconds:.1f}s of accel + a depth frame for the local floor fit "
             f"-- keep the camera perfectly still.")
    try:
        import time as _time
        t_end = _time.time() + seconds
        while _time.time() < t_end:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_raw = np.asanyarray(depth_frame.get_data()).copy()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                d = accel_frame.as_motion_frame().get_motion_data()
                accel_samples.append((d.x, d.y, d.z))
    finally:
        pipeline.stop()

    if depth_raw is None:
        raise RuntimeError("No depth frame captured during the calibration window.")

    points_cam = geometry.depth_image_to_points(depth_raw, depth_scale, intr, stride=depth_stride)

    up_hint_cam = None
    if accel_samples:
        mean_accel = np.mean(np.array(accel_samples), axis=0)
        norm = np.linalg.norm(mean_accel)
        if norm > 1e-6:
            up_hint_cam = mean_accel / norm
            log.info(f"Accel-derived up_hint_cam = {up_hint_cam} (from {len(accel_samples)} samples)")
    if up_hint_cam is None:
        log.warning("No usable accel samples -- proceeding without a gravity hint "
                    "(RANSAC over the full point cloud; verify the result visually).")

    return CalibrationCaptureData(points_cam=points_cam, up_hint_cam=up_hint_cam, intr=intr)
