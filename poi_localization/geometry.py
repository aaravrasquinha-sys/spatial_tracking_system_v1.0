"""
Camera/floor geometry primitives, used everywhere else in this package.
Pure, dependency-free (numpy only) functions, unit-tested against hand
-solved cases -- this is the module where a sign error would silently
corrupt every downstream measurement, so it gets the most direct test
coverage in the whole package.

Coordinate convention (matches the D435i's optical frame, and
poi_perception.contracts.Intrinsics): camera space is X right, Y down,
Z forward (out of the lens). Floor space (either Phase A's local frame
or Phase B's room frame) is X/Y horizontal, Z up, with the floor at
Z = 0 -- this matches the full system plan's `room` frame convention
(Section 3 of the full plan) exactly, so Phase A and Phase B share this
axis convention and only differ in *which* transform gets you there.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from poi_perception.contracts import Intrinsics


def deproject_pixel(u: float, v: float, depth_m: float, intr: Intrinsics) -> np.ndarray:
    """Pixel + depth -> 3D point in camera space. depth_m is the Z
    (forward) coordinate directly -- not range along the ray -- matching
    how RealSense depth is defined (and Section 4's "depth is z, not
    range" note in the full plan)."""
    x = (u - intr.cx) / intr.fx * depth_m
    y = (v - intr.cy) / intr.fy * depth_m
    return np.array([x, y, depth_m], dtype=np.float64)


def pixel_ray_direction(u: float, v: float, intr: Intrinsics) -> np.ndarray:
    """Unit ray direction in camera space through pixel (u, v), Z=1 plane."""
    d = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0], dtype=np.float64)
    return d / np.linalg.norm(d)


def undistort_pixel(u: float, v: float, intr: Intrinsics, coeffs: Optional[Tuple] = None) -> Tuple[float, float]:
    """Section 5 of the M4 design doc: "don't assume [negligible
    distortion] without checking." If coeffs is None or all-zero
    (the D435i's color-aligned stream typically reports zero Brown-Conrady
    coefficients, per the full plan's calibration file example), this is
    a pass-through. Otherwise applies a standard Brown-Conrady undistort
    via OpenCV, imported lazily so geometry.py has no hard cv2 dependency
    for the (common) zero-distortion case."""
    if not coeffs or all(abs(c) < 1e-9 for c in coeffs):
        return u, v
    import cv2

    pts = np.array([[[u, v]]], dtype=np.float64)
    K = intr.K()
    D = np.array(coeffs, dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, K, D, P=K)
    return float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1])


# ---------------------------------------------------------------------------
# Rigid transforms between camera space and a floor-referenced frame.
# A transform is (R, t) such that p_floor = R @ p_cam + t. Both Phase A's
# local floor frame and Phase B's room frame are represented this way
# (see frames/types.py), so every function below works identically for
# either phase.
# ---------------------------------------------------------------------------


def transform_point(R: np.ndarray, t: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    return R @ p_cam + t


def transform_direction(R: np.ndarray, d_cam: np.ndarray) -> np.ndarray:
    """Directions (rays, velocities) rotate but don't translate."""
    return R @ d_cam


def invert_rigid_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    return R_inv, -R_inv @ t


def intersect_ray_with_floor(
    ray_origin_floor: np.ndarray, ray_dir_floor: np.ndarray, eps: float = 1e-6
) -> Optional[np.ndarray]:
    """Intersect a ray (in floor space, Z up, floor at Z=0) with the
    floor plane. Returns None if the ray doesn't hit the floor in front
    of the camera (points level or upward, i.e. dir.z >= -eps), per
    Section 5's "if the ray doesn't hit the floor in front of the
    camera... discard it, don't clamp it to some arbitrary point.\""""
    if ray_dir_floor[2] >= -eps:
        return None
    t_param = -ray_origin_floor[2] / ray_dir_floor[2]
    if t_param <= 0:
        return None
    point = ray_origin_floor + t_param * ray_dir_floor
    point[2] = 0.0  # exact, by construction -- avoid float drift downstream
    return point


def depression_angle(ray_dir_floor: np.ndarray) -> float:
    """Angle below horizontal, in radians, of a (unit) ray in floor
    space. 0 = horizontal, pi/2 = straight down. Used by the ray-cast
    uncertainty model (Section 5): "as the footpoint approaches the
    horizon... a small pixel error translates into a large ground-
    distance error," i.e. this angle -> 0."""
    return float(np.arcsin(np.clip(-ray_dir_floor[2], -1.0, 1.0)))


def radial_tangential_to_xy_cov(
    sigma_radial: float, sigma_tangential: float, direction_xy: np.ndarray
) -> Tuple[float, float, float]:
    """Build a 2x2 floor-plane XY covariance from an uncertainty that's
    anisotropic along a known bearing (both measurement models are:
    "uncertain mostly along the camera-to-person direction, much less
    so perpendicular to it") and rotate it from the (radial, tangential)
    basis into floor-frame XY. Returns the upper triangle (sxx, sxy, syy)
    in m^2, matching the poi.v1 schema's cov_xy convention."""
    norm = np.linalg.norm(direction_xy)
    if norm < 1e-9:
        # No well-defined bearing (person directly at the camera's floor
        # projection) -- fall back to isotropic using the larger sigma,
        # which is the conservative (never-too-confident) choice.
        s = max(sigma_radial, sigma_tangential)
        return float(s * s), 0.0, float(s * s)
    ux, uy = direction_xy / norm  # radial unit vector
    # Rotation matrix columns: radial = (ux, uy), tangential = (-uy, ux)
    var_r = sigma_radial**2
    var_t = sigma_tangential**2
    sxx = var_r * ux * ux + var_t * uy * uy
    syy = var_r * uy * uy + var_t * ux * ux
    sxy = (var_r - var_t) * ux * uy
    return float(sxx), float(sxy), float(syy)


def extrinsic_position_variance(
    range_m: float, sigma_rot_rad: float, sigma_trans_m: float
) -> float:
    """First-order variance a small calibration (extrinsic) error adds
    to a floor-frame XY position, on top of that measurement's own
    (photometric/pixel) noise.

    This is the fix for the single biggest accuracy gap the system had:
    depth and ray-cast covariances previously modeled ONLY sensor noise
    (disparity/pixel jitter) as if T_room_cam / the Phase A floor frame
    were exact. It never is -- Phase A's IMU+depth fit is good to
    perhaps 0.3-1.0 degrees and a centimeter or two; Phase B's M2
    calibration reports its own sigma_trans/sigma_rot in the calibration
    file precisely because it isn't exact either. An extrinsic angular
    error rotates the whole ray field about the camera origin, so its
    effect on a point at range r is, to first order, an arc-length
    displacement of r * sigma_rot_rad (exact for a rotation about an
    axis perpendicular to the ray; a reasonable isotropic approximation
    otherwise, since which axis dominates -- pitch vs. yaw vs. roll --
    isn't known at the measurement site). A translation error displaces
    every point by a constant amount regardless of range. The two are
    independent, so their variances add.

    This is deliberately a simplified, isotropic model rather than a
    full per-axis Jacobian of the rigid transform (which would need to
    know which rotation axis the calibration uncertainty is actually
    concentrated on, e.g. pitch dominates for a fixed-yaw tripod mount).
    It is still far more honest than omitting it entirely: a tracker
    that doesn't know its own extrinsics might be wrong should not
    report 1-2cm covariances that are wrong by an order of magnitude the
    moment the mount is bumped or the Phase A fit is a little off (see
    FloorFrameTransform.sigma_rot_rad / .sigma_trans_m -- this is the
    only place that value is consumed).
    """
    sigma_from_rot = max(range_m, 0.0) * max(sigma_rot_rad, 0.0)
    sigma_trans = max(sigma_trans_m, 0.0)
    return float(sigma_from_rot**2 + sigma_trans**2)


def add_isotropic_variance(cov_xy: Tuple[float, float, float], extra_variance: float) -> Tuple[float, float, float]:
    """Add an isotropic (extrinsic-uncertainty) variance to an existing
    XY covariance's upper triangle, in place of the caller re-deriving
    the addition by hand at every measurement-model call site. Isotropic
    noise adds only to the diagonal; the covariance's own anisotropy
    (radial vs. tangential) is left untouched."""
    sxx, sxy, syy = cov_xy
    return float(sxx + extra_variance), float(sxy), float(syy + extra_variance)


def mahalanobis_sq(delta: np.ndarray, cov: np.ndarray) -> float:
    """Squared Mahalanobis distance of `delta` under covariance `cov`.
    Used for every chi-square gate in this package (Section 6, 9)."""
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    return float(delta.T @ cov_inv @ delta)


def cov_upper_to_matrix(cov_xy: Tuple[float, float, float]) -> np.ndarray:
    sxx, sxy, syy = cov_xy
    return np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64)


def depth_image_to_points(
    depth_raw: np.ndarray,
    depth_scale: float,
    intr: Intrinsics,
    stride: int = 4,
    min_range_m: float = 0.2,
    max_range_m: float = 6.0,
) -> np.ndarray:
    """Vectorized deprojection of a (downsampled) depth image into an
    (N, 3) camera-space point cloud, skipping invalid (zero) and out-of-
    range pixels. `stride` subsamples the pixel grid -- floor-fitting
    doesn't need every pixel, and keeping the point count in the low
    thousands keeps RANSAC fast."""
    h, w = depth_raw.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    depth_m = depth_raw[vs, us].astype(np.float64) * depth_scale
    valid = (depth_m >= min_range_m) & (depth_m <= max_range_m)
    us, vs, depth_m = us[valid], vs[valid], depth_m[valid]

    x = (us - intr.cx) / intr.fx * depth_m
    y = (vs - intr.cy) / intr.fy * depth_m
    return np.stack([x, y, depth_m], axis=1)
