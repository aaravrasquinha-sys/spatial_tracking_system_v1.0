"""
Section 4: the stronger measurement at close-to-moderate range.

Procedure exactly as specified: valid depth pixels inside the torso
polygon -> reject invalid/outlier pixels -> median depth -> deproject
the polygon centroid -> body-thickness offset along the (floor-plane)
horizontal component of the viewing direction -> transform to the floor
frame -> drop to Z=0. Returns None (never a forced bad number) when the
measurement isn't trustworthy this frame.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from poi_localization import geometry
from poi_localization.config import DepthMeasurementConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.types import Measurement
from poi_perception.contracts import Intrinsics

# Converts a median-absolute-deviation to an approximately-equivalent
# standard deviation for a normal distribution -- standard MAD-to-sigma
# constant, used only to size the outlier-rejection band.
_MAD_TO_SIGMA = 1.4826


def _polygon_mask(polygon_px: List[Tuple[float, float]], height: int, width: int) -> np.ndarray:
    import cv2

    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon_px], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def median_depth_in_polygon(
    depth_raw: np.ndarray,
    torso_polygon_px: List[Tuple[float, float]],
    torso_quality: str,
    intr: Intrinsics,
    cfg: DepthMeasurementConfig,
) -> Optional[float]:
    """The median-depth-with-outlier-rejection half of Section 4, factored
    out so tracking/height_estimator.py's fallback depth (Section 7:
    "transformed the same way as the depth measurement") can reuse the
    exact same value compute_depth_measurement used, rather than
    recomputing a slightly different number."""
    if torso_quality == "fallback" or len(torso_polygon_px) < 3:
        return None

    h, w = depth_raw.shape[:2]
    mask = _polygon_mask(torso_polygon_px, h, w)
    polygon_area_px = int(mask.sum())
    if polygon_area_px == 0:
        return None

    valid_mask = mask & (depth_raw > 0)
    n_valid = int(valid_mask.sum())
    if n_valid < cfg.min_valid_px_frac * polygon_area_px:
        return None

    depths_m = depth_raw[valid_mask].astype(np.float64) * intr.depth_scale
    median = float(np.median(depths_m))
    mad = float(np.median(np.abs(depths_m - median)))
    reject_band = max(cfg.mad_reject_k * mad * _MAD_TO_SIGMA, cfg.mad_floor_m)
    inliers = depths_m[np.abs(depths_m - median) <= reject_band]
    if inliers.size == 0:
        inliers = depths_m
    return float(np.median(inliers))


def estimate_body_thickness_offset_m(
    shoulder_l_px: Optional[Tuple[float, float]],
    shoulder_r_px: Optional[Tuple[float, float]],
    depth_m: float,
    intr: Intrinsics,
    floor_transform: FloorFrameTransform,
    viewing_dir_floor_xy: np.ndarray,
    cfg: DepthMeasurementConfig,
) -> float:
    """How far to move from the body's front surface toward its center,
    along the viewing direction. A single fixed body_thickness_m is only
    correct when the person faces (or has their back to) the camera; a
    person seen from the side needs roughly their half-SHOULDER-WIDTH
    offset instead, which is typically 1.5-2x their chest depth -- the
    old fixed constant under-corrected side-on people by several
    centimeters and over-corrected facing-on people by the same amount,
    a systematic error the Kalman filter has no way to average out
    (it's the same sign every frame for a given walking direction).

    Estimated from the shoulder line's orientation relative to the
    viewing ray: shoulder line roughly PERPENDICULAR to the view ==
    facing/away (use body_thickness_m); shoulder line roughly PARALLEL
    to the view == profile (use body_thickness_side_m); interpolated by
    |cos| of the angle between them for anything in between. Falls back
    to body_thickness_m when both shoulders aren't available (the common
    case for "partial"/"fallback" torso quality) -- an isotropic guess is
    still better than none, and this function only ever changes which
    guess is used, never removes the correction.
    """
    if shoulder_l_px is None or shoulder_r_px is None:
        return cfg.body_thickness_m

    p_l = geometry.deproject_pixel(shoulder_l_px[0], shoulder_l_px[1], depth_m, intr)
    p_r = geometry.deproject_pixel(shoulder_r_px[0], shoulder_r_px[1], depth_m, intr)
    shoulder_dir_floor = floor_transform.apply_direction(p_r - p_l)[:2]
    shoulder_norm = np.linalg.norm(shoulder_dir_floor)
    view_norm = np.linalg.norm(viewing_dir_floor_xy)
    if shoulder_norm < 1e-6 or view_norm < 1e-6:
        return cfg.body_thickness_m

    cos_angle = abs(float(np.dot(shoulder_dir_floor, viewing_dir_floor_xy)) / (shoulder_norm * view_norm))
    cos_angle = min(cos_angle, 1.0)
    return cfg.body_thickness_m + (cfg.body_thickness_side_m - cfg.body_thickness_m) * cos_angle


def compute_depth_measurement(
    depth_raw: np.ndarray,
    torso_polygon_px: List[Tuple[float, float]],
    torso_quality: str,
    intr: Intrinsics,
    floor_transform: FloorFrameTransform,
    cfg: DepthMeasurementConfig,
    shoulder_l_px: Optional[Tuple[float, float]] = None,
    shoulder_r_px: Optional[Tuple[float, float]] = None,
) -> Optional[Measurement]:
    depth_final = median_depth_in_polygon(depth_raw, torso_polygon_px, torso_quality, intr, cfg)
    if depth_final is None:
        return None
    if not (cfg.min_range_m <= depth_final <= cfg.max_range_m):
        return None

    centroid_px = np.mean(np.array(torso_polygon_px, dtype=np.float64), axis=0)
    point_cam_front = geometry.deproject_pixel(centroid_px[0], centroid_px[1], depth_final, intr)

    ray_dir_cam = point_cam_front / np.linalg.norm(point_cam_front)
    ray_dir_floor = floor_transform.apply_direction(ray_dir_cam)  # unit, since R is a rotation

    point_floor_front = floor_transform.apply_point(point_cam_front)
    camera_floor_xy = floor_transform.camera_origin_floor()[:2]
    viewing_dir_floor_xy = point_floor_front[:2] - camera_floor_xy
    xy_norm = np.linalg.norm(viewing_dir_floor_xy)
    if xy_norm < 1e-6:
        viewing_dir_unit_xy = np.array([1.0, 0.0])  # degenerate: point directly under the camera
    else:
        viewing_dir_unit_xy = viewing_dir_floor_xy / xy_norm

    # Body-thickness offset (Section 4): move from "front surface" to
    # roughly body center, along the floor-plane-parallel component of
    # the viewing direction -- computed in the floor frame (where
    # "horizontal" is unambiguous) rather than in camera space (where it
    # would depend on the camera's own tilt), a deliberate reading of the
    # doc's "horizontal component of the viewing direction" for a
    # possibly-tilted camera. Orientation-dependent when shoulder
    # keypoints are available (see estimate_body_thickness_offset_m);
    # otherwise the old fixed facing-on constant.
    thickness_m = estimate_body_thickness_offset_m(
        shoulder_l_px, shoulder_r_px, depth_final, intr, floor_transform, viewing_dir_floor_xy, cfg
    )
    offset_xy = viewing_dir_unit_xy * thickness_m
    position_xy = (float(point_floor_front[0] + offset_xy[0]), float(point_floor_front[1] + offset_xy[1]))

    # Uncertainty: sigma_z from a two-term (range-floor + range^2) noise
    # model calibrated to the D435i's actual stereo geometry, not the
    # color stream's focal length -- see DepthMeasurementConfig's
    # docstring for why the previous single-term formula understated
    # sigma_z by roughly 2-3x. Projected onto the floor plane by how
    # horizontal the viewing ray actually is (a ray close to horizontal
    # projects nearly all its depth-axis error onto floor-plane distance;
    # a ray pointing more steeply down projects much less of it).
    f_depth = cfg.depth_stereo_fx_px
    sigma_z_precision = depth_final**2 * cfg.sigma_disparity_px / (f_depth * intr.baseline)
    sigma_z_bias = depth_final * cfg.depth_bias_frac
    sigma_z = float(np.hypot(sigma_z_precision, sigma_z_bias))
    horizontal_fraction = float(np.linalg.norm(ray_dir_floor[:2]))  # in [0, 1]
    sigma_radial = sigma_z * horizontal_fraction
    sigma_tangential = depth_final * cfg.lateral_px_noise / intr.fx  # centroid pixel jitter uses the COLOR fx (correct: centroid is a color-image pixel)

    cov_xy = geometry.radial_tangential_to_xy_cov(sigma_radial, sigma_tangential, viewing_dir_floor_xy)

    # Extrinsic (calibration) uncertainty, on top of sensor noise -- see
    # geometry.extrinsic_position_variance's docstring. Without this, a
    # tracker running on a Phase A frame with a 1deg pitch error reports
    # centimeter-scale covariance while actually being off by 10-20cm at
    # a few meters' range, which both fails the floor-marker acceptance
    # test AND silently causes the chi-square gate to reject good
    # measurements (their real error no longer fits inside the too-tight
    # covariance the gate checks against).
    range_m = float(np.linalg.norm(point_floor_front[:2] - camera_floor_xy))
    extra_var = geometry.extrinsic_position_variance(range_m, floor_transform.sigma_rot_rad, floor_transform.sigma_trans_m)
    cov_xy = geometry.add_isotropic_variance(cov_xy, extra_var)

    return Measurement(position_xy=position_xy, cov_xy=cov_xy, src="depth")
