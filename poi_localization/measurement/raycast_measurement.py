"""
Section 5: the fallback/complement measurement, strongest at range or
when depth is missing. Undistort -> cast a ray through the footpoint
pixel -> intersect the floor plane -> reject if it doesn't hit in front
of the camera, or (for the noisiest footpoint source) if the resulting
uncertainty is past a sanity ceiling.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from poi_localization import geometry
from poi_localization.config import RaycastMeasurementConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.types import Measurement
from poi_perception.contracts import Intrinsics

_EPS_SIN2 = 1e-4  # floor on sin^2(depression angle) to avoid a divide-by-zero at the horizon


def _pixel_noise_for_source(footpoint_source: str, cfg: RaycastMeasurementConfig) -> float:
    if footpoint_source == "ankles":
        return cfg.pixel_noise_ankles_px
    if footpoint_source == "ankle_single":
        return cfg.pixel_noise_ankle_single_px
    return cfg.pixel_noise_bbox_fallback_px  # "bbox_fallback", or anything unrecognized -> most conservative


def compute_raycast_measurement(
    footpoint_px: Tuple[float, float],
    footpoint_source: str,
    intr: Intrinsics,
    floor_transform: FloorFrameTransform,
    cfg: RaycastMeasurementConfig,
    distortion_coeffs: Optional[Tuple] = None,
) -> Optional[Measurement]:
    u, v = geometry.undistort_pixel(footpoint_px[0], footpoint_px[1], intr, distortion_coeffs)
    ray_dir_cam = geometry.pixel_ray_direction(u, v, intr)

    ray_origin_floor = floor_transform.camera_origin_floor()
    ray_dir_floor = floor_transform.apply_direction(ray_dir_cam)  # unit, R is a rotation

    hit = geometry.intersect_ray_with_floor(ray_origin_floor, ray_dir_floor)
    if hit is None:
        return None

    ground_distance = float(np.linalg.norm(hit[:2] - ray_origin_floor[:2]))
    if ground_distance > cfg.max_range_m:
        return None

    alpha = geometry.depression_angle(ray_dir_floor)
    sigma_px = _pixel_noise_for_source(footpoint_source, cfg)
    f = intr.fx
    h = ray_origin_floor[2]

    sin2_alpha = max(np.sin(alpha) ** 2, _EPS_SIN2)
    sigma_radial = (h / sin2_alpha) * (sigma_px / f)

    if footpoint_source == "bbox_fallback" and sigma_radial > cfg.raycast_sigma_ceiling_m:
        # Section 5: "a very unreliable ray-cast is worse than no measurement."
        return None

    range_3d = float(np.linalg.norm(hit - ray_origin_floor))
    sigma_tangential = range_3d * sigma_px / f

    if footpoint_source == "ankle_single":
        # A single visible ankle is biased by up to half a stride length
        # toward whichever foot is forward, not just noisier -- see
        # RaycastMeasurementConfig.ankle_single_extra_sigma_m. Combined
        # in quadrature with the pixel-noise sigma; this term dominates
        # at short-to-moderate range, where pixel noise alone is small
        # but the gait bias is not.
        extra = cfg.ankle_single_extra_sigma_m
        sigma_radial = float(np.hypot(sigma_radial, extra))
        sigma_tangential = float(np.hypot(sigma_tangential, extra))

    direction_xy = hit[:2] - ray_origin_floor[:2]
    cov_xy = geometry.radial_tangential_to_xy_cov(sigma_radial, sigma_tangential, direction_xy)

    # Extrinsic (calibration) uncertainty -- see the matching comment in
    # depth_measurement.py and geometry.extrinsic_position_variance's
    # docstring. Ray-cast is if anything MORE sensitive to extrinsic
    # pitch error than depth is: a grazing ray at long range moves a lot
    # on the floor for a small angular error, on top of its own already-
    # large pixel-noise-driven sigma_radial.
    extra_var = geometry.extrinsic_position_variance(ground_distance, floor_transform.sigma_rot_rad, floor_transform.sigma_trans_m)
    cov_xy = geometry.add_isotropic_variance(cov_xy, extra_var)

    return Measurement(position_xy=(float(hit[0]), float(hit[1])), cov_xy=cov_xy, src="raycast")
