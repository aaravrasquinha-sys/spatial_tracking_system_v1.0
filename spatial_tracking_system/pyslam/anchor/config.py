"""
All Module 2 thresholds in one place, so a gate result is always
traceable to a named number (and the report can print the exact config
it ran with). Defaults follow the planning doc's acceptance table; each
non-obvious number has a one-line reason.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple


@dataclass
class AnchorConfig:
    # ---- Stage 0: map preparation / room frame ----
    map_voxel_m: float = 0.02                 # matches M1's VoxelHashFuser default
    sor_k: int = 16                           # statistical outlier removal neighbours
    sor_std: float = 2.5                      # ... and std-dev multiplier
    up_prior_map: Tuple[float, float, float] = (0.0, -1.0, 0.0)  # W_cam0: y is DOWN, so up = -y
    up_prior_tol_deg: float = 60.0            # M1 start pose is hand-held; prior is only a sign/rough guide
    floor_ransac_dist_m: float = 0.02
    floor_ransac_iters: int = 600
    floor_min_inliers: int = 1500             # on the 5 cm RANSAC cloud
    floor_ransac_voxel_m: float = 0.05
    floor_lowest_frac: float = 0.3            # candidate must have >= this x the biggest horizontal plane's inliers
    floor_flat_rms_m: float = 0.015           # plan: inlier RMS <= 1.5 cm
    floor_quadrant_tilt_deg: float = 1.0      # plan: quadrant normals agree within 1 deg
    floor_quadrant_height_m: float = 0.02     # ... heights within 2 cm
    wall_min_h_m: float = 0.3                 # points used for wall-direction histogram
    wall_max_h_m: float = 2.0
    wall_normal_vertical_tol: float = 0.25    # |n . up| below this = wall-like
    wall_hist_min_peak_frac: float = 0.25     # peak must hold this share of weight, else fall back to start heading
    n_walls: int = 6
    wall_ransac_dist_m: float = 0.03
    wall_min_inliers: int = 600
    # accuracy floor the map itself imposes (cannot localize the camera
    # relative to the room better than the room is known)
    map_trans_floor_m: float = 0.010
    map_rot_floor_deg: float = 0.15

    # ---- walkable grid ----
    walk_res_m: float = 0.05
    walk_floor_tol_m: float = 0.04            # |z| below this counts as floor support
    walk_band_lo_m: float = 0.10              # plan: nothing between 0.1 ...
    walk_band_hi_m: float = 1.80              # ... and 1.8 m above the floor
    walk_sit_max_m: float = 0.90              # low surfaces up to here are "sittable" (matches layers.py)
    walk_min_pts: int = 2                     # per-cell point count to trust a cell (kills floaters)
    walk_dilate_cells: int = 1                # generous gate: avoid rejecting real footpoints at obstacle edges

    # ---- Stage 1: static capture ----
    capture_frames: int = 150
    capture_warmup_s: float = 60.0            # thermal settle of D435i depth before the geometric capture
    capture_discard_s: float = 1.5            # auto-exposure settle
    capture_n_subsets: int = 5                # interleaved sub-medians for repeatability
    min_valid_frac: float = 0.8               # pixel valid in >= this share of frames
    max_mad_m: float = 0.02                   # temporal MAD gate (metres) at 1 m; grows with z^2 below
    edge_jump_m: float = 0.05                 # 3x3 max-min depth jump above which a pixel is an edge/flying pixel
    edge_jump_per_z: float = 0.02
    range_min_m: float = 0.3
    range_max_m: float = 4.0                  # D435i error ~ z^2; beyond ~4 m it hurts more than it helps
    query_stride: int = 2
    # depth noise model shape (same as M4): sigma_z = z^2 * sigma_d / (f * B), times an empirical factor
    sigma_disparity_px: float = 0.1
    noise_multiplier: float = 2.0
    map_sigma_m: float = 0.005
    imu_max_gyro_std_rad_s: float = 0.02
    imu_max_accel_std_mps2: float = 0.15

    # ---- Stage 2: physics prior ----
    floor_fit_lowest_percentile: float = 70.0  # candidate floor = lowest 30% along IMU-down
    floor_fit_ransac_iters: int = 500
    floor_fit_dist_m: float = 0.02
    floor_fit_min_inliers: int = 400
    floor_fit_min_extent_m: float = 0.10       # minor in-plane std of floor inliers (rejects a slice through a wall)
    require_imu: bool = True

    # ---- Stage 3: global search ----
    csm_res_m: float = 0.05
    csm_yaw_step_deg: float = 0.5
    csm_range_m: float = 4.0                  # query slice radius (multiple of csm_res)
    csm_sigma_m: float = 0.06                 # likelihood-field blur
    slice_lo_m: float = 0.3                   # structure slice above the floor
    slice_hi_m: float = 2.0
    csm_min_map_pts_per_cell: int = 2
    csm_peaks_per_yaw: int = 6
    csm_top_k: int = 4                        # candidates handed to ICP
    csm_distinct_xy_m: float = 0.30
    csm_distinct_yaw_deg: float = 8.0
    ambiguity_ratio_min: float = 1.15         # best/second CSM score
    ambiguity_icp_fitness_ratio: float = 0.85 # second candidate's ICP fitness / best must be below this to count as "distinct enough"
    colour_tiebreak_margin: float = 0.25      # relative colour-error advantage needed to break a tie

    # ---- Stage 4: ICP ----
    icp_scales_m: Tuple[float, ...] = (0.12, 0.05, 0.025)
    icp_iters: Tuple[int, ...] = (25, 25, 30)
    icp_trim_frac: float = 0.05               # at the final scale only: drop the worst 5% of residuals
    icp_min_delta: float = 1e-6
    icp_crop_margin_m: float = 0.5
    icp_normal_k: int = 20
    fitness_dist_m: float = 0.04
    coverage_dist_m: float = 0.15
    icp_cov_inflation: float = 4.0            # depth noise is spatially correlated; validated by subset spread

    # ---- Stage 6: acceptance gates ----
    gate_icp_rmse_m: float = 0.015
    gate_fitness_min: float = 0.60
    gate_coverage_min: float = 0.70
    gate_imu_tilt_deg: float = 0.5
    gate_floor_height_m: float = 0.015
    gate_floor_tilt_deg: float = 0.5
    gate_min_observability: float = 0.001     # normalised normal-matrix min eigenvalue; a single flat wall gives ~0
    gate_sigma_trans_m: float = 0.03
    gate_sigma_rot_deg: float = 0.5
    gate_repeat_trans_m: float = 0.010
    gate_repeat_rot_deg: float = 0.20
    gate_depth_resid_frac: float = 0.80
    gate_depth_resid_tol_m: float = 0.03
    gate_marker_err_m: float = 0.03

    # ---- watchdog ----
    wd_tilt_deg: float = 0.2
    wd_depth_move_m: float = 0.05
    wd_depth_frac: float = 0.15               # share of stable pixels moved before "suspect" (people occlude some)
    wd_consecutive: int = 3
    wd_pose_trans_m: float = 0.01
    wd_pose_rot_deg: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


def apply_overrides(cfg: AnchorConfig, items) -> AnchorConfig:
    """`--set key=value` overrides (values parsed to the field's own type; tuples as comma lists)."""
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    kw = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"override {it!r} is not KEY=VALUE")
        k, v = it.split("=", 1)
        if k not in fields:
            raise ValueError(f"unknown AnchorConfig field {k!r}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            kw[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            kw[k] = int(v)
        elif isinstance(cur, float):
            kw[k] = float(v)
        elif isinstance(cur, tuple):
            kw[k] = tuple(type(cur[0])(x) for x in v.split(","))
        else:
            kw[k] = v
    return dataclasses.replace(cfg, **kw)
