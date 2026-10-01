# Accuracy fixes — change log

## Module 5 added (this session)

Added `poi_present` (Module 5): a live WebSocket publisher plus a 2D
dashboard and a WebXR viewer, reading `poi_localization`'s already-
computed `WorldTrack` output. Zero changes to `poi_perception` or
`poi_localization` — `M4Pipeline` already accepted an `on_tracks`
callback (`runtime/m4_pipeline.py`) that nothing called; M5's
`scripts/run_present.py --source live` is the first thing to pass one.
See `PRESENT.md` for the full writeup and `SCHEMA.md` for the wire
contract. Test count: 135 → 168 (33 new, all hardware-free).

All changes verified with `pytest` (110 → 135 tests, all passing) and
cross-checked with a physically-realistic D435i simulation harness
before/after each fix. See README.md's "Accuracy fixes" section for the
narrative summary; this is the file-by-file detail.

## Modules 3+4 accuracy fixes (prior session)

## poi_perception/tracking/footpoint.py
Fixed a self-intersecting ("bowtie") torso polygon in the "full" and
"partial" branches of `compute_torso_polygon`: quad corners were
ordered by anatomical keypoint label (`left_shoulder`/`right_shoulder`)
rather than by actual pixel x-position. For a camera-facing person (the
common case — COCO's "left" is the person's own left, which is on the
image's right when facing the camera), this produced a self-intersecting
quad that `cv2.fillPoly` draws as two small opposing triangles instead
of one trapezoid — roughly halving the sampled depth area exactly in
the partial-occlusion case where samples are already scarce. Fixed by
sorting keypoints by x before building the quad; falls back correctly
when only one keypoint per edge is available. New tests in
`tests/test_footpoint_polygon_winding.py`.

## poi_localization/geometry.py
Added `extrinsic_position_variance(range_m, sigma_rot_rad, sigma_trans_m)`
and `add_isotropic_variance(cov_xy, extra_variance)` — the core of the
extrinsic-uncertainty fix. See the function's docstring for the (documented,
approximate, isotropic) model and its limitations.

## poi_localization/frames/types.py
`FloorFrameTransform` gained `sigma_rot_rad`/`sigma_trans_m` fields
(default 0.0, preserving old behavior everywhere they're not explicitly
set). `from_height_and_yaw` accepts them as optional kwargs.

## poi_localization/frames/local_floor_frame.py
Phase A's `estimate_floor_frame_from_points` now computes and attaches
`sigma_rot_rad`/`sigma_trans_m` to the returned transform, from the
fit's own diagnostics (gravity-alignment disagreement, inlier RMS)
combined with a documented conservative floor (Phase A has no
independent yaw check at all).

## poi_localization/frames/room_frame.py
Phase B's `load_room_frame` now reads the calibration file's own
`sigma` block (previously logged and discarded) and attaches it to the
transform, with a warning + conservative floor if the block is missing.

## poi_localization/measurement/depth_measurement.py
- Both measurement models now add `extrinsic_position_variance`'s
  contribution to `cov_xy`.
- New `estimate_body_thickness_offset_m`: orientation-dependent body-
  thickness offset (facing-on vs. side-profile) from shoulder keypoints,
  replacing one fixed constant. `compute_depth_measurement` takes
  optional `shoulder_l_px`/`shoulder_r_px`.
- Depth noise model recalibrated: uses a configurable `depth_stereo_fx_px`
  (the IR-pair focal length depth is actually computed from, ~390px at
  640x480) instead of the color stream's fx (~607px), which understated
  sigma_z by ~2.5x; added a proportional bias term
  (`depth_bias_frac`), combined in quadrature.

## poi_localization/measurement/raycast_measurement.py
Ray-cast measurements also add `extrinsic_position_variance`. New
`ankle_single_extra_sigma_m`: a single visible ankle now carries an
explicit range-independent uncertainty floor representing gait-sway
bias (up to ~0.35m at normal walking pace), not just pixel noise.

## poi_localization/tracking/height_estimator.py
`estimate_instantaneous_height` now applies a per-keypoint head-offset
correction (`HeightConfig.top_keypoint_head_offset_m`): nose/eye/ear
keypoints all sit below the actual crown of the head by ~9-12cm, which
was previously a systematic (not averageable) under-estimate on every
frame of every track.

## poi_localization/config.py
New/changed config fields: `DepthMeasurementConfig.max_range_m` raised
3.0→6.0m; `depth_stereo_fx_px`, `depth_bias_frac`,
`body_thickness_side_m` added. `RaycastMeasurementConfig.ankle_single_extra_sigma_m`
added. `HeightConfig.top_keypoint_head_offset_m` added.

## poi_localization/runtime/m4_pipeline.py
Wires shoulder keypoints (when confidently detected) from `Detection2D`
into `compute_depth_measurement` for the orientation-dependent thickness
correction.

## poi_perception/config.py
`CameraConfig` gained RealSense tuning fields: `enable_global_time`,
`visual_preset`, `laser_power`, `enable_post_processing`, and the
spatial/temporal/hole-filling filter parameters.

## poi_perception/capture/realsense_source.py
- Applies `global_time_enabled` on every sensor at startup (required by
  the full system plan's Section 3 timestamp contract; was previously
  never set).
- Applies the configured depth visual preset (default High Accuracy)
  and laser power.
- Builds and applies a spatial → temporal → hole-filling post-processing
  filter chain (in disparity space, per Intel's recommended ordering) to
  every depth frame before it reaches M3/M4. None of this existed
  before. All best-effort with warnings, never crashes, so
  hardware/firmware without a given option still runs.
- New `_build_post_processing_filters` is a pure function of
  `(CameraConfig, rs)`, tested hardware-free in
  `tests/test_realsense_post_processing.py` via a fake `rs` module.

## poi_localization/eval/consistency.py (new)
A covariance-honesty (NEES) check, run against the real measurement +
tracking code with a synthetic, physically-realistic sensor error model
— usable standalone (`python -m poi_localization.eval.consistency`) to
tune a real room's numbers, and wrapped in
`tests/test_consistency_nees.py` so an overconfidence regression (e.g.
someone reverting the extrinsic-uncertainty wiring) fails CI
automatically, without hardware.

## New test files
- `tests/test_footpoint_polygon_winding.py`
- `tests/test_extrinsic_uncertainty.py`
- `tests/test_body_thickness_offset.py`
- `tests/test_realsense_post_processing.py`
- `tests/test_consistency_nees.py`

## Modified test file
- `tests/test_raycast_measurement.py`: updated one assertion that
  assumed a strict ankles < ankle_single < bbox_fallback covariance
  ordering, which is no longer universally true now that ankle_single
  carries an explicit gait-bias floor (it can legitimately exceed
  bbox_fallback's purely pixel-noise-driven sigma at short range) — see
  the updated test's comments for the reasoning, plus a new test
  documenting the near/far crossover behavior directly.

## Not done in this session (still open, from the original review)
- Per-torso point-cloud/cylinder fit to replace median-centroid depth
  (removes the need to guess body thickness at all).
- Per-track bias state augmentation in the Kalman filter.
- Stacking depth+raycast into one correlated update instead of two
  independent sequential ones (they currently share extrinsics, so
  applying them as independent updates shrinks P by ~sqrt(2) too much).
- Posture (standing/sitting/lying) detection to gate raycast validity.
- IMM / adaptive process noise.
- Temporal keypoint smoothing before footpoint computation.
- AprilTag-grid ground truth for floor-marker acceptance testing.
