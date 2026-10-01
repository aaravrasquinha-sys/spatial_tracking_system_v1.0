# poi_localization — M4 (3D localization and world tracking)

**May import** `poi_perception` (its `Frame`/`Detection2D` contracts) only. Never `pyslam`, `poi_present`, `sts`.

* **Consumes:** `Detection2D` + the synced depth frame; a **floor-frame transform** from a `FrameProvider`.
* **Produces:** `WorldTrack` records — Phase A `p_local/v_local`, Phase B `p/v` in the room frame — with `cov_xy`, `height_m`,
  `src` (`depth | raycast | fused | predicted`), `state` (`tentative | confirmed | coasting | lost`); JSONL under
  `data/logs/<site>/<cam>/world_tracks/`.
* **Two measurements, fused in one constant-velocity Kalman filter per person on the floor plane:**
  1. *depth* — median of valid depth inside the torso polygon, deprojected, plus a body-thickness offset (facing vs side,
     from shoulder keypoints), dropped to the floor; noise `sqrt((stereo-fx model)² + (bias·z)²)`.
  2. *raycast* — the footpoint pixel cast onto z = 0; noise grows at grazing angles; a single ankle carries an explicit gait-bias floor.
  Each applied with its own covariance **plus the calibration's extrinsic uncertainty** (`sigma` from M2), after a χ² gate.
* **Lifecycle:** tentative → confirmed (3 hits) → coasting (≤ 1 s on prediction) → lost → ended; world-frame re-association
  catches some 2D ID swaps. Gates: walkable grid (also removes mirror reflections behind walls), height 0.5–2.2 m.
* **Phase A vs B** differ only in where the transform comes from (`frames/local_floor_frame.py` IMU + floor RANSAC, or
  `frames/room_frame.py` reading `calibration.<cam>.json`). `phase` in `M4Config` (generated from `site.json`).
* **Entry points:** `scripts/run_m4.py`, `scripts/floor_marker_capture.py`, `scripts/debug_visualize_world.py`,
  `scripts/demo_synthetic_m4.py`; in `sts run` M4 is wired through M3's `on_detections` and publishes via `on_tracks`.
* **Tests:** `tests/unit/test_{depth_measurement,raycast_measurement,body_thickness_offset,kalman_track,track_manager,frames,local_floor_frame,geometry,extrinsic_uncertainty,consistency_nees,m4_pipeline_e2e}.py`.
  `consistency_nees` is a **covariance-honesty** check (NEES under a physically-realistic D435i error model), not just accuracy.
* **Eval:** `poi_localization/eval/floor_markers/` (**the acceptance test**: 10+ taped crosses, median error ≤ 10 cm within 4 m,
  ≤ 20 cm beyond), `eval/scenarios/`, `grade.py`, `replay_diff.py`.

**Limits:** depth-noise constants are guesses until you fit them (H11) · single camera (H10) · no re-identification (R1) ·
`M4Pipeline.total_events` grows without bound (`sts`'s soak monitor trims it from outside).
