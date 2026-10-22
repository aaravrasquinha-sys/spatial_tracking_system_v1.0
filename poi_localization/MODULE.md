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

**Limits:** depth-noise constants are guesses until you fit them (H11) · single camera (H10) · re-identification (R1) is experimental and off by default (see below) ·
`M4Pipeline.total_events` grows without bound (`sts`'s soak monitor trims it from outside).

## Re-identification (R1, experimental, off by default)

`poi_localization/identity/` gives a person who leaves and comes back the same `P<n>` label. It rides on M4 without
touching tracking: with it off (the default) M4's output is bit-for-bit what it was; with it on, track positions are
still identical and each track gains an optional `person` field (`{id, label, state, conf, candidate}`) on the wire.

* **How:** the empty-room depth and colour from the M2 anchor bundle (`<cam>_reference_depth.npz`, `static_capture.npz`)
  remove the background and correct lighting; each person is measured (stature, limb lengths with error bars, clothing
  colour by height band, seen from front/back/side) only on clean frames; a returning track is compared against people who
  left, with "a stranger" as a real option, and commits `matched`, `new` or `ambiguous`.
* **Enable (Phase B):** in `site.json` set `"slots": {"identity_resolver": "multicue_v1"}` and
  `"localization": {"overrides": {"identity": {"enabled": true}}}`. `sts` fills the reference paths itself. Without the
  references it still runs, but every track gets a new id (`degraded` in the soak stats says why).
* **Status:** the likelihood constants in `poi_localization/identity/resolver.py` are placeholders until fitted on
  recordings of real people (Phase 0). Not yet: where/when cue, swap monitor, merging ambiguous ids, per-session calibration.
* **Tests:** `tests/unit/test_identity_{phase1,geometry,extract,reid}.py`, `tests/integration/test_identity_{chain,phase_b}.py`.
