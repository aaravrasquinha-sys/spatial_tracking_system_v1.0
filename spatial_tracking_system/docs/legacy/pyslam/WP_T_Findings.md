# WP-T0/T1/T2/T3/T4 Findings: trajectory export, LOST-crash fix, mirrored fixtures

Scope: (1) accurate bird's-eye + height trajectory output, (2) keyframe-
by-keyframe pose export, both prerequisite-gated on fixing what analysis
of the attached hardware logs + full codebase read turned up first. Orin
Nano (item 3) explicitly deferred, per instruction.

All changes validated against: full selftest (22/22), G2 (4/4), G3
(6/6), G4 (5/5), G5 (5/5), new G-TRAJ (3/3) -- all on the native backend
(GTSAM not installed in this environment; see "Not validated" below).

---

## WP-T0: the corridor_v2 / GTSAM crash

**Root cause.** `pipeline.py`'s LOST-recovery path (`force_new_keyframe`)
created a new graph node with `last_keyframe_link_T = None`, so no
odometry `Link` was ever added between it and the previous keyframe.
That node -- and every node chained after it until the next successful
link or loop closure -- had zero or a disconnected set of factors. GTSAM's
`LevenbergMarquardtOptimizer` needs a full elimination ordering over
every variable; a factorless node makes that impossible, producing
exactly the "inconsistent arguments" `RuntimeError` seen on the
`corridor_v2` hardware run in the attached log.

**Fix.** `pipeline.py::_run_loop` now adds a weak identity `Link(kind=
"bridge")` across every LOST gap (`Config.bridge_link_info_scale`,
default `1e-3` -- "assume no motion happened, trust this hardly at
all"). A real loop closure across the same gap, once verified, still
dominates. Both graph backends treat `"bridge"` like `"odom"`
(un-robustified; it's already given a near-zero weight at construction
instead).

**Validated (native backend):** re-ran `corridor_v2` end-to-end (the
exact scenario that crashed on hardware) -- 15 LOST events fired, 15
bridge links added, the graph stayed **one connected component** the
entire run, and `pipeline.finalize()` (the same full-graph `optimize()`
GTSAM's LM step would need to succeed on) returned finite poses for all
179 nodes, none missing. New gate `test_g_traj.py::
check_no_orphan_node_after_repeated_lost` forces this condition
deterministically (impossible `odom_min_inliers` threshold -> LOST every
few frames) and asserts the same three invariants.

**Not validated:** the actual GTSAM code path itself -- GTSAM isn't
importable in this environment. The fix removes the orphan-node
precondition the crash needed, and the native-backend re-run confirms
the graph-connectivity invariant holds, but run this on the target
machine (GTSAM installed) before considering it closed.

**Also fixed while here (GTSAM/native backend parity, same class of
gap):** `"proximity"` links (WP-P4) previously got no robust-kernel
treatment on EITHER backend (native: only `kind=="loop"` was
Huber/DCS'd; GTSAM: only `kind=="loop"` was Huber'd, and DCS wasn't
implemented on GTSAM at all despite `cfg.loop_robust_kernel="dcs"`
existing). Both backends now Huber/DCS proximity links the same as loop
links, and DCS is now implemented on the GTSAM backend
(`gtsam.noiseModel.mEstimator.DCS`). G4 (5/5) re-confirms DCS robustness
bounds hold with this change; the proximity-recall check (G4.5) now
visibly exercises the robustified path.

---

## Mirrored ground-truth fixtures (found via a metrics anomaly, not the gate)

**Root cause.** `tests/synth/world.py::poses_from_path` computed
`right = fwd x up` and stored it in the column documented as **left**
(body convention: x-forward, y-**left**, z-up). `right != left`, so
every pose built through this function was a reflection: det(R) = -1.
Affected `square6dof` and `corridor_v2` ground truth (confirmed by
det(R) audit); `room_orbit` builds its rotation directly via
`lie.so3_exp` and was never affected.

**How it was found.** Not by any existing gate -- by re-deriving
`square6dof`'s reported metrics: anchored ATE (0.94cm) was *smaller*
than Umeyama ATE (4.59cm), which is impossible for two honest proper-
rigid alignments (Umeyama is the least-squares-optimal proper
alignment, anchored has strictly less freedom, so anchored >= Umeyama
whenever both are comparing real rotations). `mirror_check` existed
already but was one-sided (only flagged ratio >> 1), so it never caught
this.

**Fix.**
- `poses_from_path`: compute `left = up x fwd` directly instead of
  negating a "right" vector into the wrong column.
- `mirror_check` (`pyslam/tools/metrics.py`): now two-sided -- flags
  either `anchored/umeyama > threshold` or its inverse.
- `validate_scenario` (`tests/synth/world.py`): new `det(R) ≈ +1` check,
  independent of every other check in that gate (none of them would
  have caught a reflection -- a mirrored path can still stay inside
  free space, keep valid depth, and have smooth rotation steps).

**Re-baselined `square6dof` (seed=1, native backend):**

| Metric | Before (mirrored) | After (fixed) |
|---|---|---|
| ATE (Umeyama), odom / graph | 4.63 / 4.59 cm | 0.63 / 0.47 cm |
| ATE (anchored), odom / graph | 1.72 / 0.94 cm | 1.46 / 0.74 cm |
| mirror_suspected | False (one-sided check missed it) | False (correctly) |

Anchored is now, correctly, >= Umeyama. **Any prior accuracy comparison
run on `square6dof` or `corridor_v2` (including the f2m-vs-f2f numbers
in WP-B1's findings) was measured against a reflected ground truth and
should be treated as unreliable until re-run.** `gravity_init_square6dof`
is built on `square6dof` and should also be re-checked, though its own
end-to-end gravity-prior correction (G5) still passed (0.42deg, well
under the 2deg bound) -- the P5.2 tilt-only correction is not sensitive
to this class of error the same way a position ATE is, since gravity
alone doesn't observe yaw/reflection about the vertical axis the way a
full pose comparison does; re-run all synthetic-fixture-derived
accuracy claims before trusting them again regardless.

---

## `static_60s` crash

**Root cause.** `evaluate.py::ate_rmse` -> `umeyama_align` asserts
`src.shape[0] >= 3`. `static_60s` produces exactly 1 keyframe by design
(never crosses the keyframe-creation motion threshold), so the very
first synthetic-gate run of this fixture in the attached log crashed
with an unhandled `AssertionError`.

**Fix.** `ate_rmse` returns `None` below 3 points instead of raising.
`run_synth.py` reports "not enough keyframes to score" and writes
`ate_odom_cm: null` / `ate_graph_cm: null` to `summary.json` rather than
crashing. Confirmed: `static_60s` now runs to completion, 1 keyframe,
`n_frames_skipped_in_final_reconstruction: 0`.

---

## WP-T1: per-frame pose bookkeeping + closing optimisation

`PipelineResult.frame_records`: one entry per **frame** (not just
keyframe) -- `(frame_id, t, ref_kf_id, T_ref_frame, status, session_id,
is_keyframe)`. `T_ref_frame` is exactly `OdomResult.T_rel`, captured
against whichever keyframe was the tracking reference *before* that
frame's `odometry.update()` call (so no new tracking math -- just
bookkeeping of what already gets computed). A frame's FINAL pose is
reconstructed at export time as `final_pose(ref_kf_id) @ T_ref_frame`,
which is what lets an ordinary (non-keyframe) frame inherit a loop
closure correction made to its reference keyframe *after* the frame
itself was originally processed.

`Pipeline.finalize(result)`: one full-graph `optimize(fixed=[first_node])`
at shutdown -- deliberately NOT the WM-only, LTM-anchored optimisation
`_accept_link` uses during the run for bounded online latency. This is
what lets a loop closure correct drift in a part of the trajectory
already evicted to LTM by the time it fires, which the online path
cannot do by design. Idempotent; safe to call on a partial
(`KeyboardInterrupt`) result.

---

## WP-T2: exporters

New `pyslam/tools/trajectory_export.py`, one shared code path for
`run_slam.py` / `run_bag.py` / `run_synth.py`:

- `trajectory_odom.tum` -- raw odometry, per frame, zero corrections.
- `trajectory_final_frames.tum` / `trajectory_final_kf.tum` -- final
  (post-`finalize()`), per-frame and per-keyframe.
- `keyframes.csv` / `keyframes.json` -- per-keyframe identity, pose in
  both W_cam0 and W_grav, path (cumulative distance), quality
  (odometry-vs-final correction magnitude), graph (loop/proximity
  partners), flags (post-LOST, was-in-LTM). Schema matches what was
  agreed in conversation.
- `graph.g2o` -- reopenable independently of this codebase.
- `trajectory_report.json` -- path length, start/end gap, height range,
  loop/proximity/bridge/LOST counts, gravity-alignment status.

TUM format puts the quaternion `w` **last**; this codebase's own frozen
convention (`pyslam.core.lie`) is `[w,x,y,z]` **first**. Every writer
reorders explicitly at the point of writing. New gate
`test_g_traj.py::check_tum_round_trip` writes 20 random poses, reads
them back, and asserts sub-micron/sub-1e-6-rad round-trip error --
specifically to catch a silent w-first/w-last regression here, since
that bug would produce a well-formed-looking file with silently wrong
orientations.

---

## WP-T3: gravity-aligned frame (W_grav)

New `pyslam/core/gravity_frame.py`. Separate from (and does not
require) `cfg.gravity_prior_enabled` -- only needs one quasi-static
window near the start of the run (`PipelineResult.startup_imu_window`,
collected unconditionally for `cfg.gravity_align_hold_s` seconds,
default 1.5s). Origin = first keyframe's camera centre (same as
W_cam0); z = up; x = the first keyframe's own forward direction
projected onto the horizontal plane.

**Real bug found by this module's own oracle test, fixed before
shipping:** `gravity_direction_body()` (existing, `pyslam/imu/gravity.py`)
returns the direction gravity **points** (down) per its own explicit
docstring. An early version of `estimate_gravity_alignment` used that
value directly as "up" without negating it -- a 180-degree error. Caught
immediately by `test_g_traj.py::check_gravity_alignment_oracle`
(`err=2.0000`, the exact fingerprint of an inverted unit vector).
Fixed by negating (`up_body = -g_body`) before rotating into camera
convention; re-verified (`err=0.0002`, world-z residual `2e-16`,
`det(R)=1.000000000`). This bug never surfaced through the *existing*
P5.2 tilt-prior gate (G5) because that code path only ever compares two
quantities computed with the *same* (consistent, if backwards-labelled)
convention against each other -- it never needed the value to mean
"up" specifically. This module does, which is exactly why it needed its
own oracle rather than trusting G5's.

**Real hardware extrinsic, not identity.** `RealSenseSource` now
queries the actual accel-to-colour rotation from the device
(`get_extrinsics_to`) instead of defaulting to identity -- the gap
flagged in WP-P5.2 as "known wrong for this project's own D435i rig,"
now closed for real hardware. Falls back to identity with a logged
warning if the query fails (e.g. IMU disabled, older firmware).

**Confirmed on synthetic data:** `square6dof`/`static_60s` correctly
report `gravity_aligned: false` (no static hold at t=0 by design in
either fixture) rather than silently presenting an unaligned frame as
if it were gravity-true. `static_60s`'s own synthetic IMU noise sits
right at the quasi-static gyro threshold (0.083 vs 0.08 rad/s) -- this
is the existing conservative `is_quasi_static` threshold interacting
with synthetic noise, not a defect; real hardware noise floors are
comfortably under it.

---

## WP-T4: plots

New `pyslam/tools/plots.py`: `trajectory_bird_eye.png` (top-down X-Y,
odometry vs final, loop/proximity links drawn, start/end markers),
`trajectory_height.png` (height vs cumulative distance and vs time).
Both degrade gracefully (skip + log, never raise) below 2 keyframes.

---

## Files changed / added

```
NEW   pyslam/core/gravity_frame.py
NEW   pyslam/tools/trajectory_export.py
NEW   pyslam/tools/plots.py
NEW   tests/gates/test_g_traj.py
MOD   pyslam/pipeline.py          (bridge link, frame_records, finalize())
MOD   pyslam/core/config.py       (bridge_link_info_scale, gravity_align_hold_s)
MOD   pyslam/graph/backend_native.py  (proximity robust-kernel parity)
MOD   pyslam/graph/backend_gtsam.py   (proximity robust-kernel parity, DCS support)
MOD   pyslam/graph/posegraph.py   (thread robust_kernel/dcs_xi to GTSAM backend)
MOD   pyslam/sensors/realsense.py (real R_body_cam from device extrinsics)
MOD   pyslam/tools/evaluate.py    (ate_rmse guards <3 points instead of asserting)
MOD   pyslam/tools/metrics.py     (mirror_check two-sided)
MOD   tests/synth/world.py        (poses_from_path handedness fix, det(R) gate check)
MOD   pyslam/selftest.py          (wire in G-TRAJ)
MOD   run_slam.py / run_bag.py / run_synth.py  (finalize() + export wiring)
```

## What's next (per the plan agreed before this work started)

- Run this on the actual target machine (GTSAM installed) -- selftest,
  G-TRAJ, then `corridor_v2` on the GTSAM backend specifically, since
  that's the one path this environment cannot execute.
- WP-T5: record one `.bag` per test case (static, straight-line,
  taped square, floor-to-table height, back-and-forth half-room),
  replay-only from then on. Cross-check one recording against upstream
  RTAB-Map via `evo`.
- Re-run WP-B1's f2m-vs-f2f comparison now that `square6dof`'s ground
  truth is no longer reflected.
- Orin Nano (item 3): still deferred. LTM will evict more aggressively
  there given the slower CPU -- worth stress-testing WP-T1's
  `finalize()` path (full-graph optimise including LTM-evicted nodes)
  under that condition specifically once T5 is done here.
