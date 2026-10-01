# Phase 0 Evaluation

This is a template for you to fill in after running on the D435i. It
tells you what to run and what to report back; I'll use it to decide
what Phase 1 should actually prioritize (odometry accuracy vs.
retrieval vs. performance vs. something we haven't hit yet).

Fill in the `>>> ` lines. Paste terminal output where asked rather than
summarizing it — exact numbers and exact errors matter more than
impressions.

---

## 0. Environment check

Run:
```
python3 -m pyslam.tools.env_probe --realsense
```

- OpenCV version reported: `>>> `
- GTSAM available + functional check result: `>>> `
- RealSense: name / serial / firmware / USB type: `>>> `
- `intrinsics_match_expected` true or false, and if false, what
  mismatched: `>>> `
- IMU streams reported available: `>>> `

## 1. Selftest (before touching the camera, and again after)

```
python3 -m pyslam.selftest
```
- Result (should be `6/6 passed`): `>>> `
- If anything failed, paste the full output: `>>> `

## 2. Static drift test

Record 60s with the camera stationary on a desk:
```
python3 run_slam.py --realsense --imu --record data/bags/desk_static.bag
```
Then:
```
python3 run_bag.py data/bags/desk_static.bag
```
Needed:
- `summary.json`: n_frames, n_keyframes: `>>> `
- Does the reported trajectory (check `telemetry.json` or just watch
  the live log) stay near-stationary, or does it visibly drift? Rough
  drift distance over the 60s if you can eyeball it from the map/PLY:
  `>>> `
- Any `LOST` events in the log during a static scene (there shouldn't
  be any)? `>>> `

## 3. Live rate / performance

While running `run_slam.py` live, note:
- Roughly how many seconds the whole run took vs. how many seconds of
  real time it covers (e.g. "captured 30s of footage, pipeline took
  90s to process") — this environment measured ~300ms/frame
  single-threaded; I need your machine's real number to know whether
  Phase 1's vectorization work is urgent or can wait: `>>> `
- Peak RAM usage if easy to check (`top`/`htop` while it runs): `>>> `

## 4. Room loop (the flagship scenario)

Walk a small loop in a room (a few meters, similar shape to the
validated synthetic square-loop scenario if possible — out, turn,
return to the start) and back to your starting position/orientation:
```
python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
```
Then replay:
```
python3 run_bag.py data/bags/room_loop.bag
```
Needed:
- `summary.json`: n_frames, n_keyframes, n_loop_closures: `>>> `
- Did at least one `LOOP CLOSURE:` line appear in the log? Paste
  them: `>>> `
- Open `map.ply` in any viewer (MeshLab, CloudCompare, even
  `python -c "import open3d"` if you have it) — does the room look
  roughly rectilinear/sane, or is it obviously torn/duplicated? `>>> `
- Does the visual return-to-start gap in the map look closed (walls
  from the start and end of the loop roughly coincide) or still
  offset? `>>> `

## 5. Corridor / longer loop (known-hard case, informational only)

If you have access to a longer corridor-like loop, try the same as #4
over a longer path (10-30m, out and back around a loop rather than a
tight square). This is **not expected to close the loop** per the
Runbook's known limitations — I want data on *how* it fails, not
whether it fails:
- n_keyframes, n_loop_closures (expected: 0): `>>> `
- Roughly how far off does the end of the trajectory look from the
  true start position, if you can tell (a rough distance estimate,
  or "didn't look obviously wrong / did look obviously wrong"):
  `>>> `

## 6. Hard case (blank wall / fast turn / low light)

```
python3 run_slam.py --realsense --imu --record data/bags/hard_case.bag
```
- Did a `LOST` event appear in the log, and did it roughly correspond
  to when you pointed at the blank wall / turned fast / lost light?
  `>>> `
- Did tracking visibly recover afterward (new keyframes resuming, no
  crash)? `>>> `

## 7. Anything that crashed or looked wrong

Paste the full traceback and the command that produced it, or a
one-line description + which run directory (`runs/run_...`) it
happened in so I can ask for the bundle if needed: `>>> `

---

## What I already know going in (so you don't need to re-report these)

- Performance is single-threaded and not real-time in this dev
  environment (~300ms/frame here); I need your number from #3 to
  calibrate.
- The corridor-scale loop-closure scenario is validated as *not yet
  working* on synthetic data — see below for why, so #5's result
  shouldn't be surprising.
- IMU is recorded but not fused into odometry yet (Phase 5).
- No threading, no LTM yet (by design, this phase).

## Why the corridor scenario doesn't close the loop yet (for context)

Diagnosed on synthetic data during Phase 0 development, in order:
1. A fixed 1000-word k-means vocabulary lost the discriminative signal
   entirely at this WM scale (word overlap was ~170-190 words between
   *any* two keyframes regardless of true similarity) — replaced
   retrieval scoring with direct batched descriptor matching, which
   discriminates correctly (see `vpr/raw_match.py`).
2. The Bayes filter's belief-decay model punishes any candidate not
   reinforced every single frame; over a ~70-keyframe corridor loop, a
   true match's belief decays by ~0.85^70 before the one frame of real
   evidence arrives at the very end, which isn't enough to win against
   a nearer-term (and correctly-rejected-by-the-verifier) false
   candidate. This is a natural fit for RTAB-Map's real
   graph-neighbor-aware transition model, deferred to Phase 2 (memory
   management) / Phase 4 (proximity + smarter transitions) rather than
   patched with a global decay-rate tweak in Phase 0.
This is why the square-loop scenario (short, ~30 keyframes, true match
reinforced across the last ~10) is Phase 0's validated gate instead —
it exercises the same mechanism honestly without depending on
infrastructure that's explicitly scoped for later phases.



# HOW TO TOGGLE PHASES:
#
# (WP-K2: --config-override now REALLY EXISTS on run_synth.py, run_bag.py and
#  run_slam.py -- it was documented here but never implemented. Values are validated;
#  the line numbers quoted under METHOD 1 below are stale, search for the field name.)

# METHOD 1: Edit config.py directly
cd ~/pyslam_repo
nano pyslam/core/config.py

# Find these lines and change True/False:

Line 34:   odometry_backend: str = "f2f"      # Change to "f2m" for Phase 1B
Line 100:  ltm_enabled: bool = True           # P2 (leave True)
Line 126:  retrieval_backend: str = "raw"     # P3 (change to "bow_incremental")
Line 157:  use_hessian_info: bool = False     # P4 (change to True)
Line 255:  proximity_enabled: bool = False    # P4 (change to True)
Line 267:  gravity_prior_enabled: bool = False # P5 (change to True)


# METHOD 2: Use command-line override (EASIER)
# Create a modified config at runtime without editing the file

# Default (Phase 0 only):
python run_synth.py --scenario square6dof

# Enable Phase 1B (F2M odometry):
python run_synth.py --scenario square6dof \
  --config-override odometry_backend=f2m

# Enable Phase 3 (incremental vocab):
python run_synth.py --scenario square6dof \
  --config-override retrieval_backend=bow_incremental

# Enable Phase 4 (Hessian info + proximity):
python run_synth.py --scenario square6dof \
  --config-override use_hessian_info=True \
  --config-override proximity_enabled=True

# Enable Phase 5 (gravity prior):
python run_synth.py --scenario square6dof \
  --config-override gravity_prior_enabled=True

# Enable EVERYTHING:
python run_synth.py --scenario square6dof \
  --config-override odometry_backend=f2m \
  --config-override retrieval_backend=bow_incremental \
  --config-override use_hessian_info=True \
  --config-override proximity_enabled=True \
  --config-override gravity_prior_enabled=True


WHAT EACH TOGGLE DOES:

✅ odometry_backend="f2f"     → Phase 0 odometry (simple, fast)
🔄 odometry_backend="f2m"     → Phase 1B odometry (3D map matching, slower, needs tuning)

✅ ltm_enabled=True           → Phase 2 (memory management, ON by default)

✅ retrieval_backend="raw"    → Phase 0/1 retrieval (fixed vocab)
🔄 retrieval_backend="bow_incremental" → Phase 3 (learns new words)

✅ use_hessian_info=False     → Phase 0 info matrices (heuristic guesses)
🔄 use_hessian_info=True      → Phase 4 (real Hessian-derived accuracy)

✅ proximity_enabled=False    → Phase 0 (appearance-only loop detection)
🔄 proximity_enabled=True     → Phase 4 (geometry-only loop detection too)

✅ gravity_prior_enabled=False → Phase 0/1/2/3/4 (no gravity)
🔄 gravity_prior_enabled=True  → Phase 5 (use IMU accelerometer for vertical)


EXAMPLE WORKFLOW:

# Test Phase 0 baseline (should be fast, accurate)
python run_synth.py --scenario square6dof

# Test Phase 1B (slower, not yet as accurate)
python run_synth.py --scenario square6dof --config-override odometry_backend=f2m

# Test Phase 3 (better loop detection)
python run_synth.py --scenario square6dof --config-override retrieval_backend=bow_incremental

# Test Phase 4 (robust to false loops)
python run_synth.py --scenario square6dof \
  --config-override use_hessian_info=True \
  --config-override proximity_enabled=True

# Test Phase 5 (gravity anchor — needs IMU calibration first!)
python run_synth.py --scenario square6dof --config-override gravity_prior_enabled=True