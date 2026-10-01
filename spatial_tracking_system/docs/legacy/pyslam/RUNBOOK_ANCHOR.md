# RUNBOOK_ANCHOR: Module 2 (static-camera anchor / `T_room_cam`)

Read `SYSTEM_SUMMARY_ANCHOR.md` §1 first: **nothing here has been run on real hardware yet.**
Every number in this runbook that says "expect" comes from a synthetic room. The first real
session is a validation exercise as much as a calibration. Do the tape-measure check (Part 6).

What this does: takes the map module 1 produced and a short capture from a D435i sitting in its
**final mount**, and computes where that camera is in a floor-referenced room frame (Z up, floor
z = 0), with an honest uncertainty. It refuses to write a calibration if independent checks disagree.

Module 1 runs separately. Module 2 never touches the camera while mapping is running, and never
modifies the map bundle: outputs go in a separate `anchors/<name>/` directory.

---

## Part 0: Prerequisites

| Need | Why | Check |
|---|---|---|
| A module 1 bundle `maps/<site>/` containing `dense/points.ply` | the thing we localise against | `ls maps/<site>/dense/points.ply` |
| The map's floor was actually mapped | the room frame is derived from the floor plane | `prepare` (Part 3) tells you |
| At least 2 non-parallel walls (or a corner) in the camera's view, plus floor | otherwise x/y/yaw are not determined; the tool rejects the view | Part 2 |
| D435i with a **calibrated IMU** | roll/pitch cross-check needs the accelerometer to read ~9.81 m/s² | `python3 -m pyslam.tools.env_probe --realsense` |
| `pip`: nothing new | numpy, scipy, opencv only (already in the repo's requirements) | |

Live-mode maps have **no manifest / map_id** (SYSTEM_SUMMARY_LIVE §4.5). Module 2 computes `map_id`
with module 1's own `compute_map_id` over `dense/points.ply` + `capture_profile.json`, so the id
matches what a later lock would produce **as long as you do not edit those two files**. If you
re-export or clean the PLY, you get a new `map_id` and must recalibrate.

### `--imu-mode`

The repo's `run_live_map.py` only accepts `--imu-mode callback|synced`. The version of
`run_live_map.py` you pasted also offers `polling`, which `RealSenseSource` rejects with a
`ValueError`. Use `callback` (default) or `synced`. `run_anchor.py` accepts the same two.

---

## Part 1: One-time camera preparation

1. **IMU calibration.** Run Intel's `rs-imu-calibration` (or the Dynamic Calibration tool) once per
   camera. Check with the capture step below: it reports `|accel|` and refuses an IMU window whose
   magnitude is outside 8.3-11.3 m/s².
2. **Firmware / depth preset.** Same as mapping. Module 2 has no independent depth setup.
3. **Capture profile.** `run_anchor.py capture` records the hash of
   `configs/capture_profile.mapping.json` (or `--capture-profile PATH`), the same file
   `run_live_map.py` uses. **Caveat:** `RealSenseSource` (both mapping and this module) does not
   push that profile to the device; the hash records what was *declared*, not what was *verified on
   the sensor*. The `capture_profile_match` gate therefore catches "different profile file", not
   "device silently at defaults". If you want a real guarantee, apply the profile in the source
   constructor (`pyslam.live.capture_profile.apply_to_realsense`) for both mapping and anchoring.

---

## Part 2: Mounting and view (do this before anything else; it decides accuracy)

Mount the camera where it will live. Do not calibrate on a tripod and then move it.

* **Height 2.2-2.5 m**, tilted **down 20-35°** so the view contains: a large patch of **floor**, and
  **at least two non-parallel walls** or a corner (fixes x, y, yaw). Furniture edges help; a single
  flat wall + floor is *rejected* (`observability_min_eig`, `global_uniqueness`).
* **Floor is mandatory**: the tool fits the camera's own floor (height + tilt cross-check). If it
  only finds a thin strip it reports "a horizontal slice through a wall": tilt down or move back.
* Avoid views dominated by glass, mirrors, screens, foliage, shiny floors: depth there is unstable
  and is filtered (or, worse, consistently wrong).
* Rigid mount, no cable strain, no fan vibration. A flexible arm defeats the whole exercise.
* Symmetric rooms (empty rectangle, camera near the centre) are ambiguous by construction; expect
  to need `--hint` (Part 4) or to move the camera to see something asymmetric.

---

## Part 3: Run order

All commands from the repo root (`pyslam_live/`). Paths are examples.

```bash
# 0. Map-only products, no camera needed. Look at the picture before you go near the camera.
python3 run_anchor.py prepare --map maps/site_001 --out anchors/site_001
#   -> anchors/site_001/{room_frame.json, walkable.json, viewer/points.ply, top_down.png}
#   Read the warnings. top_down.png shows walls (w0, w1, ...) and 1 m axis ticks.
```

Things to check in the `prepare` output:

* `floor rms` ≤ 15 mm and "quad tilt" ≤ 1° : otherwise the map itself is bowed and every
  calibration sigma is inflated (that is intended: the map's accuracy floor is part of sigma).
* No warning "no dominant wall direction": if you get it, room X falls back to the mapping-start
  heading (still a valid frame, just not wall-aligned).
* If it errors "no horizontal plane facing the up-prior": the floor was not mapped, or the map's
  up direction is far from W_cam0 −y. Pass `--up-prior-map x,y,z` (rough "up" in map coordinates).
  A wrong-sign prior can pick the ceiling; the height-vs-own-floor gate should then fail loudly.

```bash
# 1. Capture, camera seated in its final mount, room EMPTY, nobody touching it.
#    60 s warm-up (D435i depth drifts while it warms) + ~150 frames. Repeat 3-5 times.
python3 run_anchor.py capture --realsense --map maps/site_001 --out captures/cam0_a.npz
python3 run_anchor.py capture --realsense --map maps/site_001 --out captures/cam0_b.npz --warmup-s 5
python3 run_anchor.py capture --realsense --map maps/site_001 --out captures/cam0_c.npz --warmup-s 5
```

Check each capture's log line: `IMU ok=True`. If `ok=False`, the reason is printed (camera not still,
or `|accel|` not ~9.8). One camera owner at a time: this fails if another process has the device.

```bash
# 2. Solve. No camera needed; runs anywhere the files are.
python3 run_anchor.py solve --map maps/site_001 --capture captures/cam0_*.npz --out anchors/site_001
#   exit 0 = ACCEPTED (calibration.cam0.json written); exit 2 = REJECTED (.REJECTED.json + reasons)
```

Or `python3 run_anchor.py calibrate --realsense --map maps/site_001 --out anchors/site_001 --sessions 3`
for capture + solve in one go.

Runtime: ~20-30 s on a laptop CPU for the synthetic case; expect longer on the Orin Nano
(single-threaded numpy dominates). It is a one-time calibration, not a real-time process.

---

## Part 4: If the solve says "ambiguous"

`global_uniqueness` failing means a **second, distinct pose explains the view almost as well** as the
best one (ICP fitness ratio ≥ 0.85) and colour could not break the tie. The tool will not guess.

1. Open `anchors/site_001/top_down.png`. Read the camera's true position (x, y in metres, room
   frame, from the axis ticks) and roughly which way it faces (degrees from +x, counter-clockwise).
2. Re-solve with a hint. Radius is how sure you are of the position (metres):

```bash
python3 run_anchor.py solve --map maps/site_001 --capture captures/cam0_*.npz --out anchors/site_001 \
        --hint 1.0,3.4,1.0,-25        # x, y, radius_m, heading_deg
```

The hint only restricts the *search region*; ICP and every gate still decide. A wrong hint fails the
gates, it does not produce a wrong accepted calibration. The method string in the calibration then
contains `operator_hint`.

---

## Part 5: What ACCEPTED means (gates, and what to do when one fails)

Every row is in `report.json` with its value and threshold. Required rows must all pass.

| Gate | Threshold | If it fails |
|---|---|---|
| `icp_inlier_rmse_m` | ≤ 15 mm | Map noise/drift, or the scene changed a lot since mapping. Look at `depth_residual.png`. |
| `icp_fitness` | ≥ 0.60 | Not enough of the view agrees with the map: moved furniture, or wrong pose. |
| `map_coverage_in_view` | ≥ 0.70 | **Holes in the map** where the camera looks (M1 can drop keyframes when its queue fills). Re-map that area, or aim the camera elsewhere. |
| `roll_pitch_vs_imu_deg` | ≤ 0.5° | IMU miscalibrated, camera moved during the IMU window, or the map floor is tilted. |
| `height_vs_floor_fit_m` / `tilt_vs_floor_fit_deg` | ≤ 15 mm / ≤ 0.5° | Camera height/tilt per the *map* disagrees with the camera's *own* floor: wrong pose, or bowed map floor. |
| `global_uniqueness` | no distinct pose ≥ 0.85× best | Part 4. |
| `observability_min_eig` | ≥ 0.001 | Degenerate view (one wall, corridor). The report names the weak direction. Change the view. |
| `sigma_trans_m` / `sigma_rot_deg` | ≤ 3 cm / ≤ 0.5° | Too uncertain to be useful. More sessions, a view with more structure, or a better map. |
| `repeat_spread_*` | ≤ 10 mm / ≤ 0.2° | Unstable capture. With ≥2 independent capture files this is a true session-to-session number; with one file it is only sub-median spread (says so in `kind`). |
| `depth_residual_frac_within_tol` | ≥ 0.80 within 3 cm | Rendered-map depth disagrees with live depth over the image. See `overlay.png` / `depth_residual.png`. |
| `capture_profile_match` | same hash | Part 1 caveat. Informational if either side has no hash. |

`sigma` = √(ICP² + repeatability² + map-accuracy-floor²). It is what M4 propagates into every
position covariance. **On real hardware the ICP part is untuned** (noise multiplier and covariance
inflation were chosen, not measured, see SYSTEM_SUMMARY_ANCHOR §5).

Visual sign-off (do it, it takes 30 seconds):

* `overlay.png`: red = map depth edges, green = live depth edges, yellow = coincident. Correct = lines
  on top of each other. A yaw error shows as two parallel lines.
* `depth_residual.png`: blue/red = live nearer/farther than the map. A uniform colour patch on one
  wall means that wall is misplaced in the map (drift), not that the camera is wrong.
* `top_down.png` / `frustum_coverage.png`: green = floor the camera actually sees; grey = **dead zone**
  (M4 cannot see people there; M5 should show it as such). Olive = in the frustum but no depth reading.

---

## Part 6: Tape-measure check (independent of everything above)

Put 3+ markers on the floor in view, at 3-5 m from the camera. Measure each one's distance to two
**non-parallel walls** (perpendicular distance, tape). Find its pixel in the RGB image (a bright tape
cross is easy). Wall ids `w0, w1, ...` come from `top_down.png` / `room_frame.json`.

`markers.json`:
```json
{"markers": [
  {"name": "m1", "pixel": [412, 388],
   "wall_offsets": [{"wall": 0, "d": 1.85}, {"wall": 2, "d": 2.40}]},
  {"name": "m2", "pixel": [301, 352],
   "wall_offsets": [{"wall": 0, "d": 2.60}, {"wall": 3, "d": 1.10}]}
]}
```
(`"room_xy": [x, y]` also works if you already know coordinates.)

```bash
python3 run_anchor.py markers --calib anchors/site_001/calibration.cam0.json \
        --room anchors/site_001/room_frame.json --markers markers.json     # exit 0 = all within 3 cm
```

This is the check that would expose a *systematic* fault the self-consistency gates cannot (a
consistent wrong pose, a scale error in the map). If it disagrees with the gates, **trust the tape**
and investigate. Sanity: a 1° yaw error moves a point 4 m away by ~7 cm.

---

## Part 7: Wire it into M4 / M5 (STS repo, no code changes)

Copy `anchors/site_001/calibration.cam0.json` and `walkable.json` next to the STS configs (or
reference them by path).

M4 (`configs/m4.phaseB.example.json`):
```json
"phase": "B",
"phase_b": {
  "calibration_path": "anchors/site_001/calibration.cam0.json",
  "expected_map_id": "<map_id printed in the calibration / report.json>",
  "walkable_grid_path": "anchors/site_001/walkable.json"
}
```
With `expected_map_id` set, M4 refuses to start on a calibration from a different map
(`CalibrationMapIdMismatch`). `walkable.json` is in the room frame, the same frame `T_room_cam`
maps into, so the walkability gate needs no conversion.

M5 (`configs/present.example.json`): set `scene.map_bundle_dir` to `anchors/site_001` (it discovers
`viewer/room.glb` and `viewer/points.ply`; this module writes `viewer/points.ply` **in the room
frame**, so positions and the point cloud line up; no `room.glb` is produced). Use the frame name
`"room"` and the same `map_id` where the config asks for them.

`calibration.cam0.json` fields: `T_room_cam` (4×4, `p_room = R p_cam + t`, camera optical frame:
x right, y down, z forward), `sigma {trans_m, rot_deg}`, `map_id`, `intrinsics`, `gravity_up_cam`,
`checks` (every gate value), `method`, `accepted`. A rejected run writes
`calibration.cam0.REJECTED.json`, a name M4 cannot load by accident.

---

## Part 8: Watchdog (keeping the calibration honest)

Library: `pyslam.anchor.watchdog.AnchorWatchdog(calibration.json, cam0_reference_depth.npz)`.

| Layer | Cost | Catches | Blind to |
|---|---|---|---|
| `check_tilt(up_cam_now)` | ~free (accelerometer) | tilt change > 0.2° | pure yaw, translation |
| `check_depth(depth_median)` | cheap | many stable reference pixels moved > 5 cm for 3 consecutive checks (tested: ~4° yaw; a person occluding ~8% of the image does *not* trip it) | small nudges: a **1.5° yaw nudge moved only ~11% of pixels** in the synthetic test, below the 15% trigger |
| `recheck_pose(capture, map_pts_room)` | seconds (ICP) | 1 cm / 0.3° shifts (tested at 1.5°) | : |

Because the depth layer misses small yaw nudges, **run the ICP recheck on a schedule (every few
minutes to hourly) when nobody is in view, not only when a cheaper layer fires.** It never silently
re-calibrates: it reports `ok` / `suspect` and the measured shift; adopting a new pose means
re-running the full calibration and passing every gate.

CLI one-shot (camera must be free):
```bash
python3 run_anchor.py check --realsense --anchor anchors/site_001 --map maps/site_001
```

**Not wired:** the STS repo's M6 daemon does not call this library yet. `health_field()` returns the
`"ok"|"suspect"` string intended for its health message; the integration is not built.

---

## Part 9: Troubleshooting

| Symptom | Likely cause |
|---|---|
| `no horizontal plane facing the up-prior` | Floor not mapped (re-map with periodic tilt-down), or wrong up prior. |
| `only ... stable depth points survived` | Capture too sparse: dark/IR-absorbing scene, glass, or camera moved. |
| `the 'floor' RANSAC found only a thin strip` | No real floor in view. Tilt down / mount further from the wall. |
| `floor fit put the camera ... negative` | IMU up-vector wrong sign/frame or gross floor mis-fit. Check `R_body_cam` printed by `env_probe`. |
| `only N map points within ...` | Camera position outside the mapped area, or a hole in the map there. |
| `IMU ok=False` in capture | Camera moving during the window, or `|accel|` far from 9.8. Recalibrate IMU. |
| Rejected with ambiguity, room is not symmetric | Featureless view. Add a hint or change the view. |
| Accepted but tape check off by >3 cm | Trust the tape. Suspect map scale/drift or intrinsics; re-map. |
| Slow | Single-threaded numpy dominates (ICP + normals). `--set icp_normal_k=12 query_stride=3` trades accuracy for speed. |

Any config threshold can be overridden per run: `--set gate_fitness_min=0.5 capture_frames=200`.
Overriding a *gate* is a decision, and it is written into `report.json` (`config`).

---

## Part 10: Tests

```bash
PYTHONPATH=.:/path/to/spatial_tracking_system python3 -m tests.gates.test_g_anchor   # G-ANCHOR (20 checks)
python3 -m tests.gates.test_g_live       # existing suite, must stay green
python3 -m pyslam.selftest               # existing suite, must stay green
```
The M4 repo on `PYTHONPATH` is optional: with it, one check loads the produced calibration through
M4's real `load_room_frame`; without it that check validates the contract inline.
