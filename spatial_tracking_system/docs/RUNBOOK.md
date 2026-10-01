# Runbook — module by module

Everything here is **unvalidated on hardware** (see `SYSTEM_SUMMARY.md` §1). Numbers that say "expect" come from the synthetic
room or from the legacy documents; treat the first real session of each module as a validation exercise, and write what you
measure into `docs/OPEN_ITEMS.md`.

**Conventions.** Run every command from the repository root, inside `pyslam_venv`. `python -m sts …` requires
`pip install -e .` (Part 1). Hardware modes take the camera lock; scripts launched directly (`scripts/*.py`, `run_*.py`) **do
not** — never start one while another camera mode is running. Paths like `data/…` are under the repo root (git-ignored).

| Part | Module | What you finish with |
|---|---|---|
| 1 | Environment | a clean venv, `sts doctor` green except hardware-dependent rows |
| 2 | `site.json` | one file describing the installation |
| 3 | M0 | a rig that records a clean bag, and a rigid repeatable mount |
| 4 | M1 | a locked map bundle with a `map_id` |
| 5 | M2 | an accepted, tape-checked calibration for the camera in its mount |
| 6 | M3 | a TensorRT engine that sustains camera rate; ROI masks; a scenario baseline |
| 7 | M4 | Phase A floor-marker pass, then Phase B floor-marker pass |
| 8 | M5 | dashboard on a phone; WebXR on the Quest |
| 9 | M6 | the registered live run, watchdog verified, soak passed, service installed |
| 10 | Acceptance | the milestone checklist |
| 11 | Troubleshooting | symptom → cause |

---

## Part 1 — Environment (pyslam_venv on the Orin)

Your `pip freeze` showed four things that must be fixed first (details: `SYSTEM_SUMMARY.md` §6).

```bash
cd ~/spatial_tracking_system                      # this repo (copy/unzip it here)
source ~/pyslam_venv/bin/activate

# 1. snapshot so you can roll back
pip freeze > ~/pyslam_venv_before_merge.txt

# 2. remove BOTH stale editable installs (each provides a top-level poi_perception)
pip uninstall -y poi-perception spatial-tracking-system

# 3. install this repo, editable, with the numpy pin. The heavy native stack is NOT installed by pip:
pip install -c requirements/constraints.txt -r requirements/runtime.txt
pip install -e .
pip install -r requirements/dev.txt              # pytest, pytest-asyncio, jsonschema, matplotlib (tests only)

# 4. verify
python -m sts doctor
```

What `doctor` must show (row names are verbatim):

* `import poi_perception / poi_localization / poi_present / pyslam / sts resolves inside this repo` — PASS. FAIL means a stale
  install still shadows this repo: `pip uninstall` it.
* `no duplicate installs of a first-party package` — PASS.
* `numpy` 1.26.x; `websockets in [14, 18)`; `scikit-learn`.
* `pyrealsense2` — PASS, and **check which build it is**. `setup_orin.sh librealsense` builds librealsense from source with
  `-DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_CUDA=ON`; your freeze lists a `pyrealsense2 2.58.4` package. If `pip show pyrealsense2`
  lists it and `rs-enumerate-devices` does not mention RSUSB, it is shadowing the source build: `pip uninstall pyrealsense2`
  and re-run `scripts/setup_orin.sh librealsense`.
* `tensorrt` (10.3 from JetPack) and `pycuda context` — PASS. Never `pip install tensorrt` on the Orin.
* `gtsam` — PASS if your source build imports under numpy 1.26 (`python -c "import gtsam"`). WARN is survivable: M1 falls back
  to its native pose-graph backend.

Bring-up once per unit (legacy procedure, unchanged; `docs/legacy/pyslam/README_ORIN.md` Parts 1–5):

```bash
scripts/setup_orin.sh power            # shows this unit's power modes
scripts/setup_orin.sh power apply      # (sudo) fixed power mode + jetson_clocks: avoids thermal-throttle surprises on long runs
rs-imu-calibration                     # the D435i ships with its IMU uncalibrated; M2's roll/pitch gate needs it
python scripts/tare_calibration_check.py --distance-m 1.000    # after Tare in realsense-viewer against a wall at a tape-measured distance
python -m pyslam.tools.env_probe --realsense                   # camera must be free; prints intrinsics, IMU extrinsics, USB type
```

Keep `torch`/`ultralytics` installed **for engine export only**. Never set `perception.backend` to `"ultralytics"` in a runtime
config: the pip `torch 2.14 (cu13)` is not the JetPack build.

Run the hardware-free suites now, before you add a camera:

```bash
python -m sts test unit integration      # 168 + integration; the ICP test is ~2-5 min
python -m sts test selftest g_live g_anchor   # pyslam's own gates (~3 + ~3 + ~7 min)
```

---

## Part 2 — `site.json`

```bash
python -m sts init --name site_001       # writes configs/site.json (git-ignored) and creates data/ directories
cp configs/camera_model.cam0.example.json configs/camera_model.cam0.json      # all-null until you fit the noise model
```

Edit only what differs from the defaults (`configs/site.example.json` shows every field):

| Field | Set it when |
|---|---|
| `cameras[0].serial` | you want the run pinned to one physical D435i (checked against the calibration's serial) |
| `cameras[0].expected_*` | `env_probe`/`sts doctor --camera` reports intrinsics that differ from 606.75 / 606.57 / 320.19 / 237.06 |
| `cameras[0].masks_path` | after Part 6 (ROI masks) |
| `present.host` / `token` | before leaving your desk: bind to the Orin's LAN IP **and** set a long random token |
| `perception.overrides`, `localization.overrides`, `present.overrides` | tuning; deep-merged into the module's config; **unknown keys are rejected** by the module's strict loader |
| `anchor.set` | M2 threshold overrides (each one is recorded in `report.json`) |
| `watchdog.*` | cadence and `on_suspect` policy (`suppress` default) |
| `slots` | selecting a non-default implementation (`python -m sts slots`) |

`python -m sts configs --phase A` (or `B`) writes the generated module configs to `data/runtime/<site>/<cam>/{m3,m4,present}.json`
and prints their paths — use them with the legacy `scripts/*.py` (they need `--m3-config/--m4-config`). **Regenerate after
changing `site.json`.** The phase you ask for is what gets written.

---

## Part 3 — M0: capture rig and mount

**Goal:** remove the USB-cable blocker and make M2's premise true (the camera returns to the same pose).

1. **Mobile rig.** Orin dev kit + battery + D435i on its own 1 m cable on one tray/handheld frame. Power the Orin from a battery
   through its DC input (laptop-style bank with DC out, or USB-C PD bank + PD-trigger-to-barrel cable at the voltage your board
   expects — **check voltage and polarity on your board before connecting**). Run headless; start/stop over SSH from a phone.
   Recording on the Orin itself matches the legacy note that RSUSB-backend IMU timestamps differ from an x86 recording.
2. **Deployment.** Put the Orin **next to the camera at the mount**; send data out over Ethernet/Wi-Fi. No long USB cable. If you
   must use one, it must be active/optical USB 3 and `sts doctor --camera` must report `USB 3.x link` PASS.
3. **Mount.** Rigid and repeatable (quick-release plate or a dock with a detent). Not a gooseneck or clamp that can rotate.
   Height 2.2–2.5 m, tilted down 20–35° so the view contains a large patch of floor **and** at least two non-parallel walls
   or a corner (a single wall + floor is rejected by M2).
4. **Check the rig:** `python -m sts doctor --camera` (intrinsics, IMU, USB, serial), then a 5-minute headless recording through
   `sts map` (Part 4) with zero dropped frames. The remount repeatability is measured properly by M2's remount check (Part 5).

*Not built:* the plan's `tools/bag_to_tum.py` (TUM export for other SLAM tools). M1 reads bags directly; it is only needed if you
add a reference pipeline (`map_builder` slot, reserved `rtabmap_reference`).

---

## Part 4 — M1: the master map

**Purpose:** one accurate, dense, versioned model of the room. Mapping quality is decided by the capture protocol.

### Before
* `python -m sts doctor --camera` green. Nothing else may hold the camera. Close other workloads — mapping runs three
  processes plus GTSAM and uses most of the 8 GB.
* Room lit as in operation, **nobody else present**, doors/chairs in their usual state.
* Tare check done (Part 1). Capture profile: `configs/capture_profile.mapping.json` (hole-filling **off** on purpose — filled
  depth invents values at discontinuities, wrong for geometry).

### Capture protocol
1. Start: `python -m sts map` (records `data/recordings/<site>/map_<ts>.bag`; Ctrl-C stops and writes the PLY).
   Optional denser keyframes for this run: `python -m sts map -- --config-override keyframe_trans_m=0.10 keyframe_rot_deg=10`.
2. Begin with the camera **seated in its static mount, held still ≥ 5 s** (gives gravity alignment and makes the first keyframe
   the static viewpoint). Lift it out smoothly.
3. Walk the perimeter 1–2 m from the walls, **slowly** (well under 0.5 m/s, under ~30°/s rotation), sweeping yaw rather than
   pointing straight ahead. Make passes at two heights and **tilt down periodically** so the floor is seen across the whole room —
   a level camera sees very little nearby floor and the floor fit needs it.
4. Keep depth within ~3 m of surfaces; do not get closer than ~0.3 m (no depth below that).
5. Revisit the start mid-run (loop closure). **End by returning the camera to the mount and holding still 5 s.**
6. Watch the once-per-second log: `QA: frames=… keyframes=… lost=… icp_fallback=… max_speed=…m/s`. `lost` should stay 0 and
   `max_speed` under ~0.5.
7. Record 2–3 takes (cheap once the rig exists). Use a separate map directory per take if you want to compare them
   (`site.map.dir`), then keep the best.

Known-defect notes: `sts map` refuses `--mode lockstep` (K1). The script's default multiprocessing start method is `fork`, not
`spawn` as the legacy runbook states (K3); `spawn` was never validated — try `python -m sts map -- --mp-start-method spawn`
only if `fork` misbehaves. In `fork` mode the parent skips probing and uses fixed intrinsics (K2): if your unit's differ, update
`run_live_map.py` and `site.json`.

### Lock it
```bash
python -m sts map-lock        # manifest.json + map_id; bundle made read-only; map_id written into site.json
```
This keeps the `map_id` rule (it hashes the same two files M2 hashes). Afterwards **never edit** `dense/points.ply` or
`capture_profile.json` — the Phase-B gate will refuse to run ("map bundle unchanged since locking").

### Accept the map (before moving on)
Run `python -m sts anchor prepare` (Part 5) — it needs no camera and is the map's inspection report:

| Check | Pass | If it fails |
|---|---|---|
| floor rms | ≤ 15 mm | the map is bowed; every calibration sigma will be inflated (intended). Re-take with more tilt-down passes |
| quadrant tilt | ≤ 1° | same |
| "no dominant wall direction" warning | absent | room X falls back to the mapping-start heading (still valid, just not wall-aligned) |
| `top_down.png` | walls where the walls are; no doubled walls | re-take; slow down |
| tape pairs (**manual**, no script) | ≥ 8 pairs incl. 2 full-room diagonals, error ≤ max(2 cm, 1% of distance) | measure two identifiable points in `dense/points.ply` with any point-cloud viewer and compare to a tape |
| start/end mount views | agree within 2 cm (**manual**) | loop closure failed |

If `prepare` errors "no horizontal plane facing the up-prior": the floor was not mapped; re-map with tilt-down passes, or pass
`--up-prior-map x,y,z`.

**Mirrors/windows** corrupt depth locally; **low-texture walls** starve ORB of features (use the emitter; include textured objects
in each view). Anything moved after locking (a chair) is harmless to M4 but will show in VR.

Reference: `docs/legacy/pyslam/RUNBOOK_LIVE.md`, `SYSTEM_SUMMARY_LIVE.md`.

---

## Part 5 — M2: anchor the static camera

**Purpose:** `T_room_cam` for the camera **in its final mount**, with an independent check on every degree of freedom and a way
to notice later that it became wrong. M2 never touches the camera while mapping runs and never modifies the map bundle.

### 5.1 Mount and view (decides accuracy)
Mount the camera where it will live; do not calibrate on a tripod and then move it. Height 2.2–2.5 m, tilt down 20–35°, floor +
≥ 2 non-parallel walls in view. Avoid glass, mirrors, screens, foliage, shiny floors. No cable strain, no fan vibration.
A symmetric room with the camera near the centre is ambiguous by construction (expect `--hint`).

### 5.2 Map-only products (no camera)
```bash
python -m sts anchor prepare       # data/anchors/<site>/cam0/{room_frame.json, walkable.json, viewer/points.ply, top_down.png}
```
Open `top_down.png`: walls are labelled `w0, w1, …` with 1 m axis ticks (needed for `--hint` and the tape check).

### 5.3 Capture — camera seated in its mount, room **empty**, nobody touching it
```bash
python -m sts anchor capture                       # first session: 60 s warm-up (D435i depth drifts while it warms) + 150 frames
python -m sts anchor capture --warmup-s 5          # repeat 2-4 more times, separate sessions, for true repeatability
```
Each capture logs `IMU ok=True`. `ok=False` prints the reason (camera not still, or |accel| not ≈ 9.8 → IMU uncalibrated).
Captures go to `data/captures/<site>/cam0_<timestamp>.npz`. Or both steps at once: `python -m sts anchor calibrate`
(asks you to press Enter between sessions).

### 5.4 Solve (no camera)
```bash
python -m sts anchor solve          # exit 0 = ACCEPTED, exit 2 = REJECTED (+ the failed gates, each with its value)
```
Expect tens of seconds on a laptop core; **the Orin is unmeasured** (H8). If it says *ambiguous* (a second distinct pose explains
the view almost as well): read the camera's true position and heading off `top_down.png` and re-solve with a hint —
`python -m sts anchor solve --hint 1.0,3.4,1.0,-25` (x, y, radius m, heading° from +x, counter-clockwise). The hint only
restricts the search region; every gate still decides.

### 5.5 What ACCEPTED means (every row is in `report.json` with value and threshold)

| Gate | Threshold | If it fails |
|---|---|---|
| `icp_inlier_rmse_m` | ≤ 15 mm | map noise/drift, or the scene changed since mapping → `depth_residual.png` |
| `icp_fitness` | ≥ 0.60 | moved furniture, or wrong pose |
| `map_coverage_in_view` | ≥ 0.70 | holes in the map where the camera looks → re-map that area |
| `roll_pitch_vs_imu_deg` | ≤ 0.5° | IMU uncalibrated, camera moved during the IMU window, or a tilted map floor |
| `height_vs_floor_fit_m` / `tilt_vs_floor_fit_deg` | ≤ 15 mm / ≤ 0.5° | map pose disagrees with the camera's *own* floor |
| `global_uniqueness` | no distinct pose ≥ 0.85× best | hint (above) |
| `observability_min_eig` | ≥ 0.001 | degenerate view (one wall) → change the view |
| `sigma_trans_m` / `sigma_rot_deg` | ≤ 3 cm / ≤ 0.5° | more sessions, more structure in view, or a better map |
| `repeat_spread_*` | ≤ 10 mm / ≤ 0.2° | unstable capture; with ≥ 2 files this is a true session-to-session number |
| `depth_residual_frac_within_tol` | ≥ 0.80 within 3 cm | `overlay.png` / `depth_residual.png` |
| `capture_profile_match` | same hash | only means "same profile *file*" (H4) |

`sigma = sqrt(ICP² + repeatability² + map-floor²)` is what M4 adds to every position covariance. The ICP inflation (×4) and
noise multiplier (×2) are **guesses** (H1); on real data sigma may come out less pessimistic.

**Visual sign-off (30 s, do it):** `overlay.png` — red = map depth edges, green = live depth edges, yellow = coincident; correct
= lines on top of each other; a yaw error shows as two parallel lines. `depth_residual.png` — a uniform colour patch on one wall
means that wall is misplaced in the *map* (drift), not that the camera is wrong. `frustum_coverage.png` — green = floor the
camera actually sees, grey = **dead zone** (M4 cannot see people there; the tracks vanish — know this number).

### 5.6 Tape-measure check (independent of everything above — trust it over the gates)
Put 3+ markers on the floor in view at 3–5 m. Measure each one's perpendicular distance to two **non-parallel walls**; find its
pixel in the RGB image (a bright tape cross is easy). `markers.json`:
```json
{"markers": [
  {"name": "m1", "pixel": [412, 388], "wall_offsets": [{"wall": 0, "d": 1.85}, {"wall": 2, "d": 2.40}]},
  {"name": "m2", "pixel": [301, 352], "wall_offsets": [{"wall": 0, "d": 2.60}, {"wall": 3, "d": 1.10}]}]}
```
```bash
python -m sts anchor markers --markers markers.json      # exit 0 = all within 3 cm
```
Sanity: a 1° yaw error moves a point 4 m away by ~7 cm. **If the tape disagrees with the gates, trust the tape** (suspect map
scale/drift or intrinsics; re-map).

### 5.7 Remount check and repeatability
Remove and reseat the camera, recalibrate; the pose must agree with the original within **1 cm / 0.3°**, otherwise the mount needs
redesigning. Five independent calibrations should spread ≤ 1 cm / 0.2°.

### 5.8 Accept into the system
```bash
python -m sts accept        # records map_id in site.json and runs the full consistency gate (22 checks)
python -m sts check         # any time later
```
`accept` refuses unless every check passes; a rejected run is written as `calibration.cam0.REJECTED.json`, a name M4 cannot load.

Reference: `docs/legacy/pyslam/RUNBOOK_ANCHOR.md`, `SYSTEM_SUMMARY_ANCHOR.md`.

---

## Part 6 — M3: 2D perception

### 6.1 Build the engine on this Orin
```bash
python -m sts engine            # ONNX export (needs ultralytics/torch) then trtexec FP16 at the CAMERA's width x height
```
Writes `data/models/engines/yolo11n_pose_fp16.engine` and `.manifest.json` (JetPack, TensorRT, exact commands). **Engines are
device-specific:** re-run after any JetPack/TensorRT change or on a different unit. If `trtexec: command not found`, it is at
`/usr/src/tensorrt/bin/trtexec` — add it to `PATH`. `sts doctor` then compares the engine's input size to the camera and its
JetPack string to `/etc/nv_tegra_release`. (INT8 is intentionally not offered: accuracy loss outweighs the gain at the n size.)

### 6.2 Benchmark — replace the placeholders with measurements
```bash
python -m sts configs --phase A
python scripts/bench.py --config data/runtime/site_001/cam0/m3.json --source realsense --frames 300
```
Per-stage capture/inference/track latency percentiles. **Pass: the engine plus tracking sustains camera rate (30 fps) at 640×480.**
If inference is slow, profile before blaming the model (NMS/keypoint decode on CPU is a common hidden cost).

### 6.3 Watch it (debug only)
`python scripts/debug_visualize.py --config data/runtime/site_001/cam0/m3.json --source realsense` — box, skeleton, torso
outline, footpoint star, stable ID. `opencv-python-headless` cannot open a window; without a GUI build it writes JPEGs to
`debug_frames/` (or `pip install opencv-python` and use `ssh -X`).

### 6.4 ROI masks (before any unattended run)
Note the pixel regions of mirrors, TVs, windows, posters from the overlay, then add them to `configs/masks.cam0.json`
(`{"cam0": [{"name": "...", "polygon": [[x,y], …]}]}`, raw 640×480 pixel space) and set `cameras[0].masks_path` in `site.json`.
M4's 3D checks also reject these, but masking at the source keeps logs clean.

### 6.5 Scenario baselines
Record the scripted set in your room (one person looping, two crossing, exit/re-entry, sitting, partial occlusion behind
furniture, near a mirror/TV, low light): run M3 (`python scripts/run_m3.py --config …/m3.json --source realsense`; Ctrl-C to stop),
then `python -m poi_perception.eval.grade <detections.jsonl> --out baseline_summary.json` and hand-grade the overlay (count real
ID switches; confirm zero false persons from masked regions). **Pass (v1): no false people from masked regions; ID switches rare
enough to count by hand.** Record the counts as the baseline to beat; later changes are compared with
`python -m poi_perception.eval.replay_diff`. Protocol: `poi_perception/eval/scenarios/README.md`.

---

## Part 7 — M4: 3D localization

### 7.1 Hardware-free sanity
```bash
python -m sts run --phase A --source synthetic --mock-scenario two_crossing   # then open the dashboard (Part 8)
```

### 7.2 Phase A (provisional floor frame; needs no map) — validate M3+M4 independently of M1/M2
```bash
python -m sts configs --phase A
python scripts/debug_visualize_world.py --m3-config data/runtime/site_001/cam0/m3.json \
    --m4-config data/runtime/site_001/cam0/m4.json --source realsense
```
Phase A's live calibration opens its own short-lived pipeline and needs **~2.5 s of a perfectly still camera**. Expect a line like
`Local floor frame estimated: height=2.31m, inliers=…, rms=4.2mm, gravity_alignment=1.1deg`: `height` should match a tape
measurement within a few cm; `gravity_alignment` well under 15° or it aborts (`FloorFitError`: tilt the camera down — too little floor).
To skip the live calibration (you know the geometry): `localization.overrides.phase_a.fixed_transform = {height_m, pitch_deg, yaw_deg}`.
(Two sequential RealSense opens per start is legacy design and unvalidated under RSUSB — W3.)

### 7.3 The acceptance test — floor markers
Tape 10+ crosses with a real spread of distances and angles (not a grid in front); measure each from the floor point directly
below the camera (**Phase A**: X along the camera's forward direction at calibration time, Y perpendicular; **Phase B**: room
coordinates). Write `ground_truth.json` (`{"markers": [{"name": "m1", "x": 1.20, "y": 0.35}, …]}`), then:
```bash
python scripts/floor_marker_capture.py --m3-config data/runtime/site_001/cam0/m3.json \
    --m4-config data/runtime/site_001/cam0/m4.json --out poi_localization/eval/floor_markers/<room>/captured.json
python -m poi_localization.eval.floor_markers.compare --captured …/captured.json --ground-truth …/ground_truth.json
```
Stand on each marker, type its name, Enter, hold still ~1 s; one person in frame only. **Pass: median error ≤ 10 cm within 4 m of
the camera and ≤ 20 cm beyond.** Repeat in Phase B (Part 9) — that is the integration exit criterion (I1).

### 7.4 Other M4 checks (hand-graded against the JSONL + `poi_localization/eval/grade.py`)
Straight-line walk along a taped line (lateral RMS ≤ 5 cm) · depth dropout with dark IR-absorbing clothing or long range (tracking
continues on ray-cast, `src` says so) · walking behind a sofa < 1 s (same ID via coasting) · mirror/TV reflections never become
confirmed tracks.

### 7.5 If it fails — tune in data, not code
Constants are the design's own "guesses": `depth_measurement.body_thickness_m`, `sigma_disparity_px`, `depth_stereo_fx_px`,
`depth_bias_frac`, `raycast_measurement.pixel_noise_*`, `track.*`. Put depth-noise numbers in
`configs/camera_model.cam0.json` (one place for M2 **and** M4) and the rest in `site.json -> localization.overrides`; regenerate
(`python -m sts configs`), re-run 7.3. Reference: `docs/legacy/sts/RUNBOOK.md`, `CHANGES.md`.

---

## Part 8 — M5: dashboard and WebXR

```bash
python -m sts run --phase A --source synthetic         # fake walkers, no hardware; serves http://127.0.0.1:8765/
```
Open `/dashboard/` (2D radar, tracks, trails, uncertainty ellipses, camera frustum, events, health footer) and `/vr/`.

**Before leaving your desk** (`site.json`): `present.host` = the Orin's LAN IP (not `0.0.0.0`) and a long random `present.token`;
then every URL needs `?token=…` (both pages forward it to the WebSocket). The stream is a record of movement in your home.

**Quest:** WebXR immersive sessions require a secure context, so the headset will **not** enter VR from `http://<lan-ip>`.
* Development: `adb reverse tcp:8765 tcp:8765`, then open `http://localhost:8765/vr/?token=…` in the Quest browser (localhost is secure).
  Desktop testing: the *Immersive Web Emulator* browser extension.
* Untethered: serve HTTPS — set `present.tls_cert` / `present.tls_key` (a certificate the headset trusts) and use `https://`.

The viewer renders ~100 ms in the past and interpolates on `t_capture`; modes: **life-size** (teleport) and **dollhouse** (1:20 on a
table). Start with the decimated cloud and **measure frame rate on the headset before adding detail** (Quest budgets are tight).
Pass: 5a shows live tracks from the real camera; 5b at native frame rate with 5 synthetic walkers, then with the live stream, with
capture-to-photon under ~200 ms (the VR viewer estimates the clock offset with ping/pong).

---

## Part 9 — M6: the registered live run, watchdog, soak, service

### 9.1 First registered run
```bash
python -m sts check                  # the consistency gate on its own (22 checks)
python -m sts run --phase B          # refuses (exit 2) unless the gate passes; then capture -> M3 -> M4 -> M5 + watchdog
```
The run manifest (`data/logs/<site>/<cam>/run_<ts>.manifest.json`) ties the logs to code, configs, calibration hash and engine.
On start the calibration is **`suspect` until the first depth check passes**; `events.jsonl` records the verdict. **W5 risk:** the
runtime depth path (High-Accuracy preset + spatial/temporal/hole-filling) differs from the capture path M2 used for its
reference depth; if startup verification flags a clean scene, read its reason (`startup: X% … moved`) and see OPEN_ITEMS W5.

### 9.2 Degraded mode
While calibration is `suspect` and `watchdog.on_suspect = "suppress"`: `tracks` messages keep flowing **empty**,
`health.calib = "suspect"`, a `calib_suspect` event is sent (id −1), M3/M4 keep running and logging. `"flag"` keeps publishing.
Only an **ICP recheck at the calibrated pose that converges with good fitness** clears it (run while nobody is tracked for
`recheck_idle_s`); nothing ever re-calibrates on its own. To adopt a new pose, re-run Part 5 in full.

### 9.3 Nudge test (required)
Knock the camera ~4° in yaw. **Expect** `calib: "suspect"` within one depth cycle: worst case `depth_check_period_s` (120 s) plus
two 5 s confirmation checks (`wd_consecutive = 3`), then a `calib_suspect` event. A ~1.5° yaw nudge is **below** the depth layer and is
only caught by the ICP recheck (up to `recheck_period_s`, 900 s, and only while the room is idle). Then re-seat the camera:
the recheck clears it. Also test a tilt and a small translation and record what is caught (H7).

### 9.4 One-hour soak (required)
```bash
python -m sts run --phase B &        # or via the service (9.6)
python - <<'PY'
import json, glob
f = sorted(glob.glob("data/logs/site_001/cam0/soak.jsonl"))[0]
rows = [json.loads(l) for l in open(f)]
print("samples", len(rows), " rss MB first/last:", rows[0]["rss_mb"], rows[-1]["rss_mb"],
      " max temp:", max((r["temp_c_max"] or 0) for r in rows), " drops:", rows[-1].get("dropped_capture"), rows[-1].get("dropped_infer"))
PY
```
**Pass:** no crash, RSS flat (no upward trend), temperature steady (fixed power mode, active cooling), drop counters not climbing,
`health.fps` ≈ camera rate.

### 9.5 Replay and baselines (the regression workflow)
Record a small library of real bags through `sts map`-style recording or `rs-record`: a walk, two people crossing, occlusion,
mirror/TV, a bumped camera. Then:
```bash
python -m sts replay --bag data/recordings/site_001/walk.bag --save-baseline baselines/walk.json   # first time
python -m sts replay --bag data/recordings/site_001/walk.bag --compare baselines/walk.json         # after every M3/M4 change
```
Replay is **lockstep** (nothing dropped) and therefore deterministic; the live chain deliberately drops stale frames.

### 9.6 Service
```bash
sudo cp deploy/sts-run.service /etc/systemd/system/     # edit User= and the two paths first
sudo systemctl daemon-reload && sudo systemctl enable --now sts-run
sudo cp deploy/sts-retention.{service,timer} /etc/systemd/system/ && sudo systemctl enable --now sts-retention.timer
```
Exit codes: 2 (refused: configuration/consistency) and 4 (camera busy) are **not** restarted; 3 (capture/inference died) is.
`python -m sts retention` (dry run) / `--apply` prunes per `site.retention` (track logs 14 d, clips 30 d, bags 7 d).

### 9.7 Changing a module later
`docs/ARCHITECTURE.md`, "Workflow for changing a module". In short: edit inside the module → its own suite →
`sts test integration` → `sts replay --compare` on real bags → regenerate golden/baselines if the change is intentional and say
why in `docs/CHANGELOG.md` → bump `COMPATIBILITY.md` only if a contract changed.

---

## Part 10 — Acceptance checklist

| Milestone | Done when |
|---|---|
| M0 rig | 5-min headless recording, zero dropped frames; mount reseats repeatably (Part 5.7) |
| A1 perception live | 30 fps sustained at 640×480 (`bench.py`); scenario clips recorded and baselined |
| A2 3D in Phase A | floor-marker test passes in the provisional frame |
| A3 live dashboard | real trails on a phone from the real camera |
| A4 VR on synthetic | native frame rate on the headset with 5 walkers |
| B1 map locked | Part 4 acceptance table; bundle read-only; `map_id` in `site.json` |
| B2 anchor | Part 5.5 gates pass; tape check ≤ 3 cm; remount ≤ 1 cm / 0.3°; overlay aligned |
| I1 registered tracking | floor-marker test passes in the **room** frame |
| I2 live VR | watch another person walk through the room in VR, < ~200 ms capture-to-photon |
| H1 hardening | one-hour soak clean; nudge detected; replay deterministic; service survives a reboot |

---

## Part 11 — Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `sts run` exits 2 | read the gate report; each FAIL names the file. Typical: calibration from another map, map edited after locking, M5 pointed at `maps/` instead of `anchors/` |
| exit 4 "camera is in use by `sts <mode>`" | stop that mode. Legacy scripts started by hand bypass the lock and produce opaque librealsense errors instead |
| `ModuleNotFoundError: websockets` | `pip install -c requirements/constraints.txt -r requirements/runtime.txt` |
| `import poi_perception` resolves outside the repo | stale editable install: `pip uninstall poi-perception spatial-tracking-system` |
| TRT engine won't load | JetPack changed since export → `python -m sts engine` |
| `FloorFitError` | too little floor visible (Phase A/2.5 s still window) — tilt down, keep still |
| M2: `only … stable depth points survived` | dark/IR-absorbing scene, glass, or the camera moved |
| M2: "floor RANSAC found only a thin strip" | no real floor in view; tilt down / move back |
| M2: rejected, room not symmetric | featureless view → add a hint or change the view |
| Accepted but tape off by > 3 cm | trust the tape: map scale/drift or intrinsics; re-map |
| Tracks vanish but video is fine | calibration `suspect` with the suppress policy → `events.jsonl` |
| Tracks vanish in part of the room | a dead zone: see `frustum_coverage.png` |
| Positions offset by a constant | overlay + tape check; intrinsics; then M4 constants |
| `health.clock = "device"` | `global_time_enabled` failed; latency numbers are meaningless until fixed |
| Quest won't enter VR | not a secure context (Part 8) |
| Dashboard event `#-1` | it's `calib_suspect` (V1, cosmetic) |
| Everything odd after an edit | `python -m sts provenance`, `sts test integration`, `sts replay --compare` |
