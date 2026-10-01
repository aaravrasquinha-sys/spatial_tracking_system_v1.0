# Spatial Tracking System Runbook (Modules 3 + 4)

Everything you do, in order, from "Orin is on my desk with the D435i
plugged in" to "I can see live 3D tracking in a bird's-eye view." Each
step names the file/doc section it maps to so you can go deeper if
something breaks. Steps 1-11 are Module 3 (2D perception) alone; Steps
12+ add Module 4 (3D localization) on top -- you can stop after Step 11
if you only need 2D tracking.

---

## TL;DR command sequence

```bash
# 1. System packages (once per Orin)
sudo apt update && sudo apt install -y librealsense2-utils librealsense2-dev python3-pip git

# 2. Confirm the camera is seen by the OS/SDK before touching Python
rs-enumerate-devices | head -20

# 3. Get the code onto the Orin, install Python deps
git clone <your-repo-url> poi_perception && cd poi_perception   # or scp the zip over + unzip
pip install -r requirements.txt
pip install -e .
pip install pyrealsense2 ultralytics          # see Step 3 below re: pyrealsense2 on aarch64

# 4. Lock power mode + clocks BEFORE any timing/benchmark run
sudo nvpmodel -q                              # check what mode you're in and what modes exist
sudo nvpmodel -m 0                            # verify what "mode 0" actually is on YOUR kit first
sudo jetson_clocks

# 5. Export the pose model to a TensorRT engine, on-device
python3 scripts/export_model.py --weights yolo11n-pose.pt \
    --out models/engines/yolo11n_pose_fp16.engine --width 640 --height 480

# 6. Sanity-check the whole pipeline with no camera at all
pytest
python3 scripts/demo_synthetic.py --scenario two_crossing --frames 60

# 7. Benchmark on real hardware before trusting any number in the design doc
python3 scripts/bench.py --config configs/m3.example.json --source realsense --frames 300

# 8. Watch it track a real person
python3 scripts/debug_visualize.py --config configs/m3.example.json --source realsense

# 9. Run it for real (writes JSONL logs + auto-clips)
python3 scripts/run_m3.py --config configs/m3.example.json --source realsense

# --- Module 4 (3D localization) from here on ---

# 10. Try it hardware-free first (Phase A, fixed transform, no calibration)
python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.fixed.example.json --source synthetic --mock-scenario two_crossing

# 11. Watch it localize a real person in 3D (bird's-eye view)
python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --source realsense

# 12. Run the numeric ground-truth test (Section 11's actual acceptance check)
python3 scripts/floor_marker_capture.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --out logs/floor_markers/captured.json
python3 -m poi_localization.eval.floor_markers.compare \
    --captured logs/floor_markers/captured.json --ground-truth <your_taped_markers.json>

# 13. Run M3+M4 together for real
python3 scripts/run_m4.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --source realsense
```

The rest of this doc is the "why" and the "what if it breaks" for each
of those 13 steps.

---

## Step 1 — System packages

```bash
sudo apt update
sudo apt install -y librealsense2-utils librealsense2-dev python3-pip git
```

- `librealsense2-utils` gives you `rs-enumerate-devices`, `realsense-viewer`,
  `rs-fw-update` — all useful for diagnosing the camera *before* you
  involve Python at all.
- CUDA and TensorRT do **not** need to be installed separately — they
  ship with JetPack. Don't `pip install tensorrt`; it won't match the
  on-device build (see Step 5's note and `models/trt_engine.py`'s
  docstring).

**Check JetPack/L4T version** (the export manifest in Step 5 records this
automatically, but it's worth knowing up front):
```bash
cat /etc/nv_tegra_release
```

---

## Step 2 — Confirm the camera before touching Python

```bash
rs-enumerate-devices | head -20
```

You want to see the D435i listed with its serial number. If nothing
shows up:
- Try a different USB port (prefer USB 3 directly on the Orin carrier
  board, not a hub).
- `lsusb` should show an Intel device (`8086:` vendor ID).
- If you're on the long 1 m+ cable for deployment rather than the
  desk setup, confirm it's an **active/optical USB3 cable** — long
  passive cables commonly drop the D435i to USB 2, which silently caps
  frame rate and depth quality (Section 0 of the design doc's "Capture
  rig" module flags this explicitly).

Optionally launch `realsense-viewer` (needs a display / X forwarding) to
visually confirm both color and depth streams look sane before writing
any code path around them.

---

## Step 3 — Get the code + Python dependencies

Copy the repo onto the Orin (git clone, `scp`, or unzip the deliverable
zip), then:

```bash
cd poi_perception
pip install -r requirements.txt      # numpy, scipy, opencv-python-headless, pytest
pip install -e .                     # makes `poi_perception` importable everywhere
```

Then the two Orin-specific pieces, from `requirements-optional.txt`:

```bash
pip install pyrealsense2
```
**Note:** on some JetPack versions the PyPI `pyrealsense2` wheel doesn't
have an aarch64 build, or lags the librealsense version that matches
your JetPack's kernel driver. If `pip install pyrealsense2` fails or the
import fails at runtime with a version mismatch, build librealsense from
source with Python bindings enabled — follow Intel's official
"Installing on the Jetson platform" instructions, since this varies with
your exact JetPack/L4T version and is the kind of thing that's stale
within a year of being written down here.

```bash
pip install ultralytics
```
This is only needed for **exporting** the model (Step 5) and if you use
`UltralyticsPoseBackend` for dev/CPU inference. It's not needed on the
deployment path once you have a `.engine` file, since `TRTPoseBackend`
doesn't depend on it.

`tensorrt` and `pycuda` — do **not** `pip install` these. `tensorrt`
ships with JetPack (importable from the system Python once JetPack is
flashed); `pycuda` you generally do want from pip/apt but it must build
against the JetPack CUDA toolchain already on the device:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"   # should already work
pip install pycuda
```
If `import tensorrt` fails here, JetPack's TensorRT isn't on this
Python's path — check you're using the same Python that JetPack
provisioned (`which python3`), not a venv that's isolated from system
site-packages, or add `--system-site-packages` when you made the venv.

---

## Step 4 — Lock power mode and clocks

Do this **before any timing measurement** — Step 5's export doesn't need
it, but Steps 7 onward (benchmark, live run) do, every time you reboot:

```bash
sudo nvpmodel -q          # shows current mode + the list of modes this kit supports
sudo nvpmodel -m 0        # your own findings noted mode 0 is 15W on your kit, NOT MAXN --
                           # confirm what mode 0 means on YOUR board with -q first
sudo jetson_clocks
```

Watch GPU utilization and temperature with `jtop` (`sudo pip install
jetson-stats` if not already present) in a second terminal while you run
Steps 7-9 — a number measured at 15 W and a number measured at
MAXN_SUPER are not comparable.

---

## Step 5 — Export the model to a TensorRT engine

```bash
python3 scripts/export_model.py \
    --weights yolo11n-pose.pt \
    --out models/engines/yolo11n_pose_fp16.engine \
    --width 640 --height 480
```

- `yolo11n-pose.pt` doesn't need to exist locally first — `ultralytics`
  downloads the COCO-pretrained checkpoint automatically the first time
  you reference it by name.
- This writes **two** files: the `.engine` itself and
  `models/engines/yolo11n_pose_fp16.manifest.json`, recording the
  JetPack version, TensorRT version, and the exact export command used.
  `TRTPoseBackend` reads this manifest at load time and warns loudly if
  it's missing or looks stale.
- **Engines are device-specific.** If you ever re-flash JetPack, get a
  different Orin unit, or update TensorRT, re-run this step — don't
  carry an old `.engine` file forward and assume it loads.
- If `trtexec: command not found`, it ships with JetPack's TensorRT
  install, typically at `/usr/src/tensorrt/bin/trtexec` — add that to
  your `PATH`.
- This step runs fine without the camera plugged in.

---

## Step 6 — Sanity-check with no camera at all

```bash
pytest
python3 scripts/demo_synthetic.py --scenario two_crossing --frames 60
```

If these fail, the problem is in your Python environment or the code
itself, not the camera or the engine — fix it here before adding
hardware into the mix. All 168 tests should pass (all three modules —
`pytest tests/test_poi_present` alone are hardware-free too, no
websocket clients needed);
`demo_synthetic.py` should print one line per frame with track IDs and
footpoint sources.

---

## Step 7 — Benchmark on real hardware

```bash
python3 scripts/bench.py --config configs/m3.example.json --source realsense --frames 300
```

Reports per-stage (capture / inference / track) latency percentiles.
**Every number in Section 3 of the design doc is a placeholder** —
this is the step that replaces it with a real measurement on your own
board. If inference latency is much higher than expected, profile before
assuming the model is the bottleneck — NMS/keypoint-decode running on
CPU instead of vectorized is a common hidden cost (see the design doc's
risk table).

---

## Step 8 — Watch it track a real person

```bash
python3 scripts/debug_visualize.py --config configs/m3.example.json --source realsense
```

Stand in front of the camera and walk around. You should see a bounding
box, skeleton, torso polygon outline, and a star at your feet (the
footpoint) with a track ID that stays stable as you move.

- This needs a **GUI-capable OpenCV build** to actually show a window —
  `requirements.txt` installs `opencv-python-headless`, which can't
  open one. If you want live `cv2.imshow()`:
  ```bash
  pip uninstall opencv-python-headless
  pip install opencv-python
  ```
  and either have a monitor on the Orin or SSH in with X forwarding
  (`ssh -X`). Without that, the script auto-falls-back to writing
  annotated JPEGs into `debug_frames/` instead of erroring out — check
  those if you're headless.
- This script is debug-only (see its docstring) — it's not measuring
  real throughput and doesn't write logs. Use it to build intuition
  about `ankle_conf_thresh`, mask placement, and lost-buffer tuning
  before you touch `configs/m3.example.json`.

---

## Step 9 — Draw ROI masks (recommended before long unattended runs)

If your room has a mirror, TV, or window that could produce a false
detection, note the pixel region while watching Step 8's overlay window
(the bbox coordinates print in the on-screen label), then add it to
`configs/masks.cam0.example.json` (or your own `configs/masks.<cam_id>.json`,
pointed to by `mask.masks_path` in your run config):

```json
{
  "cam0": [
    { "name": "living_room_mirror", "polygon": [[400, 50], [620, 50], [620, 300], [400, 300]] }
  ]
}
```

Coordinates are raw pixel space (0-640, 0-480), same space as the
detection boxes. There's no drawing GUI for this yet — eyeball it from
the debug visualizer's overlay, or take a still frame
(`debug_visualize.py --max-frames 1 --save-dir .`) and open it in any
image viewer that shows pixel coordinates on hover.

---

## Step 10 — Run it for real

```bash
python3 scripts/run_m3.py --config configs/m3.example.json --source realsense
```

This is the actual M3 daemon (Section 4's 3-thread architecture):
capture and inference on separate threads, track+footpoint on the main
thread, writing every frame (even empty ones) to
`logs/detections/<cam_id>_<timestamp>.jsonl` and auto-saving clips
around ID switches / lost-buffer entries / low-confidence streaks to
`logs/clips/<cam_id>/`. No live window here — that's Step 8's job.

Stop it with Ctrl+C.

---

## Step 11 — Record and grade the Section 9 scenario set

Once Step 10 works, record each of the seven scripted scenarios (one
person looping, two crossing, exit/re-entry, sitting, partial occlusion,
near a mirror, low light) for real, pointing `output.jsonl_dir` /
`output.clips_dir` at `poi_perception/eval/scenarios/<name>/`, then:

```bash
python3 -m poi_perception.eval.grade poi_perception/eval/scenarios/<name>/detections.jsonl \
    --out poi_perception/eval/scenarios/<name>/baseline_summary.json
```

Hand-grade against the overlay (count real ID switches, confirm zero
false persons from masked regions, note any visibly wrong footpoint) and
write those notes down next to the summary — see
`poi_perception/eval/scenarios/README.md` for the full protocol. That
baseline is what any later change (model swap, tracker retuning) gets
compared against with `eval/replay_diff.py`.

---

## Step 10 (Module 4) — Try it hardware-free first

```bash
python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.fixed.example.json --source synthetic --mock-scenario two_crossing
```

This needs no camera, no calibration, nothing real at all — it uses
`configs/m4.phaseA.fixed.example.json`, which skips Module 4's live IMU+
depth calibration entirely (`phase_a.fixed_transform` is set) and feeds
M3's scripted mock scenarios through the real M4 measurement/tracking
code. You should see two panels: the (blank, synthetic) camera feed on
the left, and a bird's-eye view on the right with colored dots tracking
two people crossing. Use this to build intuition for the visualizer
itself and confirm your Python environment is healthy before adding a
camera.

---

## Step 11 (Module 4) — Watch it localize a real person in 3D

```bash
python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --source realsense
```

Unlike Step 10, `configs/m4.phaseA.example.json` has no
`fixed_transform` set, so this triggers Module 4's **live Phase A
calibration** at startup (Section 2 of the M4 design doc): it opens its
own short-lived RealSense pipeline (separate from the main capture
source, closed before that one opens — see
`frames/local_floor_frame.py`'s docstring for why), captures ~2.5s of
accelerometer samples plus a depth frame, and fits the floor plane via
RANSAC. **Keep the camera perfectly still during this window** — it logs
a message telling you when it's capturing.

You'll see a log line like:

```
Local floor frame estimated: height=2.31m, inliers=3812/4102, rms=4.2mm, gravity_alignment=1.1deg
```

- `height` should roughly match a tape measurement of the camera's
  actual height off the floor — if it's off by more than a few cm,
  something's wrong (see Troubleshooting).
- `gravity_alignment` should be well under
  `phase_a.gravity_alignment_deg_max` (15° by default) — a large value
  means the RANSAC fit disagreed with the IMU and the run aborts with a
  `FloorFitError` rather than silently using a bad frame.
- If it raises `FloorFitError` about too few candidate points, the
  camera doesn't have enough visible floor — tilt it down more (Section
  14's risk table flags this as a development-time-only constraint).

Once calibrated, stand in front of the camera and walk around — the
bird's-eye panel should show a dot tracking you, roughly where you
actually are, with a trail and an uncertainty ellipse. This is a rough
visual sanity check, not the real acceptance test — that's Step 12.

---

## Step 12 (Module 4) — The actual ground-truth acceptance test

Section 11's "done when" thresholds (median error ≤10cm within 4m, ≤20cm
beyond) are checked numerically, not by eyeballing the visualizer. See
`poi_localization/eval/floor_markers/README.md` for the full protocol;
short version:

1. Tape 10+ markers on the floor, tape-measure each from the point
   directly below the camera, write `ground_truth.json`.
2. Capture:
   ```bash
   python3 scripts/floor_marker_capture.py --m3-config configs/m3.example.json \
       --m4-config configs/m4.phaseA.example.json \
       --out poi_localization/eval/floor_markers/<room>/captured.json
   ```
   Stand on each marker, type its name, press Enter, hold still ~1s.
3. Compare:
   ```bash
   python3 -m poi_localization.eval.floor_markers.compare \
       --captured poi_localization/eval/floor_markers/<room>/captured.json \
       --ground-truth poi_localization/eval/floor_markers/<room>/ground_truth.json
   ```
   Prints per-marker error and a PASS/FAIL against Section 11's
   thresholds, split by range.

If it fails, the constants in `configs/m4.phaseA.example.json`'s
`depth_measurement`/`raycast_measurement` sections are the design doc's
own "guesses, tune once real data exists" (`body_thickness_m`,
`sigma_disparity_px`, the per-footpoint-source pixel noise values) —
this is that real data.

---

## Step 13 (Module 4) — Run M3+M4 together for real

```bash
python3 scripts/run_m4.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --source realsense
```

Runs M3's daemon exactly as Step 9 did (same threads, same M3 JSONL log
unless `--no-m3-log`), plus M4's processing chained through M3's
`on_detections` hook. Writes Module 4's own JSONL log to
`logs/world_tracks/<cam_id>_<timestamp>.jsonl` — one line per frame,
each with the frame's list of `WorldTrackPhaseA` records (`p_local`,
`v_local`, `cov_xy`, `height_m`, `src`, `state`). Stop with Ctrl+C.

---

## Phase A → Phase B, later (no code changes)

Once Module 2 produces a calibration file (`T_room_cam` + `map_id`, per
the full system plan's Section 4.2 shape), switch by pointing at
`configs/m4.phaseB.example.json` instead — same commands as Steps 11 and
13, just a different `--m4-config`:

```bash
python3 scripts/run_m4.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseB.example.json --source realsense
```

Fill in `phase_b.calibration_path` (and, once you know it,
`phase_b.expected_map_id` — the run refuses to start if the calibration
file's `map_id` disagrees) in that config first. Nothing else changes:
same measurement code, same tracker, same gates — only the output
records rename `p_local`/`v_local` to the real `poi.v1` `p`/`v` and the
floor-frame transform now comes from `frames/room_frame.py` instead of a
live calibration. Re-run Step 12's floor-marker test in the room frame
afterward — Section 11's Phase B exit criterion is passing it again
after the swap.

---

## Troubleshooting quick reference

| Symptom | Likely cause / fix |
|---|---|
| `RuntimeError: pyrealsense2 is not importable` | Step 3 — install/build pyrealsense2 for your JetPack version |
| Log line: "rgb8 not supported by this device/firmware; using bgr8 + flip" | Harmless — the code already falls back automatically (`capture/realsense_source.py`) |
| Log line: "Queried intrinsics drifted...px from the expected rig values" | Hard warning, not an error — either a different unit/firmware than `CameraConfig`'s defaults, or genuinely worth investigating if the drift is large. Values used are always the queried ones. |
| `trtexec not found` | Step 5 — add `/usr/src/tensorrt/bin` to `PATH` |
| `RuntimeError: tensorrt/pycuda are not importable` | Step 3 — you're not using JetPack's Python, or pycuda isn't installed |
| `No manifest found next to <engine>.engine` warning | You built the engine some other way — re-export via `scripts/export_model.py` so the JetPack/TensorRT version is on record |
| `cv2.imshow unavailable` in `debug_visualize.py` | Expected with `opencv-python-headless` — see Step 8 |
| Frame drops / low FPS only on the long cable | Confirm USB 3 link (`rs-enumerate-devices` shows the link speed) — long passive cables commonly fall back to USB 2 |
| `env_probe`-style intrinsics/IMU tooling from M1/M2 | Not part of M3 — M3 deliberately doesn't import `pyslam` (see README "Module boundary"); use `rs-enumerate-devices` / `realsense-viewer` for camera-level diagnostics instead |
| `FloorFitError: Only N depth points available` | Step 11 — camera doesn't have enough visible floor in frame; tilt it down more during Phase A setup |
| `FloorFitError: ...disagrees with the IMU gravity direction by N degrees` | Step 11 — either a genuinely bad RANSAC fit (not enough real floor visible) or the camera moved during the ~2.5s capture window; retry holding it still |
| Calibration capture hangs / times out | Two RealSense pipelines can't usually open the same device at once — `capture_calibration_data()` runs and fully closes before the main capture source opens; if it hangs, check no other process (realsense-viewer, a previous run) still has the camera open |
| `CalibrationMapIdMismatch` (Phase B) | The calibration file's `map_id` doesn't match `phase_b.expected_map_id` in your M4 config — re-run Module 2's calibration against the currently loaded map, or fix the config |
| Floor-marker test fails Section 11's thresholds | Step 12 — tune `depth_measurement.body_thickness_m`/`sigma_disparity_px` and `raycast_measurement.pixel_noise_*` in your M4 config against this real data, per the design doc's own "these are guesses" notes |
| `pyrealsense2 is not importable` during Phase A calibration | Same fix as M3's Step 3 — Module 4's live calibration capture needs it too; use `phase_a.fixed_transform` in your M4 config to skip it entirely if you already know the rig geometry |
