# Spatial Tracking System — Modules 3 + 4 + 5

## Accuracy fixes (this session)

A physically-realistic D435i error simulation (1.5% proportional depth
noise + bias, 2px keypoint noise, gait sway) against the real
measurement + tracking code found several concrete accuracy bugs, now
fixed:

- **Extrinsic (calibration) uncertainty was never modeled.** `cov_xy`
  only reflected sensor noise, so a 1deg Phase A pitch error produced
  ~14cm real error against ~1cm reported covariance (NEES ~90 — an
  order of magnitude overconfident). `FloorFrameTransform` now carries
  `sigma_rot_rad`/`sigma_trans_m`, populated from the Phase A fit's own
  diagnostics or Phase B's calibration-file `sigma` block, and both
  measurement models add its contribution to `cov_xy`. See
  `geometry.extrinsic_position_variance` and
  `poi_localization/eval/consistency.py` (a new NEES-based covariance-
  honesty check, run in CI via `tests/test_consistency_nees.py`).
- **Depth was gated out ~55% of the time** by `max_range_m=3.0` and an
  under-estimated noise model (used the color stream's focal length
  instead of the depth pair's, ignored sensor bias). Both fixed;
  `src=fused` went from ~30% to ~97% of frames in simulation, i.e.
  depth-primary tracking now actually behaves as designed.
- **Self-intersecting ("bowtie") torso polygon** for a camera-facing
  person with a partial keypoint set — silently halved the sampled
  depth area. Fixed in `poi_perception/tracking/footpoint.py`.
- **Single-ankle raycast** now carries an explicit gait-bias uncertainty
  floor instead of treating a stride-length systematic as pixel noise.
- **Body-thickness offset** is now orientation-dependent (facing vs.
  side-profile) using shoulder keypoints, instead of one fixed constant.
- **Height** now corrects for which top-of-head keypoint was used
  (nose/eye/ear all sit below the actual crown) instead of a uniform
  ~10cm under-estimate.
- **RealSense capture** now enables `global_time_enabled` (required by
  the full plan's timestamp contract), the High Accuracy depth preset,
  full laser power, and a spatial+temporal+hole-filling post-processing
  chain — see `CameraConfig`'s RealSense-tuning fields and
  `capture/realsense_source.py`. None of this was previously set
  anywhere in the repo.


Three packages, one repo:

- **`poi_perception`** (Module 3) — raw RGB frame → `Detection2D` records
  with persistent 2D track IDs, a footpoint, and a torso polygon. No 3D,
  no map, no calibration.
- **`poi_localization`** (Module 4) — `Detection2D` + the synced depth
  frame → filtered, persistent 3D `WorldTrack` records with an honest
  uncertainty. Owns all 3D math and tracking-over-time logic.
- **`poi_present`** (Module 5) — `WorldTrack` records → a live WebSocket
  feed plus a 2D dashboard and a WebXR viewer. No 3D math of its own; it
  reads M4's already-computed positions and presents them. See
  [`PRESENT.md`](./PRESENT.md) for the full writeup.

`poi_localization` depends on `poi_perception` (it consumes
`Detection2D` and `Frame` directly) — that's the intended, documented
layering. `poi_perception` in turn never imports `poi_localization` or
`pyslam`; see each package's own README/module docstrings for why that
boundary is deliberate.

**New to this repo? Start with [`RUNBOOK.md`](./RUNBOOK.md)** — the
step-by-step sequence from "Orin + D435i plugged in" through M3, then
M4 Phase A, with a Phase A → Phase B migration note for once Module 2
exists.

## Quick start (no hardware required)

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

168 tests, all hardware-free: M3's synthetic capture + mock detections,
M4's synthetic ground-truth geometry (deprojection/ray-cast round-trips
against known transforms), its own mock-measurement track-manager
tests, a covariance-honesty (NEES) regression check driven by a
physically-realistic D435i error model, and M5's 33 (schema, adapter,
health, client queues, sources, and a real end-to-end server test over
a socket). No camera, no TensorRT, no GPU needed for any of it.

Try the two visual demos:

```bash
# M3 only: bbox/skeleton/footpoint over a scripted scenario
python3 scripts/demo_synthetic.py --scenario two_crossing --frames 90

# M3+M4 together: prints world-track positions per frame
python3 scripts/demo_synthetic_m4.py --scenario two_crossing --frames 90

# M3+M4 together, live visual: camera feed + bird's-eye-view of tracked
# world positions, side by side
python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.fixed.example.json --source synthetic --mock-scenario two_crossing
```

## Phase A vs. Phase B (Module 4)

Module 4's design doc structures the whole module around this split, and
the code follows it exactly:

- **Phase A** (build and validate now): a local, camera-relative floor
  frame, computed once at startup from the D435i's own IMU (gravity) and
  depth (RANSAC floor fit). No map, no calibration file, needed.
- **Phase B** (later, once Module 2 exists): the same code, reading
  `T_room_cam` from Module 2's calibration file instead.

The switch is `M4Config.phase = "A"` vs. `"B"` in one JSON config file —
see `configs/m4.phaseA.example.json` / `configs/m4.phaseB.example.json`.
**No code changes, no regeneration.** Everything downstream of
`frame_provider.build_frame_provider()` (measurement, tracking, output
schema field names aside — Phase A logs `p_local`/`v_local`, Phase B
logs the real `poi.v1` `p`/`v`, deliberately different field names so a
Phase-A log can never be mistaken for a room-frame position) is
identical code path for both phases.

Three ready-to-use M4 configs:

| Config | Phase | What it needs |
|---|---|---|
| `configs/m4.phaseA.example.json` | A | A D435i — does a live IMU+depth calibration capture at startup |
| `configs/m4.phaseA.fixed.example.json` | A | Nothing — skips live calibration, uses a fixed (height, pitch, yaw) you already know |
| `configs/m4.phaseB.example.json` | B | Module 2's calibration file (doesn't exist yet in this build) |

## Running for real

```bash
# M3 alone
python3 scripts/run_m3.py --config configs/m3.example.json --source realsense

# M3+M4 together (recommended once you want 3D positions)
python3 scripts/run_m4.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --source realsense
```

`run_m4.py` runs M3's daemon unmodified (same capture/inference threads,
same JSONL Detection2D logging) and chains M4's processing through M3's
existing `on_detections` callback hook — M4 was integrated without
touching a single line of M3's code.

## Module 5 — dashboard + WebXR viewer

```bash
# No hardware -- fake walkers, fastest way to try the viewers:
python3 scripts/run_present.py --config configs/present.example.json \
    --source synthetic --synthetic-mode scripted

# Live, on the Orin, next to M3+M4 -- zero changes to poi_localization
# or poi_perception (see PRESENT.md for how):
python3 scripts/run_present.py --config configs/present.example.json \
    --source live --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --m4-source realsense
```

Then open `http://<this machine's IP>:8765/` for the 2D dashboard or
the WebXR viewer. Same integration pattern as M3→M4 above: M5 attaches
to `M4Pipeline`'s existing (previously unused) `on_tracks` callback
rather than changing anything upstream. Full writeup, wire protocol,
and security notes in [`PRESENT.md`](./PRESENT.md).

## Live debug visualizers (not part of either pipeline)

- `scripts/debug_visualize.py` — M3 only: bbox/skeleton/footpoint drawn
  over the live camera feed.
- `scripts/debug_visualize_world.py` — M3+M4: the same camera-feed panel
  alongside a **bird's-eye (top-down) view** of the floor frame — each
  tracked person as a colored dot with a fading trail and a 2σ
  uncertainty ellipse, the camera drawn at the origin with its FOV
  wedge, and (with `--ground-truth`) your tape-measured floor markers
  overlaid as crosses so you can stand on a known point and see at a
  glance how far off the dot is. This is the quick visual sanity check
  for Phase A accuracy; `poi_localization/eval/floor_markers/` is the
  actual numeric acceptance test (Section 11's "done when" thresholds).

Both need a GUI-capable OpenCV build to show a live window —
`requirements.txt` installs `opencv-python-headless`, which can't. See
`RUNBOOK.md` Step 8 for the fix, or use `--no-window --save-dir` to get
annotated JPEGs instead.

## Evaluation

- `poi_perception/eval/` — Section 9's scripted scenario set (M3): 6 of
  7 scenarios are automated as mock-mode regression tests; run the real
  7th (low light) on hardware and grade by hand per its README.
- `poi_localization/eval/floor_markers/` — Section 11's primary Phase A
  acceptance test: tape-measure 10+ floor markers, capture with
  `scripts/floor_marker_capture.py`, compare with
  `python3 -m poi_localization.eval.floor_markers.compare`.
- `poi_localization/eval/consistency.py` — covariance-honesty check, not
  just position accuracy: reports the median NEES (normalized estimation
  error squared) of a synthetic walk against the REAL measurement +
  tracking code, driven by a physically-realistic D435i error model.
  Position accuracy alone can pass while the reported uncertainty is
  silently wrong (this is exactly how the extrinsic-uncertainty gap was
  found -- 1deg of pitch miscalibration produced ~14cm real error
  against a covariance claiming ~1cm). Run
  `python3 -m poi_localization.eval.consistency --deg-err 1.0` to see it;
  `tests/test_consistency_nees.py` runs it in CI so an overconfidence
  regression fails automatically, without hardware.
- `poi_localization/eval/scenarios/` — Section 11's other scenarios
  (straight-line walk, depth dropout, occlusion, crossing, false-source)
  — hand-graded against the JSONL log + `eval/grade.py`'s countable stats.
- `poi_perception/eval/replay_diff.py` and
  `poi_localization/eval/replay_diff.py` — regression diff against a
  saved baseline summary, same discipline both modules.

## File layout

```
poi_perception/          # Module 3 (unchanged from its own build)
poi_localization/         # Module 4
  geometry.py               # pure math: deprojection, ray/floor intersection, covariance rotation
  config.py                  # M4Config -- every tunable the design doc calls a "guess"
  frames/
    types.py                   # FloorFrameTransform -- the shared Phase A/B shape
    local_floor_frame.py         # Phase A: IMU+depth RANSAC floor fit
    room_frame.py                 # Phase B: reads Module 2's calibration file
    frame_provider.py              # THE swappable "get current floor-frame transform" call
  measurement/
    depth_measurement.py           # Section 4: torso polygon -> position + covariance
    raycast_measurement.py          # Section 5: footpoint -> position + covariance
  tracking/
    kalman_track.py                  # Section 6: per-person constant-velocity filter
    height_estimator.py               # Section 7: smoothed height, separate from position
    gates.py                           # Section 8: chi-square, height-range, walkable-area
    track_manager.py                    # Section 9: lifecycle + world-frame re-association
  io/
    world_track.py                      # Section 10: Phase A / Phase B records + JSONL
  runtime/
    m4_pipeline.py                       # Section 13's pluggable stage
    m4_daemon.py                           # runs M3+M4 live, via M3's on_detections hook
  eval/
    grade.py, replay_diff.py               # regression tooling
    floor_markers/                          # Section 11's primary acceptance test
    scenarios/                               # Section 11's other scenarios
scripts/
  run_m3.py, run_m4.py                        # live daemons
  run_present.py                                # M5 live/replay/synthetic daemon
  demo_synthetic.py, demo_synthetic_m4.py       # hardware-free demos
  debug_visualize.py, debug_visualize_world.py   # debug-only live visualizers
  floor_marker_capture.py                         # M4 ground-truth capture tool
  bench.py, export_model.py                        # M3's model export/benchmark tools
poi_present/                                          # Module 5 -- see PRESENT.md
  schema.py, adapter.py, config.py                      # wire contract + M4-record adapter
  sources/                                                # live / replay / synthetic
  server/                                                  # WebSocket + HTTP publisher
web/                                                        # M5's dashboard (web/dashboard)
                                                              # and WebXR viewer (web/vr)
docs/schema_examples/                                        # golden poi.v1 messages
configs/                                              # example configs for all three modules
tests/                                                  # 168 tests, all three modules combined
```

## Dependencies

`requirements.txt` (numpy, scipy, opencv-python-headless, websockets,
psutil, pytest) covers everything hardware-free, all three modules.
`requirements-optional.txt` covers the Orin-specific pieces
(`pyrealsense2`, `ultralytics`, `tensorrt`/`pycuda`) — see `RUNBOOK.md`
for install order and the version-matching gotchas around TensorRT
specifically. Module 5's own dependency footprint (just `websockets` +
`psutil`, no Node/build step — three.js and fonts are vendored as
plain files under `web/common/vendor/`) is detailed in `PRESENT.md`.
