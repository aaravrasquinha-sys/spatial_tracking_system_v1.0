# Phase 0 Runbook

## What this is

A complete, working, single-threaded RTAB-Map-style SLAM pipeline:
capture -> ORB features -> frame-to-keyframe PnP odometry -> STM/WM
memory -> retrieval -> Bayes filter -> geometric verification -> pose
graph (GTSAM/native) -> point-cloud map. Deliberately the simplest
correct version of every component (see architecture doc) so later
phases are swaps behind fixed interfaces, not rewrites.

## Dependencies

```
pip install numpy scipy scikit-learn opencv-contrib-python pyrealsense2
```

GTSAM: already installed on the target machine per your setup — used
automatically as the primary graph backend. If it's ever missing or
its `numpy<2.0` pin conflicts with the rest of the environment, the
pipeline **automatically falls back to a native backend** (no crash,
just a log line: `Using native pose-graph backend.`).

## First thing to run, always

```
python3 -m pyslam.selftest
```

~50s, no hardware needed. Runs the full synthetic gate suite (SE3
identities, renderer geometric oracle, Bayes filter spike-rejection and
sustained-match-fires checks, verifier false-accept check, and a full
end-to-end synthetic loop closure with an ATE check). **If this is
red, don't touch the camera** — go straight to `tools/env_probe.py`
instead.

Expected output: all checks passed (54 as of WP-M: 13 gate + 3 trajectory + 10 Phase-A + 10 Phase-B + 8 Phase-C + 10 mutation).

## Environment check (before first hardware run)

```
python3 -m pyslam.tools.env_probe --realsense
```

Confirms OpenCV/GTSAM/pyrealsense2 versions and symbol availability,
queries the connected D435i for firmware/serial/USB mode, and compares
its measured intrinsics against the expected rig values (fx=606.75,
fy=606.57, cx=320.19, cy=237.06, 640x480, depth_scale≈0.001,
baseline≈0.0499). A mismatch is a warning, not a crash — the pipeline
always uses whatever the device actually reports.

## Running on synthetic data (no camera)

```
python3 run_synth.py                       # flagship square-loop scenario (validated, passes)
python3 run_synth.py --scenario corridor   # harder scenario, known not to close the loop yet
```

## Running live on the D435i

```
python3 run_slam.py --realsense --imu
python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
python3 run_slam.py --realsense --imu --max-frames 300
```

IMU is captured and stored on every Frame from frame 0, but **not yet
fused into odometry** — that's a Phase-5 upgrade. Recording it now
means bags captured today are already IMU-ready for later.

**Record the 4 reference bags now, while the camera is out**, per the
architecture doc's workflow — after this, hardware mostly leaves the
iteration loop and everything below runs against replays:

| Bag | What to capture |
|---|---|
| `desk_static.bag` | 60s, camera stationary on a desk |
| `room_loop.bag` | 60-90s, walk a loop around a room, return to start |
| `corridor_out_back.bag` | 90s, down a corridor and back |
| `hard_case.bag` | blank wall / fast turn / low light |

```
mkdir -p data/bags
python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
```

## Replaying a recorded bag

```
python3 run_bag.py data/bags/room_loop.bag
python3 run_bag.py data/bags/room_loop.bag --max-frames 200
```

## Every run writes a `runs/run_<timestamp>/` directory

Contains `config.json` (exact settings used), `telemetry.json`
(per-frame timing/status), `summary.json` (keyframes, loop closures,
ATE if ground truth is available), and `map.ply` (open in MeshLab /
CloudCompare / any PLY viewer). If something looks wrong, zip and send
the whole directory:

```
python3 -m pyslam.tools.bundle runs/run_1234567890
```

## Known limitations in this phase (see Phase Evaluation doc for detail)

- **Performance is not real-time yet**: ~300ms/frame in this
  environment (offline, not the target machine — numbers will differ
  on yours; report what you see). Correctness was the Phase-0
  priority; a profiling/vectorization pass is the natural Phase-1
  companion to the odometry upgrade.
- **The corridor/open-area loop-closure scenario doesn't close the
  loop yet** — only the tighter square-loop scenario is validated.
  This is a real, understood limitation (weaker retrieval signal over
  longer time gaps + no graph-neighbor-aware belief diffusion yet),
  not a crash or a silent wrong-answer. Full detail in the Phase
  Evaluation doc.
- No threading (single-threaded by design, see architecture doc §7
  rule 1). No LTM (WM only, unbounded). Fixed vocabulary code exists
  but isn't used to drive loop-closure decisions in this phase (see
  `vpr/raw_match.py` docstring for why).

## Running on the Jetson Orin Nano (WP-J, in progress)

See `WP_J_Findings.md` for the full status -- summary here.

1. `./scripts/setup_orin.sh all` (in stages, reviewing each -- it's
   long: apt/pip deps, librealsense built from source with CUDA and the
   RSUSB backend, GTSAM built from source against this interpreter's
   numpy so it doesn't pull `numpy<2` the way `pip install gtsam` does).
2. `python3 -m pyslam.tools.env_probe --realsense`. Check the `jetson`
   block for RAM/JetPack/CUDA/OpenCV-CUDA and the `POWER:` line, then
   `./scripts/setup_orin.sh power apply` (mode 0 is NOT MAXN on this kit)
   (the script prints the exact commands) before any timing comparison
   -- numbers collected without these locked aren't comparable to
   anything.
3. `python3 -m pyslam.selftest`, same as always -- if red, don't touch
   the camera.
4. `python3 run_slam.py --realsense --imu` now defaults to
   `--imu-mode callback` (motion samples captured on their own SDK
   thread, independent of frame-capture cadence, instead of only being
   drained from synced framesets -- more robust against drops on a
   loaded frontend). Pass `--imu-mode synced` to force the original
   Phase-0 behaviour if you want to compare, or if callback mode logs a
   fallback warning on your specific unit.
5. Record the 4 reference bags per the section above, on the Jetson
   itself -- RSUSB-backend IMU timestamps can behave differently from an
   x86 recording, so Orin-side work should be replay-only from its own
   bags, same principle as the rest of this doc.

**Not yet done** (see WP_J_Findings.md "What's still open"): two-stage
retrieval, iSAM2, the P6 threading pull-forward, and any GPU-offload
work. The capture-path and env_probe changes above are written but
**unvalidated against real hardware** -- this is the first thing to
confirm, not something to build further on top of blind.



## Phase A additions (WP-K) -- see `WP_K_Findings.md`

- `--config-override KEY=VALUE` (repeatable) on `run_slam.py`, `run_bag.py`, `run_synth.py`.
- `map.ply` is now built after the closing optimisation and includes keyframes evicted to LTM;
  `summary.json` / `map_stats.json` report map coverage; `summary.json` has a `timing` block.
- `python3 -m pyslam.tools.phase_a_baseline --label X [--config-override ...]` and `--compare A.json B.json`.
- `./scripts/setup_orin.sh power [apply]` -- power mode by name, not by assumed ID.
