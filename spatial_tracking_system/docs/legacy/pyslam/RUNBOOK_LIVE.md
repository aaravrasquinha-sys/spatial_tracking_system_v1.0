# WP-LIVE Runbook

Extends `Runbook.md` and `README_ORIN.md` (unchanged) with what's new
for live, multi-process mapping. Read those first for base environment
setup (JetPack, librealsense with CUDA + RSUSB backend, GTSAM from
source, power mode) -- nothing there changes.

---

## 0. First thing to run, always

```bash
python3 -m pyslam.selftest              # base repo, 54/54, ~3 min, no hardware needed
python3 -m tests.gates.test_g_live      # WP-LIVE's own gates, 22/22, ~2-3 min, no hardware needed
```

If either is red, don't touch the camera -- see `Runbook.md`'s own
guidance (`env_probe.py`) and `SYSTEM_SUMMARY_LIVE.md` section 1 for
what WP-LIVE's own gates cover.

---

## 1. New dependencies

See `requirements_live.txt`. Short version: **nothing new is required**
for lockstep mode or the CPU dense fallback -- the multi-process IPC
layer is pure standard library (`multiprocessing`), and the iSAM2
backend reuses the same GTSAM the base repo already builds from source
(`README_ORIN.md` Part 4). `nvblox_torch`/`torch` are optional and only
needed to run the bounded nvblox evaluation (section 5).

---

## 2. Sensor pre-flight: tare calibration

Before any mapping run, verify the D435i's depth **scale** accuracy
(distinct from On-Chip Calibration, which only reduces noise -- see
`scripts/tare_calibration_check.py`'s own docstring):

```bash
# In realsense-viewer: More -> Tare Calibration, against a flat wall
# at a KNOWN, tape-measured distance (0.65-1.8m recommended range for
# the D435i). Then verify quantitatively:
python3 scripts/tare_calibration_check.py --distance-m 1.000
```

This uses the same `pyslam.mapping.scale_check.check_scale` function
the live mapping scale-verification feature uses -- one shared notion
of acceptable scale error, not a second ad-hoc threshold. If it fails,
re-run Tare Calibration and check again before mapping.

Also run the base repo's own IMU calibration check (`README_ORIN.md`'s
own note: "The D435i ships with its IMU uncalibrated from the
factory") before trusting gravity alignment or the tilt prior.

---

## 3. Capture profile

One shared profile (`pyslam.live.capture_profile.CaptureProfile`) is
meant to be used identically for mapping and for any later runtime
consumer, so depth geometry is consistent everywhere it matters (ICP,
plane fitting, the watchdog's reference depth). The mapping default
(`configs/capture_profile.mapping.json`) deliberately leaves hole-
filling **off** -- see that config's own comment for why (hole-filled
depth invents values at range/edge discontinuities, which is wrong for
anything geometric).

`run_live_map.py` loads this automatically (`--capture-profile` to
override) and writes a copy into the output bundle
(`<out>/capture_profile.json`) along with its content hash, so a
downstream consumer can detect (not silently tolerate) a profile
mismatch.

---

## 4. Running live mapping

```bash
# Live, multi-process (the real architecture -- default)
python3 run_live_map.py --realsense --out maps/site_001

# Also record a .bag while mapping live, for later lockstep replay
python3 run_live_map.py --realsense --out maps/site_001 --record data/bags/site_001.bag

# Replay a recorded bag through the SAME live multi-process architecture
python3 run_live_map.py --bag data/bags/site_001.bag --out maps/site_001_replay

# Lockstep mode: single-process, deterministic -- for debugging, or if
# 'spawn' turns out not to work on your machine (see section 4.1 below)
python3 run_live_map.py --bag data/bags/site_001.bag --out maps/site_001_replay --mode lockstep
```

While a live session runs, QA output streams to the log every second:

```
QA: frames=412 keyframes=38 lost=0 icp_fallback=2 max_speed=0.41m/s
```

**`Ctrl-C` stops cleanly** and locks whatever was captured (mirrors
`run_slam.py`'s own discipline) -- a live session is never lost to an
interrupt.

### 4.1 `spawn` vs `fork`

`run_live_map.py --mp-start-method spawn` is the default and the
documented production choice (`orchestrator.py`'s own module
docstring explains why: `fork` silently duplicates device/GPU handles
across the fork boundary). **This was not validated end-to-end on real
hardware in this project's development sandbox** -- confirm it works
on your actual Orin/workstation before relying on it. If it doesn't
(same class of restriction the dev sandbox hit -- see
`SYSTEM_SUMMARY_LIVE.md` section 4.1), pass `--mp-start-method fork` as
a fallback; it was fully validated end-to-end in the dev sandbox and
accepts a real but understood risk (device handle duplication) in
exchange.

### 4.2 What's NOT wired up yet for live mode

`--mode live`'s map-locking step is incomplete -- see
`SYSTEM_SUMMARY_LIVE.md` section 4.5. Today it writes the dense PLY and
an honest `session_summary.json` explaining what's missing (a full
`manifest.json` with `map_id`), rather than a silently incomplete
bundle. `--mode lockstep` IS fully wired (writes `manifest.json`,
`map_id`, the locked read-only bundle) -- use lockstep mode against a
recorded bag if you need a complete, locked bundle today.

---

## 5. Bounded nvblox evaluation

Approved scope: a strictly time-boxed spike before considering Open3D
CUDA. Run this on the target Orin Nano:

```bash
python3 -m pyslam.tools.nvblox_eval --day-budget 7
# ... re-run on subsequent days as needed, up to the budget ...
python3 -m pyslam.tools.nvblox_eval --show-report   # see accumulated findings any time
```

The script refuses to run further checks once the day-budget is used
up -- that's deliberate, forcing a go/no-go decision rather than
letting the spike sprawl. See `pyslam/tools/nvblox_eval.py`'s own
docstring and `SYSTEM_SUMMARY_LIVE.md` section 4.4 for exactly what
this script does and doesn't establish (it's a bounded feasibility
check, not a benchmark suite or an integration).

If the recommendation is GO, wiring `nvblox_torch` in is a follow-up
work package: implement the `DenseBackend` Protocol
(`pyslam/mapping/dense_voxel.py`) with an nvblox-backed class and swap
it into `dense_process.py` in place of `VoxelHashFuser` -- the
interface is already there for exactly this.

If the recommendation is NO-GO or inconclusive after the budget is
spent, `VoxelHashFuser` (already in use by default) is the fallback --
it works today, with the honestly-documented limitations in
`dense_voxel.py`'s own docstring.

---

## 6. Multi-process debugging tips

- Each process logs under its own logger name
  (`live.orchestrator`/`live.backend_process`/`live.dense_process`) --
  grep by name to isolate one process's output.
- `session.poll_qa()` (or the log stream `run_live_map.py` prints) is
  the tracker's own view; a stalled tracker with no QA updates for
  several seconds usually means the camera pipeline itself has an
  issue -- check with `env_probe.py --realsense` first, same as any
  other hardware problem in this project.
- If the backend process appears to hang: it drains `frame_queue` with
  a 1-second timeout by default (`QueueSensorSource`'s
  `get_timeout_s`) -- a genuinely stuck backend usually means Pipeline
  itself is stuck (see the base repo's own debugging guidance for
  that), not the IPC layer.
- Resource-tracker warnings about "leaked shared_memory objects" at
  process exit are a known, harmless side effect of this module's
  explicit-ownership workaround for a well-documented CPython
  `multiprocessing.shared_memory` bug (see `pyslam/live/ipc.py`'s
  `_untrack` docstring) -- they do not indicate an actual leak as long
  as `free_shm_array()` is being called by the designated last
  consumer (verified by `tests/gates/test_g_live.py`'s IPC checks,
  which explicitly assert a freed block is actually gone).

---

## 7. File map addendum

See `SYSTEM_SUMMARY_LIVE.md` section 2 for the full new-file list. Two
entry points to remember:

```
run_live_map.py                    live mapping (multi-process or lockstep)
scripts/tare_calibration_check.py  pre-flight depth-scale verification
pyslam/tools/nvblox_eval.py        bounded dense-backend evaluation
```
