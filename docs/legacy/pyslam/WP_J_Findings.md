# WP-J0/J1/J2/J3 Findings: Jetson Orin Nano port (Tier 0/1 of 4)

Scope: bring-up tooling + the accuracy-neutral / numerically-equivalent
changes identified as highest-value-per-risk in the port plan, for an
8GB Jetson Orin Nano Developer Kit (assumed JetPack 6.2, L4T R36.4.x --
current as of this session; confirm with `scripts/setup_orin.sh probe`
against the actual unit). **Nothing in this session ran on Jetson
hardware** -- there is none available in this environment. Every change
below is either (a) validated here against the same synthetic gate suite
the rest of this project uses, or (b) explicitly flagged as
hardware-only and untested, per this project's own established honesty
convention (see every other WP_*_Findings.md).

## What was validated here (x86 container, synthetic fixtures)

**WP-J1 (`pyslam/frontend/features.py`): `grid_bucket` vectorised.**
Was a per-keypoint Python loop (dict-of-lists + one `list.sort(key=...)`
per grid cell) -- measured at ~6.4ms/frame profiling `square6dof`, the
second-largest pure-Python cost in the frontend after `associate_depth`
(already vectorised, per its own docstring). Rewritten as one stable
`np.argsort` grouping pass over all keypoints, followed by a loop over
grid CELLS only (<=48 by default config, not the ~900 pre-bucket
keypoints). Proven bit-identical to the original by construction (same
cell arithmetic, same stable-sort tie-breaking behaviour Python's
`list.sort(reverse=True)` guarantees, same first-occurrence cell-visit
order for the final truncation) and by a new gate,
`tests/gates/test_g0.py::test_grid_bucket_vectorized_matches_loop`,
which runs both implementations against real `cv2.ORB` output (not
synthetic points -- real response ties are exactly where a subtly
different stable-sort composition could diverge) over 12 trials and
asserts exact equality. **selftest: 26/26 (was 25/25).**

**WP-J2 (`pyslam/pipeline.py`, `pyslam/core/config.py`): the WM
transfer-under-budget controller decoupled from frontend/backend
speed.** `enforce_budget()` was being fed the WHOLE frame's
`duration_ms` -- capture, ORB extraction, PnP odometry, memory ops,
retrieval, verify, graph optimise, all of it lumped together. This
couples WM eviction pressure to costs eviction cannot affect: shrinking
WM doesn't make ORB extraction faster. On a platform whose frontend is
proportionally faster or slower than the reference machine -- exactly
what porting to different hardware produces -- that mis-coupling changes
how aggressively nodes get evicted to LTM, and therefore changes
loop-closure recall, for reasons that have nothing to do with actual
memory pressure. This is precisely the kind of "accuracy gets worse on
the new platform even though no logic changed" risk flagged in the port
plan.

Fix: split timing into `mem_duration_ms` -- 0.0 on a non-keyframe frame
(no memory-management work happens then); on a keyframe frame, real
elapsed time over `memory.add` + retrieval + verify + `graph.optimize`
specifically, i.e. the subset of per-frame work that actually scales
with WM size, which eviction CAN reduce. `enforce_budget()` now receives
`mem_duration_ms`, not total frame time. `Config.wm_budget_ms`'s
docstring updated to say so explicitly, including that it should be
retuned per platform from that platform's own measured
`mem_duration_ms` distribution, not from total per-frame latency.

New gate `tests/gates/test_g2.py::check_budget_decoupled_from_frontend_time`
(**G2.6**): monkeypatches `extract_signature` to sleep 120ms on every
frame (>> `wm_budget_ms=80ms`, entirely frontend-side, zero real memory
pressure on this small fixture) and asserts **zero** LTM transfers
result over 40 frames. Before this fix, that same test would have shown
transfers on nearly every keyframe, purely from the artificial frontend
slowdown. **G2: 5/5 (was 4/4).** `telemetry` entries also gained a
`mem_duration_ms` field alongside the existing `duration_ms`, so a run
directory now shows both numbers for diagnosis.

**Regression coverage for both of the above:** full `selftest` (26/26)
and `tests.gates.test_g2` (5/5) re-run after each change. `test_g3`,
`test_g4`, `test_g5` were **not** independently re-run this session --
`test_g3` alone measured >5 minutes against this environment's CPU per
this project's own prior findings (`WP_P3_Findings.md`), and none of
this session's changes touch the code paths those gates exercise
(`vpr/`, `proximity/`, `imu/` are untouched; the only pipeline.py change
is two timing markers bracketing existing calls, with the surrounding
control flow re-read and confirmed unchanged -- see the `else:
mem_duration_ms = 0.0` branch attaches to the same `if
self.odometry.last_keyframe_created:` as before). Run the full gate
suite on the target machine before trusting this claim past "low risk,"
per this project's own rule that nothing ships without the gate that
would catch it being wrong.

## What was written but NOT validated (needs the actual Jetson + D435i)

**WP-J0 (`pyslam/tools/env_probe.py`): `probe_jetson()`.** Reports
model, RAM, L4T/JetPack version, CUDA toolkit version, OpenCV CUDA
build status, and `nvpmodel`/`jetson_clocks` state, all degrading to
`None`/a `reason` field rather than raising. **Validated: (1) runs
clean on this x86 container, correctly reporting `is_jetson: false`;
(2) the file-parsing logic specifically (null-terminated
`/proc/device-tree/model`, `/etc/nv_tegra_release`) verified against a
synthetic fixture with a real null-terminated model file, confirming
the null-byte handling is correct.** NOT validated: `nvpmodel`,
`jetson_clocks`, `nvcc`, `dpkg-query nvidia-jetpack`/`nvidia-l4t-core`
subprocess calls, since none of those binaries exist in this container
-- their happy paths are unexercised. `probe_realsense_device()` also
extended to report rgb8-vs-bgr8 support and whether a combined
accel+gyro sensor exists (WP-J3's precondition); neither testable
without a connected device.

**WP-J3 (`pyslam/sensors/realsense.py`): capture-path changes.** Two
changes, both defensive with automatic fallback to the original Phase-0
behaviour, **neither exercised against real hardware**:

1. Request `rgb8` directly instead of `bgr8` + a per-frame
   `[:, :, ::-1].copy()` flip. Falls back to `bgr8` (+ the original
   flip) if `pipeline.start()` raises with rgb8 requested -- caught, not
   assumed away.
2. New `imu_capture_mode="callback"` (default): opens the motion sensor
   directly via `sensor.open()`/`sensor.start(callback)`, independent of
   `wait_for_frames()`'s cadence, instead of only draining whatever
   motion samples happen to be bundled into a synced frameset. This is
   the same class of fix as WP-J2 applied to sensor I/O instead of the
   memory controller: on a slower/loaded frontend, samples arriving
   between two `wait_for_frames()` calls were silently dropped by the
   SDK before ever reaching `_drain_motion`. The callback thread is
   librealsense's own SDK-managed thread (not this project's pipeline
   threading -- the architecture doc's "no threads before P6" rule is
   about *pyslam's own* tracking/mapping/optimiser concurrency, which
   this doesn't touch; `__iter__` stays single-threaded and
   synchronous). Falls back to `"synced"` mode automatically, with a
   full pipeline restart to recover a working motion-stream config, if
   opening the callback fails (older firmware, some RSUSB-backend
   configurations). `run_slam.py --imu-mode {callback,synced}` lets you
   force either for a direct on-device comparison.

Also added: `threading.Lock` around the shared accel/gyro buffers
(needed once a second thread can write them), `close()` now stops+closes
the directly-opened motion sensor before stopping the main pipeline.

**Validated:** module imports cleanly with `pyrealsense2` absent (this
container), syntax-checked, and the existing hardware-free gate
(`test_g0.py::test_imu_part_inference`, which exercises
`infer_imu_part` -- untouched by this session's changes) still passes
inside `selftest`. **NOT validated:** anything requiring an actual
device -- rgb8 stream acceptance, the fallback-to-bgr8 path, the
motion-sensor callback actually receiving frames, the
fallback-to-synced recovery path, or `close()`'s shutdown ordering under
real concurrent callback activity. **Run this on the target machine
before trusting it**, ideally first with `--imu-mode synced` (closest to
the already-hardware-tested Phase-0 behaviour) to confirm the rgb8
change alone is safe, then `--imu-mode callback` to exercise the new
path, comparing `env_probe.py --realsense`'s `imu_callback_mode_supported`
and `color_format_used` fields against what actually happened.

**WP-J0 (`scripts/setup_orin.sh`): bring-up script.** Builds
librealsense with `BUILD_WITH_CUDA=ON` and `FORCE_RSUSB_BACKEND=ON`
(Jetson doesn't ship the patched mainline kernel Intel's x86 UVC path
assumes), and builds GTSAM from source pinned to the target
interpreter's Python version specifically so it links against THIS
environment's numpy (>=2.0) instead of pulling the `numpy<2` pin that
blocked iSAM2 in `WP_P4_Findings.md`. **Syntax-checked (`bash -n`)
only** -- no stage has been run against real hardware. Split into
independently-callable stages (`deps`/`librealsense`/`gtsam`/`probe`)
specifically so a failure partway through GTSAM's ~30-60min build
doesn't force redoing librealsense too.

## What's still open (per the port plan, unchanged by this session)

Tier 2 (two-stage retrieval wiring via the already-existing
`bow_top_k`/incremental-vocabulary ANN index, iSAM2 switch-over now that
GTSAM builds against numpy 2, and the P6 threading pull-forward) and
Tier 3 (CUDA Hamming matcher, CUDA ORB) are not started. Tier 4
(accuracy-trading knobs) is deliberately last-resort and untouched.

## Immediate next step

Run `scripts/setup_orin.sh all` on the actual unit (in stages, reviewing
each), then `python3 -m pyslam.selftest` and the full gate suite
(`test_g2` through `test_g5`, `test_g_traj`) to get this project's first
real Orin-native numbers. Everything in Tier 2/3 of the port plan is
gated on having those numbers rather than my guesses.
