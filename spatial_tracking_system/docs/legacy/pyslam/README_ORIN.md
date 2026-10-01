# pySPLAM on Jetson Orin Nano — from a blank box to a live run

This is the complete path from an unopened Jetson Orin Nano Developer
Kit (8GB) to this pipeline running live against a D435i. It assumes you
have an **HP Z440 (x86, Ubuntu)** available as a second machine — that
matters, because it's the fastest, most reliable flashing path (see
Part 1) and it's also the reference machine every accuracy claim in
`WP_J_Findings.md` was measured against.

Read `WP_J_Findings.md` first if you haven't — it says exactly what's
validated and what isn't. This document is the "how", that one is the
"what changed and what to trust."

---

## Part 0 — What you're starting with, what you'll end with

**Start:** an unopened Jetson Orin Nano Developer Kit box, a D435i, and
an HP Z440 running Ubuntu.

**End:** `python3 -m pyslam.selftest` green on the Orin, `env_probe.py
--realsense` confirming the camera, and `run_slam.py --realsense --imu`
producing a `runs/run_.../` directory with a trajectory and a map.

**Time budget:** flashing ~30 min, firmware update (if needed) ~20 min,
`setup_orin.sh deps` ~10 min, `setup_orin.sh librealsense` ~20 min,
`setup_orin.sh gtsam` ~30-60 min (this is the long one — it's compiling
a large C++ template-heavy library on an ARM SoC). Budget half a day
for the first pass, including reading the warnings as they scroll by.

---

## Part 1 — Fresh Ubuntu on the Orin Nano

### 1.1 Check the factory firmware version first

The Orin Nano dev kit ships with older firmware that is **not**
compatible with JetPack 6.x. Power it on with no microSD inserted,
press `Esc` at the NVIDIA splash to enter UEFI, and note the firmware
version shown.

- Firmware **≥ 36.0**: skip to 1.3, flash JetPack 6.2 directly.
- Firmware **< 36.0**: you need the two-stage update in 1.2 first.

### 1.2 Firmware update (only if needed)

1. Flash a **JetPack 5.1.3** microSD image first (Part 1.3's flashing
   method, just an older image) and boot from it once — this schedules
   a firmware update to happen on next power cycle.
2. Reboot. The board updates its firmware during this boot (QSPI
   update to 36.4.0-ish). Let it finish; don't interrupt power.
3. Swap in the **JetPack 6.2** microSD (1.3) and boot from it — this
   schedules the final firmware update to 36.4.3.
4. Reboot once more. After this, `/proc/device-tree/model` and
   `/etc/nv_tegra_release` should show L4T R36.4.x or newer (the target
   unit reports R36.5.2 and works), which is what `scripts/setup_orin.sh`
   and `env_probe.py`'s Jetson probe assume.

### 1.3 Flash JetPack 6.2

Two ways — pick based on what you have:

**A. SDK Manager, from the HP Z440 (recommended, since you have it):**
Requires the Z440 running Ubuntu 22.04 or 20.04.
1. Download **NVIDIA SDK Manager** on the Z440.
2. Put the Orin Nano into Recovery Mode (hold the Recovery button while
   powering on, or per the dev kit's carrier-board jumper instructions)
   and connect it to the Z440 via USB-C.
3. Run SDK Manager, select Jetson Orin Nano, JetPack 6.2, and flash. If
   you want the OS on an **NVMe SSD instead of microSD** (recommended —
   see 1.6), this is the method that supports that target directly.

**B. microSD-only, no host PC needed for flashing:**
1. Download the JetPack 6.2 **SD card image** from NVIDIA's Jetson
   download page.
2. Flash it to a microSD card (64GB+) with **balenaEtcher** (Windows,
   Mac, or Linux — the Z440 works fine for this step too).
3. Insert the card (gold contacts toward the heatsink) and boot.

Either way, the Orin Nano only outputs over **DisplayPort, not HDMI**
— use a DP monitor, or a DP→HDMI adapter/cable if that's what you have.

### 1.4 First boot / Ubuntu OOBE

Standard Ubuntu out-of-box setup: accept EULA, set locale/keyboard,
create a user, connect Wi-Fi or plug in Ethernet, set the timezone.
Nothing SLAM-specific here.

### 1.5 Update the OS and set the power mode

```bash
sudo apt-get update && sudo apt-get -y upgrade
./scripts/setup_orin.sh power          # shows THIS unit's power modes + which one to use (changes nothing)
./scripts/setup_orin.sh power apply    # switches to it and runs jetson_clocks
```

**Do not assume `nvpmodel -m 0` is MAXN.** An earlier version of this
guide said so; on the target Orin Nano (L4T R36.5.2) `sudo nvpmodel -q`
printed `15W` / `0` and `tegrastats` showed CPU pinned at 1497 MHz and
GPU at 611 MHz — i.e. every timing taken by following the old
instructions was measured in the power-limited mode. Mode IDs and names
differ between Jetson models and JetPack releases, so the `power` stage
reads `/etc/nvpmodel.conf` and picks the `MAXN*` mode **by name**
(preferring `MAXN_SUPER` where the unit has it). `env_probe.py` now
reports the same thing (`POWER:` line in its summary). After applying,
confirm in `tegrastats` that CPU/GPU MHz are well above the 15 W-mode
values before trusting any timing.

`jetson_clocks` holds clocks at maximum continuously. That's correct
for benchmarking and for live runs, but it raises idle power draw and
heat — don't leave it set unattended for days without checking your
enclosure's airflow. There's no separate step to undo it beyond a
reboot or `sudo jetson_clocks --restore <file>` if you saved a
baseline first.

### 1.6 Storage: strongly consider adding an NVMe SSD

A microSD card is fine to boot from but is the wrong place to write
`runs/run_*/` directories, LTM's SQLite WAL file, or recorded `.bag`
files — sustained random writes on microSD are slow and wear the card
out faster than you'd expect. If your dev kit carrier board has an
M.2 NVMe slot (most do), add an SSD and either:
- flash JetPack directly to it via SDK Manager (1.3A), or
- boot from microSD and mount the NVMe at `/home` or a dedicated
  `/data` path, and point `run_slam.py`'s output/record paths there.

The 8GB RAM variant also benefits from this for another reason: see
1.7.

### 1.7 Swap space

8GB is tight for compiling GTSAM (a large templated C++ codebase) with
a full `-j$(nproc)` parallel build. If the `setup_orin.sh gtsam` stage
gets OOM-killed partway through, either:
```bash
# reduce parallelism (slower, less memory pressure)
# edit scripts/setup_orin.sh, or export JOBS=2 before running it

# OR add swap (needs the NVMe/microSD headroom from 1.6)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
Swap on microSD works but is slow and adds wear — prefer NVMe swap if
you have the SSD from 1.6.

---

## Part 2 — System and Python dependencies

Everything in this part is scripted (`scripts/setup_orin.sh`), but
here's what it actually installs and why, so you're not running a
black box.

### 2.1 apt packages
`build-essential cmake git pkg-config` (build tooling) · `libboost-all-dev
libtbb-dev` (GTSAM's dependencies) · `python3-dev python3-pip
python3-venv` · `libgtk-3-dev libcanberra-gtk3-module` (OpenCV GUI,
optional but cheap to have) · `libssl-dev libusb-1.0-0-dev
libudev-dev` (librealsense USB access) · `libglfw3-dev libgl1-mesa-dev
libglu1-mesa-dev` (librealsense's example viewers — not required by
this pipeline, but part of its default build).

### 2.2 Python packages
`numpy scipy scikit-learn opencv-contrib-python` — same set the x86
reference machine uses, so the code path is identical. **Do not** `pip
install gtsam` or `pip install pyrealsense2` — both get built from
source in Part 3/4 specifically so they link against *this*
environment's numpy instead of pulling their own pinned versions.

### 2.3 Should you use a virtualenv?

Optional, but if you do, create it with `--system-site-packages`:
```bash
python3 -m venv --system-site-packages ~/pyslam_venv
```
JetPack's OpenCV and any TensorRT/CUDA Python bindings are tied to the
system Python interpreter and are non-trivial to rebuild inside an
isolated venv. `--system-site-packages` lets the venv see them while
still letting you `pip install` project-specific packages on top
without touching the system interpreter.

Run:
```bash
cd pyslam_repo/
chmod +x scripts/setup_orin.sh
./scripts/setup_orin.sh deps
```

---

## Part 3 — librealsense (with CUDA)

```bash
./scripts/setup_orin.sh librealsense
```

What this does and why:
- Builds from source with `-DBUILD_WITH_CUDA=ON` — this is what moves
  `rs.align()` (depth-to-color alignment, called every frame) onto the
  GPU. Same output, just off the CPU.
- Builds with `-DFORCE_RSUSB_BACKEND=ON`. Intel's x86 build normally
  uses a kernel-patched UVC driver; Jetson's kernel doesn't have that
  patch, so librealsense's own userspace USB backend (RSUSB) is the
  supported path here. This is a **different code path from what was
  validated on the x86 reference machine** — see Part 6's note on this.
- Installs `udev` rules so the D435i works without `sudo`.

**Plug in the D435i to this stage's USB-C/USB-3 port directly** — not
through an unpowered hub, and note that the dev kit's USB-C port and
its USB-A 3.x ports are on different internal controllers with
different bandwidth headroom; if you see USB errors or dropped frames,
try the other port before debugging anything in software.

After this stage:
```bash
python3 -c "import pyrealsense2; print(pyrealsense2.__version__)"
python3 -m pyslam.tools.env_probe --realsense
```
Check `imu_callback_mode_supported` and `color_format_used` in the
output — those are exactly the two things Part 6's WP-J3 capture-path
changes depend on.

---

## Part 4 — GTSAM (from source, against your numpy)

```bash
./scripts/setup_orin.sh gtsam
```

This is the longest step (30-60 min on 6 cores). The reason it's built
from source rather than `pip install gtsam`: the PyPI wheel pulls a
`numpy<2` pin, which would silently downgrade the numpy every other
part of this codebase (every gate, `hnswlib`, everything) depends on.
Building from source with `-DGTSAM_PYTHON_VERSION` pinned to your
interpreter links it against whatever numpy is already installed
instead.

After this stage:
```bash
python3 -c "import gtsam; print(gtsam.__version__)"
python3 -c "import numpy; print(numpy.__version__)"   # must STILL be >=2.0
```
If numpy shows `<2.0` after this, something in the build pulled a
wheel instead of using your source build — check `pip show gtsam`
points at a local path, not PyPI, before going further.

With GTSAM now available, `Pipeline(..., backend_prefer="auto")` will
use GTSAM's `LevenbergMarquardtOptimizer` as the primary backend
instead of the native/scipy fallback — this closes the gap flagged in
`WP_P4_Findings.md` ("iSAM2 NOT attempted... numpy conflict").

---

## Part 5 — Get the code onto the Orin, and verify

```bash
git clone <your repo remote>   # or scp/rsync the repo from the Z440
cd pyslam_repo
git checkout jetson-orin-port   # the branch with everything in this README
./scripts/setup_orin.sh probe
python3 -m pyslam.selftest
```

`selftest` should show the same number it shows on the x86 reference
machine (54/54 as of WP-M -- check `SYSTEM_SUMMARY.md` for the current
total, since it grows with every work package) — the synthetic gate
suite has no hardware dependency, so a
failure here means an environment problem (numpy/opencv/gtsam version
mismatch), not a real algorithmic regression. If it's red, don't touch
the camera — go to `env_probe.py`'s output instead.

Also run the slower gates at least once on this machine, since they
were **not** re-validated on real Orin hardware in this session (only
against the x86 container that built this port):
```bash
python3 -m tests.gates.test_g2
python3 -m tests.gates.test_g3    # slow, several minutes
python3 -m tests.gates.test_g4
python3 -m tests.gates.test_g5
python3 -m tests.gates.test_g_traj
```

---

## Part 6 — First live run

```bash
python3 -m pyslam.tools.env_probe --realsense   # confirm the camera, one more time
python3 run_slam.py --realsense --imu --imu-mode synced --max-frames 300
```

**Start with `--imu-mode synced`, not the new default `callback`, for
the very first run.** `synced` is the exact behavior that was already
validated against real D435i hardware before this port (Phase 0's
G0-HW gate). `callback` is new, untested-on-hardware code (see
`WP_J_Findings.md`) — try it second, and compare:
```bash
python3 run_slam.py --realsense --imu --imu-mode callback --max-frames 300
```
Check `runs/run_.../summary.json` and `trajectory_report.json` from
both — trajectories should be near-identical; if `callback` mode logs
a fallback warning to `synced`, that's the code doing its job safely,
not a crash.

Once you're satisfied both paths work, record the 4 reference bags
**on the Orin itself** (not copied from the Z440 — RSUSB-backend
timestamps can behave slightly differently from the x86 kernel-patched
backend the original bags were recorded with):
```bash
mkdir -p data/bags
python3 run_slam.py --realsense --imu --record data/bags/desk_static.bag --max-frames 1800
python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
python3 run_slam.py --realsense --imu --record data/bags/corridor_out_back.bag
python3 run_slam.py --realsense --imu --record data/bags/hard_case.bag
```
From here on, iterate against these bags (`run_bag.py`) rather than
the live camera, same principle the architecture doc establishes for
the x86 workflow.

---

## Things to keep in mind while deploying

- **Put the unit in its MAXN power mode + `jetson_clocks` before trusting
  any timing number** (`./scripts/setup_orin.sh power apply` — see 1.5;
  mode 0 is NOT MAXN on the Orin Nano dev kit). A run measured at 15 W
  clocks isn't comparable to one at MAXN, and isn't comparable to
  anything on the x86 reference machine either.
- **Don't `pip install gtsam` or `pip install pyrealsense2` later**,
  even by accident (e.g. a stray `pip install -r requirements.txt`
  that lists them) — it'll silently replace your from-source builds
  and reintroduce the numpy<2 problem. Consider `pip freeze` right
  after Part 4/5 as a known-good baseline to diff future installs
  against.
- **Thermal throttling changes timing, and timing drives the WM
  budget controller.** If the Orin is in a poorly-ventilated enclosure,
  check `tegrastats` during a long run; `bundle.py` output should
  ideally snapshot this (not yet wired in — see `WP_J_Findings.md`'s
  open items) so note it manually for now if a run behaves oddly.
- **The D435i ships with its IMU uncalibrated from the factory.** If
  gravity alignment or the tilt prior look wrong, run Intel's IMU
  calibration tool before trusting `gravity_prior_enabled` or
  `gravity_frame.py`'s output — `env_probe.py --realsense` prints a
  reminder of this every time.
- **`wm_budget_ms` needs retuning for this machine, not inherited from
  x86.** WP-J2 fixed *what* it measures (memory-management time only,
  not frontend time), but the right *setpoint* is still whatever this
  Orin's own `mem_duration_ms` distribution looks like under load —
  check `telemetry.json`'s `mem_duration_ms` field after a real run
  before assuming the default `50.0` is right here.
- **RSUSB backend, not the kernel-patched UVC path.** If you see
  intermittent frame drops or USB errors that never showed up on the
  Z440's bags, this backend difference is the first thing to suspect,
  not the SLAM code.
- **8GB is shared between CPU and GPU (unified memory).** Each
  resident WM node keeps full RGB+depth (~1.5MB) until evicted; at
  WM~1000 that's ~1.5GB competing with CUDA allocations. Watch `free
  -h` / `tegrastats` during a long run, especially once GPU-offload
  work (Tier 3 of the port plan) lands.
- **No HDMI, DisplayPort only**, and **headless/SSH works fine** for
  everything in this doc except the very first OOBE setup (Part 1.4) —
  once the OS is configured you can do all of Parts 2-6 over SSH,
  which is usually more comfortable than working at the dev kit
  directly.

---

## System summary — what changed to port this from the HP Z440 (x86) to the Jetson Orin Nano (ARM)

Nothing about the **algorithm** changed. Every frozen contract in the
architecture doc (`Frame`, `Signature`, `Node`, `Link`, every Protocol)
is untouched, and every change below is either proven bit-identical,
proven numerically equivalent by a dedicated gate, or explicitly
flagged as unvalidated pending this machine. Full detail and rationale
for each is in `WP_J_Findings.md`; this is the condensed version.

| # | File(s) | What changed | Why it was needed for this platform | Validated how |
|---|---|---|---|---|
| WP-J1 | `pyslam/frontend/features.py` | `grid_bucket` rewritten from a per-keypoint Python loop to one vectorised `np.argsort` pass | Pure-Python per-keypoint loops are proportionally more expensive on the Orin's Cortex/Arm cores than on the Z440's x86 cores — this removes ~6ms/frame of Python-interpreter overhead that scales with keypoint count, not with any real algorithmic need | Bit-identical to the original, gated against real ORB output (12 trials) |
| WP-J2 | `pyslam/pipeline.py`, `pyslam/core/config.py` | WM eviction (`enforce_budget`) now driven by memory-management time only, not total frame time | The Z440 and the Orin have different frontend speed relative to memory/retrieval speed; feeding eviction the OLD whole-frame signal would make WM shrink at a different, wrong rate on this machine purely because the frontend timing ratio changed — not because memory pressure changed | New adversarial gate (120ms/frame artificial frontend slowdown → asserts zero spurious evictions); G2 5/5 |
| WP-J3 | `pyslam/sensors/realsense.py`, `run_slam.py` | Direct `rgb8` capture (was `bgr8` + flip-copy); new IMU callback-capture mode, independent of frame cadence, with automatic fallback | Removes a needless per-frame full-image copy; and a slower/loaded frontend (more likely on an embedded SoC under thermal pressure) was silently dropping IMU samples in the old synced-frameset capture path | **Not yet validated on real hardware** — written defensively with automatic fallback to the original, already-hardware-tested behavior; this is exactly what Part 6 above asks you to confirm first |
| WP-J0 | `pyslam/tools/env_probe.py` | New `probe_jetson()`: model/RAM/JetPack/CUDA/OpenCV-CUDA/power-mode reporting | The Z440 has none of this to report; the Orin needs its own bring-up diagnostics before any timing number means anything | Parsing logic verified against a synthetic fixture; the real subprocess calls are exercised for the first time when you run this on your unit |
| WP-J0 | `scripts/setup_orin.sh` (new) | Staged environment bring-up: librealsense w/ CUDA + RSUSB backend, GTSAM from source pinned to local numpy | The Z440 didn't need CUDA-accelerated align, doesn't use the RSUSB USB backend, and (per `WP_P4_Findings.md`) never got GTSAM built against numpy 2 in the first place — this closes that gap as part of the same effort | Syntax-checked only; every stage runs for the first time when you run it |

**Deliberately not changed yet** (next tiers of the port plan, not
started): two-stage retrieval (wiring the existing but unused
`bow_top_k`), the P6 threading pull-forward, and any GPU-offload beyond
librealsense's own CUDA align (CUDA ORB, CUDA Hamming matching). All of
these are accuracy-affecting or bigger structural changes that are
deliberately gated on having real numbers from *this* machine first —
see `WP_J_Findings.md`'s "What's still open" section.

---

## Part 7 — Phase A tooling (WP-K): what changed after the first hardware results

Full detail and rationale in `WP_K_Findings.md`. What you need day to day:

**Change any setting from the command line, no file editing** (all three
runners: `run_slam.py`, `run_bag.py`, `run_synth.py`):
```bash
python3 run_bag.py data/bags/room_loop.bag \
    --config-override proximity_enabled=true \
    --config-override odometry_backend=f2m
```
Values are type-checked and enumerated fields are validated
(`retrieval_backend=incremental` is rejected — the real value is
`bow_incremental`). The effective config is saved in the run's `config.json`,
and the overrides are listed in `summary.json`.

**Every run now finishes the same way** (`pyslam/tools/run_outputs.py`):
closing optimisation (`finalize`) → `map.ply` from the *final* poses →
trajectory/keyframe/g2o exports. `map.ply` used to be written before
`finalize`, and dropped every keyframe that had been evicted to LTM. Now
`summary.json` and `map_stats.json` state exactly how many keyframes made it
into the map (`n_nodes_in_map` / `n_nodes_total`). Evicted keyframes' imagery
is kept as lossless PNGs in `<run_dir>/imagery_cache/` for the duration of the
export and then deleted (`--keep-imagery-cache` keeps them, ~1 MB each). Set
`imagery_cache_enabled=false` to get the old behaviour.

**`summary.json` now has a `timing` block**: per-keyframe `mem_duration_ms`
percentiles, effective fps, max WM size, and how often keyframes exceeded
`wm_budget_ms`. This is the data `wm_budget_ms` must be retuned from
(see the WP-J2 note above) — read it from a real run on this machine.

**`trajectory_report.json`'s `height_range_m` is only height when
`height_range_valid` is true.** If gravity alignment failed, it is measured
along the first camera's optical axis (ordinary forward travel included).
Alignment needs a ~1.5–2 s completely still hold at the very start of the
run (`gravity_align_hold_s`); the synthetic fixtures do not provide one, so
their reports will always say "not aligned".

**Measure before you change anything else:**
```bash
python3 -m pyslam.tools.phase_a_baseline --label A5_base --config-override proximity_enabled=true
python3 -m pyslam.tools.phase_a_baseline --label A5_f2m  --config-override proximity_enabled=true \
                                         --config-override odometry_backend=f2m
python3 -m pyslam.tools.phase_a_baseline --compare baselines/A5_base.json baselines/A5_f2m.json
```
Each writes `baselines/<label>.json` + `.md` (after every run, so a crash
loses nothing). It reports what the old metrics could not: how many loop
closures are *real* (vs ground truth, and not just the last few seconds
re-matching itself), vertical vs horizontal error against ground truth, how
far the camera really moved across each LOST "bridge", map coverage, and
`mem_duration_ms` percentiles. Keep `--backend` identical between runs you
compare.


---
## Module 2 (static-camera anchor)

Localising a fixed D435i inside the module 1 map (`T_room_cam`, sigma, walkable grid for M3/M4/M5):
see `RUNBOOK_ANCHOR.md` (procedure) and `SYSTEM_SUMMARY_ANCHOR.md` (design, honest status, what is not
validated). Entry point: `python3 run_anchor.py {prepare,capture,solve,calibrate,markers,check}`.
Tests: `python3 -m tests.gates.test_g_anchor`.
