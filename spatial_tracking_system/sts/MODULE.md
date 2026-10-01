# sts — M6: orchestration, health, replay

The **only** package that imports every module. It contains no perception, SLAM, localization or presentation maths.

| File | Role |
|---|---|
| `site.py` | `site.json` (`SiteConfig`): cameras list, paths, per-module `overrides`, watchdog policy, retention, `slots`; defaults everywhere; unknown keys preserved |
| `configgen.py` | builds the per-camera M3/M4/M5 configs + M2 overrides, writes them to `data/runtime/<site>/<cam>/`, and **round-trips each through the module's own strict loader** so a typo fails here |
| `camera_model.py` | `configs/camera_model.<cam>.json`: one depth-noise model pushed into both M2 and M4 (null = leave module defaults) |
| `slots.py`, `registry.py`, `adapters/` | the slot table, registry, and today's implementations wrapped behind it |
| `consistency.py` | the Phase-B gate (22 checks): map == calibration == M4 == M5, walkable/reference/viewer assets present |
| `runtime.py` | `plan_run` (ids → configs → gate → manifest), `build_chain`, `run_live` (asyncio + M3 daemon), `run_frames` (**lockstep**, deterministic) |
| `watchdog_runtime.py`, `calib_health.py` | supervises `AnchorWatchdog` from the frame loop without blocking it; drives `health.calib` |
| `camera_lock.py` | one D435i owner at a time (flock) |
| `doctor.py` | preflight: imports/paths, duplicate installs, numpy/websockets, pyrealsense2/TensorRT/pycuda, Jetson power, engine manifest vs JetPack and camera size, generated configs, `--camera` hardware probe |
| `manifest.py`, `provenance.py` | run manifest; which legacy files you have modified |
| `boundaries.py`, `contracts.py` | import-boundary checker; JSON-schema validation |
| `replay.py`, `retention.py`, `cli.py` | replay + baseline compare; log pruning; `python -m sts …` |

* **Tests:** `tests/integration/` (see `docs/SYSTEM_SUMMARY.md` §8).
* **Rule for editing `sts`:** keep adapters thin. If you find yourself writing geometry here, it belongs in a module.
