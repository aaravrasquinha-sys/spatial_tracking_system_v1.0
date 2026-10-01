# Architecture and the rules that keep every module editable

```
            OFFLINE (run occasionally)                                    RUNTIME (Orin, next to the camera)
  ┌─────────────┐   map bundle    ┌──────────────┐  anchor bundle  ┌───────────────────────────────────────────┐
  │ M1 pyslam   │ ──────────────► │ M2 pyslam.   │ ──────────────► │  capture ─► M3 ─► M4 ─► M5  ─► dashboard │
  │ live mapper │ maps/<site>/    │ anchor       │ anchors/<site>/ │  (Frame) (Det2D)(WorldTrack)(poi.v1) VR   │
  └─────────────┘                 └──────────────┘   <cam>/        │           ▲            ▲                  │
        ▲  camera lock                 ▲ camera lock                │           └─ watchdog ─┘  health/events   │
        └──────────────── sts: site.json · slots · consistency gate · doctor · runtime · replay ──────────────┘
```

The runtime **never runs SLAM**. It needs the calibration, the walkable grid, the reference depth, the intrinsics and
the room-frame cloud. Everything heavy happens offline, once.

## The four rules

**1. Modules only talk through contracts** ([`../contracts/`](../contracts/README.md)). Allowed imports:

| Package | May import | Must never import |
|---|---|---|
| `pyslam` (M1, M2) | itself | `poi_*`, `sts` |
| `poi_perception` (M3) | itself | `pyslam`, `poi_localization`, `poi_present`, `sts` |
| `poi_localization` (M4) | `poi_perception` (its `Frame` / `Detection2D` contracts) | `pyslam`, `poi_present`, `sts` |
| `poi_present` (M5) | `poi_localization`, `poi_perception` (record types) | `pyslam`, `sts` |
| `sts` | everything | (nothing imports `sts`) |

`tests/integration/test_boundaries.py` parses every file and fails on any forbidden edge. **This one test is what lets you
edit a module without accidentally coupling it to another.** `python -m sts boundaries` runs it by hand.

**2. Existing code is wrapped, not refactored.** `sts/adapters/` presents today's implementations behind slot interfaces.
An improvement goes either *inside* a module (keeping its contract; nothing else changes) or as a *new implementation
registered in a slot*, selected per site in `site.json -> slots`. The old one stays available; `sts replay --compare` judges
the two on recorded data.

**3. Every new config field has a default.** `SiteConfig`, `M3Config`, `M4Config`, `PresentConfig`, `AnchorConfig` all load
partial JSON; `site.json` also preserves unknown keys. An old file never breaks after an upgrade.

**4. Contracts are versioned separately from code** ([`COMPATIBILITY.md`](COMPATIBILITY.md)).

## Slots (`sts/slots.py`; `python -m sts slots` lists them)

| Slot | Default (built) | Reserved names (planned, **refused** until built) | Contract |
|---|---|---|---|
| `map_builder` | `pyslam_live` | `rtabmap_reference`, `tsdf_rebuild_from_bag` | leaves a map bundle |
| `map_finalizer` | `manifest_lock` (+ `noop`) | `acceptance_suite` | must keep the `map_id` rule |
| `anchor_solver` | `pyslam_anchor` | `fpfh_cross_check` | produces the anchor bundle; rejected => `.REJECTED.json` |
| `anchor_capture_filter` | `none` | `m3_person_mask` | per-frame person mask for static capture |
| `viewer_asset_producer` | `points_only` | `room_glb` | files under `<anchor>/viewer/` in the room frame |
| `capture_source` | `m3_realsense` | `profile_enforcing_realsense`, `ir_stream` | `Frame`s |
| `detector_backend` | `m3_default` | `int8_experiment` | `PoseEstimator` backend interface |
| `frame_provider` | `m4_default` | `multi_camera`, `online_reanchor` | `FrameProvider` |
| `watchdog_layers` | `depth+icp` | -- | `+`-joined subset of `depth`, `icp`, `tilt` |
| `calib_state_provider` | `watchdog` | -- | `.state()` -> `ok/suspect/missing`, `.reason()` |
| `track_source` | `live` | `multi_camera_fusion` | `poi_present.sources.base.TrackSource` |
| `sensor_model` | `camera_model_file` | `fitted_from_flat_wall` | `CameraModel` pushed into BOTH M2 and M4 configs |

Registering a reserved name raises; selecting one in `site.json` fails with "reserved (planned, not built yet)". That is
deliberate: a config can never silently name something that does not exist.

### Adding an implementation (the whole procedure)

1. Write a class with the slot's contract anywhere (a new file in `sts/adapters/` is fine).
2. `reg.register("<slot>", "<name>", YourClass)` in `sts/adapters/*.register()`; if the name was in `reserved`, delete it
   from `sts/slots.py` in the same change.
3. `site.json -> "slots": {"<slot>": "<name>"}`.
4. Run the change workflow below.

## Workflow for changing a module

1. Edit inside the module.
2. Run that module's own suite (`python -m sts test unit`; pyslam: `python -m sts test selftest g_live g_anchor`).
3. `python -m sts test integration` — contract tests, import boundaries, the consistency gate, the golden stream.
4. `python -m sts replay --bag <real bag> --compare <baseline.json>` on recorded data from **your room**.
5. Intentional behaviour change? `python tools/make_golden.py`, store the new replay baseline, and write why in
   `docs/CHANGELOG.md`.
6. Changed a contract? Bump it in `COMPATIBILITY.md`, update **both** ends and their schema in the same commit.

`python -m sts provenance` lists exactly which legacy files differ from the versions merged here, and every run manifest
records that list, so a bug report names the modules you touched.

## Concurrency and ownership

* **One camera, one owner.** `sts map`, `sts anchor capture|calibrate|check`, `sts run` (live) and `sts doctor --camera`
  all take `data/locks/camera.<key>.lock` (`flock`; released by the kernel if the holder dies). A second mode is refused
  immediately with the holder's name and pid. Modes launched by calling the legacy scripts directly **bypass** the lock.
* **Live vs lockstep.** Live (`M3Daemon`): capture and inference threads with size-1 drop-stale queues; a fast source loses
  frames by design. Replay/regression (`sts.runtime.run_frames`): synchronous lockstep, nothing dropped, deterministic.
* **The watchdog never blocks the frame loop.** `observe()` copies one depth frame and decides whether a check is due; real
  work runs on one worker thread (single-flight).
* **Exit codes (for systemd):** 0 operator stop · 2 refused to start (configuration/consistency; do not restart-loop) ·
  3 capture/inference died on its own (restart) · 4 camera busy.
