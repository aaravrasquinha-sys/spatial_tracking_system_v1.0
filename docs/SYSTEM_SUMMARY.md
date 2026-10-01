# Spatial Tracking System — System Summary (v1.0.0)

Single Intel RealSense **D435i**, single room, **Jetson Orin Nano**: a person-tracking system that places every person on
the floor of a metric, floor-referenced room frame and renders them live in a 2D dashboard and a WebXR (Quest) viewer.

This repository merges two code bases that were built to coexist:

* `pyslam_live_wp_anchor.zip` → **M1** (master map) and **M2** (static-camera anchor) — package `pyslam`
* `spatial_tracking_system_m3_m4_m5.zip` → **M3** (2D perception), **M4** (3D localization), **M5** (presentation) —
  packages `poi_perception`, `poi_localization`, `poi_present`
* new: **M6**, the orchestration/runtime layer — package `sts` — plus `contracts/`, integration tests, deployment files, docs.

The original planning document's Module 0 (capture rig) is procedure, not code; it is covered in the runbook.

---

## 1. Honest status — read this first

| | |
|---|---|
| **Verified in a sandbox (x86, Python 3.12, no GPU, no camera)** | `pyslam.selftest` 54/54 · `tests.gates.test_g_live` 22/22 · `tests.gates.test_g_anchor` 20/20 · `pytest tests/unit` 168/168 · `pytest tests/integration` 108 passed (107 fast + 1 slow) + 1 expected failure (K1) |
| **Run history** | selftest, G-LIVE and G-ANCHOR were run on the merged tree before the last doc/test edits (pyslam is byte-identical since); unit + integration were re-run at the end. Re-run everything with `python -m sts test all` on your machine |
| **Real code, real data flow, no mocks of M1/M2 internals** | The integration tests build a synthetic room, write a map bundle with M1's own hashing, solve it with **M2's real solver and gates** (accepted, sigma 14 mm / 0.26°), and run the consistency gate, M4's loader, M5's scene, the watchdog and the full M3→M4→M5 chain over those real files |
| **NOT verified — never run** | A real D435i · a Jetson · TensorRT / pycuda / pyrealsense2 (RSUSB) / GTSAM · the VR viewer on a headset · systemd units · any timing or accuracy number on real data · M2 on a real map |
| **Consequence** | Every threshold, noise constant and "expect" number in the docs is either from the synthetic room or chosen, not measured. The first real sessions are validation exercises. `docs/OPEN_ITEMS.md` lists each one with the place it plugs in |

Two defects in the uploaded legacy code were found and are pinned by tests rather than hidden (§9): `run_live_map.py --mode
lockstep` calls an undefined function, and in fork mode it hardcodes intrinsics.

---

## 2. Architecture

```
   OFFLINE (occasionally)                                                  RUNTIME (Orin, beside the camera)
 ┌──────────┐ maps/<site>/   ┌──────────┐ anchors/<site>/<cam>/  ┌─────────────────────────────────────────────┐
 │ M1 map   │ dense/points   │ M2 anchor│ calibration · walkable │ D435i ─► M3 ─► M4 ─► M5 ─► dashboard / VR   │
 │ (pyslam) │ ─────────────► │ (pyslam) │ room_frame · reference │ Frame  Det2D  WorldTrack poi.v1             │
 └──────────┘ + profile      └──────────┘ depth · viewer cloud ─►│            ▲                               │
        ▲                          ▲                              │            └── watchdog ─► health / events │
        └─────────── sts: site.json · slots · camera lock · consistency gate · doctor · runtime · replay ──────┘
```

* **Two lives.** The offline side (map, anchor) produces two artefacts: a **map bundle** and an **anchor bundle** (calibration,
  walkable grid, reference depth, room-frame cloud). The runtime needs only the anchor bundle; **it never runs SLAM**.
* **The key simplification** (from the plan): the static camera *is* the D435i, so M2 is 3D-to-3D registration, M4 uses the
  camera's own depth as the primary measurement and footpoint ray-cast as the fallback, fused in one Kalman filter per person.
* **Runtime threads:** capture and inference threads (size-1 drop-stale queues) → track/M4 on the main loop → asyncio server
  thread → watchdog worker thread (single-flight). Target budget: capture ~33 ms, inference 10–25 ms, M4 a few ms, network a
  few ms, viewer interpolation 100 ms ⇒ ~150–200 ms capture-to-photon. **Not measured.**

### Frames, time, identity

| | |
|---|---|
| `map` (W_cam0) | M1's frame: first mapping camera's optical frame, y down, not gravity-aligned |
| `room` | Z up, fitted floor = z 0, X/Y from the dominant wall direction; derived by M2 from the map alone; `T_room_map` in `room_frame.json` |
| `cam` | the colour camera's optical frame (x right, y down, z forward); depth is aligned to colour |
| `local` | Phase A's provisional floor frame (origin below the camera, +X along its forward direction on the floor) |
| viewer | one fixed rotation `(x, y, z)_room → (x, z, −y)` at the scene root |
| Units / time | metres; capture time = librealsense global (host) time, float seconds since epoch; every message carries `t_capture` and `t_publish`; `health.clock = "device"` flags a failed global-time setup |
| Identity | `map_id` = 12-hex SHA-256 of `dense/points.ply` + `capture_profile.json`; calibrations and runtime config name it; the runtime refuses to run if they disagree |

---

## 3. Module summaries

### M0 — capture rig and mount (procedure; no code)
Remove the long-USB-cable blocker by mapping with a mobile rig (Orin + battery + D435i on the existing 1 m cable, run headless,
record a `.bag` over SSH) and, at deployment, **co-locating the Orin with the camera** (Ethernet/Wi-Fi out). Build a rigid,
repeatable mount (quick-release plate or printed dock with a detent): M2's accuracy depends on the camera returning to the same pose.
`sts doctor --camera` verifies a USB 3.x link. The legacy plan's `bag_to_tum.py` is **not built** — M1 consumes bags directly.

### M1 — master environment map (`pyslam`)
Multi-process live mapper (tracker + backend + dense) writing `maps/<site>/dense/points.ply`. `sts map` wraps it: forces
`--imu-mode callback`, **records a bag by default** (the only way to re-fuse a map with holes or rebuild with another pipeline),
passes the site capture profile and camera serial, and holds the camera lock. `sts map-lock` writes `manifest.json` + `map_id`
and makes the bundle read-only, **keeping the `map_id` rule** so anchors stay valid. Capture protocol: runbook §M1.

### M2 — static camera anchor (`pyslam.anchor`)
`prepare` (map-only: room frame, walkable grid, `top_down.png`) → `capture` ×3–5 (camera seated in its mount, room empty) →
`solve` (physics prior → correlative global search → robust multi-scale ICP → gates → bundle). Accepts only if every required
gate passes; writes `calibration.<cam>.json` (or `.REJECTED.json`). Checked independently by the tape-measure test
(`anchor markers`) and the visual overlay. Details: `docs/legacy/pyslam/SYSTEM_SUMMARY_ANCHOR.md`.

### M3 — 2D perception (`poi_perception`)
TensorRT FP16 `yolo11n-pose` + ByteTrack → `Detection2D` (box, keypoints, **footpoint from the ankles**, torso polygon for depth
sampling) with ROI masks for mirrors/TVs. Logs every frame as JSONL and clips around failures. `sts engine` builds the engine
**on the Orin** and records a manifest (JetPack, TensorRT, command).

### M4 — 3D localization and world tracking (`poi_localization`)
Depth measurement + ray-cast measurement, each with its own covariance **and the calibration's extrinsic uncertainty**,
χ²-gated, fused in a constant-velocity Kalman filter per person; lifecycle tentative → confirmed → coasting → lost; walkable-grid
and height gates. Phase A (provisional floor frame from the camera's IMU + own floor fit) needs no map; Phase B reads
`calibration.<cam>.json`. **Acceptance:** floor-marker test (10+ taped crosses: median error ≤ 10 cm within 4 m, ≤ 20 cm beyond).

### M5 — presentation (`poi_present`)
`poi.v1` WebSocket + HTTP on one port; 2D dashboard; WebXR viewer (life-size with teleport, or a 1:20 dollhouse; renders 100 ms in
the past and interpolates on `t_capture`). Token auth, optional TLS, per-client bounded queues.

### M6 — runtime, health, replay (`sts`)
* **`plan_run`** resolves `map_id`, generates and validates configs, runs the **consistency gate** (22 checks), writes a run manifest.
* **`build_chain`/`run_live`** assemble capture → M3 → M4 → M5 + watchdog; exit code 3 if the capture/inference pipeline dies
  (so systemd restarts it instead of serving a frozen stream).
* **Watchdog supervisor:** verifies calibration *before* going live (starts `suspect`, clears only after a passing depth check),
  then every 120 s compares live depth on stable reference pixels (cheap layer) and, when nobody has been tracked for 10 s, runs
  an ICP recheck (the only layer allowed to *clear* a suspect state; it is the only one that catches ~1.5° yaw nudges). A crashing
  check reads as `suspect`, never `ok`.
* **Degraded mode:** `on_suspect = "suppress"` (default) publishes **empty** `tracks` and `health.calib = "suspect"` plus a
  `calib_suspect` event; `"flag"` keeps publishing. M3/M4 keep running and logging either way.
* **Soak monitor** (`soak.jsonl`), **retention pruning**, **replay** (lockstep, deterministic) with baseline compare.

---

## 4. The `sts` command surface

```
python -m sts init | doctor [--camera] | configs | check | slots | provenance | boundaries | contracts
python -m sts map [--no-record] [-- extra run_live_map.py args]      (camera)   python -m sts map-lock
python -m sts anchor prepare|capture|solve|calibrate|markers|check   (capture/calibrate/check need the camera)
python -m sts accept | engine | run [--phase A|B] [--source realsense|synthetic] [--playback BAG]
python -m sts replay --bag BAG [--save-baseline F | --compare F]     python -m sts test [unit integration selftest g_live g_anchor all]
python -m sts retention [--apply]
```
Exit codes: 0 ok · 1 check failed · 2 refused to start (config/consistency) · 3 pipeline died · 4 camera busy.

---

## 5. Repository layout

```
pyslam/  poi_perception/  poi_localization/  poi_present/   legacy modules, verbatim (ONE file edited: see CHANGELOG)
sts/                          orchestration (M6)
web/                          dashboard + VR viewer (vendored three.js, no build step)
run_*.py relocalize.py        legacy M1/M2 entry points (repo root, byte-identical; they must STAY at the root -- see CHANGELOG)
scripts/                      legacy M3–M5 entry points + setup_orin.sh + tare_calibration_check.py
contracts/                    contract specs + JSON schemas + legacy_hashes.json (provenance baseline)
configs/                      example configs, capture_profile.mapping.json, site.example.json, camera_model example
deploy/                       systemd units
requirements/ pyproject.toml  one package set; native Orin stack documented in requirements/orin.txt
tests/{unit,integration,gates,synth,golden}   docs/ (this file, RUNBOOK, ARCHITECTURE, OPEN_ITEMS, COMPATIBILITY, legacy/)
data/  (git-ignored)          maps/ anchors/ captures/ recordings/ models/engines/ logs/ locks/ runtime/
```

---

## 6. Your Orin venv — what the `pip freeze` says (fix before anything else)

1. **Two `poi_perception` installs** (`poi-perception` from `pyslam_mod3` and `spatial-tracking-system` from `pyslam_mod4`);
   the older lacks the torso-polygon fix, the global-time/preset tuning and the extrinsic-uncertainty fix. Uninstall both, install this repo.
2. **`websockets` is not installed** — M5 cannot start.
3. `pytest-asyncio` missing (tests only). 4. numpy is 1.26.4 (correct for this venv; ignore the legacy "numpy ≥ 2").
5. `torch 2.14` with `cu13` wheels is not the JetPack build — keep it export-only; the runtime path never imports it.
6. `pyrealsense2 2.58.4` appears as a package — confirm it is not shadowing the RSUSB+CUDA build from `setup_orin.sh`.
`python -m sts doctor` checks all of these. Procedure: RUNBOOK Part 0.

---

## 7. What changed in the legacy code

Exactly **one** legacy module file differs from what you uploaded: `poi_present/server/app.py` (an additive, default-off hook for
the watchdog; full description in `docs/CHANGELOG.md`). Everything else is byte-identical: 198 of 199 module files, 12 scripts,
6 entry points (hashes of all 217 in `contracts/legacy_hashes.json`; `python -m sts provenance` lists whatever you change later).
One *test* file has a one-token path fix. The entry-point scripts keep working because they stay at the repo root next to
`configs/capture_profile.mapping.json`, where they already looked.

---

## 8. Tests — what each proves

| Suite | Count | Proves |
|---|---|---|
| `pyslam.selftest` | 54 | M1 base gates + trajectory export + mutation-sensitivity (includes `run_bag.py` dispatch — would fail if the entry points moved) |
| `tests.gates.test_g_live` / `test_g_anchor` | 22/22 / 20/20 | live mapping architecture; M2 oracle tests (accuracy, ambiguity refusal, degeneracy, map holes, IMU faults, watchdog) |
| `tests/unit` | 168/168 | M3/M4/M5 unchanged behaviour, hardware-free; includes the NEES covariance-honesty check |
| `tests/integration/test_boundaries` | import rules, with a negative test that the checker catches violations |
| `…/test_contracts` | golden wire examples + what **M2 actually wrote** validate against the schemas; **M4's own loader** reads M2's files; M4 refuses a foreign `map_id`; M5 scene consumes the anchor bundle |
| `…/test_site_and_configs` | defaults, unknown-key preservation, validation, per-camera paths, generated configs load through each module's strict loader, typos fail loudly, `camera_model` reaches M2 **and** M4 |
| `…/test_consistency` | the gate **blocks** each of 12 disagreements (foreign map_id, unaccepted/rejected/missing calibration, map edited after locking, M5 pointed at the map dir, wrong M4 id, missing walkable/reference…) and only warns where it should |
| `…/test_map_lock_and_slots` | locking **preserves** the `map_id`; manifest valid; idempotent; read-only; edited bundle refused; reserved slot names refused; an alternative implementation can be selected per site |
| `…/test_watchdog_runtime` | startup-suspect → ok; 4° yaw → suspect with **one** event and 3D suppressed; depth alone never clears it; a person-sized occluder does not trip it; a crashing check reads `suspect`; **ICP recheck catches 1.5° and clears only back at the calibrated pose** (slow) |
| `…/test_runtime_chain` | Phase A with no map; Phase B carries the real `map_id`; startup verification suppresses 3D; `flag` policy; inconsistent Phase B refuses **before** anything is built; run manifest; **deterministic replay**; baseline compare |
| `…/test_present_hook` | default behaviour unchanged; provider drives `health.calib` over a real socket; a broken provider never reads `ok`; events injected from another thread arrive and validate |
| `…/test_golden_stream` | frozen M3+M4 stream on 3 scenarios (id, state, src, position ±5 mm, covariance ±2%, height) |
| `…/test_cli_doctor_misc`, `test_camera_lock`, `test_docs`, `test_known_legacy_defects` | CLI exit codes, doctor, retention, provenance, camera lock incl. crashed-holder, docs↔code consistency, pinned legacy defects |

**Mutation evidence** (does the golden test notice a behaviour change?): body-thickness, process-noise and disparity-sigma
edits are caught; raycast-noise, height-EMA and χ²-gate edits are **not** (M3's clean mock scenarios never exercise them; M4's unit tests do). OPEN_ITEMS W2.

`sts test integration` excludes nothing; the ICP-recheck test is marked `slow` and takes ~2–5 min on one core.
The golden file should be regenerated **on the Orin** the first time (`python tools/make_golden.py`): aarch64 vs x86 float
differences can shift a gate decision by a frame.

---

## 9. Findings made while merging

| # | Finding | Handling |
|---|---|---|
| 1 | Two installs of `poi_perception` in the venv | `doctor` detects duplicates; install procedure removes both |
| 2 | `websockets` / `pytest-asyncio` undeclared | declared in `pyproject`/requirements |
| 3 | `run_anchor.py` and `run_live_map.py` resolve their default capture profile relative to their own directory; moving them silently changes the profile hash (degrading the map↔anchor gate) | kept at the repo root with the profile in `configs/`, so **no edit**; a test fails if they move |
| 4 | **`run_live_map.py --mode lockstep` calls `_lock_and_write()`, defined nowhere** (legacy docs say lockstep is fully wired) | `sts map` refuses that mode with an explanation; xfail test flips when fixed (K1) |
| 5 | fork mode hardcodes intrinsics; `fork` is the script default while the runbook says `spawn` | doctor compares the device to `site.json`; tests pin both (K2, K3) |
| 6 | Stock `run_present.py` passes `map_id=None` to the live source | `sts` passes the real id |
| 7 | Stock `run_present.py` keeps serving a frozen stream if capture dies | `sts run` exits 3 |
| 8 | `M4Pipeline.total_events` grows without bound | soak monitor trims it from outside |
| 9 | **`M3Daemon` drops frames by design**; any replay through it is non-deterministic | `run_frames` is a separate lockstep loop; replay determinism test |
| 10 | M3 dataclass default model size 640×640 vs a 640×480 camera/engine | generated configs use the camera size; `doctor` compares the engine manifest |
| 11 | M3/M4 config loaders are strict (`Cls(**raw)`) | generated configs are round-tripped through them |
| 12 | M2 and M4 each guess the same depth-noise model | `camera_model.<cam>.json` (null = no change) |
| 13 | A fresh watchdog has no runtime caller; M5's `_calib_state()` was a stub | supervisor + additive hook |
| 14 | Dashboard shows `calib_suspect` as `#-1` | documented (V1) |
| 15 | The runtime depth path (preset + filters) differs from the anchor's capture path; startup verification may flag a clean scene | documented risk (W5) with the first-run check |

---

## 10. Limits and where each one plugs in
See [`OPEN_ITEMS.md`](OPEN_ITEMS.md) (IDs H1–H11, K1–K3, V1, W1–W5, R1). Headlines: nothing validated on hardware; M2 needs an
empty room; no `room.glb`; single camera; no re-identification; the tilt watchdog layer cannot run at runtime.

## 11. Where to look when something is wrong

| Symptom | Look at |
|---|---|
| Won't start, exit 2 | the printed gate report (`python -m sts check`) — each FAIL names the file and the fix |
| Exit 4 | another mode holds the camera: `data/locks/camera.default.lock` holds the owner |
| `health.calib = suspect` | `data/logs/<site>/<cam>/events.jsonl` (reason), `soak.jsonl`, then `sts anchor check` |
| Tracks vanish but video is fine | `calib` state (suppress policy), then `world_tracks` vs `detections` logs |
| Positions offset by a constant | `anchors/<site>/<cam>/overlay.png` + tape check; then `report.json` |
| Anchor rejected | `report.json` — every gate with its value; `top_down.png` for ambiguity |
| Everything odd after an edit | `python -m sts provenance`, then `sts test integration`, then `sts replay --compare` |
| VR won't enter immersive mode | secure-context rule (HTTPS or `adb reverse`) — see RUNBOOK M5 |
