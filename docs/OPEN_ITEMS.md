# Open items — what is NOT solved, and exactly where each fix plugs in

Nothing here has been run on a Jetson or a real D435i. **Status legend:** `OPEN` (not done) · `MITIGATED` (handled in `sts`,
root cause remains) · `DEFECT` (bug in uploaded legacy code, pinned by a test).

Format: *seam* = the slot/file where the change lands · *also* = what else must change · *proof* = how you know it worked.
When you close one, record the slot used and the evidence in this file.

## A. Validation and calibration quality

| ID | Item | Status | Seam | Also must change | Proof |
|---|---|---|---|---|---|
| H1 | M2 has only met a synthetic room. Every gate threshold, the ICP covariance inflation (x4) and the depth-noise multiplier (x2) are **chosen, not measured** | OPEN | `site.json -> anchor.set` (`--set KEY=VAL`), `camera_model.<cam>.json` | nothing; `report.json` records the config used | tape check (`sts anchor markers`), 3–5 independent sessions (repeat spread), then NEES once real ground truth exists |
| H11 | Depth-noise model never fitted on a real wall | OPEN | `sensor_model` slot: fill `configs/camera_model.cam0.json` | nothing — `sts` pushes it into M2 **and** M4 configs | flat-wall fit; `consistency.py` NEES test still passes |
| W5 | The watchdog's reference depth comes from the anchor capture path (no runtime filters), while runtime depth goes through High-Accuracy preset + spatial + temporal + hole-filling. Startup verification may report `suspect` on a clean scene | OPEN (risk) | `watchdog.*` in `site.json`; `anchor.set wd_depth_move_m=…`; or re-capture the reference through the runtime source | none | first real `sts run`: `data/logs/<site>/<cam>/events.jsonl` shows `startup: X% … moved`; X should be small |
| H4 | Capture profile is *declared*, not pushed to the device by `pyslam`'s `RealSenseSource`; `capture_profile_match` therefore means "same file", not "verified on the sensor". Mapping/anchor and the runtime source also differ (hole-filling 0 vs 1) | OPEN | `capture_source` slot (reserved `profile_enforcing_realsense`): call `pyslam.live.capture_profile.apply_to_realsense` in both sources | `sts doctor` reads device options back | doctor reports no difference between readback and profile |
| H5 | No person masking during M2 capture — the room must be **empty** (a brief walker is tolerated by the temporal median; a person standing still is not) | OPEN | `anchor_capture_filter` slot (reserved `m3_person_mask`); optional `mask` arg on `build_capture` | only that optional argument | calibration with a person in view matches the empty-room result |
| H6 | No FPFH+RANSAC second global method; ambiguity rests on the correlative search alone | OPEN | `anchor_solver` (reserved `fpfh_cross_check`) writing a **non-required** row into `report.json`, promoted to required once trusted | add Open3D (check aarch64 wheel for Py3.10 first) | agrees with the correlative search on real captures |
| H8 | M2 solve speed on the Orin unmeasured (pure single-threaded numpy; ~13 s for the synthetic solve on an x86 sandbox core) | OPEN | inside `pyslam.anchor` | none (offline) | `tests.gates.test_g_anchor` 20/20 and golden bundle within tolerance |
| H7 | Watchdog thresholds untuned. The **tilt layer cannot run at runtime**: the runtime capture path has no IMU and a second accel pipeline under RSUSB is unvalidated, so `watchdog.tilt_layer` is `off` and `"auto"` is rejected | OPEN | `watchdog_layers` slot, `site.json -> watchdog.*`; inject a `tilt_source` in code to enable `external` | none | nudge test: small yaw, tilt and translation caught within one cycle |
| W4 | The ICP recheck lazily loads the map cloud into the runtime process (cropped to the camera's working volume). Time and memory on the Orin (8 GB shared with TensorRT) are unmeasured | OPEN | `sts.runtime.make_map_loader` | lower `recheck_frames`, decimate the loader | `soak.jsonl` rss stays flat across a recheck |

## B. Map (M1)

| ID | Item | Status | Seam | Also | Proof |
|---|---|---|---|---|---|
| H2 | `run_live_map.py --mode live` writes no manifest / lock | MITIGATED | `map_finalizer` (`sts map-lock`) | keeps the `map_id` rule | `resolve_map_id` -> `source=manifest.json`, `mismatch=False` |
| K1 | `run_live_map.py --mode lockstep` calls `_lock_and_write()`, **defined nowhere** -> `NameError` after mapping finished | DEFECT | fix in `run_live_map.py` (write the function, or drop the call) | `sts map` guard relaxes automatically | `tests/integration/test_known_legacy_defects.py` stops xfailing |
| K2 | In `live` + `fork` mode the script skips probing and hardcodes intrinsics `606.75/606.57/320.19/237.06` | DEFECT | `sts doctor --camera` compares the **device** to `site.json` | update both if your unit differs | doctor: "device intrinsics vs site.json" PASS |
| K3 | `--mp-start-method` defaults to `fork`; the legacy runbook says `spawn` is the default and recommends it. `spawn` was never validated on hardware | OPEN | `sts map -- --mp-start-method spawn` to try it | -- | a live run completes under `spawn` |
| H3 | No `room.glb` mesh; the viewer shows the point cloud | OPEN | `viewer_asset_producer` (reserved `room_glb`): write `anchors/<site>/<cam>/viewer/room.glb` **in the room frame** (use `T_room_map` from `room_frame.json`) | nothing — M5 already discovers `room.glb` and the VR viewer has a fallback chain | mesh and cloud overlap in VR; tracks stand on the mesh floor |
| H9 | Paused SLAM work: gyro bridge regression, f2m parity, corridor loops, full IMU fusion | OPEN | inside `pyslam`, behind `map_builder` | nothing downstream (M2 only sees the bundle) | `pyslam.selftest`, `g_live`, baselines, then a real-room take |

## C. Runtime

| ID | Item | Status | Seam | Also | Proof |
|---|---|---|---|---|---|
| W1 | Degraded mode publishes an **empty `tracks` array** while calibration is suspect; it does **not** stream 2D detections (no wire message exists for them) | OPEN | new `poi.v1` message type (additive) + `on_suspect` policy in `sts.runtime.Chain.on_tracks` | viewers must render it | suspect state shows flagged 2D boxes in the dashboard |
| V1 | Dashboard prints `calib_suspect` events as `#-1` (events carry `id=-1`) | OPEN (cosmetic) | `web/dashboard/app.js` line ~138 | -- | -- |
| W2 | The golden stream is sensitive to depth-path/tracker-dynamics changes (mutation-checked: body thickness, process noise, disparity sigma all caught) but **blind** to raycast noise, height EMA and the χ² gate, because M3's clean mock scenarios never exercise them. M4's own unit tests cover those | OPEN | add scenarios to `tools/make_golden.py` (e.g. dropout, ID churn) | regenerate golden | mutation check catches them |
| W3 | Phase A's live calibration (M4 opens its own short-lived RealSense pipeline, then the main source) is legacy design; two sequential opens under RSUSB are unvalidated | OPEN | `phase_a.fixed_transform` in `localization.overrides` skips it | -- | `sts run --phase A` starts twice in a row without a USB reset |
| H10 | One camera only: the chain refuses >1 enabled camera | OPEN | `frame_provider` (reserved `multi_camera`), `track_source` (`multi_camera_fusion`); data layout is already per camera | fusion in M4's world tracker | two cameras agree on one person within their combined sigma |
| R1 | Re-identification after leaving frame (new ID in v1) | OPEN | M4 tracker | -- | -- |
