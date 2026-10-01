# WP-ANCHOR: System Summary (Module 2: static-camera anchor)

Companion to `SYSTEM_SUMMARY_LIVE.md` (module 1, live mapping). Hardware procedure: `RUNBOOK_ANCHOR.md`.

## 0. Scope

Module 1 produces a map (`dense/points.ply`) in `W_cam0`, the first mapping camera's optical frame
(y down, not gravity-aligned). Module 2 answers: **where is a static D435i in that map, in a
floor-referenced room frame, and how sure are we?** The output feeds M4 (`T_room_cam`, sigma,
`walkable.json`) and M5 (room-frame point cloud). Module 1 is unchanged and runs separately.

Design position: this is a **3D-to-3D registration** problem, not monocular relocalisation. The old
`relocalize.py` is not used (it needs a `reloc_map` that live mode does not write). The camera measures
metric depth, the map is metric, gravity is measurable, so most degrees of freedom are known *before*
any matching.

## 1. Honest status (read this first)

| | |
|---|---|
| Validated | On a **synthetic analytic room** with known ground truth: accuracy, ambiguity handling, degeneracy handling, map holes, IMU faults, output contract, watchdog. See §4. |
| **NOT validated** | **Real hardware.** No real D435i capture, no real module 1 map, no tape-measure check has been run. The depth noise model, covariance inflation and every gate threshold are **chosen, not measured.** Expect to re-tune them from the first real sessions. |
| Deliverable state | Complete pipeline + CLI + tests + runbook. Not integrated into the STS M6 daemon. |
| Sensitive assumption | Real maps drift and bow; the synthetic map has 3 mm noise and none of that. Reported sigma on real data will be dominated by map quality, which the synthetic test does not exercise beyond one deliberately bowed-floor check. |

## 2. Pipeline

```
maps/<site>/dense/points.ply (W_cam0)                static capture (camera in mount)
        |                                                     |
 Stage 0  map prep                                     Stage 1  capture
   outlier removal (SOR)                                 warm-up, temporal MEDIAN depth + MAD,
   room frame: floor plane bounding cloud from           valid-fraction, K sub-medians,
     below -> Z up, floor z=0; yaw from wall             median RGB, resting accelerometer
     histogram + wall-plane refinement                   -> up vector (specific force, +up)
   walkable grid (STS schema)                                 |
        |                                          weighted query cloud (edge/hole/instability
        |                                          filters, 1/sigma^2 depth weights)
        |                                                     |
        |                                          Stage 2  physics prior (no map)
        |                                            roll/pitch: IMU up-vector
        |                                            height + 2nd roll/pitch: camera's own floor RANSAC
        |                                            -> leaves only x, y, yaw unknown
        |                                                     |
        +-------------------> Stage 3  global search (correlative, FFT over x,y at every yaw)
                                 top-K DISTINCT peaks, fine local refine
                                                     |
                              Stage 4  robust multi-scale point-to-plane ICP (6 DOF, NOT clamped
                                 to the IMU) from every peak; choose by ICP fitness/RMSE
                                 ambiguity check + colour tie-break; observability
                                                     |
                              Stage 5  sigma = sqrt(ICP^2 + repeatability^2 + map-floor^2)
                              Stage 6  gates (all must pass)  ->  ACCEPT / REJECT
                              Stage 7  bundle + watchdog
```

Key decisions and why:

* **Room frame from the map alone.** Live mode has no floor fit / walkable / manifest. The floor is the
  large near-horizontal plane that *bounds the cloud from below* (a tabletop has cloud beneath it; a
  floor does not). The up-prior (−y) only picks the sign and rejects non-horizontal planes and may be
  ~60° off. Yaw is the dominant wall direction (refined by wall planes; no orthogonality assumed; falls
  back to the mapping-start heading if no direction dominates).
* **Exhaustive search instead of a feature matcher.** After levelling, the unknown is a planar rigid
  motion, searched deterministically. This makes *ambiguity a measurable quantity*: symmetric rooms
  show up as a tie between distinct poses, and the calibration is **rejected** rather than confidently
  wrong.
* **ICP does not use the IMU as a constraint.** Roll/pitch/height from ICP are then compared against
  the accelerometer and the camera's own floor fit. Independent evidence, not circular.
* **Refuse rather than guess.** Every gate is a named number; a rejected run is written as
  `calibration.cam0.REJECTED.json`, which M4 cannot load by accident.
* **Sigma is what M4 trusts.** It goes into M4's `extrinsic_position_variance`. Better slightly
  pessimistic than overconfident; §5 says how far it is validated.

## 3. File map

New package `pyslam/anchor/`:

| File | Role |
|---|---|
| `config.py` | `AnchorConfig`: every threshold; `apply_overrides()` for `--set KEY=VAL` |
| `geo.py` | rot_z, pose_delta, voxel_downsample, SOR, PCA normals, RANSAC plane |
| `map_io.py` | PLY reader (M1's binary layout + ascii), `resolve_map_id` (uses M1's `compute_map_id`; manifest wins if present) |
| `room_frame.py` | floor / Z-up / yaw / walls; floor quadrant flatness; map accuracy floor |
| `walkable.py` | walkable grid in the room frame (floor-supported or low-sittable, minus tall obstacles) |
| `capture.py` | `StaticCapture` (npz); `build_capture` (pure); `capture_static` (**only hardware touchpoint**, via `RealSenseSource`) |
| `query_cloud.py` | stability/edge/range filters, deprojection, depth-noise weights |
| `physics_prior.py` | levelling rotation, camera-own floor fit (rejects wall slices, no-floor views) |
| `global_search.py` | likelihood-field raster, FFT correlative search, distinct peaks, two-stage refine |
| `icp.py` | `MapTarget`, robust ICP (Tukey, final-scale trimming, range-compressed weights), `evaluate` (covariance, observability), colour correlation |
| `render.py` | map to depth image, residual, overlay |
| `verify.py` | `Check` rows, `run_checks`, frustum coverage, wall-referenced marker check |
| `calibrate.py` | orchestrator, capture pooling, ambiguity/tie-break, repeatability, `CalibrationResult` |
| `bundle.py` | writes the anchor bundle (`calibration.cam0.json`, `room_frame.json`, `walkable.json`, `viewer/points.ply`, reference depth, report, overlays) |
| `watchdog.py` | tilt / depth / ICP-recheck layers; never auto-recalibrates |

Elsewhere: `run_anchor.py` (CLI: prepare / capture / solve / calibrate / markers / check),
`tests/synth/anchor_world.py` (analytic synthetic room), `tests/gates/test_g_anchor.py` (20 checks),
`RUNBOOK_ANCHOR.md`, this file. **No existing file's behaviour was modified.**

Output contract (checked in the tests): `calibration.cam0.json` carries `T_room_cam` (4×4, orthonormal,
det +1, `p_room = R p_cam + t`), `sigma{trans_m, rot_deg}`, `map_id`; `walkable.json` is
`{origin, resolution, grid}` with row = y, col = x. Verified in the tests to load through M4's own
`load_room_frame`. M5 discovery of `viewer/points.ply` is by directory convention; the point cloud is
written in the room frame; no `room.glb` is produced.

## 4. What the tests establish (synthetic room)

`python3 -m tests.gates.test_g_anchor`: **20/20 passed** in a final clean run (single CPU core, ~7 min). No regressions: `python3 -m tests.gates.test_g_live` **22/22**, `python3 -m pyslam.selftest` **54/54** (both re-run after all anchor changes; no existing file was modified).

What each safety property demonstrates (all synthetic):

* Accuracy: with an awkward (tilted, yawed, y-down) map frame, 40 cm of moved furniture and an unmapped
  stool, error ≈ 0.3 mm / 0.007°; reported sigma 14 mm / 0.26° ≥ actual.
* Symmetric room → rejected as ambiguous; operator hint → accepted; distinct wall colours → colour
  breaks the tie to the right pose (all four symmetric images considered).
* Map hole *across the view* → rejected; hole *behind the camera* → still calibrates correctly.
* Accelerometer 2° off → rejected by the roll/pitch cross-check; no IMU → rejected (`require_imu`) or
  accepted with a warning when waived.
* Single flat wall + floor → rejected (observability / uniqueness). No floor in view → refuses.
* ICP convergence: noise-free oracle recovers 0.07 mm / 0.0006° from 8 cm / 5° seeds; covariance mean
  NEES 0.02 (<6 dof expected), i.e. **not overconfident**. It is very pessimistic on synthetic data.
* Watchdog: a person-sized occluder is ignored; 0.6° tilt caught by the IMU layer; 4° yaw caught by
  the depth layer; **1.5° yaw is below the depth layer** and needs the ICP recheck (which measured it).

### Bugs the gate discipline caught while building

1. **ICP settled into a wrong minimum** (37 mm off from an 8 cm seed) because trimming the worst 12%
   at *every* scale discarded exactly the far walls that pin x/y/yaw, and 1/σ² weights (σ ∝ z²) let
   the near field dominate. Fix: trim only at the final scale; compress range weights for the
   optimisation (true σ is still used for covariance); wider first scale.
2. **Colour tie-break compared only two of four symmetric poses** and could pick a wrong image with
   confidence. Fix: colour-score every geometrically tied pose.
3. **"Floor" RANSAC accepted a horizontal slice through a wall** (negative camera height on a
   wall-facing view). Fix: minimum in-plane extent on all on-plane points; refuse with a message.
4. **Profile-match gate was trivially true** if the capture hash was read from the map. Fix: the
   capture records the profile it *declares*, independent of the map (with the caveat in §6).
5. Test-geometry bugs (markers out of view, insufficient pitch) were mine, not the code's.

## 5. Uncertainty: how much to trust `sigma`

* Composition: `sqrt(ICP² + repeatability² + map-floor²)`. ICP covariance is `(ΣJᵀWJ)⁻¹` scaled by
  `max(1, χ²/dof)` and a ×4 inflation for spatially correlated depth noise.
* The ×4 and the depth-noise multiplier (2×) are **engineering guesses**. On synthetic data they are
  ~50× too pessimistic in NEES terms (harmless for safety, but it costs the sigma ≤ 3 cm / 0.5° gate
  some headroom). On real data the ICP term is likely *less* pessimistic. The repeatability term
  from ≥2 independent captures is the most trustworthy number; take 3-5 sessions.
* Repeatability from sub-medians of a single capture shares systematic errors and understates the
  true session-to-session spread; the report labels which one it computed.
* The map's own flatness sets a floor on sigma (`map_trans_floor_m`, `map_rot_floor_deg`, grown when
  the floor quadrants disagree).

## 6. Known limitations / not built

Stated plainly, because the plan named several of these:

1. **Not validated on hardware** (§1).
2. **No FPFH+RANSAC global cross-check.** Open3D is not in this environment; the search is
   correlative-only. A second, independent global method would strengthen the ambiguity decision.
3. **No full coloured-ICP objective.** Colour is a *correlation tie-break* only.
4. **Person masking during capture** (using M3 detections) is not implemented. Capture requires an
   empty room; the temporal median tolerates brief walkers, not a person standing still.
5. **Depth-noise model not fitted empirically in the room** (the plan's "fit on a flat wall").
6. **Watchdog not integrated into the STS M6 daemon;** only a library + CLI one-shot. Small yaw nudges
   (~1.5°) are not caught by the cheap layers; schedule the ICP recheck.
7. **Mirrors / windows / glass** are only handled by generic stability filters, not modelled. A mirror
   can still create a consistently wrong surface; the tape check is the defence.
8. **Module 1 live-mode map lock is still incomplete** (SYSTEM_SUMMARY_LIVE §4.5): no manifest/map_id;
   Module 2 computes one from the PLY + capture profile, and any edit of those files invalidates it.
9. **Capture profile is identity-only**, `RealSenseSource` does not push it to the device (§RUNBOOK 1).
10. **Map holes** (M1's bounded keyframe queue can drop keyframes) are detected (coverage / fitness
    gates), not repaired. Re-mapping, or re-fusing from a `--record` bag, is the fix. *Open question:*
    was the accepted map made with `--mode live` alone or with `--record`?
11. **One camera.** `cam_id` is plumbed, but nothing checks consistency *between* cameras.
12. **Performance:** pure numpy, single-threaded here; ICP and normal estimation dominate (~20-30 s
    for the synthetic case on a laptop; the Orin Nano is not measured).
13. **`run_live_map.py`** was not in the uploaded zip; the `--imu-mode polling` mismatch in your pasted
    version is documented (RUNBOOK §0) but not patched here.

## 7. Where to look when something is wrong

`report.json` (every gate + value + config used + ambiguity table) → `overlay.png` /
`depth_residual.png` (does geometry agree?) → `top_down.png` (is the frame, wall numbering and
visible floor what you expect?) → tape check (`run_anchor.py markers`). The tape check is the only
one of these that does not rely on the map agreeing with itself.
