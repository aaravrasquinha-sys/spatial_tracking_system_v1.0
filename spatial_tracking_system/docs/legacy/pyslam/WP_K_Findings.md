# WP-K Findings: Orin accuracy plan, Phase A (measurement + free wins)

Scope: after the first real Orin runs (selftest green, synthetic fixtures close to the x86 numbers, two
fixtures struggling), the goal became better accuracy and map detail. Phase A changes NO algorithm: it fixes
what was mis-measured or silently thrown away, and builds the instruments the later phases need. Everything
here was validated on the x86 dev sandbox (1 core, native pose-graph backend). **Nothing in this work package
has run on the Orin yet** -- see section 5.

---

## 1. Corrections to the earlier analysis (read this first)

The plan handed over before this work package contained three claims that were wrong or unmeasured:

1. **"corridor_v2 has ~7.8 m of vertical drift" -- wrong as stated.** `trajectory_report.json` reported
   `height_range_m = [-1.70, +6.13]` with `gravity_aligned: False`. When alignment fails `T_grav_cam0` is the
   identity, so that "Z" is the FIRST CAMERA'S OPTICAL AXIS -- ordinary forward travel down a corridor -- not
   height. The report now says so (`height_range_valid`, `height_range_note`, and a title on
   `trajectory_height.png`). Real vertical error is measured against ground truth instead (section 2).
2. **"bridge links shrink path_length_m"** was a guess. It is now measured, and it is true (section 2).
3. **"each LOST bridge skips ~0.3-0.5 m"** was a guess; measured mean 0.26 m, max 0.38 m, plus a rotation
   term nobody had considered (mean 7.7 deg).

## 2. What the ground-truth audit of corridor_v2 says (seed 1, native backend, proximity on)

Reproduces your run closely (15 LOST, 15 bridge links, 64 skipped cross-session RPE segments -- identical;
rotation drift 0.707 vs your 0.735 deg/m; translation drift 1.63 vs 1.57 %/m; this sandbox's fixture ran 480
frames vs 525 in your log, cause not investigated).

| measurement | value | meaning |
|---|---|---|
| loop closures found | 14 | |
| ...of which REAL (>=10 s apart, pose agrees with GT) | **0** | every "loop closure" joins nodes 1.6-2.9 s apart |
| ...wrong (pose disagrees with GT) | 0 (max error 4.8 cm) | they are correct, and useless |
| proximity links | 5, all 1.6-1.7 s apart, max error 2.2 cm | local constraints, not loop closures |
| GT path length / estimated | 23.81 m / 20.03 m | 3.78 m missing |
| bridge links | 15, asserting ZERO motion | camera really moved 3.84 m in total (mean 0.26, max 0.38 m) |
| rotation asserted zero across bridges | mean 7.7 deg, max 13.1 deg | |
| vertical error vs GT (final poses) | RMS 0.92 m, max 1.56 m | GT height range is 0.06 m (flat floor); estimate spans 1.75 m |
| horizontal error vs GT (final poses) | RMS 1.17 m | |
| anchored ATE odom / online / final | 150.9 / 148.6 / 148.6 cm | finalize() changes nothing here: no real loop exists |

Consequences for the plan:

* **Loop closure never happens on corridor_v2.** The 14-20 "closures" are nodes leaving STM (10 keyframes) and
  matching their own recent past. NEES is ~0 because they are near-identity links; the post-optimisation
  rollback cannot catch them and graph ATE cannot improve. Phase B1 (minimum time gap on candidates) is
  confirmed; expect the loop count to fall to ~0, which is the honest number.
* **The identity bridge accounts for essentially all of the missing path** (3.84 m asserted-zero motion vs
  3.78 m path shortfall) and injects ~7.7 deg of rotation error per LOST. That makes the bridge (2.3 / 2.4 in
  the menu: relocalise after LOST, or bridge with a motion model / preintegrated gyro) at least as important
  as any tilt prior. Hypothesis, not yet tested: the vertical error is largely bridge-injected rotation.
  Cheap test for Phase C: replace bridge T_ab with the ground-truth motion in a synthetic run and re-measure
  vertical error.
* **Vertical error is real** (0.92 m RMS, comparable to horizontal on a floor that is flat to 6 cm), so
  the IMU tilt work is justified -- but on evidence from this fixture, not from the report's `height_range_m`.
* Proximity "helped the map a bit" on your hardware runs is consistent with these local links adding
  redundant constraints; on the fixtures measured here it is not adding loop closures.

## 3. What changed

| ID | Files | Change |
|---|---|---|
| K1 | `pyslam/tools/run_outputs.py` (new), `run_slam.py`, `run_bag.py`, `run_synth.py` | One shared function does `finalize()` -> `map.ply` from FINAL poses -> trajectory exports -> imagery cleanup. Before, all three runners wrote the map first, from pre-finalize poses. A failed `finalize()` is recorded and the map is still written from online poses. `run_synth` keeps every previous `*_graph_*` number unchanged (online poses) and adds `ate_final_cm` / `anchored_ate_final_cm`. |
| K2 | `pyslam/core/config.py` and the three runners | `--config-override KEY=VALUE` (repeatable), typed from the field's own annotation, enumerated fields validated (`CONFIG_CHOICES`), typo hints. Parsed before the camera opens or a run dir is made. `Phase_Evaluation.md` documented this flag but it never existed; that doc also named `retrieval_backend=incremental` (real value `bow_incremental`) -- fixed. |
| K3 | `pyslam/memory/memory.py`, `pyslam/mapping/cloud.py`, `Config.imagery_cache_enabled` | Evicted nodes' RGB+depth go to a lossless PNG side-cache (never into LTM's database -- its contract is unchanged, retrieval never sees imagery). `assemble_cloud` streams them back one node at a time (peak RAM flat). `map_stats.json` / `summary.json["map"]` report coverage. Write failure disables the cache with a warning; it can never crash SLAM. Default location `<run_dir>/imagery_cache/` (NVMe, not /tmp), deleted after export unless `--keep-imagery-cache`. |
| K4 | `scripts/setup_orin.sh` (`power [apply]`), `pyslam/tools/env_probe.py`, `README_ORIN.md` | Power mode chosen by NAME from `/etc/nvpmodel.conf`. On the target unit `nvpmodel -q` = `15W / 0` and tegrastats showed CPU 1497 MHz / GPU 611 MHz: the old README's `nvpmodel -m 0` was NOT MAXN, so all earlier timings were taken in the 15 W mode. `power` only shows; `power apply` switches + `jetson_clocks`. |
| K5 | `pyslam/tools/phase_a_baseline.py` (new), `run_outputs.summarize_telemetry`, `pipeline.py` (one timing field) | Baseline harness (section 2's table comes from it) + `timing` block in every `summary.json` (keyframe-only `mem_duration_ms` percentiles, fps, WM size, budget overshoot, eviction cost). `telemetry` gains `enforce_budget_ms`. |
| gates | `tests/gates/test_ga.py` (10 checks, in `selftest`), `pyslam/tools/phase_a_mutation_check.py` | see section 4 |

Changed behaviour to be aware of: `map.ply` now contains far more points than before (see below), and its
poses are the closing-optimisation poses; `trajectory_report.json` gained two keys (existing keys unchanged).

## 4. Validation (x86 sandbox)

* `python3 -m pyslam.selftest`: **36/36** in 129 s (13 gate + 3 trajectory + 10 Phase-A + 10 mutation).
* Phase-A gates G-A 10/10 (config coercion, nvpmodel parsing of the target unit's exact `15W / 0`, cache round
  trip bit-exact, LTM blobs still imagery-free, unwritable-cache degradation, cloud identical to a never-evicted
  run, export ORDER, export resilience, telemetry digest, audit functions vs hand-computed truth, height-range
  annotation, and a real end-to-end run that really evicts).
* Source-level mutation check (`phase_a_mutation_check`): re-introducing each of {map before finalize, no cache
  write, RGB/BGR swap in the cache, `flag=false` parsed as truthy} turns the gates red: **4/4 caught**.
* Existing gates re-run after the changes: G2 5/5, G4 5/5, G5 5/5, G-TRAJ 3/3, G3 6/6 (section 6).
* Runner-level: `run_bag.py` on a recorded synthetic `.npz` bag with forced eviction (12/12 keyframes in the map,
  5 restored from the cache, cache dir removed, bad override rejected with the valid choices);
  `run_synth.py --scenario corridor_v2` full run.
* Map coverage on the full corridor_v2 run (default config, sandbox timing): **53/179 keyframes mapped before
  -> 179/179 now**, 342,587 points. (Coverage before/after depends on how much eviction happens, which depends on
  machine speed -- on the Orin expect a different number; `map_stats.json` will tell you.)
* Eviction cost with the PNG write (x86, 1 core): mean 33 ms, p95 43 ms, max 56 ms per eviction, 126 evictions
  in the 480-frame run. It happens after `mem_duration_ms` is sampled, so it does not feed the budget controller,
  but it does add to that frame's latency. Orin cost unmeasured.
* Smoke baseline on square6dof seed 1 reproduces the known post-mirror-fix accuracy (anchored ATE after
  finalize 0.74 cm).

## 5. Not validated -- do these on the Orin

* No hardware run of any of this. `run_slam.py` was only import/argparse-checked; its export path is the same
  function `run_bag.py` exercised end to end, but the live-camera path itself is untested.
* `/etc/nvpmodel.conf`'s line format and the SUPER mode's name/ID are from memory: `./scripts/setup_orin.sh power`
  prints the raw mode list before doing anything, and the parser degrades to "run `nvpmodel -p --verbose`".
* PNG eviction cost on the Cortex-A78AE cores, and PNG size on real (not synthetic-noise) imagery (synthetic:
  ~0.8 MB/node).
* All accuracy numbers above are synthetic fixtures, seed 1, native backend. Not real-world evidence.
* Real runs will still say `gravity_aligned: false` unless the run starts with a ~1.5-2 s completely still hold.

## 6. G3

G3 (retrieval, ~10 min on this hardware): 6/6 passed.

## 7. Revised order of work (changes from the previous plan)

1. On the Orin: `power apply`, then `phase_a_baseline` twice (default f2f, and `odometry_backend=f2m`) with the
   same `--backend`, then `--compare`. Send back the two `.json` files -- they answer the f2m question with
   real (post-mirror-fix) numbers and give the real `mem_duration_ms` distribution for retuning `wm_budget_ms`.
2. Phase B1 (min loop time gap) -- now with evidence and a metric (`loops.n_real`) that will show it working.
3. Bridge replacement (relocalise after LOST / motion-model bridge) BEFORE the IMU tilt prior: it is the
   measured source of the missing path length and ~7.7 deg of rotation per LOST, and the likely main cause of the
   vertical error. Then C (tilt prior), judged by `vertical_final.vert_err_rms_m`, not by `height_range_m`.
4. D (map quality: depth filters, density), E (GPU), F (threading) unchanged.
