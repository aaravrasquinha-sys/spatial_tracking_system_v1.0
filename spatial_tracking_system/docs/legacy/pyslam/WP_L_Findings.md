# WP-L Findings: Orin accuracy plan, Phase B (loop-closure correctness)

Follows `WP_K_Findings.md`. Named WP-L because `WP_B1/B2/B6` already exist for the older Phase 1B.
Everything here was measured on the x86 dev sandbox (1 core, native pose-graph backend, synthetic fixtures,
seed 1) -- **nothing has run on the Orin**, and **every new behaviour is OFF by default**: with default settings
the pipeline is unchanged (telemetry/audit fields are additive).

## 1. What was built

| ID | Files | Change | Default |
|---|---|---|---|
| L1 | `pipeline.py` (`_loop_candidates`, per-keyframe odometry path), `Config.loop_min_path_m`, `loop_min_time_s` | A WM node is a loop-closure candidate only if the odometry path (and/or capture time) since it was created is at least the bound. Applied BEFORE retrieval scoring, so recent nodes also leave the likelihood population and retrieval cost. A node only ever goes ineligible -> eligible. Fails open for a node with no recorded path. | off (0) |
| L2 | `loop/bayes.py`, `Memory.neighbours`, `Config.bayes_diffusion_enabled/_rate` | Graph-neighbour belief diffusion in the predict step (RTAB-Map's mechanism). Mass conserved exactly; only moves between current candidates; strict no-op when disabled. | off |
| L3 | `frontend/odometry.py`, `pipeline.py` telemetry | `kf_reason` (trans / rot / inliers) recorded per keyframe; the audit tool counts them. Telemetry only. | on (additive) |
| L4 | `memory/memory.py` (`select_redundant_victim`), `Config.wm_evict_policy`, `wm_redundancy_radius_m/_angle_deg` | `redundancy` eviction: victim = the node with the most other WM nodes within radius AND viewing angle (ties -> weight -> oldest). Under a fixed WM size this thins the map to roughly uniform coverage instead of keeping only the recent stretch. Reduces to the default rule when nothing is crowded. | `oldest` |
| L5 | `memory/memory.py::enforce_budget`, `Config.wm_max_nodes` | Deterministic WM cap: when WM holds more than N nodes, evict one victim per call (per `wm_evict_policy`) regardless of timing. The time-driven budget makes WM composition depend on how fast the machine ran, so identical runs evicted different nodes and gave different loop-closure results. | off (0) |
| tools | `phase_a_baseline.py` | revisit recall vs GT, per-link path/rotation/keyframe gaps (`loops.rows`), `keyframe_reasons`, `vertical_online`, `wm_oldest_node_id_at_end`, `--no-finalize`, `--loop-gap-m` | - |
| fixture | `tests/synth/scenarios.py::corridor_lap13` | corridor_v2's ring driven 1.3 laps (624 frames, 31.8 m, 152 frames with a true revisit partner). `baseline.py --scenario all` is restricted to the 5 frozen fixtures so this cannot leak into the frozen store. | - |
| gates | `tests/gates/test_gl.py` (9 checks, in `selftest`), 3 new mutants in `phase_a_mutation_check.py` | see section 4 | - |

## 2. Why: what the measurements said

* **corridor_v2 cannot evaluate loop closure at all.** It is ONE lap that ends where it began: only 1 keyframe has a
  true revisit partner (<0.5 m, <30 deg, >=1.5 m of path away). Its 14-20 reported "loop closures" were all 1.6-2.9 s
  apart (nodes leaving STM matching their own recent past) -- confirmed again on the new fixture (below).
* **Oldest-first eviction removes the start of a run first.** Rehearsal weights are all 1 unless a vocabulary is loaded,
  so the default policy is FIFO. On corridor_lap13 the oldest node still in WM at the end was 484 of 624 frames.
* **Keyframes on corridor_v2 are inlier-driven:** 102 of 179 by `n_inliers < keyframe_min_inliers`, 69 by translation,
  15 by rotation. On the small fixtures rotation dominates (room_orbit 44/45, square6dof 21/28).

## 3. Results (x86 sandbox, synthetic, seed 1, native backend, no GTSAM)

### 3.1 The lap fixture, default behaviour
corridor_lap13 with default settings (proximity on): 20 loop closures, all 1.2-1.9 m of path / 1.6-2.9 s apart,
**none closes the lap**; oldest node in WM at the end 484 of 624 frames; 20 LOST; anchored ATE 189 cm; vertical
error RMS 0.86 m, horizontal 1.68 m. (Same failure as corridor_v2, now on a fixture that HAS a loop to close.)

### 3.2 Fixed WM size (wm_max_nodes=45, wm_budget_ms=1e6, loop_min_path_m=3.0, proximity on, no finalize)
corridor_lap13, seed 1, **first 500 frames** (stopped right after the lap closes at ~frame 480-490, before the
sandbox's slow post-loop phase), `wm_max_nodes=45`, `wm_budget_ms=1e6`, `loop_min_path_m=3.0`, proximity on, ground-truth audited:

| | default (oldest-first) eviction | `wm_evict_policy=redundancy` |
|---|---|---|
| loop closures | 0 | **4** (480, 484, 487, 490 <-> node 4) |
| real (>=3 m of path) / wrong | 0 / 0 | **4 / 0** |
| link error vs ground truth | - | **1-7 mm**; path between the nodes 23.8-24.3 m (true lap closures) |
| revisit recall (>=3 m, frame level) | 0/9 | 4/9 |
| oldest node in WM at the end | 359 (start lost) | **4** (start kept) |
| anchored ATE, online poses | 157.0 cm | **134.3 cm (-14 %)** (odometry alone: 157.8) |
| Umeyama ATE, online poses | 69.9 cm | 59.9 cm (-14 %) |
| horizontal error RMS | 1.27 m | **0.98 m (-23 %)** |
| vertical error RMS | 0.92 m | 0.92 m (unchanged -- a loop does not fix tilt) |
| LOST events | 16 | 16 |

Also (full 624 frames, control): FIFO + cap 45 + 3 m gap -> 0 loops, oldest node 490, anchored ATE 188.7 cm.
The redundancy run reproduced the same four closures bit-for-bit (same posteriors 0.780/0.916/0.951/0.958, same inlier
counts 157/143/95/44) across two separate runs once the cap made eviction deterministic. (Before the cap, the time-driven
budget gave two different results on the same fixture: closures in one run, none through frame 583 in another.)
The redundancy run over the full 624 frames did not finish in the sandbox (slow native solver after the closure), so
frames 500-624 are unmeasured. ATE figures are ONLINE poses (LTM-frozen, no closing optimisation): the final trajectory,
which can also correct the evicted part of the run, is expected to gain more but was not measured.

### 3.3 Other measurements
* **loop_min_path_m on small fixtures (rotation-heavy):** square6dof unchanged (its real loop kept); aliasing_rooms 5 -> 4
  loops (2 wrong ones unchanged); **room_orbit 3 -> 0 loops and anchored ATE 4.17 -> 6.57 cm (worse)** -- its "trivial by
  path" loops were rotation-driven revisits that helped. Path length is the wrong yardstick for rotation-dominated motion.
  => keep off by default; use only on large translation-dominated loops.
* **bayes_diffusion (L2):** on the three small fixtures results are identical to B1-only to 0.01 cm; unit-level effect is
  ~9% more belief kept on a moving true match. No evidence it helps. Off.
* **aliasing_rooms** false cross-room merges (2 wrong loops) are unchanged by any of this -- as documented, only IMU
  plausibility (Phase C) could catch them.
* **Keyframe reasons:** corridor_v2 102/179 inlier-triggered; small fixtures rotation-triggered. The keyframe-threshold
  sweep and f2m comparison were NOT completed (sandbox time), so no keyframe/f2m recommendation is made here.

## 4. Validation
* `selftest`: 46/46 passed in 171 s (13 gate + 3 trajectory + 10 Phase-A + 10 Phase-B + 10 mutation). Existing G2 5/5 re-run after the eviction changes; G3/G4/G5/G-TRAJ were last run at WP-K and not repeated (defaults unchanged; the only shared code touched is `Memory.enforce_budget`, `BayesFilter._predict`, `Pipeline._try_loop_closure`, all behaviour-identical with the new options off, and the selftest exercises them).
* G-L gates (`tests/gates/test_gl.py`, 10 checks): candidate-gap filter incl. monotonic eligibility and fail-open; diffusion mass
  conservation over 30 random graphs, strict no-op when disabled, order independence; sequence-evidence and noise-only
  non-firing; diffusion scope; recall/audit functions vs hand-built GT; end-to-end square6dof with the bound on/off;
  redundancy victim == brute force on 300 random maps; coverage under a fixed WM size; lap-fixture geometry; deterministic cap.
* Source-level mutation check (`phase_a_mutation_check`): **7/7 caught, after one fix**: the first run caught 6/7. The LEAK mutant (diffusion forgets to subtract what it hands on) went undetected because `BayesFilter.update()` renormalises every step, which hides mass created inside `_predict`; GL.2 now tests `_predict` directly (`_predict changed total belief mass 1.0 -> 1.27` under the mutant).
* `corridor_lap13` was run through the full `validate_scenario` for seed 1 only (first attempt failed on an end-of-path
  heading snap, fixed by generating and dropping look-ahead frames). Seeds 2-5 are unvalidated.

## 5. Not validated / limits
* No hardware. All numbers are synthetic, one seed, native backend; ATE figures in 3.2 are ONLINE poses (finalize skipped:
  the pure-python native backend can take >35 min to finalize a graph containing a real lap closure; GTSAM will not).
* The redundancy policy's radius/angle (0.5 m / 30 deg) and the 3 m path bound are fixture-scale choices, not tuned on
  real data. loop_min_path_m=3.0 would remove real loops in a room-scale run (room_orbit's whole path is 2.5 m).
* Redundancy eviction is untested on the frozen small fixtures under memory pressure, and untested with a loaded vocabulary
  (where rehearsal weights are no longer all 1).
* Recall numbers use frame-level opportunities (<0.5 m, <30 deg, >= path bound) and count endpoints of correct links.

## 6. Suggested settings to try on the Orin (large loop / corridor / building-scale runs only)
```
--config-override wm_max_nodes=45  --config-override wm_budget_ms=1000000   # fixed, reproducible WM size (pick N to fit your latency)
--config-override wm_evict_policy=redundancy
--config-override loop_min_path_m=3.0
```
Compare with the same run without the last two lines via `phase_a_baseline --compare`; the deciding columns are
`loops.n_real` (>=3 m), `revisit.recall`, and `anchored_ate_final_cm`. Do not use loop_min_path_m on a small orbit.

## 7. Still open
Bridge replacement after LOST (measured source of the missing path and ~7.7 deg rotation per LOST, WP_K_Findings.md), the
keyframe-rate/f2m question, then Phase C (IMU) judged by `vertical_final.vert_err_rms_m`.
