# pySLAM: System Summary & Handoff (for a new session)

**Read this first if you're picking this project up in a new conversation.**
Paste this whole file (or upload it) plus the zip's code, and say "continue
from Phase 2" or similar. Everything below is accurate as of commit
`e901514` (`git log --oneline` in the repo root shows the full history with
detailed reasoning in every commit message -- read those for the "why",
this file is the "what" and "what's next").

---

## 0. What this project is

A pure-Python, from-scratch implementation of RTAB-Map-style RGB-D visual
SLAM, targeting an Intel RealSense D435i on an HP Z440 workstation
(Ubuntu, Python 3.10, OpenCV 5.0.0.93, NumPy 2.2.6 -- see
`Runbook.md`/`Phase_Evaluation.md` for the original hardware brief).
Pipeline: capture -> ORB features -> visual odometry -> STM/WM memory ->
appearance-based retrieval -> Bayes filter -> geometric loop-closure
verification -> pose graph optimisation -> point-cloud map.

## 1. Where things stand right now

**Update, WP-M (bridge replacement after LOST) -- read `WP_M_Findings.md` first.**
Both WP-K and WP-L independently flagged this as the biggest remaining lever, ahead of the IMU
tilt prior. New: `pyslam/imu/bridge.py` (opt-in via `cfg.bridge_mode="gyro"`, default stays
`"identity"`/unchanged) replaces the near-zero-information identity bridge with a gyro-
integrated rotation (reuses P5.0's already-validated `preintegrate()` unchanged) plus a
constant-velocity-extrapolated translation, each falling back to the old identity behaviour
independently if its own inputs aren't trustworthy. Measured on corridor_v2 (seed 1, first 300
of 480 frames): vertical error RMS dropped 0.782m -> 0.123m (-84%) and online ATE improved 19%
-- but the trajectory AFTER the closing graph optimization got WORSE (42.3cm -> 52.4cm anchored
ATE). The ORIGINAL hypothesis (the new bridge is more confident than the old one and out-
competes a nearby loop closure) was tested directly across a >2000x span of info-matrix weight,
including matching the old bridge's own weight exactly, and the regression never moved at all --
**that hypothesis is FALSIFIED.** The real cause looks structural (the native backend's
`optimize()` starts each solve from the CURRENT pose_map rather than a fresh baseline, and its
own `scipy 'trf'` solver falls back toward that starting point in under-constrained directions --
so a nonzero bridge MEAN, not its weight, gets partially baked in during earlier online solves
regardless of how weakly it's weighted) -- see `WP_M_Findings.md` section 3.3.1 for the full
falsification evidence and the still-open confirmation experiment. selftest is 54/54; mutation
check 9/9; no code changed in the follow-up session, only `--config-override` experiments.
**Do not recommend `bridge_mode="gyro"` for a `finalize()`d trajectory until this is resolved**
-- it stays off by default either way.

**Update, WP-K (Orin accuracy plan, Phase A) -- read `WP_K_Findings.md` first.**
After the first real Orin runs the work turned to accuracy and map detail. Phase A
is measurement + free wins, no algorithm change: (K1) `map.ply` is now built AFTER
`finalize()` via the shared `pyslam/tools/run_outputs.py`; (K2) `--config-override
KEY=VALUE` really exists on all three runners; (K3) keyframes evicted to LTM are kept in a
PNG side-cache so they reach the map (corridor_v2: 53/179 -> 179/179 keyframes); (K4)
power mode is chosen by NAME (`setup_orin.sh power`), mode 0 is 15W on the target unit, not
MAXN; (K5) `pyslam/tools/phase_a_baseline.py` audits loops, vertical error, bridge links, map
coverage and timing against ground truth. Also: `trajectory_report.json`'s `height_range_m`
is NOT height when `gravity_aligned` is false -- an earlier analysis misread it as ~8 m of
vertical drift; do not repeat that. selftest is 36/36 (was 26/26).


**Update, latest session: WP-P5.2 completed properly, including a new
fixture and a real bug it found.** `gravity_init_square6dof`
(`tests/synth/scenarios.py`) is a new fixture -- static hold + smooth
constant-velocity non-rotating translation (quasi-static by the
gyro-rate/accel-variance criteria that actually matter, not "zero
velocity") + `square6dof`'s own unchanged loop -- built specifically
because NONE of the other 5 fixtures can ever trigger a gravity prior
end-to-end (not even `static_60s`, which never produces a second
keyframe at all). Building it surfaced a real bug: `pose_map`/
`pose_odom` are CAMERA-frame poses while `Frame.imu` measures the
physical BODY frame -- `pyslam/imu/gravity.py` was built and oracle-
tested assuming they were the same frame, correct for the isolated
oracle (which never touched a real `pose_map`) but silently wrong the
moment it ran through the real pipeline. Fixed with an explicit
`R_body_cam` parameter (measured effect: correction magnitude
1.48deg->0.76deg on the new fixture); still defaults to identity
because the frozen contract has no field for a real extrinsic yet --
documented as KNOWN WRONG for this project's own D435i rig, a decision
flagged for whoever continues, not made unilaterally here. Gate G5:
5/5 (added an end-to-end check against the new fixture). Two
scenario-building bugs (a heading-snap from assuming the loop's own
start direction, and `poses_from_path`'s known look-ahead-clamps-at-
the-end trap recurring in new code) were caught by `validate_scenario`
itself before either could ship -- see `WP_P5_Findings.md` section 4.3
for both.

**Earlier in the same session: WP-P5.2's core (gravity/tilt prior,
loose coupling) built on top of WP-P5.0.** `pyslam/imu/gravity.py`:
quasi-static detection, gravity-direction estimation, and a tilt-only
correction (oracle-validated -- a known 5deg tilt error corrects to
0.0000deg residual while a known 15deg yaw error is preserved, not
touched, since gravity alone cannot observe yaw). Wired into
`pipeline.py` as an opt-in `Link(kind="prior")`
(`cfg.gravity_prior_enabled`, default `False`) -- required zero
`GraphBackend` changes, confirmed `backend_native.py` already treats
`"prior"` links the same as `"odom"`.

**Before that, same session: WP-P5.0 (IMU preintegration) built and
validated -- the blocking prerequisite for all of Phase 5.** New:
`pyslam/imu/preintegration.py` (on-manifold Euler preintegration +
first-order bias correction, bias Jacobians via finite difference by
deliberate choice -- see WP_P5_Findings.md for why), `tests/synth/
imu_world.py` (an analytically-consistent IMU trajectory generator --
`SyntheticSource`'s existing IMU is gravity-reaction-only and cannot
validate any of this). `pyslam.selftest`: 22/22 (mutations 9 and 10,
gravity-sign inversion and info-matrix axis swap, both wired in --
mutation 10's own first draft was itself broken and caught by its own
sanity assertion before shipping, see `WP_P5_Findings.md` section 4.4).
`Frame`/`SensorSource`/the graph backend itself are all still
untouched by anything in Phase 5 so far. The `Link`/`GraphBackend`
contract still doesn't have room for IMU nav-state factors
(velocity+bias, 9-dim residual); that gap is real, documented, and
unaddressed -- P5.4's problem specifically, not P5.0/P5.2's.

**Update, post-Phase-4 session: WP-B1 continued (odometry_f2m.py).**
Picked back up from section 6's "still-open items" -- landmark data
association was found to be the actual accuracy-gap cause (the local
map was churning over ~15x per run, never persisting), fixed via
landmark fusion + an intra-batch dedup pass; a separate velocity
off-by-one-frame bug was found and fixed (was silently spanning two
frame intervals, not one); fixing the first regressed BA's per-frame
cost (roughly doubled free-landmark count -> dominant 74% of wall
time), fixed with the same sparse-Jacobian technique F5 gave the pose
graph, which in turn surfaced and fixed a real pre-existing numerical
robustness bug (weakly-constrained landmarks could silently run away
to kilometres under BA, undetected by the existing residual-only
safety net -- fixed with a second, oracle-validated displacement
check); and a 3-seed sweep found and fixed a real accuracy-gap cause
(`f2m_min_inlier_ratio` 0.35->0.5, nearly halving the worst seed's
error with zero effect on the others). Net result, 5-seed
square6dof: f2m's map_ATE went from a pre-session 39.35cm (odom,
23x worse than f2f) to a post-session median 8.47cm (post-graph,
~9x worse than f2f's ~0.9cm) -- real, substantial, multiply-validated
progress, but still not at parity. `cfg.odometry_backend` correctly
stays `"f2f"`. Full detail, all numbers, and what's still open (seed
2's own still-unexplained missed loop closure, `corridor_v2` not
re-validated with any of this) in `WP_B1_Findings.md` section 7.
`pyslam.selftest` 20/20 throughout (no regression); G2 4/4, G4 4/4,
and G3 2/6 (interrupted by an unrelated environment issue, not a code
failure) manually confirmed clean, since none of Phase 2-4's gates run
inside `selftest` itself -- see that finding's own note in section 7.7,
still unresolved.


**Update: Phase 4 (WP-P4, robust back-end + proximity) is done, on top
of Phase 3.** Headline result: WP-B2's long-standing blocker (Phase 1B)
is effectively resolved -- extending `pnp_info.py` with a second,
oracle-validated function for verify.py's other PnP direction, then
wiring real Hessian-derived info matrices into odometry AND loop-link
together (single `cfg.use_hessian_info` flag, opt-in), makes graph
optimisation measurably BETTER than the heuristic on two independent
fixtures (0.532cm vs 0.944cm on square6dof; 3.588cm vs 4.173cm on
room_orbit) rather than "functionally inert" the way wiring only one
side was. Also added: Dynamic Covariance Scaling for loop links
(opt-in via `cfg.loop_robust_kernel`) -- measured directly against a
5m injected wrong loop, DCS holds ATE at 0.933cm (vs clean's 0.944cm)
where the existing static Huber kernel lets the same wrong loop blow
ATE up to 4.806cm (5.1x worse); and geometry-only proximity detection
(`proximity/detect.py`, opt-in via `cfg.proximity_enabled`), which
found 7 real, clean (NEES~0) links on `square6dof` that appearance-
based retrieval didn't. Gate G4: 5/5 checks passing. `pyslam.selftest`:
20/20 (12 gates + 8 mutations). iSAM2 was explicitly NOT attempted --
`pip install gtsam` conflicts with this codebase's numpy 2.x dependency
globally, a real risk not worth taking for one gate check; flagged
honestly rather than silently skipped. See `WP_P4_Findings.md` for full
detail. WP-B1's f2m accuracy gap (Phase 1B) remains the one still-open
item from that phase.

**Before that, Phase 3 (WP-P3, retrieval upgrade):** incremental
vocabulary, hnswlib/numpy-fallback ANN, honestly-scoped learned-channel
descriptor. Gate G3: 6/6 passing. See `WP_P3_Findings.md`.

**Before that, Phase 2 (WP-P2, memory management):** real STM/WM/LTM
with SQLite (WAL) persistence, transfer-under-budget, graph-neighbour
retrieval, WM-only optimisation with frozen-LTM anchor priors. Gate G2:
4/4 passing. See `WP_P2_Findings.md`.


**Update, post-handoff (commits `fc253df`..`04911aa`, on top of the
`e901514` state this document originally described):** F5 is fixed
(size-gated sparse Jacobian, see `WP_B_F5_Findings.md`), `corridor_v2`'s
baseline is now a full 5-seed run (was 1-seed partial), F8's Config
hygiene is done, WP-B6's retrieval/Bayes instrumentation has run on
`corridor_v2` with real findings (see `WP_B6_Findings.md`), and WP-B1 is
mostly done: local bundle adjustment, a permanent-freeze bug, and an
eviction-ordering bug are all fixed and validated (see
`WP_B1_Findings.md`) -- f2m no longer catastrophically collapses, but
still does not reach accuracy parity with f2f (23x worse ATE, 11x worse
drift, a missed loop closure on `square6dof`). `cfg.odometry_backend`
still defaults to `"f2f"`. WP-B2's PnP-Hessian info matrix is done,
oracle-validated, and calibrated (see `WP_B2_Findings.md`), but
deliberately NOT wired into the pipeline -- doing so passes every
existing gate but makes graph optimisation functionally inert
end-to-end, because it's ~300,000x larger in scale than `verify.py`'s
still-heuristic loop-link info matrix, which is genuinely P4-scoped
work, not WP-B2's alone. **Still not started: closing f2m's remaining
accuracy gap, and giving the loop-link info matrix a comparable basis
to WP-B2's odometry one** -- both are correctness-sensitive,
deliberately not rushed; see section 6 below for what they still need.

**Phase 0** (delivered before this engagement): a working pipeline
structure with 8 real, non-obvious bugs, most critically an inverted
odometry direction (every relative motion integrated backwards) that the
original gate suite could not detect because its test fixtures were flat
and yaw-only.

**Phase 1A** (COMPLETE, frozen, fully tested): fixed the critical bugs,
built a test harness that can actually catch bugs (proven via a 5/5
mutation-injection suite), rebuilt the synthetic simulator with real 6-DoF
motion and validated fixtures, and froze a baseline. This is the
trustworthy foundation everything else sits on.

**Phase 1B** (PARTIAL): WP-B3 (session tracking) is done and validated.
WP-B1 (frame-to-local-map odometry): local bundle adjustment, a
permanent-freeze bug, and an eviction-ordering bug are all fixed and
validated -- f2m no longer collapses, but still doesn't reach accuracy
parity with f2f (23x worse ATE, 11x worse drift on `square6dof`). See
section 5.2's update and `WP_B1_Findings.md`. WP-B2 (formal uncertainty
calibration): the PnP-Hessian info matrix itself is done and calibrated,
but not wired in -- it's incompatible in scale with the still-heuristic
loop-link info matrix (`verify.py`), which is genuinely P4-scoped work.
See `WP_B2_Findings.md`. WP-B5 (perf instrumentation beyond what's
already there) is NOT started. WP-B4's F8 remainder (Config hygiene) and
WP-B6 (corridor evidence-gathering) are done -- see the update note
above.

**Phase 2 onward**: not started. See section 6 for what they need.

---

## 2. Run it yourself, right now

```bash
cd pyslam_phase1  # wherever you extracted the zip
python3 -m pyslam.selftest              # 54/54 checks, ~3 min, run this first always
python3 -m pyslam.tools.mutation_check  # standalone mutation-sensitivity check
python3 run_synth.py --scenario square6dof --seed 1     # full pipeline demo
python3 run_synth.py --scenario corridor_v2 --seed 1     # slower (~3-4 min), harder fixture
python3 -m tests.synth.scenarios         # validates all 5 fixtures' geometry (slow, ~10 min total)
python3 -m pyslam.tools.env_probe        # hardware/environment check (no RealSense needed to run it)
python3 -m pyslam.tools.baseline --scenario square6dof   # re-run the frozen baseline
```

`git log --oneline` (12 commits) is itself a readable history of every
decision made, in order, with full reasoning. When in doubt about *why*
something is the way it is, `git log -p <file>` or `git show <hash>` will
usually answer it directly -- every commit message explains the evidence
behind the change, not just what changed.

---

## 3. The 8 Phase-0 bugs found and fixed (Phase 1A)

Full detail in `Phase1_Plan.md` (the original research/architecture doc)
and each fix's own commit message. Quick reference:

| ID | Bug | Fixed in |
|---|---|---|
| F1 | Odometry integrated the INVERSE of every relative motion (`solvePnPRansac`'s result was mislabeled) | `pyslam/frontend/odometry.py`, commit `d8c46c5` |
| F2 | Test fixtures were flat/yaw-only, structurally unable to detect F1 | `tests/synth/world.py` (`handheld_poses`), commit `109439c` |
| F3 | `poses_from_path`'s look-ahead clamped instead of wrapped at loop ends, causing a heading snap (found empirically while building WP-A2, not just in the original flagship fixture -- affects ANY looped path) | `tests/synth/world.py`, commit `109439c` |
| F4 | Corridor fixture's 4 wall segments didn't geometrically overlap -- camera was outside all of them half the time | `tests/synth/scenarios.py` (`add_ring_corridor`), commit `109439c` |
| F5 | Native graph backend used dense finite-difference Jacobians, O(n^2)-ish and the dominant runtime cost | NOT fixed -- documented, deprioritized (see section 5) |
| F6 (a-e) | Various graph-layer issues: unit-inconsistent information matrices, GTSAM tangent-ordering mismatch, no map<-odom correction, LOST created ungraphable disconnected components | (a)/(b)/(d) fixed in `6bf549f`; (c) fixed in `6bf549f`; (e) mitigated by WP-B3's session_id, commit `21d6647` |
| F7 | Hardware recording was fundamentally broken (instance-`__iter__` monkeypatch that Python never calls, `.bag.npz` filename mismatch, no Ctrl-C safety) | `pyslam/sensors/realsense.py`, `run_slam.py`, `run_bag.py`, commit `6264ddd` |
| F8 | Minor: unused Config fields, USAC flag computed-but-ignored | USAC fixed in `33c4a37`; Config hygiene fixed in `751fe47` (see `WP_B_F5_Findings.md`-adjacent commit log -- removed genuinely dead fields, documented the rest as declared-but-not-wired rather than silently wiring them to different numeric defaults) |

## 4. Key architectural decisions made in Phase 1A

- **Anchored ATE + mirror detection** (`pyslam/tools/metrics.py`): Umeyama
  alignment can hide a mirrored/inverted trajectory on planar+yaw-only
  motion by finding a 180° flip that scores well. Anchored ATE (align only
  by the first pose) cannot be fooled this way. `mirror_check()` flags a
  large gap between the two as a red flag.
- **Mutation-sensitivity harness** (`pyslam/tools/mutation_check.py`):
  5 known defect classes are synthetically injected and the metrics must
  catch all 5, or the harness itself isn't trusted. Run before trusting
  any other gate.
- **6-DoF validated simulator** (`tests/synth/scenarios.py`,
  `tests/synth/world.py`): 5 fixtures (square6dof, room_orbit,
  corridor_v2, static_60s, aliasing_rooms), each checked by
  `validate_scenario()` for free-space clearance, valid depth fraction,
  angular velocity bounds, and a LOCAL-WINDOW rotation-snap detector
  (deliberately not a global-median one -- see the comment in
  `world.py::validate_scenario` for why that matters).
- **Frozen baseline** (`pyslam/tools/baseline_store.json`,
  `WP_A3_Findings.md`): the numbers WP-B1/B2 must beat, with corridor_v2
  explicitly flagged as low-confidence (1 seed, 220/480 frames -- too slow
  to fully validate in this environment; needs an offline/background run).
- **Post-optimisation loop rollback** (`pyslam/pipeline.py`,
  `_try_loop_closure`): after adding a candidate loop link and
  re-optimising, checks the link's own NEES against the OPTIMISED poses;
  rolls back if it exceeds `chi2(0.995, df=6)`. Catches a loop that's
  geometrically plausible alone but contradicts the accumulated graph.
- **session_id + session-aware RPE** (`pipeline.py`, `metrics.py`): a LOST
  recovery starts a new "session" (no measured transform back to the old
  local frame). Drift metrics must not score a distance segment that
  crosses a session boundary -- doing so compares two unrelated coordinate
  frames and produces nonsense (confirmed: fixed corridor_v2's baseline
  drift from a misleading 18.3%/m down to a real 1.6%/m).

## 5. Two significant OPEN findings (not bugs -- real limitations, documented)

### 5.1 Perceptual aliasing defeats appearance-only SLAM (aliasing_rooms fixture)

Two rooms with IDENTICAL texture produced 3 false loop closures that were
**mutually self-consistent to within 0.6-3.2cm** -- not noisy, so no
threshold (inliers/posterior/NEES) can catch them; the false evidence is
geometrically as good as genuine evidence. Worse: traced further and found
**frame-to-frame odometry itself** was fooled across the 20m room-to-room
cut (never even flagged LOST). This is not fixable within pure
appearance+geometry. The one real lead: **Phase 5's IMU fusion** would
notice zero real acceleration across an apparent 20m visual jump, which
vision alone structurally cannot. See `pipeline.py`'s
`_try_loop_closure` comment and `WP_A3_Findings.md` section 3 for full
detail. `PipelineResult.cross_session_merges` exists so a graph-connectivity-crossing
merge is at least auditable (via a union-find over node ids), though it
doesn't fire on THIS specific failure since odometry never detected
anything wrong in the first place.

### 5.2 Frame-to-local-map odometry: BA and the capacity collapse are both fixed; accuracy parity is not

**UPDATE (commits `a93bc09`, `2a92882`): resolved through several
rounds -- full history in `WP_B1_Findings.md`, summarized here.**

The original diagnosis (landmarks frozen at insertion-time pose,
compounding error, "PnP inlier ratio collapsed to 17% by frame 28/140")
turned out to be an early/soft reading of a much more severe problem
that trace didn't run far enough to see. Three things were found and
fixed, in this order:

1. **Local bundle adjustment** (`local_ba.py`) -- the originally-scoped
   fix, oracle-validated against a noise-free synthetic ground truth
   before being wired in, with a two-pass robust-outlier scheme added
   after a single Huber pass was found insufficient (a single bad
   cross-frame match pulled an otherwise-clean landmark 3.76m off
   despite robust loss being active).
2. **A permanent-freeze bug** in `_add_landmarks`, the actual root cause
   of the collapse (not landmark drift, not BA): its insertion loop
   refused to add anything once the map first reached its cap, which
   happens by frame 10/140 -- and since eviction only fires when OVER
   budget, and insertion now never exceeded it, the map froze completely
   and permanently at whatever it contained at frame 10, however far the
   camera still had left to travel. Confirmed directly via the landmark-
   creation counter, which stopped incrementing at frame 10 and never
   moved again. This fully explains why three EARLIER targeted fixes
   (eviction sort-order, a real de-duplication bug found and fixed
   along the way, and sweeping the cap size 3000-8000-unbounded) each
   failed to help: insertion/eviction had already permanently stopped
   regardless of any of them.
3. **Eviction sort-order**, re-tested in the now-corrected context (the
   earlier test of this was uninformative, since eviction essentially
   never ran while bug 2 was active): switching from `(n_obs,
   last_seen)` to `(last_seen, n_obs)` -- recency primary -- was needed
   on top of the freeze fix to fully resolve the collapse. A single
   large prune event under the old ordering was observed to zero out
   the very next frame's matching outright.

With all three in place: **zero LOST frames across the full 140-frame
square6dof run** (was: LOST by frame 33 with none of them), sustained
150-377 inliers from frame 30 onward, confirmed with BA active too (41
keyframes, still zero LOST).

**But f2m still does not reach parity with f2f.** Full pipeline
comparison, `square6dof` seed=1, f2m vs the frozen f2f baseline: anchored
ATE 39.35cm vs 1.72cm (23x worse), trans_drift 14.74%/m vs 1.29%/m (11x
worse), missed the loop closure f2f found, ~5x slower per frame. f2m
went from catastrophically broken to genuinely functional -- real,
meaningful progress -- but "no longer collapses" and "wins" are
different bars, and only the first has been cleared.

**`cfg.odometry_backend` still stays `"f2f"`.** Do not flip until the
remaining accuracy gap is closed and re-validated against the frozen
baseline. `WP_B1_Findings.md` section 6 has concrete starting points for
that: BA's 5-keyframe window relative to how much more of the scene the
(now-healthy) map spans at any moment; a per-keyframe (not just
aggregate) ATE comparison against f2f to see whether error is smooth or
concentrated at specific events; and why the loop closure was missed
specifically (retrieval never recognising the revisit, vs. f2m's own
drift by that point exceeding the verifier's geometric tolerance).

## 6. What Phase 2 (and finishing Phase 1B) needs to know

If continuing Phase 1B first (recommended before Phase 2):
- **WP-B1**: local BA, the permanent-freeze bug, and eviction ordering
  are all DONE and validated -- see 5.2 and `WP_B1_Findings.md`. What's
  left is closing the remaining accuracy gap (23x ATE, 11x drift, a
  missed loop closure) before `cfg.odometry_backend` can flip to
  `"f2m"`. See `WP_B1_Findings.md` section 6 for concrete next steps
  (BA window sizing, per-keyframe error comparison, the missed-loop
  diagnosis). `wpb6_trace.py`'s pattern (wrap, don't modify, log
  everything, join against ground truth post-hoc) is a reasonable
  template for whichever of those gets picked up first.
- **WP-B2**: PARTIAL -- see `WP_B2_Findings.md`. The PnP-Hessian
  information matrix itself is done, oracle-validated (Monte Carlo NEES
  check, permanent gate), and calibrated via NEES on `static_60s`
  (`sigma_px=1.25`, validated out-of-sample). NOT wired into
  `odometry.py`/`odometry_f2m.py` -- both still use the original
  heuristic (`eye(6) * inlier_ratio * 100`). Wiring it in was tried and
  reverted: it passes every existing gate (17/17, 5/5) but makes graph
  optimisation functionally inert end-to-end (ATE improvement from
  optimisation goes from real to ~zero), because the correctly-scaled
  odometry info matrix (tens of millions for a typical inlier count) is
  ~300,000x larger than `verify.py`'s still-heuristic loop-link info
  matrix (capped at 50), so the optimiser stops trusting loop closures
  at all. Properly fixing this means giving the loop-link info matrix a
  comparable basis too -- which is explicitly P4 scope per this
  document's own section 5 ("real estimated information matrices from
  the verifier's Hessian instead of heuristics"), not something WP-B2
  was ever meant to cover alone. See `WP_B2_Findings.md` section 4 for
  concrete next steps, including that `baseline.py` doesn't currently
  measure graph-optimised (`pose_map`) accuracy at all -- only raw
  odometry -- which is exactly the blind spot that almost let the
  wiring ship before an end-to-end check caught the problem.
- **F5 (performance)**: the native graph backend
  (`pyslam/graph/backend_native.py`) still uses dense finite-difference
  Jacobians via `scipy.optimize.least_squares`. This dominates runtime on
  anything past ~100 nodes. Analytic SE(3) Jacobians + sparse solve would
  fix it; not attempted this session.
- **corridor_v2's baseline needs a full run**: currently only 1 seed,
  220/480 frames (see `baseline_store.json`, `WP_A3_Findings.md`). Run
  `pyslam.tools.baseline --scenario corridor_v2 --seeds 1 2 3 4 5` as a
  background/offline job (each seed takes ~3-4 min with the current f2f
  backend; expect worse with f2m until BA is added).

**Phase 2 (memory management / LTM)** should know:
- `pyslam/memory/memory.py`'s STM/WM work; rehearsal is dormant (only
  fires with a loaded vocabulary) and orphans nodes when it does fire; no
  LTM exists yet.
- Retrieval (`pyslam/vpr/raw_match.py`) is O(|WM|) per keyframe --
  brute-force matching against everything in working memory. This is
  fine at Phase 1's scale (tens to low hundreds of keyframes) but won't
  scale to a long session without Phase 2's memory management.
- The original plan (`Phase1_Plan.md`, section on WP-B6, not present in
  this handoff bundle -- see `WP_B6_Findings.md` for the caveat) wanted
  to instrument retrieval rank / likelihood / Bayes trace on `corridor_v2`
  BEFORE Phase 2 design starts, to confirm or retire two hypotheses about
  why the original (broken) corridor scenario failed to close loops. That
  instrumentation was never built. Worth doing early in Phase 2 rather
  than skipping straight to implementation, per this project's own
  established practice of measuring before building.
- `PipelineResult.cross_session_merges` and `Node.session_id` (WP-B3)
  give Phase 2 a natural hook for session-aware LTM/relocalization design
  -- a session boundary is already a first-class concept in the data
  model now, not something Phase 2 has to invent.

## 7. File map (what's where)

```
pyslam/
  core/          types, config, lie algebra, PnP flag selection, logging,
                 pnp_info.py (WP-B2 info matrix, validated but not wired in)
  frontend/      features.py (ORB), odometry.py (F2F, DEFAULT),
                 odometry_f2m.py (F2M, opt-in, see section 5.2),
                 local_ba.py (WP-B1 sliding-window bundle adjustment)
  memory/        STM/WM (memory.py)
  vpr/           vocabulary, BoW index, retrieval scoring, likelihood
  loop/          Bayes filter, geometric verifier
  graph/         posegraph.py (wrapper), backend_native.py (scipy LM,
                 default), backend_gtsam.py (optional, needs gtsam pip pkg)
  mapping/       point cloud assembly + PLY export
  sensors/       realsense.py (hardware, native .bag record/playback),
                 synthetic.py (renders from tests/synth/world.py),
                 bagfile.py (legacy .npz format, kept for old bags only)
  tools/         metrics.py (WP-A1), mutation_check.py, baseline.py,
                 evaluate.py (legacy ATE), env_probe.py, bundle.py,
                 wpb6_trace.py (WP-B6 retrieval/Bayes instrumentation)
  pipeline.py    the whole thing wired together

tests/
  gates/test_g0.py   pyslam.selftest's actual checks (15 of them)
  synth/world.py     renderer, trajectory generators, validate_scenario()
  synth/scenarios.py the 5 fixtures (WP-A2)

run_synth.py     synthetic pipeline demo/eval, --scenario {5 fixtures}
run_slam.py      live RealSense capture (--record for native .bag)
run_bag.py       replay a .bag (native) or .npz (legacy)

Phase1_Plan.md         original research/architecture doc (read for "why")
WP_A3_Findings.md      baseline-freeze findings writeup
WP_B_F5_Findings.md    F5 (graph backend perf) fix findings writeup
WP_B6_Findings.md      retrieval/Bayes trace instrumentation findings
WP_B1_Findings.md      local bundle adjustment findings (+ new capacity/eviction issue found)
WP_B2_Findings.md      PnP info matrix findings (oracle-validated, not wired in -- see why)
SYSTEM_SUMMARY.md       <- this file
pyslam/tools/baseline_store.json   frozen numbers
```

## 8. Honest caveats for whoever continues this

- **No RealSense hardware was available in this environment at any
  point.** Every hardware-facing code path (`realsense.py`,
  `env_probe.py`'s device-query branches) is reviewed and unit-tested
  where possible (e.g. `infer_imu_part` is a pure function, tested) but
  NOT exercised against a live D435i. Run `env_probe.py` on the real
  hardware before trusting the intrinsics/baseline/IMU-rate assumptions.
- **corridor_v2's baseline is the least-confident number in the store**
  (1 seed, partial run) -- see section 6.
- **f2m is not production-ready** -- see section 5.2. Don't flip the
  default without adding local BA first.
- Every other claim in this document is backed by a real, reproducible
  test run in this repo's `git log` -- if in doubt, re-run
  `pyslam.selftest` and read the relevant commit.
