# WP-M Findings: gyro/constant-velocity LOST-recovery bridge

Follows `WP_L_Findings.md` (section 7: "Bridge replacement after LOST is still the biggest
open lever, ahead of the IMU tilt prior") and `WP_K_Findings.md` (section 2: the identity
bridge accounts for essentially all of corridor_v2's missing path length and injects a
measured mean 7.7deg, max 13.1deg, of rotation error per LOST event). Everything here was
measured on the x86 dev sandbox (1 core, native pose-graph backend, no GTSAM) -- **nothing
has run on the Orin**, and the new behaviour is OFF by default: with `bridge_mode="identity"`
(unchanged) the pipeline is bit-for-bit what WP-L shipped.

## 1. What was built

| ID | Files | Change | Default |
|---|---|---|---|
| M1 | `pyslam/imu/bridge.py` (new) | Pure functions, oracle-validated first (same discipline as `preintegration.py`/`gravity.py`): `gyro_bridge_rotation` (reuses `preintegrate()` unchanged, zero bias assumed, converts body-frame rotation into camera convention via the real `R_body_cam` extrinsic -- same conjugate-transform pattern `gravity.py` already uses for a direction vector, applied here to a relative rotation); `constant_velocity_translation` (v * dt); `build_bridge_link` (combines both, falls back to Phase-0's identity behaviour PER COMPONENT, never all-or-nothing). | - |
| M2 | `pipeline.py` (`_run_loop`), `Config.bridge_mode` | New `self._last_two_ok_frames` / `self._pre_lost_velocity` tracking (a 2-sample finite difference of `OdomResult.T_rel`, relative to the CURRENT reference keyframe, frozen the instant LOST fires); the identity-bridge branch now calls `build_bridge_link` when `cfg.bridge_mode="gyro"`. `PipelineResult.bridge_events` records `(a_id, b_id, dt, used_gyro, used_velocity)` per bridge, empty unless the new mode is on. | `bridge_mode="identity"` (off) |
| gates | `tests/gates/test_gm.py` (8 checks, in `selftest`), 2 new mutants in `phase_a_mutation_check.py` | see section 4 | - |

`Config` also gained `bridge_gyro_min_samples` (5), `bridge_rotation_sigma_rad` (0.05 rad,
~2.9deg -- NOT independently calibrated, same open-item status `gravity_prior_sigma_tilt_rad`
already carries), `bridge_velocity_sigma_base_mps` (0.3) and `bridge_velocity_sigma_growth_mps_per_s`
(1.0, the reason translation's info WEAKENS with elapsed gap time while rotation's does not --
see `pyslam/imu/bridge.py`'s own docstring).

## 2. Why this design

Two independent things stay available across a LOST gap even though visual tracking failed:

* **The gyroscope.** Integrating angular velocity needs no features or depth. `preintegrate()`
  (P5.0, already oracle-validated) is reused completely unchanged -- this module only adds the
  camera-frame conversion. Zero gyro bias is assumed, honestly: this project has no bias
  ESTIMATE yet (P5.4 not started), and a wrong assumed bias is a small, bounded error over a
  single bridge gap (seconds), not the multi-second drift a real bias estimate would need to
  worry about.
* **The last odometry-measured velocity**, extrapolated at constant velocity across the gap.
  Deliberately NOT double-integrated accelerometer -- accel bias and (pre-P5.4) the lack of a
  trustworthy gravity subtraction both grow with dt^2, and there is no oracle-validated
  component in this codebase yet trustworthy enough to use blind for that. Constant velocity is
  honestly a WEAKER assumption during exactly the kind of motion that tends to cause LOST
  (fast/erratic) -- which is why, unlike rotation, its info weight is deliberately WIDENED
  (weakened) with elapsed gap time rather than held fixed.

Both halves fail open, per component, independently: a missing/short gyro window degrades ONLY
the rotation block to Phase 0's identity value; a missing pre-LOST velocity (e.g. LOST on the
very first frame after a keyframe) degrades ONLY translation; both missing reproduces the old
bridge bit-for-bit (`GM.4`).

## 3. Results

### 3.1 Pure-function rotation accuracy (synthetic, exact oracle)
For a truly constant body-frame angular velocity, Euler integration over any step count is
EXACT (each step is `so3_exp(w*dt)` about the same fixed axis, and matrix exponentials about a
common axis commute), so `GM.1` is not an approximate check: with a real, non-identity rig
rotation (`T_BODY_CAM` from the synthetic fixtures), the estimated rotation matched the
analytic answer to **7e-15 rad** (floating-point noise).

### 3.2 End-to-end: a real forced LOST on square6dof (GM.8)
A 10-frame featureless stretch (solid-black RGB, real depth/IMU/ground-truth untouched) forces
three chained LOST bridges (5 blank frames each triggers `lost_consecutive_frames=5`; the
degenerate frame that declares LOST becomes the new, still-degenerate reference, so the first
gap chains into a second before real content resumes -- an honest consequence of Phase 0's own
"restart from the current frame" recovery, not a test artifact). Composing all three bridges
node 5 -> node 24 (0.9s + 0.5s + 0.5s = 1.9s total, seed 1, no added sensor noise):

| | identity bridge | gyro bridge |
|---|---|---|
| rotation error vs ground truth | 29.2 deg (the entire true rotation) | **7e-15 deg** |
| translation error vs ground truth | 0.440 m (the entire true displacement) | **0.259 m (-41%)** |

Every one of the three bridges had a real IMU window (`used_gyro=True` on all three); only the
first had a pre-LOST velocity estimate (`used_vel=True` once, `False` twice -- the second and
third bridges start from an already-degenerate reference with no real tracking behind it, so
translation on those two stays at Phase 0's identity value, per the fallback design).

### 3.3 corridor_v2, first 300 frames, seed 1, proximity on (real LOST events, not synthetic)
Same run, `bridge_mode=identity` vs `bridge_mode=gyro`, everything else identical
(`baselines/M_corr_identity.{json,md}` / `M_corr_gyro.{json,md}`, `--compare` output below):

| | identity | gyro | ratio |
|---|---|---|---|
| LOST events / bridges | 9 | 9 | - |
| anchored ATE, online poses | 120.3 cm | **96.9 cm** | x0.81 |
| vertical error RMS | 0.782 m | **0.123 m** | **x0.16** |
| horizontal error RMS | 0.915 m | 0.961 m | x1.05 |
| anchored ATE, AFTER finalize() | 42.3 cm | **52.4 cm** | **x1.24 (worse)** |

**Update (this session): the info-matrix-overconfidence hypothesis below is FALSIFIED --
re-read section 3.3.1 before drawing any conclusion from the table above.** The ONLINE
(pre-finalize) numbers are real and hold up; the post-finalize regression is real too, but its
cause is NOT what this section originally guessed.

~~**Honest finding, not yet resolved:**~~ The ONLINE (pre-finalize) trajectory improved
substantially, especially vertical error (which WP-K hypothesized was largely bridge-injected
rotation -- this is the first direct evidence for that hypothesis: an 84% reduction). But the
trajectory AFTER the closing graph optimization got WORSE (42.3cm -> 52.4cm).

~~The likely cause: `bridge_rotation_sigma_rad=0.05` (weight 400) makes the gyro bridge far
more confident than the old `bridge_link_info_scale=1e-3` identity bridge ever was.~~ **This
was tested directly (section 3.3.1) and does not hold.**

#### 3.3.1 Follow-up: the overconfidence hypothesis was tested and is FALSE

Section 6's own suggested first experiment (widen `bridge_rotation_sigma_rad`) was run, plus
three further points to fully bracket the hypothesis -- same corridor_v2/300-frame/seed-1 setup,
`bridge_mode=gyro`, only the listed override changed each time
(`baselines/M_corr_{rot2x,rot4x,vel3x,nearzero}.json`):

| run | rotation weight (1/sigma^2) | translation weight | aATE_final | vert_rms_m |
|---|---|---|---|---|
| gyro, default | 400 | ~1.6 (dt-dependent) | 52.4 cm | 0.123 |
| rotation sigma x2 (0.10 rad) | 100 | ~1.6 | **52.4 cm** | 0.124 |
| rotation sigma x4 (0.20 rad) | 25 | ~1.6 | **52.4 cm** | 0.123 |
| translation sigma x3 (base 0.9, growth 3.0) | 400 | ~0.17 | **52.4 cm** | 0.123 |
| **both set to `1/31.6^2 = 0.001` -- IDENTICAL to `bridge_link_info_scale`, identity's own weight** | 0.001 | 0.001 | **52.4 cm** | 0.123 |
| identity (reference) | 0.001 (uniform) | 0.001 (uniform) | 42.3 cm | 0.782 |

Every single point is bit-identical to the default at 52.4cm/0.123m -- **including the last
row, where the bridge's info matrix was set to numerically match `bridge_mode="identity"`'s
own weight exactly.** If overconfidence-relative-to-identity were the cause, that row should
have reproduced identity's 42.3cm. It did not move at all across a 400x span of rotation
weight and a ~2400x span of translation weight (25 down to 0.001, and separately 1.6 down to
0.001). **The regression is not caused by the bridge link's WEIGHT. It is caused by its MEAN
estimate** (the real rotation/velocity-extrapolated values vs. identity's exact zero),
independent of how that mean is weighted in the tested range.

**Working hypothesis for the actual mechanism (code-supported, NOT yet confirmed by a direct
pose trace -- see section 6):** `backend_native.py`'s `optimize()` starts every solve from
`x0 = zeros`, i.e. a perturbation of the poses **already stored in the backend** from the
previous `optimize()` call (`base[i] = self._poses[i]`) -- this is an iterative refinement, not
a fresh global solve each time. Its own code comment notes `scipy`'s `'trf'` method "falls back
toward the initial guess for unconstrained directions" on rank-deficient/underdetermined
problems -- and a LOST bridge gap is close to the definition of a weakly-constrained region.
Every online loop-closure `optimize()` call during the run includes every earlier bridge link as
an active residual (`active_links` is every link with both endpoints present, not just new
ones), regardless of `bridge_mode`. So even a bridge link weighted down to `identity`'s own
0.001 still has a *different mean* than `identity`'s exact-zero mean at every one of those
intermediate online solves, and if TRF is falling back toward its current `x0` in the
under-constrained bridge-gap direction rather than being pulled there by relative link
strength, that small per-step difference gets baked into `pose_map` before `finalize()` ever
runs -- and `finalize()` itself starts from that already-diverged state, not from a common
baseline. This would explain why *weight* doesn't matter in the tested range (a `weight`-based
mechanism should have shown at least partial recovery somewhere in a 2400x span) while the
*mean* clearly does. This is a plausible, code-supported explanation, not a confirmed one -- an
attempt to confirm it directly (diffing `pose_map` node-by-node right after `run()`, before
`finalize()`, between `identity` and the weight-matched `gyro` run) was started this session but
did not complete within the available sandbox time; it is the concrete next step, not a done
one -- see section 6.

## 4. Validation
* `selftest`: **54/54** in 165.7s (13 gate + 3 trajectory + 10 Phase-A + 10 Phase-B + 8 Phase-C
  (WP-M) + 10 mutation). G2 re-run after the shared `pipeline.py` edit: 5/5, unaffected.
* G-M gates (`tests/gates/test_gm.py`, 8 checks): rotation oracle (exact to float precision, incl.
  an identity-rig sanity check that would catch a reversed conjugate-transform bug the general
  case alone might not); gyro/velocity fallback to `None`; the full-fallback-equals-Phase-0-bridge
  regression guarantee (bit-for-bit); independent per-component upgrade; translation info widens
  monotonically with gap time while rotation info does not; the end-to-end forced-LOST comparison
  in 3.2.
* Source-level mutation check (`phase_a_mutation_check`): **9/9 caught**, including the 2 new
  WP-M mutants (`ROTFLIP`: conjugate-transform reversed; `NOGROW`: translation info stops
  widening with dt) -- both caught on the first attempt, no gate fixes needed this time.
* `run_synth.py --scenario square6dof --config-override bridge_mode=gyro`: bit-identical result
  to the existing frozen baseline (ATE anchored 0.74cm after finalize) -- confirms the new code
  path is a true no-op on any run with zero LOST events, as designed.
* Follow-up session (section 3.3.1): 4 additional corridor_v2/300-frame/seed-1 comparisons at
  different `bridge_rotation_sigma_rad`/`bridge_velocity_sigma_*` settings, each independently
  reproduced end-to-end (not just read from a saved file) before being reported. No code
  changed this session -- `pipeline.py`/`bridge.py`/`config.py` are identical to what shipped
  with the initial WP-M commit; `selftest` (54/54) and `phase_a_mutation_check` (9/9) both still
  pass unmodified.

## 5. Not validated / limits
* No hardware. All numbers are synthetic or corridor_v2 (still one seed, one fixture, native
  backend, no GTSAM).
* corridor_v2's comparison used only the first 300 of 480 frames (sandbox time budget); the
  full run, and seeds 2-5, are unmeasured.
* `bridge_rotation_sigma_rad` / `bridge_velocity_sigma_*` are NOT calibrated against real gyro
  noise or real velocity-estimate error -- they are the values this shipped with, flagged as an
  open item exactly like `gravity_prior_sigma_tilt_rad` already is.
* The section 3.3 post-finalize regression is a real, reproducible, MEASURED result on this
  fixture -- not a guess. Its cause is NOT info-matrix overconfidence (section 3.3.1 tested
  this directly across a >2000x weight span, including matching identity's own weight exactly,
  and the regression never moved). The current best explanation (the native backend's iterative
  `x0`-from-current-pose solve, per section 3.3.1) is a code-supported hypothesis, not a
  confirmed root cause -- the direct pose-trace diff that would confirm it has not completed.
* Only tested against `proximity_enabled=true` (WP-K/L's own standard baseline setting); not
  tested against `gravity_prior_enabled=true` together (the two opt-ins have not been run
  simultaneously).

## 6. Suggested next steps, in order
1. ~~Resolve the section 3.3 regression before recommending `bridge_mode=gyro` for general
   use. Cheapest first experiment: widen `bridge_rotation_sigma_rad`...~~ **Done, and it
   disproved the hypothesis it was meant to test (section 3.3.1).** Widening rotation sigma
   (x2, x4), widening translation sigma (x3), and matching BOTH to identity's own exact weight
   all left `aATE_final` bit-identical at 52.4cm. The cause is the bridge's non-identity MEAN,
   not its confidence.
2. **Confirm the `x0`-from-current-pose mechanism directly.** Run `identity` and a
   weight-matched `gyro` (section 3.3.1's last row) side by side, and diff `pose_map` for every
   node right after `run()` returns (before calling `finalize()`). If the two have already
   diverged online despite matched weight, that confirms the mean-not-weight mechanism and
   narrows the fix to either (a) not including old bridge links as active residuals in later
   online `optimize()` calls, or (b) starting `finalize()`'s solve from a fresh `x0` rather than
   the run's accumulated online state. If they have NOT diverged online, the divergence must be
   introduced somewhere in `finalize()` itself and the mechanism above is wrong -- look there
   next. A first attempt at this diff was started this session and did not finish within the
   sandbox's time budget; it needs a shorter fixture (fewer frames, or a scenario with fewer
   loop closures) to fit the available turn budget, not a re-run of the same 300-frame corridor.
3. Once the mechanism is confirmed, the fix is probably structural (see 2's two options) rather
   than a `Config` calibration constant -- do not spend further effort tuning
   `bridge_rotation_sigma_rad` / `bridge_velocity_sigma_*`, section 3.3.1 already rules that out
   as sufficient.
4. Full corridor_v2 (480 frames) and corridor_lap13, both bridge modes, once (2)/(3) are
   resolved -- the online-vs-final split needs the FULL run, not a 300-frame slice, to say
   anything about loop-closure interaction with any confidence.
5. Real hardware: confirm `R_body_cam` (WP-T3) is right on the actual rig before trusting any
   of this on real data -- a wrong extrinsic would silently corrupt the rotation half exactly
   the way WP-P5.2's own gravity prior was found to be corrupted before that fix.
6. Then the IMU tilt prior (Phase C, WP-P5.2) on real data, per WP-L's own ordering -- vertical
   error was already the tilt prior's stated justification, and section 3.2/3.3 here now give a
   first real number for how much of it the bridge alone can recover.

**Bottom line for now: do not recommend `bridge_mode="gyro"` for a trajectory that will be
`finalize()`d until step 2/3 above resolve the post-finalize regression.** The ONLINE
improvement (vertical RMS -84%, online ATE -19%) is real and reproducible, and the whole
feature stays off by default (`bridge_mode="identity"`) either way, so nothing here changes
default behaviour -- this is scoping an opt-in, not shipping a default.
