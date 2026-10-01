# WP-P5 Findings: Phase 5 (IMU fusion) -- P5.0 only

## What this covers, and what it deliberately does NOT cover yet

This session built and validated **only WP-P5.0**: IMU preintegration,
the blocking prerequisite identified in this project's own Phase 5
planning notes. It does **not** touch the pipeline, the graph backend,
`SyntheticSource`, or any real sensor data path -- all of that is
P5.1-P5.4, still ahead. `Frame.imu` and `SensorSource` are untouched;
nothing in `pyslam/pipeline.py` calls anything in this section. This is
foundation work, not integration, and is presented as such.

## 1. Why P5.0 had to come first

Two blocking problems were identified before any of this was written:

1. **`SyntheticSource._imu_between` cannot validate anything Phase 5
   needs.** It sets `a_specific = R.T @ (-g_world)` -- gravity reaction
   only, with zero acceleration from actual motion, ever. No
   preintegration, bias-convergence, or gravity-estimation check could
   ever be exercised against it; it would pass trivially regardless of
   whether the code under test was correct.
2. **The `Link`/`GraphBackend` contract doesn't fit preintegrated IMU
   factors.** A 6-DoF relative pose + 6x6 info matrix has no room for
   the velocity/bias states and 9-dimensional residual an IMU factor
   needs. This is real, not yet addressed, and is P5.4's problem, not
   P5.0's -- flagged here so it isn't forgotten.

P5.0 solves the first blocker only. The second is still fully open.

## 2. The IMU-consistent trajectory generator (`tests/synth/imu_world.py`)

A trajectory where position is a closed-form sum of sinusoids (exact
velocity/acceleration by direct differentiation, no finite-differencing
anywhere) and body-frame angular velocity is likewise a closed-form
function (exactly what a gyro measures -- no need to differentiate an
orientation function at all). Orientation `R(t)` is obtained by RK4
integration of `R' = R @ skew(w_body(t))` at a fixed 5000Hz step,
accepted as ground truth (its own error is many orders of magnitude
below the ~1st-order discretisation error the actual oracle is
measuring -- see section 3's convergence numbers, which bottom out
following a clean O(1/hz) pattern with no sign of an RK4-floor
contaminating the result even at 1600Hz test rate).

**Performance note, found and fixed during development**: an initial
version re-integrated from t=0 on every single query, which would have
made generating one IMU stream O(n^2) in the number of samples (every
`rotation(t)` call redoing the whole sweep from zero). Fixed by
`integrate_checkpoints()`: one continuous forward pass records
orientation at every requested checkpoint as it's crossed, shared by
both `rotation(t)` (single checkpoint) and `sample_imu()` (one
checkpoint per output sample). Generating 5 IMU streams (100-1600Hz,
~2700 samples total) over a 0.5s window: 1.3s.

## 3. Preintegration (`pyslam/imu/preintegration.py`) -- Gate G5

Standard on-manifold discrete (Euler, not midpoint/RK4) preintegration
between two keyframe times, following Forster et al.'s IMU-factor
formulation as used in GTSAM: `delta_R`/`delta_v`/`delta_p` accumulated
from bias-corrected gyro/accel samples, composed with a starting
`(T_wb0, v0)` and gravity to predict `(T_wb1, v1)`. Convention matches
this codebase's own `base_pose @ exp(xi)` pattern throughout
(`backend_native.py`, `local_ba.py`): gyro measures body angular
velocity such that `R(t+dt) = R(t) @ so3_exp(w_body*dt)`, a RIGHT
multiplication.

**Bias-correction Jacobians are computed via central finite difference
on the bias, not hand-derived analytically.** Deliberate, not a
shortcut: WP-B2's own findings document a hand-derived Adjoint-
placement bug that passed visual inspection and was only caught by a
Monte Carlo oracle. The equivalent risk for IMU preintegration is
getting the recursive analytic bias-Jacobian propagation formula wrong
-- exactly the class of bug this project's established practice
prefers to avoid by construction. The finite-difference Jacobians cost
12 extra integration passes per call (real, if this is ever wired into
a real-time path at P5.4) but are correct by construction given
`_integrate` itself is correct -- which Gate G5 verifies independently.

### Gate G5 (`python -m tests.gates.test_g5`) -- 3/3 checks

| # | Check | Result |
|---|---|---|
| G5.1 | Convergence: noise-free preintegration vs RK4 ground truth, 100-1600Hz | 200Hz trans_err=2.61cm; error ratio between consecutive doublings stays in [1.5,2.5] at every step (textbook O(1/hz) first-order convergence) |
| G5.2 | Bias-Jacobian first-order correction vs direct re-integration at realistic MEMS bias magnitudes | rot_err=0.00005deg, v_err=0.11mm/s, p_err=0.014mm at \|bg\|=0.027rad/s, \|ba\|=0.137m/s2 |
| G5.3 | Degenerate input (empty/single-sample IMU) | correctly degenerates to identity, no crash or fabricated motion |

G5.1 is the check that actually matters most: it doesn't just assert
"error is small" (which a lucky bug could satisfy at one specific
rate), it asserts the error **shrinks at the rate a correct first-order
integrator should**, across 5 independent sample rates. A wrong sign,
a swapped gyro/accel column, or a wrong composition order would not
produce this clean halving pattern -- it would produce an error that
doesn't shrink with rate, or shrinks at the wrong order, or is simply
enormous regardless of rate. This is a stronger correctness signal than
a single-point tolerance check, in the same spirit as this project's
existing oracle tests (Monte Carlo NEES for `pnp_info_matrix`,
ground-truth convergence for `local_bundle_adjust`).

### Mutation 9 (`pyslam/tools/mutation_check.py`, covered by `pyslam.selftest`)

Per the project's own standing rule ("any new subsystem ships its own
mutation class before trusting the gate"): injects a gravity-sign
inversion into `compose_prediction`'s gravity term and confirms it
produces a gross (>1m) error over a 0.7s window, vs the correct
sign's <5cm. **This is the one piece of P5.0 that IS covered by
`pyslam.selftest`** (21/21 now, was 20/20) -- G5 itself, like G2/G3/G4
before it, is standalone and not yet wired into `selftest` (an
existing, still-open gap this session did not fix -- see
SYSTEM_SUMMARY.md).

## 4. WP-P5.2: loose coupling -- gravity/tilt prior

**What this covers**: `pyslam/imu/gravity.py` (quasi-static detection,
gravity-direction estimation, tilt-only correction) and its wiring into
`pipeline.py` as an opt-in `Link(kind="prior")`, per architecture doc
section 5's own P5.2 description. Deliberately the simpler half of
Phase 5's coupling spectrum: no velocity/bias states, no
`GraphBackend` changes -- confirmed `backend_native.py`'s
`_factor_residual` treats `kind="prior"` identically to `"odom"` (only
`"loop"` gets robust-kernel treatment), so this plugs into the
EXISTING optimiser with zero backend code changes.

### 4.1 The math, oracle-validated (Gate G5.4)

`tilt_correct_rotation` finds the minimal body-frame rotation that
makes a vision pose's own implied "up" direction match an independent
accelerometer observation, while disturbing yaw as little as possible.
Validated with a KNOWN injected error, decomposed into tilt and yaw
components: a 5deg injected tilt is corrected to a 0.0000deg residual
(exact, as the algebra guarantees), and a 15deg injected yaw is
PRESERVED at 15.01deg -- the property that actually matters here,
since gravity fundamentally cannot observe yaw and a correct
implementation must not silently invent that information. The
anisotropic info matrix (`w_tilt*(I-uu^T) + eps*uu^T`, `eps=1e-6` to
keep it positive-DEFINITE, not merely PSD, for `backend_native.py`'s
`cholesky` call) was checked directly: eigenvalues came back
`[1e-6]*4, ~821, ~821` for `sigma_tilt=2deg` -- position and yaw
correctly near-zero, both tilt directions correctly equal to
`1/sigma_tilt^2`.

### 4.2 Pipeline wiring

`cfg.gravity_prior_enabled` (default `False`). IMU samples are
accumulated per-frame into a since-last-keyframe window, reset at every
keyframe. The first quasi-static window ESTABLISHES the gravity
direction; every later keyframe's own window is then checked
independently and, if quasi-static, produces a `Link(kind="prior")`
between node0 and that keyframe.

**Directly tested first** (bypassing the frame loop, crafting quasi-
static IMU windows and calling `Pipeline._try_gravity_prior` directly,
before any fixture existed to drive it through a real run): gravity
direction established exactly matching the crafted window's own input
direction; a node with an injected 3.33deg tilt error correctly got a
prior link; running `graph.optimize()` with that single prior link
pulled the tilt error from 3.33deg to 0.0deg. **This test used an
implicit identity `R_body_cam`** (the crafted IMU samples were built
AS IF the accelerometer measured directly in the camera frame) -- which
made it a valid test of the OPTIMISATION math, but not of the real
camera/body frame relationship, which is exactly what section 4.3
below found wrong once a real fixture exercised the full path.

At the time this wiring was first built, no fixture existed that could
drive it through a real `run_synth.py` run at all -- see section 4.3
for why (none of the other 5 fixtures can ever trigger it) and how that
gap was closed.

### 4.3 The fixture: `gravity_init_square6dof`, and a real bug it found

Needed because none of the other 5 fixtures can ever trigger a gravity
prior through a real run. `square6dof`/`room_orbit`/`corridor_v2`/
`aliasing_rooms` all have continuous 6-DoF motion from frame 0 by
design (deliberately, to avoid the original Phase 0 F2 bug --
flat/yaw-only fixtures), so their gyro rate exceeds
`gravity_prior_gyro_thresh_rad_s` (0.08 rad/s, ~4.6deg/s) essentially
immediately. More surprising: `static_60s` can't trigger it either,
despite being the one fixture that's supposed to be stationary --
because a genuinely static camera never crosses the keyframe-creation
motion threshold (`baseline.py`'s own long-standing note: "static_60s
never crosses... only ever sees ONE ground-truthed node"), and
gravity-prior windows are defined relative to KEYFRAME-to-KEYFRAME
intervals. One keyframe means zero "since last keyframe" windows to
ever establish or use a prior against.

Built:
smooth constant-velocity non-rotating translation (30 frames,
comfortably crossing `cfg.keyframe_trans_m` while staying quasi-static
by the gyro-rate/accel-variance criteria that actually matter) +
`square6dof`'s own unchanged 6-DoF loop. Two real snap bugs were found
and fixed while building it, both instances of documented, known
failure modes recurring in new code rather than novel ones:

- The lead-in's approach direction was originally hardcoded toward
  world +x, assuming the main loop starts heading "toward its second
  waypoint". Wrong: `square6dof`'s own CLOSED LOOP path has a BLENDED
  starting tangent (Catmull-Rom wraps the last waypoint's incoming
  direction into the first point's tangent), giving an actual starting
  heading with a substantial off-axis component. `validate_scenario`
  caught a 30deg snap at the phase boundary. Fixed by measuring the
  loop's own actual starting heading directly instead of assuming it.
- Fixing that surfaced the SAME class of bug a second time, in a place
  this project has already named and fixed once before (Phase 0's F3):
  `poses_from_path`'s own look-ahead clamps (rather than wraps) at the
  LAST sample of any `loop=False` path, causing a heading collapse
  right at the phase boundary. Both the static-hold and translate
  phases hit this at their own tails. Fixed the same way F3 itself was
  scoped to be fixed for closed loops: generate one extra sample beyond
  what's needed, then drop it, so the sample that actually needs a
  valid look-ahead is never the array's last element.

**A third, more significant bug was found once the fixture actually
ran through the real pipeline**: `pose_map`/`pose_odom` are CAMERA-
frame poses (confirmed via this project's own `wpb1_eval.py`/
`baseline.py` convention: `node_gt[i] @ T_BODY_CAM`), while `Frame.imu`
measures the physical BODY frame -- different coordinate systems
related by the fixed `T_BODY_CAM` rotation, not the same frame under
different names. `pyslam/imu/gravity.py`'s functions were built (and
component-oracle-tested) assuming they were the same frame -- correct
for the isolated oracle test (which never involved a real camera/body
distinction at all), silently wrong the moment it touched real
`pose_map` values. Found because `gravity_init_square6dof` was the
first thing to exercise the FULL path (accelerometer -> established
`u_world` -> comparison against a real vision pose) -- exactly the
"component-level assumption misleads, only an end-to-end check catches
it" pattern this project's history keeps repeating (WP-B1's landmark
duplication, WP-P4's info-matrix scale mismatch, this session's own
mutation-10 near-miss in section 4.4 below).

**Fixed**: `build_gravity_prior_link` and `Pipeline.__init__` both now
take an explicit `R_body_cam` (rotation mapping a camera-frame vector
into the physical body frame). Every accelerometer-derived direction is
rotated through it before being compared against or stored alongside a
`pose_map` rotation. Measured effect on `gravity_init_square6dof`: the
constructed correction's magnitude dropped from 1.48deg (wrong,
identity-assumed frame) to 0.76deg (correct frame) -- a real, expected-
direction change (this fixture's vision-only tracking is already
accurate, so a correctly-framed correction should be small, which 0.76
deg plausibly is and 1.48deg less plausibly is).

**Still an open gap, not fully closed**: the frozen `Intrinsics`/
`Config` contract has no field for a real extrinsic, so `R_body_cam`
defaults to identity -- explicitly documented as KNOWN WRONG for this
project's own D435i rig, not a safe default, kept only so the feature
doesn't crash when nothing better is available. Every real caller
(including every synthetic-fixture one, which has
`tests.synth.world.T_BODY_CAM` on hand) must pass the real rotation
explicitly. Threading a real calibration-derived extrinsic through the
frozen contract is a bigger decision than this session made
unilaterally -- flagged for whoever continues, not resolved.

Gate G5.5 (`check_gravity_prior_end_to_end`) makes this a permanent
regression check: runs the real `Pipeline` on `gravity_init_square6dof`
with the correct `R_body_cam`, asserts a prior actually fires, and
asserts its correction stays under 2deg -- a regression in the frame
conversion would blow well past that bound (the pre-fix magnitude was
already at the 1.5deg mark with a MUCH simpler two-phase geometry;
`check_gravity_prior_end_to_end`'s own margin was set having seen that
number directly, not guessed). Gate G5: 5/5. `pyslam.selftest`: 22/22
(unaffected -- no existing code path changed).

### 4.4 Mutation 10 (`pyslam.selftest`): the mutation check itself needed a fix

Per the standing rule, a 10th mutation class validates the info
matrix's axis assignment (a swapped tilt/yaw projector -- plausible,
passes casual inspection, silently makes the optimiser trust yaw and
ignore tilt). **The first version of this check was itself broken**: it
ran a full two-node graph optimisation and found BOTH the correct and
the mutated info matrix converged to zero residual -- with only one
link and no competing constraint, `backend_native.py`'s tight solver
tolerances drive every residual direction to zero eventually regardless
of how weakly it's weighted, since nothing opposes it. The mutation
check's own sanity assertion caught this before it could ship. Fixed by
testing the info matrix's weighted residual directly (`cholesky(info) @
err`, the exact computation `_factor_residual` performs) rather than
via optimiser convergence. `pyslam.selftest`: 22/22.

### 4.5 What's still open

- **`R_body_cam` has no home in the frozen contract.** It defaults to
  identity in `Pipeline.__init__`, explicitly documented as KNOWN WRONG
  for this project's own D435i rig. Every real caller must pass the
  true rotation explicitly (synthetic callers have
  `tests.synth.world.T_BODY_CAM` on hand; real hardware would need it
  from calibration, which `env_probe.py` doesn't currently surface in
  a form this could consume). Threading a real, calibration-derived
  extrinsic through `Intrinsics`/`Config` is a bigger decision than
  this session made unilaterally -- flagged for whoever continues, not
  resolved. This generalises the "zero-lever-arm" translation gap
  already flagged in section 1 to the (arguably more fundamental,
  since EVERY IMU reading needs it) rotational part too.
- **`gravity_prior_sigma_tilt_rad` (2deg default) is not calibrated**
  against any real or synthetic accelerometer noise model -- a
  reasonable-looking guess, unlike `pnp_sigma_px`'s own NEES-based
  calibration (WP-B2). `gravity_init_square6dof` now exists to
  calibrate against, but that calibration itself wasn't done this
  session.
- **Only ever anchors to node0.** If node0's own establishing window
  turns out to be wrong (e.g. the "quasi-static" heuristic passes on a
  window with real but low-rate rotation), every later prior inherits
  that error with no mechanism to detect or correct it. A real
  limitation of the "establish once" design, not fixed here.
- **`gravity_init_square6dof` only produces ONE prior event** (node 42,
  see Gate G5.5) across its own 185 frames -- Phase C (the unchanged
  `square6dof` loop) never gets another quasi-static window once real
  6-DoF motion resumes. Sufficient to prove the wiring and the frame
  conversion both work, but this fixture alone cannot answer "does the
  prior measurably improve accuracy over many keyframes" -- that needs
  either a longer/richer quasi-static-interval fixture or a different
  validation approach (e.g. injecting multiple synthetic tilt errors
  directly and checking recovery, closer to how G4's DCS check works).
- **P5.1 (real IMU front-end: buffering, async accel/gyro alignment for
  actual hardware, NVRAM calibration gate) is still not started.**
  This session went straight to P5.2's math because it didn't need
  P5.1's hardware-specific plumbing to validate against synthetic data
  -- P5.1 becomes necessary before ANY of this can run on the D435i.
- **P5.3/P5.4 unchanged from the prior session's notes**: P5.3 still
  needs a new fixture (a physically-rendered, visually-uninformative
  transit, not `aliasing_rooms`' instantaneous cut) -- possibly
  reusable machinery from `gravity_init_square6dof`'s own lead-in
  pattern, not attempted; P5.4 is still blocked on the
  `Link`/`GraphBackend` contract gap.

