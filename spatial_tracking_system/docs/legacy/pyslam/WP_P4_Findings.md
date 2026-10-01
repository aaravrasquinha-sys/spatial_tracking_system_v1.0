# WP-P4 Findings: Phase 4 robust back-end + proximity

## What this covers

Phase 4 (architecture doc section 5, P4): "GTSAM iSAM2 for incremental
optimisation instead of batch; Dynamic Covariance Scaling or switchable
constraints on loop factors; real estimated information matrices from
the verifier's Hessian instead of heuristics; local-space and
local-time proximity detection."

## Honesty notes up front

1. **iSAM2 was NOT attempted.** `pip install gtsam` pulls `numpy<2.0`,
   which conflicts with the numpy 2.x this entire codebase (every gate
   already written, hnswlib, everything) depends on. Downgrading numpy
   globally to get one gate check risked regressing everything already
   built and verified, for a check that needs its own isolated
   environment to do safely. Flagged explicitly as not attempted, not
   silently skipped -- see "What's still open" below.
2. **Gate G4's hardware-bag-specific numbers** (proximity recall on
   `corridor_out_back` specifically) are scored against synthetic
   surrogates with real measured numbers, same situation as every other
   gate in this project.
3. **Proximity's `corridor_v2` run was too slow for this session's
   interactive budget** -- O(|WM|²) candidate scanning plus a
   `GeometricVerifier.verify()` call (PnP+RANSAC in two directions) per
   candidate pair adds up fast on corridor_v2's denser keyframing.
   Scored on `square6dof` instead; see section 4.

## 1. Real information matrices -- WP-B2's blocker, finally closed

This is the headline result. WP-B2 (Phase 1B) built and oracle-validated
`pnp_info.py`'s Hessian-derived information matrix, wired it into
odometry, and found it made graph optimisation **functionally inert**:
properly-scaled odometry info (diagonal in the tens of millions,
correctly reflecting 200-350 real inliers) against the still-heuristic
loop-link info (capped at 50) created a ~300,000x mismatch, so the
optimiser stopped trusting loop closures almost entirely. WP-B2
correctly reverted rather than ship that.

**What closed it**: extending `pnp_info.py` with a second function,
`pnp_info_matrix_direct_frame`, for `verify.py`'s OTHER PnP direction
(see section 2), and wiring `pnp_info_matrix`/`_direct_frame` into
`odometry.py`, `odometry_f2m.py`, AND `verify.py` together, behind a
single `cfg.use_hessian_info` flag covering all three call sites at
once -- deliberately not three independent flags, because flipping only
one again would reproduce WP-B2's exact failure.

**Measured end-to-end, not just unit-tested** (same discipline WP-B2
itself insisted on -- "measuring end-to-end... surfaced something the
component-level oracle couldn't see"), on two independent fixtures:

| Fixture | Heuristic after-graph ATE | Hessian-info after-graph ATE |
|---|---|---|
| `square6dof` | 0.944cm | **0.532cm** |
| `room_orbit` | 4.173cm | **3.588cm** |

Not "functionally inert" this time -- genuinely better on both, because
both sides of every `optimize()` call are now consistently scaled. Gate
G4.4 checks this directly (asserts Hessian-info ATE isn't dramatically
worse than heuristic, which would indicate the same failure mode
recurring); both fixtures pass with real improvement, not just
non-regression.

`use_hessian_info` defaults to `False` (opt-in, same pattern as
`odometry_backend`/`retrieval_backend`) -- this result is a strong case
for eventually flipping it, but that's a deliberate follow-up decision
for whoever reviews this, not something this session unilaterally
defaults on.

**`pnp_sigma_px=1.25` is reused directly from WP-B2's own calibration**
(static_60s, NEES-based) rather than independently re-calibrated for
verify.py's loop-link context. Justified (both odometry's and verify's
PnP calls measure the same underlying pixel-localisation noise process;
what differs -- inlier count, point geometry -- is already captured by
the Hessian itself), but flagged as worth an independent direct check,
not asserted as beyond doubt.

## 2. The direction bug WP-B2's own methodology predicts, caught before shipping

`verify.py`'s bidirectional check calls PnP in two directions and keeps
whichever gives more inliers. Direction 1's accepted `T_ab` is the
**inverse** of its raw PnP solve (same case `pnp_info_matrix` was built
for). Direction 2's accepted `T_ab` **is** its raw PnP solve directly --
using `pnp_info_matrix` there would silently apply an Adjoint transform
meant for a pose that was never inverted, exactly WP-B2's own bug class
("an earlier version of this derivation got the adjoint's placement
wrong... caught by the Monte Carlo oracle, not by inspection").

Built `pnp_info_matrix_direct_frame` for direction 2 (no Adjoint step)
and gave it its own Monte Carlo oracle (`check_pnp_info_direction_oracle`
in gate G4), checking NEES against `T_cam_obj`'s own ground truth
directly rather than its inverse -- the exact distinction that matters.
**Mean NEES = 6.095** over 600 trials (chi2(6) mean = 6.0) -- confirms
the two functions are correctly NOT interchangeable, verified
numerically rather than trusted from the docstring's algebra.

## 3. Dynamic Covariance Scaling (`backend_native.py`)

Added as an alternative to the existing static Huber kernel for loop
links, opt-in via `cfg.loop_robust_kernel` (default `"huber"`,
unchanged). Formula: `s = min(1, 2*Xi/(Xi+chi2))`, `Xi = 6` (the
factor's DOF, i.e. the expected chi-square for a genuinely correct
factor). Computed fresh from the CURRENT residual on every
`_factor_residual` call -- since `scipy.optimize.least_squares` calls
this at every trial iterate, this is naturally an iteratively-reweighted
scheme with no separate outer loop needed.

**Measured directly against gate G4's own numbers**, injecting one
5m-wrong loop link into `square6dof`'s otherwise-real graph:

| Condition | After-graph ATE |
|---|---|
| Clean (no wrong loop) | 0.944cm |
| Huber + 1 wrong loop | 4.806cm (5.1x worse) |
| **DCS + 1 wrong loop** | **0.933cm** (essentially unchanged) |

DCS doesn't just clear the gate's 15%-of-clean bound (1.085cm) -- it's
within noise of clean entirely, while the existing static Huber kernel
lets the same wrong loop through to a 5x ATE blowup. A real, measured,
substantial difference, not a marginal one.

10%-wrong-loops check (G4.2) also passes: 0.931cm vs. clean's 0.944cm,
comfortably under the 50%-of-clean bound.

**Mutation 8** (`check_dcs_scale_formula`, per the project's standing
rule): checks the formula's own defining properties directly -- scale=1
at chi2=0, monotonically non-increasing, ->0 for gross outliers -- so a
sign or placement error (the DCS equivalent of WP-B2's Adjoint bug)
would be caught structurally, not just by one favourable fixture run.

**Not implemented for the GTSAM backend** (`backend_gtsam.py` still only
takes `huber_delta`) -- moot in this environment since GTSAM isn't
importable here (see honesty note 1), but a real gap if the target
machine's GTSAM path is used instead of native.

## 4. Proximity detection (`proximity/detect.py`, wired into `pipeline.py`)

Geometry-only candidate generation: WM node pairs whose CURRENT
`pose_map` positions are within `proximity_radius_m` (default 0.5m),
excluding pairs too close in creation-index (`proximity_min_index_gap`,
default 15) to be a meaningful revisit. Every candidate still goes
through the exact same `GeometricVerifier.verify()` appearance-triggered
loop closure uses -- proximity widens WHICH pairs get proposed, it does
not relax what counts as a verified link. Accepted links get the SAME
post-optimisation NEES-rollback rigour as loop closure (`_accept_link`,
extracted as a shared helper from what was `_try_loop_closure`'s
tail -- both paths now go through identical acceptance logic, not a
lighter-weight proximity variant).

Opt-in via `cfg.proximity_enabled` (default `False`). On `square6dof`:
**7 real proximity links found**, all NEES~0.00 (clean, geometrically
consistent), none of them the one node pair the appearance-based
Bayes/retrieval path already found via the deliberate loop -- these are
genuinely different, additional revisits appearance-based retrieval
either didn't attempt or didn't score highly enough to trigger on.
Combined with the one real loop closure: 8/54 of this fixture's own
generously-defined "true close pairs" (same ground-truth methodology as
WP-P3's gate) found, **zero false positives** (gate G4.5).

**`corridor_v2`'s full run timed out** under this session's interactive
budget with proximity enabled -- O(|WM|²) candidate scanning plus a
verify() call per candidate is a real cost that adds up on a fixture
with denser keyframing than `square6dof`. Scored on `square6dof`
instead (honestly flagged, same pattern as WP-P3's own `corridor_v2`
partial-run precedent). A spatial index (k-d tree / grid) over WM
positions would remove the O(|WM|²) scan cost -- not attempted, out of
this session's scope, same category of open item as WP-P3's incremental
vocabulary's own O(N·V) scaling note.

## 5. Gate G4 (`tests/gates/test_g4.py`, `python -m tests.gates.test_g4`)

5/5 checks pass:

| # | Check | Result |
|---|---|---|
| G4.1 | DCS vs Huber, single 5m wrong loop | DCS: 0.933cm (within 15% of clean's 0.944cm); Huber: 4.806cm (5.1x worse) |
| G4.2 | DCS, ~10% wrong loops | 0.931cm (well under the 50%-of-clean bound) |
| G4.3 | `pnp_info_matrix_direct_frame` oracle | mean NEES=6.095 over 600 trials |
| G4.4 | `use_hessian_info` end-to-end, 2 fixtures | both fixtures IMPROVE (not just non-regress) vs. heuristic |
| G4.5 | proximity+loop recall, `square6dof` | 8/54 true pairs found, 0 false positives |

`pyslam.selftest`: **20/20** (12 gates + 8 mutations, P4's own class
added). Default-path baseline (`square6dof`, all P4 flags off)
reproduces the exact frozen numbers -- `baseline_store.json` untouched.

## What's still open for whoever continues from here

- **iSAM2**: not attempted (numpy version conflict, see honesty note
  1). Needs an isolated environment (venv/container) with its own numpy
  pin before this can be done without risking the rest of the codebase.
- **Proximity's O(|WM|²) scan**: fine at `square6dof` scale, a real cost
  at `corridor_v2` scale. A spatial index is the natural next step if
  this needs to run on denser fixtures.
- **DCS not implemented for the GTSAM backend path**: only
  `backend_native.py` has it. Moot here (GTSAM unavailable), real gap
  on the target machine if GTSAM is the backend actually used there.
- **`pnp_sigma_px` reuse**: justified but not independently verified
  for the loop-link context specifically (section 1).
- **`use_hessian_info` staying opt-in despite a clear measured win**:
  a deliberate choice to leave the default-flip decision to whoever
  reviews this, consistent with how every other backend/kernel choice
  in this codebase has been handled -- the numbers are there, the
  decision isn't made unilaterally here.
- **Phase 1B's still-open items are unchanged**: WP-B1's f2m accuracy
  gap remains open. (WP-B2's loop-link wiring is, in effect, now
  resolved by this session's work -- see section 1.)
