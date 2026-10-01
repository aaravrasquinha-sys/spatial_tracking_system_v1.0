# WP-B2: PnP Information Matrix -- Findings

Code state: `pyslam/core/pnp_info.py` (new, validated, kept) exists as a
correct, oracle-tested standalone module. It is NOT wired into
`odometry.py`/`odometry_f2m.py` -- both trackers still use the original
heuristic (`eye(6) * inlier_ratio * 100`). `cfg.pnp_sigma_px` does not
exist in `Config` (the wiring attempt that would have added it was
reverted). This is a deliberate stopping point, not an oversight -- see
section 3.

## 1. The derivation, and a real bug caught by the oracle

Goal: replace the heuristic odometry info matrix with one derived from
the PnP solve's own Gauss-Newton Hessian (reprojection Jacobians over
the inlier set), the way the Phase 1 plan describes ("come from the PnP
refinement's Hessian, calibrated via NEES on static_60s").

The derivation has two parts: (1) the per-point reprojection Jacobian
w.r.t. a right-tangent perturbation of the RAW PnP output `T_cam_obj`
(same convention both trackers already use for `solvePnPRansac`'s
result), giving a Gauss-Newton Hessian `H`; (2) propagating that
uncertainty through to `T_obj_cam = inverse(T_cam_obj)` -- the pose
actually used as the emitted Link's relative transform -- via the SE(3)
adjoint, since backend_native.py parameterises every pose graph variable
with a RIGHT perturbation.

Built and tested as a pure function (`pnp_info_matrix`), with a Monte
Carlo oracle as the validation method (appropriate here: unlike WP-B1's
noise-free convergence oracle, this is a STATISTICAL claim -- "this
matrix correctly predicts the scatter of PnP estimates under pixel
noise" -- so the right check is inject KNOWN Gaussian pixel noise many
times, re-solve PnP each trial, and verify the predicted information
matrix's NEES, `e^T @ Info @ e` against the true pose, averages to the
chi-square(6) mean of 6.0).

**First version was wrong, caught immediately by the oracle, not by
inspection.** Initial derivation gave `Info = Adj(T_obj_cam) @ H @
Adj(T_obj_cam)^T / sigma_px^2`. Oracle result: mean NEES=23.9 (should be
~6.0), with per-component diagnosis showing the ROTATION components
specifically wrong (translation components matched fine, ~1.0 ratio;
rotation, especially the in-plane rotation component, badly off -- one
ratio was 0.57, meaning the predicted uncertainty there was ~1.8x too
large). Isolated the bug methodically: first confirmed the base Hessian
`H` was itself correct by testing it directly against `T_cam_obj`'s own
uncertainty (no adjoint involved) -- NEES=6.12, correct. So the bug was
specifically in the adjoint-propagation step. Rather than re-derive by
hand a second time (risk of a second sign/ordering error), tested all 4
plausible variants (which pose's adjoint; transpose on the left vs.
right) numerically against the empirical Monte Carlo samples:

| formula | mean NEES |
|---|---|
| `Adj(T_obj_cam) @ H @ Adj(T_obj_cam)^T` (original, wrong) | 23.888 |
| `Adj(T_obj_cam)^T @ H @ Adj(T_obj_cam)` | 6.122 |
| `Adj(T_cam_obj) @ H @ Adj(T_cam_obj)^T` | 19.057 |
| `Adj(T_cam_obj)^T @ H @ Adj(T_cam_obj)` | 6.540 |

The second formula matches a corrected hand re-derivation exactly
(confirmed algebraically after the fact, not just numerically) and is
what's implemented. Validated across 4 independent configurations
(varying point count, noise level, pose) after the fix: NEES 5.7-6.5 in
every case, no re-tuning between them. This is now a permanent gate,
`test_pnp_info_matrix_oracle` in `tests/gates/test_g0.py` (selftest
16/16 -> 17/17).

## 2. Calibration and out-of-sample validation

`sigma_px` (the info matrix's one free scale parameter) calibrated via
NEES on `static_60s`, per the plan's own instruction. Instrumented every
tracked frame (not just keyframe-emitted links -- `static_60s`'s camera
barely moves, so F2F never actually creates a SECOND keyframe across the
whole 600-frame sequence under its own keyframe-trigger thresholds,
meaning zero odom Links exist to calibrate against; per-frame relative-
to-reference-keyframe data, compared against each frame's own ground
truth, is what's actually available and used). Result: mean NEES at
`sigma_px=1` was 9.40 over 599 frames, giving `sigma_px =
sqrt(9.40/6) = 1.25` (closed form, since NEES scales as `1/sigma_px^2`
-- no grid search needed).

Validated out-of-sample on `square6dof` and `room_orbit` (real motion,
real ground truth, same NEES methodology) using this calibrated value:
mean NEES ~4.5-4.6 on both -- somewhat below the target 6.0, meaning the
static-calibrated info matrix is modestly CONSERVATIVE (too uncertain,
not overconfident) when transferred to fixtures with real motion. This
is a genuine, honestly-reported generalisation gap, not treated as a
failure: re-tuning `sigma_px` to fit square6dof/room_orbit too would
have defeated the point of holding them out as a validation set rather
than a second calibration set. The gap errs in the safe direction for a
SLAM system (underclaiming confidence, not overclaiming it).

## 3. Why this is NOT wired into the pipeline (the actual finding of this pass)

Wired both trackers to use the calibrated `pnp_info_matrix` and ran the
full trust-gate suite: selftest and mutation_check both passed
unchanged (17/17, 5/5) -- BUT the selftest's own end-to-end loop-closure
test's graph-optimised ATE moved from 2.29cm to 2.84cm, worse, not
better. That's exactly the kind of "looks fine on the isolated gates,
something's wrong end-to-end" signal this engagement has learned to
chase rather than wave past.

Root cause, found by direct measurement rather than assumption: the
calibrated info matrix's ABSOLUTE MAGNITUDE is enormous relative to
what the rest of the system expects. A typical square6dof frame with
200-350 inliers produces diagonal info values in the TENS OF MILLIONS
(confirmed directly: e.g. frame 2 of a seed=1 run gave diagonal
`[29.1M, 29.1M, 3.2M, 90.6M, 93.8M, 9.1M]`). The OLD heuristic
(`eye(6) * inlier_ratio * 100`) is capped at 100 by construction. The
LOOP-LINK info matrix (set separately, in `verify.py`, untouched by this
work) is clipped to `[0.05, 50.0]`. This is not a bug in the new
derivation -- more inlier observations genuinely DO constrain a 6-DoF
pose far more tightly than a bare inlier RATIO can express, and the old
heuristic never scaled with absolute inlier count at all (200 inliers
and 20 inliers at the same ratio got IDENTICAL info under the old
formula, which is itself an accuracy problem the old heuristic had).
The new derivation is more correct in exactly this respect -- but that
correctness is precisely what produces a ~300,000x scale mismatch
against the still-heuristic, non-count-scaling loop-link info matrix.

Direct before/after comparison confirms the practical consequence:
graph optimisation, which meaningfully improves ATE over raw odometry
with the OLD heuristic (e.g. seed=1: 4.635cm odom -> 4.592cm graph),
does essentially NOTHING with the NEW calibrated odometry info matrix
(4.635cm odom -> 4.635cm graph, unchanged). With odometry links this
overwhelmingly "confident" in absolute terms, the optimiser has no
practical reason to let a loop-closure correction move anything --
which defeats the core value proposition of the whole RTAB-Map-style
architecture (correcting accumulated drift via loop closures), not a
minor accuracy nuance.

Properly fixing this means giving the loop-link info matrix a
comparably principled, absolute-count-aware basis too -- which is
exactly what the architecture doc's own P4 section already scopes
("real estimated information matrices from the verifier's Hessian
instead of heuristics"), not something WP-B2 (explicitly an odometry-
only item, per the plan's own wording: "the PnP refinement's Hessian")
was ever meant to cover alone. A quick patch here -- e.g. rescaling the
new odometry info matrix down to roughly the old heuristic's ballpark --
was considered and rejected: it would throw away exactly the part of
the fix that's actually correct (scaling with absolute observation
count), just to paper over an incompatibility with a heuristic that has
its own, separate, un-addressed problem. Shipping a "statistically
correct in isolation" component that measurably breaks the system's
actual use of that information is not a win, and this pass stopped
rather than rush a fix to the fix without the same oracle-driven care
everything else in this engagement has had.

**Decision: reverted the pipeline wiring.** `odometry.py` and
`odometry_f2m.py` are back to the original heuristic, confirmed via
re-running selftest (ATE back to 2.29cm, bit-identical to pre-WP-B2).
`pnp_info.py` itself -- the derivation, its oracle, and the calibration
methodology -- is correct and kept; it's the missing loop-link
counterpart that blocks using it yet.

## 4. What a continuation needs to know

- `pnp_info.py`'s derivation and calibration (`sigma_px=1.25`, NOT
  currently read by any Config field since the wiring was reverted) are
  ready to use once the loop-link side is addressed. Re-wiring is a
  small, mechanical change (see the reverted diff in this session's git
  history around the WP-B2 commits, or just redo the two call-site
  edits in `update()` for each tracker -- straightforward, the hard part
  was the derivation and calibration, both already done).
- The real remaining work is giving `verify.py`'s loop-link info matrix
  a comparable, absolute-observation-count-aware basis. This doesn't
  necessarily require a full Hessian derivation from the verifier's own
  geometric solve (though that's the "proper" P4-scoped version) --
  even a smaller fix that makes the loop-link heuristic scale with
  absolute inlier count the same way the corrected odometry derivation
  does (rather than normalising it away via `n_inliers/min_inliers`)
  might be enough to restore compatible scales, but this needs its own
  measurement, not an assumption: check whether such a fix keeps
  `pipeline.py`'s post-optimisation loop-rollback check (which compares
  a link's NEES against `chi2(0.995, df=6)` -- a THRESHOLD, sensitive to
  absolute scale) still behaving sensibly before trusting it.
- Whichever direction is chosen, the validation should be the same
  end-to-end check this pass used to catch the problem in the first
  place: don't just check the new info matrix's own NEES in isolation
  (that already passes) -- check graph-optimised ATE against raw
  odometry ATE, across fixtures, and confirm optimisation still helps
  rather than becoming inert. `baseline.py` does not currently measure
  graph-optimised (`pose_map`-based) accuracy at all, only raw
  odometry -- worth adding as a permanent metric there regardless of
  how this specific issue gets resolved, since it's exactly the blind
  spot that let this pass initially believe (from selftest/mutation_check
  alone) that the wiring was safe to ship.
