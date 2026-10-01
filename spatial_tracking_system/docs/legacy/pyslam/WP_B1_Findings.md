# WP-B1 Finish: Local Bundle Adjustment -- Findings

Code state: `pyslam/frontend/local_ba.py` (new), integrated into
`LocalMapOdometry.update()` (`odometry_f2m.py`), plus a real de-duplication
bug found and fixed along the way. `cfg.odometry_backend` default is
UNCHANGED (`"f2f"`) -- see section 3 below for why.

## 1. What was fixed, and how it was validated

The structural gap odometry_f2m.py's own docstring described: landmarks
are inserted into the local map using whatever pose estimate exists at
insertion time and never corrected afterward, so pose error present at
insertion compounds permanently into the map.

The fix is a sliding-window local bundle adjustment (`local_ba.py`):
jointly re-optimises the last ~5 keyframe poses and the landmark
positions they share, against actual 2D pixel observations, instead of
trusting a single per-frame PnP solve forever. Built and validated as a
pure function first, independent of the tracker, against a synthetic
oracle with known ground truth (same pattern this codebase already uses
for its GTSAM-backend gate: "noise-free synthetic graph, recovers GT to
~1e-8"). This is now a permanent gate,
`test_local_bundle_adjustment_oracle` in `tests/gates/test_g0.py`
(selftest is 16/16, was 15/15).

**A single robust-loss pass was not enough**, found by direct testing,
not assumed: `loss="huber"` down-weights outlier residuals but never
fully rejects them. Injecting one 40px cross-frame outlier (a match that
individually passes its own frame's RANSAC but is wrong relative to a
different keyframe -- never filtered before reaching BA) into an
otherwise noise-free 3-observation landmark pulled it 3.76m from ground
truth despite huber_px=3.0 being active, and measurably degraded several
OTHER, uncorrupted landmarks through the shared pose estimates. Fixed
with the standard two-pass pattern: solve once, inspect residuals per
INDIVIDUAL OBSERVATION (not aggregated per landmark) against that first
solve, physically remove any observation whose residual is still large,
re-solve on the reduced set, then apply a final landmark-level residual
check as a last-resort safety net (same shape as `backend_native.py`'s
own post-optimisation NEES rollback for loop links -- optimise, then
verify the result, then possibly reject that specific piece of it,
rather than trusting the optimiser unconditionally). With this in place,
the same outlier test converges every returned pose and every surviving
landmark to ground truth exactly; the corrupted landmark itself is
conservatively left un-refined this cycle rather than corrupted (see
"KNOWN CONSERVATIVE BEHAVIOUR" in `local_ba.py`'s own docstring for why
that's an accepted trade-off, not a bug).

Integrated into `LocalMapOdometry.update()`: at each new-keyframe event,
this keyframe's re-observations of EXISTING map landmarks are recorded,
BA runs once the window has >=2 keyframes, and both the CURRENT
keyframe's pose and the odometry link fed to the pose graph are updated
to reflect the refined result (not the raw pre-BA PnP solve). Verified
this doesn't touch anything on the default (`f2f`) path or the existing
3-frame f2m convention oracle (`test_local_map_odometry_direction_convention`,
which never reaches a second real keyframe, so never exercises BA):
selftest 16/16, mutation_check 5/5, unaffected.

## 2. A second, separate bug found and fixed: no landmark de-duplication

`_add_landmarks`'s own module docstring claimed "a simple 'map already
has points here' density skip" existed as a known simplification. It
didn't -- no such check was implemented anywhere. Every valid keypoint
got inserted as a brand-new landmark at every keyframe, including
keypoints that had JUST been matched to an existing landmark that same
frame. Fixed by threading the matched keypoint indices through
`_match_to_map`/`_match_guided`/`_match_unguided` (all three now return
the original `sig`-keypoint index alongside their existing outputs) and
excluding them from insertion.

This is a real, independently-valid bug fix (verified: selftest/
mutation_check unaffected), kept regardless of section 3's outcome.

## 3. What's still broken: f2m still does not reach parity, and this is NOT WP-B1's original scope

While validating BA end-to-end against `square6dof` (per the Phase 1
plan's own instruction: "re-run baseline.py with cfg.odometry_backend=
'f2m' and compare against baseline_store.json's frozen f2f numbers, only
flip the default once f2m clearly wins"), found `LocalMapOdometry`
collapses to total tracking failure starting frame 29/140 (0 inliers,
sustained, LOST by frame 33, never recovers) -- REGARDLESS of BA (traced
with `f2m_ba_window=1` to disable BA without touching any other code
path, identical collapse).

Traced to the map's capacity/eviction system (`f2m_max_landmarks=2000`,
"RTAB-Map's own F2M default" per the code comment, plus `_prune_map()`'s
eviction policy). Three independent, targeted fixes were tried and each
FAILED to resolve it:

1. **Eviction sort-order.** Original policy sorts by `(n_obs,
   last_seen)` ascending -- hypothesis: this unfairly penalises
   just-inserted landmarks (all start at n_obs=1) once the map first
   hits cap, causing a self-reinforcing collapse. Tried `(last_seen,
   n_obs)` instead (recency-primary, so just-inserted landmarks are
   automatically protected). Direct A/B test: **identical collapse
   pattern**, frame-for-frame. Reverted (no evidence it helps, added
   complexity for nothing).
2. **De-duplication** (section 2's real bug). Hypothesis: eliminating
   continuous duplicate-landmark insertion would reduce churn pressure
   enough to avoid the collapse. Direct test with BA still disabled:
   **identical collapse pattern**, frame-for-frame, no measurable
   difference at all. Kept anyway (it's still a real, valid bug fix,
   just not sufficient alone), but ruled out as a fix for this.
3. **Cap size.** Swept `f2m_max_landmarks` in {3000, 5000, 8000,
   unbounded (50000, never actually reached)}. At 50000 (never hit),
   zero collapse through frame 45 -- but this only shows "removing
   eviction from the picture entirely avoids the problem it causes",
   not "a bigger cap fixes it": at every FINITE cap actually tested
   (3000/5000/8000), the map eventually grows to fill it and the SAME
   collapse pattern recurs by frame 140 regardless of cap size. This
   scene's actual landmark needs during a full 6-DoF loop don't show a
   clean plateau within the range tested.

Three different, well-reasoned, independently-tested fixes each failing
to resolve it is a strong signal this is a genuinely deeper structural
problem -- capacity management under a hard cap, with churn-under-pressure
degrading match quality across the board once eviction becomes a
steady-state process rather than an occasional trim -- not a tunable
constant or a narrow patch. This is substantively the same class of
problem Phase 2 (memory management: "STM/WM/LTM... rehearsal weight
update and merge... transfer of oldest-of-least-weighted under a
wall-clock budget controller") is explicitly scoped to solve properly,
just manifesting inside `LocalMapOdometry`'s own private local map
rather than the pipeline's shared WM. It was not chased further in this
pass: three failed targeted attempts is the right point to stop and
report rather than attempt a fourth without a clearer hypothesis for
why it would work where the others didn't.

**Consequence: `cfg.odometry_backend` stays `"f2f"`.** No full
baseline.py comparison against the frozen f2f numbers was run, because
the collapse means it would not show a competitive result and would not
be informative beyond confirming what's already documented above. Not
run in order to avoid manufacturing a misleading number; the honest
status is "blocked", not a specific bad drift percentage.

## 5. UPDATE: the capacity/eviction issue itself is now understood and fixed

Section 3 above stopped after three failed targeted attempts and
recommended a dedicated follow-up investigation. That follow-up happened
in the same continued session, via more precise instrumentation (per-
keyframe insertion/eviction counts, not just aggregate map size), and
found the ACTUAL cause -- which was neither eviction policy nor
insertion volume, the two things section 3's three attempts targeted.

**The real bug: a permanent-freeze condition in `_add_landmarks`.** Its
insertion loop had `if len(self.map) >= cap: break`. Once the map first
reached EXACTLY the cap (square6dof: by frame 10 of 140), this condition
is true on the very first candidate keypoint of every subsequent
keyframe, so NOTHING is ever inserted again -- and since `_prune_map()`
only evicts when OVER budget, and insertion now never pushes the map
over budget (it stops exactly at it), eviction never fires either. The
map doesn't churn too aggressively, as section 3 suspected; it does the
opposite -- it freezes completely and permanently the instant it first
fills up, however much of the scene the camera has left to see.
Confirmed directly: `_next_lm_id` (the monotonic creation counter)
stopped incrementing at frame 10 and never moved again through frame
140. This fully explains why section 3's three attempts each failed:
eviction-order tuning changes nothing if eviction never runs;
de-duplication changes nothing if insertion had already permanently
stopped regardless of duplicate count; and raising the cap only delayed
the same one-time freeze rather than preventing it (consistent with
"bigger cap survives a bit longer, still eventually collapses").

**Fix:** stop gating insertion on current map occupancy at all --
`_prune_map()`, already called immediately after `_add_landmarks` by
every call site, is what's supposed to enforce the budget, by evicting
the worst EXISTING landmarks to make room. Tested in isolation (BA still
disabled via `f2m_ba_window=1`, to isolate this one change): frame 30
went from 0 inliers (previously) to 102; real, substantial improvement,
but collapse still recurred later (LOST by frame 38, `n_lost=103/140`).

**A second fix was needed on top: eviction sort-order, revisited.**
Section 3's original eviction-order test is now understood to have been
uninformative -- with the freeze bug active, eviction essentially never
ran, so that test wasn't really exercising ordering at all. Re-tested
after the freeze fix: a single large prune event (hundreds of landmarks
evicted in one shot, right after a keyframe's own hundreds of new
insertions) was observed to zero out the VERY NEXT frame's matching
outright, sorted `(n_obs, last_seen)`. Switching to `(last_seen, n_obs)`
-- recency primary, since a landmark's `last_seen` is set to the current
frame at insertion, so anything just added is automatically among the
most-recently-seen and protected -- combined with the freeze fix:
**zero LOST frames across the full 140-frame square6dof run**, healthy
150-377 inliers sustained from frame 30 onward. Confirmed with BA
re-enabled too (default `f2m_ba_window=5`): still zero LOST, 41
keyframes, sustained tracking throughout.

Both fixes are now in `odometry_f2m.py`, each documented in place with
the measurement that justified it. selftest 16/16, mutation_check 5/5,
unaffected (f2f default path untouched; the 3-frame f2m convention
oracle never reaches a second real keyframe either way).

### 5.1 f2m no longer collapses, but does NOT yet reach parity with f2f

With both fixes and BA all active, the full pipeline (not just the raw
odometry loop) was run on `square6dof` seed=1 and compared directly
against the frozen f2f baseline entry for the same seed:

| metric | f2f (frozen) | f2m (this session) |
|---|---|---|
| anchored ATE | 1.72cm | 39.35cm (23x worse) |
| trans_drift_pct | 1.29%/m | 14.74%/m (11x worse) |
| n_loop_closures | 1 | 0 (missed the loop f2f found) |
| ms_per_frame | ~150-170ms | 888ms (~5x slower) |
| n_lost_events | 0 | 0 (both healthy now) |

f2m is now a FUNCTIONING tracker (no longer catastrophically broken --
that distinction matters and is real progress), but it is clearly not
competitive with f2f on accuracy, loop-closure recall, or speed. This
was not chased further in this pass: the collapse-fixing investigation
was the planned scope for this continuation, and the remaining accuracy
gap is a new, open question (is it BA's window size, the corrected
eviction still discarding useful spatial diversity, the guided-match
window being too tight against a now-much-larger and differently-
distributed map, or something not yet identified) that deserves its own
dedicated measurement pass rather than another same-session guess.

**`cfg.odometry_backend` still stays `"f2f"`.** The bar was always "only
flip once f2m clearly wins," and f2m does not win yet -- it merely no
longer catastrophically loses. That's real, meaningful progress, worth
keeping, but not the same thing.

## 7. WP-B1 continued (post-handoff session): landmark fusion, velocity fix, BA scaling, inlier-ratio retune

Picked up from section 6's own suggested starting points. Investigated
via direct instrumentation (wrap, don't modify -- `diag_f2m.py`/
`diag_f2m2.py`, not committed, same throwaway-script pattern as
`wpb6_trace.py`), same order as this section's own priority list.

### 7.1 The map was never actually persistent

A per-frame trace on square6dof seed=1 found 30,817 landmarks created
over 140 frames against a 2,000 cap -- the map was turning over ~15x
per run, holding only 2-3 keyframes' worth of history at any moment,
which defeats F2M's entire reason for existing. Root cause: no data
association existed for keypoints that FAILED to match this frame's own
guided/unguided search (which starves under near-duplicate descriptors
-- a sampled census found 100% of final landmarks had >=1 near-
duplicate within 3cm/Hamming<40). **Fix**: `_add_landmarks` now checks
unmatched candidates against the existing map (spatial pre-filter, then
Hamming) before inserting, fusing into a matched landmark instead
(running-mean position update, descriptor kept from original creation
-- documented simplification, not updated on fuse). A second pass
catches intra-batch duplicates (candidates that duplicate EACH OTHER
within one frame's own insertion set, not caught by the first pass).
Result: landmarks created dropped to 12,229 (same run); median free
landmarks reaching local BA per call roughly doubled (66 -> 138);
landmarks with n_obs>=6 roughly doubled (247 -> 504).

### 7.2 Velocity was computed one frame stale

Direct measurement: median |predicted velocity| / |true per-frame step|
= 2.057 -- every constant-velocity prediction spanned TWO frame
intervals, not one. Root cause: `self._velocity` was computed against
`self._prev_pose`, which (given the assignment order in `update()`) lags
`self.cur_pose` by one additional frame. Fixed to anchor both the
guided-matching prediction and `_plausibility_gate`'s own motion-gate
check on `self.cur_pose` (the correct "one frame back" anchor) instead.
Result: velocity ratio -> 1.015-1.044; unguided-matching fallback rate
dropped from 68/140 frames to 13-14/140 (guided matching's 25px window
was, in effect, always aimed a full frame-step past where landmarks
actually projected, before this fix).

### 7.3 Local BA's dense Jacobian became the dominant cost (found while validating 7.1)

Profiling a real pipeline run after 7.1/7.2 found `local_bundle_adjust`
responsible for 74% of total wall time (28s/38s over 50 frames),
because 7.1's fix roughly doubled the free-landmark count reaching each
BA call, and `local_ba.py` never received the sparse-Jacobian treatment
F5 gave `backend_native.py`. **Fix**: same technique as F5 -- a
structural sparsity pattern for `least_squares`'s finite-difference
Jacobian (each observation's 2-row residual block depends on exactly
one window pose's 6 columns and, if free, one landmark's 3 columns),
gated by a measured size threshold (`_BA_SPARSE_THRESHOLD_VARS=120`).
Benchmarked on 8 real captured problem instances (n_vars 126-417): net
2x aggregate speedup (20.2s dense vs 9.9s sparse), though NOT uniformly
faster per-instance (3/8 instances were 0.71-0.80x, i.e. slightly
slower) -- the same conditioning-dependent variance F5 documented for
the pose graph, not a clean monotonic-in-size relationship.

**A real, pre-existing bug found while validating this (not introduced
by it -- reproduces identically on the unmodified dense path)**: for 3
of the 8 captured real instances, one weakly-constrained landmark's
position ran away to kilometres from a room-scale scene, undetected by
the existing residual-based safety net. Root cause: a landmark and its
observing pose's correction can drift together along an under-
constrained direction, keeping REPROJECTION residual near zero (what
the existing check measures) even as the ABSOLUTE position becomes
physically absurd -- a classic gauge/rank-deficiency failure the
residual-only check cannot catch by construction. **Fix**: a second,
physically-grounded safety net (`reject_landmark_displacement_m`) that
rejects a landmark whose BA-refined position moved implausibly far from
its pre-BA estimate, same pattern as `verify.py`'s
`verify_max_translation_m` and this module's own motion-gate check.
Calibration mattered and was validated, not guessed: an initial 1.0m
bound didn't fully catch the observed >15km divergence pattern's edge
cases (found via dense-vs-sparse cross-checking); a subsequent 0.15m
bound was TOO tight and broke `test_local_bundle_adjustment_oracle`
(rejected legitimate corrections the noise-free oracle expects to
succeed); **0.5m** passes the oracle exactly (pose/landmark error <1um,
30/30 free landmarks correctly identified, on BOTH the dense and forced-
sparse paths) while remaining ~20,000-30,000x tighter than the observed
real-world divergence -- a wide, validated margin on both sides.

### 7.4 The "missed loop closure" from section 6 was never a tracking bug

Traced directly: node 0's Bayes-filter likelihood DOES rise to the top
candidate in f2m's final 3 keyframes on square6dof seed=1 (raw
likelihood 3.87-4.15). With `hypothesis_hysteresis` relaxed to 1 (
diagnostic-only, not a config change), f2m closes the EXACT SAME loop
f2f finds -- node 136<->0, 116 inliers, essentially identical geometry.
The real cause: f2m creates its qualifying keyframe one cycle later
than f2f on this 140-frame fixture, leaving only one candidate-
evaluation cycle before the run ends -- not enough for the default
2-frame hysteresis. Not a data-association or geometry defect at all;
a fixture-timing artifact both backends approach right at the wire.
`hypothesis_hysteresis` was NOT changed in config.py -- this was a
diagnostic-only override, and changing it globally would need its own
cross-fixture validation this session didn't do.

### 7.5 Seed-to-seed variance: found and partially closed

Multi-seed evaluation after 7.1-7.3 (new `pyslam/tools/wpb1_eval.py`,
deliberately separate from the frozen `baseline_store.json` -- see that
file's own header) found real, substantial improvement but real
remaining variance: square6dof map_ATE ranged 2.76cm (seed 1) to
17.66cm (seed 2), median 8.47cm, vs f2f's tight ~0.9cm/all-seeds. A
per-frame trace on seed=2 (never LOST, inliers mostly healthy 100-400)
found map_ATE accumulating in discrete JUMPS that each correlated
directly with a keyframe whose `inlier_ratio` sat in the 0.35-0.5 band
-- RANSAC still accepted these as its best available solution and the
`f2m_min_inlier_ratio=0.35` plausibility gate let them through, after
which `_add_landmarks` permanently baked the resulting slightly-off
pose into new map landmarks (no later mechanism un-bakes this). A
3-seed x 3-value sweep (0.35/0.5/0.6) found: **0.5 nearly halves
seed=2's map_ATE (17.66cm -> 9.86cm) with ZERO change to seeds 1/3**
(bit-identical map_ATE at 0.35 and 0.5 on both), confirming it screens
out only genuinely marginal keyframes; **0.6 was tested and rejected**
-- it starts rejecting keyframes the OTHER plausibility checks would
have accepted, causing real LOST events (0/0/0 -> 1/1/3 across the
three seeds) and large ATE regressions even on seeds 1/3.
`f2m_min_inlier_ratio` default changed 0.35 -> 0.5 (see config.py's own
note for the full sweep numbers).

### 7.6 Net result, 5-seed square6dof, all fixes applied

| | f2f (median) | f2m (median) |
|---|---|---|
| map_ATE (post-graph) | ~0.9cm | 8.47cm |
| trans drift | ~1.1%/m | 8.60%/m |
| loop closures found | 5/5 seeds | 4/5 seeds |
| ms/frame | ~190 | 541 |

For comparison, the pre-this-session state (section 5.2): odom_ATE
39.35cm (23x worse than f2f), missed loop closure, ~5x slower. Order-
of-magnitude real progress -- but f2m is **still not at parity** and
**`cfg.odometry_backend` correctly stays `"f2f"`**. Seed 2 specifically
still finds zero loop closures even after the inlier-ratio retune
(map_ATE improved, but the revisit is never detected) -- an open,
not-yet-diagnosed question distinct from the timing artifact in 7.4.

### 7.7 What's still open for whoever continues this

- **Seed 2's missed loop closure post-retune**: distinct from 7.4's
  timing artifact (this one doesn't recover even with more hysteresis
  margin, on the evidence gathered so far -- not re-checked this
  session). Needs the same Bayes-filter-likelihood trace 7.4 used,
  applied to seed 2 specifically.
- **`corridor_v2`** was not re-evaluated with any of this session's
  fixes -- it's the fixture f2m was originally justified by (long
  low-overlap segments where F2F's single-keyframe reference starves),
  and per-fixture behaviour doesn't reliably transfer from
  square6dof alone (see WP-P4's own `corridor_v2`-too-slow precedent).
  Should be the next validation step before any `odometry_backend`
  default-flip decision.
- **BA's sparse-vs-dense per-instance variance (7.3)**: same open
  question F5 left for the pose graph -- a fully analytic SE(3)
  Jacobian would remove the FD-evaluation cost on the sparse path
  entirely and likely close the 3/8-instances-slower gap, not
  attempted here (same "higher-risk to hand-derive correctly, judged
  not worth it for an incremental win" reasoning F5 gave).
- **`pyslam.selftest` still doesn't cover G2/G3/G4**: unrelated to this
  session's changes, but worth flagging again -- none of Phase 2-4's
  gates ran automatically while this work was in progress; they were
  run manually (see this section's own validation, all green:
  selftest 20/20, G2 4/4, G4 4/4 confirmed, G3 2/6 confirmed before an
  unrelated background-process interruption in this environment, not a
  code failure).

## 6. What a continuation needs to know (updated)

- Sections 1, 2, and 5 (BA, de-duplication, the freeze bug, and eviction
  ordering) are all done, validated, and should not need revisiting.
- The accuracy gap in section 5.1 is the next open question. Suggested
  starting points, in rough order of how directly they're implicated by
  what's already been measured: (a) BA's window is only 5 keyframes and
  the map now legitimately spans much more of the scene at any given
  time (41 keyframes' worth of landmarks, healthy diversity) -- check
  whether BA is actually being invoked on the RIGHT landmarks, or
  whether 5 keyframes is too narrow a window relative to how much the
  map now moves; (b) compare f2m's per-keyframe ATE trajectory against
  f2f's directly (not just the final aggregate number) to see whether
  error is smoothly worse throughout or concentrated at specific events
  (e.g. right after a large prune, or right after a LOST-free but still
  imperfect keyframe); (c) the missed loop closure specifically --
  is retrieval/verification failing to recognise the revisit at all, or
  is f2m's own drift by that point large enough that the loop candidate
  falls outside the verifier's own geometric tolerance. `wpb6_trace.py`
  is a reusable template for (b)/(c) (wrap, don't modify, log
  everything, join against ground truth post-hoc).
- Whether the (fixed) capacity/eviction system still belongs as an
  explicit consideration in Phase 2's own scope is worth keeping in
  mind even though it's no longer broken: Phase 2's memory-management
  redesign (real STM/WM/LTM for the pipeline's shared memory) might
  eventually want to subsume f2m's local map entirely rather than the
  two maintaining separate capacity-management logic long-term. Not
  urgent now that f2m's own version works, but worth revisiting once
  Phase 2 design starts.
