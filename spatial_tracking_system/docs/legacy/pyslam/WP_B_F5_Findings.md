# F5 Fix: Sparse Jacobian for the Native Graph Backend -- Findings

Code state: everything WP_A3_Findings.md describes, plus this fix to
`pyslam/graph/backend_native.py`. No fixture behaviour changed; this is
a pure runtime fix, verified against the frozen baseline in
`pyslam/tools/baseline_store.json`, which this change does NOT alter
(every re-run reproduced the frozen drift/ATE/loop-closure numbers to
full float precision -- only `wall_s`/`ms_per_frame` differ, and those
diffs were not committed, see below).

## What F5 was

`NativeBackend.optimize()` called `scipy.optimize.least_squares` with no
`jac` argument, so scipy estimated the Jacobian by dense 2-point finite
differences: one extra residual evaluation per unknown variable, every
LM/TRF iteration, even though each factor residual only depends on 2 of
the N unknown poses (12 of the 6N columns are structurally zero). Worse,
`method="trf"` then solved the resulting mostly-zero dense system with a
dense linear solve. This is exactly what the original architecture doc
flagged as "O(n^2)-ish and the dominant runtime cost," and it shows up
directly in the baseline: `square6dof` (28 keyframes) ran at
149-170ms/frame while `corridor_v2` (89 keyframes) ran at 949ms/frame --
worse than linear scaling for roughly 3x the node count.

## The fix, and why it's gated by size rather than unconditional

The natural fix is `jac_sparsity`: telling `least_squares` which
Jacobian entries can possibly be nonzero lets scipy's finite-difference
estimator use graph colouring to batch non-conflicting columns into far
fewer function evaluations (the SAME finite-difference values as the
dense path, not an approximation -- fewer redundant evaluations, not
lower precision). A first pass applied this unconditionally and looked
like a clean win on a standalone synthetic benchmark (chain + random
loop links, isotropic info matrices): ~60-100x faster from n=30 to
n=400 nodes.

Running the actual selftest end-to-end fixture (a ~25-31 keyframe loop
closure) with that unconditional version caught a real regression before
it shipped: the fixture got SLOWER (43.1s vs 31.4s total selftest time),
not faster. Direct instrumentation of `optimize()` call-by-call
confirmed it wasn't noise:

| n_poses | n_links | dense (old) | sparse (unconditional) |
|---|---|---|---|
| 25 | 25 | 1.05s | 5.14s |
| 26 | 27 | 0.95s | 5.26s |
| 27 | 29 | 1.14s | 3.93s |
| 28 | 31 | 1.39s | 3.24s |
| 29 | 33 | 1.35s | 2.33s |
| 31 | 36 | 2.38s | 3.87s |

`jac_sparsity` forces scipy onto the iterative `lsmr` trust-region
solver (scipy disallows combining a sparse Jacobian with the dense
`exact` solver). These small graphs' info matrices are the heuristic
`eye(6) * inlier_ratio * 100` (see `odometry.py`), which vary hugely in
magnitude link-to-link (inlier counts 72-385 seen in the same run) --
real ill-conditioning, not a benchmark artifact. An iterative Krylov
solver needs many more inner iterations than a direct dense SVD solve in
that regime; a synthetic benchmark with clean isotropic weights simply
didn't reproduce it, which is itself worth remembering: the topology-only
synthetic benchmark was NOT sufficient evidence on its own, and the real
fixture caught what it missed.

Re-running the actual target case, `corridor_v2` (89 keyframes, the
fixture F5 was originally diagnosed against, where loop-closure events
fire late -- observed node ids 99-175 out of an eventual ~89-120 total,
i.e. the graph is already large by the time `optimize()` is called with
a loop link in it), confirmed the win is real there:

- Unconditional sparse: 949ms/frame -> 414ms/frame (2.3x), drift/ATE
  numbers identical to the frozen baseline to full float precision.

So the fix is gated on `len(unknown_ids) >= _SPARSE_THRESHOLD_POSES`
(60), sitting between the two measured regimes. Every currently-existing
small fixture (square6dof, room_orbit, aliasing_rooms, static_60s, and
the selftest loop-closure gate -- all 25-45 keyframes) keeps the
original dense path unchanged; only corridor_v2-scale graphs take the
sparse path.

## Verification performed

- `pyslam.selftest`: 15/15, 40.2s total (matches the pre-fix 42.2s;
  the 30.0s loop-closure sub-test matches the pre-fix 31.4s -- confirms
  the small-graph regression is gone and the dense path is genuinely
  what's running below threshold).
- `pyslam.tools.mutation_check`: 5/5.
- `pyslam.tools.baseline --scenario corridor_v2 --seeds 1`: 429ms/frame
  (94.4s wall vs the frozen 208.9s), `trans_drift_pct`,
  `rot_drift_deg_per_m`, `anchored_ate_odom_cm`, `n_loop_closures`,
  `n_lost_events` all bit-identical to the frozen entry.
- `pyslam.tools.baseline --scenario square6dof --seeds 1`: seed-1
  `anchored_ate_odom_cm` and `trans_drift_pct` bit-identical to the
  frozen entry (confirms the below-threshold/dense path is unchanged in
  behaviour, not just in timing).
- **Caveat discovered during verification, not related to F5 itself:**
  `baseline.py --scenario X --seeds N` overwrites that scenario's entire
  `runs` array rather than merging into it. Running it with a seed
  subset for a quick check will silently destroy other seeds' frozen
  data if the resulting `baseline_store.json` gets committed. Every
  verification run above was diffed against git and reverted
  (`git checkout -- pyslam/tools/baseline_store.json`); none of these
  numbers are the frozen numbers of record. `baseline_store.json` itself
  is unchanged by this fix. Worth fixing `baseline.py` itself (append/
  merge by scenario+seed rather than replace) before anyone runs partial
  reseeds again -- not done here, out of scope for F5.

## What's still open

- A fully analytic (closed-form) SE(3) Jacobian would remove the
  remaining FD-evaluation cost on the sparse path too (it still pays for
  coloured finite differences, just far fewer of them). Not attempted:
  the SE(3) right-Jacobian's rho/phi coupling block (Barfoot's `Q` term)
  is non-trivial to hand-derive correctly from memory, and the dominant
  O(n^2) cost is already gone at the scale (`corridor_v2`) where it
  mattered. If ever revisited, it should ship with its own numerical-vs-
  analytic cross-check gate (compare against the existing FD Jacobian to
  a tight tolerance on every fixture) before being trusted, the same way
  `mutation_check` guards every other fix in this codebase.
- The threshold (60 poses) is an empirical choice sitting between two
  measured regimes, not a derived optimum. If a future fixture lands
  between ~45 and ~85 keyframes, it's worth re-measuring at that size
  specifically rather than assuming the threshold generalises.
- `corridor_v2`'s baseline is still the 1-seed partial run flagged in
  WP_A3_Findings.md. F5 being fixed is what makes a full 5-seed run
  affordable (see SYSTEM_SUMMARY.md section 6) -- that run has not been
  done yet as part of this change; it's the natural next step.
