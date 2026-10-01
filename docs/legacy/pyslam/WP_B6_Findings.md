# WP-B6: Retrieval Rank / Likelihood / Bayes Trace on corridor_v2 -- Findings

Instrumentation: `pyslam/tools/wpb6_trace.py`. Non-invasive (wraps
`score_candidates` and `BayesFilter.update`, does not touch pipeline
behaviour -- verified bit-identical trajectory/loop-closure output with
instrumentation on vs off, see the module's own docstring and the git
history of this file).

## Caveat, stated up front

`Phase1_Plan.md` -- which SYSTEM_SUMMARY.md says is where the specific
"two hypotheses about why the original corridor scenario failed to close
loops" are written down -- is not present in this repo/handoff bundle. I
checked WP_A3_Findings.md and Runbook.md for the wording too; neither
mentions it. **This document does not confirm or retire those two named
hypotheses; it reports what the instrumentation actually shows.** If the
original plan document turns up, check its wording against the data
below rather than against any claim made here.

There's also a structural mismatch worth naming: the plan's wording
refers to the "original (broken) corridor scenario," which is Phase 0's
geometry, already replaced by WP-A2 (finding F4: the four wall segments
didn't overlap). That geometry no longer exists to instrument. Everything
below is `corridor_v2` (F4's fix), not the original broken fixture.

## What the instrumentation found

Two seeds run to full 220 frames, `TRUE_MATCH_DIST_M=2.5m` (calibrated
against corridor_v2's actual measured loop-closure separations of
1.2-1.9m, which reflect the ~2x2m corner overlap from F4 -- NOT
square6dof's 0.20m room-loop bound, which is a different physical scale;
using it here initially produced a false "0/79 true matches found"
result that was a bug in the instrumentation's threshold, not a real
finding, until corrected).

| Seed | Attempts | True match found | Median rank | Top-1 rate | Median belief (true match) |
|---|---|---|---|---|---|
| 1 | 79 | 79/79 | 10 | 15/79 (19%) | 0.101 |
| 2 | 75 | 75/75 | 10 | 15/75 (20%) | 0.091 |

**1. Raw retrieval score rarely ranks the single geometrically-nearest
candidate first, but the pipeline works anyway because corridor loop
regions are geometrically redundant, not because retrieval is precise.**
Median rank of the true-nearest candidate is 10th out of the WM pool on
both seeds -- direct-descriptor BFMatcher scoring (the Phase-0 choice
documented in `raw_match.py`, chosen specifically because the
fixed-vocabulary BoW alternative erases the signal entirely) is a
meaningfully noisy ranking signal in this fixture, not a crisp one. Yet
every actual accepted loop closure succeeded. Looking at which node the
Bayes filter actually fired against vs. the true-nearest node by GT
distance:

```
seed=1: node 157<->136 (true-nearest=143, rank 4)
        node 160<->136 (true-nearest=144, rank 2)
        node 163<->136 (true-nearest=145, rank 1)
        node 169<->145 (true-nearest=147, rank 1)
        node 175<->147 (true-nearest=149, rank 12)
seed=2: node  35<->18  (true-nearest=21,  rank 3)
        node  38<->18  (true-nearest=22,  rank 1)
        node 157<->136 (true-nearest=141, rank 5)
        node 160<->136 (true-nearest=142, rank 1)
        node 163<->136 (true-nearest=143, rank 1)
        node 166<->136 (true-nearest=144, rank 1)
```
In every single row, the node the Bayes filter actually fired against
(`fired_id`) is DIFFERENT from the single closest-by-GT-distance node
(`true_match_id`), yet still within the 2.5m true-match radius (a
correct loop closure, geometrically and per the pipeline's own
post-optimisation NEES check). One node (136 on seed 1, both 18 and 136
on seed 2) repeatedly wins across 3-4 consecutive query keyframes each,
rather than the filter picking a different "best" candidate each time.
This is consistent with the Bayes filter's own predict/decay design
(loop/bayes.py) doing what it's meant to do: accumulate sustained
evidence toward one candidate rather than chasing whichever single frame
scores highest, and corridor_v2's loop-overlap geometry provides multiple
valid candidates for it to converge on -- the system's robustness here
comes from redundancy in the candidate pool, not from retrieval
precision.

**2. This is a real, previously-undocumented risk surface for Phase 2's
memory-management design**, worth flagging explicitly since Phase 2 is
the next thing this instrumentation was meant to inform. Phase 2's
WM/LTM transfer policy will, by design, evict some working-memory nodes
under a budget. If a future transfer policy happens to evict most of the
2-4 "redundant" near-duplicate candidates in a loop-overlap region and
leaves only one, loop-closure recall in corridor-like geometry could
degrade even though nothing about retrieval SCORING changed -- because
the mechanism that currently compensates for retrieval's noisy ranking
(candidate redundancy) would have been designed away. This wasn't
visible from the summary drift/ATE numbers alone; it only shows up by
tracing which specific candidates fired against which. Concretely: any
Phase 2 transfer-policy gate should include a corridor-like fixture and
check loop-closure recall, not just STM/WM size and dangling-reference
correctness (which is what G2's stated gate already covers per the
architecture doc -- this suggests ADDING a recall check to G2, not
replacing anything there).

**3. This trace data is the empirical anchor P3's recall gate should be
checked against.** The architecture doc's G3 gate targets "recall at
100% precision >= 60% (BoW) / >= 80% (learned)" on `room_loop` +
`corridor_out_back` (the hardware bags, which don't exist -- see
SYSTEM_SUMMARY.md's own honest caveat about this). Absent those bags,
`corridor_v2`'s median-rank-10/19%-top-1 numbers above are a concrete,
reproducible synthetic stand-in P3's incremental-vocabulary and
learned-descriptor channels can be checked against directly: if P3's
descriptor channel doesn't materially improve rank/top-1 on this exact
fixture, its improvement over Phase 0's direct-BFMatcher approach is not
yet demonstrated where it currently matters most.

## What this instrumentation does NOT show

- It does not identify the two named hypotheses from `Phase1_Plan.md` --
  see the caveat above.
- It's synthetic-only, and specifically two seeds of one fixture; it
  should not be read as a claim about hardware behaviour (no D435i
  bags exist to check it against, per SYSTEM_SUMMARY.md section 8).
- `TRUE_MATCH_DIST_M` is a per-scenario constant here (2.5m for
  corridor_v2), chosen from one measurement of actual loop-closure
  separations on this fixture. It is a reasonable calibration, not a
  principled derivation -- if corridor_v2's geometry changes, re-measure
  it before trusting numbers built on it, the same caveat that applies
  to the module's own `TRUE_MATCH_DIST_M` dict.
