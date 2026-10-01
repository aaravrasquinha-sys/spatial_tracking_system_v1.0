# WP-P2 Findings: Phase 2 memory management (STM/WM/LTM)

## What this covers

Phase 2 (architecture doc section 5, P2): "STM/WM/LTM with SQLite (WAL)
persistence, rehearsal weight update and merge, transfer of
oldest-of-least-weighted under a wall-clock budget controller,
retrieval of graph neighbours on loop closure. WM-only optimisation
with frozen-LTM anchor priors."

Two prerequisites carried in from the Phase 1B handoff were done first,
per SYSTEM_SUMMARY.md section 6's explicit instruction ("Two things to
carry into it... before Phase 2 design starts"):

## 0a. Rehearsal orphaning bug (fixed)

Phase 0/1's `Memory.add()` merged a new node into an existing STM
entry (bumping its weight) whenever BoW word-overlap exceeded
`rehearsal_threshold`, but the merged-away node's id was never added to
`stm`, `wm`, or anywhere else — it existed only in `_all` (so `get()`
still worked) and was invisible to `working_set()`, `stm_ids()`, and
therefore to every future retrieval or transfer decision. Dormant in
Phase 0/1 because rehearsal only fires with a loaded vocabulary
(`word_ids is not None`), which no gate or baseline run has used so
far — exactly why the architecture doc flagged it as something to fix
*before* Phase 2, not discover mid-Phase-2 once LTM transfer starts
moving things around.

Fix: explicit merge groups (`_merge_rep` / `_merge_children` in
`memory.py`). A merge never discards an id — it redirects the child to
its representative, and every subsequent operation that touches a
representative (transfer to LTM, promotion back to WM, retrieval) moves
the whole group together. `audit_consistency()` checks every merge
child resolves to a representative that's actually resident somewhere.

## 0b. Phase-2 mutation-harness defect class (added)

Per the doc's own rule ("any new subsystem ships its own mutation class
or its gate isn't trusted"): mutation 6, `check_ltm_dangling_reference`,
injects the plausible real Phase-2 regression — a transfer path that
updates `ltm_ids` but forgets to remove the id from `wm` (or the
reverse on retrieval) — directly against `Memory.audit_consistency()`,
since this defect class is a bookkeeping inconsistency, not a numeric
error the existing 5 mutations' style would catch. `pyslam.selftest`
now reports 18/18 (12 gates + 6 mutations); confirmed no regression on
the existing 5.

## 1. LTM persistence (`pyslam/memory/ltm_store.py`)

SQLite, WAL mode, `isolation_level=None` (autocommit — every `put()` is
its own committed transaction, never a multi-row batch a crash could
split). Per the frozen contract's own note on `Signature.rgb`/`depth`
("kept only while in WM"), the store never persists those two fields at
all — a node that reaches LTM permanently loses its raw imagery; only
what retrieval/re-verification need survives (`kp`, `kp3d`, `desc`,
`valid`, `word_ids`, `global_desc`, poses, weight, session_id).

## 2. Transfer-under-budget (`Memory.enforce_budget`)

A simple proportional controller: called every frame with that frame's
measured `duration_ms`; if it exceeds `cfg.wm_budget_ms`, transfer
**exactly one** victim group, otherwise do nothing. Deliberately not
"transfer everything currently over budget" — that shape overshoots
then starves on the next call, which is precisely the oscillation gate
G2 asks the p95 latency curve NOT to show. One-victim-per-over-budget-
frame is the simplest controller that can't do that.

Victim selection (`select_transfer_victim`) is a pure function, kept
separate from `Memory` specifically so it can be checked against a
brute-force reference without touching sqlite/graph state — done in
`tests/gates/test_g2.py::check_victim_selection_matches_brute_force`
(500 randomised trials, exact match every time). Policy: lowest weight
first, ties broken by smallest id (oldest), excluding the
`wm_min_resident` most-recently-created ids — "oldest-of-least-
weighted" per the doc's own phrasing, with a floor so the immediate
neighbourhood of "now" is never evicted out from under active tracking.

## 3. `get()` stays total over every id ever seen

Rather than have every caller special-case "was this id evicted",
`Memory.get()` lazily rehydrates from the LTM store into the resident
cache (`_all`) on demand — but does NOT re-admit it into `wm`/`ltm_ids`
bookkeeping (a `get()` is a read, not a retrieval event). This is what
lets `pipeline.py`'s per-optimisation pose write-back
(`self.memory.get(nid).pose_map = T` for every optimised node) and
`run_synth.py`'s end-of-run trajectory extraction keep working
unmodified regardless of what's currently WM-resident vs LTM-resident.
Verified directly: `run_synth.py --scenario square6dof` reproduces the
exact frozen-baseline numbers (1.72cm anchored ATE odom-only) at
default config (no eviction pressure at that scale — `wm_min_resident`
20 > this run's 18 total WM entries, so nothing transfers, matching
Phase 0/1 behaviour exactly). No baseline re-freeze needed for this
work — `pyslam/tools/baseline_store.json` is untouched.

## 4. Graph-neighbour retrieval on loop closure (`Memory.on_loop`)

`Memory` keeps its own lightweight adjacency map (`record_adjacency`,
called by `pipeline.py` whenever an odom or loop `Link` is added) — not
a duplicate of the pose graph, just enough to answer "what's spatially
near this id". `on_loop(node_id)` pulls `node_id` itself plus any
1-hop neighbour currently LTM-resident back into WM, so the verifier
and optimiser have local context around a revisit rather than a single
isolated node. Tested directly in isolation
(`check_graph_neighbour_retrieval`) and wired into
`pipeline._try_loop_closure` right after a loop link is accepted.

**Open, honestly-reported limitation, not a bug:** retrieval
*candidates* (`score_candidates` in `_try_loop_closure`) are still drawn
only from `working_set()` (WM), matching RTAB-Map's own design. This
means if the actual revisited node itself has already been evicted to
LTM *and* none of its WM-resident neighbours happen to be the ones
scored, the loop is never found in the first place — `on_loop`'s
neighbour-retrieval only helps once something in WM has ALREADY been
recognised as a hit. Confirmed empirically: an aggressive
`wm_budget_ms=0.001` run on `square6dof` (forces transfer almost every
frame) drains WM down to its 5-node floor and the run finds **zero**
loop closures (vs. 1 at default config) — not because retrieval broke,
but because the revisit target left WM long before the camera returned
to it. This is the expected, documented trade-off of budget pressure
against recall, not a regression; RTAB-Map's real fix for this
(full-LTM signature search as a fallback / active relocalisation) is
more than this work package's scope. Left as a named open item for
whoever picks up retrieval-quality work next, alongside Phase 1B's
still-open f2m accuracy gap.

## 5. WM-only optimisation with frozen-LTM anchor priors

No new graph code needed — `GraphBackend.optimize(fixed)` already holds
every id in `fixed` constant during optimisation (confirmed by reading
`backend_native.py`'s `unknown_ids = [i for i in node_ids if i not in
fixed]`). `pipeline._try_loop_closure` now extends the existing
`fixed` list (previously just the gauge-fixed first node) with every id
`Memory` currently considers LTM-resident AND still present in the
graph (`Memory.ltm_ids_expanded()`, which — like transfer/promotion —
returns merge-group members alongside representatives, so a rehearsed-
away node that went to LTM as part of its group is frozen too, not
silently left as a free variable). WM nodes optimise normally; LTM
nodes act as anchor priors. This is a body-swap of an existing
parameter, not new machinery — which is exactly what section 3's
"frozen seam" design rule predicts a well-shaped interface should cost.

## 6. Gate G2 (`tests/gates/test_g2.py`, `python -m tests.gates.test_g2`)

Lighter-weight than the doc's original 10⁴-cycle spec (this
environment's iterate-and-verify budget doesn't cover an
offline/background stress run of that size — flagged honestly rather
than claimed), but exercises the same five properties against real
code paths:

1. Victim selection vs. brute force — 500 trials, exact match.
2. STM never exceeds `cfg.stm_size` across a real synthetic run.
3. `audit_consistency()` clean (zero dangling refs) after a run with
   eviction pressure deliberately forced on.
4. Simulated crash mid-batch (exception injected after 12/20 `put()`
   calls, store closed, path reopened as a fresh `LtmStore`) — exactly
   the 12 committed rows are present and loadable, nothing corrupted.
   This is achieved by WAL + autocommit-per-row (see section 1), not
   emulated after the fact.
5. Graph-neighbour retrieval round-trip (section 4).

All 4 checks pass (`4/4 G2 checks passed`). `pyslam.selftest` remains
18/18 with these changes in place (no regression to any Phase 0/1A/1B
gate).

## What's still open for whoever continues from here

- **Retrieval-under-eviction-pressure** (section 4's limitation) —
  candidate scoring is WM-only; a full-LTM fallback search is real
  future work, not attempted here.
- **10⁴-cycle stress form of G2** as originally specified — the lighter
  version above is a reasonable substitute for iterate-and-verify scale
  but is not the same claim; run the fuller version as a background job
  if this needs to be trusted at production scale.
- **Phase 1B's still-open items are unchanged by this work**: f2m's
  accuracy gap (`cfg.odometry_backend` stays `"f2f"`), and WP-B2's
  info-matrix wiring (blocked on P4's loop-link Hessian rework). Phase
  2 does not depend on either — the `Memory` interface change here is
  orthogonal to which odometry backend or loop-link info source is
  active.
