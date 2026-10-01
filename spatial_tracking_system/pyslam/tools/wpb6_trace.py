"""
WP-B6 instrumentation (Phase 1 plan): retrieval rank / likelihood / Bayes
trace on a scenario, meant to be done BEFORE Phase 2 (memory management)
design starts -- see SYSTEM_SUMMARY.md section 6. The intent per the
handoff was to "confirm or retire two hypotheses about why the original
(broken) corridor scenario failed to close loops."

CAVEAT, stated up front: Phase1_Plan.md, which is where those two named
hypotheses apparently live, is not present in this repo/handoff bundle
(checked: not in the zip, not referenced by wording in WP_A3_Findings.md
or Runbook.md). This module does not confirm or retire hypotheses it has
never seen the text of. It builds the actual instrumentation and reports
what it empirically shows; if the original two hypotheses turn up later,
re-check them against the trace data this produces rather than against
anything asserted here.

Also worth noting: the "original (broken) corridor scenario" the plan
refers to is Phase 0's, which WP-A2 already replaced (finding F4: the
four wall segments didn't geometrically overlap). This instruments
corridor_v2, the fixed geometry -- there is no way to run this against
the original broken geometry any more, so this is necessarily reporting
on a different (successor) fixture than whatever the original two
hypotheses were about. That's a second, structural reason this module
reports observations rather than verdicts on the original hypotheses.

Mechanism: non-invasive. Wraps `score_candidates` and
`BayesFilter.update` to record their inputs/outputs as a side effect,
and wraps `Pipeline._try_loop_closure` ONLY to stash the current node id
for the two inner wraps to tag their records with -- it still calls the
real, unmodified method. No pipeline behaviour is touched by any of
this: every wrapped function returns exactly what the original would.
Verified (see WP_B6_Findings.md) that trajectory/loop-closure results
are bit-identical with instrumentation on vs off.
"""
from __future__ import annotations
import numpy as np

from pyslam.core.config import Config
from pyslam.core.log import get_logger
import pyslam.pipeline as pipeline_module
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from pyslam.loop.bayes import BayesFilter
from tests.synth.scenarios import build

log = get_logger("wpb6_trace")

# "True match" distance bound is scenario-scale-dependent, not universal:
# square6dof's tight room loop uses 0.20m (test_g0's own bound), but
# corridor_v2's loop closures were measured directly (this module, first
# run) to sit at 1.2-1.9m GT separation -- consistent with F4's ~2x2m
# corner overlap, not a tight room revisit. Pass the right one in.
TRUE_MATCH_DIST_M = {
    "default": 0.20,
    "corridor_v2": 2.5,
}


def _true_match_id(node_id: int, candidate_ids: list[int], node_gt: dict,
                    max_dist_m: float) -> int | None:
    """Ground-truth nearest candidate to node_id, if within max_dist_m."""
    if node_id not in node_gt:
        return None
    gt_pos = node_gt[node_id][:3, 3]
    best_id, best_d = None, None
    for cid in candidate_ids:
        if cid not in node_gt:
            continue
        d = np.linalg.norm(node_gt[cid][:3, 3] - gt_pos)
        if best_d is None or d < best_d:
            best_id, best_d = cid, d
    if best_id is not None and best_d <= max_dist_m:
        return best_id
    return None


def trace_scenario(scenario_name: str, seed: int = 1, max_frames: int | None = None) -> list[dict]:
    """Runs the real pipeline once with instrumentation active. Returns a
    list of per-keyframe trace records (one per _try_loop_closure call
    that actually had candidates to score)."""
    records: list[dict] = []
    ctx = {"node_id": None}

    orig_score = pipeline_module.score_candidates

    def wrapped_score(query_sig, candidate_nodes, ratio=0.75):
        scores = orig_score(query_sig, candidate_nodes, ratio)
        ctx["candidate_ids"] = [n.id for n in candidate_nodes]
        ctx["raw_scores"] = scores.copy()
        return scores

    orig_update = BayesFilter.update

    def wrapped_update(self, likelihood, candidate_ids):
        hyp = orig_update(self, likelihood, candidate_ids)
        belief_snapshot = {cid: self.belief.get(cid, 0.0) for cid in candidate_ids}
        records.append({
            "node_id": ctx["node_id"],
            "candidate_ids": list(candidate_ids),
            "raw_scores": ctx.get("raw_scores"),
            "likelihood": likelihood.copy(),
            "belief_after": belief_snapshot,
            "belief_new_after": self.belief_new,
            "fired": hyp is not None,
            "fired_id": hyp.node_id if hyp is not None else None,
            "fired_posterior": hyp.posterior if hyp is not None else None,
        })
        return hyp

    orig_try_loop = Pipeline._try_loop_closure

    def wrapped_try_loop(self, node, result):
        ctx["node_id"] = node.id
        return orig_try_loop(self, node, result)

    pipeline_module.score_candidates = wrapped_score
    BayesFilter.update = wrapped_update
    Pipeline._try_loop_closure = wrapped_try_loop
    try:
        scen = build(scenario_name, seed=seed, validate=False)
        source = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt,
                                  add_noise=True, seed=seed)
        cfg = Config()
        pipe = Pipeline(cfg, backend_prefer="native")
        result = pipe.run(source, max_frames=max_frames, verbose=False)
    finally:
        pipeline_module.score_candidates = orig_score
        BayesFilter.update = orig_update
        Pipeline._try_loop_closure = orig_try_loop

    # Post-hoc join against ground truth, now that the whole run's
    # node_gt is available.
    for rec in records:
        max_dist = TRUE_MATCH_DIST_M.get(scenario_name, TRUE_MATCH_DIST_M["default"])
        rec["true_match_id"] = _true_match_id(rec["node_id"], rec["candidate_ids"], result.node_gt, max_dist)
        if rec["true_match_id"] is not None and rec["raw_scores"] is not None:
            idx = rec["candidate_ids"].index(rec["true_match_id"])
            order = np.argsort(-rec["raw_scores"])  # descending score
            rec["true_match_rank"] = int(np.where(order == idx)[0][0]) + 1  # 1-based
            rec["true_match_score"] = float(rec["raw_scores"][idx])
            rec["true_match_likelihood"] = float(rec["likelihood"][idx])
            rec["true_match_belief_after"] = float(rec["belief_after"][rec["true_match_id"]])
        else:
            rec["true_match_rank"] = None
    return records, result


def summarize(records: list[dict], result) -> None:
    with_true = [r for r in records if r["true_match_id"] is not None]
    log.info(f"{len(records)} retrieval attempts total, "
              f"{len(with_true)} had a ground-truth-true candidate present in WM.")
    if with_true:
        ranks = [r["true_match_rank"] for r in with_true]
        log.info(f"true-match rank among candidates: "
                  f"median={np.median(ranks):.1f} best={min(ranks)} worst={max(ranks)} "
                  f"(1=top-scored)")
        top1 = sum(1 for r in ranks if r == 1)
        log.info(f"true match was top-ranked (rank 1) in {top1}/{len(ranks)} attempts")
        beliefs = [r["true_match_belief_after"] for r in with_true]
        log.info(f"true-match belief after Bayes update: "
                  f"median={np.median(beliefs):.3f} max={max(beliefs):.3f}")
        never_fired_despite_present = [
            r for r in with_true if not r["fired"] and r["true_match_belief_after"] > 0.5
        ]
        if never_fired_despite_present:
            log.info(f"{len(never_fired_despite_present)} attempts had true-match belief "
                     f">0.5 but did NOT fire a hypothesis that update (hysteresis not yet met, "
                     f"or belief dropped again next step) -- see per-node detail for streak behaviour")
    log.info(f"actual accepted loop closures this run: {len(result.loop_events)}")
    for new_id, old_id, link in result.loop_events:
        matching = [r for r in records if r["node_id"] == new_id]
        if matching:
            r = matching[0]
            log.info(f"  node {new_id}<->{old_id}: true_match_id={r['true_match_id']} "
                     f"rank={r['true_match_rank']} fired_id={r['fired_id']}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="corridor_v2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    records, result = trace_scenario(args.scenario, seed=args.seed, max_frames=args.max_frames)
    summarize(records, result)


if __name__ == "__main__":
    main()
