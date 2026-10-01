"""
Same regression discipline as poi_perception.eval.replay_diff (Section
9/11: compare a run's countable stats against a saved baseline), applied
to world-track logs.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from poi_localization.eval.grade import summarize_log

_CHECKS = [
    ("distinct_track_ids", "up", 0),  # more distinct IDs than baseline for the same scenario -> possible ID switch
    ("lost_rows", "up", 0),
]


@dataclass
class DiffResult:
    metric: str
    baseline: float
    current: float
    delta: float
    regressed: bool


def diff_summaries(baseline: dict, current: dict, checks=_CHECKS) -> list[DiffResult]:
    results = []
    for metric, direction, tol in checks:
        b = baseline.get(metric, 0)
        c = current.get(metric, 0)
        delta = c - b
        regressed = (delta > tol) if direction == "up" else (-delta > tol)
        results.append(DiffResult(metric, b, c, delta, regressed))
    return results


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-summary", default=None)
    ap.add_argument("--baseline-log", default=None)
    ap.add_argument("--current", required=True)
    args = ap.parse_args(argv)

    if not args.baseline_summary and not args.baseline_log:
        print("Provide --baseline-summary or --baseline-log", file=sys.stderr)
        sys.exit(2)

    baseline = (
        json.loads(Path(args.baseline_summary).read_text())
        if args.baseline_summary
        else summarize_log(args.baseline_log).to_dict()
    )
    current = summarize_log(args.current).to_dict()

    results = diff_summaries(baseline, current)
    any_regressed = False
    print(f"{'metric':<24}{'baseline':>10}{'current':>10}{'delta':>10}")
    for r in results:
        flag = "  <-- regressed" if r.regressed else ""
        any_regressed = any_regressed or r.regressed
        print(f"{r.metric:<24}{r.baseline:>10}{r.current:>10}{r.delta:>+10}{flag}")

    print(f"\nsrc_counts: baseline={baseline.get('src_counts')}")
    print(f"             current={current.get('src_counts')}")

    if any_regressed:
        print("\nOne or more metrics regressed -- review against the floor-marker/scenario baseline before accepting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
