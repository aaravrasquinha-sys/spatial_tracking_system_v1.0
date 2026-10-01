"""
Section 9: "A simple script that replays a saved JSONL log through a
diff against a previous run's output is enough" -- this is that script.

Two modes:
  --baseline-summary base_summary.json --current run.jsonl
      Compare a fresh run's grade.summarize_log() output against a saved
      baseline summary (the normal case: you changed the model/tracker
      tuning and want to know what moved).
  --baseline-log base.jsonl --current run.jsonl
      Both are re-graded and compared -- convenient for a one-off diff
      without a saved baseline file.

Exits non-zero if any tracked metric regresses beyond its tolerance, so
this can sit in a CI-style check without a human reading the output every
time -- though Section 9 is explicit that the *baseline itself* (what
counts as acceptable) is still set by hand-grading the overlay video.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from poi_perception.eval.grade import LogSummary, summarize_log

# (metric name, direction that's "worse", tolerance) -- direction "up"
# means an increase beyond tolerance is a regression, "down" means a
# decrease is.
_CHECKS = [
    ("new_track_events", "up", 0),  # any increase is worth a look (candidate ID switches)
    ("low_confidence_rows", "up", 0),
    ("lost_buffer_rows", "up", 0),
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
        if direction == "up":
            regressed = delta > tol
        else:
            regressed = -delta > tol
        results.append(DiffResult(metric, b, c, delta, regressed))
    return results


def _load_summary_dict(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-summary", default=None, help="a summary JSON previously written by eval.grade")
    ap.add_argument("--baseline-log", default=None, help="a raw JSONL log to re-grade as the baseline")
    ap.add_argument("--current", required=True, help="the JSONL log from the run being checked")
    args = ap.parse_args(argv)

    if not args.baseline_summary and not args.baseline_log:
        print("Provide --baseline-summary or --baseline-log", file=sys.stderr)
        sys.exit(2)

    baseline = (
        _load_summary_dict(args.baseline_summary)
        if args.baseline_summary
        else summarize_log(args.baseline_log).to_dict()
    )
    current = summarize_log(args.current).to_dict()

    results = diff_summaries(baseline, current)
    any_regressed = False
    print(f"{'metric':<24}{'baseline':>10}{'current':>10}{'delta':>10}  ")
    for r in results:
        flag = "  <-- regressed" if r.regressed else ""
        if r.regressed:
            any_regressed = True
        print(f"{r.metric:<24}{r.baseline:>10}{r.current:>10}{r.delta:>+10}{flag}")

    print()
    print(f"distinct_track_ids: baseline={baseline.get('distinct_track_ids')} current={current.get('distinct_track_ids')}")
    print(f"footpoint_source_counts: baseline={baseline.get('footpoint_source_counts')}")
    print(f"                          current={current.get('footpoint_source_counts')}")

    if any_regressed:
        print("\nOne or more metrics regressed beyond tolerance -- review by hand against the overlay before accepting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
