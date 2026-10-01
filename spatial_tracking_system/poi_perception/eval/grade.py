"""
Section 9: "For v1, grade by hand rather than building a full MOT-metrics
harness... count ID switches per scenario, count false persons from
masked regions (should be zero), and note any footpoint that's visibly
wrong on the overlay video."

This module doesn't try to replace that hand grading -- inferring true
ID switches or false-person status from the JSONL alone, without ground
truth, isn't reliable. What it does is turn a raw JSONL log into the
countable numbers a human needs to do that grading quickly, and gives a
single summary dict that eval/replay_diff.py can compare run to run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from poi_perception.io.detection2d import read_jsonl


@dataclass
class LogSummary:
    path: str
    n_frames: int
    n_detection_rows: int
    distinct_track_ids: int
    new_track_events: int  # candidate ID-switch points -- confirm by hand against the overlay
    lost_buffer_rows: int
    low_confidence_rows: int
    footpoint_source_counts: Dict[str, int]
    torso_quality_counts: Dict[str, int]
    max_concurrent_tracks: int

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_log(path: str | Path) -> LogSummary:
    n_frames = 0
    n_rows = 0
    track_ids = set()
    new_events = 0
    lost_rows = 0
    low_conf_rows = 0
    fp_counts: Counter = Counter()
    torso_counts: Counter = Counter()
    max_concurrent = 0

    for t_capture, frame_id, dets in read_jsonl(path):
        n_frames += 1
        max_concurrent = max(max_concurrent, len(dets))
        for d in dets:
            n_rows += 1
            track_ids.add(d.track_id_2d)
            if d.track_state == "new":
                new_events += 1
            if d.track_state == "lost_buffer":
                lost_rows += 1
            if d.low_confidence:
                low_conf_rows += 1
            fp_counts[d.footpoint_source] += 1
            torso_counts[d.torso_quality] += 1

    return LogSummary(
        path=str(path),
        n_frames=n_frames,
        n_detection_rows=n_rows,
        distinct_track_ids=len(track_ids),
        new_track_events=new_events,
        lost_buffer_rows=lost_rows,
        low_confidence_rows=low_conf_rows,
        footpoint_source_counts=dict(fp_counts),
        torso_quality_counts=dict(torso_counts),
        max_concurrent_tracks=max_concurrent,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl_path")
    ap.add_argument("--out", default=None, help="write the summary as JSON to this path")
    args = ap.parse_args(argv)

    summary = summarize_log(args.jsonl_path)
    text = json.dumps(summary.to_dict(), indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
