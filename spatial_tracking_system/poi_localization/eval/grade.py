"""
Section 11's evaluation protocol is mostly hands-on (floor markers,
straight-line walks, watching whether IDs swap during a crossing) --
this module is the machine-countable half: summarize a world-track
JSONL log into the numbers a human needs to grade it, and give
eval/replay_diff.py something to diff run-to-run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from poi_localization.io.world_track import read_jsonl


@dataclass
class WorldTrackLogSummary:
    path: str
    phase: str
    n_frames: int
    n_track_rows: int
    distinct_track_ids: int
    confirmed_rows: int
    coasting_rows: int
    lost_rows: int  # one-time publish events, so this ~= number of track endings
    src_counts: Dict[str, int]
    max_concurrent_tracks: int
    mean_height_m: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_log(path: str | Path) -> WorldTrackLogSummary:
    n_frames = 0
    n_rows = 0
    ids = set()
    confirmed = coasting = lost = 0
    src_counts: Counter = Counter()
    max_concurrent = 0
    heights = []
    phase = "A"

    for t_capture, frame_id, ph, map_id, tracks in read_jsonl(path):
        phase = ph
        n_frames += 1
        max_concurrent = max(max_concurrent, len(tracks))
        for t in tracks:
            n_rows += 1
            tid = t.world_track_id if phase == "A" else t.id
            ids.add(tid)
            if t.state == "confirmed":
                confirmed += 1
            elif t.state == "coasting":
                coasting += 1
            elif t.state == "lost":
                lost += 1
            src_counts[t.src] += 1
            h = t.height_m if phase == "A" else t.height
            if h is not None:
                heights.append(h)

    return WorldTrackLogSummary(
        path=str(path),
        phase=phase,
        n_frames=n_frames,
        n_track_rows=n_rows,
        distinct_track_ids=len(ids),
        confirmed_rows=confirmed,
        coasting_rows=coasting,
        lost_rows=lost,
        src_counts=dict(src_counts),
        max_concurrent_tracks=max_concurrent,
        mean_height_m=float(sum(heights) / len(heights)) if heights else 0.0,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    summary = summarize_log(args.jsonl_path)
    text = json.dumps(summary.to_dict(), indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
