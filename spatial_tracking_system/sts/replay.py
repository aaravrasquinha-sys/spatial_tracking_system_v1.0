"""Replay + regression: run the SAME chain the live service runs over a recorded bag (or the
synthetic scenario), server-less, and summarise / diff the world-track log.

This is the workflow for editing M3 or M4 (docs/ARCHITECTURE.md, "changing a module"):
record real bags once, store the baseline summary, re-run after every change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from sts.runtime import RunPlan, build_chain, run_frames


def replay_bag(plan: RunPlan, bag: str, max_frames: Optional[int] = None, backend=None) -> Path:
    """Returns the path of the world-track JSONL this run wrote."""
    chain = build_chain(plan, source_kind="realsense", playback=bag, serve=False, backend=backend, inline_watchdog=True)
    m4w = chain.m4.writer
    path = Path(m4w.base_path) if m4w is not None and hasattr(m4w, "base_path") else None
    run_frames(chain, max_frames=max_frames)
    if path is None:
        raise RuntimeError("world-track writer exposes no .base_path")
    return path


def summarize(log_path: str | Path) -> dict:
    from poi_localization.eval.grade import summarize_log
    return summarize_log(log_path).to_dict()


def compare(baseline_summary: dict, current_summary: dict) -> dict:
    from poi_localization.eval.replay_diff import diff_summaries
    rows = diff_summaries(baseline_summary, current_summary)
    return {"regressed": any(r.regressed for r in rows),
            "rows": [r.__dict__ for r in rows]}
