"""
Replays a recorded poi_localization JSONL log (Phase A or B --
read_jsonl already tells them apart via each line's "phase" field),
paced by the real gaps between consecutive t_capture values so
playback timing matches how the run actually looked, not a fixed
frame rate. Every frame in the log is replayed, including empty
ones, which is exactly what "stream stalled" vs. "nobody here"
testing needs.

This is the primary way to develop M5a/M5b without hardware: record
once at the desk with run_m4.py, then iterate on the viewers against
the log all day.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List, Optional

from poi_present.adapter import record_to_wire
from poi_present.config import SourceConfig
from poi_present.sources.base import FrameBundle, Sink, TrackSource


class ReplaySource(TrackSource):
    def __init__(self, cfg: SourceConfig, cam_id: str = "cam0"):
        if not cfg.replay_path:
            raise ValueError("SourceConfig.replay_path is required for kind='replay'")
        self.path = Path(cfg.replay_path)
        self.speed = max(cfg.replay_speed, 1e-6)
        self.loop = cfg.replay_loop
        self.cam_id = cam_id
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    async def run(self, sink: Sink) -> None:
        from poi_localization.io.world_track import read_jsonl

        while not self._stopped:
            rows = list(read_jsonl(self.path))
            if not rows:
                return
            prev_t: Optional[float] = None
            for t_capture, _frame_id, phase, map_id, tracks in rows:
                if self._stopped:
                    return
                if prev_t is not None:
                    gap = max(0.0, (t_capture - prev_t) / self.speed)
                    # Never stall more than a second on a log gap (a
                    # paused recording, a dropped stretch) -- replay is
                    # for iterating on the viewer, not reproducing dead
                    # air exactly.
                    await asyncio.sleep(min(gap, 1.0))
                prev_t = t_capture

                frame_name = "room" if phase == "B" else "local"
                wire_tracks = [record_to_wire(rec) for rec in tracks]
                await sink(
                    FrameBundle(t_capture=t_capture, frame=frame_name, map_id=map_id, cam_id=self.cam_id, tracks=wire_tracks)
                )
            if not self.loop:
                return
