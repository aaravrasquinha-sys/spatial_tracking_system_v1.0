"""
Section 8: "auto-clip a few seconds of raw video (or just the JSONL
slice plus rendered overlay) around every ID switch, every track entering
lost_buffer, and every low_confidence footpoint streak longer than ~1s
... capturing them automatically beats scrubbing hour-long logs later."

Keeps a short ring buffer of (rgb frame, Detection2D list) and, when a
trigger fires, writes out the buffered JSONL slice plus (if OpenCV is
importable) a short .mp4 built from the buffered frames. Video writing
is best-effort: a missing cv2 degrades to JSONL-only clips rather than
failing the run.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

from poi_perception.io.detection2d import Detection2D


@dataclass
class _BufferedFrame:
    t_capture: float
    frame_id: int
    rgb: Optional[np.ndarray]
    detections: List[Detection2D]


class ClipRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        fps: float = 30.0,
        pre_seconds: float = 1.5,
        post_seconds: float = 1.5,
        low_confidence_streak_s: float = 1.0,
        keep_rgb: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pre_frames = max(1, int(pre_seconds * fps))
        self.post_frames = max(1, int(post_seconds * fps))
        self.low_confidence_streak_frames = max(1, int(low_confidence_streak_s * fps))
        self.keep_rgb = keep_rgb

        self._buf: Deque[_BufferedFrame] = deque(maxlen=self.pre_frames + 1)
        self._pending_post: List[Tuple[str, List[_BufferedFrame]]] = []  # (trigger_label, frames_after_trigger)
        self._known_track_ids: set[int] = set()
        self._prev_state: dict[int, str] = {}
        self._low_conf_streak: dict[int, int] = {}

    def on_frame(self, t_capture: float, frame_id: int, rgb: Optional[np.ndarray], detections: List[Detection2D]) -> None:
        frame = _BufferedFrame(t_capture, frame_id, rgb.copy() if (rgb is not None and self.keep_rgb) else None, detections)
        self._buf.append(frame)

        triggers = self._detect_triggers(detections)
        for label in triggers:
            self._pending_post.append((f"{label}_f{frame_id}", []))

        still_pending = []
        for label, collected in self._pending_post:
            collected.append(frame)
            if len(collected) >= self.post_frames:
                self._flush(label, collected)
            else:
                still_pending.append((label, collected))
        self._pending_post = still_pending

    def _detect_triggers(self, detections: List[Detection2D]) -> List[str]:
        triggers: List[str] = []
        seen_ids = set()
        for d in detections:
            seen_ids.add(d.track_id_2d)
            prev = self._prev_state.get(d.track_id_2d)
            if d.track_id_2d not in self._known_track_ids and d.track_state == "new":
                triggers.append("id_switch")
            if d.track_state == "lost_buffer" and prev != "lost_buffer":
                triggers.append("lost_buffer")
            self._prev_state[d.track_id_2d] = d.track_state

            if d.low_confidence:
                self._low_conf_streak[d.track_id_2d] = self._low_conf_streak.get(d.track_id_2d, 0) + 1
                if self._low_conf_streak[d.track_id_2d] == self.low_confidence_streak_frames:
                    triggers.append("low_confidence_streak")
            else:
                self._low_conf_streak[d.track_id_2d] = 0
        self._known_track_ids = seen_ids
        return triggers

    def _flush(self, label: str, post_frames: List[_BufferedFrame]) -> None:
        # Pre-buffer + collected post frames, de-duplicated by frame_id
        # (they overlap, since on_frame appends every frame to both) and
        # ordered oldest first.
        by_id = {f.frame_id: f for f in self._buf}
        for f in post_frames:
            by_id[f.frame_id] = f
        ordered = [by_id[k] for k in sorted(by_id.keys())]

        clip_dir = self.out_dir / label
        clip_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = clip_dir / "detections.jsonl"
        with open(jsonl_path, "w") as f:
            for frame in ordered:
                f.write(
                    json.dumps(
                        {
                            "t_capture": frame.t_capture,
                            "frame_id": frame.frame_id,
                            "detections": [d.to_dict() for d in frame.detections],
                        }
                    )
                    + "\n"
                )

        if self.keep_rgb and any(f.rgb is not None for f in ordered):
            self._write_video(clip_dir / "clip.mp4", ordered)

    def _write_video(self, path: Path, frames: List[_BufferedFrame]) -> None:
        try:
            import cv2
        except ImportError:
            return  # best-effort only -- JSONL slice above is always written
        frames_with_rgb = [f for f in frames if f.rgb is not None]
        if not frames_with_rgb:
            return
        h, w = frames_with_rgb[0].rgb.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        try:
            for f in frames_with_rgb:
                bgr = f.rgb[:, :, ::-1]
                writer.write(bgr)
        finally:
            writer.release()
