"""
A frame source that needs no camera: yields correctly-shaped, correctly-
timestamped Frame objects at a fixed rate, with blank (or optionally
patterned) pixel content.

This exists so the rest of M3 -- inference backend swap-out, tracking,
footpoint/torso logic, masking, the daemon's threading, JSONL logging --
can be developed and unit-tested on a laptop with no D435i, no Jetson,
and no TensorRT installed. Pair it with inference.mock_infer.MockPoseBackend,
which produces synthetic *detections* directly (bypassing pixel content
entirely), to exercise the full pipeline end to end deterministically.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from poi_perception.config import CameraConfig
from poi_perception.contracts import Frame, Intrinsics


class SyntheticSource:
    """Deterministic fake capture. `n_frames=None` means infinite."""

    def __init__(
        self,
        cam_cfg: Optional[CameraConfig] = None,
        n_frames: Optional[int] = None,
        start_t: float = 0.0,
        pattern: str = "blank",  # "blank" | "noise"
        seed: int = 0,
    ):
        self.cam_cfg = cam_cfg or CameraConfig()
        self.n_frames = n_frames
        self.start_t = start_t
        self.pattern = pattern
        self._rng = np.random.default_rng(seed)
        self.intr = Intrinsics(
            fx=self.cam_cfg.expected_fx,
            fy=self.cam_cfg.expected_fy,
            cx=self.cam_cfg.expected_cx,
            cy=self.cam_cfg.expected_cy,
            width=self.cam_cfg.width,
            height=self.cam_cfg.height,
            depth_scale=0.001,
        )

    def intrinsics(self) -> Intrinsics:
        return self.intr

    def __iter__(self) -> Iterator[Frame]:
        h, w = self.cam_cfg.height, self.cam_cfg.width
        dt = 1.0 / max(self.cam_cfg.fps, 1)
        i = 0
        while self.n_frames is None or i < self.n_frames:
            if self.pattern == "noise":
                rgb = self._rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
            else:
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
            depth = np.full((h, w), 2000, dtype=np.uint16)  # flat "floor" ~2m at 0.001 scale
            yield Frame(
                t=self.start_t + i * dt,
                rgb=rgb,
                depth=depth,
                intr=self.intr,
                frame_id=i,
                cam_id=self.cam_cfg.cam_id,
            )
            i += 1

    def close(self) -> None:
        pass
