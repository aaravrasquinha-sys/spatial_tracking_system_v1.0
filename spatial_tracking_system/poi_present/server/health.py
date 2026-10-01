"""
Section 8's health skeleton (fps, drops, pipeline latency, hardware,
client count), sourced the way the design doc calls for:

  fps / latency  -- rolling window of (t_capture, t_publish) pairs
                    this module receives, not re-derived elsewhere.
  temperature    -- /sys/class/thermal/thermal_zone*/temp directly.
                    NOT nvidia-ml-py: NVML doesn't see Jetson's
                    integrated GPU, so this deliberately does not
                    reach for it.
  memory / cpu   -- psutil (already in the venv for every other
                    module).
  clock sanity   -- if |t_capture - time.time()| is large at the
                    first frames, global_time_enabled likely failed
                    in capture/realsense_source.py and t_capture is
                    device time, not epoch time; every latency number
                    downstream would be nonsense, so this is reported
                    explicitly as clock="device" rather than silently
                    producing a bad latency_ms.
"""
from __future__ import annotations

import glob
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from poi_present.config import HealthConfig
from poi_present.log import get_logger

log = get_logger("server.health")

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is in every deployment venv,
    psutil = None    # but the test suite shouldn't hard-require it.


@dataclass
class HealthSnapshot:
    fps: float
    latency_ms: Optional[float]
    drop_rate: float
    temp_c: Optional[float]
    mem_percent: Optional[float]
    clock: str  # "synced" | "device"


class HealthTracker:
    def __init__(self, cfg: HealthConfig, window_s: float = 3.0):
        self.cfg = cfg
        self.window_s = window_s
        self._frames: Deque[Tuple[float, float]] = deque()  # (t_capture, t_publish)
        self._dropped = 0
        self._delivered = 0
        self._clock_state = "synced"
        self._clock_checked = False

    def record_frame(self, t_capture: float, t_publish: float) -> None:
        now = t_publish
        if not self._clock_checked:
            self._clock_checked = True
            if abs(t_capture - time.time()) > self.cfg.clock_skew_warn_s:
                self._clock_state = "device"
                log.warning(
                    "t_capture looks like device time, not epoch time "
                    "(global_time_enabled likely failed) -- latency figures "
                    "will be hidden in both viewers until this is fixed."
                )
        self._frames.append((t_capture, t_publish))
        self._delivered += 1
        cutoff = now - self.window_s
        while self._frames and self._frames[0][1] < cutoff:
            self._frames.popleft()

    def record_client_drops(self, n: int) -> None:
        self._dropped += n

    def snapshot(self) -> HealthSnapshot:
        n = len(self._frames)
        fps = 0.0
        latency_ms = None
        if n >= 2:
            span = self._frames[-1][1] - self._frames[0][1]
            fps = (n - 1) / span if span > 0 else 0.0
        if n >= 1 and self._clock_state == "synced":
            latencies = [(tp - tc) * 1000.0 for tc, tp in self._frames]
            latency_ms = sum(latencies) / len(latencies)

        total = self._delivered + self._dropped
        drop_rate = (self._dropped / total) if total > 0 else 0.0

        return HealthSnapshot(
            fps=fps,
            latency_ms=latency_ms,
            drop_rate=drop_rate,
            temp_c=_read_max_temp(self.cfg.thermal_zone_globs),
            mem_percent=(psutil.virtual_memory().percent if psutil is not None else None),
            clock=self._clock_state,
        )


def _read_max_temp(globs) -> Optional[float]:
    best: Optional[float] = None
    for pattern in globs:
        for path in glob.glob(pattern):
            try:
                with open(path) as f:
                    milli = float(f.read().strip())
                deg = milli / 1000.0
                if best is None or deg > best:
                    best = deg
            except (OSError, ValueError):
                continue
    return best
