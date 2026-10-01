"""
One interface, three implementations (live / replay / synthetic),
per the M5 design. Whichever one is configured, the server and both
web clients see an identical stream -- that's the point of freezing
this shape.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from poi_present.schema import TrackWire


@dataclass
class FrameBundle:
    """One frame's worth of output, already in wire-ready form (each
    element of `tracks` is a TrackWire, produced by
    adapter.record_to_wire). This is the unit every TrackSource
    produces and the server's Publisher consumes."""

    t_capture: float
    frame: str  # "local" | "room"
    map_id: Optional[str]
    cam_id: str
    tracks: List[TrackWire]


Sink = Callable[[FrameBundle], Awaitable[None]]


class TrackSource(abc.ABC):
    """Runs on the server's asyncio loop. `run(sink)` drives frames
    into `sink` at whatever pace is appropriate for this source
    (real-time for live, log-timed for replay, tick-timed for
    synthetic) until `stop()` is called or the source is exhausted."""

    @abc.abstractmethod
    async def run(self, sink: Sink) -> None:
        ...

    def stop(self) -> None:
        """Best-effort cooperative stop; run() should notice and
        return promptly. Default no-op for sources with no internal
        loop state to flag (e.g. a live source stopped externally)."""
        return None
