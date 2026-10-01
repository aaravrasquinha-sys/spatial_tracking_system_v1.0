"""
The live hook. M4Pipeline already accepts an `on_tracks(frame,
records)` callback (poi_localization/runtime/m4_pipeline.py) -- it
was simply never wired up by m4_daemon.py. scripts/run_present.py
attaches LiveSource.submit as that callback, exactly the way M4
originally attached itself to M3's on_detections hook. No M4 code
changes.

M4's `process()` runs on the daemon's single-threaded main loop
(Section 4's capture/inference-thread, process-on-main-thread
design, unchanged by M4 or M5). The asyncio server loop is a
*different* thread. LiveSource.submit() is called from the M4
thread and must never block it -- it does the cheapest possible
thing (build the FrameBundle, which is a handful of Python objects)
and hands it to the event loop with call_soon_threadsafe, matching
the plan's "adapter does the cheapest possible thing" design note.
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from poi_present.adapter import record_to_wire
from poi_present.sources.base import FrameBundle, Sink, TrackSource


class LiveSource(TrackSource):
    def __init__(self, cam_id: str, frame_name: str, map_id: Optional[str]):
        self.cam_id = cam_id
        self.frame_name = frame_name
        self.map_id = map_id
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sink: Optional[Sink] = None
        self._stopped = False

    def submit(self, frame, records: List) -> None:
        """Call from the M4 thread (i.e. pass this as M4Pipeline's
        on_tracks). Never blocks; drops the frame with a warning if
        the server loop isn't ready yet (startup race) or if this
        source has been stopped."""
        if self._loop is None or self._sink is None or self._stopped:
            return
        bundle = FrameBundle(
            t_capture=frame.t,
            frame=self.frame_name,
            map_id=self.map_id,
            cam_id=self.cam_id,
            tracks=[record_to_wire(r) for r in records],
        )
        try:
            self._loop.call_soon_threadsafe(self._deliver, bundle)
        except RuntimeError:
            pass  # loop already closed (shutdown race) -- fine to drop

    def _deliver(self, bundle: FrameBundle) -> None:
        if self._sink is not None:
            asyncio.ensure_future(self._sink(bundle))

    def stop(self) -> None:
        self._stopped = True

    async def run(self, sink: Sink) -> None:
        self._loop = asyncio.get_running_loop()
        self._sink = sink
        while not self._stopped:
            await asyncio.sleep(0.5)
