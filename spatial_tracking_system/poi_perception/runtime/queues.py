"""
Section 4: "size-1, drop-stale queues (never let old frames back up -- a
queued frame from 200ms ago is worse than no frame, since it corrupts
velocity estimates downstream)".

A plain queue.Queue(maxsize=1) with block=False on put would raise
queue.Full instead of dropping the old item, so this is a small
dedicated class rather than a stdlib queue with a wrapper.
"""
from __future__ import annotations

import threading
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class DropStaleQueue(Generic[T]):
    def __init__(self):
        self._item: Optional[T] = None
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self.dropped_count = 0

    def put(self, item: T) -> None:
        """Overwrites whatever was there, unread or not."""
        with self._not_empty:
            if self._item is not None:
                self.dropped_count += 1
            self._item = item
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Blocks until an item is available, the queue is closed, or
        timeout elapses. Returns None on close/timeout."""
        with self._not_empty:
            if self._item is None and not self._closed:
                self._not_empty.wait(timeout=timeout)
            item = self._item
            self._item = None
            return item

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed
