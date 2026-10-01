"""Thread-safe calibration-health holder.

The watchdog worker thread writes it; the M5 server's 1 Hz health loop and the
track-suppression policy read it. Implements the `calib_state_provider` slot
contract: .state() -> "ok" | "suspect" | "missing", .reason() -> str.
"""
from __future__ import annotations

import threading
import time
from typing import List, Tuple

VALID = ("ok", "suspect", "missing")


class CalibState:
    def __init__(self, initial: str = "missing", reason: str = ""):
        assert initial in VALID
        self._lock = threading.Lock()
        self._state = initial
        self._reason = reason
        self._since = time.time()
        self._history: List[Tuple[float, str, str]] = [(self._since, initial, reason)]

    def set(self, state: str, reason: str = "") -> bool:
        """Returns True if this CHANGED the state."""
        assert state in VALID, state
        with self._lock:
            changed = state != self._state
            self._state, self._reason = state, reason
            if changed:
                self._since = time.time()
                self._history.append((self._since, state, reason))
            return changed

    def state(self) -> str:
        with self._lock:
            return self._state

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def since(self) -> float:
        with self._lock:
            return self._since

    def history(self) -> List[Tuple[float, str, str]]:
        with self._lock:
            return list(self._history)
