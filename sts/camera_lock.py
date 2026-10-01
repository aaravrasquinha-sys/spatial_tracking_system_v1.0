"""One D435i, one owner at a time.

M1 mapping, M2 capture/check, M4's Phase-A calibration capture and the live
runtime all open the camera through librealsense. Two of them at once fail
with an opaque USB/"device busy" error -- or hang. Every hardware mode in
`sts` takes this lock first, so the second one refuses immediately and says
who holds it.

Uses flock(2): the kernel releases the lock when the holder dies, so a crashed
run never leaves a stale lock behind. (Scripts launched directly, bypassing
`sts`, don't take it -- run hardware modes through `sts` only.)
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Optional


class CameraBusy(RuntimeError):
    pass


class CameraLock:
    def __init__(self, lock_dir: str | os.PathLike, key: str = "default", mode: str = "unknown"):
        self.path = Path(lock_dir) / f"camera.{key or 'default'}.lock"
        self.mode = mode
        self._fd: Optional[int] = None

    def holder(self) -> Optional[dict]:
        try:
            txt = self.path.read_text().strip()
            return json.loads(txt) if txt else None
        except (OSError, json.JSONDecodeError):
            return None

    def acquire(self) -> "CameraLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            h = self.holder() or {}
            os.close(fd)
            raise CameraBusy(
                f"camera is in use by `sts {h.get('mode', '?')}` (pid {h.get('pid', '?')}, "
                f"since {time.strftime('%H:%M:%S', time.localtime(h.get('since', 0))) if h.get('since') else '?'}). "
                f"Stop it first; only one mode may own the D435i at a time."
            )
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "mode": self.mode, "since": time.time()}).encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.ftruncate(self._fd, 0)
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "CameraLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
