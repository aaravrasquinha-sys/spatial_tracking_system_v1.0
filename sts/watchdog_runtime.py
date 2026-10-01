"""Runtime supervisor for the calibration watchdog (the M6 piece).

`pyslam.anchor.watchdog.AnchorWatchdog` is a LIBRARY: three layers, nobody calling
them. This supervisor calls them from the live frame loop without ever stalling it:

  observe(frame, n_tracks)   -- called once per frame on the pipeline thread. It only
                                copies the depth frame into a small ring and decides
                                whether a check is due. Microseconds.
  depth layer  (cheap)       -- every `depth_check_period_s` (and every `confirm_period_s`
                                while a breach is awaiting confirmation): temporal-median
                                of the last `depth_window_frames` frames vs the stored
                                reference on its stable pixels. The library needs
                                `wd_consecutive` consecutive breaches; the faster confirm
                                cadence makes a nudge visible well inside ONE period.
  ICP layer    (expensive)   -- every `recheck_period_s`, only while nobody has been
                                tracked for `recheck_idle_s`, and immediately after the
                                depth layer goes suspect (if the scene is idle). It is the
                                only layer that can CLEAR a suspect state, and only with a
                                converged, good-fitness "ok" verdict -- never silently.
  tilt layer   (optional)    -- an injected `tilt_source()` returning the current up-vector
                                in the camera frame. Off by default: the runtime capture
                                path has no IMU and a second accel pipeline under RSUSB is
                                UNVALIDATED (docs/OPEN_ITEMS.md).

Heavy work runs on ONE worker thread (single-flight); the frame loop never waits.
It never re-calibrates. The worst it does is flip CalibState to "suspect" and emit a
`calib_suspect` event; what the system does about it is `on_suspect` policy
("suppress" | "flag"), applied by sts.runtime.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

import numpy as np

from sts.calib_health import CalibState
from sts.site import WatchdogSpec


def _log():
    from pyslam.core.log import get_logger
    return get_logger("sts.watchdog")


class WatchdogSupervisor:
    def __init__(self, spec: WatchdogSpec, calib_state: CalibState, watchdog=None, *,
                 anchor_cfg=None,
                 map_loader: Optional[Callable[[], tuple]] = None,
                 on_event: Optional[Callable[[str, str], None]] = None,
                 tilt_source: Optional[Callable[[], Optional[np.ndarray]]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 inline: bool = False,
                 confirm_period_s: float = 5.0):
        self.spec = spec
        self.calib = calib_state
        self.wd = watchdog                    # AnchorWatchdog, or None => calibration "missing"
        self.anchor_cfg = anchor_cfg
        self._map_loader = map_loader
        self._map = None
        self._on_event = on_event
        self._tilt_source = tilt_source
        self._clock = clock
        self._inline = inline
        self._confirm_period_s = confirm_period_s

        n = max(spec.depth_window_frames, spec.recheck_frames)
        self._depth: Deque[np.ndarray] = deque(maxlen=n)
        self._rgb: Deque[np.ndarray] = deque(maxlen=3)
        self._intr = None
        self._frames_since_depth_check = 0
        self._frames_total = 0
        t0 = clock()
        self._next_depth = t0
        self._next_recheck = t0 + spec.recheck_period_s
        self._next_suspect_recheck = t0
        self.suspect_retry_s = 60.0
        self._last_track_t = t0               # "idle" = no tracks since (now - idle_s)
        self._busy = threading.Lock()
        self._startup_done = not (spec.startup_check and watchdog is not None)
        self.last_recheck: Optional[dict] = None
        self.stats = {"depth_checks": 0, "rechecks": 0, "tilt_checks": 0, "errors": 0}

        if watchdog is None:
            self.calib.set("missing", "no calibration/reference loaded")
        elif not self._startup_done:
            # publish nothing wrong while the first check is pending
            self.calib.set("suspect", "startup verification pending")
        else:
            self.calib.set("ok", "")

    # ------------------------------------------------------------------ frame hook
    def observe(self, frame, n_active_tracks: int) -> None:
        if self.wd is None or not self.spec.enabled:
            return
        now = self._clock()
        if n_active_tracks > 0:
            self._last_track_t = now
        self._frames_total += 1
        self._frames_since_depth_check += 1
        self._intr = frame.intr
        # copy: capture sources are free to reuse their buffers
        self._depth.append(np.array(frame.depth, dtype=np.uint16, copy=True))
        if frame.rgb is not None and (self._frames_total % 10 == 1):
            self._rgb.append(np.array(frame.rgb, copy=True))

        due_depth = (now >= self._next_depth and self._frames_since_depth_check >= self.spec.depth_window_frames
                     and len(self._depth) >= self.spec.depth_window_frames)
        idle = (now - self._last_track_t) >= self.spec.recheck_idle_s
        suspect = self.calib.state() == "suspect"
        can_recheck = (self._startup_done and idle and self._map_loader is not None and self.anchor_cfg is not None
                       and len(self._depth) >= self.spec.recheck_frames)
        # (a) the periodic schedule, (b) while suspect: try to characterise/clear it, retrying every
        # `suspect_retry_s` for as long as the scene is idle.
        due_recheck = can_recheck and (now >= self._next_recheck or (suspect and now >= self._next_suspect_recheck))
        if due_recheck:
            self._next_suspect_recheck = now + self.suspect_retry_s
        if not (due_depth or due_recheck or self._tilt_due(now)):
            return
        if not self._busy.acquire(blocking=False):
            return                             # a check is already running; try again next frame
        job = (due_depth, due_recheck, self._tilt_due(now))
        if self._inline:
            self._work(job)
        else:
            threading.Thread(target=self._work, args=(job,), name="sts-watchdog", daemon=True).start()

    def _tilt_due(self, now: float) -> bool:
        return (self.spec.tilt_layer == "external" and self._tilt_source is not None
                and now >= self._next_depth)

    # ------------------------------------------------------------------ worker
    def _work(self, job) -> None:
        try:
            due_depth, due_recheck, due_tilt = job
            if due_tilt:
                self._run_tilt()
            if due_depth:
                self._run_depth()
            if due_recheck:
                self._run_recheck()
        except Exception as e:                 # a broken check must not read as "ok"
            self.stats["errors"] += 1
            _log().warning(f"watchdog check failed: {type(e).__name__}: {e}")
            self._to_suspect(f"watchdog check error: {type(e).__name__}: {e}")
        finally:
            self._busy.release()

    def _median_depth_m(self, n: int) -> np.ndarray:
        stack = np.stack(list(self._depth)[-n:], axis=0)
        scale = float(self._intr.depth_scale)
        out = np.full(stack.shape[1:], np.nan, np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for s in range(0, stack.shape[1], 60):
                blk = stack[:, s:s + 60].astype(np.float32) * scale
                blk[stack[:, s:s + 60] == 0] = np.nan
                out[s:s + 60] = np.nanmedian(blk, axis=0)
        return out

    def _run_depth(self) -> None:
        wd = self.wd
        med = self._median_depth_m(self.spec.depth_window_frames)
        if med.shape != wd.ref_depth.shape:
            self._to_suspect(f"live depth shape {med.shape} != reference {wd.ref_depth.shape} (stream config changed?)")
            return
        self.stats["depth_checks"] += 1
        self._frames_since_depth_check = 0
        frac = wd.check_depth(med)
        now = self._clock()
        if not self._startup_done:
            self._startup_done = True
            if frac > wd.cfg.wd_depth_frac:
                wd._raise(f"startup: {frac:.0%} of stable reference pixels moved > "
                          f"{wd.cfg.wd_depth_move_m * 100:.0f} cm vs the stored reference")
            else:
                # startup verified: clear the "pending" state
                self.calib.set("ok", "")
                wd.state.status, wd.state.reason = "ok", ""
        pending_confirm = wd.state.consecutive > 0 and wd.state.status == "ok"
        self._next_depth = now + (self._confirm_period_s if pending_confirm else self.spec.depth_check_period_s)
        if wd.state.status == "suspect":
            self._to_suspect(wd.state.reason)
            # nothing else to do here: the ICP layer is the only one allowed to clear it

    def _run_tilt(self) -> None:
        up = self._tilt_source()
        self.stats["tilt_checks"] += 1
        if up is not None:
            self.wd.check_tilt(np.asarray(up, float))
            if self.wd.state.status == "suspect":
                self._to_suspect(self.wd.state.reason)

    def _run_recheck(self) -> None:
        from pyslam.anchor.capture import build_capture
        wd = self.wd
        if self._map is None:
            self._map = self._map_loader()
        pts, cols = self._map
        n = min(self.spec.recheck_frames, len(self._depth))
        stack = np.stack(list(self._depth)[-n:], axis=0)
        rgb = np.stack(list(self._rgb)) if self._rgb else np.zeros((1,) + stack.shape[1:] + (3,), np.uint8)
        i = self._intr
        intr = {"fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy, "width": i.width, "height": i.height,
                "depth_scale": i.depth_scale, "baseline": getattr(i, "baseline", 0.05)}
        cap = build_capture(stack, rgb, intr, None, None, self.anchor_cfg, source="runtime-recheck")
        res = wd.recheck_pose(cap, pts, cols)
        res.pop("T_measured", None)
        self.last_recheck = res
        self.stats["rechecks"] += 1
        self._next_recheck = self._clock() + self.spec.recheck_period_s
        _log().info(f"ICP recheck: {res}")
        if res["verdict"] == "moved":
            self._to_suspect(wd.state.reason)
        elif res["verdict"] == "ok":
            self._startup_done = True
            if self.calib.set("ok", ""):
                _log().info("calibration cleared by ICP recheck (converged, good fitness, shift within tolerance)")
        # "inconclusive": leave whatever state we had

    # ------------------------------------------------------------------ state
    def _to_suspect(self, reason: str) -> None:
        changed = self.calib.set("suspect", reason)
        if changed:
            _log().warning(f"CALIBRATION SUSPECT: {reason}")
            if self._on_event:
                try:
                    self._on_event("calib_suspect", reason)
                except Exception:
                    pass

    def should_publish_3d(self) -> bool:
        """Policy hook used by sts.runtime: only `suspect` + on_suspect=='suppress' withholds 3D."""
        return not (self.calib.state() == "suspect" and self.spec.on_suspect == "suppress")
