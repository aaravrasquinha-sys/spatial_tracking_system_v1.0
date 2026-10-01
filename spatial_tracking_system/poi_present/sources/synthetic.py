"""
What M5b (and M5a) get built against before there's a room map or a
tolerance for standing in front of the camera all day. Two modes:

  "scripted"  -- a small hand-written state machine per walker. Cheap,
                 fast, deterministic, and it deliberately exercises
                 every wire state (tentative/confirmed/coasting/lost),
                 every src value, id churn (a walker "re-entering"
                 under a new id, since Section 3 scopes re-ID out of
                 v1), and whole-frame dropouts -- so the viewer is
                 built for real stream behaviour, not just the happy
                 path.

  "realistic" -- feeds walker ground truth through the SAME sensor
                 error model and the SAME real TrackManager /
                 measurement code that
                 poi_localization.eval.consistency.run_consistency_check
                 uses for its NEES check, generalized from one walker
                 to N. What the viewer sees (covariance shapes,
                 depth/raycast fusion switching, track lifecycle
                 timing) is then produced by the actual M4 code, not
                 an imitation of it -- this is cheap to build because
                 every piece already exists in poi_localization.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from poi_present.config import SourceConfig
from poi_present.schema import TrackWire
from poi_present.sources.base import FrameBundle, Sink, TrackSource


# --------------------------------------------------------------------------
# Scripted mode
# --------------------------------------------------------------------------


@dataclass
class _ScriptedWalker:
    id: int
    x: float
    y: float
    waypoint: Tuple[float, float]
    state: str = "tentative"
    hits: int = 0
    speed: float = 1.2
    coasting_left_s: float = 0.0
    lost_left_s: float = 0.0
    height: float = 1.72
    src_cycle: int = 0


class ScriptedSyntheticSource(TrackSource):
    def __init__(self, cfg: SourceConfig, tick_hz: float = 30.0):
        self.cfg = cfg
        self.tick_hz = tick_hz
        self._rng = np.random.default_rng(cfg.synthetic_seed)
        self._rect = cfg.synthetic_rect_m
        self._walkers: Dict[int, _ScriptedWalker] = {}
        self._next_id = 1
        self._stopped = False
        for _ in range(cfg.synthetic_n_walkers):
            self._spawn()

    def stop(self) -> None:
        self._stopped = True

    def _spawn(self) -> _ScriptedWalker:
        wid = self._next_id
        self._next_id += 1
        x0, x1, y0, y1 = self._rect
        x = self._rng.uniform(x0, x1)
        y = self._rng.uniform(y0, y1)
        w = _ScriptedWalker(id=wid, x=x, y=y, waypoint=self._new_waypoint())
        self._walkers[wid] = w
        return w

    def _new_waypoint(self) -> Tuple[float, float]:
        x0, x1, y0, y1 = self._rect
        return (float(self._rng.uniform(x0, x1)), float(self._rng.uniform(y0, y1)))

    def _step_walker(self, w: _ScriptedWalker, dt: float) -> None:
        dx, dy = w.waypoint[0] - w.x, w.waypoint[1] - w.y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            w.waypoint = self._new_waypoint()
        else:
            w.x += w.speed * dt * dx / dist
            w.y += w.speed * dt * dy / dist

    def _tick(self, t: float, dt: float) -> List[TrackWire]:
        out: List[TrackWire] = []
        for w in list(self._walkers.values()):
            # Occasionally start a short coasting (occlusion) episode.
            if w.state == "confirmed" and self._rng.random() < 0.0006:
                w.state = "coasting"
                w.coasting_left_s = float(self._rng.uniform(0.2, 0.9))

            if w.state == "coasting":
                w.coasting_left_s -= dt
                if w.coasting_left_s <= 0:
                    w.state = "confirmed"

            # Occasionally lose and end a track (walker exits frame),
            # then respawn a fresh walker under a NEW id -- exercises
            # id churn, since re-ID after exit is out of scope for v1.
            if w.state == "confirmed" and self._rng.random() < 0.0002:
                w.state = "lost"
                w.lost_left_s = 1.0

            self._step_walker(w, dt)

            if w.state == "tentative":
                w.hits += 1
                if w.hits >= 3:
                    w.state = "confirmed"

            vx = (w.waypoint[0] - w.x)
            vy = (w.waypoint[1] - w.y)
            norm = math.hypot(vx, vy) or 1.0
            v = (w.speed * vx / norm, w.speed * vy / norm, 0.0)

            w.src_cycle += 1
            src = ("fused", "fused", "fused", "depth", "raycast")[w.src_cycle % 5]
            if w.state == "coasting":
                src = "predicted"

            cov_scale = 0.0025 if w.state != "coasting" else 0.02 * (1.0 + (1.0 - max(w.coasting_left_s, 0)))
            out.append(
                TrackWire(
                    id=w.id,
                    state=w.state,
                    p=(round(w.x, 4), round(w.y, 4), 0.0),
                    v=v,
                    cov_xy=(cov_scale, 0.0002, cov_scale),
                    height=w.height,
                    conf=float(self._rng.uniform(0.7, 0.95)),
                    src=src,
                    age_s=float(t),
                    bbox=(0.0, 0.0, 0.0, 0.0),
                )
            )

            if w.state == "lost":
                w.lost_left_s -= dt
                if w.lost_left_s <= 0:
                    del self._walkers[w.id]
                    self._spawn()

        return out

    async def run(self, sink: Sink) -> None:
        dt = 1.0 / self.tick_hz
        t = 0.0
        period = 1.0 / self.tick_hz
        next_tick = time.monotonic()
        while not self._stopped:
            tracks = self._tick(t, dt)
            await sink(
                FrameBundle(t_capture=time.time(), frame="local", map_id=None, cam_id="cam0-sim", tracks=tracks)
            )
            t += dt
            next_tick += period
            sleep_for = max(0.0, next_tick - time.monotonic())
            await asyncio.sleep(sleep_for)


# --------------------------------------------------------------------------
# Realistic mode: real TrackManager + real measurement models
# --------------------------------------------------------------------------


class RealisticSyntheticSource(TrackSource):
    """Same sensor model and code path as
    poi_localization.eval.consistency.run_consistency_check,
    generalized to N simultaneous walkers sharing one TrackManager --
    so fusion, gating, and lifecycle timing are all the real M4
    behaviour, not a re-implementation of it."""

    def __init__(self, cfg: SourceConfig, tick_hz: float = 30.0, camera_height_m: float = 2.3, camera_pitch_deg: float = 25.0):
        # Imported lazily: these come from poi_localization, which is a
        # sibling package in the merged repo, not a hard dependency of
        # poi_present's own test suite (scripted mode needs none of it).
        from poi_localization.config import M4Config
        from poi_localization.frames.types import FloorFrameTransform
        from poi_localization.tracking.track_manager import TrackManager
        from poi_perception.contracts import Intrinsics

        self.cfg = cfg
        self.tick_hz = tick_hz
        self._rng = np.random.default_rng(cfg.synthetic_seed)
        self._rect = cfg.synthetic_rect_m
        self._m4_cfg = M4Config.default()
        self._intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480, depth_scale=0.001, baseline=0.05)
        self._T = FloorFrameTransform.from_height_and_yaw(camera_height_m, camera_pitch_deg, 0.0)
        self._tm = TrackManager(self._m4_cfg)
        self._stopped = False
        self._t = 0.0
        self._walkers: List[Dict] = []
        for i in range(cfg.synthetic_n_walkers):
            self._walkers.append(self._new_walker(i + 1))

    def stop(self) -> None:
        self._stopped = True

    def _new_walker(self, det_id: int) -> Dict:
        x0, x1, y0, y1 = self._rect
        return {
            "det_id": det_id,
            "x": float(self._rng.uniform(x0, x1)),
            "y": float(self._rng.uniform(y0, y1)),
            "wp": (float(self._rng.uniform(x0, x1)), float(self._rng.uniform(y0, y1))),
            "speed": float(self._rng.uniform(0.8, 1.4)),
        }

    def _step(self, w: Dict, dt: float) -> None:
        dx, dy = w["wp"][0] - w["x"], w["wp"][1] - w["y"]
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            x0, x1, y0, y1 = self._rect
            w["wp"] = (float(self._rng.uniform(x0, x1)), float(self._rng.uniform(y0, y1)))
        else:
            w["x"] += w["speed"] * dt * dx / dist
            w["y"] += w["speed"] * dt * dy / dist

    def _tick(self, t: float, dt: float):
        from poi_localization.measurement.depth_measurement import compute_depth_measurement
        from poi_localization.measurement.raycast_measurement import compute_raycast_measurement
        from poi_localization.tracking.track_manager import DetectionMeasurements

        dets = []
        for w in self._walkers:
            self._step(w, dt)
            gp = np.array([w["x"], w["y"], 0.0])
            torso = np.array([w["x"], w["y"], 1.25])

            p_cam = self._T.R.T @ (torso - self._T.t)
            if p_cam[2] <= 0.05:
                continue
            u = self._intr.fx * p_cam[0] / p_cam[2] + self._intr.cx
            v = self._intr.fy * p_cam[1] / p_cam[2] + self._intr.cy
            z_true = p_cam[2]

            z_meas = (z_true - 0.12) + self._rng.normal(0, 0.015 * z_true + 0.005) + 0.008 * z_true
            un, vn = u + self._rng.normal(0, 1.5), v + self._rng.normal(0, 1.5)
            depth_raw = np.zeros((self._intr.height, self._intr.width), dtype=np.uint16)
            x0i, x1i = max(0, int(un) - 12), min(self._intr.width, int(un) + 12)
            y0i, y1i = max(0, int(vn) - 28), min(self._intr.height, int(vn) + 28)
            depth_raw[y0i:y1i, x0i:x1i] = int(round(z_meas / self._intr.depth_scale))
            poly = [(un - 12, vn - 28), (un + 12, vn - 28), (un + 12, vn + 28), (un - 12, vn + 28)]
            depth_m = compute_depth_measurement(depth_raw, poly, "full", self._intr, self._T, self._m4_cfg.depth_measurement)

            ankle_p_cam = self._T.R.T @ (gp - self._T.t)
            ray_m = None
            if ankle_p_cam[2] > 0.05:
                au = self._intr.fx * ankle_p_cam[0] / ankle_p_cam[2] + self._intr.cx
                av = self._intr.fy * ankle_p_cam[1] / ankle_p_cam[2] + self._intr.cy
                ray_m = compute_raycast_measurement(
                    (au + self._rng.normal(0, 2.0), av + self._rng.normal(0, 2.0)),
                    "ankles", self._intr, self._T, self._m4_cfg.raycast_measurement,
                )

            if depth_m is None and ray_m is None:
                continue
            dets.append(DetectionMeasurements(w["det_id"], 0.9, (un - 20, vn - 60, un + 20, vn + 10), depth_m, ray_m, 1.72))

        outputs, _events = self._tm.update(t, dt, dets)
        wire = []
        for o in outputs:
            wire.append(
                TrackWire(
                    id=o.world_track_id,
                    state=o.state,
                    p=(o.position_xy[0], o.position_xy[1], 0.0),
                    v=(o.velocity_xy[0], o.velocity_xy[1], 0.0),
                    cov_xy=o.cov_xy,
                    height=o.height_m,
                    conf=o.conf,
                    src=o.src,
                    age_s=o.age_s,
                    bbox=o.bbox_px,
                )
            )
        return wire

    async def run(self, sink: Sink) -> None:
        dt = 1.0 / self.tick_hz
        period = dt
        next_tick = time.monotonic()
        while not self._stopped:
            tracks = self._tick(self._t, dt)
            await sink(
                FrameBundle(t_capture=time.time(), frame="local", map_id=None, cam_id="cam0-sim", tracks=tracks)
            )
            self._t += dt
            next_tick += period
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))


def build_synthetic_source(cfg: SourceConfig, tick_hz: float = 30.0) -> TrackSource:
    if cfg.synthetic_mode == "realistic":
        return RealisticSyntheticSource(cfg, tick_hz=tick_hz)
    if cfg.synthetic_mode == "scripted":
        return ScriptedSyntheticSource(cfg, tick_hz=tick_hz)
    raise ValueError(f"unknown synthetic_mode {cfg.synthetic_mode!r}")
