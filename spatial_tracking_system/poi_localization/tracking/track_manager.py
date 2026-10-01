"""
Section 9: track lifecycle (tentative -> confirmed -> coasting ->
lost -> ended) and 2D-to-world association, re-verified in the world
frame rather than inherited blindly from Module 3's 2D track ID.

Section 6: each available measurement (depth, raycast, either, or both)
is applied as its own independent update, each with its own chi-square
gate (Section 8) -- so even a detection that geometrically matched a
track can end up contributing nothing this frame if both its individual
measurements get rejected (logged distinctly from a plain miss, since
"a lot of rejections is a sign something upstream is wrong").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from poi_localization import geometry
from poi_localization.config import M4Config
from poi_localization.measurement.types import Measurement
from poi_localization.tracking.gates import WalkableGrid, chi_square_gate_pass, walkable_gate
from poi_localization.tracking.height_estimator import HeightEMA
from poi_localization.tracking.kalman_track import KalmanTrack
from poi_localization.log import get_logger

log = get_logger("tracking.track_manager")

_DISALLOWED_COST = 1e12


@dataclass
class DetectionMeasurements:
    """One frame's per-person bundle: M3's 2D hint plus whatever
    measurements Section 4/5 managed to compute this frame. At least one
    of depth_measurement/raycast_measurement should be non-None for this
    detection to participate in tracking at all; if both are None,
    the caller (m4_pipeline) shouldn't include it."""

    track_id_2d: int
    det_conf: float
    bbox_px: Tuple[float, float, float, float]
    depth_measurement: Optional[Measurement]
    raycast_measurement: Optional[Measurement]
    height_sample_m: Optional[float]


@dataclass
class WorldTrackOutput:
    """Phase-agnostic track output -- runtime/m4_pipeline.py converts
    this into WorldTrackPhaseA or WorldTrackPhaseB (Section 10)."""

    world_track_id: int
    state: str  # "confirmed" | "coasting" | "lost"
    position_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    cov_xy: Tuple[float, float, float]
    height_m: Optional[float]
    conf: float
    src: str  # "depth" | "raycast" | "fused" | "predicted"
    age_s: float
    bbox_px: Optional[Tuple[float, float, float, float]]


@dataclass
class TrackEvent:
    kind: str  # "measurement_rejected" | "2d_id_mismatch" | "track_confirmed" | "track_lost"
    world_track_id: Optional[int]
    detail: str


class _Track:
    def __init__(self, track_id: int, t_capture: float, cfg: M4Config):
        self.id = track_id
        self.kf: Optional[KalmanTrack] = None
        self.height_ema = HeightEMA(cfg.height.ema_alpha)
        self.state = "tentative"
        self.hits = 0
        self.time_since_update_s = 0.0
        self.created_t = t_capture
        self.tentative_start_t = t_capture
        self.last_2d_id: Optional[int] = None
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_det_conf: float = 0.0
        self.last_src = "predicted"
        self.just_transitioned_to_lost = False


def _reference_measurement(det: DetectionMeasurements) -> Measurement:
    """Section 4: depth is "the stronger measurement" when available --
    used only to seed association cost / new-track initialization, never
    as the sole thing actually applied (both available measurements are
    still applied independently, Section 6)."""
    return det.depth_measurement if det.depth_measurement is not None else det.raycast_measurement


class TrackManager:
    def __init__(self, cfg: M4Config):
        self.cfg = cfg
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def active_track_count(self) -> int:
        return len(self._tracks)

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _filter_walkable(self, det: DetectionMeasurements, grid) -> DetectionMeasurements:
        """Section 8: reject a position measurement outright (Phase B
        only) if it lands outside the walkable grid -- applied per
        measurement, before anything else sees it."""
        if grid is None:
            return det
        depth_m = det.depth_measurement
        if depth_m is not None and not walkable_gate(depth_m.position_xy[0], depth_m.position_xy[1], grid):
            depth_m = None
        ray_m = det.raycast_measurement
        if ray_m is not None and not walkable_gate(ray_m.position_xy[0], ray_m.position_xy[1], grid):
            ray_m = None
        if depth_m is det.depth_measurement and ray_m is det.raycast_measurement:
            return det
        return DetectionMeasurements(
            track_id_2d=det.track_id_2d,
            det_conf=det.det_conf,
            bbox_px=det.bbox_px,
            depth_measurement=depth_m,
            raycast_measurement=ray_m,
            height_sample_m=det.height_sample_m,
        )

    def _apply_measurements(self, track: _Track, det: DetectionMeasurements, events: List[TrackEvent]) -> List[str]:
        """Section 6: apply each available measurement as its own
        independent update, each individually chi-square gated (Section
        8). Returns the list of src strings actually applied ("depth",
        "raycast", both, or neither)."""
        applied: List[str] = []
        for m, label in ((det.depth_measurement, "depth"), (det.raycast_measurement, "raycast")):
            if m is None:
                continue
            z = np.array(m.position_xy)
            R = geometry.cov_upper_to_matrix(m.cov_xy)
            d2 = track.kf.mahalanobis_distance_sq(z, R)
            if chi_square_gate_pass(d2, self.cfg.gates):
                track.kf.update(z, R)
                applied.append(label)
            else:
                events.append(
                    TrackEvent(
                        "measurement_rejected",
                        track.id,
                        f"src={label} mahalanobis_sq={d2:.2f} (gate={self.cfg.gates.chi2_gate_2dof})",
                    )
                )
        return applied

    def update(
        self,
        t_capture: float,
        dt: float,
        detections: List[DetectionMeasurements],
        walkable_grid: Optional[WalkableGrid] = None,
    ) -> Tuple[List[WorldTrackOutput], List[TrackEvent]]:
        events: List[TrackEvent] = []

        detections = [self._filter_walkable(d, walkable_grid) for d in detections]
        # Only detections with at least one surviving measurement participate.
        usable = [d for d in detections if d.depth_measurement is not None or d.raycast_measurement is not None]

        for track in self._tracks.values():
            if track.kf is not None:
                track.kf.predict(dt)

        track_ids = list(self._tracks.keys())
        n_tracks, n_dets = len(track_ids), len(usable)
        matched_track_to_det: Dict[int, int] = {}

        if n_tracks and n_dets:
            cost = np.full((n_tracks, n_dets), _DISALLOWED_COST)
            for i, tid in enumerate(track_ids):
                track = self._tracks[tid]
                if track.kf is None:
                    continue
                for j, det in enumerate(usable):
                    # Gate on whichever available measurement agrees best
                    # with the track's prediction, not just the "stronger"
                    # one (depth) alone: depth carries a deliberate,
                    # systematic body-thickness offset (Section 4), and
                    # once a track's covariance has been tightened by a
                    # precise ray-cast update, checking only depth's
                    # (offset) position against that tight covariance can
                    # spuriously fail even a correct association. Using
                    # the best-agreeing measurement is still a legitimate
                    # "is this plausibly the same person" test -- it only
                    # takes one piece of consistent evidence to associate;
                    # Section 6 already applies BOTH independently once
                    # matched, so this doesn't weaken the actual update.
                    best_d2 = None
                    for m in (det.depth_measurement, det.raycast_measurement):
                        if m is None:
                            continue
                        z = np.array(m.position_xy)
                        R = geometry.cov_upper_to_matrix(m.cov_xy)
                        d2 = track.kf.mahalanobis_distance_sq(z, R)
                        if best_d2 is None or d2 < best_d2:
                            best_d2 = d2
                    if best_d2 is not None and chi_square_gate_pass(best_d2, self.cfg.gates):
                        hint_bonus = self.cfg.track.association_2d_hint_bonus if det.track_id_2d == track.last_2d_id else 0.0
                        cost[i, j] = max(best_d2 - hint_bonus, 0.0)
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] >= _DISALLOWED_COST:
                    continue
                matched_track_to_det[track_ids[r]] = c

        matched_det_indices = set(matched_track_to_det.values())

        for tid in track_ids:
            track = self._tracks[tid]
            det_idx = matched_track_to_det.get(tid)
            applied: List[str] = []
            if det_idx is not None:
                det = usable[det_idx]
                applied = self._apply_measurements(track, det, events)
                if applied:
                    if track.last_2d_id is not None and det.track_id_2d != track.last_2d_id:
                        events.append(
                            TrackEvent(
                                "2d_id_mismatch",
                                track.id,
                                f"world track kept, but M3's 2D id changed {track.last_2d_id} -> {det.track_id_2d}",
                            )
                        )
                    if track.state in ("coasting", "lost"):
                        track.state = "confirmed"  # re-acquired after a gap
                        track.just_transitioned_to_lost = False
                    track.last_2d_id = det.track_id_2d
                    track.last_bbox = det.bbox_px
                    track.last_det_conf = det.det_conf
                    track.last_src = "fused" if len(applied) == 2 else applied[0]
                    track.height_ema.update(det.height_sample_m)
                    track.hits += 1
                    track.time_since_update_s = 0.0

            if not applied:
                track.time_since_update_s += dt
                if track.state == "tentative":
                    del self._tracks[tid]
                    continue
                if track.time_since_update_s <= self.cfg.track.coasting_budget_s:
                    if track.state != "coasting":
                        track.state = "coasting"
                elif track.state != "lost":
                    track.state = "lost"
                    track.just_transitioned_to_lost = True
                    events.append(TrackEvent("track_lost", track.id, f"age_s={t_capture - track.created_t:.2f}"))
                elif track.time_since_update_s > self.cfg.track.coasting_budget_s + self.cfg.track.lost_grace_s:
                    del self._tracks[tid]
                    continue

        for j, det in enumerate(usable):
            if j in matched_det_indices:
                continue
            ref = _reference_measurement(det)
            track = _Track(self._new_id(), t_capture, self.cfg)
            track.kf = KalmanTrack(ref.position_xy[0], ref.position_xy[1], self.cfg.track.process_accel_noise)
            track.kf.P[:2, :2] = geometry.cov_upper_to_matrix(ref.cov_xy)
            applied = self._apply_measurements(track, det, events)
            track.last_2d_id = det.track_id_2d
            track.last_bbox = det.bbox_px
            track.last_det_conf = det.det_conf
            track.last_src = "fused" if len(applied) == 2 else (applied[0] if applied else ref.src)
            track.height_ema.update(det.height_sample_m)
            track.hits = 1
            self._tracks[track.id] = track

        # Unified confirmation-check-and-output pass over every track now
        # in self._tracks (both pre-existing and just-spawned) -- doing
        # this in one place, after spawning, is what lets
        # tentative_confirm_hits=1 publish on a track's very first frame
        # instead of being one frame late.
        outputs: List[WorldTrackOutput] = []
        for track in list(self._tracks.values()):
            if track.state == "tentative":
                confirm_by_hits = track.hits >= self.cfg.track.tentative_confirm_hits
                stale = (t_capture - track.tentative_start_t) > self.cfg.track.tentative_window_s and not confirm_by_hits
                if confirm_by_hits:
                    track.state = "confirmed"
                    events.append(TrackEvent("track_confirmed", track.id, f"hits={track.hits}"))
                elif stale:
                    del self._tracks[track.id]
                    continue

            if track.state in ("confirmed", "coasting"):
                outputs.append(self._to_output(track, t_capture))
            elif track.state == "lost" and track.just_transitioned_to_lost:
                outputs.append(self._to_output(track, t_capture))
                track.just_transitioned_to_lost = False

        return outputs, events

    def _to_output(self, track: _Track, t_capture: float) -> WorldTrackOutput:
        x, y = track.kf.predicted_position()
        vx, vy = float(track.kf.x[2]), float(track.kf.x[3])
        cov = track.kf.position_cov()
        cov_xy = (float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1]))
        src = track.last_src if track.time_since_update_s < 1e-9 else "predicted"
        return WorldTrackOutput(
            world_track_id=track.id,
            state=track.state,
            position_xy=(x, y),
            velocity_xy=(vx, vy),
            cov_xy=cov_xy,
            height_m=track.height_ema.value,
            conf=track.last_det_conf,
            src=src,
            age_s=t_capture - track.created_t,
            bbox_px=track.last_bbox,
        )
