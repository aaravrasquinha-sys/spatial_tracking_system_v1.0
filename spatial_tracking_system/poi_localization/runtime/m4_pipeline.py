"""
Section 13: "Build Module 4 as a stateful-but-pluggable stage (input:
one frame's detections + depth; output: this frame's track list) so it
slots into [Module 6's] daemon shape later."

This is that stage. M4Pipeline.process(frame, detections) is the entire
public surface runtime/m4_daemon.py (or, later, Module 6) needs to call.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from poi_localization.config import M4Config
from poi_localization.frames.frame_provider import FrameProvider
from poi_localization.io.world_track import (
    JsonlWorldTrackWriter,
    WorldTrackPhaseA,
    WorldTrackPhaseB,
)
from poi_localization.log import get_logger
from poi_localization.measurement.depth_measurement import compute_depth_measurement, median_depth_in_polygon
from poi_localization.measurement.raycast_measurement import compute_raycast_measurement
from poi_localization.tracking.gates import WalkableGrid
from poi_localization.tracking.height_estimator import estimate_instantaneous_height
from poi_localization.tracking.track_manager import DetectionMeasurements, TrackManager, TrackEvent, WorldTrackOutput
from poi_perception.contracts import Frame
from poi_perception.io.detection2d import Detection2D

log = get_logger("runtime.m4_pipeline")

TracksCallback = Callable[[Frame, list], None]


class M4Pipeline:
    def __init__(
        self,
        cfg: M4Config,
        frame_provider: FrameProvider,
        walkable_grid: Optional[WalkableGrid] = None,
        distortion_coeffs: Optional[Tuple] = None,
        writer: Optional[JsonlWorldTrackWriter] = None,
        on_tracks: Optional[TracksCallback] = None,
        default_dt: float = 1.0 / 30.0,
    ):
        self.cfg = cfg
        # Fetched once (Section 2: Phase A doesn't change during a run;
        # Phase B is loaded once from the calibration file) -- not
        # re-queried on every frame.
        self.floor_transform = frame_provider.get_transform()
        self.walkable_grid = walkable_grid
        self.distortion_coeffs = distortion_coeffs
        self.writer = writer
        self.on_tracks = on_tracks
        self.default_dt = default_dt

        self.track_manager = TrackManager(cfg)
        self._prev_t: Optional[float] = None
        self.total_events: List[TrackEvent] = []

    def _build_detection_measurements(self, frame: Frame, det: Detection2D) -> Optional[DetectionMeasurements]:
        # Reuse height's keypoint-confidence threshold: both are "is this
        # upper-body keypoint trustworthy enough to use for geometry"
        # checks, and adding a third near-duplicate config knob just for
        # this wasn't worth it.
        shoulder_conf_thresh = self.cfg.height.keypoint_conf_thresh
        l_sh = det.keypoints.get("left_shoulder")
        r_sh = det.keypoints.get("right_shoulder")
        shoulder_l_px = (l_sh[0], l_sh[1]) if l_sh is not None and l_sh[2] >= shoulder_conf_thresh else None
        shoulder_r_px = (r_sh[0], r_sh[1]) if r_sh is not None and r_sh[2] >= shoulder_conf_thresh else None
        depth_m = compute_depth_measurement(
            frame.depth, det.torso_polygon_px, det.torso_quality, frame.intr,
            self.floor_transform, self.cfg.depth_measurement,
            shoulder_l_px=shoulder_l_px, shoulder_r_px=shoulder_r_px,
        )
        ray_m = compute_raycast_measurement(
            det.footpoint_px, det.footpoint_source, frame.intr, self.floor_transform,
            self.cfg.raycast_measurement, self.distortion_coeffs,
        )
        if depth_m is None and ray_m is None:
            return None

        fallback_depth_m = median_depth_in_polygon(
            frame.depth, det.torso_polygon_px, det.torso_quality, frame.intr, self.cfg.depth_measurement
        )
        height_sample = estimate_instantaneous_height(
            frame.depth, det.keypoints, frame.intr, self.floor_transform, self.cfg.height,
            fallback_depth_m=fallback_depth_m,
        )

        return DetectionMeasurements(
            track_id_2d=det.track_id_2d,
            det_conf=det.det_conf,
            bbox_px=det.bbox,
            depth_measurement=depth_m,
            raycast_measurement=ray_m,
            height_sample_m=height_sample,
        )

    def _to_phase_record(self, o: WorldTrackOutput, t_capture: float):
        if self.cfg.phase == "A":
            return WorldTrackPhaseA(
                t_capture=t_capture,
                world_track_id=o.world_track_id,
                state=o.state,
                p_local=(o.position_xy[0], o.position_xy[1], 0.0),
                v_local=(o.velocity_xy[0], o.velocity_xy[1], 0.0),
                cov_xy=o.cov_xy,
                height_m=o.height_m,
                src=o.src,
                age_s=o.age_s,
            )
        return WorldTrackPhaseB(
            id=o.world_track_id,
            state=o.state,
            p=(o.position_xy[0], o.position_xy[1], 0.0),
            v=(o.velocity_xy[0], o.velocity_xy[1], 0.0),
            cov_xy=o.cov_xy,
            height=o.height_m,
            conf=o.conf,
            src=o.src,
            age_s=o.age_s,
            bbox=o.bbox_px if o.bbox_px is not None else (0.0, 0.0, 0.0, 0.0),
        )

    def process(self, frame: Frame, detections: List[Detection2D]):
        dt = (frame.t - self._prev_t) if self._prev_t is not None else self.default_dt
        dt = max(dt, 1e-4)
        self._prev_t = frame.t

        det_measurements = []
        for det in detections:
            dm = self._build_detection_measurements(frame, det)
            if dm is not None:
                det_measurements.append(dm)

        outputs, events = self.track_manager.update(frame.t, dt, det_measurements, walkable_grid=self.walkable_grid)
        for e in events:
            if e.kind == "measurement_rejected":
                log.debug(f"track {e.world_track_id}: {e.detail}")
            elif e.kind == "2d_id_mismatch":
                log.info(f"track {e.world_track_id}: {e.detail}")
            elif e.kind in ("track_lost", "track_confirmed"):
                log.info(f"track {e.world_track_id}: {e.kind} ({e.detail})")
        self.total_events.extend(events)

        records = [self._to_phase_record(o, frame.t) for o in outputs]

        if self.writer is not None:
            self.writer.write_frame(frame.t, frame.frame_id, records)
        if self.on_tracks is not None:
            self.on_tracks(frame, records)
        return records
