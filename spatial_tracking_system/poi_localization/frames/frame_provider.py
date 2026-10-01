"""
Section 3: "The floor-frame transform: Phase A computes this itself at
startup from IMU + depth; Phase B reads it from Module 2's calibration
file instead. Module 4's internal code should treat 'get the floor-frame
transform' as a single swappable function so this substitution is clean."

This is that function. Everything else in poi_localization (measurement,
tracking, the visualizer) takes a FrameProvider and calls .get_transform()
-- it never knows or cares which phase produced it.
"""
from __future__ import annotations

from typing import Optional, Protocol

from poi_localization.config import M4Config
from poi_localization.frames.local_floor_frame import (
    CalibrationCaptureData,
    capture_calibration_data,
    estimate_floor_frame_from_points,
)
from poi_localization.frames.room_frame import load_room_frame
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.log import get_logger

log = get_logger("frames.frame_provider")


class FrameProvider(Protocol):
    def get_transform(self) -> FloorFrameTransform: ...


class FixedFrameProvider:
    """Wraps an already-known transform. Used for Phase A with
    fixed_transform set, and directly in tests/the synthetic demo."""

    def __init__(self, transform: FloorFrameTransform):
        self._transform = transform

    def get_transform(self) -> FloorFrameTransform:
        return self._transform


class LocalFloorFrameProvider(FixedFrameProvider):
    """Phase A, computed once at startup from a live calibration
    capture (Section 2). Subclasses FixedFrameProvider since, once
    estimated, Phase A's transform doesn't change during a run -- there's
    no live re-fitting on every frame (that would need its own
    stability/drift analysis this design doc doesn't ask for)."""

    def __init__(self, transform: FloorFrameTransform, capture: Optional[CalibrationCaptureData] = None):
        super().__init__(transform)
        self.capture = capture


class RoomFrameProvider(FixedFrameProvider):
    """Phase B, loaded once from Module 2's calibration file."""

    pass


def build_phase_a_provider(cfg: M4Config, cam_cfg=None) -> LocalFloorFrameProvider:
    """cam_cfg: a poi_perception.config.CameraConfig, required unless
    cfg.phase_a.fixed_transform is set. Performs the live IMU+depth
    calibration capture (hardware) unless a fixed_transform override is
    configured, in which case no camera access is needed at all."""
    fixed = cfg.phase_a.fixed_transform
    if fixed is not None:
        log.info(
            f"Phase A: using fixed_transform override (height={fixed.height_m}m, "
            f"pitch={fixed.pitch_deg}deg, yaw={fixed.yaw_deg}deg) -- skipping live "
            f"IMU+depth calibration capture entirely."
        )
        transform = FloorFrameTransform.from_height_and_yaw(
            fixed.height_m, fixed.pitch_deg, fixed.yaw_deg, frame_name="local"
        )
        return LocalFloorFrameProvider(transform, capture=None)

    if cam_cfg is None:
        raise ValueError("cam_cfg is required for a live Phase A calibration capture "
                          "(no fixed_transform is set in config)")

    capture = capture_calibration_data(
        cam_cfg,
        seconds=cfg.phase_a.calibration_capture_seconds,
        depth_stride=cfg.phase_a.depth_point_stride,
    )
    transform, diagnostics = estimate_floor_frame_from_points(
        capture.points_cam,
        up_hint_cam=capture.up_hint_cam,
        ransac_dist_thresh_m=cfg.phase_a.ransac_dist_thresh_m,
        ransac_iterations=cfg.phase_a.ransac_iterations,
        gravity_alignment_deg_max=cfg.phase_a.gravity_alignment_deg_max,
        candidate_percentile=cfg.phase_a.candidate_percentile,
        min_candidate_points=cfg.phase_a.min_candidate_points,
    )
    return LocalFloorFrameProvider(transform, capture=capture)


def build_phase_b_provider(cfg: M4Config) -> RoomFrameProvider:
    if not cfg.phase_b.calibration_path:
        raise ValueError("phase_b.calibration_path is required for Phase B")
    transform = load_room_frame(cfg.phase_b.calibration_path, expected_map_id=cfg.phase_b.expected_map_id)
    return RoomFrameProvider(transform)


def build_frame_provider(cfg: M4Config, cam_cfg=None) -> FrameProvider:
    """The single entry point everything else should call. Section 2's
    Phase A -> Phase B swap is, at the call site, changing cfg.phase
    from "A" to "B" (plus filling in phase_b.calibration_path) -- no
    other code changes."""
    if cfg.phase == "A":
        return build_phase_a_provider(cfg, cam_cfg=cam_cfg)
    if cfg.phase == "B":
        return build_phase_b_provider(cfg)
    raise ValueError(f"Unknown M4Config.phase: {cfg.phase!r} (expected 'A' or 'B')")
