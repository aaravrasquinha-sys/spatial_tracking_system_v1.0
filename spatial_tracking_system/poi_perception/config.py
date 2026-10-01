"""
M3 run configuration.

One JSON file holds every tunable mentioned in the design doc as
"start here, tune from Section 8/9's data": ankle confidence threshold,
ByteTrack's two thresholds and lost-track buffer, torso-shrink factor,
and camera/model paths. Keeping these in config rather than constants
is what makes the Section 9 evaluation protocol ("tune the lost-track
buffer for this room, not COCO video defaults") a config edit instead
of a code change.

See configs/m3.example.json for a starting point.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CameraConfig:
    cam_id: str = "cam0"
    width: int = 640
    height: int = 480
    fps: int = 30
    serial: Optional[str] = None
    # Expected D435i intrinsics for this rig (Section 1: "matching the
    # D435i color stream exactly"). A mismatch at runtime is a hard
    # warning, never a silent override -- see capture/realsense_source.py.
    expected_fx: float = 606.75
    expected_fy: float = 606.57
    expected_cx: float = 320.19
    expected_cy: float = 237.06
    # --- RealSense tuning ---
    enable_global_time: bool = True
    visual_preset: str = "high_accuracy"
    laser_power: Optional[float] = 360.0
    enable_post_processing: bool = True
    spatial_filter_magnitude: float = 2.0
    spatial_filter_smooth_alpha: float = 0.5
    spatial_filter_smooth_delta: float = 20.0
    temporal_filter_smooth_alpha: float = 0.4
    temporal_filter_smooth_delta: float = 20.0
    hole_filling_mode: int = 1


@dataclass
class FootpointConfig:
    # Section 5: "start at 0.5, tune from Section 8's data"
    ankle_conf_thresh: float = 0.5
    # Section 5: shrink the shoulder/hip quad ~15% toward its centroid
    torso_shrink_frac: float = 0.15
    # Section 5 fallback strip: fraction of box height, (top, bottom)
    fallback_strip_frac: tuple = (0.20, 0.60)


@dataclass
class TrackConfig:
    # ByteTrack two-tier thresholds (Section 6). Defaults are stock
    # ByteTrack starting points; both are meant to be tuned against this
    # room's Section 9 scenario clips, not trusted as-is.
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    match_iou_thresh: float = 0.3
    # How much a detection's mean keypoint confidence can boost its
    # matching score, on top of IoU (Section 6: "track with keypoint
    # confidence in the loop, not just box IoU + score"). 0 disables it.
    keypoint_conf_weight: float = 0.15
    # "Lost track" buffer, in frames -- tune from measured furniture
    # occlusion durations (Section 6), not left at a COCO video default.
    lost_buffer_frames: int = 30
    # Consecutive hits required before a track is "confirmed" rather
    # than "new" -- kept separate from M4's own tentative/confirmed
    # semantics; this is purely the 2D track's own bookkeeping.
    min_hits_to_confirm: int = 3


@dataclass
class ModelConfig:
    backend: str = "trt"  # "trt" | "ultralytics" | "mock"
    engine_path: Optional[str] = None  # .engine, required for backend="trt"
    weights_path: Optional[str] = None  # .pt, required for backend="ultralytics"
    input_width: int = 640
    input_height: int = 640  # Updated to match square model input requirements
    det_conf_thresh: float = 0.1  # pre-NMS floor; ByteTrack does its own low-thresh filtering
    nms_iou_thresh: float = 0.45
    keypoint_conf_thresh_for_decode: float = 0.05  # below this, keypoint is zeroed, not decoded


@dataclass
class MaskConfig:
    masks_path: Optional[str] = None  # JSON file, keyed by cam_id -- see runtime/masks.py


@dataclass
class OutputConfig:
    jsonl_dir: str = "logs/detections"
    clips_dir: str = "logs/clips"
    low_confidence_streak_s: float = 1.0  # auto-clip trigger (Section 8)


@dataclass
class M3Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    footpoint: FootpointConfig = field(default_factory=FootpointConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path) -> "M3Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            camera=CameraConfig(**raw.get("camera", {})),
            footpoint=FootpointConfig(**raw.get("footpoint", {})),
            track=TrackConfig(**raw.get("track", {})),
            model=ModelConfig(**raw.get("model", {})),
            mask=MaskConfig(**raw.get("mask", {})),
            output=OutputConfig(**raw.get("output", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def default(cls) -> "M3Config":
        return cls()
