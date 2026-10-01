"""
M4 run configuration. Every constant Section 4/5/14 of the design doc
calls a "guess" or "starting estimate" ("tune this once real data
exists", "don't treat the formulas as final") lives here, not hardcoded
in the measurement/tracking code, so Section 11's floor-marker data can
tune it without touching code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FixedTransformConfig:
    """Phase A override for hardware-free operation (tests, the
    synthetic demo, or simply skipping the live IMU+depth calibration
    capture because you already know the rig geometry). See
    frames.types.FloorFrameTransform.from_height_and_yaw."""

    height_m: float = 2.3
    pitch_deg: float = 20.0
    yaw_deg: float = 0.0


@dataclass
class PhaseAConfig:
    calibration_capture_seconds: float = 2.5
    depth_point_stride: int = 4
    ransac_dist_thresh_m: float = 0.02
    ransac_iterations: int = 500
    gravity_alignment_deg_max: float = 15.0
    candidate_percentile: float = 80.0
    min_candidate_points: int = 50
    # If set, skip the live hardware calibration capture entirely and
    # build the transform from these numbers instead (Section 2's "a
    # real, physically grounded frame" doesn't strictly require a fresh
    # live capture every run if you already trust the rig's geometry).
    fixed_transform: Optional[FixedTransformConfig] = None


@dataclass
class PhaseBConfig:
    calibration_path: Optional[str] = None
    expected_map_id: Optional[str] = None
    walkable_grid_path: Optional[str] = None  # Section 8: Phase B only


@dataclass
class DepthMeasurementConfig:
    min_valid_px_frac: float = 0.20  # Section 4: "start around 20% of the polygon's pixel area"
    min_range_m: float = 0.3
    # Was 3.0. The old cutoff hard-dropped depth (the "primary" 3D
    # measurement per the full plan's Module 4 rationale) for over half
    # the floor of a typical single-camera room, forcing the system onto
    # ray-cast-primary in exactly the situation the depth-primary design
    # was meant to avoid. Raised to the D435i's realistic usable range;
    # the (now-calibrated) noise model below de-weights long range
    # itself, via cov_xy, rather than a hard cliff.
    max_range_m: float = 6.0
    mad_reject_k: float = 3.5  # median-absolute-deviation outlier multiplier
    mad_floor_m: float = 0.03  # never reject tighter than this, even if MAD is ~0 on a flat surface
    body_thickness_m: float = 0.10  # facing-on default; see body_thickness_side_m for the orientation-dependent model
    body_thickness_side_m: float = 0.18  # half-depth when seen from the side (shoulder-to-shoulder axis parallel to the viewing ray)

    # --- Depth noise model ---
    # sigma_z(z) = sqrt((depth_noise_a_m + depth_noise_b_frac * z^2)^2)
    #              i.e. a range-independent floor plus a term that grows
    #              with the SQUARE of range, matching stereo depth's
    #              actual error shape (error ~ z^2 * sigma_disparity /
    #              (f * baseline)), PLUS a proportional bias term added
    #              in quadrature.
    #
    # The previous formula used sigma_disparity_px / (fx * baseline)
    # with fx = the COLOR stream's focal length (~607px at 640x480).
    # D435i depth is computed from the IR stereo pair, whose focal
    # length at this resolution is closer to 385-390px -- using the
    # color fx understated sigma_z by roughly (607/385)^2 =~ 2.5x, and
    # ignored the sensor's own ~0.5-1% proportional bias entirely. These
    # defaults are a documented, still-approximate correction (D435i
    # published spec: <2% RMS error at 2m under ideal conditions,
    # degrading with distance/surface/lighting); Section 11's
    # floor-marker acceptance data should replace them with a fit of
    # this same two-parameter form, not be treated as final.
    depth_stereo_fx_px: float = 390.0  # IR-pair focal length, NOT intr.fx (color)
    sigma_disparity_px: float = 0.1  # Section 4: "order 0.1px as a starting assumption"
    depth_bias_frac: float = 0.008  # proportional bias/precision term, added in quadrature
    lateral_px_noise: float = 2.0  # centroid-pixel jitter -> tangential uncertainty


@dataclass
class RaycastMeasurementConfig:
    pixel_noise_ankles_px: float = 2.0
    pixel_noise_ankle_single_px: float = 4.0
    pixel_noise_bbox_fallback_px: float = 8.0
    raycast_sigma_ceiling_m: float = 2.0  # Section 5: reject bbox_fallback rays beyond this uncertainty
    max_range_m: float = 8.0
    # A single confident ankle isn't just noisier than the midpoint of
    # both -- it's systematically biased by up to half a stride length
    # (~0.15-0.35m at normal walking pace) toward whichever foot is
    # forward, because the raycast model treats "the one ankle we can
    # see" as if it were "the point below the person's center of mass."
    # pixel_noise_ankle_single_px alone (a few pixels) cannot represent a
    # decimeter-scale gait bias; this additive, range-independent sigma
    # (in metres, combined in quadrature with the pixel-noise-derived
    # sigma) is a direct, physically-motivated correction for it. See
    # raycast_measurement.py.
    ankle_single_extra_sigma_m: float = 0.15


@dataclass
class HeightConfig:
    ema_alpha: float = 0.3
    min_height_m: float = 0.5
    max_height_m: float = 2.2
    keypoint_conf_thresh: float = 0.4
    top_keypoint_patch_radius_px: int = 2
    # Whichever top-of-head-ish keypoint is used (see footpoint.py's
    # _TOP_KEYPOINT_PRIORITY), none of them ARE the top of the head:
    # nose sits ~10-13cm below crown, eyes ~10-12cm, ears ~7-10cm. Using
    # the raw keypoint depth as "height" was a systematic ~10cm
    # UNDER-estimate for every single track, every single frame -- not
    # noise the EMA could average out. One fixed offset per keypoint
    # name, added after computing the raw floor-frame height.
    top_keypoint_head_offset_m: dict = field(
        default_factory=lambda: {
            "nose": 0.12,
            "left_eye": 0.11,
            "right_eye": 0.11,
            "left_ear": 0.09,
            "right_ear": 0.09,
        }
    )


@dataclass
class GateConfig:
    chi2_gate_2dof: float = 9.21  # 99% confidence, 2 DOF chi-square


@dataclass
class TrackLifecycleConfig:
    process_accel_noise: float = 1.5  # m/s^2, Section 6
    tentative_confirm_hits: int = 3
    tentative_window_s: float = 1.0
    coasting_budget_s: float = 1.0  # Section 9: "up to roughly a second"
    lost_grace_s: float = 1.0
    association_2d_hint_bonus: float = 2.0  # subtracted from cost when 2D IDs agree (never enough to pass a failed gate)


@dataclass
class OutputConfig:
    jsonl_dir: str = "logs/world_tracks"


@dataclass
class M4Config:
    phase: str = "A"  # "A" | "B"
    phase_a: PhaseAConfig = field(default_factory=PhaseAConfig)
    phase_b: PhaseBConfig = field(default_factory=PhaseBConfig)
    depth_measurement: DepthMeasurementConfig = field(default_factory=DepthMeasurementConfig)
    raycast_measurement: RaycastMeasurementConfig = field(default_factory=RaycastMeasurementConfig)
    height: HeightConfig = field(default_factory=HeightConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    track: TrackLifecycleConfig = field(default_factory=TrackLifecycleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path) -> "M4Config":
        raw = json.loads(Path(path).read_text())
        phase_a_raw = dict(raw.get("phase_a", {}))
        fixed_raw = phase_a_raw.pop("fixed_transform", None)
        phase_a = PhaseAConfig(
            fixed_transform=FixedTransformConfig(**fixed_raw) if fixed_raw else None,
            **phase_a_raw,
        )
        return cls(
            phase=raw.get("phase", "A"),
            phase_a=phase_a,
            phase_b=PhaseBConfig(**raw.get("phase_b", {})),
            depth_measurement=DepthMeasurementConfig(**raw.get("depth_measurement", {})),
            raycast_measurement=RaycastMeasurementConfig(**raw.get("raycast_measurement", {})),
            height=HeightConfig(**raw.get("height", {})),
            gates=GateConfig(**raw.get("gates", {})),
            track=TrackLifecycleConfig(**raw.get("track", {})),
            output=OutputConfig(**raw.get("output", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def default(cls) -> "M4Config":
        return cls()
