"""
WP-LIVE N0: one shared capture profile, applied identically wherever a
D435i is opened -- live mapping, M2 calibration, and (in the STS repo)
M3/M4 runtime. Before this, pyslam's own RealSenseSource used device
defaults (no preset, no laser-power pin, no post-processing filters,
global_time not enabled) while the STS repo's realsense_source.py set
all of these explicitly. Two different depth pipelines feeding the same
downstream map/calibration/watchdog is exactly the kind of mismatch
that shows up as an unexplained few-percent scale or noise difference
weeks later -- see the planning notes this module implements.

This is deliberately NOT RealSense-API-shaped: it's a plain, hashable
dataclass that both this repo and the STS repo can import (or
re-implement field-for-field against) without either one depending on
the other. apply_to_realsense() is the only place pyrealsense2 is
touched, and it degrades exactly like realsense.py's existing option
calls -- best-effort, warn, never crash.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import hashlib
import json


@dataclass(frozen=True)
class CaptureProfile:
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_global_time: bool = True
    visual_preset: str = "high_accuracy"
    laser_power: Optional[float] = 360.0
    enable_post_processing: bool = True
    spatial_filter_magnitude: float = 2.0
    spatial_filter_smooth_alpha: float = 0.5
    spatial_filter_smooth_delta: float = 20.0
    temporal_filter_smooth_alpha: float = 0.4
    temporal_filter_smooth_delta: float = 20.0
    # WP-LIVE: hole filling is deliberately EXCLUDED from the mapping
    # profile default (0 = off), unlike STS's M3 profile. Hole-filled
    # depth invents values at range/edge discontinuities -- fine for
    # M4's torso-median sampling, wrong for anything geometric (ICP,
    # plane fitting, keyframe depth fusion, the watchdog's reference
    # depth). A capture profile used for LIVE TRACKING/PERCEPTION may
    # set this back to a real mode; the MAPPING profile (below) must not.
    hole_filling_mode: int = 0
    # Raw depth (pre-filter) is what gets recorded to .bag; filters are
    # always applied in-process from raw, per stream, never baked into
    # the recording -- see capture_profile.mapping.json's own comment
    # and RUNBOOK_LIVE.md Part 0.
    depth_min_m: float = 0.3
    depth_max_m: float = 6.0

    def content_hash(self) -> str:
        """Short, stable hash identifying this exact profile -- stored in
        bag metadata, the manifest, and calibration.json's own
        capture_profile_sha so a downstream consumer can detect (not
        silently tolerate) a profile mismatch between mapping and
        runtime."""
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def load(cls, path: str) -> "CaptureProfile":
        raw = json.loads(open(path).read())
        return cls(**raw)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


_VISUAL_PRESET_NAMES = {"default", "high_accuracy", "high_density", "medium_density"}


def apply_to_realsense(profile: CaptureProfile, rs, depth_sensor, color_sensor=None, log=None) -> None:
    """Applies `profile` to an already-started RealSense sensor pair.
    Mirrors the STS repo's realsense_source.py::_configure_sensors,
    generalised to take any (rs module, sensor) pair rather than being
    embedded in one capture class -- both pyslam's RealSenseSource and
    (once this profile is adopted there) the STS repo's own capture
    class should call this instead of each hand-rolling the option
    calls. Every option is best-effort: log + continue, never raise,
    same discipline realsense.py already uses for R_body_cam and
    baseline queries.
    """
    def _warn(msg):
        if log is not None:
            log.warning(msg)

    if profile.enable_global_time:
        for sensor in (depth_sensor, color_sensor):
            if sensor is None:
                continue
            try:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            except Exception as e:
                _warn(f"Could not enable global_time_enabled: {e}")

    if profile.visual_preset:
        if profile.visual_preset not in _VISUAL_PRESET_NAMES:
            _warn(f"CaptureProfile.visual_preset={profile.visual_preset!r} not in "
                  f"{sorted(_VISUAL_PRESET_NAMES)} -- leaving sensor preset unchanged.")
        else:
            try:
                if depth_sensor.supports(rs.option.visual_preset):
                    preset_enum = {
                        "default": rs.rs400_visual_preset.default,
                        "high_accuracy": rs.rs400_visual_preset.high_accuracy,
                        "high_density": rs.rs400_visual_preset.high_density,
                        "medium_density": rs.rs400_visual_preset.medium_density,
                    }[profile.visual_preset]
                    depth_sensor.set_option(rs.option.visual_preset, float(preset_enum))
            except Exception as e:
                _warn(f"Could not set visual_preset={profile.visual_preset!r}: {e}")

    if profile.laser_power is not None:
        try:
            if depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                power = float(min(max(profile.laser_power, rng.min), rng.max))
                depth_sensor.set_option(rs.option.laser_power, power)
        except Exception as e:
            _warn(f"Could not set laser_power={profile.laser_power}: {e}")


def build_post_processing_filters(profile: CaptureProfile, rs):
    """Same ordering Intel recommends and the STS repo already uses:
    spatial -> temporal -> (optional) hole-filling, all in disparity
    space. Returns an ordered list of filter objects to .process()
    through in sequence; empty list if enable_post_processing is False.
    Mapping profiles should leave hole_filling_mode=0 -- see the
    dataclass docstring."""
    if not profile.enable_post_processing:
        return []
    filters = []
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, profile.spatial_filter_magnitude)
    spatial.set_option(rs.option.filter_smooth_alpha, profile.spatial_filter_smooth_alpha)
    spatial.set_option(rs.option.filter_smooth_delta, profile.spatial_filter_smooth_delta)
    filters.append(spatial)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, profile.temporal_filter_smooth_alpha)
    temporal.set_option(rs.option.filter_smooth_delta, profile.temporal_filter_smooth_delta)
    filters.append(temporal)

    if profile.hole_filling_mode != 0:
        hf = rs.hole_filling_filter()
        hf.set_option(rs.option.holes_fill, profile.hole_filling_mode)
        filters.append(hf)
    return filters
