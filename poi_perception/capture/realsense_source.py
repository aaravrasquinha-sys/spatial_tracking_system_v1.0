"""
Intel RealSense D435i capture for M3.

This intentionally duplicates (rather than imports) the relevant slice of
pyslam.sensors.realsense.RealSenseSource from M1/M2: same rgb8-with-
bgr8-fallback logic, same hardware-timestamp discipline, same "warn on
intrinsics mismatch, never silently override" policy. M3 doesn't need
IMU, recording-for-mapping, or anything else that module carries for
SLAM's sake -- see poi_perception/contracts.py for why the duplication
is deliberate rather than an oversight.

Frame.depth passes straight through, aligned to color, for M4 to
consume later; M3 itself only ever reads frame.rgb.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from poi_perception.config import CameraConfig
from poi_perception.contracts import Frame, Intrinsics
from poi_perception.log import get_logger

log = get_logger("capture.realsense")

# How far a queried intrinsic may drift from the expected value (Section 1
# / CameraConfig) before we warn. This is advisory only -- we always use
# the device's own reported values, never the expected ones, to build
# Intrinsics; see the constructor.
_INTRINSICS_WARN_TOL_PX = 2.0

_VISUAL_PRESET_NAMES = {"default", "high_accuracy", "high_density", "medium_density"}


def _build_post_processing_filters(cam_cfg: CameraConfig, rs):
    """Ordered per Intel's own recommended pipeline: decimation (skipped
    here -- would change resolution, and M3/M4 both assume the color
    grid's resolution) -> depth-to-disparity -> spatial -> temporal ->
    disparity-to-depth -> hole-filling. Spatial/temporal filters operate
    more correctly in disparity space (their edge-preserving smoothing
    assumes inverse-depth-like error statistics), which is why the
    domain-transform filters bracket them rather than filtering the raw
    depth frame directly."""
    filters = []
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, cam_cfg.spatial_filter_magnitude)
    spatial.set_option(rs.option.filter_smooth_alpha, cam_cfg.spatial_filter_smooth_alpha)
    spatial.set_option(rs.option.filter_smooth_delta, cam_cfg.spatial_filter_smooth_delta)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, cam_cfg.temporal_filter_smooth_alpha)
    temporal.set_option(rs.option.filter_smooth_delta, cam_cfg.temporal_filter_smooth_delta)

    filters.append(depth_to_disparity)
    filters.append(spatial)
    filters.append(temporal)
    filters.append(disparity_to_depth)

    if cam_cfg.hole_filling_mode != 0:
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, cam_cfg.hole_filling_mode)
        filters.append(hole_filling)

    return filters


class RealSenseSource:
    """Live capture from a connected D435i, or (playback_path set)
    deterministic replay of a native .bag. Color+depth are aligned to the
    color stream so Frame.rgb and Frame.depth always share one pixel
    grid -- required for M3's keypoints and M4's later depth sampling to
    index the same pixels without a remapping step."""

    def __init__(
        self,
        cam_cfg: CameraConfig,
        record_path: Optional[str] = None,
        playback_path: Optional[str] = None,
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 is not importable. Install with `pip install "
                "pyrealsense2` (or the Jetson-specific wheel/apt package on "
                "the Orin). Use capture.synthetic_source.SyntheticSource for "
                "hardware-free development."
            ) from e

        if record_path is not None and playback_path is not None:
            raise ValueError("record_path and playback_path are mutually exclusive")

        self._rs = rs
        self.cam_cfg = cam_cfg
        self._playback = playback_path is not None
        self._frame_id = 0

        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # Prefer rgb8 directly (no channel-flip copy needed); fall back to
        # bgr8 + flip if this unit/firmware doesn't support it, exactly as
        # M1/M2's capture does.
        self._color_format = rs.format.rgb8
        if playback_path is not None:
            cfg.enable_device_from_file(playback_path, repeat_playback=False)
        else:
            if cam_cfg.serial:
                cfg.enable_device(cam_cfg.serial)
            cfg.enable_stream(
                rs.stream.color, cam_cfg.width, cam_cfg.height, rs.format.rgb8, cam_cfg.fps
            )
            cfg.enable_stream(
                rs.stream.depth, cam_cfg.width, cam_cfg.height, rs.format.z16, cam_cfg.fps
            )
            if record_path:
                cfg.enable_record_to_file(record_path)

        profile = self.pipeline.start(cfg)

        try:
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        except Exception:
            # rgb8 not supported on this unit -- fall back to bgr8.
            self.pipeline.stop()
            self._color_format = rs.format.bgr8
            cfg2 = rs.config()
            if playback_path is not None:
                cfg2.enable_device_from_file(playback_path, repeat_playback=False)
            else:
                if cam_cfg.serial:
                    cfg2.enable_device(cam_cfg.serial)
                cfg2.enable_stream(
                    rs.stream.color, cam_cfg.width, cam_cfg.height, rs.format.bgr8, cam_cfg.fps
                )
                cfg2.enable_stream(
                    rs.stream.depth, cam_cfg.width, cam_cfg.height, rs.format.z16, cam_cfg.fps
                )
                if record_path:
                    cfg2.enable_record_to_file(record_path)
            profile = self.pipeline.start(cfg2)
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            log.warning("rgb8 not supported by this device/firmware; using bgr8 + flip.")

        self._configure_sensors(profile, cam_cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        intr = color_stream.get_intrinsics()
        self.intr = Intrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            width=intr.width,
            height=intr.height,
            depth_scale=depth_scale,
        )
        self._warn_if_intrinsics_drifted(cam_cfg)

        self.align = rs.align(rs.stream.color)
        self._post_filters = (
            _build_post_processing_filters(cam_cfg, rs) if cam_cfg.enable_post_processing else []
        )
        log.info(
            f"RealSense started: {self.intr.width}x{self.intr.height} "
            f"fx={self.intr.fx:.2f} fy={self.intr.fy:.2f} "
            f"cx={self.intr.cx:.2f} cy={self.intr.cy:.2f} "
            f"depth_scale={self.intr.depth_scale:.6f} "
            f"post_processing={'on' if self._post_filters else 'off'}"
        )

    def _configure_sensors(self, profile, cam_cfg: CameraConfig) -> None:
        """Apply the tuning CameraConfig calls for: global time, visual
        preset, laser power. Every option is set best-effort with a
        warning (never a crash) on failure -- a recorded .bag played
        back in playback mode, an older firmware missing an option, or a
        depth-sensor-less test double should all still run, just without
        that particular improvement, rather than refusing to start."""
        rs = self._rs
        device = profile.get_device()

        if cam_cfg.enable_global_time:
            for sensor in device.query_sensors():
                try:
                    if sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled, 1)
                except Exception as e:
                    log.warning(f"Could not enable global_time_enabled on {sensor.get_info(rs.camera_info.name)}: {e}")

        try:
            depth_sensor = device.first_depth_sensor()
        except Exception:
            depth_sensor = None
        if depth_sensor is None:
            return

        if cam_cfg.visual_preset:
            if cam_cfg.visual_preset not in _VISUAL_PRESET_NAMES:
                log.warning(
                    f"CameraConfig.visual_preset={cam_cfg.visual_preset!r} is not one of "
                    f"{sorted(_VISUAL_PRESET_NAMES)} -- leaving the sensor's current preset "
                    f"as-is. A typo here silently keeps the noisier default preset, which is "
                    f"exactly the failure mode this option exists to prevent, so it's a hard "
                    f"warning rather than a best-effort guess at what you meant."
                )
            else:
                try:
                    if depth_sensor.supports(rs.option.visual_preset):
                        preset_enum = {
                            "default": rs.rs400_visual_preset.default,
                            "high_accuracy": rs.rs400_visual_preset.high_accuracy,
                            "high_density": rs.rs400_visual_preset.high_density,
                            "medium_density": rs.rs400_visual_preset.medium_density,
                        }[cam_cfg.visual_preset]
                        depth_sensor.set_option(rs.option.visual_preset, float(preset_enum))
                except Exception as e:
                    log.warning(f"Could not set visual_preset={cam_cfg.visual_preset!r}: {e}")

        if cam_cfg.laser_power is not None:
            try:
                if depth_sensor.supports(rs.option.laser_power):
                    rng = depth_sensor.get_option_range(rs.option.laser_power)
                    power = float(np.clip(cam_cfg.laser_power, rng.min, rng.max))
                    depth_sensor.set_option(rs.option.laser_power, power)
            except Exception as e:
                log.warning(f"Could not set laser_power={cam_cfg.laser_power}: {e}")

    def _apply_post_filters(self, depth_frame):
        for f in self._post_filters:
            depth_frame = f.process(depth_frame)
        return depth_frame

    def _warn_if_intrinsics_drifted(self, cam_cfg: CameraConfig) -> None:
        expected = (cam_cfg.expected_fx, cam_cfg.expected_fy, cam_cfg.expected_cx, cam_cfg.expected_cy)
        actual = (self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy)
        drift = max(abs(a - b) for a, b in zip(expected, actual))
        if drift > _INTRINSICS_WARN_TOL_PX:
            log.warning(
                f"Queried intrinsics drifted {drift:.2f}px from the expected "
                f"rig values in CameraConfig (expected fx={expected[0]:.2f} "
                f"fy={expected[1]:.2f} cx={expected[2]:.2f} cy={expected[3]:.2f}; "
                f"got fx={actual[0]:.2f} fy={actual[1]:.2f} cx={actual[2]:.2f} "
                f"cy={actual[3]:.2f}). Using the queried values as-is -- this is "
                f"a hard warning, not a silent override. Update CameraConfig if "
                f"this is expected (different unit / firmware)."
            )

    def intrinsics(self) -> Intrinsics:
        return self.intr

    def __iter__(self) -> Iterator[Frame]:
        rs = self._rs
        while True:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                if self._playback:
                    log.info("Playback reached end of file.")
                return
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            if self._post_filters:
                # Filtering the ALIGNED depth frame (rather than filtering
                # before align) keeps this a small, local change -- align
                # is a per-pixel remap and commutes with these filters
                # closely enough for tracking purposes, and doing it here
                # means the filters never need to know about the color
                # stream at all.
                depth_frame = self._apply_post_filters(depth_frame)
            t = color_frame.get_timestamp() / 1000.0

            raw = np.asanyarray(color_frame.get_data())
            if self._color_format == rs.format.rgb8:
                rgb = raw.copy()
            else:
                rgb = raw[:, :, ::-1].copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()

            yield Frame(
                t=t,
                rgb=rgb,
                depth=depth,
                intr=self.intr,
                frame_id=self._frame_id,
                cam_id=self.cam_cfg.cam_id,
            )
            self._frame_id += 1

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
