"""Runtime-slot adapters: capture_source, detector_backend, frame_provider,
watchdog_layers, calib_state_provider, track_source, sensor_model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

from sts.calib_health import CalibState
from sts.camera_model import CameraModel, load_camera_model
from sts.site import SiteConfig, SiteError


@dataclass
class CaptureChoice:
    kind: str                       # "realsense" | "synthetic"  (M3's build_source kinds)
    playback: Optional[str] = None  # a recorded .bag for replay


class M3CaptureSource:
    def choose(self, site: SiteConfig, cam_id: str, playback: Optional[str] = None,
               synthetic: bool = False) -> CaptureChoice:
        if synthetic:
            return CaptureChoice("synthetic")
        return CaptureChoice("realsense", playback)


class M3DetectorBackend:
    def build(self, m3_cfg):
        from poi_perception.runtime.m3_daemon import build_backend
        return build_backend(m3_cfg)


class M4FrameProvider:
    def build(self, m4_cfg, cam_cfg):
        from poi_localization.frames.frame_provider import build_frame_provider
        return build_frame_provider(m4_cfg, cam_cfg=cam_cfg)


KNOWN_LAYERS = ("depth", "icp", "tilt")


class DepthIcpLayers:
    """`watchdog_layers: "depth+icp"` -> the set of layers the supervisor may run."""

    def parse(self, spec: str) -> Set[str]:
        layers = {s.strip() for s in spec.split("+") if s.strip()}
        bad = layers - set(KNOWN_LAYERS)
        if bad:
            raise SiteError(f"unknown watchdog layer(s) {sorted(bad)}; known: {KNOWN_LAYERS}")
        if not layers:
            raise SiteError("watchdog_layers must name at least one layer")
        return layers


class WatchdogCalibState:
    """The supervisor drives this; M5 reads it."""
    def __new__(cls, initial: str = "missing") -> CalibState:
        return CalibState(initial)


class LiveTrackSource:
    def build(self, cam_id: str, frame_name: str, map_id: Optional[str]):
        from poi_present.sources.live import LiveSource
        return LiveSource(cam_id=cam_id, frame_name=frame_name, map_id=map_id)


class CameraModelFile:
    def load(self, site: SiteConfig, cam_id: str) -> CameraModel:
        return load_camera_model(site.camera_model_path(cam_id))


def register(reg) -> None:
    reg.register("capture_source", "m3_realsense", M3CaptureSource)
    reg.register("detector_backend", "m3_default", M3DetectorBackend)
    reg.register("frame_provider", "m4_default", M4FrameProvider)
    for name in ("depth+icp", "depth", "depth+icp+tilt", "depth+tilt"):
        reg.register("watchdog_layers", name, DepthIcpLayers)
    reg.register("calib_state_provider", "watchdog", WatchdogCalibState)
    reg.register("track_source", "live", LiveTrackSource)
    reg.register("sensor_model", "camera_model_file", CameraModelFile)
