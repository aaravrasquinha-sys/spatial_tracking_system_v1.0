"""
Every tunable M5 needs, in one place, same discipline as
poi_localization.config.M4Config. Loaded from JSON; see
configs/present.example.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"          # bind to the LAN interface's own IP in
                                     # production, not this -- see README's
                                     # "Security" section. 0.0.0.0 is the
                                     # dev-friendly default.
    port: int = 8765
    token: Optional[str] = None     # required in both the page URL
                                     # (?token=) and the WS URL for anything
                                     # beyond localhost. None only permitted
                                     # when host is 127.0.0.1/localhost.
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None
    static_dir: str = "web"
    max_clients: int = 8
    client_queue_size: int = 4      # bounded per-client outgoing queue;
                                     # tracks messages drop-oldest when full,
                                     # event messages are never dropped.
    slow_client_timeout_s: float = 5.0


@dataclass
class SourceConfig:
    kind: str = "synthetic"         # "live" | "replay" | "synthetic"
    replay_path: Optional[str] = None
    replay_speed: float = 1.0
    replay_loop: bool = True
    synthetic_mode: str = "scripted"  # "scripted" | "realistic"
    synthetic_n_walkers: int = 3
    synthetic_seed: int = 7
    synthetic_rect_m: tuple = (0.5, 4.5, -2.0, 2.0)  # x_min,x_max,y_min,y_max


@dataclass
class SceneConfig:
    cam_id: str = "cam0"
    frame: str = "local"            # "local" (Phase A) | "room" (Phase B)
    map_id: Optional[str] = None
    map_bundle_dir: Optional[str] = None  # maps/room_<id>/ once M1 exists
    walkable_grid_path: Optional[str] = None
    # Fallback intrinsics/camera pose to publish in /api/scene when no
    # live M4 pipeline is attached (e.g. running the dashboard purely
    # against a replay log or synthetic source that predates this run).
    fallback_intrinsics: dict = field(
        default_factory=lambda: {"fx": 606.75, "fy": 606.57, "cx": 320.19, "cy": 237.06, "width": 640, "height": 480}
    )


@dataclass
class HealthConfig:
    tick_hz: float = 1.0
    thermal_zone_globs: tuple = ("/sys/class/thermal/thermal_zone*/temp",)
    clock_skew_warn_s: float = 2.0


@dataclass
class PresentConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    camera_preview_enabled: bool = False
    camera_preview_fps: float = 5.0
    camera_preview_max_wh: tuple = (320, 240)
    event_log_max: int = 500        # cap on M4Pipeline.total_events-style
                                     # growth on the M5 side (see README's
                                     # "M4 finding" about unbounded growth)

    @classmethod
    def load(cls, path: str | Path) -> "PresentConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            server=ServerConfig(**raw.get("server", {})),
            source=SourceConfig(**raw.get("source", {})),
            scene=SceneConfig(**raw.get("scene", {})),
            health=HealthConfig(**raw.get("health", {})),
            camera_preview_enabled=raw.get("camera_preview_enabled", False),
            camera_preview_fps=raw.get("camera_preview_fps", 5.0),
            camera_preview_max_wh=tuple(raw.get("camera_preview_max_wh", (320, 240))),
            event_log_max=raw.get("event_log_max", 500),
        )

    def to_dict(self) -> dict:
        return asdict(self)
