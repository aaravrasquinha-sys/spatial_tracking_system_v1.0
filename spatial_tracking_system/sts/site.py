"""site.json -- the ONE file you edit per installation.

Every field has a default (rule 3 in docs/ARCHITECTURE.md): an old site.json
never breaks after an upgrade, and unknown keys are preserved, not rejected,
so a newer file still loads on an older checkout.

Cameras are a LIST from day one. The run chain currently supports exactly one
enabled camera (multi-camera fusion is a reserved slot, see docs/OPEN_ITEMS.md)
but every path, config and log is keyed per camera so adding a second is a data
change, not a restructure.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from sts.paths import REPO_ROOT, resolve

SITE_SCHEMA_VERSION = 1


@dataclass
class CameraSpec:
    cam_id: str = "cam0"
    enabled: bool = True
    serial: Optional[str] = None            # pin to one physical D435i; null = first device
    width: int = 640
    height: int = 480
    fps: int = 30
    # Intrinsics you expect this unit to report (env_probe prints the real ones).
    expected_fx: float = 606.75
    expected_fy: float = 606.57
    expected_cx: float = 320.19
    expected_cy: float = 237.06
    intrinsics_tol_px: float = 2.0          # `sts doctor` fails if the device differs by more than this
    masks_path: Optional[str] = None        # ROI masks json (M3); null = none


@dataclass
class MapSpec:
    dir: Optional[str] = None               # default: <data>/maps/<site>
    map_id: Optional[str] = None            # filled in by `sts map-lock` / `sts anchor accept`


@dataclass
class AnchorSpec:
    dir: Optional[str] = None               # base dir; default <data>/anchors/<site>; bundle = <base>/<cam_id>
    sessions: int = 3
    set: Dict[str, Any] = field(default_factory=dict)   # AnchorConfig overrides (--set KEY=VAL)


@dataclass
class PerceptionSpec:
    backend: str = "trt"                    # "trt" | "ultralytics" (export/dev only) -- runtime must be trt
    engine_path: Optional[str] = None       # default: <data>/models/engines/yolo11n_pose_fp16.engine
    weights_path: str = "yolo11n-pose.pt"
    overrides: Dict[str, Any] = field(default_factory=dict)   # deep-merged into the M3Config dict


@dataclass
class LocalizationSpec:
    phase: str = "B"                        # "A" (provisional floor frame) | "B" (registered room frame)
    overrides: Dict[str, Any] = field(default_factory=dict)   # deep-merged into the M4Config dict


@dataclass
class PresentSpec:
    host: str = "127.0.0.1"                 # loopback by default; set your LAN IP + a token to go live
    port: int = 8765
    token: Optional[str] = None
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)   # deep-merged into the PresentConfig dict


@dataclass
class WatchdogSpec:
    enabled: bool = True
    depth_check_period_s: float = 120.0     # cheap layer cadence
    depth_window_frames: int = 15           # temporal-median window for each depth check
    recheck_period_s: float = 900.0         # expensive ICP layer cadence (only when the scene is idle)
    recheck_idle_s: float = 10.0            # ... and nobody was tracked for this long
    recheck_frames: int = 30                # frames pooled for the ICP recheck
    tilt_layer: str = "off"                 # "off" | "external" (a TiltSource injected by code). "auto" is reserved
    on_suspect: str = "suppress"            # "suppress": publish empty tracks | "flag": keep publishing, health says suspect
    startup_check: bool = True              # run one depth check before going live


@dataclass
class RetentionSpec:
    track_log_days: int = 14                # world-track + detection JSONL are a record of people's movements
    clip_days: int = 30
    bag_days: int = 7


@dataclass
class SiteConfig:
    schema: int = SITE_SCHEMA_VERSION
    site: str = "site_001"
    data_dir: str = "data"                  # relative to repo root, or absolute
    cameras: List[CameraSpec] = field(default_factory=lambda: [CameraSpec()])
    map: MapSpec = field(default_factory=MapSpec)
    anchor: AnchorSpec = field(default_factory=AnchorSpec)
    perception: PerceptionSpec = field(default_factory=PerceptionSpec)
    localization: LocalizationSpec = field(default_factory=LocalizationSpec)
    present: PresentSpec = field(default_factory=PresentSpec)
    watchdog: WatchdogSpec = field(default_factory=WatchdogSpec)
    retention: RetentionSpec = field(default_factory=RetentionSpec)
    capture_profile: Optional[str] = None   # default: configs/capture_profile.mapping.json
    slots: Dict[str, str] = field(default_factory=dict)   # slot name -> implementation name (see sts/slots.py)
    extra: Dict[str, Any] = field(default_factory=dict)   # unknown top-level keys survive a load/save round-trip

    # ---- derived paths -------------------------------------------------
    def data_root(self) -> Path:
        return resolve(self.data_dir)

    def map_dir(self) -> Path:
        return resolve(self.map.dir) if self.map.dir else self.data_root() / "maps" / self.site

    def anchor_base(self) -> Path:
        return resolve(self.anchor.dir) if self.anchor.dir else self.data_root() / "anchors" / self.site

    def anchor_dir(self, cam_id: str = "cam0") -> Path:
        """One anchor bundle PER CAMERA: M2 writes a capture, overlays and a reference
        depth into it, which would collide between cameras if the directory were shared."""
        return self.anchor_base() / cam_id

    def captures_dir(self) -> Path:
        return self.data_root() / "captures" / self.site

    def recordings_dir(self) -> Path:
        return self.data_root() / "recordings" / self.site

    def logs_dir(self) -> Path:
        return self.data_root() / "logs" / self.site

    def runtime_dir(self) -> Path:
        """Generated per-camera configs + run manifests live here."""
        return self.data_root() / "runtime" / self.site

    def engine_path(self) -> Path:
        if self.perception.engine_path:
            return resolve(self.perception.engine_path)
        return self.data_root() / "models" / "engines" / "yolo11n_pose_fp16.engine"

    def capture_profile_path(self) -> Path:
        from sts.paths import default_capture_profile
        return resolve(self.capture_profile) if self.capture_profile else default_capture_profile()

    def enabled_cameras(self) -> List[CameraSpec]:
        return [c for c in self.cameras if c.enabled]

    def camera(self, cam_id: Optional[str] = None) -> CameraSpec:
        cams = self.enabled_cameras()
        if cam_id is None:
            if len(cams) != 1:
                raise SiteError(f"{len(cams)} cameras enabled; pass --cam to choose one "
                                f"(the live run chain supports exactly one camera today)")
            return cams[0]
        for c in self.cameras:
            if c.cam_id == cam_id:
                return c
        raise SiteError(f"no camera {cam_id!r} in site.json (have: {[c.cam_id for c in self.cameras]})")

    def calibration_path(self, cam_id: str) -> Path:
        return self.anchor_dir(cam_id) / f"calibration.{cam_id}.json"

    def reference_depth_path(self, cam_id: str) -> Path:
        return self.anchor_dir(cam_id) / f"{cam_id}_reference_depth.npz"

    def walkable_path(self, cam_id: str = "cam0") -> Path:
        return self.anchor_dir(cam_id) / "walkable.json"

    def camera_model_path(self, cam_id: str) -> Path:
        return resolve(f"configs/camera_model.{cam_id}.json")

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        return d

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "SiteConfig":
        p = Path(path)
        if not p.exists():
            raise SiteError(f"site config not found: {p}  (create it with `python -m sts init`)")
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise SiteError(f"{p} is not valid JSON: {e}") from e
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "SiteConfig":
        raw = dict(raw)
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for k in list(raw):
            if k in known and k != "extra":
                kwargs[k] = raw.pop(k)
        kwargs["extra"] = raw                          # everything unknown is preserved
        sub = {"map": MapSpec, "anchor": AnchorSpec, "perception": PerceptionSpec,
               "localization": LocalizationSpec, "present": PresentSpec,
               "watchdog": WatchdogSpec, "retention": RetentionSpec}
        for name, klass in sub.items():
            if name in kwargs and isinstance(kwargs[name], dict):
                kwargs[name] = _build(klass, kwargs[name])
        if "cameras" in kwargs:
            kwargs["cameras"] = [_build(CameraSpec, c) if isinstance(c, dict) else c for c in kwargs["cameras"]]
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.site or any(ch in self.site for ch in "/\\ "):
            raise SiteError(f"site name {self.site!r} must be a non-empty string without slashes or spaces")
        ids = [c.cam_id for c in self.cameras]
        if len(set(ids)) != len(ids):
            raise SiteError(f"duplicate cam_id in cameras: {ids}")
        if not self.cameras:
            raise SiteError("cameras must list at least one camera")
        if self.localization.phase not in ("A", "B"):
            raise SiteError(f"localization.phase must be 'A' or 'B', got {self.localization.phase!r}")
        if self.watchdog.on_suspect not in ("suppress", "flag"):
            raise SiteError("watchdog.on_suspect must be 'suppress' or 'flag'")
        if self.watchdog.tilt_layer not in ("off", "external"):
            raise SiteError("watchdog.tilt_layer must be 'off' or 'external' ('auto' is reserved: an accel side-pipeline under RSUSB is unvalidated)")
        if self.perception.backend not in ("trt", "ultralytics"):
            raise SiteError("perception.backend must be 'trt' or 'ultralytics'")


class SiteError(RuntimeError):
    pass


def _build(klass, raw: dict):
    """Construct a dataclass from a dict, ignoring unknown keys (forward-compat) and
    leaving omitted keys at their defaults (rule 3)."""
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in raw.items() if k in names})


def default_site_path() -> Path:
    return REPO_ROOT / "configs" / "site.json"
