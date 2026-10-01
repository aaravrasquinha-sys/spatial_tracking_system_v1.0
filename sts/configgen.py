"""Generate the per-camera module configs from site.json (+ camera_model.<cam>.json).

The legacy modules keep their own config formats (M3Config / M4Config / PresentConfig /
AnchorConfig). This module builds those configs as plain dicts from defaults, applies
site-level values and the user's `overrides`, writes them under
`data/runtime/<site>/<cam>/`, and round-trips each through the module's OWN loader so a
typo in an override fails here with the module's error, not at 3 a.m. inside a daemon.

Note the legacy loaders are STRICT (`Cls(**raw)`): an unknown key raises TypeError.
That is why everything is validated by loading.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from sts.camera_model import load_camera_model
from sts.paths import resolve
from sts.site import SiteConfig, SiteError


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class GeneratedConfigs:
    cam_id: str
    out_dir: Path
    m3_path: Path
    m4_path: Path
    present_path: Path
    anchor_overrides: Dict[str, Any]
    map_id: Optional[str]
    phase: str

    def as_dict(self) -> dict:
        return {"cam_id": self.cam_id, "m3": str(self.m3_path), "m4": str(self.m4_path),
                "present": str(self.present_path), "anchor_overrides": self.anchor_overrides,
                "map_id": self.map_id, "phase": self.phase}


def build_m3_dict(site: SiteConfig, cam_id: str) -> dict:
    from poi_perception.config import M3Config
    cam = site.camera(cam_id)
    d = asdict(M3Config())
    d["camera"].update(cam_id=cam.cam_id, width=cam.width, height=cam.height, fps=cam.fps, serial=cam.serial,
                       expected_fx=cam.expected_fx, expected_fy=cam.expected_fy,
                       expected_cx=cam.expected_cx, expected_cy=cam.expected_cy)
    # The shipped example config and the runbook's export command both use a 640x480 engine input;
    # the dataclass default (640x640) would silently mismatch that engine. `sts doctor` compares
    # these against the engine manifest's img_size_hw.
    d["model"].update(backend=site.perception.backend, engine_path=str(site.engine_path()),
                      weights_path=site.perception.weights_path, input_width=cam.width, input_height=cam.height)
    if cam.masks_path:
        d["mask"]["masks_path"] = str(resolve(cam.masks_path))
    d["output"].update(jsonl_dir=str(site.logs_dir() / cam_id / "detections"),
                       clips_dir=str(site.logs_dir() / cam_id / "clips"))
    return deep_merge(d, site.perception.overrides)


def build_m4_dict(site: SiteConfig, cam_id: str, phase: Optional[str] = None,
                  map_id: Optional[str] = None) -> dict:
    from poi_localization.config import M4Config
    phase = phase or site.localization.phase
    d = asdict(M4Config())
    d["phase"] = phase
    d["output"]["jsonl_dir"] = str(site.logs_dir() / cam_id / "world_tracks")
    # one noise model for M2 and M4
    model = load_camera_model(site.camera_model_path(cam_id))
    d["depth_measurement"].update(model.m4_depth_overrides())
    if phase == "B":
        d["phase_b"].update(calibration_path=str(site.calibration_path(cam_id)),
                            expected_map_id=map_id or site.map.map_id,
                            walkable_grid_path=str(site.walkable_path(cam_id)))
    return deep_merge(d, site.localization.overrides)


def build_present_dict(site: SiteConfig, cam_id: str, phase: Optional[str] = None,
                       map_id: Optional[str] = None) -> dict:
    from poi_present.config import PresentConfig
    phase = phase or site.localization.phase
    cam = site.camera(cam_id)
    d = asdict(PresentConfig())
    d["server"].update(host=site.present.host, port=site.present.port, token=site.present.token,
                       tls_cert=site.present.tls_cert, tls_key=site.present.tls_key,
                       static_dir=str(resolve("web")))
    d["scene"].update(cam_id=cam_id, frame="room" if phase == "B" else "local", map_id=map_id,
                      fallback_intrinsics={"fx": cam.expected_fx, "fy": cam.expected_fy, "cx": cam.expected_cx,
                                           "cy": cam.expected_cy, "width": cam.width, "height": cam.height})
    if phase == "B":
        # The ANCHOR folder, never the map folder: maps/<site>/dense/points.ply is in M1's
        # W_cam0 frame (y-down, not gravity aligned); only anchors/<site>/viewer/points.ply is
        # in the room frame the tracks are published in.
        d["scene"].update(map_bundle_dir=str(site.anchor_dir(cam_id)), walkable_grid_path=str(site.walkable_path(cam_id)))
    # tuples survive JSON as lists; PresentConfig.load re-tuples what it needs
    return deep_merge(d, site.present.overrides)


def anchor_overrides(site: SiteConfig, cam_id: str) -> Dict[str, Any]:
    model = load_camera_model(site.camera_model_path(cam_id))
    return {**model.anchor_overrides(), **site.anchor.set}


def write_configs(site: SiteConfig, cam_id: str, phase: Optional[str] = None,
                  map_id: Optional[str] = None) -> GeneratedConfigs:
    from poi_localization.config import M4Config
    from poi_perception.config import M3Config
    from poi_present.config import PresentConfig

    phase = phase or site.localization.phase
    out = site.runtime_dir() / cam_id
    out.mkdir(parents=True, exist_ok=True)
    m3p, m4p, prp = out / "m3.json", out / "m4.json", out / "present.json"
    m3p.write_text(json.dumps(build_m3_dict(site, cam_id), indent=2))
    m4p.write_text(json.dumps(build_m4_dict(site, cam_id, phase, map_id), indent=2))
    prp.write_text(json.dumps(build_present_dict(site, cam_id, phase, map_id), indent=2))
    for path, loader, label in ((m3p, M3Config.load, "M3"), (m4p, M4Config.load, "M4"), (prp, PresentConfig.load, "M5")):
        try:
            loader(path)
        except TypeError as e:
            raise SiteError(f"generated {label} config {path} is rejected by the {label} loader ({e}). "
                            f"Check the `overrides` for that module in site.json: unknown keys are not allowed.") from e
    (out / "anchor_overrides.json").write_text(json.dumps(anchor_overrides(site, cam_id), indent=2))
    return GeneratedConfigs(cam_id, out, m3p, m4p, prp, anchor_overrides(site, cam_id), map_id, phase)
