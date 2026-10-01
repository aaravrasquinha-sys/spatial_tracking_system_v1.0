"""camera_model.<cam>.json -- the single source of truth for the D435i depth-noise model.

Before this file existed, M2 (`AnchorConfig.noise_multiplier / sigma_disparity_px`)
and M4 (`DepthMeasurementConfig.depth_stereo_fx_px / sigma_disparity_px /
depth_bias_frac`) each held their own GUESS of the same physical quantity. When
you later fit the noise model on a flat wall (the plan's "fit empirically in
your room"), you edit ONE file and `sts` pushes the values into both modules'
generated configs. No module code changes.

Every field defaults to null = "leave that module's own default alone", so
adding this file changes nothing until you put a number in it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CameraModel:
    sigma_disparity_px: Optional[float] = None      # both M2 and M4
    depth_stereo_fx_px: Optional[float] = None      # M4 (IR-pair focal length)
    depth_bias_frac: Optional[float] = None         # M4
    lateral_px_noise: Optional[float] = None        # M4
    anchor_noise_multiplier: Optional[float] = None  # M2 (scales its color-fx-based sigma)
    fitted_on: Optional[str] = None                 # free text: when/where/how this was fitted
    notes: str = ""

    def m4_depth_overrides(self) -> Dict[str, Any]:
        out = {}
        for k in ("sigma_disparity_px", "depth_stereo_fx_px", "depth_bias_frac", "lateral_px_noise"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out

    def anchor_overrides(self) -> Dict[str, Any]:
        out = {}
        if self.sigma_disparity_px is not None:
            out["sigma_disparity_px"] = self.sigma_disparity_px
        if self.anchor_noise_multiplier is not None:
            out["noise_multiplier"] = self.anchor_noise_multiplier
        return out


def load_camera_model(path: str | Path) -> CameraModel:
    p = Path(path)
    if not p.exists():
        return CameraModel()
    raw = json.loads(p.read_text())
    names = {f.name for f in fields(CameraModel)}
    return CameraModel(**{k: v for k, v in raw.items() if k in names})


def save_camera_model(model: CameraModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(model), indent=2) + "\n")
