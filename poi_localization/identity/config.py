"""IdentityConfig: every field defaulted, identity disabled by default (R1 spec section 13)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BackgroundConfig:
    margin_m: float = 0.08
    person_band_m: float = 0.6


@dataclass
class PhotometricConfig:
    enabled: bool = True
    ring_scale: float = 1.3
    min_ring_px: int = 300
    gain_clamp: List[float] = field(default_factory=lambda: [0.5, 2.0])


@dataclass
class HistConfig:
    hue_bins: int = 12
    sat_bins: int = 4
    achrom_bins: int = 4
    sat_achrom_thresh: float = 0.18
    v_dark_thresh: float = 0.12


@dataclass
class SwapMonitorConfig:
    enabled: bool = True
    swap_correction: bool = False


@dataclass
class WhereConfig:
    enabled: bool = False
    v_max_mps: float = 2.5
    cell_m: float = 0.2
    hz: float = 5.0
    p_det: float = 0.9


@dataclass
class TextureConfig:
    enabled: bool = False


@dataclass
class GaitConfig:
    enabled: bool = False


@dataclass
class LogConfig:
    enabled: bool = True


_NESTED = {
    "background": BackgroundConfig,
    "photometric": PhotometricConfig,
    "hist": HistConfig,
    "swap_monitor": SwapMonitorConfig,
    "where": WhereConfig,
    "texture": TextureConfig,
    "gait": GaitConfig,
    "log": LogConfig,
}


@dataclass
class IdentityConfig:
    enabled: bool = False                       # master switch (slot identity_resolver must also be multicue_v1)
    reference_depth_path: str = ""              # filled by sts/configgen in Phase B (later phase)
    static_capture_path: str = ""
    priors_path: str = "configs/identity_priors.default.json"
    sample_hz_per_track: float = 5.0
    max_extractions_per_frame: int = 2
    budget_ms_p95: float = 2.0
    pixel_stride: int = 2
    min_range_m: float = 1.0
    max_range_m: float = 4.5
    kp_conf_min: float = 0.5
    upright_max_tilt_deg: float = 15.0
    knee_min_angle_deg: float = 155.0
    isolation_iou_max: float = 0.05
    isolation_min_dist_m: float = 0.6
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)
    bands: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.28, 0.48, 0.55, 0.70, 0.82, 0.88, 1.0])
    hist: HistConfig = field(default_factory=HistConfig)
    appearance_ema_halflife_s: float = 10.0
    max_outfits: int = 3
    struct_max_samples_per_s: float = 3.0
    soft_feature_extra_sd_m: float = 0.03
    kp_pixel_sigma_px: float = 3.0
    prior_strength_n0: int = 30
    llr_clip: float = 6.0
    rho_upper_change: float = 0.10
    rho_lower_change: float = 0.03
    pi_new: float = 0.2
    p_commit: float = 0.97
    p_commit_new: float = 0.95
    min_margin: float = 0.5
    n_min_obs: int = 5
    t_min_s: float = 0.5
    t_max_resolve_s: float = 6.0
    missing_ttl_s: float = 45.0
    swap_monitor: SwapMonitorConfig = field(default_factory=SwapMonitorConfig)
    where: WhereConfig = field(default_factory=WhereConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    persist_gallery: bool = False
    log: LogConfig = field(default_factory=LogConfig)
    portals: List[list] = field(default_factory=list)   # optional room-frame polygons (used by the where cue)

    @classmethod
    def from_dict(cls, raw) -> "IdentityConfig":
        """Strict like the other M4 sections: an unknown key raises TypeError."""
        raw = dict(raw or {})
        kw = {}
        for name, sub in _NESTED.items():
            if name in raw:
                kw[name] = sub(**raw.pop(name))
        return cls(**raw, **kw)
