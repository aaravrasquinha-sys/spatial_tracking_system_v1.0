"""G3 -- photometric normalisation against the empty room's colours (R1 spec section 6.3).

The anchor's `rgb_med` is a colour photo of the same pixels under calibration-time conditions, so the
background around a person is a colour calibration target. Per channel, gain g_c = median(live_c / ref_c)
over a ring of unoccluded stable background pixels, clamped; normalised colour = live / g. |log g| is
returned as a quality signal (large corrections -> lower observation weight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from poi_localization.identity.config import PhotometricConfig

BBox = Tuple[float, float, float, float]


@dataclass
class PhotoResult:
    gain: np.ndarray            # (3,) per-channel gain (live = ref * gain); ones when mode == "none"
    n_px: int
    mode: str                   # "ring" | "global" | "none"
    quality: float              # mean |log g| (0 = no correction needed)


def _gain_from_mask(rgb_live: np.ndarray, rgb_ref: np.ndarray, mask: np.ndarray, clamp) -> np.ndarray:
    live = rgb_live[mask].astype(np.float64)
    ref = rgb_ref[mask].astype(np.float64)
    g = np.median(live / np.maximum(ref, 1.0), axis=0)
    return np.clip(g, clamp[0], clamp[1])


def _valid_background(rgb_ref, depth_m, ref_depth, ref_stable, margin_m):
    mid = np.all((rgb_ref >= 20) & (rgb_ref <= 235), axis=-1)
    with np.errstate(invalid="ignore"):
        same = np.isfinite(depth_m) & np.isfinite(ref_depth) & (np.abs(depth_m - ref_depth) <= margin_m)
    return mid & same & ref_stable.astype(bool)


def estimate_global_gain(rgb_live, rgb_ref, depth_m, ref_depth, ref_stable, cfg: PhotometricConfig,
                         margin_m: float = 0.08, stride: int = 8) -> Optional[np.ndarray]:
    """Whole-frame gain from a fixed subsample of stable background pixels (call at ~1 Hz and cache)."""
    sub = (slice(None, None, stride), slice(None, None, stride))
    m = _valid_background(rgb_ref[sub], depth_m[sub], ref_depth[sub], ref_stable[sub], margin_m)
    if int(m.sum()) < 50:
        return None
    return _gain_from_mask(rgb_live[sub], rgb_ref[sub], m, cfg.gain_clamp)


def estimate_gain(rgb_live: np.ndarray, rgb_ref: np.ndarray, depth_m: np.ndarray, ref_depth: np.ndarray,
                  ref_stable: np.ndarray, bbox: BBox, cfg: Optional[PhotometricConfig] = None,
                  margin_m: float = 0.08, global_gain: Optional[np.ndarray] = None, stride: int = 2) -> PhotoResult:
    cfg = cfg or PhotometricConfig()
    ones = np.ones(3)
    if not cfg.enabled or rgb_ref.shape != rgb_live.shape:
        return PhotoResult(ones, 0, "none", 0.0)
    h, w = depth_m.shape
    x1, y1, x2, y2 = bbox
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    X1, X2 = int(max(0, cx - hw * cfg.ring_scale)), int(min(w, cx + hw * cfg.ring_scale))
    Y1, Y2 = int(max(0, cy - hh * cfg.ring_scale)), int(min(h, cy + hh * cfg.ring_scale))
    sl = (slice(Y1, Y2, stride), slice(X1, X2, stride))
    V, U = np.mgrid[Y1:Y2:stride, X1:X2:stride]
    ring = ~((U >= x1) & (U <= x2) & (V >= y1) & (V <= y2))
    ok = ring & _valid_background(rgb_ref[sl], depth_m[sl], ref_depth[sl], ref_stable[sl], margin_m)
    if int(ok.sum()) >= cfg.min_ring_px:
        g = _gain_from_mask(rgb_live[sl], rgb_ref[sl], ok, cfg.gain_clamp)
        return PhotoResult(g, int(ok.sum()), "ring", float(np.mean(np.abs(np.log(g)))))
    if global_gain is not None:
        g = np.clip(np.asarray(global_gain, dtype=np.float64), cfg.gain_clamp[0], cfg.gain_clamp[1])
        return PhotoResult(g, 0, "global", float(np.mean(np.abs(np.log(g)))))
    return PhotoResult(ones, int(ok.sum()), "none", 0.0)


def normalise(pixels: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """(N, 3) colours -> colours as they would look under calibration-time conditions (float32, 0..255)."""
    return np.clip(pixels.astype(np.float32) / np.asarray(gain, dtype=np.float32), 0.0, 255.0)
