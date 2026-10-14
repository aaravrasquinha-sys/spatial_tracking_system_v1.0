"""C1 -- height-banded, stature-normalised appearance (R1 spec section 7.4).

Bands are fractions of the person's stature, so the same band is the same body part for everyone.
Per band: after photometric normalisation, sampled foreground pixels -> HSV; chromatic pixels (S >= 0.18 and
V >= 0.12) go into a hue x saturation histogram (12 x 4 = 48 bins, soft/linear on hue), achromatic pixels into
4 value bins (black / dark grey / light grey / white): 52 bins total, L1-normalised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from poi_localization.identity.config import HistConfig
from poi_localization.identity.visibility import VISIBLE

UPPER_BANDS = (4, 5, 6, 7)       # abdomen, chest, neck/shoulders, head/hair   (default band edges, section 7.4)
LOWER_BANDS = (0, 1, 2, 3)       # shoes, shins, thighs, hips
MIN_BAND_PIXELS = 40


def n_bins(h: HistConfig) -> int:
    return h.hue_bins * h.sat_bins + h.achrom_bins


def rgb_to_hsv(rgb01: np.ndarray) -> np.ndarray:
    """(N, 3) float RGB in [0, 1] -> (N, 3) HSV with H in [0, 1)."""
    r, g, b = rgb01[:, 0], rgb01[:, 1], rgb01[:, 2]
    mx, mn = rgb01.max(axis=1), rgb01.min(axis=1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-9
    rc = nz & (mx == r)
    gc = nz & (mx == g) & ~rc
    bc = nz & ~rc & ~gc
    h[rc] = ((g[rc] - b[rc]) / d[rc]) % 6.0
    h[gc] = (b[gc] - r[gc]) / d[gc] + 2.0
    h[bc] = (r[bc] - g[bc]) / d[bc] + 4.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h / 6.0, s, mx], axis=1)


def histogram52(rgb255: np.ndarray, hc: HistConfig) -> np.ndarray:
    """(N, 3) colours 0..255 -> L1-normalised histogram (52 bins by default). Empty input -> zeros."""
    nb = n_bins(hc)
    if rgb255.shape[0] == 0:
        return np.zeros(nb, dtype=np.float32)
    hsv = rgb_to_hsv(np.clip(rgb255.astype(np.float64) / 255.0, 0.0, 1.0))
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    chroma = (s >= hc.sat_achrom_thresh) & (v >= hc.v_dark_thresh)
    hist = np.zeros(nb, dtype=np.float64)
    if chroma.any():
        hh = h[chroma] * hc.hue_bins
        lo = np.floor(hh).astype(int) % hc.hue_bins
        hi = (lo + 1) % hc.hue_bins
        frac = hh - np.floor(hh)
        sb = np.clip(((s[chroma] - hc.sat_achrom_thresh) / (1 - hc.sat_achrom_thresh) * hc.sat_bins).astype(int),
                     0, hc.sat_bins - 1)
        hist[:hc.hue_bins * hc.sat_bins] += np.bincount(lo * hc.sat_bins + sb, weights=1 - frac,
                                                        minlength=hc.hue_bins * hc.sat_bins)
        hist[:hc.hue_bins * hc.sat_bins] += np.bincount(hi * hc.sat_bins + sb, weights=frac,
                                                        minlength=hc.hue_bins * hc.sat_bins)
    ach = ~chroma
    if ach.any():
        vb = np.clip((v[ach] * hc.achrom_bins).astype(int), 0, hc.achrom_bins - 1)
        hist[hc.hue_bins * hc.sat_bins:] += np.bincount(vb, minlength=hc.achrom_bins)
    tot = hist.sum()
    return (hist / tot).astype(np.float32) if tot > 0 else hist.astype(np.float32)


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya coefficient in [0, 1] (1 = identical distributions)."""
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))


@dataclass
class AppearanceSample:
    view: Optional[str]                                   # "front" | "back" | "side" | None
    hists: Dict[int, np.ndarray] = field(default_factory=dict)     # band index -> histogram
    mass: Dict[int, int] = field(default_factory=dict)             # band index -> sampled pixel count
    quality: float = 1.0                                           # lowered by large photometric corrections


def extract_appearance(colours_norm: np.ndarray, heights_m: np.ndarray, stature_m: float, edges: Sequence[float],
                       band_states: Sequence[str], hc: HistConfig, view: Optional[str] = None,
                       photo_quality: float = 0.0) -> AppearanceSample:
    """`colours_norm`: (N, 3) photometrically normalised foreground colours (0..255) at pixels whose room
    height is `heights_m` (N,). Only bands whose visibility state is VISIBLE are accumulated."""
    frac = heights_m / max(stature_m, 1e-6)
    idx = np.clip(np.digitize(frac, edges) - 1, -1, len(edges) - 1)
    out = AppearanceSample(view=view, quality=float(np.exp(-2.0 * photo_quality)))
    for b in range(len(edges) - 1):
        if band_states[b] != VISIBLE:
            continue
        m = idx == b
        n = int(m.sum())
        if n < MIN_BAND_PIXELS:
            continue
        out.hists[b] = histogram52(colours_norm[m], hc)
        out.mass[b] = n
    return out
