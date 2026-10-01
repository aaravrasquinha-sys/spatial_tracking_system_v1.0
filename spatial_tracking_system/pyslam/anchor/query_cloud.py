"""
Static capture -> weighted metric point cloud in the CAMERA frame.

Filters, in order (each one exists because a specific real D435i failure mode needs it):
  * valid in >= min_valid_frac of frames       (dropout / dark or IR-absorbing surfaces)
  * temporal MAD small (grows ~z^2)             (shimmering glass, screens, foliage, fans)
  * range in [range_min, range_max]             (below 0.3 m no depth; error ~z^2 beyond ~4 m)
  * not a depth edge / next to a hole           (flying pixels at silhouettes -> phantom geometry)
Weights are 1/(sigma_z^2 + sigma_map^2): the same noise-model SHAPE M4 uses
(sigma_z = z^2 * sigma_d / (f*B)), times an empirical multiplier.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from pyslam.anchor.config import AnchorConfig


def depth_sigma(z: np.ndarray, intr: dict, cfg: AnchorConfig) -> np.ndarray:
    f = 0.5 * (intr["fx"] + intr["fy"])
    b = intr.get("baseline", 0.05)
    return cfg.noise_multiplier * (z ** 2) * cfg.sigma_disparity_px / (f * b)


def stable_mask(depth: np.ndarray, mad: Optional[np.ndarray], valid_frac: Optional[np.ndarray],
                cfg: AnchorConfig) -> np.ndarray:
    ok = np.isfinite(depth) & (depth >= cfg.range_min_m) & (depth <= cfg.range_max_m)
    if valid_frac is not None:
        ok &= valid_frac >= cfg.min_valid_frac
    if mad is not None:
        ok &= np.nan_to_num(mad, nan=1e9) <= cfg.max_mad_m * np.maximum(depth, 1.0) ** 2
    # edge / hole-adjacent rejection on the (finite) median depth
    fin = np.isfinite(depth)
    mx = maximum_filter(np.where(fin, depth, -np.inf), size=3)
    mn = minimum_filter(np.where(fin, depth, np.inf), size=3)
    all_fin = minimum_filter(fin.astype(np.uint8), size=3).astype(bool)      # no invalid neighbour
    jump = np.where(all_fin, mx - mn, np.inf)
    ok &= jump <= (cfg.edge_jump_m + cfg.edge_jump_per_z * np.nan_to_num(depth, nan=0.0))
    return ok


def deproject(depth: np.ndarray, mask: np.ndarray, intr: dict, stride: int = 1):
    """-> pts (N,3) cam frame, pix (N,2) int (row, col). Depth is z, not range."""
    h, w = depth.shape
    m = np.zeros_like(mask)
    m[::stride, ::stride] = mask[::stride, ::stride]
    rows, cols = np.nonzero(m)
    z = depth[rows, cols].astype(np.float64)
    x = (cols - intr["cx"]) / intr["fx"] * z
    y = (rows - intr["cy"]) / intr["fy"] * z
    return np.stack([x, y, z], axis=1), np.stack([rows, cols], axis=1)


def capture_to_cloud(cap, cfg: Optional[AnchorConfig] = None, depth: Optional[np.ndarray] = None,
                     stride: Optional[int] = None, use_stability: bool = True):
    """-> dict(pts, weights, colors, pix, sigma). `depth` overrides cap.depth_med (used for sub-medians,
    where per-subset MAD isn't available, so the FULL capture's stability mask is reused)."""
    cfg = cfg or AnchorConfig()
    d = cap.depth_med if depth is None else depth
    mask = stable_mask(d, cap.depth_mad if use_stability else None,
                       cap.valid_frac if use_stability else None, cfg)
    if depth is not None and use_stability:
        # sub-median: it must itself be finite where we sample, and inherit the stability mask
        mask &= stable_mask(cap.depth_med, cap.depth_mad, cap.valid_frac, cfg)
    pts, pix = deproject(d, mask, cap.intr, stride or cfg.query_stride)
    z = pts[:, 2]
    sig = depth_sigma(z, cap.intr, cfg)
    w = 1.0 / (sig ** 2 + cfg.map_sigma_m ** 2)
    w = w / w.mean() if w.size else w
    colors = cap.rgb_med[pix[:, 0], pix[:, 1]] if cap.rgb_med is not None and cap.rgb_med.shape[:2] == d.shape else None
    return {"pts": pts, "weights": w, "colors": colors, "pix": pix, "sigma": sig}
