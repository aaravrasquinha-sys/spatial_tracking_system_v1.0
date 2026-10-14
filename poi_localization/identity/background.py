"""G2 -- foreground mask for ONE person against the empty-room reference (R1 spec section 6.2).

A pixel inside the person's (expanded) bbox is foreground for this person when ALL hold:
  1. live depth is valid;
  2. it is in front of the empty room: reference is NaN, or d < ref - max(margin, 3*mad);
  3. it is near the person: |d - d_torso| <= person_band_m;
  4. it is not claimed by a nearer person (another bbox covers it, that person's torso depth is smaller,
     and |d - d_other| <= person_band_m).
The mask is then eroded once (drops the outer ring: hole-filling smears depth at silhouettes) and
sampled at `stride` to bound cost. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from poi_localization.identity.config import BackgroundConfig

BBox = Tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)


@dataclass
class ForegroundResult:
    vs: np.ndarray                      # full-resolution row indices of foreground samples
    us: np.ndarray                      # full-resolution column indices
    flags: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.vs.size)


def _erode4(m: np.ndarray) -> np.ndarray:
    """4-neighbour binary erosion; pixels on the array border are dropped."""
    out = m.copy()
    out[0, :] = out[-1, :] = False
    out[:, 0] = out[:, -1] = False
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    out[1:-1, 1:-1] = inner
    return out


def foreground_pixels(depth_m: np.ndarray, ref_depth: Optional[np.ndarray], bbox: BBox, d_torso: float,
                      others: Sequence[Tuple[BBox, float]] = (), cfg: Optional[BackgroundConfig] = None,
                      stride: int = 2, ref_mad: Optional[np.ndarray] = None,
                      expand: float = 0.05) -> ForegroundResult:
    cfg = cfg or BackgroundConfig()
    h, w = depth_m.shape
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, np.floor(x1 - expand * bw))), int(min(w, np.ceil(x2 + expand * bw)))
    y1, y2 = int(max(0, np.floor(y1 - expand * bh))), int(min(h, np.ceil(y2 + expand * bh)))
    flags: List[str] = []
    empty = ForegroundResult(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), flags)
    if x2 - x1 < 3 * stride or y2 - y1 < 3 * stride:
        return empty

    vs = np.arange(y1, y2, stride)
    us = np.arange(x1, x2, stride)
    V, U = np.meshgrid(vs, us, indexing="ij")
    d = depth_m[V, U].astype(np.float64)

    ok = np.isfinite(d) & (d > 0)                                           # rule 1
    if ref_depth is not None and ref_depth.shape == depth_m.shape:          # rule 2
        ref = ref_depth[V, U].astype(np.float64)
        mad = ref_mad[V, U].astype(np.float64) if ref_mad is not None and ref_mad.shape == depth_m.shape else 0.0
        thr = np.maximum(cfg.margin_m, 3.0 * np.nan_to_num(mad, nan=0.0))
        with np.errstate(invalid="ignore"):
            ok &= np.isnan(ref) | (d < ref - thr)
    else:
        flags.append("no_reference_background")
    with np.errstate(invalid="ignore"):
        ok &= np.abs(d - d_torso) <= cfg.person_band_m                      # rule 3
        for (ob, od) in others:                                             # rule 4
            if od < d_torso:
                ox1, oy1, ox2, oy2 = ob
                inside = (U >= ox1) & (U <= ox2) & (V >= oy1) & (V <= oy2)
                ok &= ~(inside & (np.abs(d - od) <= cfg.person_band_m))
    ok = _erode4(ok)
    iv, iu = np.nonzero(ok)
    return ForegroundResult(vs[iv].astype(np.int64), us[iu].astype(np.int64), flags)
