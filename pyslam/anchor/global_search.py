"""
Stage 3: global search for (x, y, yaw) -- no initial guess needed.

After the physics prior levels the query, only a planar rigid motion is unknown, so we can search it
EXHAUSTIVELY and deterministically (correlative scan matching, the idea Cartographer uses for loop
closure) instead of hoping a feature matcher gets lucky:

  map   : room-frame points in the structure slice [slice_lo, slice_hi] above the floor
          -> 2D occupancy at csm_res -> distance transform -> likelihood field L = exp(-d^2 / 2 sigma^2)
  query : levelled camera points in the same height slice -> xy
  score(yaw, t) = mean_p L(R_yaw p + t)  for EVERY t at once, via one FFT cross-correlation per yaw.

The result is a score landscape, not one answer, so AMBIGUITY is a measurable quantity: the ratio of the
best peak to the best DISTINCT second peak. A symmetric room shows up here as a tie instead of as a
confidently wrong calibration.

Slice/height details that matter: the query's floor is at z = -h in L, so 'height above floor' = z_L + h.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import distance_transform_edt, maximum_filter, map_coordinates

from pyslam.anchor import geo
from pyslam.anchor.config import AnchorConfig
from pyslam.core.log import get_logger

log = get_logger("anchor.global_search")


@dataclass
class MapRaster:
    L: np.ndarray            # (H,W) likelihood field, padded
    origin: np.ndarray       # world xy of the min corner of cell (0,0)
    res: float
    bbox_cells: Tuple[int, int, int, int]   # occupied-cell bbox (r0, r1, c0, c1) inside the padded grid
    n_occupied: int


@dataclass
class Candidate:
    x: float
    y: float
    yaw: float               # rad
    score: float
    refined_score: Optional[float] = None


def build_map_raster(pts_room: np.ndarray, cfg: AnchorConfig, pad_m: Optional[float] = None) -> MapRaster:
    r = cfg.csm_res_m
    pad = (cfg.csm_range_m + 0.5) if pad_m is None else pad_m
    z = pts_room[:, 2]
    m = (z >= cfg.slice_lo_m) & (z <= cfg.slice_hi_m)
    p = pts_room[m, :2]
    if p.shape[0] < 500:
        raise ValueError("structure slice of the map has < 500 points: nothing to match against")
    lo = p.min(axis=0) - pad
    hi = p.max(axis=0) + pad
    lo = np.floor(lo / r) * r
    W = int(np.ceil((hi[0] - lo[0]) / r)) + 1
    H = int(np.ceil((hi[1] - lo[1]) / r)) + 1
    cx = np.floor((p[:, 0] - lo[0]) / r).astype(int)
    cy = np.floor((p[:, 1] - lo[1]) / r).astype(int)
    cnt = np.bincount(cy * W + cx, minlength=H * W).reshape(H, W)
    occ = cnt >= cfg.csm_min_map_pts_per_cell
    dist = distance_transform_edt(~occ) * r
    L = np.exp(-0.5 * (dist / cfg.csm_sigma_m) ** 2).astype(np.float32)
    rr, cc = np.nonzero(occ)
    return MapRaster(L=L, origin=lo, res=r, bbox_cells=(int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())),
                     n_occupied=int(occ.sum()))


def query_slice(pts_L: np.ndarray, h: float, cfg: AnchorConfig, weights: Optional[np.ndarray] = None):
    """Levelled-camera points -> xy of the structure slice (uniformly thinned in 3D so nearby surfaces don't dominate)."""
    hz = pts_L[:, 2] + h
    rng_xy = np.linalg.norm(pts_L[:, :2], axis=1)
    m = (hz >= cfg.slice_lo_m) & (hz <= cfg.slice_hi_m) & (rng_xy <= cfg.csm_range_m - 0.1)
    p = pts_L[m]
    w = weights[m] if weights is not None else np.ones(p.shape[0])
    if p.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0)
    keys = np.floor(p / 0.04).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return p[first, :2], w[first]


def csm_search(raster: MapRaster, q_xy: np.ndarray, q_w: np.ndarray, cfg: AnchorConfig,
               hint: Optional[Tuple[float, float, float, float]] = None) -> List[Candidate]:
    """Exhaustive (x,y,yaw) search. `hint` = (x, y, radius_m, yaw_rad) restricts camera position to a disc and
    yaw to +-30 deg around the hint (operator-assisted disambiguation)."""
    if q_xy.shape[0] < 50:
        raise ValueError(f"only {q_xy.shape[0]} query points in the structure slice: not enough visible structure "
                         f"(camera sees only floor/ceiling?)")
    res = raster.res
    half = int(np.ceil(cfg.csm_range_m / res))
    nq = 2 * half + 1
    Hm, Wm = raster.L.shape
    F_M = sfft.rfft2(raster.L.astype(np.float64), workers=-1)
    r0, r1, c0, c1 = raster.bbox_cells
    # valid camera cells: (ty + half, tx + half) inside the occupied bbox (and inside the FFT valid region)
    tyv = np.arange(Hm - nq + 1) + half
    txv = np.arange(Wm - nq + 1) + half
    ymask = (tyv >= r0) & (tyv <= r1)
    xmask = (txv >= c0) & (txv <= c1)
    valid = ymask[:, None] & xmask[None, :]
    if hint is not None:
        hx, hy, hr, hyaw = hint
        ccx = raster.origin[0] + (txv + 0.5) * res
        ccy = raster.origin[1] + (tyv + 0.5) * res
        valid &= ((ccx[None, :] - hx) ** 2 + (ccy[:, None] - hy) ** 2) <= hr ** 2
    yaws = np.arange(0.0, 360.0, cfg.csm_yaw_step_deg)
    if hint is not None:
        d = np.abs(((yaws - np.degrees(hint[3])) + 180) % 360 - 180)
        yaws = yaws[d <= 30.0]
    wsum = float(q_w.sum())
    cap = 4.0 * float(np.mean(q_w))
    cands: List[Candidate] = []
    for yd in yaws:
        yaw = np.radians(yd)
        c, s = np.cos(yaw), np.sin(yaw)
        xr = q_xy[:, 0] * c - q_xy[:, 1] * s
        yr = q_xy[:, 0] * s + q_xy[:, 1] * c
        ix = np.rint(xr / res).astype(int) + half
        iy = np.rint(yr / res).astype(int) + half
        ok = (ix >= 0) & (ix < nq) & (iy >= 0) & (iy < nq)
        Q = np.bincount(iy[ok] * nq + ix[ok], weights=q_w[ok], minlength=nq * nq).reshape(nq, nq)
        np.minimum(Q, cap, out=Q)                        # a tall wall column must not out-vote everything else
        corr = sfft.irfft2(F_M * np.conj(sfft.rfft2(Q, s=(Hm, Wm), workers=-1)), s=(Hm, Wm), workers=-1)
        corr = corr[: Hm - nq + 1, : Wm - nq + 1] / max(Q.sum(), 1e-9)
        corr = np.where(valid, corr, -1.0)
        pk = (corr == maximum_filter(corr, size=9)) & (corr > 0)
        ys, xs = np.nonzero(pk)
        if ys.size == 0:
            continue
        vals = corr[ys, xs]
        top = np.argsort(-vals)[: cfg.csm_peaks_per_yaw]
        for k in top:
            cands.append(Candidate(x=float(raster.origin[0] + (xs[k] + half + 0.5) * res),
                                   y=float(raster.origin[1] + (ys[k] + half + 0.5) * res),
                                   yaw=float(yaw), score=float(vals[k])))
    return cands


def distinct_peaks(cands: List[Candidate], cfg: AnchorConfig, k: Optional[int] = None) -> List[Candidate]:
    """Greedy non-maximum suppression across yaw and xy."""
    k = k or cfg.csm_top_k
    out: List[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.score):
        far = True
        for o in out:
            dxy = np.hypot(c.x - o.x, c.y - o.y)
            dyaw = abs(np.degrees(geo.wrap_pi(c.yaw - o.yaw)))
            if dxy < cfg.csm_distinct_xy_m and dyaw < cfg.csm_distinct_yaw_deg:
                far = False
                break
        if far:
            out.append(c)
            if len(out) >= k:
                break
    return out


def fine_config(cfg: AnchorConfig) -> AnchorConfig:
    """Sharper likelihood field for the local refinement stage (coarse field is deliberately blurry so the
    exhaustive search tolerates quantisation; the fine field is what discriminates within a coarse plateau)."""
    import dataclasses
    return dataclasses.replace(cfg, csm_res_m=cfg.csm_res_m / 2.0, csm_sigma_m=cfg.csm_sigma_m / 2.0)


def _eval_poses(raster: MapRaster, q_xy, w, cx, cy, yaw, offs_x, offs_y):
    """score grid over (offs_y, offs_x) for one yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    xr = q_xy[:, 0] * c - q_xy[:, 1] * s
    yr = q_xy[:, 0] * s + q_xy[:, 1] * c
    DX, DY = np.meshgrid(offs_x, offs_y)
    px = (cx + DX.reshape(-1, 1) + xr[None] - raster.origin[0]) / raster.res - 0.5
    py = (cy + DY.reshape(-1, 1) + yr[None] - raster.origin[1]) / raster.res - 0.5
    v = map_coordinates(raster.L, [py.ravel(), px.ravel()], order=1, mode="constant", cval=0.0)
    return (v.reshape(px.shape) @ w).reshape(DX.shape)


def refine_local(raster_fine: MapRaster, q_xy: np.ndarray, q_w: np.ndarray, cand: Candidate,
                 stages=((0.20, 0.025, 4.0, 0.5), (0.03, 0.01, 0.5, 0.125))) -> Candidate:
    """Two-stage local search (span_m, step_m, yaw_span_deg, yaw_step_deg) around a coarse peak on the SHARP
    field with bilinear lookups. The coarse peak is quantised (5 cm / 0.5 deg) and sits on a plateau; this
    hands ICP a seed comfortably inside its basin."""
    w = q_w / q_w.sum()
    cx, cy, cyaw = cand.x, cand.y, cand.yaw
    best_sc = -1.0
    for span, step, yspan, ystep in stages:
        offs = np.arange(-span, span + 1e-9, step)
        best = (-1.0, cx, cy, cyaw)
        for dyaw in np.radians(np.arange(-yspan, yspan + 1e-9, ystep)):
            g = _eval_poses(raster_fine, q_xy, w, cx, cy, cyaw + dyaw, offs, offs)
            k = np.unravel_index(np.argmax(g), g.shape)
            if g[k] > best[0]:
                best = (float(g[k]), cx + offs[k[1]], cy + offs[k[0]], cyaw + dyaw)
        best_sc, cx, cy, cyaw = best
    return Candidate(x=cx, y=cy, yaw=cyaw, score=cand.score, refined_score=best_sc)
