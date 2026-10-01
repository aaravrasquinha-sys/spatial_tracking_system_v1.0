"""Depth rendering from the map (point splat + guarded hole fill) and the visual-check overlays."""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter


def render_depth(pts_room: np.ndarray, T_room_cam: np.ndarray, intr: dict, max_range: float = 8.0,
                 fill: bool = True) -> np.ndarray:
    """Map -> depth image (float32 metres, NaN = no map). z-buffered (nearest wins); a 5x5 min-filter fills
    sub-pixel gaps between 2 cm map points, but ONLY where the neighbourhood is depth-continuous (so it
    never bridges a true silhouette)."""
    H, W = int(intr["height"]), int(intr["width"])
    R, t = T_room_cam[:3, :3], T_room_cam[:3, 3]
    pc = (pts_room - t) @ R                                   # R^T (p - t)
    z = pc[:, 2]
    m = (z > 0.2) & (z < max_range)
    pc, z = pc[m], z[m]
    u = np.rint(intr["fx"] * pc[:, 0] / z + intr["cx"]).astype(int)
    v = np.rint(intr["fy"] * pc[:, 1] / z + intr["cy"]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    order = np.argsort(-z)                                    # far first, near last => near wins
    buf = np.full((H, W), np.nan, np.float32)
    buf[v[order], u[order]] = z[order].astype(np.float32)
    if not fill:
        return buf
    fin = np.isfinite(buf)
    mn = minimum_filter(np.where(fin, buf, np.inf), size=5)
    mx = maximum_filter(np.where(fin, buf, -np.inf), size=5)
    cont = np.isfinite(mn) & np.isfinite(mx) & ((mx - mn) < (0.06 + 0.03 * np.nan_to_num(mn, posinf=0)))
    out = buf.copy()
    fillm = ~fin & cont
    out[fillm] = mn[fillm]
    return out


def depth_residual(live: np.ndarray, rendered: np.ndarray, stable: np.ndarray, tol: float) -> dict:
    both = stable & np.isfinite(rendered) & np.isfinite(live)
    n_stable = int(stable.sum())
    if both.sum() == 0:
        return {"frac_within_tol": 0.0, "render_coverage": 0.0, "n_compared": 0, "median_abs_m": float("nan"),
                "residual_img": np.full(live.shape, np.nan, np.float32)}
    res = np.where(both, live - rendered, np.nan).astype(np.float32)
    ra = np.abs(res[both])
    return {"frac_within_tol": float(np.mean(ra <= tol)), "render_coverage": float(both.sum() / max(n_stable, 1)),
            "n_compared": int(both.sum()), "median_abs_m": float(np.median(ra)), "residual_img": res}


def _edges(d: np.ndarray) -> np.ndarray:
    fin = np.isfinite(d)
    mx = maximum_filter(np.where(fin, d, -np.inf), size=3)
    mn = minimum_filter(np.where(fin, d, np.inf), size=3)
    return fin & np.isfinite(mx - mn) & ((mx - mn) > (0.08 + 0.03 * np.nan_to_num(d, nan=0)))


def overlay_edges(rgb: np.ndarray, rendered: np.ndarray, live: np.ndarray) -> np.ndarray:
    """RGB with MAP depth-edges in red and LIVE depth-edges in green. A correct calibration shows red and
    green lines lying on top of each other (yellow where they coincide); a yaw error shows two parallel lines."""
    img = rgb.copy() if rgb is not None and rgb.ndim == 3 else np.zeros(rendered.shape + (3,), np.uint8)
    img = (img * 0.6).astype(np.uint8)
    er, el = _edges(rendered), _edges(live)
    img[el] = (0, 255, 0)
    img[er] = (255, 0, 0)
    img[er & el] = (255, 255, 0)
    return img


def colorize_residual(res: np.ndarray, vmax: float = 0.06) -> np.ndarray:
    import cv2
    x = np.clip(np.nan_to_num(res, nan=0.0) / vmax, -1, 1)
    g = ((x * 0.5 + 0.5) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(g, cv2.COLORMAP_JET)[:, :, ::-1].copy()
    cm[~np.isfinite(res)] = (30, 30, 30)
    return cm


def save_png(path: str, rgb: np.ndarray) -> None:
    import cv2
    cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
