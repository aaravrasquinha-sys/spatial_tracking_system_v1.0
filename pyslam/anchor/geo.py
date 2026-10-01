"""Small geometry helpers shared across the anchor package (pure numpy/scipy)."""
from __future__ import annotations
from typing import Optional, Tuple, Callable
import numpy as np
from scipy.spatial import cKDTree

from pyslam.core import lie


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = float(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(lie.so3_log(R))))


def pose_delta(Ta: np.ndarray, Tb: np.ndarray) -> Tuple[float, float]:
    """(translation metres, rotation degrees) between two poses, measured
    in Ta's frame -- symmetric in magnitude."""
    d = lie.se3_inverse(Ta) @ Tb
    return float(np.linalg.norm(d[:3, 3])), rot_angle_deg(d[:3, :3])


def voxel_downsample(pts: np.ndarray, voxel: float,
                     colors: Optional[np.ndarray] = None):
    """Mean point (and mean colour) per voxel."""
    if pts.shape[0] == 0:
        return pts, colors
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    n = counts.size
    out = np.zeros((n, 3))
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=pts[:, k], minlength=n) / counts
    if colors is None:
        return out, None
    cout = np.zeros((n, 3))
    for k in range(3):
        cout[:, k] = np.bincount(inv, weights=colors[:, k].astype(np.float64), minlength=n) / counts
    return out, np.clip(np.round(cout), 0, 255).astype(np.uint8)


def statistical_outlier_mask(pts: np.ndarray, k: int = 16, std_mult: float = 2.5) -> np.ndarray:
    """True = keep."""
    if pts.shape[0] <= k + 1:
        return np.ones(pts.shape[0], dtype=bool)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1, workers=-1)
    md = d[:, 1:].mean(axis=1)
    thr = md.mean() + std_mult * md.std()
    return md <= thr


def estimate_normals(pts: np.ndarray, tree: Optional[cKDTree] = None, k: int = 20,
                     chunk: int = 200_000) -> np.ndarray:
    """PCA normals, unit length, sign unspecified (point-to-plane doesn't
    care). Vectorised in chunks -- the covariance is built with einsum,
    the smallest-eigenvalue eigenvector via batched eigh."""
    if tree is None:
        tree = cKDTree(pts)
    n = pts.shape[0]
    out = np.zeros((n, 3))
    kk = min(k, n)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        _, idx = tree.query(pts[s:e], k=kk, workers=-1)
        nb = pts[idx]                                  # (m,k,3)
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / kk
        _, v = np.linalg.eigh(cov)
        out[s:e] = v[:, :, 0]
    return out


def fit_plane_ls(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """(unit normal, offset) with n.x = offset, via SVD."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    n = n / max(np.linalg.norm(n), 1e-12)
    return n, float(n @ c)


def ransac_plane(pts: np.ndarray, dist: float, iters: int, rng: np.random.Generator,
                 normal_ok: Optional[Callable[[np.ndarray], bool]] = None,
                 max_pts: int = 200_000):
    """Best plane by inlier count. `normal_ok` lets the caller reject
    hypotheses outright (e.g. 'must be near-horizontal') BEFORE counting
    inliers, which both speeds this up and stops a big wall from beating
    a floor when we asked for a floor. Returns (normal, offset, inlier_mask)
    or None. Refined by least squares on the final inliers."""
    n = pts.shape[0]
    if n < 3:
        return None
    sub = pts if n <= max_pts else pts[rng.choice(n, max_pts, replace=False)]
    best = None
    best_cnt = -1
    for _ in range(iters):
        i = rng.choice(sub.shape[0], 3, replace=False)
        p0, p1, p2 = sub[i]
        nv = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        if normal_ok is not None and not normal_ok(nv):
            continue
        off = float(nv @ p0)
        cnt = int(np.count_nonzero(np.abs(sub @ nv - off) < dist))
        if cnt > best_cnt:
            best_cnt, best = cnt, (nv, off)
    if best is None or best_cnt < 3:
        return None
    nv, off = best
    for _ in range(2):   # LS refine, twice (inlier set changes after the first refit)
        m = np.abs(pts @ nv - off) < dist
        if m.sum() < 3:
            return None
        nv, off = fit_plane_ls(pts[m])
    mask = np.abs(pts @ nv - off) < dist
    return nv, off, mask


def horizontal_basis(up: np.ndarray, forward_hint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Right-handed (u, v) in the plane orthogonal to `up`, u = forward_hint
    projected onto that plane (falls back to any orthogonal axis)."""
    up = up / np.linalg.norm(up)
    u = forward_hint - (forward_hint @ up) * up
    if np.linalg.norm(u) < 1e-3:
        for cand in (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])):
            u = cand - (cand @ up) * up
            if np.linalg.norm(u) > 1e-3:
                break
    u = u / np.linalg.norm(u)
    v = np.cross(up, u)
    return u, v / np.linalg.norm(v)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return lie.make_T(R, np.asarray(t, float))
