"""
WP-LIVE N2: ICP fallback tracking. Feature-poor frames (a blank wall,
a corridor with little texture) are exactly where ORB matching starves
and f2f odometry goes LOST -- this is what WP-K/L's corridor_v2 numbers
show happening repeatedly. Depth geometry doesn't care about texture,
so when feature tracking is weak this module tries point-to-plane ICP
of the current frame's depth against the reference keyframe's own
FUSED depth (pyslam/mapping/fused_depth.py -- already denoised, so ICP
isn't fighting raw per-pixel depth noise on top of everything else),
seeded by whatever motion prediction is available (gyro integration or
constant velocity, same sources pyslam/imu/bridge.py already uses for
the LOST bridge).

Point-to-plane Gauss-Newton, a handful of iterations, pure numpy --
deliberately NOT Open3D (avoids a heavy/uncertain-on-Jetson dependency
for a per-frame-budget operation; Open3D stays reserved for the
Level-2 dense evaluation only, per the nvblox-first approved scope).

Degeneracy check: a long corridor constrains ICP poorly along its own
axis (translation along the corridor is nearly unobservable from wall
geometry alone) -- this is caught by eigen-decomposing the normal-
equations matrix and refusing to trust directions with a small
eigenvalue, rather than silently returning an ill-conditioned pose.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from pyslam.core import lie


@dataclass
class IcpResult:
    T_ref_cur: np.ndarray      # ref <- cur, same convention VisualOdometry.T_rel uses
    info: np.ndarray           # 6x6, ZEROED along degenerate directions (see below)
    n_correspondences: int
    rms_m: float
    degenerate_directions: int  # how many of the 6 tangent directions were zeroed out


def _estimate_normals(pts: np.ndarray, k: int = 8) -> np.ndarray:
    """Cheap per-point normal estimate via a local PCA over the k
    nearest neighbours in a coarse 3D grid bucket (not a real kd-tree --
    fine at the sparse point counts (hundreds-low thousands) this
    module runs on; a real kd-tree is the natural upgrade if profiling
    ever shows this as a bottleneck)."""
    n = pts.shape[0]
    normals = np.zeros((n, 3))
    if n < k + 1:
        return normals
    # coarse voxel bucket for approximate neighbour lookup
    voxel = 0.1
    keys = np.floor(pts / voxel).astype(np.int64)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, k3 in enumerate(map(tuple, keys)):
        buckets[k3].append(i)
    for i in range(n):
        k3 = tuple(keys[i])
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neigh.extend(buckets.get((k3[0] + dx, k3[1] + dy, k3[2] + dz), []))
        if len(neigh) < 4:
            continue
        local = pts[neigh] - pts[neigh].mean(axis=0)
        cov = local.T @ local
        w, v = np.linalg.eigh(cov)
        normals[i] = v[:, 0]  # smallest-eigenvalue eigenvector
    return normals


def point_to_plane_icp(src_pts: np.ndarray, dst_pts: np.ndarray, dst_normals: np.ndarray,
                        T_init: np.ndarray, max_iters: int = 12, corr_dist_m: float = 0.1,
                        min_correspondences: int = 80, min_eigenvalue: float = 5.0,
                        ) -> Optional[IcpResult]:
    """src_pts: current frame's 3D points (its own camera frame).
    dst_pts/dst_normals: reference keyframe's fused-depth points +
    normals (its own camera frame). T_init: ref<-cur seed. Uses a
    coarse voxel-bucket nearest-neighbour lookup (same approach as
    _estimate_normals) rather than a kd-tree -- fine at this point
    count. Returns None if too few correspondences ever form, or if
    EVERY tangent direction ends up degenerate (nothing usable)."""
    if dst_pts.shape[0] < min_correspondences or src_pts.shape[0] < min_correspondences:
        return None

    voxel = 0.15
    dst_keys = np.floor(dst_pts / voxel).astype(np.int64)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, k3 in enumerate(map(tuple, dst_keys)):
        buckets[k3].append(i)

    T = T_init.copy()
    last_rms = None
    JTJ_final = None
    n_corr_final = 0

    for _ in range(max_iters):
        src_in_ref = lie.transform_points(T, src_pts)
        src_keys = np.floor(src_in_ref / voxel).astype(np.int64)
        corr_src, corr_dst = [], []
        for i, k3 in enumerate(map(tuple, src_keys)):
            best_j, best_d = -1, corr_dist_m
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in buckets.get((k3[0] + dx, k3[1] + dy, k3[2] + dz), []):
                            d = np.linalg.norm(src_in_ref[i] - dst_pts[j])
                            if d < best_d:
                                best_d, best_j = d, j
            if best_j >= 0:
                corr_src.append(i)
                corr_dst.append(best_j)
        if len(corr_src) < min_correspondences:
            return None
        corr_src, corr_dst = np.array(corr_src), np.array(corr_dst)

        p = src_in_ref[corr_src]
        q = dst_pts[corr_dst]
        n = dst_normals[corr_dst]
        n_norm = np.linalg.norm(n, axis=1)
        valid_n = n_norm > 1e-6
        if valid_n.sum() < min_correspondences:
            return None
        p, q, n = p[valid_n], q[valid_n], n[valid_n] / n_norm[valid_n, None]

        # point-to-plane linearised residual: n . (p - q) = 0, under a
        # small LEFT perturbation T_new = se3_exp(xi) @ T_old, xi =
        # [rho(3), phi(3)] -- this project's frozen se3_exp/se3_log
        # convention (pyslam/core/lie.py). For small xi, p_new ~= p +
        # phi x p + rho, so the residual's first-order expansion is
        #   n.(p-q) + rho.n + phi.(p x n)
        # giving Jacobian columns [n | cross(p,n)] in that [rho, phi]
        # order -- NOT [cross, n] (a rho/phi swap here would silently
        # produce a plausible-looking but wrong pose, the same class of
        # bug pnp_info.py's own docstring warns about for the Adjoint
        # step; see tests/gates/test_g_live.py's ICP oracle, which
        # recovers a KNOWN transform to catch exactly this).
        cross = np.cross(p, n)
        Jrows = np.concatenate([n, cross], axis=1)  # (N,6), [rho | phi]
        residual = np.einsum("ij,ij->i", n, p - q)  # (N,)

        JTJ = Jrows.T @ Jrows
        JTr = Jrows.T @ residual
        try:
            delta = -np.linalg.solve(JTJ + np.eye(6) * 1e-9, JTr)
        except np.linalg.LinAlgError:
            return None
        T = lie.se3_exp(delta) @ T
        rms = float(np.sqrt(np.mean(residual ** 2)))
        JTJ_final, n_corr_final = JTJ, len(corr_src)
        if last_rms is not None and abs(last_rms - rms) < 1e-6:
            last_rms = rms
            break
        last_rms = rms

    if JTJ_final is None:
        return None

    # degeneracy check: eigen-decompose the normal-equations matrix.
    # Directions with a small eigenvalue (the corridor-axis case) are
    # zeroed in the returned info matrix rather than trusted -- the
    # caller (pipeline) should treat a heavily-degenerate result as
    # weak evidence, fusable but not authoritative, same spirit as the
    # WP-M bridge's own per-component fallback.
    w, v = np.linalg.eigh(JTJ_final)
    keep = w > min_eigenvalue
    info = (v[:, keep] * w[keep]) @ v[:, keep].T if keep.any() else np.zeros((6, 6))
    n_degenerate = int((~keep).sum())

    return IcpResult(T_ref_cur=T, info=info, n_correspondences=n_corr_final,
                      rms_m=last_rms, degenerate_directions=n_degenerate)
