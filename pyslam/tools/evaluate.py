from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core import lie


def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Find (R, t, s) minimising ||dst - (s*R@src + t)||^2. src,dst: (N,3)."""
    assert src.shape == dst.shape and src.shape[0] >= 3
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def ate_rmse(est_positions: np.ndarray, gt_positions: np.ndarray, align: bool = True) -> Optional[float]:
    """Returns None (not a crash) when there are fewer than 3 keyframes --
    Umeyama alignment is undefined below that, and a single-keyframe run
    (static_60s by design, or any very short live capture) is a valid
    outcome, not a bug. Callers must handle None explicitly rather than
    assuming a float; see run_synth.py/trajectory_export.py for the
    'ATE: not enough keyframes to score' path."""
    if est_positions.shape[0] < 3 or gt_positions.shape[0] < 3:
        return None
    if align:
        R, t, s = umeyama_align(est_positions, gt_positions)
        est_aligned = (s * (R @ est_positions.T).T) + t
    else:
        est_aligned = est_positions
    err = np.linalg.norm(est_aligned - gt_positions, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def rpe(est_poses: list[np.ndarray], gt_poses: list[np.ndarray], delta: int = 1):
    """Mean relative-pose error at a fixed index delta. Returns (trans_m, rot_deg)."""
    n = len(est_poses)
    if n <= delta:
        return 0.0, 0.0
    t_errs, r_errs = [], []
    for i in range(n - delta):
        rel_est = lie.se3_inverse(est_poses[i]) @ est_poses[i + delta]
        rel_gt = lie.se3_inverse(gt_poses[i]) @ gt_poses[i + delta]
        diff = lie.se3_inverse(rel_gt) @ rel_est
        xi = lie.se3_log(diff)
        t_errs.append(np.linalg.norm(xi[:3]))
        r_errs.append(np.degrees(np.linalg.norm(xi[3:])))
    return float(np.mean(t_errs)), float(np.mean(r_errs))


def path_length(positions: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
