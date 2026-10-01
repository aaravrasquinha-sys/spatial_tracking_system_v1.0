"""
Stage 4: robust multi-scale point-to-plane ICP, 6 DOF, roll/pitch NOT clamped to the IMU (so the IMU
comparison in verification stays a genuinely independent check).

  * Gauss-Newton on se(3) with the project's [rho, phi] (translation-first) convention and a LEFT
    perturbation  T <- exp(xi) @ T  ->  for a transformed point p and map normal n:  J = [ n , p x n ].
    (Same convention lesson icp_fallback.py's Jacobian-ordering bug taught; checked by the oracle test.)
  * Robustness: correlation gating at each scale, trimming of the worst `icp_trim_frac` residuals, then a
    Tukey kernel with c = the scale's correspondence distance. Furniture moved since mapping is an outlier,
    not a bias.
  * Weights: 1/(sigma_z^2 + sigma_map^2) from the D435i noise model (far points count less).
  * Degeneracy/observability: eigen-decomposition of the (range-normalised) normal matrix built about the
    CAMERA origin; the worst-conditioned direction is named in human terms.
  * Covariance: (sum w J J^T)^-1 with w = 1/sigma_i^2, scaled by max(1, chi2/dof) and an inflation for the
    spatial correlation of depth noise -- validated against the empirical sub-capture spread, not trusted blind.

Colour: a colour-correlation DIAGNOSTIC is provided (used only to break geometric ties); a full coloured-ICP
objective is NOT implemented (see SYSTEM_SUMMARY_ANCHOR.md, 'Not built').
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from scipy.spatial import cKDTree

from pyslam.core import lie
from pyslam.anchor import geo
from pyslam.anchor.config import AnchorConfig
from pyslam.core.log import get_logger

log = get_logger("anchor.icp")


class MapTarget:
    """A cropped, normal-annotated piece of the room-frame map with a KD-tree."""

    def __init__(self, pts: np.ndarray, colors: Optional[np.ndarray], k_normals: int = 20):
        self.pts = np.ascontiguousarray(pts, dtype=np.float64)
        self.colors = colors
        self.tree = cKDTree(self.pts)
        self.normals = geo.estimate_normals(self.pts, self.tree, k=k_normals)

    @classmethod
    def crop(cls, pts_room: np.ndarray, colors: Optional[np.ndarray], center: np.ndarray, radius: float,
             k_normals: int = 20) -> "MapTarget":
        m = np.linalg.norm(pts_room - center[None], axis=1) <= radius
        if m.sum() < 1000:
            raise ValueError(f"only {int(m.sum())} map points within {radius:.1f} m of the seed camera position -- "
                             f"seed is outside the mapped area or the map has a hole here")
        return cls(pts_room[m], None if colors is None else colors[m], k_normals)


@dataclass
class IcpResult:
    T: np.ndarray
    rmse_m: float
    fitness: float
    coverage: float
    n_inliers: int
    n_query: int
    converged: bool
    cov: np.ndarray                     # 6x6, camera-origin-centred [trans(3), rot(3)], room-frame axes
    obs_min_eig: float
    obs_worst: str
    chi2_scale: float
    stage_log: List[dict] = field(default_factory=list)

    def sigma_trans_m(self) -> float:
        return float(np.sqrt(np.max(np.linalg.eigvalsh(self.cov[:3, :3]))))

    def sigma_rot_deg(self) -> float:
        return float(np.degrees(np.sqrt(np.max(np.linalg.eigvalsh(self.cov[3:, 3:])))))


def _tukey(r_abs: np.ndarray, c: float) -> np.ndarray:
    u = np.clip(r_abs / c, 0.0, 1.0)
    return (1.0 - u * u) ** 2


def _gn_step(p: np.ndarray, q: np.ndarray, n: np.ndarray, w: np.ndarray, r: np.ndarray, lam: float = 1e-9):
    J = np.concatenate([n, np.cross(p, n)], axis=1)          # (m,6)
    A = (J * w[:, None]).T @ J
    b = -(J * w[:, None]).T @ r
    A = A + lam * np.eye(6) * max(np.trace(A) / 6.0, 1e-12)
    return np.linalg.solve(A, b)


def _describe_direction(v: np.ndarray, lever: float) -> str:
    t, rvec = v[:3], v[3:]
    if np.linalg.norm(t) >= np.linalg.norm(rvec):
        d = t / max(np.linalg.norm(t), 1e-12)
        return f"translation along room axis ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})"
    d = rvec / max(np.linalg.norm(rvec), 1e-12)
    return f"rotation about room axis ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})"


def evaluate(T: np.ndarray, q_cam: np.ndarray, sigma: np.ndarray, target: MapTarget, cfg: AnchorConfig) -> dict:
    """Metrics + covariance + observability at a given pose (no optimisation)."""
    p = geo_transform(T, q_cam)
    d, idx = target.tree.query(p, distance_upper_bound=cfg.coverage_dist_m, workers=-1)
    covered = np.isfinite(d)
    coverage = float(covered.mean()) if q_cam.shape[0] else 0.0
    fit_m = np.isfinite(d) & (d <= cfg.fitness_dist_m)
    fitness = float(fit_m.mean()) if q_cam.shape[0] else 0.0
    if fit_m.sum() < 30:
        return {"rmse": np.inf, "fitness": fitness, "coverage": coverage, "n_inl": int(fit_m.sum()),
                "cov": np.full((6, 6), np.inf), "obs_min": 0.0, "obs_worst": "no inliers", "chi2_scale": np.inf}
    pi = p[fit_m]
    qi = target.pts[idx[fit_m]]
    ni = target.normals[idx[fit_m]]
    r = np.einsum("ij,ij->i", ni, pi - qi)
    rmse = float(np.sqrt(np.mean(r ** 2)))
    cam = T[:3, 3]
    rel = pi - cam[None]
    J = np.concatenate([ni, np.cross(rel, ni)], axis=1)
    s2 = sigma[fit_m] ** 2 + cfg.map_sigma_m ** 2
    A = (J / s2[:, None]).T @ J
    chi2 = float(np.sum(r ** 2 / s2))
    dof = max(int(fit_m.sum()) - 6, 1)
    scale = max(1.0, chi2 / dof)
    try:
        cov = np.linalg.inv(A) * scale * cfg.icp_cov_inflation
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.inf)
    lever = float(np.mean(np.linalg.norm(rel, axis=1)))
    Jn = np.concatenate([ni, np.cross(rel, ni) / max(lever, 1e-6)], axis=1)
    An = Jn.T @ Jn / Jn.shape[0]
    ev, evec = np.linalg.eigh(An)
    return {"rmse": rmse, "fitness": fitness, "coverage": coverage, "n_inl": int(fit_m.sum()), "cov": cov,
            "obs_min": float(ev[0]), "obs_worst": _describe_direction(evec[:, 0], lever), "chi2_scale": scale,
            "obs_eigs": ev.tolist()}


def geo_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def icp(q_cam: np.ndarray, weights: np.ndarray, sigma: np.ndarray, target: MapTarget, T0: np.ndarray,
        cfg: Optional[AnchorConfig] = None, scales=None, iters=None) -> IcpResult:
    cfg = cfg or AnchorConfig()
    scales = scales or cfg.icp_scales_m
    iters = iters or cfg.icp_iters
    T = T0.copy()
    log_stages = []
    converged = True
    n_stage = len(scales)
    # Range weights: statistically 1/sigma^2 (sigma ~ z^2) spans >100x between 1 m and 4 m, which lets the
    # near field alone decide convergence and starves the far walls that pin x/y/yaw. Compress the dynamic
    # range for the OPTIMISATION (the covariance below still uses the true per-point sigma).
    weights = np.clip(weights / max(float(np.median(weights)), 1e-12), 0.15, 3.0)
    for si, (scale, n_it) in enumerate(zip(scales, iters)):
        last = None
        final_stage = (si == n_stage - 1)
        for it in range(n_it):
            p = geo_transform(T, q_cam)
            d, idx = target.tree.query(p, distance_upper_bound=scale, workers=-1)
            m = np.isfinite(d)
            if m.sum() < 100:
                converged = False
                log.warning(f"ICP scale {scale}: only {int(m.sum())} correspondences -- seed too far off")
                break
            pm, qm, nm, wm = p[m], target.pts[idx[m]], target.normals[idx[m]], weights[m]
            r = np.einsum("ij,ij->i", nm, pm - qm)
            ra = np.abs(r)
            if final_stage and cfg.icp_trim_frac > 0:
                # trimming only at the FINAL scale: earlier, the worst residuals are often exactly the far
                # walls that carry the constraint, and dropping them lets ICP settle into a wrong minimum
                keep = ra <= np.quantile(ra, 1.0 - cfg.icp_trim_frac)
                pm, qm, nm, wm, r, ra = pm[keep], qm[keep], nm[keep], wm[keep], r[keep], ra[keep]
            wt = wm * _tukey(ra, scale)
            if wt.sum() < 1e-9:
                converged = False
                break
            xi = _gn_step(pm, qm, nm, wt, r)
            T = lie.se3_exp(xi) @ T
            last = (float(np.linalg.norm(xi[:3])), float(np.linalg.norm(xi[3:])))
            if last[0] < 2e-5 and last[1] < 2e-6:
                break
        log_stages.append({"scale_m": scale, "iters": it + 1, "last_step": last})
    ev = evaluate(T, q_cam, sigma, target, cfg)
    return IcpResult(T=T, rmse_m=ev["rmse"], fitness=ev["fitness"], coverage=ev["coverage"], n_inliers=ev["n_inl"],
                     n_query=int(q_cam.shape[0]), converged=converged, cov=ev["cov"], obs_min_eig=ev["obs_min"],
                     obs_worst=ev["obs_worst"], chi2_scale=ev["chi2_scale"], stage_log=log_stages)


def colour_correlation(T: np.ndarray, q_cam: np.ndarray, q_rgb: Optional[np.ndarray], target: MapTarget,
                       cfg: AnchorConfig) -> Optional[float]:
    """Pearson correlation of luminance between query pixels and their nearest map points (gain/offset
    invariant, since the map's fused colours and a single static exposure never match absolutely).
    Higher = colours agree better. None if either side has no colour."""
    if q_rgb is None or target.colors is None or q_cam.shape[0] < 100:
        return None
    p = geo_transform(T, q_cam)
    d, idx = target.tree.query(p, distance_upper_bound=cfg.fitness_dist_m, workers=-1)
    m = np.isfinite(d)
    if m.sum() < 100:
        return None
    lum = lambda c: c[:, 0] * 0.299 + c[:, 1] * 0.587 + c[:, 2] * 0.114
    a = lum(q_rgb[m].astype(np.float64))
    b = lum(target.colors[idx[m]].astype(np.float64))
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])
