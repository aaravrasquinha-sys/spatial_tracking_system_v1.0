"""
Fast sparse SE(3) pose-graph optimiser for the OFFLINE mapper.

Why a new one: the native backend uses finite-difference Jacobians through
scipy.least_squares, which is fine for ~100 keyframes but takes minutes per
solve at the 1000-2000 keyframes a long walk produces, and the offline
mapper re-solves many times (robust loop selection). This one uses analytic
Jacobians and a sparse Levenberg-Marquardt step, so a 2000-node graph solves
in seconds.

Conventions are the codebase's own: pose T = world<-node, link T_ab maps b's
frame into a's frame (T_ab ~= inv(T_a) @ T_b), tangent order [rho(3), phi(3)],
right perturbation T <- T @ exp(d).

Residual   e = log( inv(Z) @ inv(T_a) @ T_b )
Jacobians  de/dd_b =  Jr^-1(e)
           de/dd_a = -Jr^-1(e) @ Ad( inv(T_b) @ T_a )
(Jr^-1 approximated to first order, I + ad(e)/2, standard for pose graphs.)

Edges carry a robust kernel name: None (plain least squares) or "cauchy"
(param c, in chi2 units: weight 1/(1+chi2/c^2)). Weights are recomputed from
the current estimate every iteration (iteratively reweighted least squares).
A robust kernel started from a badly drifted initial guess simply ignores
every loop (all look like outliers), so the caller anneals c from very large
to small (graduated non-convexity) -- see loops.robust_solve.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pyslam.core import lie


@dataclass
class Edge:
    a: int
    b: int
    T_ab: np.ndarray
    info: np.ndarray            # 6x6
    kind: str = "odom"          # odom | bridge | loop
    robust: Optional[str] = None
    robust_param: float = 1.0
    meta: dict = field(default_factory=dict)


def _ad(xi: np.ndarray) -> np.ndarray:
    rho, phi = xi[:3], xi[3:]
    A = np.zeros((6, 6))
    A[:3, :3] = lie.skew(phi)
    A[:3, 3:] = lie.skew(rho)
    A[3:, 3:] = lie.skew(phi)
    return A


def edge_error(T_a: np.ndarray, T_b: np.ndarray, T_ab: np.ndarray) -> np.ndarray:
    return lie.se3_log(lie.se3_inverse(T_ab) @ lie.se3_inverse(T_a) @ T_b)


def robust_weight(kind: Optional[str], chi2: float, p: float) -> float:
    if kind is None or chi2 <= 0.0:
        return 1.0
    if kind == "cauchy":
        return 1.0 / (1.0 + chi2 / (p * p))
    raise ValueError(f"unknown robust kernel {kind!r}")


def robust_cost(kind: Optional[str], chi2: float, p: float) -> float:
    if kind is None:
        return chi2
    if kind == "cauchy":
        return p * p * np.log1p(chi2 / (p * p))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- batched SE(3)

def _bskew(v: np.ndarray) -> np.ndarray:
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def _bso3_exp(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = _bskew(w)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0, np.sin(ths) / ths)
    b = np.where(small, 0.5, (1 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + a * K + b * (K @ K)


def _bso3_log(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R, axis1=-2, axis2=-1)
    th = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    small = th < 1e-6
    s = np.where(small, 1.0, np.sin(np.where(small, 1.0, th)))
    out = np.where(small[..., None], 0.5 * v, v * (th / (2.0 * s))[..., None])
    near_pi = th > np.pi - 1e-3                   # rare (only grossly wrong loops): exact per-item
    for k in np.where(near_pi)[0]:
        out[k] = lie.so3_log(R[k])
    return out


def _bjl(w: np.ndarray, inverse: bool) -> np.ndarray:
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = _bskew(w)
    small = th < 1e-6
    ths = np.where(small, 1.0, th)
    if inverse:
        half = ths / 2.0
        c = np.where(small, 1.0 / 12.0, (1.0 - half * np.cos(half) / np.sin(half)) / ths ** 2)
        return np.eye(3) - 0.5 * K + c * (K @ K)
    b = np.where(small, 0.5, (1 - np.cos(ths)) / ths ** 2)
    c = np.where(small, 1.0 / 6.0, (1 - np.sin(ths) / ths) / ths ** 2)
    return np.eye(3) + b * K + c * (K @ K)


def _bse3_log(T: np.ndarray) -> np.ndarray:
    phi = _bso3_log(T[..., :3, :3])
    rho = np.einsum("nij,nj->ni", _bjl(phi, True), T[..., :3, 3])
    return np.concatenate([rho, phi], -1)


def _bse3_exp(xi: np.ndarray) -> np.ndarray:
    T = np.zeros(xi.shape[:-1] + (4, 4))
    T[..., :3, :3] = _bso3_exp(xi[..., 3:])
    T[..., :3, 3] = np.einsum("nij,nj->ni", _bjl(xi[..., 3:], False), xi[..., :3])
    T[..., 3, 3] = 1.0
    return T


def _binv(T: np.ndarray) -> np.ndarray:
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("nij,nj->ni", Rt, T[..., :3, 3])
    Ti[..., 3, 3] = 1.0
    return Ti


def _badjoint(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    A = np.zeros(T.shape[:-2] + (6, 6))
    A[..., :3, :3] = R
    A[..., :3, 3:] = _bskew(T[..., :3, 3]) @ R
    A[..., 3:, 3:] = R
    return A


def _bad(xi: np.ndarray) -> np.ndarray:
    A = np.zeros(xi.shape[:-1] + (6, 6))
    A[..., :3, :3] = _bskew(xi[..., 3:])
    A[..., :3, 3:] = _bskew(xi[..., :3])
    A[..., 3:, 3:] = A[..., :3, :3]
    return A


class _Packed:
    def __init__(self, edges: List[Edge], index: Dict[int, int]):
        self.ia = np.array([index[e.a] for e in edges])
        self.ib = np.array([index[e.b] for e in edges])
        self.Zinv = _binv(np.array([e.T_ab for e in edges], dtype=np.float64))
        self.info = np.array([e.info for e in edges], dtype=np.float64)
        # Cauchy parameter per edge; inf = plain least squares
        self.c = np.array([e.robust_param if e.robust == "cauchy" else np.inf for e in edges], dtype=np.float64)
        for e in edges:
            if e.robust not in (None, "cauchy"):
                raise ValueError(f"unknown robust kernel {e.robust!r}")

    def residuals(self, P: np.ndarray):
        E = self.Zinv @ _binv(P[self.ia]) @ P[self.ib]
        r = _bse3_log(E)
        chi2 = np.einsum("ni,nij,nj->n", r, self.info, r)
        return r, chi2

    def cost(self, chi2: np.ndarray) -> float:
        fin = np.isfinite(self.c)
        c2 = np.where(fin, self.c, 1.0) ** 2
        return float(np.sum(np.where(fin, c2 * np.log1p(chi2 / c2), chi2)))

    def weights(self, chi2: np.ndarray) -> np.ndarray:
        fin = np.isfinite(self.c)
        c2 = np.where(fin, self.c, 1.0) ** 2
        return np.where(fin, 1.0 / (1.0 + chi2 / c2), 1.0)


def total_cost(poses: Dict[int, np.ndarray], edges: List[Edge]) -> float:
    ids = sorted(poses)
    index = {k: i for i, k in enumerate(ids)}
    pk = _Packed(edges, index)
    P = np.array([poses[k] for k in ids])
    return pk.cost(pk.residuals(P)[1])


def optimize(poses: Dict[int, np.ndarray], edges: List[Edge], fixed: List[int],
             max_iters: int = 60, tol: float = 1e-7, verbose: bool = False) -> Dict[int, np.ndarray]:
    """Levenberg-Marquardt on the sparse normal equations; Cauchy weights re-evaluated every
    iteration. Returns a NEW dict of optimised poses (inputs untouched)."""
    ids = sorted(poses)
    index = {k: i for i, k in enumerate(ids)}
    P = np.array([poses[k] for k in ids], dtype=np.float64)
    fixed_set = set(fixed)
    free = np.array([k not in fixed_set for k in ids])
    if not free.any() or not edges:
        return {k: P[i].copy() for i, k in enumerate(ids)}
    col = -np.ones(len(ids), dtype=np.int64)
    col[free] = np.arange(int(free.sum()))
    n = 6 * int(free.sum())
    pk = _Packed(edges, index)
    ca, cb = col[pk.ia], col[pk.ib]
    # COO index pattern for the four 6x6 blocks of every edge (aa, ab, ba, bb), built once
    o = np.arange(6)
    def blk(ci, cj):
        rr = (6 * ci)[:, None, None] + o[None, :, None] + 0 * o[None, None, :]
        cc = (6 * cj)[:, None, None] + 0 * o[None, :, None] + o[None, None, :]
        return rr.reshape(len(ci), -1), cc.reshape(len(ci), -1)
    pats = [blk(ca, ca), blk(ca, cb), blk(cb, ca), blk(cb, cb)]
    oks = [(ca >= 0) & (ca >= 0), (ca >= 0) & (cb >= 0), (cb >= 0) & (ca >= 0), (cb >= 0) & (cb >= 0)]

    r, chi2 = pk.residuals(P)
    cost = pk.cost(chi2)
    lam = 1e-4
    for it in range(max_iters):
        w = pk.weights(chi2)
        Om = pk.info * w[:, None, None]
        Jinv = np.eye(6) + 0.5 * _bad(r)
        Jb = Jinv
        Ja = -Jinv @ _badjoint(_binv(P[pk.ib]) @ P[pk.ia])
        JaT, JbT = np.swapaxes(Ja, 1, 2), np.swapaxes(Jb, 1, 2)
        ga = np.einsum("nij,njk,nk->ni", JaT, Om, r)
        gb = np.einsum("nij,njk,nk->ni", JbT, Om, r)
        Hs = [JaT @ Om @ Ja, JaT @ Om @ Jb, JbT @ Om @ Ja, JbT @ Om @ Jb]
        rows = np.concatenate([pats[k][0][oks[k]].ravel() for k in range(4)])
        cols = np.concatenate([pats[k][1][oks[k]].ravel() for k in range(4)])
        vals = np.concatenate([Hs[k][oks[k]].reshape(-1) for k in range(4)])
        H = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        g = np.zeros(n)
        ma, mb = ca >= 0, cb >= 0
        np.add.at(g, (6 * ca[ma])[:, None] + o[None, :], ga[ma])
        np.add.at(g, (6 * cb[mb])[:, None] + o[None, :], gb[mb])
        diag = H.diagonal()
        improved, rel = False, 0.0
        for _ in range(10):
            Hd = H + sp.diags(lam * (diag + 1e-9) + 1e-12, format="csc")
            try:
                dx = -spla.spsolve(Hd, g)
            except Exception:  # noqa: BLE001
                lam *= 10.0
                continue
            if not np.all(np.isfinite(dx)):
                lam *= 10.0
                continue
            Pn = P.copy()
            Pn[free] = P[free] @ _bse3_exp(dx.reshape(-1, 6))
            rn, chi2n = pk.residuals(Pn)
            cn = pk.cost(chi2n)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-12)
                P, r, chi2, cost = Pn, rn, chi2n, cn
                lam = max(lam / 5.0, 1e-9)
                improved = True
                break
            lam *= 10.0
        if verbose:
            print(f"  pgo iter {it}: cost {cost:.4f} lambda {lam:.1e}")
        if not improved or rel < tol:
            break
    return {k: P[i].copy() for i, k in enumerate(ids)}


def edge_residual_m_deg(poses: Dict[int, np.ndarray], e: Edge) -> tuple:
    r = edge_error(poses[e.a], poses[e.b], e.T_ab)
    return float(np.linalg.norm(r[:3])), float(np.degrees(np.linalg.norm(r[3:])))
