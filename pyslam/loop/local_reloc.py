"""
WP-LIVE N2: local RGB-D relocalization on LOST.

Before this module, a LOST event (pyslam/pipeline.py's _run_loop) always
did two things: bumped the session counter (breaking drift-metric
continuity) and added either an identity or gyro-predicted BRIDGE link
(no real measurement at all). WP-K/L/M's own findings rank that bridge
as the single largest measured error source on corridor-like fixtures
(WP-K: ~7.7deg rotation and essentially all the missing path length per
LOST event).

This module tries something better FIRST: relocalize the current
(depth-bearing) frame against a handful of recent working-memory
keyframes using 3D-3D correspondence (Kabsch/Umeyama over RANSAC
inliers), which is a strictly better-conditioned problem than the
existing monocular relocalizer's 2D-3D PnP -- both sides have real
metric depth, so there's no scale ambiguity and no need for the
existing pack's global-descriptor shortlist (the search set here is
just "the last N working-memory keyframes plus a couple of spatial
neighbours of the last confirmed pose", already small).

On success: a real, verified Link (kind="local_reloc") with an honest
information matrix, the session does NOT change, and tracking resumes
from the matched keyframe as the new reference -- no bridge is created
at all. On failure (every candidate rejected): the caller falls back to
pipeline.py's existing bridge-link machinery unchanged.

Reuses pyslam.loop.verify's descriptor matcher (bit-identical matching
to every other part of this codebase) but NOT its PnP-based solve --
see solve_3d3d_ransac below for why 3D-3D is the right tool here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

from pyslam.core.types import Signature, Node, Link
from pyslam.core.config import Config
from pyslam.core import lie
from pyslam.loop.verify import match_descriptors

log = None
try:
    from pyslam.core.log import get_logger
    log = get_logger("loop.local_reloc")
except Exception:
    pass


@dataclass
class LocalRelocResult:
    node_id: int
    T_node_query: np.ndarray   # node <- query (maps query-frame points into node's frame)
    n_inliers: int
    inlier_ratio: float
    rms_m: float
    info: np.ndarray


def _kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid (no-scale) transform T such that T @ src ~= dst, both
    (N,3) point sets, N>=3, ALREADY CORRESPONDING (same index = same
    physical point). Standard Umeyama/Kabsch, no reflection allowed
    (det(R)=+1 enforced) -- mirrors this codebase's own det(R)>=+1
    discipline (tests/synth/world.py's mirrored-fixture fix, WP-T)."""
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst.mean(axis=0) - R @ src.mean(axis=0)
    return lie.make_T(R, t)


def solve_3d3d_ransac(src_pts: np.ndarray, dst_pts: np.ndarray, dist_thresh_m: float = 0.05,
                       iters: int = 300, min_inliers: int = 20,
                       rng: Optional[np.random.Generator] = None
                       ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """src_pts/dst_pts: (N,3) CORRESPONDING points (same index = same
    matched keypoint pair, both with valid depth) in each side's own
    camera frame. Returns (T_dst_src, inlier_idx, rms_m) or None.
    RANSAC over minimal 3-point Kabsch solves, same discipline as
    solvePnPRansac elsewhere in this codebase (random minimal sample,
    count inliers, refit on the full inlier set)."""
    rng = rng or np.random.default_rng(0)
    n = src_pts.shape[0]
    if n < min_inliers:
        return None
    best_inliers = None
    for _ in range(iters):
        idx3 = rng.choice(n, size=3, replace=False)
        try:
            T = _kabsch(src_pts[idx3], dst_pts[idx3])
        except np.linalg.LinAlgError:
            continue
        pred = lie.transform_points(T, src_pts)
        dist = np.linalg.norm(pred - dst_pts, axis=1)
        inliers = np.where(dist < dist_thresh_m)[0]
        if best_inliers is None or inliers.size > best_inliers.size:
            best_inliers = inliers
    if best_inliers is None or best_inliers.size < min_inliers:
        return None
    T = _kabsch(src_pts[best_inliers], dst_pts[best_inliers])
    pred = lie.transform_points(T, src_pts[best_inliers])
    rms = float(np.sqrt(np.mean(np.sum((pred - dst_pts[best_inliers]) ** 2, axis=1))))
    return T, best_inliers, rms


def try_local_relocalization(query_sig: Signature, candidate_nodes: list[Node], cfg: Config,
                              min_inliers: int = 25, min_inlier_ratio: float = 0.35,
                              max_rms_m: float = 0.03, dist_thresh_m: float = 0.05,
                              ) -> Optional[LocalRelocResult]:
    """query_sig: current frame's Signature (kp3d/valid populated from
    ITS OWN depth -- unlike the frozen-map relocalizer, this is the
    depth-assisted path from the start, since both sides are live RGB-D
    frames). candidate_nodes: recent working-memory Nodes to try, in
    the order the caller wants them attempted (nearest-in-time first is
    the sensible default). Returns the FIRST candidate that clears
    every gate below -- unlike the frozen-map relocalizer's cross-
    candidate consensus requirement, a single well-conditioned 3D-3D
    match is accepted directly here: cross-candidate consensus exists
    there specifically to compensate for monocular PnP's weaker single-
    view conditioning (RELOCALIZATION.md's own reasoning), which does
    not apply when both sides carry real depth and every accepted match
    is still individually inlier/ratio/RMS-gated.
    """
    for node in candidate_nodes:
        matches = match_descriptors(node.sig.desc, query_sig.desc)
        if len(matches) < min_inliers:
            continue
        idx_node = np.array([m[0] for m in matches])
        idx_query = np.array([m[1] for m in matches])
        both_valid = node.sig.valid[idx_node] & query_sig.valid[idx_query]
        if both_valid.sum() < min_inliers:
            continue
        src = query_sig.kp3d[idx_query[both_valid]].astype(np.float64)  # query frame
        dst = node.sig.kp3d[idx_node[both_valid]].astype(np.float64)    # node frame

        solved = solve_3d3d_ransac(src, dst, dist_thresh_m=dist_thresh_m, min_inliers=min_inliers)
        if solved is None:
            continue
        T_node_query, inlier_idx, rms = solved
        n_inliers = inlier_idx.size
        inlier_ratio = n_inliers / max(both_valid.sum(), 1)
        if inlier_ratio < min_inlier_ratio or rms > max_rms_m:
            continue

        # Information matrix: crude but honest -- scales with inlier
        # count and inversely with residual spread, capped the same way
        # verify.py's own heuristic loop-link info is capped, so this
        # link doesn't dominate a real optimize() call the way an
        # over-confident bridge link would. A Hessian-based version is
        # the natural WP-LIVE follow-up (mirrors pnp_info_matrix's role
        # for the 2D-3D case) but isn't required for this to be a much
        # better constraint than the identity/gyro bridge it replaces.
        info_scale = min(n_inliers, 200) / max(rms, 1e-3)
        info = np.eye(6) * info_scale

        if log is not None:
            log.info(f"local_reloc: matched node {node.id}, inliers={n_inliers} "
                     f"ratio={inlier_ratio:.2f} rms={rms*1000:.1f}mm")
        return LocalRelocResult(node_id=node.id, T_node_query=T_node_query,
                                 n_inliers=n_inliers, inlier_ratio=inlier_ratio,
                                 rms_m=rms, info=info)
    return None


def build_local_reloc_link(a_node_id: int, b_query_node: Node, result: LocalRelocResult) -> Link:
    """Builds the graph Link once the caller has decided to keep a
    successful local_reloc as the new keyframe's anchor (mirrors the
    shape of an ordinary odom Link -- a=matched existing node,
    b=new keyframe created from the query frame). T_ab = a<-b, matching
    every other Link in this codebase's convention (T_ab maps b's frame
    into a's)."""
    # result.T_node_query IS already "a<-b" (matched node <- query frame),
    # this codebase's Link.T_ab convention (see e.g. pipeline.py's odom
    # Link construction: a=prev keyframe, b=new keyframe, T_ab=ref<-cur)
    # -- no inversion needed.
    return Link(a=a_node_id, b=b_query_node.id, T_ab=result.T_node_query, info=result.info,
                kind="local_reloc", n_inliers=result.n_inliers)
