"""
Single-snapshot monocular relocalization against a frozen map pack
(pyslam/mapping/reloc_map.py).

Deliberately NOT a thin wrapper around GeometricVerifier.verify(): that
method solves PnP in both directions and requires them to agree, which
is exactly the check this pipeline cannot run -- direction 1 there
needs the QUERY's own 3D points, and a monocular snapshot has none.
Calling verify() anyway would not fail loudly; it would silently fall
through to verify.py's own single-direction elif branch, quietly
disabling the cheapest false-positive guard in the system. So this
module calls the same two reused primitives (loop.verify.match_descriptors,
loop.verify.solve_pnp_ransac) directly and replaces the bidirectional
check with cross-candidate consensus (two INDEPENDENT map nodes must
place the camera in the same spot) -- see _decide() below.

Transform convention, worth re-deriving explicitly since it is the one
place an inverted or mis-Adjoint'd pose would look plausible and be
wrong (same class of bug pyslam.core.pnp_info's own docstring describes
being caught only by a Monte Carlo oracle, never by inspection):

  A map node n stores kp3d in ITS OWN camera frame (core/types.py's
  Signature docstring) and pose_map[n]: world <- n.

  solve_pnp_ransac(obj_pts=node's kp3d, img_pts=query's kp, K=query's K)
  returns raw cv2 PnP output T_cam_obj, where by OpenCV's own convention
  obj_pts are transformed INTO the camera that captured img_pts. Here
  that is exactly T_query_node (query-camera <- node-frame), directly,
  no inversion needed -- the same "direction 2" shape verify.py's own
  bidirectional check uses (node's 3D -> other image).

      T_world_query = pose_map[n] @ inverse(T_query_node)

  For the covariance: pose_map[n] is a fixed, already-known left
  constant in that composition, so (same argument pnp_info_matrix's own
  docstring makes for odometry_f2m.py's case) a right-tangent
  perturbation of inverse(T_query_node) induces the IDENTICAL
  right-tangent perturbation of T_world_query, with NO further Adjoint
  needed. And since T_world_query's own uncertainty is inverse(raw PnP
  output)'s uncertainty -- the "direction 1" shape, not "direction 2" --
  the correct covariance function is pnp_info_matrix (WITH its Adjoint
  step), not pnp_info_matrix_direct_frame. Getting this backwards
  produces a covariance that is off by an Adjoint transform: plausible
  numbers, wrong meaning, and nothing about the gate that uses them
  would look broken. See tests/gates/test_greloc.py's covariance oracle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import cv2

from pyslam.core.types import Signature, Intrinsics
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.core import lie
from pyslam.core.gravity_frame import pose_to_grav
from pyslam.core.pnp_info import pnp_info_matrix
from pyslam.frontend.features import make_orb, grid_bucket
from pyslam.loop.verify import match_descriptors, solve_pnp_ransac
from pyslam.vpr.likelihood import normalize_likelihood
from pyslam.vpr.global_desc import GlobalDescriptor
from pyslam.mapping.reloc_map import RelocMap

log = get_logger("loop.relocalize")


# --------------------------------------------------------------- query sig

def build_query_signature(rgb: np.ndarray, cfg: Config, orb=None) -> Signature:
    """Same feature stage extract_signature() runs (make_orb + grid_bucket),
    minus depth association -- there is no query depth in this pipeline's
    design (see module docstring: metric scale comes from the MAP's
    depth, not the query's). kp3d is all-NaN and valid is all-False by
    construction, matching core/types.py's own frozen contract for what
    an invalid row looks like, so this Signature is safe to hand to any
    code that checks `.valid` first -- which match/solve below always do.
    """
    if orb is None:
        orb = make_orb(cfg)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kps, descs = orb.detectAndCompute(gray, None)
    if descs is None:
        descs = np.zeros((0, 32), dtype=np.uint8)
    kps, descs = grid_bucket(kps, descs, cfg, rgb.shape[1], rgb.shape[0])
    kp_xy = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
    n = kp_xy.shape[0]
    return Signature(
        id=-1, t=time.time(),
        kp=kp_xy,
        kp3d=np.full((n, 3), np.nan, dtype=np.float32),
        valid=np.zeros(n, dtype=bool),
        desc=descs,
    )


# ------------------------------------------------------------------ result

@dataclass
class CandidateTrace:
    node_id: int
    shortlist_likelihood: float
    matched: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_rms_px: float = 0.0
    grid_cells: int = 0
    sigma_trans_m: Optional[float] = None
    passed_base_gates: bool = False
    reject_reason: Optional[str] = None
    T_world_query: Optional[np.ndarray] = None
    info6: Optional[np.ndarray] = None
    inlier_img_pts: Optional[np.ndarray] = None  # (n_inliers,2) query pixel coords,
        # for the optional --depth-crosscheck diagnostic only -- never
        # read anywhere in the accept/reject decision itself.


@dataclass
class RelocResult:
    status: str  # "LOCALIZED" | "AMBIGUOUS" | "DEGENERATE" | "NOT_FOUND"
    stage: str   # where the decision was made, for a NOT_FOUND trace
    pose_world_query: Optional[np.ndarray] = None      # 4x4, W_cam0 <- query
    pose_grav_query: Optional[np.ndarray] = None        # 4x4, W_grav <- query
    sigma_trans_m: Optional[float] = None
    node_id: Optional[int] = None
    consensus: Optional[str] = None  # "dual" | "single"
    reason: Optional[str] = None
    n_shortlisted: int = 0
    n_verified: int = 0
    retrieval_ambiguous: bool = False
    candidates: list = field(default_factory=list)   # list[CandidateTrace]
    timing_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def pose_or_none(T):
            if T is None:
                return None
            q = lie.rot_to_quat(T[:3, :3])
            return {"xyz": T[:3, 3].tolist(), "qwxyz": q.tolist(), "matrix": T.tolist()}

        return {
            "status": self.status,
            "stage": self.stage,
            "node_id": self.node_id,
            "consensus": self.consensus,
            "reason": self.reason,
            "sigma_trans_m": self.sigma_trans_m,
            "pose_world_query": pose_or_none(self.pose_world_query),
            "pose_grav_query": pose_or_none(self.pose_grav_query),
            "n_shortlisted": self.n_shortlisted,
            "n_verified": self.n_verified,
            "retrieval_ambiguous": self.retrieval_ambiguous,
            "timing_ms": self.timing_ms,
            "candidates": [
                {
                    "node_id": c.node_id,
                    "shortlist_likelihood": c.shortlist_likelihood,
                    "matched": c.matched,
                    "n_inliers": c.n_inliers,
                    "inlier_ratio": c.inlier_ratio,
                    "reproj_rms_px": c.reproj_rms_px,
                    "grid_cells": c.grid_cells,
                    "sigma_trans_m": c.sigma_trans_m,
                    "passed_base_gates": c.passed_base_gates,
                    "reject_reason": c.reject_reason,
                }
                for c in self.candidates
            ],
        }


# ------------------------------------------------------------------ stages

def _shortlist(rmap: RelocMap, query_sig: Signature, cfg: Config) -> tuple[list[int], np.ndarray]:
    """Returns (candidate_node_ids, likelihoods) -- likelihoods has one
    more entry than candidate_node_ids, the trailing 'new place' baseline
    (see vpr/likelihood.py). Two backends, selected by
    cfg.reloc_shortlist_backend; both are read-only against the mmapped
    pack arrays."""
    node_ids = rmap.node_ids
    if not node_ids:
        return [], np.array([1.0])

    if cfg.reloc_shortlist_backend == "brute":
        if query_sig.desc.shape[0] == 0 or rmap.desc.shape[0] == 0:
            raw = np.zeros(len(node_ids), dtype=np.float64)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            knn = matcher.knnMatch(query_sig.desc, np.asarray(rmap.desc), k=2)
            owner = np.asarray(rmap.owner)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            counts = np.zeros(len(node_ids), dtype=np.float64)
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, nn = pair
                if m.distance < 0.75 * nn.distance:
                    counts[id_to_idx[int(owner[m.trainIdx])]] += 1.0
            desc_per_node = np.bincount(owner, minlength=int(owner.max()) + 1 if owner.size else 0)
            denom = np.array([max(desc_per_node[nid], 1) if nid < len(desc_per_node) else 1
                               for nid in node_ids], dtype=np.float64)
            raw = counts / denom
    else:  # "global_desc" default
        # Dimension (and the seeded random-projection matrix it implies)
        # must match whatever the pack was BUILT with, not this process's
        # own cfg.p3_global_desc_dim -- read it from the pack itself.
        gdesc_fn = GlobalDescriptor(out_dim=rmap.gdesc.shape[1])
        q = gdesc_fn.compute(query_sig.desc)
        mat = np.asarray(rmap.gdesc)  # (N, dim), already L2-normalised
        raw = mat @ q if mat.shape[0] else np.zeros(0, dtype=np.float64)
        # rmap.gdesc rows correspond to rmap.gdesc_node_ids, which may not
        # be in the same order as rmap.node_ids if a future writer ever
        # changes that -- realign defensively rather than assume.
        gid_to_score = {int(nid): float(s) for nid, s in zip(rmap.gdesc_node_ids, raw)}
        raw = np.array([gid_to_score.get(nid, 0.0) for nid in node_ids], dtype=np.float64)

    order = np.argsort(-raw)
    top = order[: cfg.reloc_shortlist_topk]
    cand_ids = [node_ids[i] for i in top]
    likelihood = normalize_likelihood(raw[top])
    return cand_ids, likelihood


def _grid_cells(img_pts: np.ndarray, cfg: Config, width: int, height: int) -> int:
    if img_pts.shape[0] == 0:
        return 0
    cell_w, cell_h = width / cfg.grid_cols, height / cfg.grid_rows
    cx = np.minimum(cfg.grid_cols - 1, (img_pts[:, 0] // cell_w).astype(np.int64))
    cy = np.minimum(cfg.grid_rows - 1, (img_pts[:, 1] // cell_h).astype(np.int64))
    return len(set(zip(cx.tolist(), cy.tolist())))


def _verify_one(rmap: RelocMap, query_sig: Signature, node_id: int, K: np.ndarray,
                 cfg: Config, likelihood: float) -> CandidateTrace:
    trace = CandidateTrace(node_id=node_id, shortlist_likelihood=likelihood)
    node = rmap.get_node(node_id)
    matches = match_descriptors(node.sig.desc, query_sig.desc)
    trace.matched = len(matches)
    if len(matches) < cfg.reloc_min_inliers:
        trace.reject_reason = f"only {len(matches)} raw matches (< reloc_min_inliers={cfg.reloc_min_inliers})"
        return trace

    idx_node = np.array([m[0] for m in matches])
    idx_query = np.array([m[1] for m in matches])
    valid = node.sig.valid[idx_node]
    if valid.sum() < cfg.reloc_min_inliers:
        trace.reject_reason = (f"only {int(valid.sum())} of {len(matches)} matches have a valid "
                                f"map-side 3D point")
        return trace

    obj_pts = node.sig.kp3d[idx_node[valid]]
    img_pts = query_sig.kp[idx_query[valid]]
    solved = solve_pnp_ransac(obj_pts, img_pts, K, cfg.reloc_reproj_px, cfg.reloc_min_inliers)
    if solved is None:
        trace.reject_reason = "PnP+RANSAC did not converge to reloc_min_inliers"
        return trace
    T_cam_obj, inlier_idx, rms = solved  # T_cam_obj == T_query_node, see module docstring

    n_inliers = len(inlier_idx)
    inlier_ratio = n_inliers / max(len(matches), 1)
    # Grid cells are computed against the map's own resolution -- caller
    # (relocalize()) is responsible for having already confirmed the
    # live query stream matches it via RelocMap.check_intrinsics(); the
    # two are the same pixel grid by that point.
    grid_cells = _grid_cells(img_pts[inlier_idx], cfg, rmap.intrinsics.width, rmap.intrinsics.height)

    trace.n_inliers = n_inliers
    trace.inlier_ratio = inlier_ratio
    trace.reproj_rms_px = rms
    trace.grid_cells = grid_cells

    if n_inliers < cfg.reloc_min_inliers:
        trace.reject_reason = f"{n_inliers} inliers < reloc_min_inliers={cfg.reloc_min_inliers}"
        return trace
    if inlier_ratio < cfg.reloc_min_inlier_ratio:
        trace.reject_reason = f"inlier_ratio {inlier_ratio:.2f} < reloc_min_inlier_ratio={cfg.reloc_min_inlier_ratio}"
        return trace
    if grid_cells < cfg.reloc_min_grid_cells:
        trace.reject_reason = (f"inliers span only {grid_cells} grid cells "
                                f"(< reloc_min_grid_cells={cfg.reloc_min_grid_cells}) -- a spatial "
                                f"coverage floor, catching inliers clustered on one small image "
                                f"region (e.g. a single poster or object) rather than spread across "
                                f"the scene. NOTE: this is not a 3D-planarity test -- a fronto-parallel "
                                f"wall filling the whole frame can pass it easily, since perspective "
                                f"gives real depth variation across the image even for a flat surface. "
                                f"Genuine ill-conditioning (bas-relief-style ambiguity) is caught below, "
                                f"by the covariance gate, not here.")
        return trace

    R, t = T_cam_obj[:3, :3], T_cam_obj[:3, 3]
    T_node_query = lie.se3_inverse(T_cam_obj)
    pose_map_n = rmap.poses[node_id]
    T_world_query = pose_map_n @ T_node_query

    # See module docstring: this is the "direction 1" shape (T_world_query
    # is the INVERSE of the raw PnP solve, composed with a fixed left
    # constant), so the Adjoint-carrying function is the correct one.
    info = pnp_info_matrix(obj_pts[inlier_idx], R, t, K, cfg.pnp_sigma_px)
    try:
        cov = np.linalg.inv(info + np.eye(6) * 1e-9)
        sigma_trans_m = float(np.sqrt(max(np.trace(cov[:3, :3]), 0.0)))
    except np.linalg.LinAlgError:
        sigma_trans_m = float("inf")

    trace.sigma_trans_m = sigma_trans_m
    trace.T_world_query = T_world_query
    trace.info6 = info
    trace.inlier_img_pts = img_pts[inlier_idx].copy()

    if sigma_trans_m > cfg.reloc_max_sigma_trans_m:
        trace.reject_reason = (f"sigma_trans={sigma_trans_m:.3f}m exceeds reloc_max_sigma_trans_m="
                                f"{cfg.reloc_max_sigma_trans_m} -- geometry converged but is poorly "
                                f"conditioned")
        return trace

    trace.passed_base_gates = True
    return trace


# --------------------------------------------------------------- decision

def _decide(rmap: RelocMap, candidates: list[CandidateTrace], cfg: Config,
            retrieval_ambiguous: bool) -> tuple[str, Optional[CandidateTrace], Optional[str], Optional[str]]:
    """Returns (status, winning_candidate_or_None, consensus_label, reason)."""
    verified = [c for c in candidates if c.passed_base_gates]
    if not verified:
        return "NOT_FOUND", None, None, "no shortlisted candidate passed geometric verification"

    if len(verified) == 1:
        c = verified[0]
        if (c.n_inliers >= cfg.reloc_single_candidate_min_inliers and
                c.grid_cells >= cfg.reloc_single_candidate_min_grid_cells and
                c.sigma_trans_m <= cfg.reloc_single_candidate_max_sigma_trans_m):
            if retrieval_ambiguous:
                return ("DEGENERATE", c, "single",
                        "only one candidate verified and the shortlist itself was ambiguous "
                        "(no clear retrieval margin over the runner-up) -- geometry alone isn't "
                        "enough corroboration here")
            return "LOCALIZED", c, "single", None
        return ("DEGENERATE", c, "single",
                f"only one candidate verified and it did not clear the stricter single-candidate "
                f"bar (inliers={c.n_inliers}/{cfg.reloc_single_candidate_min_inliers}, "
                f"grid_cells={c.grid_cells}/{cfg.reloc_single_candidate_min_grid_cells}, "
                f"sigma_trans={c.sigma_trans_m:.3f}/{cfg.reloc_single_candidate_max_sigma_trans_m})")

    c1, c2 = verified[0], verified[1]
    xi = lie.se3_log(lie.se3_inverse(c1.T_world_query) @ c2.T_world_query)
    trans_diff = float(np.linalg.norm(xi[:3]))
    rot_diff = float(np.degrees(np.linalg.norm(xi[3:])))
    if trans_diff <= cfg.reloc_consensus_trans_m and rot_diff <= cfg.reloc_consensus_rot_deg:
        return "LOCALIZED", c1, "dual", None
    return ("AMBIGUOUS", None, None,
            f"top two verified candidates (nodes {c1.node_id}, {c2.node_id}) disagree: "
            f"{trans_diff:.3f}m / {rot_diff:.2f}deg apart "
            f"(limits {cfg.reloc_consensus_trans_m}m / {cfg.reloc_consensus_rot_deg}deg)")


# --------------------------------------------------------------------- API

def relocalize(rmap: RelocMap, rgb: np.ndarray, live_intr: Intrinsics, cfg: Config,
                orb=None) -> RelocResult:
    """rgb: HxWx3 uint8, from the SAME kind of stream (resolution, sensor)
    the map was built with -- see rmap.check_intrinsics(), which the
    caller should have already run. Returns a RelocResult; never raises
    for an ordinary not-found/ambiguous/degenerate outcome -- only for a
    programming error (bad shapes, an unreadable pack)."""
    t0 = time.perf_counter()
    query_sig = build_query_signature(rgb, cfg, orb=orb)
    t_extract = time.perf_counter()

    if query_sig.desc.shape[0] < cfg.reloc_min_inliers:
        return RelocResult(status="NOT_FOUND", stage="extraction",
                            reason=f"only {query_sig.desc.shape[0]} features extracted from the query",
                            timing_ms={"extract_ms": (t_extract - t0) * 1000.0})

    cand_ids, likelihood = _shortlist(rmap, query_sig, cfg)
    t_shortlist = time.perf_counter()
    if not cand_ids:
        return RelocResult(status="NOT_FOUND", stage="shortlist", reason="map pack is empty",
                            timing_ms={"extract_ms": (t_extract - t0) * 1000.0,
                                       "shortlist_ms": (t_shortlist - t_extract) * 1000.0})

    new_place_idx = len(cand_ids)
    if int(np.argmax(likelihood)) == new_place_idx:
        return RelocResult(status="NOT_FOUND", stage="shortlist",
                            reason="no map candidate stood out from the 'new place' baseline",
                            n_shortlisted=len(cand_ids),
                            timing_ms={"extract_ms": (t_extract - t0) * 1000.0,
                                       "shortlist_ms": (t_shortlist - t_extract) * 1000.0})

    retrieval_ambiguous = False
    if len(cand_ids) >= 2:
        top1_score, top2_score = float(likelihood[0]), float(likelihood[1])
        are_neighbours = cand_ids[1] in rmap.neighbours(cand_ids[0])
        if not are_neighbours and top1_score < cfg.reloc_retrieval_margin * max(top2_score, 1e-9):
            retrieval_ambiguous = True

    K = live_intr.K()
    candidates = [
        _verify_one(rmap, query_sig, nid, K, cfg, float(likelihood[i]))
        for i, nid in enumerate(cand_ids)
    ]
    t_verify = time.perf_counter()

    status, winner, consensus, reason = _decide(rmap, candidates, cfg, retrieval_ambiguous)
    timing = {"extract_ms": (t_extract - t0) * 1000.0,
              "shortlist_ms": (t_shortlist - t_extract) * 1000.0,
              "verify_ms": (t_verify - t_shortlist) * 1000.0,
              "total_ms": (t_verify - t0) * 1000.0}

    result = RelocResult(status=status, stage="decision", reason=reason,
                          n_shortlisted=len(cand_ids), n_verified=sum(c.passed_base_gates for c in candidates),
                          retrieval_ambiguous=retrieval_ambiguous, candidates=candidates, timing_ms=timing)

    if winner is not None:
        result.node_id = winner.node_id
        result.consensus = consensus
        result.sigma_trans_m = winner.sigma_trans_m
        result.pose_world_query = winner.T_world_query
        result.pose_grav_query = (pose_to_grav(winner.T_world_query, rmap.T_grav_cam0)
                                   if rmap.gravity_aligned else winner.T_world_query.copy())
    return result
