"""
Offline loop closing over EVERY keyframe of a finished run.

The live pipeline only searches the keyframes still in working memory, and
on the Jetson's 50 ms budget the first lap is evicted long before the second
lap starts -- so it never closes a lap-to-lap loop. Offline we have all the
keyframes and no time budget, so:

  1. candidates by appearance: a bag-of-words vocabulary is trained on THIS
     run's own ORB descriptors (k-means), every keyframe becomes a tf-idf
     vector, and each keyframe's top-k most similar keyframes from more than
     min_gap_s away are verified;
  2. geometric verification with the pipeline's own bidirectional PnP check
     (pyslam.loop.verify.GeometricVerifier), with a stricter inlier floor;
  3. pairwise consistency: a loop is kept only if another loop close by in
     time on BOTH ends agrees with it through the short odometry pieces
     between them (or it is very strong on its own) -- this removes isolated
     false matches from repeated texture (window grilles, tiles);
  4. robust solve with graduated non-convexity: Cauchy kernel on loops,
     annealed from "trust everything" to "trust only what fits", so a
     drifted starting guess does not make every real loop look like an
     outlier;
  5. proximity rounds: after a solve, keyframes from different times that
     now sit close together and look the same way are verified too, which
     finds revisits the appearance search missed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
import cv2

from pyslam.core import lie
from pyslam.core.log import get_logger
from pyslam.offline import pgo

log = get_logger("offline.loops")


@dataclass
class LoopParams:
    min_gap_s: float = 20.0          # a loop must join keyframes at least this far apart in time
    top_k: int = 5                   # appearance candidates verified per keyframe
    vocab_k: int = 500               # bag-of-words vocabulary size
    vocab_sample: int = 120000       # descriptors used to train it
    min_inliers: int = 25            # stricter than the live verifier's 20
    strong_inliers: int = 80         # a loop this strong needs no consistency partner
    sigma_t: float = 0.03            # loop translation std (m) at min_inliers
    sigma_r_deg: float = 1.0         # loop rotation std (deg) at min_inliers
    consist_window: int = 40         # keyframes: how far apart two loops may be to check each other
    consist_t: float = 0.15          # cycle translation tolerance (m) ...
    consist_t_per_m: float = 0.05    # ... plus this per metre of odometry inside the cycle
    consist_r_deg: float = 4.0       # cycle rotation tolerance (deg)
    gap_span: int = 3                # keyframes on each side of a lost-tracking gap tried against each other
    prox_rounds: int = 2
    prox_radius_m: float = 1.0
    prox_angle_deg: float = 35.0
    prox_per_node: int = 3
    gnc_schedule: Tuple[float, ...] = (1e4, 1e3, 300.0, 100.0, 30.0, 10.0, 5.0)
    gnc_iters: int = 20
    keep_weight: float = 0.5         # final Cauchy weight below this = loop rejected
    odom_robust_c: float = 0.0       # >0: Cauchy kernel on odometry links in the final stage


# ----------------------------------------------------------------------------- appearance

def _unpack(desc: np.ndarray) -> np.ndarray:
    return np.unpackbits(desc, axis=1).astype(np.float32)


def build_bow(nodes: Dict[int, object], p: LoopParams, seed: int = 0) -> Tuple[List[int], np.ndarray]:
    """Returns (node_ids, V) with V an (N, K) L2-normalised tf-idf matrix."""
    ids = sorted(nodes)
    descs = [nodes[i].sig.desc for i in ids]
    allD = np.concatenate([d for d in descs if d.shape[0] > 0], axis=0)
    rng = np.random.default_rng(seed)
    sample = allD[rng.choice(allD.shape[0], min(p.vocab_sample, allD.shape[0]), replace=False)]
    K = int(min(p.vocab_k, max(8, sample.shape[0] // 20)))
    t0 = time.time()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1e-3)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(_unpack(sample), K, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    c2 = (centers ** 2).sum(1)
    counts = np.zeros((len(ids), K), dtype=np.float32)
    for r, d in enumerate(descs):
        if d.shape[0] == 0:
            continue
        x = _unpack(d)
        dist = c2[None, :] - 2.0 * x @ centers.T          # |x|^2 is constant per row
        w = np.argmin(dist, axis=1)
        np.add.at(counts[r], w, 1.0)
    df = (counts > 0).sum(0)
    idf = np.log(len(ids) / np.maximum(df, 1)).astype(np.float32)
    tf = counts / np.maximum(counts.sum(1, keepdims=True), 1.0)
    V = tf * idf[None, :]
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    log.info(f"vocabulary: {K} words from {sample.shape[0]} descriptors, {len(ids)} keyframes "
             f"({time.time() - t0:.1f}s)")
    return ids, V


def appearance_candidates(ids: List[int], V: np.ndarray, times: Dict[int, float],
                          p: LoopParams) -> List[Tuple[int, int, float]]:
    S = V @ V.T
    t = np.array([times[i] for i in ids])
    far = np.abs(t[:, None] - t[None, :]) >= p.min_gap_s
    S = np.where(far, S, -1.0)
    pairs = {}
    for r in range(len(ids)):
        order = np.argsort(-S[r])[:p.top_k]
        for c in order:
            if S[r, c] <= 0.0:
                continue
            a, b = (ids[r], ids[c]) if ids[r] < ids[c] else (ids[c], ids[r])
            pairs[(a, b)] = max(pairs.get((a, b), 0.0), float(S[r, c]))
    return [(a, b, s) for (a, b), s in pairs.items()]


# ----------------------------------------------------------------------------- verification

def loop_info(n_inliers: int, p: LoopParams) -> np.ndarray:
    scale = float(np.sqrt(max(n_inliers, 1) / p.min_inliers))       # more inliers -> tighter
    st, sr = p.sigma_t / scale, np.radians(p.sigma_r_deg) / scale
    return np.diag([1 / st ** 2] * 3 + [1 / sr ** 2] * 3)


def verify_pairs(pairs, nodes, verifier, p: LoopParams, tried: set) -> List[pgo.Edge]:
    out = []
    t0 = time.time()
    for k, (a, b, score) in enumerate(pairs):
        if (a, b) in tried:
            continue
        tried.add((a, b))
        link = verifier.verify(nodes[a], nodes[b])
        if link is None or link.n_inliers < p.min_inliers:
            continue
        out.append(pgo.Edge(a, b, link.T_ab, loop_info(link.n_inliers, p), kind="loop",
                            robust="cauchy", robust_param=p.gnc_schedule[0],
                            meta={"inliers": int(link.n_inliers), "score": float(score),
                                  "rms": float(link.residual_rms)}))
    log.info(f"verified {len(pairs)} candidate pairs -> {len(out)} geometric loops ({time.time() - t0:.0f}s)")
    return out


# ----------------------------------------------------------------------------- consistency

def consistent_loops(loops: List[pgo.Edge], init: Dict[int, np.ndarray], order: Dict[int, int],
                     p: LoopParams) -> List[pgo.Edge]:
    """Keep a loop if a second loop near it on both ends agrees through odometry, or it is strong."""
    keep = []
    oa = np.array([order[L.a] for L in loops])
    srt = np.argsort(oa, kind="stable")
    oa_sorted = oa[srt]
    for i, L in enumerate(loops):
        if L.meta.get("inliers", 0) >= p.strong_inliers:
            keep.append(L)
            continue
        ok = False
        lo = int(np.searchsorted(oa_sorted, order[L.a] - p.consist_window, side="left"))
        hi = int(np.searchsorted(oa_sorted, order[L.a] + p.consist_window, side="right"))
        for j in srt[lo:hi]:
            M = loops[j]
            if i == j:
                continue
            if abs(order[L.a] - order[M.a]) > p.consist_window or abs(order[L.b] - order[M.b]) > p.consist_window:
                continue
            if order[L.a] == order[M.a] and order[L.b] == order[M.b]:
                continue
            Oa = lie.se3_inverse(init[L.a]) @ init[M.a]          # odometry L.a -> M.a
            Ob = lie.se3_inverse(init[L.b]) @ init[M.b]          # odometry L.b -> M.b
            cyc = L.T_ab @ Ob @ lie.se3_inverse(M.T_ab) @ lie.se3_inverse(Oa)
            xi = lie.se3_log(cyc)
            dist = np.linalg.norm(Oa[:3, 3]) + np.linalg.norm(Ob[:3, 3])
            if (np.linalg.norm(xi[:3]) < p.consist_t + p.consist_t_per_m * dist and
                    np.degrees(np.linalg.norm(xi[3:])) < p.consist_r_deg):
                ok = True
                break
        if ok:
            keep.append(L)
    log.info(f"consistency check: {len(keep)} of {len(loops)} loops have a consistent partner or are strong")
    return keep


# ----------------------------------------------------------------------------- robust solve

def robust_solve(init: Dict[int, np.ndarray], base_edges: List[pgo.Edge], loops: List[pgo.Edge],
                 fixed: List[int], p: LoopParams) -> Tuple[Dict[int, np.ndarray], List[float]]:
    poses = init
    edges = base_edges + loops
    for c in p.gnc_schedule:
        for L in loops:
            L.robust, L.robust_param = "cauchy", c
        poses = pgo.optimize(poses, edges, fixed, max_iters=p.gnc_iters)
    if p.odom_robust_c > 0:
        for e in base_edges:
            if e.kind == "odom":
                e.robust, e.robust_param = "cauchy", p.odom_robust_c
        poses = pgo.optimize(poses, edges, fixed, max_iters=p.gnc_iters)
        for e in base_edges:
            if e.kind == "odom":
                e.robust = None
    w = []
    for L in loops:
        r = pgo.edge_error(poses[L.a], poses[L.b], L.T_ab)
        w.append(pgo.robust_weight("cauchy", float(r @ L.info @ r), p.gnc_schedule[-1]))
    return poses, w


# ----------------------------------------------------------------------------- proximity

def proximity_candidates(poses: Dict[int, np.ndarray], times: Dict[int, float], p: LoopParams,
                         tried: set) -> List[Tuple[int, int, float]]:
    ids = sorted(poses)
    P = np.array([poses[i][:3, 3] for i in ids])
    Z = np.array([poses[i][:3, 2] for i in ids])            # optical axis (camera z) in world
    t = np.array([times[i] for i in ids])
    out = []
    for r in range(len(ids)):
        d = np.linalg.norm(P - P[r], axis=1)
        ang = np.degrees(np.arccos(np.clip(Z @ Z[r], -1, 1)))
        ok = (d < p.prox_radius_m) & (ang < p.prox_angle_deg) & (np.abs(t - t[r]) >= p.min_gap_s)
        cand = np.where(ok)[0]
        cand = cand[np.argsort(d[cand])][:p.prox_per_node]
        for c in cand:
            a, b = (ids[r], ids[c]) if ids[r] < ids[c] else (ids[c], ids[r])
            if (a, b) not in tried:
                out.append((a, b, -float(d[c])))
    return list({(a, b): (a, b, s) for a, b, s in out}.values())


# ----------------------------------------------------------------------------- driver

def close_loops(nodes: Dict[int, object], init: Dict[int, np.ndarray], base_edges: List[pgo.Edge],
                verifier, p: LoopParams) -> Tuple[Dict[int, np.ndarray], List[pgo.Edge], dict]:
    ids = sorted(nodes)
    order = {nid: k for k, nid in enumerate(ids)}
    times = {nid: float(nodes[nid].sig.t) for nid in ids}
    fixed = [ids[0]]
    stats: dict = {}

    tried: set = set()
    # 0. lost-tracking gaps: a bridge link is a guess (no measured motion), and with nothing else
    # tying the two sides together everything between two gaps floats. Try to measure it directly:
    # the keyframes just before and just after a short gap usually still see the same scene.
    gap_pairs = []
    for e in base_edges:
        if e.kind != "bridge" or e.a not in order or e.b not in order:
            continue
        ka, kb = order[e.a], order[e.b]
        for x in ids[max(0, ka - p.gap_span):ka + 1]:
            for y in ids[kb:kb + p.gap_span + 1]:
                if x != y:
                    gap_pairs.append((min(x, y), max(x, y), 1.0))
    gap_pairs = list({(a, b): (a, b, s) for a, b, s in gap_pairs}.values())
    gap_loops = verify_pairs(gap_pairs, nodes, verifier, p, tried)
    for L in gap_loops:
        L.meta["gap"] = True
    bridged = {(order[e.a]) for e in base_edges if e.kind == "bridge"}
    stats["bridges"] = len(bridged)
    stats["bridge_gap_pairs_tried"] = len(gap_pairs)
    stats["bridge_gap_links_verified"] = len(gap_loops)

    if not any(n.sig.desc.shape[0] > 0 for n in nodes.values()):
        log.warning("no keyframe has image features -- no loops can be searched; keeping the odometry/bridge poses")
        stats.update({"appearance_candidates": 0, "appearance_loops_verified": 0, "loops_final": 0,
                      "loop_time_gap_s": None, "no_descriptors": True})
        return init, [], stats
    bow_ids, V = build_bow(nodes, p)
    pairs = appearance_candidates(bow_ids, V, times, p)
    stats["appearance_candidates"] = len(pairs)
    loops = gap_loops + verify_pairs(pairs, nodes, verifier, p, tried)
    stats["appearance_loops_verified"] = len(loops) - len(gap_loops)

    poses = init
    kept, weights = [], []
    for rnd in range(p.prox_rounds + 1):
        kept = consistent_loops(loops, init, order, p)
        poses, weights = robust_solve(init, base_edges, kept, fixed, p)
        n_in = sum(1 for w in weights if w >= p.keep_weight)
        log.info(f"round {rnd}: {len(kept)} loops into the solve, {n_in} fit the final map")
        stats[f"round{rnd}_loops_in_solve"] = len(kept)
        stats[f"round{rnd}_loops_fitting"] = n_in
        if rnd == p.prox_rounds:
            break
        prox = proximity_candidates(poses, times, p, tried)
        new = verify_pairs(prox, nodes, verifier, p, tried)
        stats[f"round{rnd}_proximity_candidates"] = len(prox)
        stats[f"round{rnd}_proximity_loops"] = len(new)
        if not new:
            break
        loops = loops + new

    final_loops = [L for L, w in zip(kept, weights) if w >= p.keep_weight]
    for L, w in zip(kept, weights):
        L.meta["final_weight"] = float(w)
    # last clean solve: only the loops that fit, plain least squares on everything
    for L in final_loops:
        L.robust = None
    poses = pgo.optimize(poses, base_edges + final_loops, fixed, max_iters=50)
    stats["loops_final"] = len(final_loops)
    gaps = [abs(times[L.b] - times[L.a]) for L in final_loops]
    stats["loop_time_gap_s"] = {"median": float(np.median(gaps)) if gaps else None,
                                "over_60s": int(sum(g > 60 for g in gaps))}
    return poses, final_loops, stats
