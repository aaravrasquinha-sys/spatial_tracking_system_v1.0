"""
WP-K5 (Phase A5): measurement harness for the Orin accuracy plan.

    # run + record a labelled baseline (3 fixtures x 3 seeds by default)
    python3 -m pyslam.tools.phase_a_baseline --label A5_prox_on \\
        --config-override proximity_enabled=true

    # the f2m-vs-f2f re-measurement Phase A calls for
    python3 -m pyslam.tools.phase_a_baseline --label A5_f2m \\
        --config-override proximity_enabled=true --config-override odometry_backend=f2m

    # compare two recorded baselines, median per scenario
    python3 -m pyslam.tools.phase_a_baseline --compare baselines/A5_prox_on.json baselines/A5_f2m.json

Deliberately SEPARATE from tools/baseline.py + baseline_store.json: that
store is the frozen WP-A3 gate (native backend, f2f, proximity off) and
its numbers are what old work packages were judged against -- do not
overwrite it. This tool writes labelled files under baselines/ instead.

What it measures that nothing did before (each item exists because a real
run log showed the old metrics could not answer a question we needed
answered):

  * LOOP AUDIT vs ground truth. corridor_v2's 20 "loop closures" were all
    ~10 keyframes apart (the moment a node leaves STM it becomes a
    candidate and trivially matches its own recent past). Every loop is
    classified here by time gap and by whether its measured relative
    pose actually agrees with ground truth, so "20 loops" can no longer
    be mistaken for "20 real loop closures".
  * VERTICAL vs HORIZONTAL error, referenced to ground truth (not the
    IMU-derived height_range_m in trajectory_report.json, which is
    meaningless whenever gravity alignment failed -- it then measures
    travel along the first camera's optical axis). This is the evidence
    the IMU/tilt work (Phase C) has to be justified by.
  * BRIDGE AUDIT: each LOST recovery inserts an identity "no motion"
    bridge link; this reports how far the camera REALLY moved across
    each one (ground truth), i.e. how wrong that assumption is.
  * ESTIMATED vs GROUND-TRUTH path length.
  * MAP COVERAGE: how many keyframes made it into the exported cloud.
  * TIMING/MEMORY: per-keyframe mem_duration_ms percentiles (needed to
    retune wm_budget_ms for this machine, per WP-J2), fps, WM size.
  * Post-finalize ATE: the trajectory and map a user actually receives.

Everything here is computed from a normal Pipeline run + the synthetic
fixture's ground truth; no pipeline code is modified or wrapped.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import numpy as np

sys.path.insert(0, ".")

from pyslam.core import lie
from pyslam.core.config import Config, add_config_args, config_from_args
from pyslam.core.log import get_logger
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from pyslam.tools import metrics
from pyslam.tools.evaluate import ate_rmse
from pyslam.tools.run_outputs import summarize_telemetry
from pyslam.mapping.cloud import assemble_cloud_with_stats

log = get_logger("phase_a_baseline")

DEFAULT_SCENARIOS = ["square6dof", "room_orbit", "corridor_v2"]
DEFAULT_SEEDS = [1, 2, 3]

# A loop/proximity link whose measured relative pose disagrees with ground
# truth by more than this is counted as WRONG. Deliberately generous
# (a correct link at ~1-2cm error is far inside it): the point is to catch
# false matches, not to grade accuracy.
WRONG_LINK_TRANS_M = 0.25
WRONG_LINK_ROT_DEG = 5.0


# --------------------------------------------------------------------------
# Pure analysis functions (unit-tested in tests/gates/test_ga.py)
# --------------------------------------------------------------------------
def link_gt_error(link, gt_by_id: dict) -> Optional[dict]:
    """How far a link's measured T_ab is from ground truth. Link.T_ab maps
    points in b's frame into a's frame, so the true value is
    inv(gt_a) @ gt_b (gt_* are world<-cam). None if either node has no GT."""
    if link.a not in gt_by_id or link.b not in gt_by_id:
        return None
    T_true = lie.se3_inverse(gt_by_id[link.a]) @ gt_by_id[link.b]
    xi = lie.se3_log(lie.se3_inverse(T_true) @ link.T_ab)
    return {"trans_err_m": float(np.linalg.norm(xi[:3])),
            "rot_err_deg": float(np.degrees(np.linalg.norm(xi[3:]))),
            "true_trans_m": float(np.linalg.norm(T_true[:3, 3])),
            "true_rot_deg": float(np.degrees(np.linalg.norm(lie.se3_log(T_true)[3:])))}


def gt_path_positions(gt_by_id: dict) -> dict:
    """node_id -> cumulative ground-truth path length (m) along the keyframe
    sequence (ids sorted ascending). The yardstick for 'how much unique
    ground was covered between two nodes' -- independent of the config
    under test, of walking speed and of keyframe density."""
    ids = sorted(gt_by_id)
    out, acc, prev = {}, 0.0, None
    for i in ids:
        p = gt_by_id[i][:3, 3]
        if prev is not None:
            acc += float(np.linalg.norm(p - prev))
        out[i] = acc
        prev = p
    return out


def gt_chain_stats(gt_by_id: dict) -> dict:
    """node_id -> (cumulative GT path m, cumulative GT rotation deg, keyframe rank)
    along the keyframe sequence. Three different yardsticks for 'how much
    odometry has accumulated between two nodes' -- WP-L found that path alone
    misclassifies rotation-dominated motion (room_orbit), so all three are
    reported per link and the choice of bound is made from data."""
    ids = sorted(gt_by_id)
    out, path, rot, prev = {}, 0.0, 0.0, None
    for k, i in enumerate(ids):
        T = gt_by_id[i]
        if prev is not None:
            path += float(np.linalg.norm(T[:3, 3] - prev[:3, 3]))
            c = (np.trace(prev[:3, :3].T @ T[:3, :3]) - 1.0) / 2.0
            rot += float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        out[i] = (path, rot, k)
        prev = T
    return out


def revisit_recall(gt_by_id: dict, links: list, min_path_m: float = 1.5,
                   max_dist_m: float = 0.5, max_rot_deg: float = 30.0) -> dict:
    """Loop-closure RECALL against ground truth. A keyframe is a revisit
    OPPORTUNITY if some earlier keyframe is within max_dist_m and max_rot_deg
    of it but at least min_path_m of ground-truth path away. It is a HIT if a
    non-wrong loop link with that keyframe as an endpoint reaches a partner
    at least min_path_m away. Frame-based, deliberately simple; the point is
    to make 'the fixture has 8 frames that could close a loop, and we found 0'
    a number instead of an argument."""
    ids = sorted(gt_by_id)
    cum = gt_path_positions(gt_by_id)
    opp = set()
    for jj, j in enumerate(ids):
        Tj = gt_by_id[j]
        for i in ids[:jj]:
            if cum[j] - cum[i] < min_path_m:
                continue
            Ti = gt_by_id[i]
            if np.linalg.norm(Ti[:3, 3] - Tj[:3, 3]) >= max_dist_m:
                continue
            cosang = (np.trace(Ti[:3, :3].T @ Tj[:3, :3]) - 1.0) / 2.0
            if np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) < max_rot_deg:
                opp.add(j)
                break
    hit = set()
    for l in links:
        err = link_gt_error(l, gt_by_id)
        if err is None or l.a not in cum or l.b not in cum:
            continue
        if err["trans_err_m"] > WRONG_LINK_TRANS_M or err["rot_err_deg"] > WRONG_LINK_ROT_DEG:
            continue
        if abs(cum[l.b] - cum[l.a]) < min_path_m:
            continue
        hit.update((l.a, l.b))
    hits = len(opp & hit)
    return {"n_keyframes": len(ids), "n_opportunity_frames": len(opp), "n_hit_frames": hits,
            "recall": (hits / len(opp)) if opp else None, "min_path_m": min_path_m}


def loop_audit(links_or_events: list, gt_by_id: dict, node_t: dict, min_gap_s: float = 10.0,
               min_gap_m: float = 0.0) -> dict:
    """Classify loop (or proximity) links.

    links_or_events: list of Link objects.
    node_t: node_id -> capture time (s).
    A link is TRIVIAL if the two nodes were captured < min_gap_s apart
    (a revisit of the last few seconds -- not a loop closure in any useful
    sense, just the retrieval window's own edge), and WRONG if its relative
    pose disagrees with ground truth by more than the WRONG_LINK_* bounds.
    'n_real' = non-trivial AND not wrong: the number that actually
    measures loop-closure capability."""
    rows = []
    cum = gt_path_positions(gt_by_id) if min_gap_m > 0.0 else {}
    for l in links_or_events:
        err = link_gt_error(l, gt_by_id)
        if err is None or l.a not in node_t or l.b not in node_t:
            continue
        gap = abs(node_t[l.b] - node_t[l.a])
        pgap = abs(cum[l.b] - cum[l.a]) if (l.a in cum and l.b in cum) else None
        # WP-L: with min_gap_m > 0 the yardstick is GROUND-TRUTH PATH between the two
        # nodes (speed- and keyframe-density-independent); the time bound still applies if set
        trivial = (gap < min_gap_s) or (pgap is not None and pgap < min_gap_m)
        wrong = err["trans_err_m"] > WRONG_LINK_TRANS_M or err["rot_err_deg"] > WRONG_LINK_ROT_DEG
        chain = gt_chain_stats(gt_by_id)
        cg = ({"rot_gap_deg": abs(chain[l.b][1] - chain[l.a][1]), "kf_gap": abs(chain[l.b][2] - chain[l.a][2])}
              if (l.a in chain and l.b in chain) else {"rot_gap_deg": None, "kf_gap": None})
        rows.append({"a": l.a, "b": l.b, "time_gap_s": float(gap), "path_gap_m": pgap, **cg,
                     "trivial": bool(trivial), "wrong": bool(wrong), **err})
    n = len(rows)
    n_trivial = sum(1 for r in rows if r["trivial"])
    n_wrong = sum(1 for r in rows if r["wrong"])
    n_real = sum(1 for r in rows if (not r["trivial"]) and (not r["wrong"]))
    gaps = [r["time_gap_s"] for r in rows]
    return {"n": n, "n_trivial": n_trivial, "n_wrong": n_wrong, "n_real": n_real,
            "rows": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                      if k in ("a", "b", "time_gap_s", "path_gap_m", "rot_gap_deg", "kf_gap", "trivial", "wrong",
                               "trans_err_m")} for r in rows],
            "min_gap_s_threshold": min_gap_s, "min_gap_m_threshold": min_gap_m,
            "path_gap_m": ({"min": float(min(r["path_gap_m"] for r in rows if r["path_gap_m"] is not None)),
                             "max": float(max(r["path_gap_m"] for r in rows if r["path_gap_m"] is not None))}
                            if any(r["path_gap_m"] is not None for r in rows) else None),
            "time_gap_s": ({"min": float(min(gaps)), "median": float(np.median(gaps)), "max": float(max(gaps))}
                            if gaps else None),
            "max_trans_err_m": float(max((r["trans_err_m"] for r in rows), default=0.0))}


def vertical_horizontal_error(est_T: list, gt_T: list) -> dict:
    """Position error split into vertical (GT world z) and horizontal,
    after aligning the estimate to ground truth by the FIRST pose only
    (same anchoring as metrics.anchored_ate: no least-squares freedom to
    hide drift). Synthetic fixtures are z-up (tests/synth/world.py), so
    GT world z is true gravity-vertical. est_T/gt_T: lists of world<-cam."""
    assert len(est_T) == len(gt_T) >= 1
    A = gt_T[0] @ lie.se3_inverse(est_T[0])
    est_pos = np.array([(A @ T)[:3, 3] for T in est_T])
    gt_pos = np.array([T[:3, 3] for T in gt_T])
    e = est_pos - gt_pos
    vert, horiz = e[:, 2], np.linalg.norm(e[:, :2], axis=1)
    return {"vert_err_rms_m": float(np.sqrt(np.mean(vert ** 2))),
            "vert_err_max_abs_m": float(np.max(np.abs(vert))),
            "horiz_err_rms_m": float(np.sqrt(np.mean(horiz ** 2))),
            "height_range_gt_m": float(gt_pos[:, 2].max() - gt_pos[:, 2].min()),
            "height_range_est_m": float(est_pos[:, 2].max() - est_pos[:, 2].min())}


def bridge_audit(links: list, gt_by_id: dict) -> dict:
    """For every identity 'bridge' link (added across a LOST gap), how far
    did the camera REALLY move between its two nodes? The bridge asserts
    zero motion; this is the size of that lie."""
    trans, rot = [], []
    for l in links:
        if l.kind != "bridge":
            continue
        err = link_gt_error(l, gt_by_id)
        if err is None:
            continue
        # measured T_ab is identity, so its error IS the true motion
        trans.append(err["true_trans_m"])
        rot.append(err["true_rot_deg"])
    return {"n": len(trans),
            "true_trans_m_sum": float(np.sum(trans)) if trans else 0.0,
            "true_trans_m_mean": float(np.mean(trans)) if trans else 0.0,
            "true_trans_m_max": float(np.max(trans)) if trans else 0.0,
            "true_rot_deg_mean": float(np.mean(rot)) if rot else 0.0,
            "true_rot_deg_max": float(np.max(rot)) if rot else 0.0}


def _kf_reason_counts(telemetry: list) -> dict:
    """WP-L3: how many keyframes each trigger created (a keyframe hit by two
    triggers counts once per trigger). Tells you WHICH threshold to move when
    the keyframe rate is too high, instead of guessing."""
    out: dict = {}
    for t in telemetry:
        r = t.get("kf_reason")
        if r:
            for part in r.split("+"):
                out[part] = out.get(part, 0) + 1
    return out


def _path_len(T_list: list) -> float:
    p = np.array([T[:3, 3] for T in T_list])
    return float(metrics.path_length_cumulative(p)[-1]) if len(p) > 1 else 0.0


# --------------------------------------------------------------------------
def run_one(scenario: str, seed: int, cfg: Config, backend: str = "auto",
            max_frames: Optional[int] = None, loop_gap_s: float = 0.0,
            with_map: bool = True, loop_gap_m: float = 1.5, do_finalize: bool = True) -> dict:
    from tests.synth.scenarios import build
    from tests.synth.world import T_BODY_CAM

    scen = build(scenario, seed=seed, validate=False)  # fixtures are validated by WP-A2's own gate
    source = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=seed)
    cache_dir = tempfile.mkdtemp(prefix="pyslam_pa_")
    try:
        pipe = Pipeline(cfg, vocab_path=None, backend_prefer=backend, R_body_cam=T_BODY_CAM[:3, :3],
                        imagery_cache_dir=os.path.join(cache_dir, "img"))
        t0 = time.perf_counter()
        result = pipe.run(source, max_frames=max_frames, verbose=False)
        wall_s = time.perf_counter() - t0

        node_ids = sorted(result.node_gt.keys())
        if len(node_ids) < 3:
            return {"scenario": scenario, "seed": seed, "error": "fewer than 3 keyframes with ground truth"}
        gt_T = [result.node_gt[i] @ T_BODY_CAM for i in node_ids]   # world<-cam
        gt_by_id = dict(zip(node_ids, gt_T))
        odom_T = [pipe.memory.get(i).pose_odom for i in node_ids]
        online_T = [pipe.memory.get(i).pose_map for i in node_ids]   # BEFORE finalize
        session_ids = [pipe.memory.get(i).session_id for i in node_ids]
        node_t = {i: pipe.memory.get(i).sig.t for i in pipe.memory.all_node_ids()}

        pos = lambda Ts: np.array([T[:3, 3] for T in Ts])
        gt_pos = pos(gt_T)
        ate_odom = ate_rmse(pos(odom_T), gt_pos)
        ate_online = ate_rmse(pos(online_T), gt_pos)
        odom_links = [l for l in pipe.graph.links if l.kind == "odom"]
        rpe_link = metrics.per_link_rpe_vs_gt(odom_links, gt_by_id)
        path_gt_cum = metrics.path_length_cumulative(gt_pos)
        rpe_dist = metrics.rpe_by_distance(odom_T, gt_T, path_gt_cum, segment_m=1.0, session_ids=session_ids)

        loops = [l for (_, _, l) in result.loop_events]
        prox = [l for (_, _, l) in result.proximity_events]
        bridges = [l for l in pipe.graph.links if l.kind == "bridge"]
        loop_a = loop_audit(loops, gt_by_id, node_t, loop_gap_s, loop_gap_m)
        prox_a = loop_audit(prox, gt_by_id, node_t, loop_gap_s, loop_gap_m)
        revisit = revisit_recall(gt_by_id, loops, loop_gap_m)
        bridge_a = bridge_audit(bridges, gt_by_id)

        # closing optimisation -> what the exported trajectory/map really are
        finalized, final_T, ate_final, anch_final, vh_final = False, None, None, None, None
        vh_online = vertical_horizontal_error(online_T, gt_T)   # WP-L: always available, needs no finalize
        try:
            if not do_finalize:
                raise RuntimeError("finalize skipped (--no-finalize)")
            pipe.finalize(result)
            finalized = True
            fin_ids = [i for i in node_ids if i in result.final_poses]
            if len(fin_ids) >= 3:
                final_T = [result.final_poses[i] for i in fin_ids]
                fin_gt = [gt_by_id[i] for i in fin_ids]
                ate_final = ate_rmse(pos(final_T), pos(fin_gt))
                anch_final = metrics.anchored_ate(final_T, fin_gt)
                vh_final = vertical_horizontal_error(final_T, fin_gt)
        except Exception as e:  # noqa: BLE001 -- record, keep going
            if do_finalize:
                log.warning(f"finalize failed for {scenario}/seed{seed}: {type(e).__name__}: {e}")

        map_stats = None
        if with_map:
            try:
                nodes = [pipe.memory.get(i) for i in pipe.memory.all_node_ids()]
                _, _, map_stats = assemble_cloud_with_stats(
                    nodes, scen.intr.K(), scen.intr.depth_scale, imagery_loader=pipe.memory.load_imagery)
            except Exception as e:  # noqa: BLE001
                map_stats = {"error": f"{type(e).__name__}: {e}"}

        status = result.status_log
        return {
            "scenario": scenario, "seed": seed,
            "graph_backend": type(pipe.graph.backend).__name__,
            "n_frames": result.n_frames, "n_keyframes": result.n_keyframes,
            "wall_s": wall_s, "ms_per_frame": 1000.0 * wall_s / max(result.n_frames, 1),
            "n_lost_events": sum(1 for l in status if "LOST" in l and "bridge" not in l),
            "n_bridge_links": sum(1 for l in status if "bridge link" in l),
            "n_sessions": len(set(session_ids)),
            "n_cross_session_merges": len(result.cross_session_merges),
            "loops": loop_a, "proximity": prox_a, "bridges": bridge_a, "revisit": revisit,
            "ate_odom_cm": None if ate_odom is None else ate_odom * 100,
            "ate_online_cm": None if ate_online is None else ate_online * 100,
            "ate_final_cm": None if ate_final is None else ate_final * 100,
            "anchored_ate_odom_cm": metrics.anchored_ate(odom_T, gt_T) * 100,
            "anchored_ate_online_cm": metrics.anchored_ate(online_T, gt_T) * 100,
            "anchored_ate_final_cm": None if anch_final is None else anch_final * 100,
            "rpe_link_trans_mean_cm": rpe_link.get("trans_err_mean_m", 0.0) * 100,
            "trans_drift_pct": rpe_dist["trans_drift_pct"],
            "rot_drift_deg_per_m": rpe_dist["rot_drift_deg_per_m"],
            "n_rpe_segments_skipped_cross_session": rpe_dist.get("n_skipped_cross_session"),
            "path_length_gt_m": _path_len(gt_T),
            "path_length_est_odom_m": _path_len(odom_T),
            "path_length_est_final_m": _path_len(final_T) if final_T else None,
            "vertical_odom": vertical_horizontal_error(odom_T, gt_T),
            "vertical_final": vh_final,
            "vertical_online": vh_online,
            "finalized": finalized,
            "map": map_stats,
            "timing": summarize_telemetry(result.telemetry, cfg.wm_budget_ms),
            "keyframe_reasons": _kf_reason_counts(result.telemetry),
            "keyframes_per_metre": (result.n_keyframes / _path_len(gt_T)) if _path_len(gt_T) > 0 else None,
            "wm_oldest_node_id_at_end": (min(pipe.memory.working_set()) if pipe.memory.working_set() else None),
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Aggregation / reporting
# --------------------------------------------------------------------------
def _get(run: dict, path: str):
    cur = run
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# (dotted path in a run dict, column header, format)
REPORT_COLUMNS = [
    ("n_keyframes", "kf", "{:.0f}"),
    ("n_lost_events", "LOST", "{:.0f}"),
    ("loops.n", "loops", "{:.0f}"),
    ("loops.n_real", "real", "{:.0f}"),
    ("loops.n_trivial", "trivial", "{:.0f}"),
    ("loops.n_wrong", "wrong", "{:.0f}"),
    ("revisit.n_opportunity_frames", "revisit_opp", "{:.0f}"),
    ("revisit.recall", "recall", "{:.2f}"),
    ("proximity.n", "prox", "{:.0f}"),
    ("proximity.n_wrong", "prox_wrong", "{:.0f}"),
    ("anchored_ate_odom_cm", "aATE_odom", "{:.1f}"),
    ("anchored_ate_online_cm", "aATE_online", "{:.1f}"),
    ("anchored_ate_final_cm", "aATE_final", "{:.1f}"),
    ("ate_final_cm", "ATE_final", "{:.1f}"),
    ("rot_drift_deg_per_m", "rot_deg/m", "{:.3f}"),
    ("trans_drift_pct", "trans_%", "{:.2f}"),
    ("vertical_final.vert_err_rms_m", "vert_rms_m", "{:.3f}"),
    ("vertical_final.horiz_err_rms_m", "horiz_rms_m", "{:.3f}"),
    ("bridges.true_trans_m_sum", "bridge_lie_m", "{:.2f}"),
    ("path_length_gt_m", "path_gt_m", "{:.1f}"),
    ("path_length_est_final_m", "path_est_m", "{:.1f}"),
    ("map.n_nodes_in_map", "map_kf", "{:.0f}"),
    ("map.n_nodes_resident", "map_kf_old", "{:.0f}"),
    ("timing.keyframe_mem_duration_ms.p95", "mem_p95_ms", "{:.0f}"),
    ("timing.effective_fps", "fps", "{:.1f}"),
    ("timing.wm_size.max", "wm_max", "{:.0f}"),
    ("keyframe_reasons.inliers", "kf_by_inl", "{:.0f}"),
    ("keyframe_reasons.trans", "kf_by_tr", "{:.0f}"),
    ("keyframe_reasons.rot", "kf_by_rot", "{:.0f}"),
    ("keyframes_per_metre", "kf/m", "{:.1f}"),
    ("wm_oldest_node_id_at_end", "wm_oldest", "{:.0f}"),
]


def aggregate(runs: list) -> dict:
    """{scenario: {column_path: {"median","min","max","n"}}} over seeds."""
    out: dict = {}
    for scen in sorted({r["scenario"] for r in runs}):
        rs = [r for r in runs if r["scenario"] == scen and "error" not in r]
        agg = {}
        for path, _, _ in REPORT_COLUMNS:
            vals = [v for v in (_get(r, path) for r in rs) if isinstance(v, (int, float))]
            if vals:
                agg[path] = {"median": float(np.median(vals)), "min": float(min(vals)),
                             "max": float(max(vals)), "n": len(vals)}
        out[scen] = agg
    return out


def format_table(agg: dict) -> str:
    heads = ["scenario"] + [h for _, h, _ in REPORT_COLUMNS]
    rows = []
    for scen, a in agg.items():
        row = [scen]
        for path, _, fmt in REPORT_COLUMNS:
            row.append(fmt.format(a[path]["median"]) if path in a else "-")
        rows.append(row)
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(heads)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(heads), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows])


def compare(path_a: str, path_b: str) -> str:
    with open(path_a) as f:
        A = json.load(f)
    with open(path_b) as f:
        B = json.load(f)
    lines = [f"A = {A['label']} (overrides {A.get('config_overrides')})",
             f"B = {B['label']} (overrides {B.get('config_overrides')})",
             "median over seeds; ratio = B/A (lower is better for error/time columns)\n"]
    for scen in sorted(set(A["aggregate"]) & set(B["aggregate"])):
        lines.append(f"[{scen}]")
        for path, head, fmt in REPORT_COLUMNS:
            a, b = A["aggregate"][scen].get(path), B["aggregate"][scen].get(path)
            if a is None or b is None:
                continue
            ratio = (b["median"] / a["median"]) if abs(a["median"]) > 1e-12 else float("nan")
            lines.append(f"  {head:<13} {fmt.format(a['median']):>10} -> {fmt.format(b['median']):>10}   x{ratio:.2f}")
    return "\n".join(lines)


def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--label", type=str, default=None, help="name for this baseline (file: baselines/<label>.json)")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto",
                    help="graph backend; keep it identical between runs you intend to compare")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--loop-gap-s", type=float, default=0.0,
                    help="loops between nodes captured closer than this many seconds are 'trivial' (0 = off)")
    ap.add_argument("--loop-gap-m", type=float, default=1.5,
                    help="loops between nodes closer than this many metres of GROUND-TRUTH path are 'trivial'; "
                         "also the min path for a revisit opportunity (WP-L)")
    ap.add_argument("--no-map", action="store_true", help="skip map-coverage measurement (faster)")
    ap.add_argument("--no-finalize", action="store_true",
                    help="skip the closing full-graph optimisation (ate_final/vertical_final stay empty; every "
                         "online/loop/recall number is unaffected). The pure-python native backend can take "
                         "hours to finalize a graph containing a real lap closure; GTSAM does not.")
    ap.add_argument("--out-dir", type=str, default="baselines")
    add_config_args(ap)
    args = ap.parse_args()

    if args.compare:
        print(compare(*args.compare))
        return 0
    if not args.label:
        ap.error("--label is required (or use --compare)")

    cfg = config_from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    runs = []
    for scen in args.scenarios:
        for seed in args.seeds:
            log.info(f"=== {scen} seed={seed} ===")
            try:
                r = run_one(scen, seed, cfg, args.backend, args.max_frames, args.loop_gap_s, not args.no_map, args.loop_gap_m, not args.no_finalize)
            except Exception as e:  # noqa: BLE001 -- one bad run must not lose the rest
                import traceback
                traceback.print_exc()
                r = {"scenario": scen, "seed": seed, "error": f"{type(e).__name__}: {e}"}
            runs.append(r)
            # write after EVERY run: these take minutes each, a crash at run 8/9 must not cost 1-7
            _write(args, cfg, runs)
    agg = aggregate(runs)
    print("\n" + format_table(agg))
    return 0


def _write(args, cfg: Config, runs: list) -> None:
    agg = aggregate(runs)
    doc = {"label": args.label, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "git_commit": _git_commit(), "config_overrides": args.config_override,
           "config_hash": cfg.hash(), "backend_requested": args.backend,
           "max_frames": args.max_frames, "loop_gap_s": args.loop_gap_s, "loop_gap_m": args.loop_gap_m,
           "runs": runs, "aggregate": agg}
    with open(os.path.join(args.out_dir, f"{args.label}.json"), "w") as f:
        json.dump(doc, f, indent=2, default=float)
    with open(os.path.join(args.out_dir, f"{args.label}.md"), "w") as f:
        f.write(f"# {args.label}\n\noverrides: `{args.config_override}`  commit: `{doc['git_commit']}`  "
                f"backend: `{args.backend}`\n\nmedian over seeds\n\n{format_table(agg)}\n")


if __name__ == "__main__":
    sys.exit(main())
