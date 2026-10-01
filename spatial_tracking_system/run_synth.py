"""
Run the pipeline against a synthetic scene -- no hardware, no recorded
bag needed.

    python3 run_synth.py --scenario square6dof
    python3 run_synth.py --scenario corridor_v2
    python3 run_synth.py --scenario room_orbit
    python3 run_synth.py --scenario static_60s
    python3 run_synth.py --scenario aliasing_rooms

WP-A2 note: scenarios now come from tests/synth/scenarios.py, which
validates every fixture's geometry (free space, valid depth, motion
smoothness) before handing it back -- see that module's docstring for
the two Phase-0 fixture defects (F3, F4) this replaces. The old
square/corridor generators lived in this file directly; they're gone.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import sys
import numpy as np

sys.path.insert(0, ".")

from pyslam.core.config import Config, add_config_args, config_from_args
from pyslam.core.log import get_logger
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from pyslam.tools.evaluate import ate_rmse
from pyslam.tools import metrics
from pyslam.tools.run_outputs import finalize_and_export, summarize_telemetry
from tests.synth.scenarios import build, ALL_SCENARIO_BUILDERS
from tests.synth.world import T_BODY_CAM

log = get_logger("run_synth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(ALL_SCENARIO_BUILDERS.keys()), default="square6dof")
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-validate", action="store_true", help="skip the scenario validity gate (debugging only)")
    add_config_args(ap)  # WP-K2: --config-override KEY=VALUE (repeatable)
    ap.add_argument("--keep-imagery-cache", action="store_true",
                     help="WP-K3: keep <run_dir>/imagery_cache/ instead of deleting it after map.ply is written")
    args = ap.parse_args()

    run_dir = os.path.join("runs", f"run_{int(time.time())}_synth_{args.scenario}")
    os.makedirs(run_dir, exist_ok=True)

    scen = build(args.scenario, seed=args.seed, validate=not args.no_validate)
    source = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=args.seed)

    cfg = config_from_args(args)
    cfg.save(os.path.join(run_dir, "config.json"))
    if args.config_override:
        log.info(f"Config overrides in effect: {args.config_override}")
    # WP-T3: same R_body_cam convention test_g5.py's oracle gate already
    # validates (T_BODY_CAM[:3,:3]) -- synthetic fixtures' IMU is
    # gravity-reaction-only (see WP-P5.0's finding), but that's still
    # enough for a plausible static-hold gravity-alignment export on any
    # scenario with a still start.
    pipeline = Pipeline(cfg, vocab_path=None, backend_prefer=args.backend, R_body_cam=T_BODY_CAM[:3, :3],
                        imagery_cache_dir=os.path.join(run_dir, "imagery_cache"))  # WP-K3
    result = pipeline.run(source, verbose=True)

    node_ids = sorted(result.node_gt.keys())
    est_odom_T = [pipeline.memory.get(i).pose_odom for i in node_ids]
    est_map_T = [pipeline.memory.get(i).pose_map for i in node_ids]
    session_ids = [pipeline.memory.get(i).session_id for i in node_ids]
    # ground truth is body<-world; convert to camera frame to match est_*, which are world<-cam
    gt_T = [result.node_gt[i] @ T_BODY_CAM for i in node_ids]

    est_odom = np.array([T[:3, 3] for T in est_odom_T])
    est_map = np.array([T[:3, 3] for T in est_map_T])
    gt = np.array([T[:3, 3] for T in gt_T])
    ate_odom = ate_rmse(est_odom, gt)
    ate_map = ate_rmse(est_map, gt)
    anchored_ate_odom = metrics.anchored_ate(est_odom_T, gt_T)
    anchored_ate_map = metrics.anchored_ate(est_map_T, gt_T)
    # WP-T0: ate_rmse returns None below 3 keyframes (Umeyama is
    # undefined there) instead of raising -- static_60s (1 keyframe, by
    # design) used to crash the whole run here with an AssertionError.
    if ate_odom is None or len(node_ids) < 3:
        mirror = {"anchored_ate_m": anchored_ate_odom, "umeyama_ate_m": None,
                   "ratio": None, "mirror_suspected": False,
                   "note": "fewer than 3 keyframes -- Umeyama ATE and mirror check are undefined"}
    else:
        mirror = metrics.mirror_check(est_odom, gt, anchored_ate_odom, ate_odom)

    odom_links = [l for l in pipeline.graph.links if l.kind == "odom"]
    gt_by_id = dict(zip(node_ids, gt_T))
    rpe_link = metrics.per_link_rpe_vs_gt(odom_links, gt_by_id)
    path_len = metrics.path_length_cumulative(gt)
    rpe_dist = metrics.rpe_by_distance(est_odom_T, gt_T, path_len, segment_m=1.0, session_ids=session_ids)

    summary = {
        "scenario": args.scenario,
        "seed": args.seed,
        "n_frames": result.n_frames,
        "n_keyframes": result.n_keyframes,
        "n_loop_closures": len(result.loop_events),
        "ate_odom_cm": (ate_odom * 100) if ate_odom is not None else None,
        "ate_graph_cm": (ate_map * 100) if ate_map is not None else None,
        "anchored_ate_odom_cm": anchored_ate_odom * 100,
        "anchored_ate_graph_cm": anchored_ate_map * 100,
        "mirror_check": mirror,
        "rpe_link_odom": {k: v for k, v in rpe_link.items() if k != "per_link"},
        "rpe_by_distance_odom": rpe_dist,
        # NOTE (WP-K1): every *_graph_* number above is the ONLINE pose_map -- the
        # graph as it stood when the run ended, BEFORE pipeline.finalize()'s
        # closing optimisation. Kept exactly as-is so numbers stay comparable
        # with every earlier baseline; the post-finalize equivalents (what the
        # exported trajectory and map actually use) are added below as *_final_*.
        "config_overrides": args.config_override,
        "timing": summarize_telemetry(result.telemetry, cfg.wm_budget_ms),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    log.info(f"Scenario: {args.scenario} (seed={args.seed})")
    log.info(f"Frames: {result.n_frames}  Keyframes: {result.n_keyframes}  "
             f"Loop closures: {len(result.loop_events)}")
    if ate_odom is not None:
        log.info(f"ATE (Umeyama) odom-only: {ate_odom*100:.2f}cm | after graph: {ate_map*100:.2f}cm")
    else:
        log.info(f"ATE (Umeyama): not enough keyframes ({len(node_ids)}) to score -- need >= 3")
    log.info(f"ATE (anchored) odom-only: {anchored_ate_odom*100:.2f}cm | after graph: {anchored_ate_map*100:.2f}cm "
             f"(mirror_suspected={mirror['mirror_suspected']})")
    log.info(f"RPE/link (odom): mean={rpe_link.get('trans_err_mean_m', 0)*100:.2f}cm "
             f"max={rpe_link.get('trans_err_max_m', 0)*100:.2f}cm")
    log.info(f"RPE/distance (odom): {rpe_dist['trans_drift_pct']:.2f}% trans drift, "
             f"{rpe_dist['rot_drift_deg_per_m']:.3f}deg/m rot drift")

    # WP-K1/K3: finalize -> map -> trajectory via the shared implementation
    # (pyslam/tools/run_outputs.py). The map used to be written BEFORE
    # finalize(), from pre-finalize poses, and without any keyframe that had
    # been evicted to LTM.
    summary_path = os.path.join(run_dir, "summary.json")
    out = finalize_and_export(run_dir, pipeline, result, scen.intr, summary_path=summary_path,
                               keep_imagery_cache=args.keep_imagery_cache)

    # Post-finalize accuracy: what the exported trajectory/map actually are.
    if out["finalized"] and result.final_poses:
        fin_ids = [i for i in node_ids if i in result.final_poses]
        if len(fin_ids) >= 3:
            fin_T = [result.final_poses[i] for i in fin_ids]
            fin_gt_T = [gt_by_id[i] for i in fin_ids]
            fin_pos = np.array([T[:3, 3] for T in fin_T])
            fin_gt_pos = np.array([T[:3, 3] for T in fin_gt_T])
            ate_fin = ate_rmse(fin_pos, fin_gt_pos)
            anch_fin = metrics.anchored_ate(fin_T, fin_gt_T)
            with open(summary_path) as f:
                summ = json.load(f)
            summ["ate_final_cm"] = (ate_fin * 100) if ate_fin is not None else None
            summ["anchored_ate_final_cm"] = anch_fin * 100
            with open(summary_path, "w") as f:
                json.dump(summ, f, indent=2, default=float)
            log.info(f"ATE after finalize() (the trajectory/map you actually get): "
                     f"Umeyama {ate_fin*100:.2f}cm | anchored {anch_fin*100:.2f}cm")

    log.info(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
