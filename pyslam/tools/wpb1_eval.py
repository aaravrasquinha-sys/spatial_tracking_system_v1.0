"""
WP-B1 continued: f2f-vs-f2m evaluation, run AFTER the landmark-fusion +
velocity-ordering fixes (see odometry_f2m.py's WP-B1-continued comments
and WP_B1_Findings.md).

Deliberately a SEPARATE tool from pyslam/tools/baseline.py, not an edit
to it: baseline_store.json is the frozen WP-A3 reference other work is
measured against ("treat edits to this file as seriously as any other
frozen threshold" -- baseline.py's own docstring). WP-B1's own
before/after comparison doesn't belong inside that frozen artifact.

Closes a real blind spot along the way: baseline.py's run_one() only
ever scored pose_odom (pre-graph) accuracy -- WP-B2's own findings
flagged this as "exactly the blind spot that almost let the [bad]
wiring ship before an end-to-end check caught the problem". This tool
scores BOTH pose_odom and post-graph pose_map against ground truth, for
both odometry backends, so a decision to flip cfg.odometry_backend has
the same end-to-end evidence WP-P4's flags were held to.

    python3 -m pyslam.tools.wpb1_eval --scenario square6dof --seeds 1 2 3 4 5
    python3 -m pyslam.tools.wpb1_eval --scenario corridor_v2 --seeds 1 --max-frames 220
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import numpy as np

sys.path.insert(0, ".")

from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from pyslam.tools.evaluate import ate_rmse
from pyslam.tools import metrics
from tests.synth.scenarios import build
from tests.synth.world import T_BODY_CAM

log = get_logger("wpb1_eval")


def run_one(scenario_name: str, seed: int, backend: str, max_frames=None) -> dict:
    scen = build(scenario_name, seed=seed, validate=False)
    source = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=seed)
    cfg = Config(odometry_backend=backend)
    pipe = Pipeline(cfg, backend_prefer="native")
    t0 = time.perf_counter()
    result = pipe.run(source, max_frames=max_frames, verbose=False)
    wall_s = time.perf_counter() - t0

    node_ids = sorted(result.node_gt.keys())
    if len(node_ids) < 3:
        return {"seed": seed, "backend": backend, "error": "too few keyframes with ground truth"}

    gt_T = [result.node_gt[i] @ T_BODY_CAM for i in node_ids]
    gt_pos = np.array([T[:3, 3] for T in gt_T])
    session_ids = [pipe.memory.get(i).session_id for i in node_ids]
    path_len = metrics.path_length_cumulative(gt_pos)

    def score(attr: str) -> dict:
        est_T = [getattr(pipe.memory.get(i), attr) for i in node_ids]
        est_pos = np.array([T[:3, 3] for T in est_T])
        ate = ate_rmse(est_pos, gt_pos)
        anchored = metrics.anchored_ate(est_T, gt_T)
        mirror = metrics.mirror_check(est_pos, gt_pos, anchored, ate)
        rpe_dist = metrics.rpe_by_distance(est_T, gt_T, path_len, segment_m=1.0, session_ids=session_ids)
        return {
            "ate_cm": ate * 100,
            "anchored_ate_cm": anchored * 100,
            "mirror_suspected": mirror["mirror_suspected"],
            "trans_drift_pct": rpe_dist["trans_drift_pct"],
            "rot_drift_deg_per_m": rpe_dist["rot_drift_deg_per_m"],
        }

    out = {
        "seed": seed, "backend": backend,
        "n_frames": result.n_frames, "n_keyframes": result.n_keyframes,
        "n_loop_closures": len(result.loop_events),
        "n_lost_events": len(result.status_log),
        "wall_s": wall_s, "ms_per_frame": 1000.0 * wall_s / max(result.n_frames, 1),
        "pose_odom": score("pose_odom"),   # pre-graph -- what baseline.py measures today
        "pose_map": score("pose_map"),     # post-graph -- the blind spot WP-B2 flagged
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="square6dof")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--backends", nargs="+", default=["f2f", "f2m"], choices=["f2f", "f2m"])
    args = ap.parse_args()

    all_results = {}
    for backend in args.backends:
        rows = []
        for seed in args.seeds:
            r = run_one(args.scenario, seed, backend, max_frames=args.max_frames)
            rows.append(r)
            if "error" in r:
                log.warning(f"  [{backend}] seed={seed}: {r['error']}")
            else:
                log.info(f"  [{backend}] seed={seed}: "
                         f"odom_ATE={r['pose_odom']['anchored_ate_cm']:.2f}cm  "
                         f"map_ATE={r['pose_map']['anchored_ate_cm']:.2f}cm  "
                         f"drift={r['pose_odom']['trans_drift_pct']:.2f}%/m  "
                         f"loops={r['n_loop_closures']}  lost={r['n_lost_events']}  "
                         f"ms/frame={r['ms_per_frame']:.0f}")
        all_results[backend] = rows

    print("\n=== summary (median over seeds) ===")
    for backend, rows in all_results.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            print(f"{backend}: no successful runs")
            continue
        odom_ate = np.median([r["pose_odom"]["anchored_ate_cm"] for r in ok])
        map_ate = np.median([r["pose_map"]["anchored_ate_cm"] for r in ok])
        drift = np.median([r["pose_odom"]["trans_drift_pct"] for r in ok])
        lost = np.median([r["n_lost_events"] for r in ok])
        loops = np.median([r["n_loop_closures"] for r in ok])
        ms = np.median([r["ms_per_frame"] for r in ok])
        print(f"{backend:>4}: odom_ATE={odom_ate:6.2f}cm  map_ATE={map_ate:6.2f}cm  "
              f"drift={drift:5.2f}%/m  lost={lost:.0f}  loops={loops:.0f}  ms/frame={ms:.0f}")

    with open("wpb1_eval_result.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    log.info("Full results written to wpb1_eval_result.json")


if __name__ == "__main__":
    main()
