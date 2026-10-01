"""
WP-A3 baseline freeze (Phase 1 plan section 4, gate G1A.6).

Runs the CURRENT odometry (Phase 0's frame-to-keyframe PnP tracker, with
WP-A0's direction fix and WP-B4's graph-layer fixes applied, but BEFORE
WP-B1's frame-to-local-map upgrade) over the WP-A2 fixture set and stores
the result as the frozen baseline. Phase 1B's gate thresholds (Phase 1
plan section 6, G1B.1) are then min(absolute_target, 0.7 * this baseline)
-- this script computes both sides of that min and prints the frozen
number, so the bar WP-B1 has to clear is written down BEFORE that work
starts, not adjusted afterward to whatever it happens to achieve.

    python3 -m pyslam.tools.baseline                    # run + freeze
    python3 -m pyslam.tools.baseline --scenario square6dof --seeds 1 2 3

Output: pyslam/tools/baseline_store.json (checked into git; treat edits
to this file as seriously as any other frozen threshold).
"""
from __future__ import annotations
import argparse
import json
import os
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
from tests.synth.scenarios import build, ALL_SCENARIO_BUILDERS
from tests.synth.world import T_BODY_CAM

log = get_logger("baseline")

STORE_PATH = os.path.join(os.path.dirname(__file__), "baseline_store.json")

# corridor_v2 finding (WP-A3, seed=1, 220-frame capped run): trans_drift_pct
# came back at 18.3%/m -- an order of magnitude worse than every other
# fixture -- while per-link RPE stayed excellent (0.63cm mean). Traced
# this directly: 7 LOST events fired in 220 frames, and EVERY ONE
# produced a node with no "odom" link into it (confirmed 7 LOST == 7
# link-less nodes exactly). force_new_keyframe() re-anchors tracking
# locally but the pipeline has no concept of a NEW SESSION yet (that's
# WP-B3): it just keeps writing pose_odom into the same trajectory as if
# nothing happened, so each LOST silently inserts a discontinuity that
# naive cumulative drift metrics (trans_drift_pct) read as catastrophic
# odometry error, when the actual per-link tracking accuracy elsewhere
# is fine. This is strong direct evidence that WP-B3 (session-on-LOST)
# is not a nice-to-have refinement but the fix for corridor_v2's
# specific failure mode -- more urgent than previously ranked relative
# to WP-B1's tracking-accuracy improvements, which this data suggests
# are not actually the bottleneck here.

# Absolute Phase 1B targets from the plan's gate table (G1B.1), used as
# the OTHER side of the min() -- see this module's docstring.
ABSOLUTE_TARGETS = {
    "trans_drift_pct_median": 1.0,
    "trans_drift_pct_worst": 2.0,
    "rot_drift_deg_per_m": 0.3,
}
# corridor_v2 is explicitly allowed a looser absolute target per the plan
ABSOLUTE_TARGETS_CORRIDOR = {
    "trans_drift_pct_median": 1.5,
    "trans_drift_pct_worst": 3.0,
    "rot_drift_deg_per_m": 0.4,
}


def run_one(scenario_name: str, seed: int, max_frames: int = None) -> dict:
    scen = build(scenario_name, seed=seed, validate=False)  # already validated once in WP-A2; skip re-validating here for speed
    source = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=seed)
    cfg = Config()
    pipe = Pipeline(cfg, backend_prefer="native")
    t0 = time.perf_counter()
    result = pipe.run(source, max_frames=max_frames, verbose=False)
    wall_s = time.perf_counter() - t0

    node_ids = sorted(result.node_gt.keys())
    if len(node_ids) < 3:
        return {"seed": seed, "error": "too few keyframes with ground truth to score"}
    est_odom_T = [pipe.memory.get(i).pose_odom for i in node_ids]
    gt_T = [result.node_gt[i] @ T_BODY_CAM for i in node_ids]
    session_ids = [pipe.memory.get(i).session_id for i in node_ids]
    est_pos = np.array([T[:3, 3] for T in est_odom_T])
    gt_pos = np.array([T[:3, 3] for T in gt_T])

    ate_odom = ate_rmse(est_pos, gt_pos)
    anchored = metrics.anchored_ate(est_odom_T, gt_T)
    mirror = metrics.mirror_check(est_pos, gt_pos, anchored, ate_odom)
    path_len = metrics.path_length_cumulative(gt_pos)
    rpe_dist = metrics.rpe_by_distance(est_odom_T, gt_T, path_len, segment_m=1.0, session_ids=session_ids)
    odom_links = [l for l in pipe.graph.links if l.kind == "odom"]
    rpe_link = metrics.per_link_rpe_vs_gt(odom_links, dict(zip(node_ids, gt_T)))

    n_frames = result.n_frames
    return {
        "seed": seed,
        "n_frames": n_frames,
        "max_frames_cap": max_frames,
        "n_keyframes": result.n_keyframes,
        "n_loop_closures": len(result.loop_events),
        "wall_s": wall_s,
        "ms_per_frame": 1000.0 * wall_s / max(n_frames, 1),
        "n_lost_events": len(result.status_log),
        "ate_odom_cm": ate_odom * 100,
        "anchored_ate_odom_cm": anchored * 100,
        "mirror_suspected": mirror["mirror_suspected"],
        "trans_drift_pct": rpe_dist["trans_drift_pct"],
        "rot_drift_deg_per_m": rpe_dist["rot_drift_deg_per_m"],
        "rpe_link_trans_mean_cm": rpe_link.get("trans_err_mean_m", 0.0) * 100,
        "rpe_link_trans_max_cm": rpe_link.get("trans_err_max_m", 0.0) * 100,
        "n_cross_session_merges": len(result.cross_session_merges),
        "n_sessions": len(set(session_ids)),
    }


def freeze_thresholds(runs: list[dict], targets: dict) -> dict:
    drifts = [r["trans_drift_pct"] for r in runs if "error" not in r]
    rots = [r["rot_drift_deg_per_m"] for r in runs if "error" not in r]
    if not drifts:
        return {"error": "no successful runs"}
    median_drift, worst_drift = float(np.median(drifts)), float(np.max(drifts))
    median_rot = float(np.median(rots))
    return {
        "baseline_trans_drift_pct_median": median_drift,
        "baseline_trans_drift_pct_worst": worst_drift,
        "baseline_rot_drift_deg_per_m_median": median_rot,
        "frozen_gate_trans_drift_pct_median": min(targets["trans_drift_pct_median"], 0.7 * median_drift),
        "frozen_gate_trans_drift_pct_worst": min(targets["trans_drift_pct_worst"], 0.7 * worst_drift),
        "frozen_gate_rot_drift_deg_per_m": min(targets["rot_drift_deg_per_m"], 0.7 * median_rot),
    }


def run_static_drift(scenario_name: str, seed: int) -> dict:
    """static_60s never crosses the keyframe-creation motion threshold
    (that's the correct behaviour for a genuinely stationary camera), so
    Pipeline.run()'s keyframe-gated ground-truth bookkeeping only ever
    sees ONE ground-truthed node -- not enough to measure drift against.
    Drift is exactly what this fixture exists to measure (see the
    Phase 1 plan's desk_static bag protocol: drift <=1cm/<=0.3deg over
    60s), so this drives VisualOdometry directly, frame by frame,
    bypassing the keyframe/graph/loop-closure machinery entirely and
    comparing odometry.cur_pose against frame.gt_pose on every frame."""
    from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
    from pyslam.frontend.odometry import VisualOdometry
    from pyslam.core import lie

    scen = build(scenario_name, seed=seed, validate=False)
    source = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=seed)
    cfg = Config()
    orb = make_orb(cfg)
    _reset_id_counter(0)
    odo = VisualOdometry(cfg)

    t0 = time.perf_counter()
    trans_errs, rot_errs, n_lost = [], [], 0
    T_wc0_true = None
    for frame in source:
        sig = extract_signature(frame, cfg, orb)
        res = odo.update(sig, frame)
        if odo.status == "LOST":
            n_lost += 1
        T_wc_true = frame.gt_pose @ T_BODY_CAM if frame.gt_pose is not None else None
        if T_wc_true is None:
            continue
        if T_wc0_true is None:
            T_wc0_true = T_wc_true
        # odometry's own origin is its first frame; compare RELATIVE
        # drift from that first frame onward, same convention as the
        # convention oracle in tests/gates/test_g0.py.
        T_rel_true = lie.se3_inverse(T_wc0_true) @ T_wc_true
        err = lie.se3_log(lie.se3_inverse(T_rel_true) @ res.pose)
        trans_errs.append(np.linalg.norm(err[:3]))
        rot_errs.append(np.degrees(np.linalg.norm(err[3:])))
    wall_s = time.perf_counter() - t0
    if not trans_errs:
        return {"seed": seed, "error": "no ground-truthed frames"}
    return {
        "seed": seed,
        "n_frames": len(trans_errs),
        "wall_s": wall_s,
        "ms_per_frame": 1000.0 * wall_s / len(trans_errs),
        "n_lost_events": n_lost,
        "final_drift_trans_cm": trans_errs[-1] * 100,
        "final_drift_rot_deg": rot_errs[-1],
        "max_drift_trans_cm": max(trans_errs) * 100,
        "max_drift_rot_deg": max(rot_errs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(ALL_SCENARIO_BUILDERS.keys()) + ["all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    # Default seed counts are deliberately uneven: corridor_v2 costs
    # ~2-3 minutes per seed to render+validate+run (24 walls vs 6, plus
    # 4x the frame count of the cheapest fixture), so this script uses
    # fewer seeds there rather than making every baseline run take an
    # impractical amount of time. This is an honest limitation, not a
    # hidden one -- see the run count recorded per fixture in the output
    # JSON, and treat corridor_v2's baseline as lower-confidence until a
    # fuller sweep is run as a background/full-tier job.
    default_seeds = {
        "square6dof": [1, 2, 3, 4, 5],
        "room_orbit": [1, 2, 3, 4, 5],
        "corridor_v2": [1, 2],
        "static_60s": [1, 2, 3],
        "aliasing_rooms": [1, 2, 3],
    }

    # WP-L: "all" means the five FROZEN fixtures the store was built from -- fixtures added later
    # (e.g. corridor_lap13) must not silently start writing into the frozen baseline.
    FROZEN = ("square6dof", "room_orbit", "static_60s", "aliasing_rooms", "corridor_v2")
    scenarios = [n for n in ALL_SCENARIO_BUILDERS if n in FROZEN] if args.scenario == "all" else [args.scenario]
    store = {}
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH) as f:
            store = json.load(f)

    for name in scenarios:
        seeds = args.seeds if args.seeds is not None else default_seeds[name]
        log.info(f"--- {name}: seeds {seeds} ---")

        if name == "static_60s":
            runs = [run_static_drift(name, seed) for seed in seeds]
            for r in runs:
                if "error" in r:
                    log.warning(f"  seed={r['seed']}: {r['error']}")
                else:
                    log.info(f"  seed={r['seed']}: final_drift={r['final_drift_trans_cm']:.2f}cm/"
                             f"{r['final_drift_rot_deg']:.2f}deg (max {r['max_drift_trans_cm']:.2f}cm/"
                             f"{r['max_drift_rot_deg']:.2f}deg) lost={r['n_lost_events']} "
                             f"ms/frame={r['ms_per_frame']:.0f}")
            ok_runs = [r for r in runs if "error" not in r]
            frozen = {
                "baseline_final_drift_trans_cm_median": float(np.median([r["final_drift_trans_cm"] for r in ok_runs])) if ok_runs else None,
                "baseline_final_drift_rot_deg_median": float(np.median([r["final_drift_rot_deg"] for r in ok_runs])) if ok_runs else None,
                "absolute_target_note": "Phase 1 plan hardware gate: desk_static drift <=1cm/<=0.3deg over 60s -- "
                                         "this synthetic run is the pre-hardware sanity check for that same bound.",
            }
            store[name] = {"runs": runs, "frozen_thresholds": frozen,
                            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "code_commit_note": "Phase 0 F2F odometry + WP-A0/WP-B4 fixes, pre-WP-B1. "
                                                 "Evaluated via run_static_drift (bypasses keyframe gating -- see its docstring)."}
            log.info(f"  FROZEN: {json.dumps(frozen, indent=2)}")
            continue

        runs = []
        for seed in seeds:
            # corridor_v2 costs ~1s/frame (24 walls vs 6, plus growing
            # retrieval cost over a longer WM -- see the Phase 1 plan's
            # F5 finding) and a full 480-frame run does not complete in
            # this environment's interactive tool timeout even once;
            # capped here at 220 frames (a bit under half the loop, no
            # closure) so the baseline at least captures real raw-
            # odometry drift numbers rather than nothing. Documented,
            # not hidden -- see max_frames_cap in the stored result and
            # treat this fixture's baseline as the least-confident one
            # in the store until a full run is done as an offline job.
            max_frames = 220 if name == "corridor_v2" else None
            r = run_one(name, seed, max_frames=max_frames)
            runs.append(r)
            if "error" in r:
                log.warning(f"  seed={seed}: {r['error']}")
            else:
                log.info(f"  seed={seed}: drift={r['trans_drift_pct']:.2f}%/m rot={r['rot_drift_deg_per_m']:.3f}deg/m "
                         f"ATE(anchored)={r['anchored_ate_odom_cm']:.2f}cm loops={r['n_loop_closures']} "
                         f"lost={r['n_lost_events']} ms/frame={r['ms_per_frame']:.0f} mirror={r['mirror_suspected']}")
        targets = ABSOLUTE_TARGETS_CORRIDOR if name == "corridor_v2" else ABSOLUTE_TARGETS
        frozen = freeze_thresholds(runs, targets)
        store[name] = {"runs": runs, "frozen_thresholds": frozen,
                        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "code_commit_note": "Phase 0 F2F odometry + WP-A0/WP-B4 fixes, pre-WP-B1"}
        log.info(f"  FROZEN: {json.dumps(frozen, indent=2)}")

    with open(STORE_PATH, "w") as f:
        json.dump(store, f, indent=2, default=float)
    log.info(f"Baseline written to {STORE_PATH}")


if __name__ == "__main__":
    main()
