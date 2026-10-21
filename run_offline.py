"""
Offline mapper: second pass over a FINISHED run_bag.py run.

    python run_offline.py runs/run_XXXX_bag --bag ~/bags/room2.db3

What it does (no real-time limits, every keyframe is used):
  1. prints the evidence from the live run (how far apart its loop closures
     were, how loop and odometry links were weighted);
  2. re-weights the odometry links with physical uncertainties;
  3. searches EVERY pair of keyframes for loop closures (bag-of-words
     trained on this run + the pipeline's PnP verification + a consistency
     check) and solves the pose graph robustly, then looks again near the
     corrected poses (proximity rounds);
  4. writes the corrected trajectory (same file names the scoring script
     uses) into <run>/offline/;
  5. with --bag: re-reads the recording and fuses depth into one clean
     surface (TSDF mesh if open3d is installed, else a voxel-averaged cloud).

The live run is untouched; everything new goes into <run>/offline/.
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
from pyslam.core import lie
from pyslam.loop.verify import GeometricVerifier
from pyslam.offline import pgo
from pyslam.offline.loops import LoopParams, close_loops, loop_info
from pyslam.offline.rundir import RunData, write_tum

log = get_logger("run_offline")


def evidence(rd: RunData) -> dict:
    t_of = dict(zip(rd.ids, rd.kf_t))
    ev = {"keyframes": len(rd.ids), "frames": int(rd.frame_t.size)}
    kinds = {}
    for e in rd.g2o_edges:
        kinds.setdefault(e["kind"], []).append(e)
    ev["live_links"] = {k: len(v) for k, v in kinds.items()}
    loops = kinds.get("loop", []) + kinds.get("proximity", [])
    gaps = [abs(t_of[e["b"]] - t_of[e["a"]]) for e in loops if e["a"] in t_of and e["b"] in t_of]
    if gaps:
        ev["live_loop_time_gap_s"] = {"median": round(float(np.median(gaps)), 1),
                                      "max": round(float(np.max(gaps)), 1),
                                      "over_60s": int(sum(g > 60 for g in gaps))}
    for k in ("odom", "loop"):
        if kinds.get(k):
            d = np.array([np.diag(e["info"]) for e in kinds[k]])
            ev[f"live_{k}_info_median"] = {"translation": round(float(np.median(d[:, :3])), 2),
                                            "rotation": round(float(np.median(d[:, 3:])), 2)}
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bag", default=None, help="the recording, to build the fused map")
    ap.add_argument("--min-gap", type=float, default=20.0, help="loop closures only between keyframes this many s apart")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2, help="fuse every Nth frame")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--depth-trunc", type=float, default=3.0)
    ap.add_argument("--no-open3d", action="store_true")
    ap.add_argument("--odom-robust", type=float, default=0.0,
                    help=">0: let a few odometry links give way in the final solve (Cauchy c, chi2 units)")
    args = ap.parse_args()

    t_start = time.time()
    out_dir = os.path.join(args.run_dir, "offline")
    os.makedirs(out_dir, exist_ok=True)
    rd = RunData(args.run_dir)
    log.info(f"loaded {len(rd.ids)} keyframes, {rd.frame_t.size} frames from {args.run_dir}")

    report = {"run_dir": args.run_dir, "evidence_from_live_run": evidence(rd)}
    log.info(f"evidence from the live run: {json.dumps(report['evidence_from_live_run'])}")

    p = LoopParams(min_gap_s=args.min_gap, top_k=args.top_k, odom_robust_c=args.odom_robust)
    base, counts = rd.base_edges(reloc_info=loop_info(30, p))
    init = {i: rd.pose_odom[i] for i in rd.ids}
    # the chain must be connected for a single gauge: report (don't hide) any break
    linked = set()
    for e in base:
        linked.add(e.a); linked.add(e.b)
    report["unlinked_keyframes"] = len([i for i in rd.ids if i not in linked]) if len(rd.ids) > 1 else 0

    verifier = GeometricVerifier(Config(), rd.intr)
    poses, loops, stats = close_loops(rd.nodes, init, base, verifier, p)
    report["offline_loops"] = stats

    corr = [np.linalg.norm((lie.se3_inverse(init[i]) @ poses[i])[:3, 3]) for i in rd.ids]
    report["correction_vs_odometry_m"] = {"median": round(float(np.median(corr)), 3),
                                          "max": round(float(np.max(corr)), 3)}
    first, last = rd.ids[0], rd.ids[-1]
    report["start_end_gap_m"] = {
        "odometry": round(float(np.linalg.norm(init[last][:3, 3] - init[first][:3, 3])), 3),
        "live_final": round(float(np.linalg.norm(rd.online_final[last][:3, 3] - rd.online_final[first][:3, 3])), 3)
        if last in rd.online_final and first in rd.online_final else None,
        "offline": round(float(np.linalg.norm(poses[last][:3, 3] - poses[first][:3, 3])), 3)}

    frames = rd.frame_poses(poses)
    write_tum(os.path.join(out_dir, "trajectory_final_frames.tum"), frames)
    write_tum(os.path.join(out_dir, "trajectory_final_kf.tum"),
              [(float(rd.nodes[i].sig.t), poses[i]) for i in rd.ids])
    t_of = dict(zip(rd.ids, rd.kf_t))
    with open(os.path.join(out_dir, "loops.json"), "w") as f:
        json.dump([{"a": L.a, "b": L.b, "t_a": float(t_of[L.a]), "t_b": float(t_of[L.b]), **L.meta}
                   for L in loops], f, indent=1)
    log.info(f"trajectory written: {out_dir}/trajectory_final_frames.tum")

    if args.bag:
        from pyslam.offline.fuse import fuse_bag
        report["map"] = fuse_bag(args.bag, frames, rd.frame_status, out_dir, stride=args.stride,
                                 voxel=args.voxel, depth_trunc=args.depth_trunc,
                                 use_open3d=not args.no_open3d)
    report["seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    log.info(f"done in {report['seconds']}s -> {out_dir}")


if __name__ == "__main__":
    main()
