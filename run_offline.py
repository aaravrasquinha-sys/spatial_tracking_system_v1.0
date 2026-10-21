"""
Offline mapper: turn a finished run_bag.py run into a loop-closed trajectory
and a fused map. No real-time limits, no working-memory eviction.

    python run_offline.py runs/run_XXXX_bag --bag ~/bags/room2.db3

Steps (all on what the run already saved; the bag is only re-read for the map):
  1. load every keyframe (reloc_map/), the odometry + bridge links (graph.g2o),
     the per-frame odometry (trajectory_odom.tum) and tracking status (telemetry.json);
  2. evidence: what the live run's loop closures and graph weights looked like;
  3. loop closing over ALL keyframe pairs (pyslam/offline/loops.py) with
     physically weighted links and a robust solve (pyslam/offline/pgo.py);
  4. writes <run>/offline/: trajectory_final_frames.tum, trajectory_final_kf.tum,
     loops.csv, graph.g2o, report.json;
  5. with --bag: re-reads the recording and fuses every 2nd tracked frame into one
     surface (pyslam/offline/fuse.py) -> map_mesh.ply + map_cloud.ply
     (TSDF, needs open3d) or map_cloud.ply (voxel average, no open3d).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyslam.core import lie
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.core.types import Link
from pyslam.loop.verify import GeometricVerifier
from pyslam.offline import pgo
from pyslam.offline.loops import LoopParams, close_loops
from pyslam.offline.rundir import RunData, write_tum

log = get_logger("run_offline")


def evidence(rd: RunData, base_counts: dict) -> dict:
    """What the live run did, in numbers -- the reasons this offline pass exists."""
    ev = {"keyframes": len(rd.ids), "frames": int(rd.frame_t.size), "live_graph_links": base_counts}
    kf_path = os.path.join(rd.run_dir, "keyframes.json")
    if os.path.exists(kf_path):
        with open(kf_path) as f:
            kfs = json.load(f)
        t_of = {r["node_id"]: r["timestamp_s"] for r in kfs}
        gaps = []
        for r in kfs:
            for o in r.get("loop_closure_partner_ids", []):
                if r["node_id"] < o or o not in t_of:
                    continue
                gaps.append(abs(r["timestamp_s"] - t_of[o]))
        ev["live_loop_closures"] = len(gaps)
        if gaps:
            g = np.array(gaps)
            ev["live_loop_time_gap_s"] = {"median": round(float(np.median(g)), 1),
                                          "max": round(float(g.max()), 1),
                                          "over_30s": int((g > 30).sum()), "over_60s": int((g > 60).sum())}
        ev["live_keyframes_evicted_to_ltm"] = int(sum(1 for r in kfs if r.get("was_in_ltm")))
    od = [e["info"] for e in rd.g2o_edges if e["kind"] == "odom"]
    lp = [e["info"] for e in rd.g2o_edges if e["kind"] in ("loop", "proximity")]
    if od:
        d = np.median([np.diag(i) for i in od], axis=0)
        ev["live_odom_link_sigma"] = {"m": round(float(1 / np.sqrt(d[:3].mean())), 3),
                                      "deg": round(float(np.degrees(1 / np.sqrt(d[3:].mean()))), 2)}
    if lp:
        d = np.median([np.diag(i) for i in lp], axis=0)
        ev["live_loop_link_sigma"] = {"m": round(float(1 / np.sqrt(d[:3].mean())), 3),
                                      "deg": round(float(np.degrees(1 / np.sqrt(d[3:].mean()))), 2)}
    lost = sum(1 for s in rd.frame_status.values() if s == "LOST")
    ev["frames_lost"] = lost
    return ev


def connect_orphans(rd: RunData, edges: list) -> int:
    """Every keyframe must be reachable through odometry/bridge links (a keyframe the live run
    tied in only through a loop/reloc link would otherwise float free once those are dropped)."""
    adj = {i: set() for i in rd.ids}
    for e in edges:
        adj[e.a].add(e.b)
        adj[e.b].add(e.a)
    seen, stack = {rd.ids[0]}, [rd.ids[0]]
    while stack:
        for m in adj[stack.pop()]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    added = 0
    for k, nid in enumerate(rd.ids):
        if nid in seen or k == 0:
            continue
        prev = rd.ids[k - 1]
        T_ab = lie.se3_inverse(rd.pose_odom[prev]) @ rd.pose_odom[nid]
        info = np.diag([1 / 1.0 ** 2] * 3 + [1 / np.radians(30.0) ** 2] * 3)   # very weak: "somewhere near"
        edges.append(pgo.Edge(prev, nid, T_ab, info, kind="bridge"))
        seen.add(nid)
        added += 1
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--bag", default=None, help="the recording, to build the fused map (optional)")
    ap.add_argument("--out", default=None, help="output folder (default <run_dir>/offline)")
    ap.add_argument("--stride", type=int, default=2, help="fuse every Nth frame (default 2)")
    ap.add_argument("--voxel", type=float, default=0.02, help="map voxel size in m (default 0.02)")
    ap.add_argument("--depth-trunc", type=float, default=3.0, help="ignore depth beyond this (m)")
    ap.add_argument("--no-open3d", action="store_true", help="voxel-average cloud even if open3d exists")
    ap.add_argument("--min-gap-s", type=float, default=LoopParams.min_gap_s)
    ap.add_argument("--top-k", type=int, default=LoopParams.top_k)
    ap.add_argument("--min-inliers", type=int, default=LoopParams.min_inliers)
    ap.add_argument("--odom-robust", type=float, default=0.0,
                    help=">0: let a few bad odometry links give way to the loops (Cauchy c, e.g. 30)")
    ap.add_argument("--no-bridge-velocity", action="store_true",
                    help="keep the live run's 'no movement' guess across lost-tracking gaps")
    ap.add_argument("--map-only", action="store_true",
                    help="skip loop closing, fuse the run's own final poses (for comparison)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run_dir, "offline")
    os.makedirs(out, exist_ok=True)
    t_all = time.time()
    rd = RunData(args.run_dir)
    base, counts = rd.base_edges()
    n_orph = connect_orphans(rd, base)
    ev = evidence(rd, counts)
    ev["orphan_keyframes_reconnected"] = n_orph
    if not args.no_bridge_velocity:
        ev["bridge_velocity"] = rd.improve_bridges(base)
    log.info("EVIDENCE from the live run: " + json.dumps(ev))

    report = {"run_dir": args.run_dir, "evidence": ev}
    if args.map_only:
        kf_final = {i: rd.online_final[i] for i in rd.ids}
        loops, stats = [], {"mode": "map-only (live run's final poses)"}
    else:
        p = LoopParams(min_gap_s=args.min_gap_s, top_k=args.top_k, min_inliers=args.min_inliers,
                       odom_robust_c=args.odom_robust)
        verifier = GeometricVerifier(Config(), rd.intr)
        init = {i: rd.pose_odom[i].copy() for i in rd.ids}
        t0 = time.time()
        kf_final, loops, stats = close_loops(rd.nodes, init, base, verifier, p)
        stats["seconds"] = round(time.time() - t0, 1)
        stats["params"] = {k: getattr(p, k) for k in ("min_gap_s", "top_k", "min_inliers", "odom_robust_c")}
    report["loops"] = stats

    frames = rd.frame_poses(kf_final)
    write_tum(os.path.join(out, "trajectory_final_frames.tum"), frames)
    write_tum(os.path.join(out, "trajectory_final_kf.tum"),
              [(float(rd.nodes[i].sig.t), kf_final[i]) for i in rd.ids])
    with open(os.path.join(out, "loops.csv"), "w") as f:
        f.write("node_a,node_b,t_a,t_b,gap_s,inliers,appearance_score,final_weight\n")
        for L in loops:
            ta, tb = float(rd.nodes[L.a].sig.t), float(rd.nodes[L.b].sig.t)
            f.write(f"{L.a},{L.b},{ta:.3f},{tb:.3f},{abs(tb - ta):.1f},{L.meta.get('inliers', 0)},"
                    f"{L.meta.get('score', 0):.3f},{L.meta.get('final_weight', 1):.3f}\n")
    try:
        from pyslam.tools.trajectory_export import write_g2o
        links = [Link(e.a, e.b, e.T_ab, e.info, e.kind) for e in base + loops]
        write_g2o(os.path.join(out, "graph.g2o"), kf_final, links)
    except Exception as e:  # noqa: BLE001
        log.warning(f"graph.g2o not written: {e}")

    P = np.array([T[:3, 3] for _, T in frames])
    report["trajectory"] = {"start_to_end_gap_m": round(float(np.linalg.norm(P[-1] - P[0])), 3),
                            "path_length_m": round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 2)}
    shift = [np.linalg.norm(kf_final[i][:3, 3] - rd.online_final[i][:3, 3]) for i in rd.ids if i in rd.online_final]
    if shift:
        report["trajectory"]["keyframe_shift_vs_live_m"] = {"median": round(float(np.median(shift)), 3),
                                                            "max": round(float(np.max(shift)), 3)}

    if args.bag:
        from pyslam.offline.fuse import fuse_bag
        report["map"] = fuse_bag(os.path.expanduser(args.bag), frames, rd.frame_status, out, stride=args.stride,
                                 voxel=args.voxel, depth_trunc=args.depth_trunc, use_open3d=not args.no_open3d)
    report["seconds_total"] = round(time.time() - t_all, 1)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n================ OFFLINE MAPPER SUMMARY ================")
    print(f"keyframes {ev['keyframes']}, frames {ev['frames']}, lost frames {ev['frames_lost']}")
    if "live_loop_closures" in ev:
        print(f"LIVE run: {ev['live_loop_closures']} loop closures, time gap {ev.get('live_loop_time_gap_s')}, "
              f"{ev.get('live_keyframes_evicted_to_ltm')} keyframes evicted from the loop search")
    print(f"lost-tracking gaps: {ev.get('bridge_velocity')}")
    print(f"LIVE weights: odometry link sigma {ev.get('live_odom_link_sigma')}, "
          f"loop link sigma {ev.get('live_loop_link_sigma')}")
    if not args.map_only:
        print(f"OFFLINE: {stats.get('appearance_candidates')} appearance candidates -> "
              f"{stats.get('appearance_loops_verified')} verified -> {stats.get('loops_final')} loops kept; "
              f"time gap {stats.get('loop_time_gap_s')}")
    print(f"trajectory: {report['trajectory']}")
    if "map" in report:
        print(f"map: {report['map'].get('method')}, {report['map'].get('n_points')} points, "
              f"{report['map'].get('frames_fused')} frames fused -> {report['map'].get('mesh') or report['map'].get('cloud')}")
    print(f"outputs in {out}  ({report['seconds_total']} s)")


if __name__ == "__main__":
    main()
