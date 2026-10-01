"""
WP-K1/K3 (Phase A): the ONE place every runner (run_slam.py, run_bag.py,
run_synth.py) turns a finished Pipeline run into files on disk.

Why this exists as a shared function rather than three copies:

  * Before this, each runner wrote map.ply FIRST and only then called
    pipeline.finalize() -- so the exported point cloud was assembled from
    the ONLINE poses, while trajectory_final_*.tum / graph.g2o were written
    from the closing full-graph optimisation. Same run, two different
    answers, and the map was always the worse one (it never saw whatever
    a loop closure fixed after a node had already left WM). Three copy-
    pasted blocks are exactly how that ordering bug got into all three
    runners at once. Putting the sequence here means the order is enforced
    by ONE function, and tests/gates/test_ga.py checks it structurally.

  * The imagery side-cache (WP-K3) needs an export-time hook and a cleanup
    step; doing that in three places would triple the chance of a leak.

Sequence (order matters, and is the whole point):
    1. pipeline.finalize(result)            closing optimisation
    2. map.ply from the FINAL pose_map      (streams evicted nodes' imagery
                                             back from the side-cache)
    3. trajectory / keyframe / g2o exports + plots
    4. drop the imagery side-cache          (unless keep_imagery_cache)

Any step failing is logged and the others still run: a run directory with
a map but no trajectory (or vice versa) is far more useful than a
traceback and nothing.
"""
from __future__ import annotations
import json
import os
from typing import Optional

import numpy as np

from pyslam.core.log import get_logger
from pyslam.mapping.cloud import assemble_cloud_with_stats
from pyslam.mapping.export import write_ply
from pyslam.tools import trajectory_export, plots

log = get_logger("run_outputs")


def finalize_and_export(run_dir: str, pipeline, result, intr,
                        summary_path: Optional[str] = None,
                        keep_imagery_cache: bool = False,
                        cloud_kwargs: Optional[dict] = None) -> dict:
    """Returns a dict: finalized (bool), finalize_error, map (stats dict incl.
    ply path), trajectory_report (dict or None), imagery_cache (stats).
    If summary_path is given, the same information is merged into that
    JSON file under the keys "finalized" and "map" so a run's summary.json
    says how complete its own map is."""
    out: dict = {"finalized": False, "finalize_error": None, "map": None,
                 "trajectory_report": None, "imagery_cache": None}

    # 1. closing optimisation ------------------------------------------------
    try:
        pipeline.finalize(result)
        out["finalized"] = True
    except Exception as e:  # noqa: BLE001
        out["finalize_error"] = f"{type(e).__name__}: {e}"
        log.warning(f"finalize() failed ({out['finalize_error']}); the map will use ONLINE poses "
                    f"and no final trajectory will be exported.")

    # 2. map, from whatever pose_map now holds --------------------------------
    try:
        nodes = [pipeline.memory.get(i) for i in pipeline.memory.all_node_ids()]
        pts, colors, stats = assemble_cloud_with_stats(
            nodes, intr.K(), intr.depth_scale,
            imagery_loader=pipeline.memory.load_imagery, **(cloud_kwargs or {}))
        ply_path = os.path.join(run_dir, "map.ply")
        write_ply(ply_path, pts, colors)
        stats["ply_path"] = ply_path
        stats["poses"] = "final (post-finalize)" if out["finalized"] else "online (finalize failed)"
        out["map"] = stats
        with open(os.path.join(run_dir, "map_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        log.info(f"Map: {ply_path} ({stats['n_points']} points; "
                 f"{stats['n_nodes_in_map']}/{stats['n_nodes_total']} keyframes contributed, "
                 f"{stats['n_nodes_from_imagery_cache']} of them restored from the imagery cache; "
                 f"poses: {stats['poses']})")
        if stats["n_nodes_skipped_no_imagery"] > 0:
            log.warning(f"{stats['n_nodes_skipped_no_imagery']} keyframes have no imagery and are NOT in "
                        f"the map (imagery_cache_enabled=False, cache write failed, or empty depth).")
    except Exception as e:  # noqa: BLE001
        log.warning(f"map export failed: {type(e).__name__}: {e}")
        out["map"] = {"error": f"{type(e).__name__}: {e}"}

    # 3. trajectory + plots ---------------------------------------------------
    T_grav_cam0, gravity_aligned, gravity_reason = np.eye(4), False, "trajectory export did not run"
    if out["finalized"]:
        try:
            report, T_grav_cam0 = trajectory_export.export_all(run_dir, pipeline, result, source_intr=intr)
            plots.export_plots(run_dir, pipeline, result, T_grav_cam0)
            out["trajectory_report"] = report
            gravity_aligned = report.get("gravity_aligned", False)
            gravity_reason = report.get("gravity_alignment_reason", "unknown")
            log.info(f"Trajectory report: {report}")
            if not report.get("gravity_aligned", True):
                log.warning("gravity alignment unavailable for this run -- height_range_m / trajectory_height.png "
                            "are measured along the FIRST CAMERA's own axes, not vertical, and must not be "
                            "read as height drift. Start every run with a ~2 s still hold to fix this.")
        except Exception as e:  # noqa: BLE001
            log.warning(f"Trajectory export failed (run directory otherwise intact): {e}")

    # 3.5. relocalization map pack (WP-RELOC) ----------------------------------
    # Same "log and move on" discipline as every step here: a failure
    # here must never take down an otherwise-good run directory. Needs
    # out["finalized"] for the same reason step 3 does -- the pack
    # describes the CLOSED graph (result.final_poses), never the online
    # one; see mapping/reloc_map.py's own docstring on why that ordering
    # is non-negotiable (it's the exact bug WP-K1 fixed for map.ply).
    if out["finalized"]:
        try:
            from pyslam.mapping.reloc_map import write_map_pack
            reloc_dir = os.path.join(run_dir, "reloc_map")
            reloc_meta = write_map_pack(reloc_dir, pipeline, result, intr, source_run_dir=run_dir,
                                         T_grav_cam0=T_grav_cam0, gravity_aligned=gravity_aligned,
                                         gravity_reason=gravity_reason)
            out["reloc_map"] = {"path": reloc_dir, "n_nodes": reloc_meta["n_nodes"]}
        except Exception as e:  # noqa: BLE001
            log.warning(f"relocalization map pack export failed (run directory otherwise intact): "
                        f"{type(e).__name__}: {e}")
            out["reloc_map"] = {"error": f"{type(e).__name__}: {e}"}

    # 4. cleanup --------------------------------------------------------------
    try:
        out["imagery_cache"] = pipeline.memory.imagery_cache_stats()
        if not keep_imagery_cache:
            pipeline.memory.drop_imagery_cache()
    except Exception as e:  # noqa: BLE001
        log.warning(f"imagery cache cleanup failed: {e}")

    if summary_path is not None:
        try:
            summ = {}
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summ = json.load(f)
            summ["finalized"] = out["finalized"]
            if out["finalize_error"]:
                summ["finalize_error"] = out["finalize_error"]
            summ["map"] = out["map"]
            if "reloc_map" in out:
                summ["reloc_map"] = out["reloc_map"]
            with open(summary_path, "w") as f:
                json.dump(summ, f, indent=2, default=float)
        except Exception as e:  # noqa: BLE001
            log.warning(f"could not merge map stats into {summary_path}: {e}")
    return out


def summarize_telemetry(telemetry: list, wm_budget_ms: Optional[float] = None) -> dict:
    """Timing / memory-pressure digest of PipelineResult.telemetry, small
    enough to live in summary.json. Exists because the WP-J2 note says
    wm_budget_ms must be retuned per platform from THIS machine's own
    mem_duration_ms distribution -- and nothing previously computed that
    distribution, so retuning meant hand-parsing telemetry.json.

    mem_duration_ms is 0.0 on non-keyframe frames by construction (no
    memory work happens then), so its percentiles are taken over KEYFRAME
    frames only; including the zeros would drag every percentile toward 0
    and make the budget look far more comfortable than it is."""
    if not telemetry:
        return {"n_frames": 0}

    def pct(a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return None
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                "max": float(a.max())}

    dur = [t["duration_ms"] for t in telemetry]
    kf = [t for t in telemetry if t.get("keyframe")]
    mem = [t.get("mem_duration_ms", 0.0) for t in kf]
    wm = [t.get("wm_size", 0) for t in telemetry]
    out = {
        "n_frames": len(telemetry),
        "frame_duration_ms": pct(dur),
        "effective_fps": (1000.0 / float(np.mean(dur))) if np.mean(dur) > 0 else None,
        "n_keyframe_frames": len(kf),
        "keyframe_mem_duration_ms": pct(mem),
        "wm_size": {"max": int(max(wm)), "final": int(wm[-1])},
        # cost of the transfer itself (incl. the WP-K3 imagery PNG write), which
        # happens after mem_duration_ms is taken; only frames where it did real
        # work (an eviction) are counted, so the numbers describe evictions.
        "eviction_cost_ms": pct([t["enforce_budget_ms"] for t in telemetry
                                  if t.get("enforce_budget_ms", 0.0) > 1.0]),
        "n_frames_odom_lost": sum(1 for t in telemetry if t.get("odom_status") == "LOST"),
    }
    if wm_budget_ms is not None and mem:
        over = sum(1 for m in mem if m > wm_budget_ms)
        out["wm_budget_ms"] = float(wm_budget_ms)
        out["keyframes_over_wm_budget"] = over
        out["fraction_keyframes_over_wm_budget"] = over / len(mem)
    return out
