"""
python -m pyslam.tools.build_reloc_map <run_directory> [--out PATH]

Best-effort retrofit: builds a reloc_map/ pack for a run that finished
BEFORE this feature existed, so it never wrote one at the time.

This is salvage, not a guaranteed reconstruction -- said plainly because
the two sources it can pull from are both incomplete by construction:

  1. The imagery side-cache (<run_dir>/imagery_cache/), if the run used
     --keep-imagery-cache. Nodes found here get their features
     RE-EXTRACTED from the cached rgb+depth PNGs, using the run's own
     recorded config.json -- byte-for-byte the same extraction path a
     live mapping run uses (pyslam.frontend.features.extract_signature),
     so the recovered descriptors are exactly what would have been
     packed at the time.

  2. Orphaned LTM SQLite files under the system temp directory. Before
     this feature's run_slam.py change (ltm_path=<run_dir>/ltm.sqlite3),
     every run's LTM landed in a randomly-named tempfile.mkstemp() file
     that nothing ever cleaned up or pointed back to the run that made
     it -- see memory/ltm_store.py and pipeline.py's own Memory
     constructor. This scans for such files, opens each read-only, and
     keeps any node whose id AND timestamp match a row in this run's
     own keyframes.json (a coincidental id collision between two
     unrelated runs is possible in principle; the timestamp match is
     what makes a false hit implausible in practice). These nodes need
     no re-extraction -- the LTM blob already holds desc/kp3d/valid.

Coverage is reported honestly, not silently rounded up: the pack's
meta.json carries a retrofit_coverage block, and any keyframe recovered
by neither route is simply absent from the pack -- relocalize.py will
just never propose it as a candidate, which is the safe failure mode.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import pickle
import sqlite3
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np
import cv2

from pyslam.core.config import Config
from pyslam.core.types import Frame, Intrinsics, Node
from pyslam.core.log import get_logger
from pyslam.core import lie
from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
from pyslam.mapping.reloc_map import write_map_pack_from_nodes

log = get_logger("tools.build_reloc_map")


def _load_keyframes(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "keyframes.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- is {run_dir} a completed run directory?")
    with open(path) as f:
        return json.load(f)


def _load_config(run_dir: str) -> Config:
    path = os.path.join(run_dir, "config.json")
    with open(path) as f:
        d = json.load(f)
    return Config(**d["config"])


def _pose_from_kf_row(row: dict) -> np.ndarray:
    R = lie.quat_to_rot(np.array([row["qw"], row["qx"], row["qy"], row["qz"]]))
    t = np.array([row["x_cam0"], row["y_cam0"], row["z_cam0"]])
    return lie.make_T(R, t)


def _recover_from_imagery_cache(run_dir: str, kf_rows: list[dict], intr: Intrinsics,
                                 cfg: Config) -> dict:
    cache_dir = os.path.join(run_dir, "imagery_cache")
    if not os.path.isdir(cache_dir):
        return {}
    orb = make_orb(cfg)
    recovered = {}
    for row in kf_rows:
        nid = row["node_id"]
        rgb_p = os.path.join(cache_dir, f"n{nid:07d}_rgb.png")
        dep_p = os.path.join(cache_dir, f"n{nid:07d}_depth.png")
        if not (os.path.exists(rgb_p) and os.path.exists(dep_p)):
            continue
        bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        depth = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None or depth.dtype != np.uint16:
            log.warning(f"node {nid}: imagery cache entry unreadable, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = Frame(t=row["timestamp_s"], rgb=rgb, depth=depth, intr=intr, frame_id=nid)
        sig = extract_signature(frame, cfg, orb=orb)
        sig.id = nid
        recovered[nid] = Node(id=nid, sig=sig, pose_odom=_pose_from_kf_row(row),
                               pose_map=_pose_from_kf_row(row),
                               session_id=row.get("session_id", 0))
    return recovered


def _find_orphaned_ltm_files() -> list[str]:
    """Same prefix/suffix memory.py's Memory.__init__ uses for its
    tempfile.mkstemp() fallback path."""
    pattern = os.path.join(tempfile.gettempdir(), "pyslam_ltm_*.sqlite3")
    return sorted(glob.glob(pattern))


def _recover_from_orphaned_ltm(kf_rows: list[dict]) -> dict:
    by_id = {row["node_id"]: row for row in kf_rows}
    recovered = {}
    for db_path in _find_orphaned_ltm_files():
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            rows = conn.execute("SELECT id, t, blob FROM nodes").fetchall()
            conn.close()
        except sqlite3.Error as e:
            log.warning(f"{db_path}: unreadable ({e}), skipping")
            continue
        for nid, t, blob in rows:
            if nid in recovered or nid not in by_id:
                continue
            kf_t = by_id[nid]["timestamp_s"]
            if abs(t - kf_t) > 1e-3:
                continue  # id collision with an unrelated run -- see module docstring
            try:
                node = pickle.loads(blob)
            except Exception as e:  # noqa: BLE001
                log.warning(f"{db_path}: node {nid} blob unreadable ({e}), skipping")
                continue
            node.pose_odom = _pose_from_kf_row(by_id[nid])
            node.pose_map = _pose_from_kf_row(by_id[nid])
            recovered[nid] = node
        if rows:
            log.info(f"{db_path}: {len(rows)} rows scanned, "
                     f"{sum(1 for nid,_,_ in rows if nid in recovered)} matched this run")
    return recovered


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="defaults to <run_dir>/reloc_map")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    out_dir = args.out or os.path.join(run_dir, "reloc_map")

    kf_rows = _load_keyframes(run_dir)
    cfg = _load_config(run_dir)
    with open(os.path.join(run_dir, "map_stats.json")) as f:
        map_stats = json.load(f)
    intr_dict = map_stats.get("intrinsics")
    if intr_dict is None:
        raise RuntimeError(f"{run_dir}/map_stats.json has no intrinsics recorded -- cannot "
                            f"retrofit without knowing the mapping camera's K")
    intr = Intrinsics(**intr_dict)

    _reset_id_counter(0)
    log.info(f"{len(kf_rows)} keyframes in this run's keyframes.json. Attempting recovery...")

    from_cache = _recover_from_imagery_cache(run_dir, kf_rows, intr, cfg)
    log.info(f"imagery cache: recovered {len(from_cache)} of {len(kf_rows)}")

    remaining_rows = [r for r in kf_rows if r["node_id"] not in from_cache]
    from_ltm = _recover_from_orphaned_ltm(remaining_rows) if remaining_rows else {}
    log.info(f"orphaned LTM scan: recovered {len(from_ltm)} of {len(remaining_rows)} remaining")

    nodes = {**from_cache, **from_ltm}
    n_total = len(kf_rows)
    n_recovered = len(nodes)
    if n_recovered == 0:
        log.error("recovered 0 of %d keyframes -- nothing to pack. This run's descriptors are "
                  "genuinely gone; the only fix is a fresh mapping run.", n_total)
        sys.exit(1)
    if n_recovered < n_total:
        log.warning(f"partial recovery: {n_recovered}/{n_total} keyframes ({100*n_recovered/n_total:.0f}%). "
                    f"The pack will relocalize correctly against the recovered nodes, but a query whose "
                    f"only overlap was with a missing keyframe will return NOT_FOUND -- coverage is "
                    f"recorded in meta.json's retrofit_coverage block, not hidden.")

    final_poses = {nid: n.pose_map for nid, n in nodes.items()}
    adjacency: dict[int, list[int]] = {}
    for row in kf_rows:
        nid = row["node_id"]
        if nid not in nodes:
            continue
        for other in row.get("loop_closure_partner_ids", []) + row.get("proximity_partner_ids", []):
            if other in nodes:
                adjacency.setdefault(nid, []).append(other)

    coverage = {
        "n_keyframes_in_run": n_total,
        "n_recovered": n_recovered,
        "n_recovered_from_imagery_cache": len(from_cache),
        "n_recovered_from_orphaned_ltm": len(from_ltm),
        "missing_node_ids": sorted(set(r["node_id"] for r in kf_rows) - set(nodes)),
    }

    write_map_pack_from_nodes(
        out_dir, nodes, final_poses, adjacency, intr, cfg,
        source_run_dir=run_dir, retrofit_coverage=coverage)
    log.info(f"Retrofit pack written to {out_dir} ({n_recovered}/{n_total} keyframes).")


if __name__ == "__main__":
    main()
