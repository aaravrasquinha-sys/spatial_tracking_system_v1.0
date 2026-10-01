"""
Relocalization map pack: writer + reader.

A finished mapping run (pipeline.finalize() already called, so
result.final_poses reflects the closing optimisation, not the online
poses -- see tools/run_outputs.py's own ordering discipline, which this
follows) has everything relocalize.py needs, but not in a form that
survives the process: descriptors and 3D points live only in
Memory._all (RAM) and, for whatever happened to be evicted, in an LTM
SQLite file run_slam.py never even names (tempfile.mkstemp -> /tmp).

This module is the fix: one self-contained, versioned directory,
written once, read many times, by a separate process on a separate
invocation.

Design mirrors two things already in this codebase rather than
inventing new conventions:
  - memory/ltm_store.py's WAL + one-row-per-node + pickled-blob shape,
    for exactly the same "SIGKILL mid-write leaves an openable DB"
    reason.
  - tools/trajectory_export.py's "one shared function every caller
    goes through" pattern, so a hand run and an automated run produce
    byte-identical packs.

Frozen contract, mirrored from core/types.py's own comment: a pack
NEVER stores Signature.rgb/Signature.depth. Nothing in relocalize.py
needs imagery, and not storing it keeps the pack small regardless of
map size.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Optional
import json
import os
import pickle
import sqlite3
import numpy as np

from pyslam.core.types import Node, Signature, Intrinsics
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.vpr.global_desc import GlobalDescriptor

log = get_logger("mapping.reloc_map")

SCHEMA_VERSION = 1

_NODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    t REAL NOT NULL,
    blob BLOB NOT NULL
);
"""


def _strip_for_pack(node: Node) -> Node:
    """Same rule as ltm_store.py's _strip_for_ltm: never persist rgb/depth."""
    light_sig = replace(node.sig, rgb=None, depth=None)
    return replace(node, sig=light_sig)


# --------------------------------------------------------------------- write

def write_map_pack_from_nodes(out_dir: str, nodes: dict, final_poses: dict, adjacency: dict,
                               intr: Intrinsics, cfg: Config, source_run_dir: Optional[str] = None,
                               T_grav_cam0: Optional[np.ndarray] = None,
                               gravity_aligned: bool = False,
                               gravity_reason: str = "not computed",
                               gdesc_dim: int = 256,
                               retrofit_coverage: Optional[dict] = None) -> dict:
    """Low-level writer, taking plain data rather than a live Pipeline --
    the single source of truth both write_map_pack() (the normal,
    end-of-run path) and tools/build_reloc_map.py (the best-effort
    retrofit path) go through, so the pack FORMAT never has two
    implementations to keep in sync.

    nodes: {node_id: Node}, adjacency: {node_id: [neighbour_ids]}.
    retrofit_coverage, if given, is folded into meta.json verbatim (see
    build_reloc_map.py) so a retrofit pack always says honestly how much
    of the source run it could actually recover.
    """
    os.makedirs(out_dir, exist_ok=True)
    node_ids = sorted(nid for nid in nodes if nid in final_poses)
    if not node_ids:
        raise ValueError("no nodes with both a Node and a final pose to pack")

    db_path = os.path.join(out_dir, "nodes.sqlite3")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit, one txn per row --
        # same crash-safety reasoning as LtmStore.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(_NODES_SCHEMA)

    gdesc_fn = GlobalDescriptor(out_dim=gdesc_dim)

    all_desc = []
    owner = []
    gdesc_rows = []
    poses = {}
    n_desc_total = 0
    n_nodes_no_desc = 0

    for nid in node_ids:
        node = nodes[nid]
        light = _strip_for_pack(node)
        blob = pickle.dumps(light, protocol=pickle.HIGHEST_PROTOCOL)
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, session_id, t, blob) VALUES (?,?,?,?)",
            (nid, light.session_id, float(light.sig.t), blob),
        )

        d = node.sig.desc
        if d.shape[0] == 0:
            n_nodes_no_desc += 1
        else:
            all_desc.append(d)
            owner.append(np.full(d.shape[0], nid, dtype=np.int64))
            n_desc_total += d.shape[0]

        gdesc_rows.append(gdesc_fn.compute(d))
        poses[nid] = final_poses[nid]

    conn.close()

    node_ids_arr = np.array(node_ids, dtype=np.int64)
    gdesc = np.stack(gdesc_rows, axis=0).astype(np.float32) if gdesc_rows else \
        np.zeros((0, gdesc_dim), dtype=np.float32)
    np.save(os.path.join(out_dir, "gdesc.npy"), gdesc)
    np.save(os.path.join(out_dir, "gdesc_node_ids.npy"), node_ids_arr)

    if all_desc:
        desc_concat = np.concatenate(all_desc, axis=0)
        owner_concat = np.concatenate(owner, axis=0)
    else:
        desc_concat = np.zeros((0, 32), dtype=np.uint8)
        owner_concat = np.zeros((0,), dtype=np.int64)
    np.save(os.path.join(out_dir, "desc.npy"), desc_concat)
    np.save(os.path.join(out_dir, "owner.npy"), owner_concat)

    pose_arrays = {f"pose_{nid}": poses[nid] for nid in node_ids}
    node_id_set = set(node_ids)
    adj_json = {str(k): [n for n in v if n in node_id_set]
                for k, v in adjacency.items() if k in node_id_set}
    np.savez(os.path.join(out_dir, "poses.npz"), node_ids=node_ids_arr, **pose_arrays)
    with open(os.path.join(out_dir, "adjacency.json"), "w") as f:
        json.dump(adj_json, f)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_run_dir": source_run_dir,
        "n_nodes": len(node_ids),
        "n_nodes_no_descriptors": n_nodes_no_desc,
        "n_descriptors_total": n_desc_total,
        "gdesc_dim": gdesc_dim,
        "intrinsics": {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "width": intr.width, "height": intr.height,
            "depth_scale": intr.depth_scale, "baseline": intr.baseline,
        },
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "T_grav_cam0": (T_grav_cam0.tolist() if T_grav_cam0 is not None else np.eye(4).tolist()),
        "gravity_aligned": bool(gravity_aligned),
        "gravity_alignment_reason": gravity_reason,
    }
    if retrofit_coverage is not None:
        meta["retrofit_coverage"] = retrofit_coverage
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)

    log.info(f"Map pack written: {out_dir} ({len(node_ids)} nodes, {n_desc_total} descriptors"
             f"{f', {n_nodes_no_desc} nodes with no descriptors' if n_nodes_no_desc else ''})")
    return meta


def write_map_pack(out_dir: str, pipeline, result, intr: Intrinsics,
                    source_run_dir: Optional[str] = None,
                    T_grav_cam0: Optional[np.ndarray] = None,
                    gravity_aligned: bool = False,
                    gravity_reason: str = "not computed",
                    gdesc_dim: int = 256) -> dict:
    """The normal, end-of-run path: writes <out_dir>/{meta.json,
    nodes.sqlite3, desc.npy, owner.npy, gdesc.npy, poses.npz} straight
    from a live Pipeline. Call AFTER pipeline.finalize(result) -- same
    ordering rule tools/run_outputs.py enforces for map.ply, and for the
    same reason: this pack must describe the CLOSED graph, not the
    online one.

    Returns a small stats dict, also written into meta.json, so
    run_outputs.py can merge it into summary.json the same way it does
    for the point-cloud stats.
    """
    final_poses = result.final_poses or {}
    node_ids = [nid for nid in pipeline.memory.all_node_ids() if nid in final_poses]
    if not node_ids:
        raise ValueError("no finalized nodes to pack -- did pipeline.finalize() run and "
                          "succeed? (write_map_pack must be called with result.final_poses set)")
    nodes = {nid: pipeline.memory.get(nid) for nid in node_ids}

    # adjacency, for the ambiguity/consensus test's "are these two just
    # neighbours" check -- built from the graph's own link list, the
    # same source keyframes.csv's loop/proximity columns use.
    adjacency: dict[int, list[int]] = {}
    for link in pipeline.graph.links:
        if link.a in final_poses and link.b in final_poses:
            adjacency.setdefault(link.a, []).append(link.b)
            adjacency.setdefault(link.b, []).append(link.a)

    return write_map_pack_from_nodes(
        out_dir, nodes, final_poses, adjacency, intr, pipeline.cfg,
        source_run_dir=source_run_dir, T_grav_cam0=T_grav_cam0, gravity_aligned=gravity_aligned,
        gravity_reason=gravity_reason, gdesc_dim=gdesc_dim)


# ---------------------------------------------------------------------- read

class RelocMap:
    """Read-only handle on a map pack. Opens SQLite in mode=ro (a real
    open-mode restriction, not just a convention this class promises to
    honour) and mmaps every array -- see class docstring in
    relocalize.py's caller for why this matters at query time: no
    concurrent writer can ever exist because the process never asks the
    OS for write access in the first place."""

    def __init__(self, pack_dir: str):
        self.pack_dir = pack_dir
        meta_path = os.path.join(pack_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{pack_dir} is not a reloc map pack (no meta.json). Did you run "
                f"tools/build_reloc_map.py or a mapping run with reloc-pack export enabled?")
        with open(meta_path) as f:
            self.meta = json.load(f)
        if self.meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"map pack schema_version={self.meta.get('schema_version')} but this "
                              f"reader expects {SCHEMA_VERSION} -- rebuild the pack.")

        db_path = os.path.join(pack_dir, "nodes.sqlite3")
        # mode=ro via URI: this is a real read-only file-descriptor-level
        # restriction, not a promise this class keeps on its own honour --
        # any write attempt from here raises sqlite3.OperationalError.
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        self.desc = np.load(os.path.join(pack_dir, "desc.npy"), mmap_mode="r")
        self.owner = np.load(os.path.join(pack_dir, "owner.npy"), mmap_mode="r")
        self.gdesc = np.load(os.path.join(pack_dir, "gdesc.npy"), mmap_mode="r")
        self.gdesc_node_ids = np.load(os.path.join(pack_dir, "gdesc_node_ids.npy"))

        poses_npz = np.load(os.path.join(pack_dir, "poses.npz"))
        self.node_ids = poses_npz["node_ids"].tolist()
        self.poses: dict[int, np.ndarray] = {
            int(nid): poses_npz[f"pose_{int(nid)}"] for nid in self.node_ids
        }

        adj_path = os.path.join(pack_dir, "adjacency.json")
        with open(adj_path) as f:
            raw_adj = json.load(f)
        self.adjacency: dict[int, set] = {int(k): set(v) for k, v in raw_adj.items()}

        self.intrinsics = Intrinsics(**self.meta["intrinsics"])
        self.T_grav_cam0 = np.array(self.meta["T_grav_cam0"], dtype=np.float64)
        self.gravity_aligned = bool(self.meta.get("gravity_aligned", False))
        self._node_cache: dict[int, Node] = {}

    def __len__(self) -> int:
        return len(self.node_ids)

    def get_node(self, node_id: int) -> Node:
        if node_id in self._node_cache:
            return self._node_cache[node_id]
        row = self.conn.execute("SELECT blob FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"node {node_id} not in this map pack")
        node = pickle.loads(row[0])
        self._node_cache[node_id] = node
        return node

    def neighbours(self, node_id: int) -> set:
        return self.adjacency.get(node_id, set())

    def check_intrinsics(self, live_intr: Intrinsics, rtol: float = 0.02) -> Optional[str]:
        """Returns None if the live camera's intrinsics are close enough
        to the ones this map was built with, else a human-readable
        mismatch description. Resolution mismatch is an automatic fail
        (K wouldn't even be describing the same pixel grid); fx/fy/cx/cy
        get a relative tolerance since two D435i units, or a wobble in
        auto-calibration, differ slightly run to run."""
        m = self.intrinsics
        if (live_intr.width, live_intr.height) != (m.width, m.height):
            return (f"resolution mismatch: map={m.width}x{m.height} "
                    f"live={live_intr.width}x{live_intr.height}")
        for name in ("fx", "fy", "cx", "cy"):
            a, b = getattr(m, name), getattr(live_intr, name)
            if abs(a - b) > rtol * max(abs(a), 1e-6):
                return f"{name} mismatch: map={a:.2f} live={b:.2f} (>{rtol*100:.0f}% relative)"
        return None

    def close(self) -> None:
        self.conn.close()
