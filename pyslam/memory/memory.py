"""
WP-P2: Phase 2 memory management.

Upgrades the Phase-0 STM/(unbounded WM) skeleton to the real RTAB-Map
core: STM -> WM -> LTM with SQLite (WAL) persistence, rehearsal
weight-merge, transfer-under-budget, and graph-neighbour retrieval on
loop closure. The `Memory` protocol (architecture doc section 3) does
not change -- this is the body swap that section always described,
`enforce_budget`/`on_loop` go from no-ops to real policy.

Two things carried in from Phase 1B's handoff, both addressed here:
  1. The rehearsal orphaning bug (a merged-away node became unreachable
     from working_set()/stm_ids()/LTM -- present in `_all` but in no
     traversal structure). Fixed via explicit merge groups: a merge
     never discards an id, it redirects it to a representative, and
     every subsequent memory operation (transfer, eviction, retrieval)
     moves the whole group together. See `_merge_into` and
     `audit_consistency()`.
  2. `wm_rehearsal_similarity` moved into Config (was a hardcoded class
     attribute) -- see config.py's note. No behaviour change at the
     default 0.6.
"""
from __future__ import annotations
from collections import deque
from typing import Optional
import os
import tempfile
import numpy as np
import cv2

from pyslam.core.types import Node
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.memory.ltm_store import LtmStore

log = get_logger("memory")


def bow_cosine_similarity(a_ids: np.ndarray, b_ids: np.ndarray) -> float:
    """Cheap rehearsal similarity: Jaccard-like overlap of assigned word ids.
    Used only for STM merge decisions, not for real place recognition."""
    if a_ids is None or b_ids is None or len(a_ids) == 0 or len(b_ids) == 0:
        return 0.0
    sa, sb = set(a_ids.tolist()), set(b_ids.tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def select_transfer_victim(wm_ids_ordered: list[int], weight_of: dict[int, int],
                            protect_recent: int) -> Optional[int]:
    """Pure victim-selection function, deliberately separated from
    Memory so gate G2 can check it against a brute-force reference
    without touching sqlite, the graph, or anything stateful.

    `wm_ids_ordered` is oldest-first (ascending creation order, which
    node ids already are -- see core/types.py's monotonic id counter).
    Eligible = every id except the `protect_recent` most-recently-added
    ones. Victim = eligible id with the lowest weight; ties broken by
    smallest id (oldest first) -- "oldest-of-least-weighted", per the
    architecture doc's own phrasing for this exact policy.
    """
    if protect_recent < 0:
        protect_recent = 0
    eligible = wm_ids_ordered[:-protect_recent] if protect_recent > 0 else list(wm_ids_ordered)
    if not eligible:
        return None
    return min(eligible, key=lambda nid: (weight_of[nid], nid))


def select_redundant_victim(wm_ids_ordered: list, weight_of: dict, poses: dict,
                             protect_recent: int, radius_m: float, angle_deg: float) -> Optional[int]:
    """WP-L4 (cfg.wm_evict_policy == "redundancy"). Pure function, like
    select_transfer_victim, so a gate can check it against a brute-force
    reference. `poses`: node id -> 4x4 world<-cam. Eligible set is the same as
    the default policy (everything except the `protect_recent` newest).
    Crowd(n) = number of OTHER nodes in wm_ids_ordered (protected ones
    included: they are real neighbours) within radius_m of n's position and
    within angle_deg of its optical axis. Victim = max crowd; ties -> lowest
    weight -> smallest id. If nobody is crowded (max crowd 0) this reduces to
    the default oldest-of-least-weighted rule, so nothing is ever evicted
    from a sparse, well-spread map ahead of the ordinary rule."""
    if protect_recent < 0:
        protect_recent = 0
    eligible = wm_ids_ordered[:-protect_recent] if protect_recent > 0 else list(wm_ids_ordered)
    if not eligible:
        return None
    ids = list(wm_ids_ordered)
    pos = np.array([poses[i][:3, 3] for i in ids], dtype=np.float64)
    axis = np.array([poses[i][:3, 2] for i in ids], dtype=np.float64)
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2)
    cosang = np.clip(axis @ axis.T, -1.0, 1.0)
    near = (d2 < radius_m ** 2) & (cosang > np.cos(np.radians(angle_deg)))
    np.fill_diagonal(near, False)
    crowd = {nid: int(near[k].sum()) for k, nid in enumerate(ids)}
    return min(eligible, key=lambda nid: (-crowd[nid], weight_of[nid], nid))


class Memory:
    def __init__(self, cfg: Config, ltm_path: Optional[str] = None,
                 imagery_cache_dir: Optional[str] = None):
        self.cfg = cfg
        self.stm: deque[int] = deque()
        self.wm: dict[int, Node] = {}
        self._all: dict[int, Node] = {}  # resident cache: every id ever seen,
            # loaded from LTM lazily by get() if evicted. See class docstring.
        self.rehearsal_threshold = cfg.wm_rehearsal_similarity

        # --- merge groups (fixes the rehearsal-orphaning bug) ---
        self._merge_rep: dict[int, int] = {}       # merged-away id -> representative id
        self._merge_children: dict[int, list[int]] = {}  # representative -> [merged ids]

        # --- adjacency, for graph-neighbour retrieval on loop closure ---
        self._adjacency: dict[int, set[int]] = {}

        # --- LTM ---
        self.ltm_ids: set[int] = set()  # representative ids currently resident in LTM
        if ltm_path is not None:
            self._ltm_path = ltm_path
        else:
            # A fresh file per Memory instance, not just per-process: a
            # single process (selftest, scenario sweeps, tests that build
            # several Pipelines in a row) can construct many Memory
            # objects, each starting its own node-id numbering back at 0
            # (Pipeline._reset_id_counter) -- reusing one pid-keyed file
            # across them would silently collide ids from unrelated runs.
            fd, self._ltm_path = tempfile.mkstemp(prefix="pyslam_ltm_", suffix=".sqlite3")
            os.close(fd)
        self.ltm_store: Optional[LtmStore] = LtmStore(self._ltm_path) if cfg.ltm_enabled else None
        self.retrieval_events: list[tuple] = []  # (trigger_id, [rehydrated_ids]) audit trail

        # WP-K3 (Phase A3): export-time imagery side-cache. See
        # Config.imagery_cache_enabled for the full rationale. Deliberately
        # NOT part of LTM (ltm_store.py's contract is unchanged: it never
        # persists rgb/depth): retrieval, verification and optimisation
        # can never get imagery back into WM through this -- only
        # load_imagery(), which the map exporter calls one node at a time.
        self._imagery_dir: Optional[str] = imagery_cache_dir  # created lazily on first eviction
        self._imagery_dir_is_ours: bool = imagery_cache_dir is None
        self._imagery_index: dict[int, tuple[str, str]] = {}  # node_id -> (rgb_png, depth_png)
        self._imagery_disabled_reason: Optional[str] = None

        # WP-P3 hooks: pipeline.py wires these to PlaceRecognizer.remove/add
        # so the retrieval index stays scoped to "currently resident in
        # STM/WM", matching RTAB-Map's own behaviour (LTM isn't part of
        # the fast retrieval index). None (no-op) when P3 isn't active.
        self.on_transfer_out: Optional[callable] = None  # called with node_id, on WM->LTM
        self.on_transfer_in: Optional[callable] = None   # called with node_id, on LTM->WM

    # ------------------------------------------------------------ core API
    def add(self, node: Node) -> None:
        self._all[node.id] = node
        merged_into: Optional[int] = None
        if node.sig.word_ids is not None:
            for other_id in list(self.stm):
                other = self._all[other_id]
                sim = bow_cosine_similarity(node.sig.word_ids, other.sig.word_ids)
                if sim >= self.rehearsal_threshold:
                    other.weight += node.weight + 1
                    merged_into = other_id
                    break
        if merged_into is not None:
            self._merge_into(node.id, merged_into)
        else:
            self.stm.append(node.id)
        while len(self.stm) > self.cfg.stm_size:
            old_id = self.stm.popleft()
            self.wm[old_id] = self._all[old_id]

    def _merge_into(self, child_id: int, rep_id: int) -> None:
        """Fold `child_id` into representative `rep_id`'s group instead of
        discarding it. `rep_id` may itself already be someone's child
        (chased to the root) -- chains collapse to a single representative
        so downstream code never has to walk a chain."""
        root = self._merge_rep.get(rep_id, rep_id)
        self._merge_rep[child_id] = root
        self._merge_children.setdefault(root, []).append(child_id)

    def working_set(self) -> list[int]:
        return list(self.wm.keys())

    def get(self, node_id: int) -> Node:
        if node_id in self._all:
            return self._all[node_id]
        # Lazy rehydrate from LTM into the resident cache (but NOT into
        # wm/ltm_ids bookkeeping -- a plain get() is a read, not a
        # retrieval event). This is what lets pipeline.py write back
        # optimised pose_map for an LTM-resident node without every
        # caller having to special-case "is this id currently loaded".
        if self.ltm_store is not None and self.ltm_store.contains(node_id):
            node = self.ltm_store.get(node_id)
            self._all[node_id] = node
            return node
        raise KeyError(f"unknown node id {node_id}")

    def on_loop(self, node_id: int) -> None:
        """Retrieval hook: if `node_id` (a loop-closure hit) or any of
        its 1-hop graph neighbours are currently parked in LTM, pull
        them back into WM so the verifier/optimiser have local context
        around the revisit, not just the single matched node."""
        rep = self._merge_rep.get(node_id, node_id)
        candidates = {rep} | self._adjacency.get(rep, set())
        rehydrated = []
        for cid in candidates:
            if cid in self.ltm_ids:
                self._promote_group(cid)
                rehydrated.append(cid)
        if rehydrated:
            self.retrieval_events.append((node_id, rehydrated))
            log.info(f"on_loop({node_id}): rehydrated {rehydrated} from LTM into WM")

    def _promote_group(self, rep_id: int) -> None:
        """Move a representative + all its merge children from LTM back
        into WM, in one step -- see class docstring on why groups always
        move together."""
        group = [rep_id] + self._merge_children.get(rep_id, [])
        for nid in group:
            if self.ltm_store is not None and self.ltm_store.contains(nid):
                self._all[nid] = self.ltm_store.get(nid)
                self.ltm_store.delete(nid)
        self.ltm_ids.discard(rep_id)
        self.wm[rep_id] = self._all[rep_id]
        if self.on_transfer_in is not None:
            for nid in group:
                self.on_transfer_in(nid)
    def enforce_budget(self, last_update_ms: float) -> None:
        """Transfer-under-budget controller (WP-P2). Deliberately moves
        at most ONE victim group per call: the gate G2 requirement is
        that p95 update latency TRACKS the setpoint "without
        oscillation" -- transferring everything eligible in one shot
        the instant the budget is exceeded would alternately overshoot
        then starve, exactly the oscillation the gate flags. One victim
        per over-budget frame is a simple proportional controller."""
        if self.ltm_store is None:
            return
        over_cap = self.cfg.wm_max_nodes > 0 and len(self.wm) > self.cfg.wm_max_nodes  # WP-L5
        if not over_cap and last_update_ms <= self.cfg.wm_budget_ms:
            return
        wm_ids_ordered = sorted(self.wm.keys())  # ids are monotonic -> this IS creation order
        weight_of = {nid: self.wm[nid].weight for nid in wm_ids_ordered}
        if self.cfg.wm_evict_policy == "redundancy":
            victim = select_redundant_victim(
                wm_ids_ordered, weight_of, {nid: self.wm[nid].pose_map for nid in wm_ids_ordered},
                self.cfg.wm_min_resident, self.cfg.wm_redundancy_radius_m, self.cfg.wm_redundancy_angle_deg)
        else:
            victim = select_transfer_victim(wm_ids_ordered, weight_of, self.cfg.wm_min_resident)
        if victim is None:
            return
        self._transfer_to_ltm(victim)

    def _transfer_to_ltm(self, rep_id: int) -> None:
        group = [rep_id] + self._merge_children.get(rep_id, [])
        for nid in group:
            node = self._all[nid]
            self.ltm_store.put(node)
            # Drop heavy fields from the resident copy too (frozen
            # contract: Signature.rgb/depth "kept only while in WM").
            # We keep the lightweight rest resident in `_all` so
            # pose_map write-backs after optimisation (pipeline.py)
            # never have to special-case an evicted id.
            self._cache_imagery(node)  # WP-K3: BEFORE the RAM copy is dropped
            node.sig.rgb = None
            node.sig.depth = None
        del self.wm[rep_id]
        self.ltm_ids.add(rep_id)
        if self.on_transfer_out is not None:
            for nid in group:
                self.on_transfer_out(nid)

    # ------------------------------------------------ WP-K3 imagery cache
    _PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # level 1: ~3-5x smaller than raw
        # for a few ms of CPU -- this runs inside the frame loop, so speed
        # matters more than the last bytes of compression.

    def _ensure_imagery_dir(self) -> str:
        if self._imagery_dir is None:
            self._imagery_dir = tempfile.mkdtemp(prefix="pyslam_img_")
        else:
            os.makedirs(self._imagery_dir, exist_ok=True)
        return self._imagery_dir

    def _disable_imagery_cache(self, reason: str) -> None:
        if self._imagery_disabled_reason is None:
            self._imagery_disabled_reason = reason
            log.warning(f"imagery cache disabled for the rest of this run ({reason}); nodes "
                        f"evicted from now on will not appear in map.ply. SLAM itself is unaffected.")

    def _cache_imagery(self, node: Node) -> None:
        """Write this node's rgb+depth to lossless PNGs, once. Never raises:
        a full disk or an unexpected dtype must degrade the MAP, never
        crash tracking/optimisation mid-run."""
        if not self.cfg.imagery_cache_enabled or self._imagery_disabled_reason is not None:
            return
        sig = node.sig
        if sig.rgb is None or sig.depth is None:
            return  # already stripped (a second eviction of a node that was rehydrated
                    # from LTM without imagery) -- the first eviction's cache entry stands
        if node.id in self._imagery_index:
            return
        try:
            if sig.depth.dtype != np.uint16 or sig.rgb.dtype != np.uint8 or sig.rgb.ndim != 3:
                self._disable_imagery_cache(f"unexpected dtypes rgb={sig.rgb.dtype}/depth={sig.depth.dtype}")
                return
            d = self._ensure_imagery_dir()
            rgb_p = os.path.join(d, f"n{node.id:07d}_rgb.png")
            dep_p = os.path.join(d, f"n{node.id:07d}_depth.png")
            ok1 = cv2.imwrite(rgb_p, cv2.cvtColor(sig.rgb, cv2.COLOR_RGB2BGR), self._PNG_PARAMS)
            ok2 = cv2.imwrite(dep_p, sig.depth, self._PNG_PARAMS)
            if not (ok1 and ok2):
                for p in (rgb_p, dep_p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                self._disable_imagery_cache("cv2.imwrite returned False (disk full / path not writable?)")
                return
            self._imagery_index[node.id] = (rgb_p, dep_p)
        except Exception as e:  # noqa: BLE001 -- see docstring: must never propagate
            self._disable_imagery_cache(f"{type(e).__name__}: {e}")

    def has_imagery(self, node_id: int) -> bool:
        """True if load_imagery(node_id) can return something: either the
        node still holds rgb+depth in RAM, or it was cached on eviction."""
        node = self._all.get(node_id)
        if node is not None and node.sig.rgb is not None and node.sig.depth is not None:
            return True
        return node_id in self._imagery_index

    def load_imagery(self, node_id: int):
        """(rgb HxWx3 uint8 RGB, depth HxW uint16) for a node, or None.
        Returns the resident arrays if the node is still in RAM, else reads
        the PNG side-cache. Does NOT put anything back on the Node -- the
        exporter streams one node at a time so peak RAM stays flat no matter
        how many keyframes the run produced."""
        node = self._all.get(node_id)
        if node is not None and node.sig.rgb is not None and node.sig.depth is not None:
            return node.sig.rgb, node.sig.depth
        paths = self._imagery_index.get(node_id)
        if paths is None:
            return None
        bgr = cv2.imread(paths[0], cv2.IMREAD_COLOR)
        depth = cv2.imread(paths[1], cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None or depth.dtype != np.uint16:
            log.warning(f"imagery cache entry for node {node_id} is unreadable; skipping it in the map")
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth

    def imagery_cache_stats(self) -> dict:
        n_bytes = 0
        for rgb_p, dep_p in self._imagery_index.values():
            for p in (rgb_p, dep_p):
                try:
                    n_bytes += os.path.getsize(p)
                except OSError:
                    pass
        return {"enabled": bool(self.cfg.imagery_cache_enabled),
                "n_cached_nodes": len(self._imagery_index),
                "bytes_on_disk": n_bytes,
                "dir": self._imagery_dir,
                "disabled_reason": self._imagery_disabled_reason}

    def drop_imagery_cache(self) -> int:
        """Delete every file this Memory wrote (and the directory, if it
        is now empty). Returns the number of files removed. Call after the
        map has been exported unless the caller wants to keep the imagery."""
        n = 0
        for rgb_p, dep_p in list(self._imagery_index.values()):
            for p in (rgb_p, dep_p):
                try:
                    os.remove(p)
                    n += 1
                except OSError:
                    pass
        self._imagery_index.clear()
        if self._imagery_dir is not None:
            try:
                os.rmdir(self._imagery_dir)
            except OSError:
                pass  # not empty (user-supplied dir with other content) or already gone
        return n

    def neighbours(self, node_id: int) -> set:
        """WP-L2: graph neighbours of a node (its merge-representative's
        adjacency: odom/bridge/loop-linked nodes), as a set of representative
        ids. Empty set for an unknown node. Read-only view for the Bayes
        filter's belief diffusion."""
        return self._adjacency.get(self._merge_rep.get(node_id, node_id), set())

    def record_adjacency(self, a: int, b: int) -> None:
        """Called by the pipeline whenever an odom or loop Link is added.
        Memory keeps its own lightweight adjacency map (not a copy of
        the whole pose graph) purely so `on_loop` knows which LTM ids
        are "near" a retrieval hit."""
        ra, rb = self._merge_rep.get(a, a), self._merge_rep.get(b, b)
        self._adjacency.setdefault(ra, set()).add(rb)
        self._adjacency.setdefault(rb, set()).add(ra)

    def ltm_ids_expanded(self) -> list[int]:
        """`ltm_ids` tracks representatives only; a merge group's
        children are graph nodes too (every constructed Node is added
        to the graph regardless of whether memory.add() rehearsal-merged
        it) and need the same "resident in LTM" treatment -- e.g.
        pipeline.py's WM-only-optimisation fixed-node set must freeze
        the whole group, not just its representative."""
        out = list(self.ltm_ids)
        for rep in self.ltm_ids:
            out.extend(self._merge_children.get(rep, []))
        return out

    def all_node_ids(self) -> list[int]:
        return list(self._all.keys())

    def stm_ids(self) -> list[int]:
        return list(self.stm)

    # ------------------------------------------------------------ audit
    def audit_consistency(self) -> list[str]:
        """Gate G2's "zero dangling references" check, as a callable
        rather than a one-off script: returns a list of problem
        descriptions (empty == clean). Checks:
          - every merged-away id resolves to a representative that
            actually exists somewhere (stm/wm/ltm)
          - no id is simultaneously in wm and ltm_ids
          - every wm id is resolvable via get()
          - stm never exceeds cfg.stm_size (should be a structural
            invariant, checked here too as a cheap sanity net)
        """
        problems = []
        if len(self.stm) > self.cfg.stm_size:
            problems.append(f"STM over cap: {len(self.stm)} > {self.cfg.stm_size}")
        overlap = set(self.wm.keys()) & self.ltm_ids
        if overlap:
            problems.append(f"ids present in BOTH wm and ltm_ids: {sorted(overlap)}")
        for child, rep in self._merge_rep.items():
            resident = rep in self.wm or rep in self.ltm_ids or rep in self.stm
            if not resident:
                problems.append(f"merge child {child} points to representative {rep} "
                                 f"which is in none of stm/wm/ltm")
        for nid in list(self.wm.keys()):
            try:
                self.get(nid)
            except KeyError:
                problems.append(f"wm id {nid} not resolvable via get()")
        if self.ltm_store is not None:
            for nid in self.ltm_ids:
                if not self.ltm_store.contains(nid):
                    problems.append(f"ltm_ids claims {nid} resident but store has no row for it")
        return problems

    def close(self) -> None:
        if self.ltm_store is not None:
            self.ltm_store.close()
