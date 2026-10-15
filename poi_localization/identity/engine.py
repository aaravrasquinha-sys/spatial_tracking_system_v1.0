"""IdentityEngine: the only object M4Pipeline talks to (R1 spec sections 5 and 10.8).

PHASE 1 SCOPE (scaffolding): lifecycle sync only.
  * track confirmed          -> a case opens; with no matching yet, every case commits `new`
  * track_ended              -> the person goes `missing` (exit info stored, TTL started)
  * TTL expiry               -> the person is removed from the gallery
  * freeze()/unfreeze()      -> no commits while frozen; cases stay `resolving`
Later phases plug the extractors, profiles, cue calibration and resolver in behind `_open_case`.

Determinism: decisions use `t` (frame time) only. Wall-clock (perf_counter) is used solely for the
timing statistics reported in health(), never for a decision.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from poi_localization.identity.config import IdentityConfig
from poi_localization.identity.gallery.base import BaseGalleryBackend
from poi_localization.identity.gallery.memory import InMemoryGallery
from poi_localization.identity.types import NEW, RESOLVING, IdentityView, PersonProfile
from poi_localization.log import get_logger

log = get_logger("identity.engine")


class IdentityEngine:
    def __init__(self, cfg: IdentityConfig, gallery: Optional[BaseGalleryBackend] = None, log_writer: Any = None):
        self.cfg = cfg
        self.gallery: BaseGalleryBackend = gallery if gallery is not None else InMemoryGallery()
        self.log_writer = log_writer
        self.frozen = False
        self.frozen_reason = ""
        self._pending: Dict[int, float] = {}                 # world_track_id -> t the case opened (unresolved)
        self._views: Dict[int, IdentityView] = {}
        self._last_out: Dict[int, Any] = {}                  # world_track_id -> last WorldTrackOutput seen
        self._case_seq = 0
        self._step_ms: deque = deque(maxlen=300)
        self.counters = {"cases_opened": 0, "commits_new": 0, "commits_match": 0, "merges": 0,
                         "abstentions": 0, "expired": 0, "steps": 0}

    # ------------------------------------------------------------------ control
    def freeze(self, reason: str) -> None:
        """Calibration is suspect: metric features are invalid. No commits, no learning."""
        if not self.frozen:
            self._log({"kind": "freeze", "t": None, "reason": reason})
        self.frozen = True
        self.frozen_reason = reason

    def unfreeze(self) -> None:
        if self.frozen:
            self._log({"kind": "unfreeze", "t": None, "reason": "calibration ok"})
        self.frozen = False
        self.frozen_reason = ""

    def close(self) -> None:
        if self.log_writer is not None:
            try:
                self.log_writer.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ main entry
    def step(self, frame, t: float, assignments: Dict[int, Tuple[Any, Any]], snapshot: Dict[int, dict],
             events: List[Any], outputs: List[Any]) -> Dict[int, IdentityView]:
        """One frame. `assignments` maps world_track_id -> (Detection2D, DetectionMeasurements) for tracks
        measured this frame (unused in Phase 1). Returns {world_track_id: IdentityView} for every output track."""
        t_start = time.perf_counter()

        self._expire(t)

        for e in events:
            if e.kind == "track_ended" and e.world_track_id is not None:
                self._on_track_ended(e.world_track_id, t)

        for o in outputs:
            tid = o.world_track_id
            self._last_out[tid] = o
            if tid not in self._views:
                self._open_case(tid, t)

        if not self.frozen:
            for tid in sorted(self._pending):
                self._commit_new(tid, t)

        views = {o.world_track_id: self._views[o.world_track_id] for o in outputs if o.world_track_id in self._views}
        self.counters["steps"] += 1
        self._step_ms.append((time.perf_counter() - t_start) * 1000.0)
        return views

    # ------------------------------------------------------------------ lifecycle
    def _open_case(self, tid: int, t: float) -> None:
        self._case_seq += 1
        self.counters["cases_opened"] += 1
        self._pending[tid] = t
        self._views[tid] = IdentityView(id=None, label=None, state=RESOLVING, conf=0.0, candidate=None)
        self._log({"kind": "case_opened", "t": t, "world_track_id": tid, "case_id": self._case_seq,
                   "n_candidates": len(list(self.gallery.candidates(t)))})

    def _commit_new(self, tid: int, t: float) -> None:
        opened_t = self._pending.pop(tid)
        pid = self.gallery.new_person_id()
        self.gallery.put(PersonProfile(person_id=pid, created_t=t, last_seen_t=t))
        self.gallery.link(pid, tid)
        self._views[tid] = IdentityView(id=pid, label=f"P{pid}", state=NEW, conf=1.0, candidate=None)
        self.counters["commits_new"] += 1
        self._log({"kind": "commit", "t": t, "world_track_id": tid, "person_id": pid, "decision": "new",
                   "reason": "phase1: no matching yet", "elapsed_s": t - opened_t})

    def _on_track_ended(self, tid: int, t: float) -> None:
        self._pending.pop(tid, None)
        self._views.pop(tid, None)
        last = self._last_out.pop(tid, None)
        pid = self.gallery.unlink(tid)
        if pid is None:
            return
        exit_info: Dict[str, Any] = {"world_track_id": tid, "t": t}
        if last is not None:
            exit_info.update(position_xy=list(last.position_xy), velocity_xy=list(last.velocity_xy),
                             height_m=last.height_m)
        self.gallery.mark_missing(pid, t, exit_info, self.cfg.missing_ttl_s)
        self._log({"kind": "person_missing", "t": t, "world_track_id": tid, "person_id": pid})

    def _expire(self, t: float) -> List[int]:
        gone = self.gallery.expire(t)
        for pid in gone:
            self.counters["expired"] += 1
            self._log({"kind": "person_expired", "t": t, "person_id": pid})
        return gone

    # ------------------------------------------------------------------ reporting
    def health(self) -> dict:
        ms = sorted(self._step_ms)
        p50 = ms[len(ms) // 2] if ms else 0.0
        p95 = ms[min(len(ms) - 1, int(0.95 * len(ms)))] if ms else 0.0
        st = self.gallery.stats()
        by_state = st.get("by_state", {})
        return {"enabled": True, "active": by_state.get("active", 0), "missing": by_state.get("missing", 0),
                "open_cases": len(self._pending), "ms_p50": round(p50, 4), "ms_p95": round(p95, 4),
                "frozen": self.frozen, "degraded": [], "gallery_bytes": st.get("approx_bytes", 0),
                **{k: v for k, v in self.counters.items()}}

    def _log(self, row: dict) -> None:
        if self.log_writer is None or not self.cfg.log.enabled:
            return
        try:
            self.log_writer.write(row)
        except Exception as e:                                   # logging must never break tracking
            log.warning(f"identity log write failed: {e}")
