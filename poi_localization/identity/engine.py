"""IdentityEngine: the only object M4Pipeline talks to (R1 spec sections 5 and 10.8).

PHASE 1-4 SCOPE:
  * track confirmed -> a case opens. With no scene (no floor registration / reference depth) or no missing
    candidates, the case commits `new` immediately (Phase 1 behaviour, unchanged).
  * With a scene and >= 1 missing candidate, the case accumulates evidence (clean-gated structure + appearance
    observations), and resolver.decide() commits matched / new / ambiguous.
  * committed tracks keep learning their person's profile from training-clean observations only.
  * track_ended -> person `missing` (exit info, TTL); TTL expiry -> removed. freeze() -> no commits, no learning.
Later: session cue calibration, where-cue, swap monitor, joint assignment across simultaneous cases, merge of ambiguous.

Determinism: decisions use `t` (frame time) only. perf_counter feeds health() timing stats only.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from poi_localization.identity import resolver
from poi_localization.identity.config import IdentityConfig
from poi_localization.identity.extract_appearance import n_bins
from poi_localization.identity.gallery.base import BaseGalleryBackend
from poi_localization.identity.gallery.memory import InMemoryGallery
from poi_localization.identity.observer import Observation, OtherTrack, Scene, depth_to_metres, observe
from poi_localization.identity.profile import AppearanceAccumulator, StructureModel
from poi_localization.identity.room_geometry import RoomGeometry
from poi_localization.identity.types import AMBIGUOUS, MATCHED, NEW, RESOLVING, IdentityView, PersonProfile
from poi_localization.log import get_logger

log = get_logger("identity.engine")


class _Case:
    def __init__(self, opened_t: float, cfg: IdentityConfig):
        self.opened_t = opened_t
        self.structure = StructureModel(cfg.struct_max_samples_per_s)
        self.appearance = AppearanceAccumulator(cfg.appearance_ema_halflife_s, n_bins(cfg.hist))
        self.n_obs = 0
        self.llrs: Dict[int, float] = {}


class IdentityEngine:
    def __init__(self, cfg: IdentityConfig, gallery: Optional[BaseGalleryBackend] = None, log_writer: Any = None):
        self.cfg = cfg
        self.gallery: BaseGalleryBackend = gallery if gallery is not None else InMemoryGallery()
        self.log_writer = log_writer
        self.frozen = False
        self.frozen_reason = ""
        self._cases: Dict[int, _Case] = {}                   # world_track_id -> open case
        self._views: Dict[int, IdentityView] = {}
        self._last_out: Dict[int, Any] = {}
        self._models: Dict[int, Tuple[StructureModel, AppearanceAccumulator]] = {}
        self._last_sample_t: Dict[int, float] = {}
        self._last_mismatch_t: Dict[int, float] = {}
        self._case_seq = 0
        self._step_ms: deque = deque(maxlen=300)
        self._floor: Optional[Tuple[np.ndarray, np.ndarray, Any]] = None
        self.scene: Optional[Scene] = None
        self._scene_tried = False
        self.degraded: List[str] = []
        self.counters = {"cases_opened": 0, "commits_new": 0, "commits_match": 0, "commits_ambiguous": 0, "merges": 0,
                         "abstentions": 0, "expired": 0, "steps": 0, "extractions": 0, "extract_errors": 0,
                         "train_updates": 0}

    # ------------------------------------------------------------------ scene
    def attach_floor(self, R, t, distortion_coeffs=None) -> None:
        """Called by M4Pipeline: the room registration. The scene is built lazily on the first frame (needs intrinsics)."""
        self._floor = (np.asarray(R, float), np.asarray(t, float), distortion_coeffs)

    def set_scene(self, scene: Optional[Scene]) -> None:
        self.scene = scene
        self._scene_tried = True

    def _ensure_scene(self, frame) -> None:
        if self.scene is not None or self._scene_tried or self._floor is None:
            return
        self._scene_tried = True
        path = self.cfg.reference_depth_path
        if not path or not Path(path).exists():
            self.degraded.append("no_reference_depth")
            return
        try:
            ref = np.load(path).astype(np.float32)
            ref[~np.isfinite(ref) | (ref <= 0)] = np.nan
            rgb = stable = None
            if self.cfg.static_capture_path and Path(self.cfg.static_capture_path).exists():
                z = np.load(self.cfg.static_capture_path)
                rgb = z["rgb"] if "rgb" in z.files else None
                stable = z["stable"].astype(bool) if "stable" in z.files else None
            R, t, dist = self._floor
            geom = RoomGeometry(R, t, frame.intr, dist)
            if ref.shape != (geom.height, geom.width):
                self.degraded.append("reference_depth_shape_mismatch")
                return
            self.scene = Scene(geom, ref, rgb, stable)
        except Exception as e:
            self.degraded.append(f"scene_load_failed:{type(e).__name__}")
            log.warning(f"identity scene load failed: {e}")

    # ------------------------------------------------------------------ control
    def freeze(self, reason: str) -> None:
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
        t_start = time.perf_counter()
        self._ensure_scene(frame)
        self._expire(t)

        for e in events:
            if e.kind == "track_ended" and e.world_track_id is not None:
                self._on_track_ended(e.world_track_id, t)
            elif e.kind == "2d_id_mismatch" and e.world_track_id is not None:
                self._last_mismatch_t[e.world_track_id] = t

        for o in outputs:
            tid = o.world_track_id
            self._last_out[tid] = o
            if tid not in self._views:
                self._open_case(tid, t)

        observations = self._observe(frame, t, assignments, outputs) if (self.scene is not None and not self.frozen) else {}
        self._learn(observations, t)

        if not self.frozen:
            self._resolve(t, observations)

        views = {o.world_track_id: self._views[o.world_track_id] for o in outputs if o.world_track_id in self._views}
        self.counters["steps"] += 1
        self._step_ms.append((time.perf_counter() - t_start) * 1000.0)
        return views

    # ------------------------------------------------------------------ observation
    def _observe(self, frame, t, assignments, outputs) -> Dict[int, Observation]:
        cfg, scene = self.cfg, self.scene
        if not scene.geom.matches_frame(frame.depth):
            return {}
        state_of = {o.world_track_id: o for o in outputs}
        tids = [tid for tid in assignments if tid in state_of and tid in self._views]
        due = [tid for tid in tids if t - self._last_sample_t.get(tid, -1e9) >= 1.0 / cfg.sample_hz_per_track]
        due.sort(key=lambda tid: (0 if tid in self._cases else 1, self._last_sample_t.get(tid, -1e9), tid))
        due = due[:cfg.max_extractions_per_frame]
        if not due:
            return {}
        depth_m = depth_to_metres(frame)
        geom = scene.geom
        out: Dict[int, Observation] = {}
        for tid in due:
            det, meas = assignments[tid]
            o = state_of[tid]
            others = []
            for otid, (odet, _) in assignments.items():
                if otid == tid or otid not in state_of:
                    continue
                oo = state_of[otid]
                z = float(geom.project(np.array([[oo.position_xy[0], oo.position_xy[1], 1.2]]))[2][0])
                others.append(OtherTrack(odet.bbox, tuple(oo.position_xy), z))
            since = t - self._last_mismatch_t[tid] if tid in self._last_mismatch_t else None
            try:
                out[tid] = observe(scene, frame, depth_m, det, meas, tuple(o.position_xy), tuple(o.velocity_xy), o.state,
                                   tid in self._cases, others, cfg, s_since_2d_mismatch=since, calib_ok=not self.frozen)
                self.counters["extractions"] += 1
            except Exception as e:                                   # one bad frame must not stop identity
                self.counters["extract_errors"] += 1
                if self.counters["extract_errors"] == 1:
                    log.warning(f"identity extraction error ({type(e).__name__}): {e}")
            self._last_sample_t[tid] = t
        return out

    def _learn(self, observations: Dict[int, Observation], t: float) -> None:
        """Committed tracks train their person's profile; only from training-clean observations."""
        for tid, ob in observations.items():
            view = self._views.get(tid)
            if view is None or view.id is None or view.state not in (NEW, MATCHED) or not ob.gate.ok_for_training:
                continue
            sm, am = self._models_of(view.id)
            sm.add(ob.features, t)
            if ob.appearance is not None and ob.appearance.hists:
                am.add(ob.appearance.hists, ob.appearance.mass, ob.view, t)
            self.counters["train_updates"] += 1

    # ------------------------------------------------------------------ resolving
    def _models_of(self, pid: int) -> Tuple[StructureModel, AppearanceAccumulator]:
        m = self._models.get(pid)
        if m is None:
            prof = self.gallery.get(pid)
            sm = StructureModel.from_dict(prof.structure, self.cfg.struct_max_samples_per_s) if prof and prof.structure \
                else StructureModel(self.cfg.struct_max_samples_per_s)
            am = AppearanceAccumulator.from_dict(prof.outfits[0], self.cfg.appearance_ema_halflife_s, n_bins(self.cfg.hist)) \
                if prof and prof.outfits else AppearanceAccumulator(self.cfg.appearance_ema_halflife_s, n_bins(self.cfg.hist))
            m = self._models[pid] = (sm, am)
        return m

    def _resolve(self, t: float, observations: Dict[int, Observation]) -> None:
        cands = list(self.gallery.candidates(t))
        decisions: List[Tuple[float, int, resolver.Decision]] = []
        for tid in sorted(self._cases):
            case = self._cases[tid]
            if self.scene is None or not cands:
                decisions.append((0.0, tid, resolver.Decision("new", None, 1.0, "no candidates" if self.scene else "phase1: no scene")))
                continue
            ob = observations.get(tid)
            if ob is not None and ob.gate.ok_for_evidence and (ob.features or (ob.appearance and ob.appearance.hists)):
                case.n_obs += 1
                case.structure.add(ob.features, t)
                if ob.appearance is not None and ob.appearance.hists:
                    case.appearance.add(ob.appearance.hists, ob.appearance.mass, ob.view, t)
            llrs = {}
            for c in cands:
                sm, am = self._models_of(c.person_id)
                ls, _ = resolver.structure_llr(case.structure, sm, self.cfg)
                la, _ = resolver.appearance_llr(case.appearance, am, self.cfg)
                llrs[c.person_id] = ls + la
            case.llrs = llrs
            d = resolver.decide(llrs, case.n_obs, t - case.opened_t, self.cfg)
            if d is not None:
                decisions.append((max(llrs.values()) if d.kind == "matched" else -1e9, tid, d))
        # most confident matches first; a candidate can be claimed once (a loser keeps resolving)
        taken = set()
        for _, tid, d in sorted(decisions, key=lambda x: (-x[0], x[1])):
            if d.kind == "matched":
                if d.person_id in taken:
                    continue
                taken.add(d.person_id)
                self._commit_match(tid, d, t)
            elif d.kind == "ambiguous":
                self._commit_ambiguous(tid, d, t)
            else:
                self._commit_new(tid, t, d.reason, d.conf)

    # ------------------------------------------------------------------ lifecycle
    def _open_case(self, tid: int, t: float) -> None:
        self._case_seq += 1
        self.counters["cases_opened"] += 1
        self._cases[tid] = _Case(t, self.cfg)
        self._views[tid] = IdentityView(id=None, label=None, state=RESOLVING, conf=0.0, candidate=None)
        self._log({"kind": "case_opened", "t": t, "world_track_id": tid, "case_id": self._case_seq,
                   "n_candidates": len(list(self.gallery.candidates(t)))})

    def _new_person(self, tid: int, t: float) -> int:
        pid = self.gallery.new_person_id()
        self.gallery.put(PersonProfile(person_id=pid, created_t=t, last_seen_t=t))
        self.gallery.link(pid, tid)
        return pid

    def _commit_new(self, tid: int, t: float, reason: str = "", conf: float = 1.0) -> None:
        case = self._cases.pop(tid)
        pid = self._new_person(tid, t)
        self._views[tid] = IdentityView(id=pid, label=f"P{pid}", state=NEW, conf=float(conf), candidate=None)
        self.counters["commits_new"] += 1
        self._log({"kind": "commit", "t": t, "world_track_id": tid, "person_id": pid, "decision": "new",
                   "reason": reason, "elapsed_s": t - case.opened_t, "n_obs": case.n_obs})

    def _commit_match(self, tid: int, d: "resolver.Decision", t: float) -> None:
        case = self._cases.pop(tid)
        pid = d.person_id
        self.gallery.link(pid, tid)
        self.gallery.mark_active(pid, t)
        self._views[tid] = IdentityView(id=pid, label=f"P{pid}", state=MATCHED, conf=float(d.conf), candidate=None)
        self.counters["commits_match"] += 1
        self._log({"kind": "commit", "t": t, "world_track_id": tid, "person_id": pid, "decision": "matched",
                   "reason": d.reason, "elapsed_s": t - case.opened_t, "n_obs": case.n_obs,
                   "llrs": {str(k): round(v, 3) for k, v in case.llrs.items()}})

    def _commit_ambiguous(self, tid: int, d: "resolver.Decision", t: float) -> None:
        case = self._cases.pop(tid)
        pid = self._new_person(tid, t)
        self._views[tid] = IdentityView(id=pid, label=f"P{pid}", state=AMBIGUOUS, conf=float(d.conf), candidate=d.person_id)
        self.counters["commits_ambiguous"] += 1
        self._log({"kind": "commit", "t": t, "world_track_id": tid, "person_id": pid, "decision": "ambiguous",
                   "candidate": d.person_id, "reason": d.reason, "elapsed_s": t - case.opened_t, "n_obs": case.n_obs,
                   "llrs": {str(k): round(v, 3) for k, v in case.llrs.items()}})

    def _sync_profile(self, pid: int) -> None:
        m = self._models.get(pid)
        if m is None:
            return

        def mut(p: PersonProfile) -> None:
            p.structure = m[0].to_dict()
            p.outfits = [m[1].to_dict()]
        self.gallery.update(pid, mut)

    def _on_track_ended(self, tid: int, t: float) -> None:
        self._cases.pop(tid, None)
        self._views.pop(tid, None)
        self._last_sample_t.pop(tid, None)
        self._last_mismatch_t.pop(tid, None)
        last = self._last_out.pop(tid, None)
        pid = self.gallery.unlink(tid)
        if pid is None:
            return
        self._sync_profile(pid)
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
            self._models.pop(pid, None)
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
                "open_cases": len(self._cases), "ms_p50": round(p50, 4), "ms_p95": round(p95, 4),
                "frozen": self.frozen, "degraded": list(self.degraded), "gallery_bytes": st.get("approx_bytes", 0),
                "scene": self.scene is not None, **{k: v for k, v in self.counters.items()}}

    def _log(self, row: dict) -> None:
        if self.log_writer is None or not self.cfg.log.enabled:
            return
        try:
            self.log_writer.write(row)
        except Exception as e:
            log.warning(f"identity log write failed: {e}")
