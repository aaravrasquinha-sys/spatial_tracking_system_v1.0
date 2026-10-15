"""InMemoryGallery: dict-backed default backend. Deterministic iteration (ordered by person id)."""
from __future__ import annotations

import json
from typing import Callable, Dict, Iterable, List, Optional

from poi_localization.identity.gallery.base import BaseGalleryBackend
from poi_localization.identity.types import ACTIVE, MISSING, PersonProfile


class InMemoryGallery(BaseGalleryBackend):
    def __init__(self):
        self._persons: Dict[int, PersonProfile] = {}
        self._track_to_person: Dict[int, int] = {}
        self._next_id = 1
        self._last_query_n = 0

    def new_person_id(self) -> int:
        pid = self._next_id
        self._next_id += 1
        return pid

    def put(self, profile: PersonProfile) -> None:
        self._persons[profile.person_id] = profile
        self._next_id = max(self._next_id, profile.person_id + 1)

    def get(self, person_id: int) -> Optional[PersonProfile]:
        return self._persons.get(person_id)

    def update(self, person_id: int, mutation: Callable[[PersonProfile], None]) -> None:
        p = self._persons.get(person_id)
        if p is not None:
            mutation(p)

    def link(self, person_id: int, world_track_id: int) -> None:
        if person_id not in self._persons:
            raise KeyError(f"unknown person {person_id}")
        self._track_to_person[world_track_id] = person_id

    def unlink(self, world_track_id: int) -> Optional[int]:
        return self._track_to_person.pop(world_track_id, None)

    def person_of_track(self, world_track_id: int) -> Optional[int]:
        return self._track_to_person.get(world_track_id)

    def mark_missing(self, person_id: int, t: float, exit_info: Dict, ttl_s: float) -> None:
        p = self._persons.get(person_id)
        if p is None:
            return
        p.state = MISSING
        p.last_seen_t = t
        p.expires_at = t + ttl_s
        p.exit_info = dict(exit_info)

    def mark_active(self, person_id: int, t: float) -> None:
        p = self._persons.get(person_id)
        if p is None:
            return
        p.state = ACTIVE
        p.last_seen_t = t
        p.expires_at = None

    def candidates(self, t: float, exclude_active: bool = True,
                   prefilter: Optional[Callable[[PersonProfile], bool]] = None) -> Iterable[PersonProfile]:
        linked = set(self._track_to_person.values())
        out = []
        for pid in sorted(self._persons):
            p = self._persons[pid]
            if p.state != MISSING:
                continue
            if p.expires_at is not None and p.expires_at <= t:
                continue
            if exclude_active and pid in linked:
                continue
            if prefilter is not None and not prefilter(p):
                continue
            out.append(p)
        self._last_query_n = len(out)
        return out

    def merge(self, src_person_id: int, dst_person_id: int, t: float) -> None:
        if src_person_id == dst_person_id:
            return
        src, dst = self._persons.get(src_person_id), self._persons.get(dst_person_id)
        if src is None or dst is None:
            raise KeyError("merge needs both persons to exist")
        for tid, pid in list(self._track_to_person.items()):
            if pid == src_person_id:
                self._track_to_person[tid] = dst_person_id
        dst.provenance.append({"merged_from": src_person_id, "t": t})
        del self._persons[src_person_id]

    def expire(self, t: float) -> List[int]:
        gone = [pid for pid in sorted(self._persons)
                if self._persons[pid].state == MISSING
                and self._persons[pid].expires_at is not None
                and self._persons[pid].expires_at <= t]
        for pid in gone:
            del self._persons[pid]
        return gone

    def snapshot(self) -> dict:
        return {"next_id": self._next_id,
                "persons": [self._persons[pid].to_dict() for pid in sorted(self._persons)],
                "links": {str(k): v for k, v in sorted(self._track_to_person.items())}}

    def restore(self, blob: dict) -> None:
        self._persons = {d["person_id"]: PersonProfile.from_dict(d) for d in blob["persons"]}
        self._track_to_person = {int(k): v for k, v in blob.get("links", {}).items()}
        self._next_id = blob["next_id"]

    def stats(self) -> dict:
        by_state: Dict[str, int] = {}
        for p in self._persons.values():
            by_state[p.state] = by_state.get(p.state, 0) + 1
        approx = sum(len(json.dumps(p.to_dict())) for p in self._persons.values())
        return {"persons": len(self._persons), "by_state": by_state, "approx_bytes": approx,
                "last_query_candidates": self._last_query_n}
