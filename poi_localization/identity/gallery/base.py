"""BaseGalleryBackend: the storage contract for person profiles (R1 spec section 11.1).

Any backend must pass the same contract suite (tests/unit/test_identity_phase1.py). Profiles are
plain dataclasses with a versioned to_dict/from_dict, so a backend can serialise them as it likes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional

from poi_localization.identity.types import PersonProfile


class BaseGalleryBackend(ABC):
    @abstractmethod
    def new_person_id(self) -> int:
        """Allocate the next person id (P1, P2, ... -- never reused within a session)."""

    @abstractmethod
    def put(self, profile: PersonProfile) -> None: ...

    @abstractmethod
    def get(self, person_id: int) -> Optional[PersonProfile]: ...

    @abstractmethod
    def update(self, person_id: int, mutation: Callable[[PersonProfile], None]) -> None:
        """Apply an in-place accumulator update to the stored profile."""

    @abstractmethod
    def link(self, person_id: int, world_track_id: int) -> None: ...

    @abstractmethod
    def unlink(self, world_track_id: int) -> Optional[int]:
        """Remove the track link; returns the person id that was linked (or None)."""

    @abstractmethod
    def person_of_track(self, world_track_id: int) -> Optional[int]: ...

    @abstractmethod
    def mark_missing(self, person_id: int, t: float, exit_info: Dict, ttl_s: float) -> None: ...

    @abstractmethod
    def mark_active(self, person_id: int, t: float) -> None: ...

    @abstractmethod
    def candidates(self, t: float, exclude_active: bool = True,
                   prefilter: Optional[Callable[[PersonProfile], bool]] = None) -> Iterable[PersonProfile]:
        """Missing, non-expired persons (the re-entry candidate set), in deterministic order."""

    @abstractmethod
    def merge(self, src_person_id: int, dst_person_id: int, t: float) -> None:
        """Fold `src` into `dst`; links of src move to dst; provenance is recorded on dst."""

    @abstractmethod
    def expire(self, t: float) -> List[int]:
        """Remove persons whose TTL has passed; returns their ids."""

    @abstractmethod
    def snapshot(self) -> dict: ...

    @abstractmethod
    def restore(self, blob: dict) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...
