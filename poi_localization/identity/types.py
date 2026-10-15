"""Plain dataclasses shared by the identity package (R1 spec sections 4 and 11)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PROFILE_SCHEMA = 1

# person states (gallery)
ACTIVE = "active"
MISSING = "missing"
EXPIRED = "expired"

# identity states of a track (what consumers see)
RESOLVING = "resolving"
MATCHED = "matched"
NEW = "new"
AMBIGUOUS = "ambiguous"


@dataclass
class IdentityView:
    """Per-track identity as published on the wire (`tracks[i].person`)."""

    id: Optional[int]
    label: Optional[str]
    state: str
    conf: float = 0.0
    candidate: Optional[int] = None

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "state": self.state,
                "conf": float(self.conf), "candidate": self.candidate}


@dataclass
class PersonProfile:
    """Accumulated knowledge about one person. Phase 1 carries only bookkeeping; later phases
    fill `structure` / `outfits` / `shape`. Stable to_dict/from_dict so any backend can serialise it."""

    person_id: int
    created_t: float
    state: str = ACTIVE
    last_seen_t: float = 0.0
    expires_at: Optional[float] = None
    exit_info: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    outfits: List[Any] = field(default_factory=list)
    shape: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    profile_schema: int = PROFILE_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PersonProfile":
        return cls(**d)


@dataclass
class TrackSnapshot:
    """One world track as the identity layer sees it (built from TrackManager.snapshot())."""

    world_track_id: int
    state: str
    position_xy: tuple
    velocity_xy: tuple
    time_since_update_s: float
    last_2d_id: Optional[int]
    last_bbox: Optional[tuple]
    hits: int

    @classmethod
    def from_dict(cls, d: dict) -> "TrackSnapshot":
        return cls(world_track_id=d["id"], state=d["state"], position_xy=tuple(d["position_xy"]),
                   velocity_xy=tuple(d["velocity_xy"]), time_since_update_s=d["time_since_update_s"],
                   last_2d_id=d["last_2d_id"], last_bbox=d["last_bbox"], hits=d["hits"])
