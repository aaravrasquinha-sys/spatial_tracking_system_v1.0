"""
The only place that knows how to turn a frame's worth of
poi_localization records into poi.v1 wire messages. Pure functions,
no I/O, so they're trivial to test against the golden examples in
docs/schema_examples/ and reusable identically from the live hook,
the replay source, and the synthetic generator.

Two responsibilities:

  1. record_to_wire(): WorldTrackPhaseA | WorldTrackPhaseB -> TrackWire.
     Phase A has no conf/bbox -- those go out as None (schema.py turns
     that into JSON null, never a fabricated value).

  2. EventDeriver: M4Pipeline emits internal TrackEvent objects for
     some transitions (track_lost, track_confirmed) but not others
     (there is no track_end at all -- a track just stops appearing
     once poi_localization.tracking.track_manager.TrackManager deletes
     it). Rather than change M4, this diffs the track-id/state set
     between consecutive frames and derives the wire events from
     that -- works identically for live data, replayed JSONL, and the
     synthetic generator, and needs zero M4 changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from poi_present.schema import TrackWire, event_message


def record_to_wire(record: Any) -> TrackWire:
    """Accepts a WorldTrackPhaseA or WorldTrackPhaseB instance (or any
    duck-typed object/dict with the same fields) and returns the
    phase-independent TrackWire. This is the one spot where the
    p_local/p, v_local/v, world_track_id/id field-name split (Section
    10's deliberate Phase A vs B naming) gets resolved -- everything
    past this function only ever sees the wire shape."""
    if isinstance(record, dict):
        d = record
        has_p = "p" in d
    else:
        d = record.__dict__
        has_p = hasattr(record, "p")

    track_id = d.get("id", d.get("world_track_id"))
    p = d.get("p", d.get("p_local"))
    v = d.get("v", d.get("v_local"))
    height = d.get("height", d.get("height_m"))
    conf = d.get("conf")  # absent entirely on Phase A -> None, correct
    bbox = d.get("bbox")  # absent entirely on Phase A -> None, correct

    return TrackWire(
        id=int(track_id),
        state=str(d["state"]),
        p=tuple(p),
        v=tuple(v),
        cov_xy=tuple(d["cov_xy"]),
        height=None if height is None else float(height),
        conf=None if conf is None else float(conf),
        src=str(d["src"]),
        age_s=float(d["age_s"]),
        bbox=None if bbox is None else tuple(bbox),
    )


@dataclass
class _TrackMemory:
    state: str
    last_seen_t: float


class EventDeriver:
    """Diffs the track-id/state set between consecutive calls and
    yields the events a viewer needs: track_start (new id),
    track_confirmed (tentative -> confirmed), track_lost (-> lost),
    track_end (id vanished from the output entirely -- the one event
    M4 doesn't produce at all, because TrackManager just deletes the
    dict entry silently after coasting_budget_s + lost_grace_s)."""

    def __init__(self) -> None:
        self._known: Dict[int, _TrackMemory] = {}

    def diff(self, t: float, tracks: Sequence[TrackWire]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        seen_ids = set()

        for tr in tracks:
            seen_ids.add(tr.id)
            prev = self._known.get(tr.id)
            if prev is None:
                events.append(event_message(event="track_start", track_id=tr.id, t=t))
                if tr.state == "confirmed":
                    events.append(event_message(event="track_confirmed", track_id=tr.id, t=t))
            elif prev.state != tr.state:
                if tr.state == "confirmed" and prev.state != "confirmed":
                    events.append(event_message(event="track_confirmed", track_id=tr.id, t=t))
                elif tr.state == "lost" and prev.state != "lost":
                    events.append(event_message(event="track_lost", track_id=tr.id, t=t))
            self._known[tr.id] = _TrackMemory(state=tr.state, last_seen_t=t)

        # Any id we knew about last time but didn't see this frame has
        # been deleted by TrackManager -- that's track_end.
        vanished = [tid for tid in self._known if tid not in seen_ids]
        for tid in vanished:
            events.append(event_message(event="track_end", track_id=tid, t=t))
            del self._known[tid]

        return events

    def reset(self) -> None:
        """Call when the frame changes (e.g. a Phase A -> Phase B
        daemon restart) so stale ids from the previous session don't
        spuriously fire track_end on the next diff."""
        self._known.clear()


class SeqCounter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n
