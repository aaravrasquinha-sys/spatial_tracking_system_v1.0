"""R1: `person` on the wire is optional and additive; identity events derive from per-track transitions."""
import json
from pathlib import Path

from poi_localization.io.world_track import WorldTrackPhaseB
from poi_present.adapter import EventDeriver, record_to_wire
from poi_present.schema import TrackWire, event_message, tracks_message

EXAMPLES = Path(__file__).resolve().parents[3] / "docs" / "schema_examples"


def _wire(tid=9, state="confirmed", person=None):
    return TrackWire(id=tid, state=state, p=(1.2, 0.4, 0.0), v=(0.3, 0.0, 0.0), cov_xy=(0.01, 0.0, 0.01),
                     height=1.72, conf=0.9, src="fused", age_s=2.0, bbox=(210.0, 80.0, 330.0, 410.0), person=person)


def _person(pid, state, conf=0.98, **extra):
    return {"id": pid, "label": None if pid is None else f"P{pid}", "state": state, "conf": conf,
            "candidate": None, **extra}


def test_golden_examples_match_the_builders():
    person = {"id": 1, "label": "P1", "state": "matched", "conf": 0.983, "candidate": None}
    msg = tracks_message(map_id="a3f9c2e1b7d4", cam_id="cam0", frame="room", seq=812, t_capture=1790070000.0,
                         t_publish=1790070000.029, tracks=[_wire(person=person)])
    assert msg == json.loads((EXAMPLES / "tracks_identity.json").read_text())
    ev = event_message(event="identity_matched", track_id=9, t=1790070000.0, detail="P1 conf=0.98", person_id=1)
    assert ev == json.loads((EXAMPLES / "event_identity_matched.json").read_text())


def test_person_is_omitted_from_the_wire_when_absent():
    assert "person" not in _wire().to_dict()
    msg = event_message(event="track_end", track_id=1, t=0.0)
    assert "person_id" not in msg and "from_person_id" not in msg


def test_record_to_wire_copies_person_from_records_and_dicts():
    rec = WorldTrackPhaseB(1, "confirmed", (0, 0, 0), (0, 0, 0), (1, 0, 1), 1.7, 0.9, "depth", 1.0, (0, 0, 1, 1),
                           person=_person(3, "new"))
    assert record_to_wire(rec).person["label"] == "P3"
    assert record_to_wire(rec.to_dict()).person["id"] == 3
    rec.person = None
    assert record_to_wire(rec).person is None


def _kinds(evs):
    return [(e["event"], e.get("person_id"), e.get("from_person_id")) for e in evs if e["event"].startswith("identity")]


def test_identity_events_follow_transitions():
    d = EventDeriver()
    assert _kinds(d.diff(0.0, [_wire(person=_person(None, "resolving", 0.0))])) == []          # resolving: silent
    assert _kinds(d.diff(0.1, [_wire(person=_person(1, "new"))])) == [("identity_new", 1, None)]
    assert _kinds(d.diff(0.2, [_wire(person=_person(1, "new"))])) == []                         # no repeat
    # a different track commits to an existing person
    d.diff(0.3, [_wire(person=_person(1, "new")), _wire(tid=10, state="tentative", person=_person(None, "resolving"))])
    evs = d.diff(0.4, [_wire(person=_person(1, "new")), _wire(tid=10, person=_person(2, "matched"))])
    assert _kinds(evs) == [("identity_matched", 2, None)]
    # ambiguous (provisional id 5) later merged into person 2 on the same track
    d.diff(0.5, [_wire(tid=11, person=_person(5, "ambiguous", 0.5))])
    evs = d.diff(0.6, [_wire(tid=11, person=_person(2, "matched"))])
    assert _kinds(evs) == [("identity_merged", 2, 5)]


def test_conflict_marker_is_emitted_once_per_increment_and_no_person_means_no_identity_events():
    d = EventDeriver()
    d.diff(0.0, [_wire(person=_person(1, "new"))])
    marked = _person(1, "new", event_seq=1, last_event="identity_conflict")
    assert _kinds(d.diff(0.1, [_wire(person=marked)])) == [("identity_conflict", 1, None)]
    assert _kinds(d.diff(0.2, [_wire(person=marked)])) == []
    swapped = _person(1, "new", event_seq=2, last_event="identity_swapped")
    assert _kinds(d.diff(0.3, [_wire(person=swapped)])) == [("identity_swapped", 1, None)]
    d2 = EventDeriver()
    assert _kinds(d2.diff(0.0, [_wire()])) == [] and _kinds(d2.diff(0.1, [_wire()])) == []
