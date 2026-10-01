from poi_present.adapter import EventDeriver, record_to_wire


def test_record_to_wire_phase_a_dict():
    rec = {
        "t_capture": 10.0,
        "world_track_id": 3,
        "state": "confirmed",
        "p_local": (1.0, 2.0, 0.0),
        "v_local": (0.1, 0.2, 0.0),
        "cov_xy": (0.01, 0.0, 0.01),
        "height_m": 1.7,
        "src": "fused",
        "age_s": 5.0,
    }
    wire = record_to_wire(rec)
    assert wire.id == 3
    assert wire.p == (1.0, 2.0, 0.0)
    assert wire.v == (0.1, 0.2, 0.0)
    assert wire.height == 1.7
    assert wire.conf is None  # Phase A has no conf field at all
    assert wire.bbox is None  # nor bbox


def test_record_to_wire_phase_b_dict():
    rec = {
        "id": 9,
        "state": "coasting",
        "p": (2.0, -1.0, 0.0),
        "v": (0.0, 0.0, 0.0),
        "cov_xy": (0.02, 0.0, 0.02),
        "height": 1.6,
        "conf": 0.77,
        "src": "predicted",
        "age_s": 4.4,
        "bbox": (10.0, 20.0, 30.0, 40.0),
    }
    wire = record_to_wire(rec)
    assert wire.id == 9
    assert wire.conf == 0.77
    assert wire.bbox == (10.0, 20.0, 30.0, 40.0)


def test_record_to_wire_dataclass_instance():
    from dataclasses import dataclass

    @dataclass
    class FakePhaseA:
        world_track_id: int
        state: str
        p_local: tuple
        v_local: tuple
        cov_xy: tuple
        height_m: float
        src: str
        age_s: float

    rec = FakePhaseA(1, "tentative", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.1), 1.5, "depth", 0.1)
    wire = record_to_wire(rec)
    assert wire.id == 1
    assert wire.state == "tentative"


def _wire(id_, state):
    return record_to_wire({
        "id": id_, "state": state, "p": (0, 0, 0), "v": (0, 0, 0),
        "cov_xy": (0, 0, 0), "height": None, "conf": None, "src": "fused",
        "age_s": 0.0, "bbox": None,
    })


def test_event_deriver_new_track_emits_start():
    ed = EventDeriver()
    events = ed.diff(1.0, [_wire(1, "tentative")])
    kinds = [e["event"] for e in events]
    assert kinds == ["track_start"]


def test_event_deriver_new_confirmed_track_emits_start_and_confirmed():
    ed = EventDeriver()
    events = ed.diff(1.0, [_wire(1, "confirmed")])
    kinds = [e["event"] for e in events]
    assert kinds == ["track_start", "track_confirmed"]


def test_event_deriver_full_lifecycle():
    ed = EventDeriver()
    e1 = [e["event"] for e in ed.diff(0.0, [_wire(1, "tentative")])]
    e2 = [e["event"] for e in ed.diff(0.1, [_wire(1, "confirmed")])]
    e3 = [e["event"] for e in ed.diff(1.5, [_wire(1, "lost")])]
    e4 = [e["event"] for e in ed.diff(2.5, [])]  # id 1 vanished entirely

    assert e1 == ["track_start"]
    assert e2 == ["track_confirmed"]
    assert e3 == ["track_lost"]
    assert e4 == ["track_end"]


def test_event_deriver_multiple_tracks_independent():
    ed = EventDeriver()
    events = ed.diff(0.0, [_wire(1, "confirmed"), _wire(2, "tentative")])
    kinds = sorted(e["event"] + str(e["id"]) for e in events)
    assert kinds == ["track_confirmed1", "track_start1", "track_start2"]

    events2 = ed.diff(0.1, [_wire(1, "confirmed")])  # id 2 vanished
    assert [e["event"] for e in events2] == ["track_end"]
    assert events2[0]["id"] == 2


def test_event_deriver_reset_clears_state():
    ed = EventDeriver()
    ed.diff(0.0, [_wire(1, "confirmed")])
    ed.reset()
    events = ed.diff(1.0, [_wire(1, "confirmed")])
    # After reset, id 1 looks brand new again -- correct behaviour
    # across a Phase A -> Phase B daemon restart.
    assert [e["event"] for e in events] == ["track_start", "track_confirmed"]
