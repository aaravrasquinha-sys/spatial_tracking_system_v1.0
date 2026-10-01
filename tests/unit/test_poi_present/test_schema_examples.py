"""
Every golden message in docs/schema_examples/ must be exactly
reproducible by schema.py's own builders (shape-for-shape, not
byte-for-byte -- key order doesn't matter). This is what keeps
SCHEMA.md, schema.py, and the JS clients honest against each other:
if someone changes a field name in schema.py without updating the
golden file (or vice versa), this test catches it, not a confused
person staring at the dashboard three weeks later.
"""
import json
from pathlib import Path

import pytest

from poi_present.schema import (
    TrackWire,
    event_message,
    health_message,
    pong_message,
    tracks_message,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "docs" / "schema_examples"


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text())


def test_tracks_confirmed_matches_golden():
    golden = _load("tracks_confirmed.json")
    wire = TrackWire(
        id=7, state="confirmed", p=(2.31, 1.08, 0.0), v=(0.42, -0.05, 0.0),
        cov_xy=(0.004, 0.0003, 0.0031), height=1.71, conf=None, src="fused",
        age_s=12.4, bbox=None,
    )
    msg = tracks_message(
        map_id=None, cam_id="cam0", frame="local", seq=18233,
        t_capture=1790061364.1234, t_publish=1790061364.1712, tracks=[wire],
    )
    assert msg == golden


def test_tracks_empty_matches_golden():
    golden = _load("tracks_empty.json")
    msg = tracks_message(
        map_id=None, cam_id="cam0", frame="local", seq=18234,
        t_capture=1790061364.1567, t_publish=1790061364.1998, tracks=[],
    )
    assert msg == golden
    assert msg["tracks"] == []  # "nobody here", not "no message" -- see SCHEMA.md


def test_tracks_room_phase_b_matches_golden():
    golden = _load("tracks_room_phase_b.json")
    wire = TrackWire(
        id=2, state="coasting", p=(1.0, -0.5, 0.0), v=(0.0, 0.0, 0.0),
        cov_xy=(0.02, 0.0, 0.02), height=1.65, conf=0.81, src="predicted",
        age_s=3.9, bbox=(300.0, 90.0, 400.0, 400.0),
    )
    msg = tracks_message(
        map_id="a3f9c2e1b7d4", cam_id="cam0", frame="room", seq=501,
        t_capture=1790062000.0, t_publish=1790062000.031, tracks=[wire],
    )
    assert msg == golden


def test_event_track_start_matches_golden():
    golden = _load("event_track_start.json")
    assert event_message(event="track_start", track_id=7, t=1790061364.1) == golden


def test_event_track_end_matches_golden():
    golden = _load("event_track_end.json")
    assert event_message(event="track_end", track_id=7, t=1790061377.4) == golden


def test_health_matches_golden():
    golden = _load("health_ok.json")
    msg = health_message(
        fps=29.8, calib="missing", map_id=None, temp_c=54.2,
        drop_rate=0.0, latency_ms=41.3, clients=2, clock="synced",
    )
    assert msg == golden


def test_pong_matches_golden():
    golden = _load("pong.json")
    assert pong_message(client_t=1790061364.001, server_t=1790061364.014) == golden


def test_tracks_message_rejects_unknown_frame():
    with pytest.raises(AssertionError):
        tracks_message(map_id=None, cam_id="cam0", frame="galaxy", seq=1, t_capture=0, t_publish=0, tracks=[])


def test_event_message_rejects_unknown_kind():
    with pytest.raises(AssertionError):
        event_message(event="teleported", track_id=1, t=0)
