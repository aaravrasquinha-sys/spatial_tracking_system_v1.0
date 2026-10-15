"""R1 Phase 1: config, TrackManager hooks, gallery contract, engine lifecycle, wire omission."""
import json
from types import SimpleNamespace

import pytest

from poi_localization.config import M4Config
from poi_localization.identity.config import IdentityConfig
from poi_localization.identity.engine import IdentityEngine
from poi_localization.identity.gallery.memory import InMemoryGallery
from poi_localization.identity.types import PersonProfile
from poi_localization.io.world_track import WorldTrackPhaseA, WorldTrackPhaseB
from poi_localization.measurement.types import Measurement
from poi_localization.tracking.track_manager import DetectionMeasurements, TrackManager

DT = 1.0 / 30.0


def _det(x, y, tid2d=1):
    m = Measurement(position_xy=(x, y), cov_xy=(0.01, 0.0, 0.01), src="depth")
    return DetectionMeasurements(track_id_2d=tid2d, det_conf=0.9, bbox_px=(0, 0, 60, 170),
                                 depth_measurement=m, raycast_measurement=None, height_sample_m=1.7)


def _cfg():
    c = M4Config.default()
    c.track.tentative_confirm_hits = 1
    return c


# ---------------- config ----------------
def test_identity_is_off_by_default_and_config_is_strict():
    assert M4Config.default().identity.enabled is False
    with pytest.raises(TypeError):
        IdentityConfig.from_dict({"nope": 1})
    c = IdentityConfig.from_dict({"enabled": True, "where": {"enabled": True}})
    assert c.enabled and c.where.enabled and c.p_commit == 0.97


def test_m4config_roundtrip_with_identity(tmp_path):
    p = tmp_path / "m4.json"
    c = M4Config.default()
    c.identity.enabled = True
    c.save(p)
    assert M4Config.load(p).identity.enabled is True
    old = json.loads(p.read_text())
    old.pop("identity")                       # a config written before R1 still loads
    p.write_text(json.dumps(old))
    assert M4Config.load(p).identity.enabled is False


# ---------------- TrackManager hooks ----------------
def test_last_assignment_and_new_tracks_index_the_input_list():
    tm = TrackManager(_cfg())
    tm.update(0.0, DT, [_det(1.0, 0.0), _det(3.0, 1.0)])
    assert sorted(tm.last_new_tracks.values()) == [0, 1] and tm.last_assignment == {}
    t1 = {tid: i for tid, i in tm.last_new_tracks.items()}
    tm.update(DT, DT, [_det(3.0, 1.0), _det(1.0, 0.0)])          # order swapped on purpose
    assert tm.last_new_tracks == {}
    for tid, i in tm.last_assignment.items():
        assert tid in t1 and i in (0, 1)
    pos = {tid: tm.snapshot()[tid]["position_xy"] for tid in tm.last_assignment}
    for tid, i in tm.last_assignment.items():
        assert abs(pos[tid][0] - (3.0 if i == 0 else 1.0)) < 0.2


def test_track_ended_is_emitted_exactly_once_after_lost_grace():
    cfg = _cfg()
    tm = TrackManager(cfg)
    tm.update(0.0, DT, [_det(1.0, 0.0)])
    t, ended = DT, []
    while t < 4.0:
        _, ev = tm.update(t, DT, [])
        ended += [e for e in ev if e.kind == "track_ended"]
        t += DT
    assert len(ended) == 1 and tm.active_track_count() == 0
    snap_keys = {"id", "state", "position_xy", "velocity_xy", "time_since_update_s", "last_2d_id", "last_bbox", "hits"}
    tm.update(t, DT, [_det(1.0, 0.0)])
    assert set(next(iter(tm.snapshot().values()))) == snap_keys


# ---------------- gallery contract ----------------
def _g():
    g = InMemoryGallery()
    for _ in range(3):
        pid = g.new_person_id()
        g.put(PersonProfile(person_id=pid, created_t=0.0))
    return g


def test_gallery_candidates_exclude_active_and_expired_and_are_ordered():
    g = _g()
    g.link(1, 10)
    g.mark_missing(1, 5.0, {}, ttl_s=45)
    assert [p.person_id for p in g.candidates(6.0)] == []         # still linked => excluded (W1)
    g.unlink(10)
    g.mark_missing(2, 5.0, {}, ttl_s=10)
    assert [p.person_id for p in g.candidates(6.0)] == [1, 2]
    assert [p.person_id for p in g.candidates(16.0)] == [1]       # 2 expired by TTL
    assert g.expire(16.0) == [2] and g.get(2) is None


def test_gallery_merge_snapshot_restore():
    g = _g()
    g.link(2, 7)
    g.merge(2, 1, t=9.0)
    assert g.get(2) is None and g.person_of_track(7) == 1
    assert g.get(1).provenance == [{"merged_from": 2, "t": 9.0}]
    h = InMemoryGallery()
    h.restore(json.loads(json.dumps(g.snapshot())))
    assert h.snapshot() == g.snapshot() and h.new_person_id() == g.new_person_id()


# ---------------- engine lifecycle ----------------
def _run(engine, tm, t, dets):
    outs, ev = tm.update(t, DT, dets)
    views = engine.step(None, t, {}, tm.snapshot(), ev, outs)
    return outs, views


def test_engine_assigns_new_person_then_marks_missing_then_expires():
    cfg = IdentityConfig(enabled=True, missing_ttl_s=2.0)
    eng, tm = IdentityEngine(cfg), TrackManager(_cfg())
    t = 0.0
    outs, views = _run(eng, tm, t, [_det(1.0, 0.0), _det(3.0, 1.0)])
    assert sorted(v.label for v in views.values()) == ["P1", "P2"]
    assert all(v.state == "new" for v in views.values())
    t += DT
    while t < 4.5:                                   # both leave; Kalman tracks die (~2 s), TTL 2 s more
        _run(eng, tm, t, [])
        t += DT
        if eng.health()["missing"] == 2:
            break
    assert eng.health()["missing"] == 2 and eng.health()["active"] == 0
    while t < 10.0:
        _run(eng, tm, t, [])
        t += DT
    assert eng.health()["missing"] == 0 and eng.counters["expired"] == 2
    outs, views = _run(eng, tm, t, [_det(1.0, 0.0)])
    assert next(iter(views.values())).label == "P3"  # ids are never reused


def test_freeze_blocks_commits_until_unfrozen():
    eng, tm = IdentityEngine(IdentityConfig(enabled=True)), TrackManager(_cfg())
    eng.freeze("calibration suspect")
    _, views = _run(eng, tm, 0.0, [_det(1.0, 0.0)])
    v = next(iter(views.values()))
    assert v.state == "resolving" and v.id is None
    eng.unfreeze()
    _, views = _run(eng, tm, DT, [_det(1.0, 0.0)])
    assert next(iter(views.values())).state == "new"


def test_engine_log_rows_are_written(tmp_path):
    from poi_localization.identity.io import IdentityJsonlWriter, read_identity_jsonl
    w = IdentityJsonlWriter(tmp_path / "id.jsonl")
    eng, tm = IdentityEngine(IdentityConfig(enabled=True), log_writer=w), TrackManager(_cfg())
    _run(eng, tm, 0.0, [_det(1.0, 0.0)])
    eng.close()
    kinds = [r["kind"] for r in read_identity_jsonl(tmp_path / "id.jsonl")]
    assert kinds == ["case_opened", "commit"]


# ---------------- wire / records ----------------
def test_person_field_is_omitted_when_none_and_present_when_set():
    a = WorldTrackPhaseA(0.0, 1, "confirmed", (0, 0, 0), (0, 0, 0), (1, 0, 1), 1.7, "depth", 1.0)
    b = WorldTrackPhaseB(1, "confirmed", (0, 0, 0), (0, 0, 0), (1, 0, 1), 1.7, 0.9, "depth", 1.0, (0, 0, 1, 1))
    assert "person" not in a.to_dict() and "person" not in b.to_dict()
    b.person = {"id": 1, "label": "P1", "state": "new", "conf": 1.0, "candidate": None}
    assert b.to_dict()["person"]["label"] == "P1"


def test_identity_exceptions_never_break_tracking_and_engine_disables_itself():
    from poi_localization.runtime.m4_pipeline import M4Pipeline

    class Boom:
        def step(self, *a, **k):
            raise RuntimeError("boom")

    pipe = M4Pipeline.__new__(M4Pipeline)            # exercise only the guard, no hardware/frames needed
    pipe.identity_engine, pipe.identity_errors = Boom(), {}
    pipe._identity_consecutive_failures, pipe.identity_disabled_reason = 0, None
    pipe.track_manager = SimpleNamespace(last_assignment={}, last_new_tracks={}, snapshot=lambda: {})
    for _ in range(25):
        assert pipe._run_identity(SimpleNamespace(t=0.0), [], [], [], []) == {}
    assert pipe.identity_errors == {"RuntimeError": 20} and pipe.identity_engine is None
    assert "20 consecutive" in pipe.identity_disabled_reason
