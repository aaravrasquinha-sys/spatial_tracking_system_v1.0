import numpy as np
import pytest

from poi_localization.config import M4Config
from poi_localization.measurement.types import Measurement
from poi_localization.tracking.gates import WalkableGrid
from poi_localization.tracking.track_manager import DetectionMeasurements, TrackManager

DT = 1.0 / 30.0
TIGHT_COV = (0.01, 0.0, 0.01)


def _det(x, y, track_id_2d=1, conf=0.9, src="depth", cov=TIGHT_COV, height=1.7, bbox=(0, 0, 60, 170)):
    m = Measurement(position_xy=(x, y), cov_xy=cov, src=src)
    return DetectionMeasurements(
        track_id_2d=track_id_2d,
        det_conf=conf,
        bbox_px=bbox,
        depth_measurement=m if src == "depth" else None,
        raycast_measurement=m if src == "raycast" else None,
        height_sample_m=height,
    )


def _cfg(**overrides):
    cfg = M4Config.default()
    for k, v in overrides.items():
        setattr(cfg.track, k, v)
    return cfg


def test_tentative_then_confirmed_after_enough_hits():
    cfg = _cfg(tentative_confirm_hits=3)
    tm = TrackManager(cfg)
    t = 0.0
    for i in range(3):
        outputs, _ = tm.update(t, DT, [_det(1.0 + 0.01 * i, 0.0)])
        t += DT
        if i < 2:
            assert outputs == []  # still tentative -- not published
        else:
            assert len(outputs) == 1
            assert outputs[0].state == "confirmed"


def test_tentative_track_dies_on_single_miss():
    cfg = _cfg(tentative_confirm_hits=3)
    tm = TrackManager(cfg)
    tm.update(0.0, DT, [_det(1.0, 0.0)])
    assert tm.active_track_count() == 1
    tm.update(DT, DT, [])  # miss
    assert tm.active_track_count() == 0


def test_confirmed_track_coasts_through_short_gap_then_resumes():
    cfg = _cfg(tentative_confirm_hits=2, coasting_budget_s=1.0)
    tm = TrackManager(cfg)
    t = 0.0
    outputs = []
    for _ in range(2):
        outputs, _ = tm.update(t, DT, [_det(1.0, 0.0)])
        t += DT
    world_id = outputs[0].world_track_id
    assert outputs[0].state == "confirmed"

    # 5 frames with no detection at all, well within coasting_budget_s
    for _ in range(5):
        outputs, _ = tm.update(t, DT, [])
        t += DT
        assert len(outputs) == 1
        assert outputs[0].world_track_id == world_id
        assert outputs[0].state == "coasting"
        assert outputs[0].src == "predicted"

    outputs, _ = tm.update(t, DT, [_det(1.0, 0.0)])
    assert len(outputs) == 1
    assert outputs[0].world_track_id == world_id
    assert outputs[0].state == "confirmed"


def test_track_becomes_lost_once_then_removed_after_grace():
    cfg = _cfg(tentative_confirm_hits=1, coasting_budget_s=0.05, lost_grace_s=0.05)
    tm = TrackManager(cfg)
    t = 0.0
    outputs, _ = tm.update(t, DT, [_det(1.0, 0.0)])
    t += DT
    world_id = outputs[0].world_track_id

    saw_lost = False
    for _ in range(20):
        outputs, events = tm.update(t, DT, [])
        t += DT
        states = [o.state for o in outputs]
        if "lost" in states:
            saw_lost = True
            assert states.count("lost") == 1  # published exactly once
        else:
            assert outputs == [] or "coasting" in states
        if tm.active_track_count() == 0:
            break
    assert saw_lost
    assert tm.active_track_count() == 0


def test_measurement_rejected_event_logged_for_bad_secondary_measurement():
    cfg = _cfg(tentative_confirm_hits=1)
    tm = TrackManager(cfg)
    tm.update(0.0, DT, [_det(1.0, 0.0, src="depth")])

    # depth agrees with the track; raycast is wildly off -- should be
    # individually rejected (but depth still applied) and logged.
    good_depth = Measurement(position_xy=(1.02, 0.0), cov_xy=TIGHT_COV, src="depth")
    bad_raycast = Measurement(position_xy=(50.0, 50.0), cov_xy=TIGHT_COV, src="raycast")
    det = DetectionMeasurements(
        track_id_2d=1, det_conf=0.9, bbox_px=(0, 0, 60, 170),
        depth_measurement=good_depth, raycast_measurement=bad_raycast, height_sample_m=1.7,
    )
    outputs, events = tm.update(DT, DT, [det])
    assert len(outputs) == 1
    assert outputs[0].src == "depth"  # only depth was actually applied
    rejected = [e for e in events if e.kind == "measurement_rejected"]
    assert len(rejected) == 1
    assert "raycast" in rejected[0].detail


def test_2d_id_mismatch_is_logged_but_world_id_stays_stable():
    cfg = _cfg(tentative_confirm_hits=1)
    tm = TrackManager(cfg)
    outputs, _ = tm.update(0.0, DT, [_det(1.0, 0.0, track_id_2d=7)])
    world_id = outputs[0].world_track_id

    # geometrically the same person, but M3 assigned a different 2D id
    # this frame (e.g. a brief 2D ID swap during a crossing) -- world
    # track should NOT split, and the disagreement should be logged.
    outputs, events = tm.update(DT, DT, [_det(1.02, 0.0, track_id_2d=9)])
    assert len(outputs) == 1
    assert outputs[0].world_track_id == world_id
    mismatches = [e for e in events if e.kind == "2d_id_mismatch"]
    assert len(mismatches) == 1


def test_crossing_two_people_keep_distinct_world_ids_even_with_swapped_2d_hints():
    cfg = _cfg(tentative_confirm_hits=1)
    tm = TrackManager(cfg)
    # two people, well separated in world position
    outputs, _ = tm.update(0.0, DT, [_det(1.0, 0.0, track_id_2d=1), _det(1.0, 3.0, track_id_2d=2)])
    assert len(outputs) == 2
    id_near_zero = next(o.world_track_id for o in outputs if abs(o.position_xy[1]) < 1.0)
    id_near_three = next(o.world_track_id for o in outputs if abs(o.position_xy[1] - 3.0) < 1.0)
    assert id_near_zero != id_near_three

    # next frame: both move slightly, but M3's 2D ids got swapped between
    # them (simulating a 2D ID swap at a crossing) -- world association
    # must go by position, not by the (wrong) 2D hint.
    outputs, events = tm.update(DT, DT, [_det(1.0, 0.05, track_id_2d=2), _det(1.0, 2.95, track_id_2d=1)])
    assert len(outputs) == 2
    new_id_near_zero = next(o.world_track_id for o in outputs if abs(o.position_xy[1]) < 1.0)
    new_id_near_three = next(o.world_track_id for o in outputs if abs(o.position_xy[1] - 3.0) < 1.0)
    assert new_id_near_zero == id_near_zero
    assert new_id_near_three == id_near_three
    assert len(events) >= 2  # both should log a 2d_id_mismatch


def test_walkable_gate_filters_measurement_and_prevents_any_track():
    cfg = _cfg(tentative_confirm_hits=1)
    grid = WalkableGrid(origin_xy=(0.0, -5.0), resolution_m=0.05, grid=np.zeros((200, 200), dtype=bool))
    tm = TrackManager(cfg)
    # target position (1.0, 0.0) is NOT walkable (grid is all False) --
    # e.g. a mirror reflection landing behind a wall (Section 8).
    outputs, _ = tm.update(0.0, DT, [_det(1.0, 0.0)], walkable_grid=grid)
    assert outputs == []
    assert tm.active_track_count() == 0


def test_walkable_gate_passes_through_on_marked_walkable_cell():
    cfg = _cfg(tentative_confirm_hits=1)
    grid = WalkableGrid(origin_xy=(0.0, -5.0), resolution_m=0.05, grid=np.ones((200, 200), dtype=bool))
    tm = TrackManager(cfg)
    outputs, _ = tm.update(0.0, DT, [_det(1.0, 0.0)], walkable_grid=grid)
    assert len(outputs) == 1
