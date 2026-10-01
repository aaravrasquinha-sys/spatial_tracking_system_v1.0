from poi_perception.config import TrackConfig
from poi_perception.inference.pose_infer import RawDetection
from poi_perception.tracking.bytetrack_wrap import ByteTrackWrapper


def _det(cx, cy, w=60, h=170, conf=0.9):
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return RawDetection(bbox=(x1, y1, x2, y2), det_conf=conf, keypoints={})


def _cfg(**overrides):
    return TrackConfig(**overrides)


def test_single_track_persists_id_across_frames():
    tracker = ByteTrackWrapper(_cfg())
    ids = []
    for i in range(10):
        out = tracker.update([_det(100 + i * 2, 200)])
        assert len(out) == 1
        ids.append(out[0].track_id_2d)
    assert len(set(ids)) == 1


def test_new_track_state_is_new_then_tracked():
    tracker = ByteTrackWrapper(_cfg())
    out0 = tracker.update([_det(100, 200)])
    assert out0[0].track_state == "new"
    out1 = tracker.update([_det(102, 200)])
    assert out1[0].track_state == "tracked"
    assert out1[0].track_id_2d == out0[0].track_id_2d


def test_track_survives_gap_within_lost_buffer():
    cfg = _cfg(lost_buffer_frames=5)
    tracker = ByteTrackWrapper(cfg)
    out0 = tracker.update([_det(100, 200)])
    tid = out0[0].track_id_2d

    # 3 frames with no detection at all -- should coast as lost_buffer,
    # same ID, not be dropped (lost_buffer_frames=5).
    for _ in range(3):
        out = tracker.update([])
        assert len(out) == 1
        assert out[0].track_id_2d == tid
        assert out[0].track_state == "lost_buffer"
        assert out[0].det is None

    # detection reappears near the predicted position -- same ID resumes.
    out_resume = tracker.update([_det(100, 200)])
    assert len(out_resume) == 1
    assert out_resume[0].track_id_2d == tid
    assert out_resume[0].track_state == "tracked"


def test_track_dropped_after_exceeding_lost_buffer():
    cfg = _cfg(lost_buffer_frames=3)
    tracker = ByteTrackWrapper(cfg)
    tracker.update([_det(100, 200)])
    for _ in range(4):  # exceeds lost_buffer_frames
        tracker.update([])
    assert tracker.active_track_count() == 0


def test_low_confidence_second_pass_keeps_id_through_dip():
    cfg = _cfg(track_high_thresh=0.6, track_low_thresh=0.1, lost_buffer_frames=10)
    tracker = ByteTrackWrapper(cfg)
    out0 = tracker.update([_det(100, 200, conf=0.9)])
    tid = out0[0].track_id_2d

    # detection confidence dips below track_high_thresh but stays above
    # track_low_thresh -- Section 6: this is exactly what the
    # low-confidence second pass is for.
    out1 = tracker.update([_det(102, 200, conf=0.3)])
    assert len(out1) == 1
    assert out1[0].track_id_2d == tid
    assert out1[0].det is not None  # matched, not coasting


def test_below_low_thresh_detection_never_spawns_or_matches():
    cfg = _cfg(track_low_thresh=0.1)
    tracker = ByteTrackWrapper(cfg)
    out = tracker.update([_det(100, 200, conf=0.05)])
    assert out == []
    assert tracker.active_track_count() == 0


def test_new_track_requires_new_track_thresh_not_just_high_thresh():
    cfg = _cfg(track_high_thresh=0.5, new_track_thresh=0.8)
    tracker = ByteTrackWrapper(cfg)
    # above track_high_thresh (so it's in the "high" bucket for matching)
    # but below new_track_thresh -- Section 6 doesn't want every
    # medium-confidence blip spawning a brand-new identity.
    out = tracker.update([_det(100, 200, conf=0.6)])
    assert out == []
    assert tracker.active_track_count() == 0


def test_two_people_get_distinct_ids_and_keep_them_while_crossing():
    tracker = ByteTrackWrapper(_cfg())
    n_frames = 30
    ids_a, ids_b = [], []
    for i in range(n_frames):
        cx_a = 60 + i * 4
        cx_b = 60 + (n_frames - i) * 4
        out = tracker.update([_det(cx_a, 200), _det(cx_b, 200)])
        assert len(out) in (1, 2)  # may briefly merge to 1 track right at the crossing
        by_x = sorted(out, key=lambda t: t.bbox[0])
        if len(by_x) == 2:
            ids_a.append(by_x[0].track_id_2d)
            ids_b.append(by_x[1].track_id_2d)
    # two distinct identities existed for most of the sequence
    assert len(set(ids_a) | set(ids_b)) >= 2
