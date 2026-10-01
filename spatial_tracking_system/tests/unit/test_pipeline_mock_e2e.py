from poi_perception.capture.synthetic_source import SyntheticSource
from poi_perception.config import M3Config
from poi_perception.inference import mock_infer
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.runtime.m3_daemon import M3Pipeline
from poi_perception.runtime.masks import MaskRegion, MaskSet, empty_mask_set


def _run(cfg, scenario_fn, n_frames, mask_set=None):
    source = SyntheticSource(cfg.camera, n_frames=n_frames)
    estimator = PoseEstimator(mock_infer.MockPoseBackend(scenario_fn))
    pipeline = M3Pipeline(cfg, mask_set=mask_set or empty_mask_set())
    all_frames_dets = []
    for frame in source:
        raw_dets = estimator.infer(frame)
        dets = pipeline.process(frame, raw_dets)
        all_frames_dets.append(dets)
    return all_frames_dets, pipeline


def test_single_loop_mostly_tracks_via_ankles():
    cfg = M3Config.default()
    n_frames = 90
    scenario = mock_infer.scenario_single_loop(cfg.camera.width, cfg.camera.height, n_frames)
    all_dets, _ = _run(cfg, scenario, n_frames)

    ankles_count = sum(1 for dets in all_dets for d in dets if d.footpoint_source == "ankles")
    total = sum(len(dets) for dets in all_dets)
    assert total > 0
    assert ankles_count / total > 0.9  # "clear standing majority" (Section 9 done-when)

    # exactly one track for the whole scenario
    ids = {d.track_id_2d for dets in all_dets for d in dets}
    assert len(ids) == 1


def test_two_crossing_keeps_two_identities_most_of_the_time():
    cfg = M3Config.default()
    n_frames = 90
    scenario = mock_infer.scenario_two_crossing(cfg.camera.width, cfg.camera.height, n_frames)
    all_dets, _ = _run(cfg, scenario, n_frames)

    counts = [len(dets) for dets in all_dets]
    # for most of the sequence both people should be tracked as 2 people
    assert sum(1 for c in counts if c == 2) > n_frames * 0.6


def test_exit_reenter_produces_a_new_track_id_on_reentry():
    cfg = M3Config.default()
    n_frames = 90
    scenario = mock_infer.scenario_exit_reenter(cfg.camera.width, cfg.camera.height, n_frames)
    all_dets, pipeline = _run(cfg, scenario, n_frames)

    ids_in_order = []
    for dets in all_dets:
        for d in dets:
            if d.track_id_2d not in ids_in_order:
                ids_in_order.append(d.track_id_2d)
    # re-identification is out of v1 scope (Section 3) -- exit+reentry
    # must produce at least 2 distinct 2D track IDs, not 1.
    assert len(ids_in_order) >= 2


def test_sitting_scenario_degrades_footpoint_gracefully():
    cfg = M3Config.default()
    n_frames = 90
    scenario = mock_infer.scenario_sitting(cfg.camera.width, cfg.camera.height, n_frames)
    all_dets, _ = _run(cfg, scenario, n_frames)

    second_half = all_dets[n_frames // 2 :]
    sources = [d.footpoint_source for dets in second_half for d in dets]
    assert sources  # still produced detections while sitting
    # ankle confidence was dropped for the sitting half -- footpoint
    # should have fallen back, not silently kept reporting "ankles".
    assert all(s in ("bbox_fallback", "ankle_single") for s in sources)


def test_partial_occlusion_track_survives_via_lost_buffer():
    cfg = M3Config.default()
    cfg.track.lost_buffer_frames = 40  # generous enough to span the occlusion window
    n_frames = 90
    scenario = mock_infer.scenario_partial_occlusion(cfg.camera.width, cfg.camera.height, n_frames, occlude_frac=0.3)
    all_dets, _ = _run(cfg, scenario, n_frames)

    ids = {d.track_id_2d for dets in all_dets for d in dets}
    assert len(ids) == 1  # same identity survives the occlusion window

    lost_buffer_seen = any(d.track_state == "lost_buffer" for dets in all_dets for d in dets)
    assert lost_buffer_seen


def test_near_mirror_reflection_is_dropped_by_mask():
    cfg = M3Config.default()
    n_frames = 60
    mirror_bbox = (400, 50, 620, 300)
    scenario = mock_infer.scenario_near_mirror(cfg.camera.width, cfg.camera.height, n_frames, mirror_bbox)
    mask_set = MaskSet([MaskRegion(name="mirror", polygon=[(400, 50), (620, 50), (620, 300), (400, 300)])])

    all_dets, _ = _run(cfg, scenario, n_frames, mask_set=mask_set)

    # zero false persons from masked regions (Section 9 done-when)
    ids = {d.track_id_2d for dets in all_dets for d in dets}
    assert len(ids) == 1  # only the real person, never the reflection


def test_near_mirror_without_mask_would_produce_two_tracks():
    """Sanity check that the scenario itself is meaningful: without a
    mask, the reflection *is* a second track, so the masked test above
    is actually exercising the mask and not just an empty scenario."""
    cfg = M3Config.default()
    n_frames = 60
    mirror_bbox = (400, 50, 620, 300)
    scenario = mock_infer.scenario_near_mirror(cfg.camera.width, cfg.camera.height, n_frames, mirror_bbox)
    all_dets, _ = _run(cfg, scenario, n_frames, mask_set=empty_mask_set())
    ids = {d.track_id_2d for dets in all_dets for d in dets}
    assert len(ids) == 2
