from poi_perception.capture.synthetic_source import SyntheticSource
from poi_perception.config import M3Config
from poi_perception.eval.grade import summarize_log
from poi_perception.eval.replay_diff import diff_summaries
from poi_perception.inference import mock_infer
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.io.detection2d import JsonlDetectionWriter
from poi_perception.runtime.m3_daemon import M3Pipeline
from poi_perception.runtime.masks import empty_mask_set


def _record_scenario(tmp_path, n_frames=60):
    cfg = M3Config.default()
    scenario = mock_infer.scenario_single_loop(cfg.camera.width, cfg.camera.height, n_frames)
    source = SyntheticSource(cfg.camera, n_frames=n_frames)
    estimator = PoseEstimator(mock_infer.MockPoseBackend(scenario))
    jsonl_path = tmp_path / "log.jsonl"
    writer = JsonlDetectionWriter(jsonl_path)
    pipeline = M3Pipeline(cfg, mask_set=empty_mask_set(), writer=writer)
    for frame in source:
        raw_dets = estimator.infer(frame)
        pipeline.process(frame, raw_dets)
    writer.close()
    return jsonl_path


def test_summarize_log_basic_counts(tmp_path):
    path = _record_scenario(tmp_path)
    summary = summarize_log(path)
    assert summary.n_frames == 60
    assert summary.distinct_track_ids == 1
    assert summary.footpoint_source_counts.get("ankles", 0) > 0


def test_diff_summaries_flags_regression():
    baseline = {"new_track_events": 1, "low_confidence_rows": 0, "lost_buffer_rows": 0}
    current_ok = {"new_track_events": 1, "low_confidence_rows": 0, "lost_buffer_rows": 0}
    current_bad = {"new_track_events": 3, "low_confidence_rows": 0, "lost_buffer_rows": 0}

    results_ok = diff_summaries(baseline, current_ok)
    assert not any(r.regressed for r in results_ok)

    results_bad = diff_summaries(baseline, current_bad)
    assert any(r.regressed and r.metric == "new_track_events" for r in results_bad)
