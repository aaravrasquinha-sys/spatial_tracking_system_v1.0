"""The live chain (sts.runtime), hardware-free: SyntheticSource + MockPoseBackend through the REAL M3, M4,
watchdog supervisor and M5 LiveSource, in Phase A (no map) and Phase B (real synthetic site)."""
import json

import pytest

from poi_perception.inference import mock_infer
from sts.consistency import ConsistencyError
from sts.runtime import build_chain, plan_run, run_frames
from sts.site import SiteConfig


class Recorder:
    """Stands in for LiveSource: records exactly what would be published."""
    def __init__(self, inner):
        self.inner, self.frames = inner, []
        self.cam_id, self.frame_name, self.map_id = inner.cam_id, inner.frame_name, inner.map_id

    def submit(self, frame, records):
        self.frames.append((frame.t, list(records)))


def _parts(site, n=60, scenario="two_crossing"):
    from sts.cli import _synthetic_parts
    return _synthetic_parts(site, "cam0", scenario, n)


def _phase_a_site(tmp_path, **kw):
    return SiteConfig.from_dict({"data_dir": str(tmp_path / "d"), "localization": {"phase": "A", "overrides": {
        "phase_a": {"fixed_transform": {"height_m": 2.3, "pitch_deg": 20.0, "yaw_deg": 0.0}},
        "track": {"tentative_confirm_hits": 2}}}, **kw})


def _run(site_or_plan, n=60, scenario="two_crossing", watchdog_layers=None, **bk):
    plan = site_or_plan if hasattr(site_or_plan, "gen") else plan_run(site_or_plan, "cam0", write_run_manifest=False)
    src, be = _parts(plan.site, n, scenario)
    chain = build_chain(plan, serve=False, source=src, backend=be, inline_watchdog=True, write_logs=False, **bk)
    chain.live_source = Recorder(chain.live_source)
    run_frames(chain, max_frames=n)
    return chain


def test_phase_a_runs_end_to_end_with_no_map_and_no_calibration(tmp_path):
    chain = _run(_phase_a_site(tmp_path), n=60)
    assert chain.frames_seen >= 55
    assert any(recs for _, recs in chain.live_source.frames), "two_crossing must produce confirmed world tracks"
    assert chain.calib.state() == "missing", "Phase A is a provisional frame, not a registered one"
    assert chain.live_source.frame_name == "local" and chain.supervisor is None
    assert chain.frames_suppressed == 0


def test_phase_b_plan_carries_the_real_map_id_into_live_source(synth):
    plan = plan_run(synth.site, "cam0", write_run_manifest=False)
    assert plan.phase == "B" and plan.map_id == synth.map_id and plan.report.ok
    synth.site.watchdog.enabled = False
    chain = _run(plan, n=10)
    assert chain.live_source.map_id == synth.map_id and chain.live_source.frame_name == "room"
    assert chain.m4.floor_transform.map_id == synth.map_id
    assert chain.calib.state() == "ok"


def test_startup_verification_suppresses_3d_when_live_depth_does_not_match_the_reference(synth):
    """The synthetic source's flat depth does NOT match the room's reference depth -> the supervisor must
    refuse to publish 3D (policy 'suppress') while still tracking and logging."""
    synth.site.slots["watchdog_layers"] = "depth"
    chain = _run(plan_run(synth.site, "cam0", write_run_manifest=False), n=40)
    assert chain.calib.state() == "suspect"
    assert chain.frames_suppressed > 0
    late = chain.live_source.frames[-10:]
    assert all(recs == [] for _, recs in late), "no 3D position may be published while calibration is suspect"
    assert chain.m4.total_events or chain.frames_seen > 0      # M4 itself kept running


def test_flag_policy_publishes_tracks_even_when_suspect(synth):
    synth.site.slots["watchdog_layers"] = "depth"
    synth.site.watchdog.on_suspect = "flag"
    chain = _run(plan_run(synth.site, "cam0", write_run_manifest=False), n=40)
    assert chain.calib.state() == "suspect" and chain.frames_suppressed == 0
    assert any(recs for _, recs in chain.live_source.frames)


def test_tilt_layer_without_a_source_is_refused(synth):
    synth.site.slots["watchdog_layers"] = "depth+tilt"
    plan = plan_run(synth.site, "cam0", write_run_manifest=False)
    src, be = _parts(synth.site, 5)
    from sts.site import SiteError
    with pytest.raises(SiteError, match="tilt"):
        build_chain(plan, serve=False, source=src, backend=be, write_logs=False)


@pytest.mark.parametrize("breakage", ["map_id", "no_calibration", "unaccepted"])
def test_inconsistent_phase_b_refuses_before_anything_is_built(synth, breakage):
    p = synth.site.calibration_path("cam0")
    if breakage == "no_calibration":
        p.unlink()
    else:
        c = json.loads(p.read_text())
        if breakage == "map_id":
            c["map_id"] = "deadbeef0000"
        else:
            c["accepted"] = False
        p.write_text(json.dumps(c))
    with pytest.raises(ConsistencyError) as e:
        plan_run(synth.site, "cam0", write_run_manifest=False)
    assert "BLOCKED" in str(e.value)


def test_run_manifest_ties_a_log_to_code_config_and_data(synth):
    plan = plan_run(synth.site, "cam0", write_run_manifest=True)
    m = json.loads(plan.manifest_path.read_text())
    assert m["map_id"] == synth.map_id and m["phase"] == "B" and m["cam_id"] == "cam0"
    assert set(m["config_hashes"]) == {"m3", "m4", "present"} and all(m["config_hashes"].values())
    assert m["calibration"]["sha"] and m["sts_version"]
    assert "modified_legacy_files" in m


def test_replay_is_deterministic(tmp_path):
    """Same input -> byte-identical world-track log (M6's 'replays deterministically to the same tracks')."""
    outs = []
    for i in range(2):
        site = _phase_a_site(tmp_path / f"r{i}")
        plan = plan_run(site, "cam0", write_run_manifest=False)
        src, be = _parts(site, 90, "two_crossing")
        chain = build_chain(plan, serve=False, source=src, backend=be, write_logs=True)
        path = chain.m4.writer.base_path
        run_frames(chain, max_frames=90)
        rows = [json.loads(l) for l in open(path)]
        for r in rows:
            r.pop("map_id", None)
        outs.append(rows)
    assert outs[0] and outs[0] == outs[1]


def test_world_track_summary_and_baseline_compare(tmp_path):
    from sts.replay import compare, summarize
    site = _phase_a_site(tmp_path)
    plan = plan_run(site, "cam0", write_run_manifest=False)
    src, be = _parts(site, 90)
    chain = build_chain(plan, serve=False, source=src, backend=be, write_logs=True)
    path = chain.m4.writer.base_path
    run_frames(chain, max_frames=90)
    s = summarize(path)
    assert s["distinct_track_ids"] >= 1
    assert compare(s, s)["regressed"] is False
    worse = dict(s, distinct_track_ids=s["distinct_track_ids"] + 3)
    assert compare(s, worse)["regressed"] is True
