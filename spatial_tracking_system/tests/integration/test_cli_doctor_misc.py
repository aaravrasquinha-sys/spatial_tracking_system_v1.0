import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sts import cli
from sts.doctor import Doctor
from sts.paths import REPO_ROOT
from sts.retention import plan_prune, prune
from sts.site import SiteConfig


def test_cli_init_slots_boundaries_provenance(tmp_path, capsys):
    site = tmp_path / "site.json"
    assert cli.main(["--site", str(site), "init", "--name", "t1"]) == 0
    s = SiteConfig.load(site)
    assert s.site == "t1" and (s.data_root() / "locks").exists()
    assert cli.main(["--site", str(site), "init"]) == 1                      # refuses to overwrite
    assert cli.main(["--site", str(site), "slots"]) == 0
    assert "map_finalizer" in capsys.readouterr().out
    assert cli.main(["boundaries"]) == 0
    assert cli.main(["--site", str(site), "provenance"]) == 0


def test_cli_check_and_accept_on_a_real_synthetic_site(synth, capsys):
    assert cli.main(["--site", str(synth.site_path), "check"]) == 0
    assert "RESULT: OK" in capsys.readouterr().out
    synth.site.map.map_id = None
    synth.site.save(synth.site_path)
    assert cli.main(["--site", str(synth.site_path), "accept"]) == 0
    assert SiteConfig.load(synth.site_path).map.map_id == synth.map_id


def test_cli_map_lock_records_the_map_id(synth):
    synth.site.map.map_id = None; synth.site.save(synth.site_path)
    assert cli.main(["--site", str(synth.site_path), "map-lock", "--writable"]) == 0
    assert SiteConfig.load(synth.site_path).map.map_id == synth.map_id
    assert (synth.site.map_dir() / "manifest.json").exists()


def test_cli_run_refuses_inconsistent_phase_b_with_exit_2(synth, capsys):
    c = json.loads(synth.site.calibration_path("cam0").read_text()); c["map_id"] = "deadbeef0000"
    synth.site.calibration_path("cam0").write_text(json.dumps(c))
    rc = cli.main(["--site", str(synth.site_path), "run", "--phase", "B", "--source", "synthetic"])
    assert rc == 2
    assert "REFUSING TO START" in capsys.readouterr().err


def test_cli_run_refuses_when_the_camera_is_busy(tmp_path):
    from sts.camera_lock import CameraLock
    site = SiteConfig.from_dict({"data_dir": str(tmp_path / "d"), "localization": {"phase": "A", "overrides": {
        "phase_a": {"fixed_transform": {"height_m": 2.3, "pitch_deg": 20.0, "yaw_deg": 0.0}}}}})
    p = tmp_path / "s.json"; site.save(p)
    with CameraLock(site.data_root() / "locks", "default", "map"):
        assert cli.main(["--site", str(p), "run", "--phase", "A"]) == 4


def test_unknown_anchor_hint_and_missing_markers_are_errors(synth):
    assert cli.main(["--site", str(synth.site_path), "anchor", "markers"]) == 2


def test_doctor_reports_and_fails_only_on_real_problems(tmp_path):
    s = SiteConfig.from_dict({"data_dir": str(tmp_path)})
    d = Doctor(s); d.run()
    names = {r.name: r for r in d.results}
    assert names["import poi_perception resolves inside this repo"].level == "PASS"
    assert names["TensorRT engine file"].level == "FAIL"               # no engine yet
    assert names["camera hardware checks"].level == "SKIP"
    assert not d.ok and "FAIL" in d.render()


def test_doctor_catches_engine_camera_size_mismatch(tmp_path):
    s = SiteConfig.from_dict({"data_dir": str(tmp_path)})
    eng = s.engine_path(); eng.parent.mkdir(parents=True, exist_ok=True); eng.write_bytes(b"x")
    eng.with_suffix(".manifest.json").write_text(json.dumps({"img_size_hw": [640, 640], "jetpack_version": None}))
    d = Doctor(s); d.run()
    r = {x.name: x for x in d.results}["engine input size == camera size"]
    assert r.level == "FAIL" and "640" in r.detail


def test_doctor_flags_a_missing_site_file():
    d = Doctor(None); d.run()
    assert any(r.name == "site.json" and r.level == "FAIL" for r in d.results)


def test_retention_deletes_only_old_files(tmp_path):
    s = SiteConfig.from_dict({"data_dir": str(tmp_path), "retention": {"track_log_days": 14, "clip_days": 30, "bag_days": 7}})
    wt = s.logs_dir() / "cam0" / "world_tracks"; wt.mkdir(parents=True)
    old, new = wt / "old.jsonl", wt / "new.jsonl"
    for f in (old, new):
        f.write_text("{}")
    t_old = time.time() - 20 * 86400
    os.utime(old, (t_old, t_old))
    rec = s.recordings_dir(); rec.mkdir(parents=True)
    bag = rec / "a.bag"; bag.write_bytes(b"0"); os.utime(bag, (time.time() - 8 * 86400,) * 2)
    items = plan_prune(s)
    assert {p.name for _, p in items} == {"old.jsonl", "a.bag"}
    prune(s, dry_run=True); assert old.exists()
    prune(s, dry_run=False); assert not old.exists() and not bag.exists() and new.exists()


def test_provenance_baseline_covers_every_legacy_file_and_reports_shape():
    from sts import provenance
    d = provenance.compare()
    assert set(d) == {"modified", "missing", "added"}
    base = provenance.load_baseline()
    assert "poi_present/server/app.py" in base and "pyslam/anchor/calibrate.py" in base and "run_anchor.py" in base
    assert d["missing"] == [], "a legacy file was deleted"


def test_legacy_scripts_resolve_the_capture_profile_from_the_new_layout():
    """run_anchor.py / run_live_map.py compute their default profile path relative to __file__; the merge
    edits that one line in each. If it resolves to a missing file they silently fall back to CaptureProfile()
    defaults and the map<->anchor capture_profile_match gate degrades."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ra", REPO_ROOT / "run_anchor.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert os.path.exists(m._DEFAULT_PROFILE), m._DEFAULT_PROFILE
    txt = (REPO_ROOT / "run_live_map.py").read_text()
    assert "capture_profile.mapping.json" in txt
    from sts.paths import default_capture_profile
    assert default_capture_profile().exists() and os.path.samefile(m._DEFAULT_PROFILE, default_capture_profile())


def test_python_dash_m_sts_works_as_a_subprocess():
    r = subprocess.run([sys.executable, "-m", "sts", "--version"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("sts ")
