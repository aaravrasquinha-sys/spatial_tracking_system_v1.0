"""The Phase-B gate must BLOCK every way the map / calibration / configs can disagree."""
import json
import shutil

import pytest

from sts.configgen import write_configs
from sts.consistency import check_phase_b


def _gate(synth, mutate_cfg=None):
    g = write_configs(synth.site, "cam0", phase="B", map_id=synth.map_id)
    if mutate_cfg:
        mutate_cfg(g)
    return check_phase_b(synth.site, "cam0", g.m4_path, g.present_path)


def _failed(rep):
    return {c.name for c in rep.errors}


def test_consistent_site_passes(synth):
    rep = _gate(synth)
    assert rep.ok, rep.render()
    assert rep.map_id == synth.map_id


def test_calibration_from_another_map_is_blocked(synth):
    p = synth.site.calibration_path("cam0")
    c = json.loads(p.read_text()); c["map_id"] = "deadbeef0000"; p.write_text(json.dumps(c))
    assert "calibration map_id == map bundle map_id" in _failed(_gate(synth))


def test_unaccepted_calibration_is_blocked(synth):
    p = synth.site.calibration_path("cam0")
    c = json.loads(p.read_text()); c["accepted"] = False; p.write_text(json.dumps(c))
    assert "calibration.accepted is true" in _failed(_gate(synth))


def test_only_a_rejected_calibration_present_is_blocked_with_a_clear_message(synth):
    p = synth.site.calibration_path("cam0")
    p.rename(p.with_name("calibration.cam0.REJECTED.json"))
    rep = _gate(synth)
    assert not rep.ok and any("REJECTED" in c.detail for c in rep.errors)


def test_map_edited_after_locking_is_blocked(synth):
    from sts.adapters.offline import ManifestLockFinalizer
    ManifestLockFinalizer().finalize(synth.site, read_only=False)
    ply = synth.site.map_dir() / "dense" / "points.ply"
    ply.write_bytes(ply.read_bytes() + b"\0")            # one stray byte
    assert "map bundle unchanged since locking" in _failed(_gate(synth))


def test_site_json_map_id_mismatch_is_blocked(synth):
    synth.site.map.map_id = "000000000000"
    assert "site.json map.map_id matches the map bundle" in _failed(_gate(synth))


def test_m5_pointed_at_the_map_dir_is_blocked(synth):
    def mutate(g):
        d = json.loads(g.present_path.read_text()); d["scene"]["map_bundle_dir"] = str(synth.site.map_dir()); g.present_path.write_text(json.dumps(d))
    assert "M5 map_bundle_dir is the ANCHOR dir, not the map dir" in _failed(_gate(synth, mutate))


def test_m4_with_wrong_expected_map_id_is_blocked(synth):
    def mutate(g):
        d = json.loads(g.m4_path.read_text()); d["phase_b"]["expected_map_id"] = "000000000000"; g.m4_path.write_text(json.dumps(d))
    assert "M4 expected_map_id == map_id" in _failed(_gate(synth, mutate))


def test_m5_scene_map_id_or_frame_mismatch_is_blocked(synth):
    def mutate(g):
        d = json.loads(g.present_path.read_text()); d["scene"]["map_id"] = "x"; d["scene"]["frame"] = "local"; g.present_path.write_text(json.dumps(d))
    f = _failed(_gate(synth, mutate))
    assert "M5 scene map_id == map_id" in f and "M5 frame is 'room'" in f


def test_missing_walkable_and_reference_depth_are_blocked(synth):
    synth.site.walkable_path("cam0").unlink()
    synth.site.reference_depth_path("cam0").unlink()
    f = _failed(_gate(synth))
    assert "walkable.json exists with {origin, resolution, grid}" in f
    assert "reference depth exists (calibration watchdog)" in f


def test_missing_reference_depth_is_only_a_warning_if_watchdog_disabled(synth):
    synth.site.watchdog.enabled = False
    synth.site.reference_depth_path("cam0").unlink()
    rep = _gate(synth)
    assert rep.ok and any(c.name.startswith("reference depth") for c in rep.warnings)


def test_no_calibration_at_all_is_blocked(synth):
    shutil.rmtree(synth.anchor_dir)
    rep = _gate(synth)
    assert not rep.ok and "accepted calibration exists" in _failed(rep)


def test_intrinsics_disagreement_warns_but_does_not_block(synth):
    synth.site.cameras[0].expected_fx = 700.0
    rep = _gate(synth)
    assert rep.ok and any("intrinsics" in c.name for c in rep.warnings)
