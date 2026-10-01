import os
import stat

import pytest

from sts.registry import SlotError, SlotRegistry, default_registry
from sts.slots import SLOTS


# ---------------- map finalizer + the map_id rule ----------------
def test_lock_keeps_the_map_id_that_anchors_were_solved_against(synth):
    """THE rule (contracts/map_bundle.md): locking must NOT change map_id, or every anchor is invalidated."""
    from pyslam.anchor.map_io import resolve_map_id
    from sts.adapters.offline import ManifestLockFinalizer
    before = resolve_map_id(str(synth.site.map_dir()))
    assert before["source"].startswith("computed")
    info = ManifestLockFinalizer().finalize(synth.site, read_only=False)
    after = resolve_map_id(str(synth.site.map_dir()))
    assert info["map_id"] == before["map_id"] == after["map_id"] == synth.map_id
    assert after["source"] == "manifest.json" and after["mismatch"] is False


def test_lock_writes_a_valid_manifest_and_is_idempotent(synth):
    import json
    from sts.adapters.offline import ManifestLockFinalizer
    f = ManifestLockFinalizer()
    a = f.finalize(synth.site, read_only=False)
    b = f.finalize(synth.site, read_only=False)
    assert b["already_locked"] is True and a["map_id"] == b["map_id"]
    jsonschema = pytest.importorskip("jsonschema")
    from sts.contracts import validate_file
    assert validate_file(a["manifest_path"], "map_manifest") == []


def test_lock_makes_bundle_read_only(synth):
    from sts.adapters.offline import ManifestLockFinalizer
    ManifestLockFinalizer().finalize(synth.site, read_only=True)
    ply = synth.site.map_dir() / "dense" / "points.ply"
    assert not (ply.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(ply, 0o644)   # let the tmp dir be cleaned up


def test_relocking_an_edited_bundle_refuses(synth):
    from sts.adapters.offline import ManifestLockFinalizer
    from sts.site import SiteError
    f = ManifestLockFinalizer()
    f.finalize(synth.site, read_only=False)
    ply = synth.site.map_dir() / "dense" / "points.ply"
    ply.write_bytes(ply.read_bytes() + b"\0")
    with pytest.raises(SiteError, match="edited"):
        f.finalize(synth.site, read_only=False)


# ---------------- slots ----------------
def test_every_slot_has_a_built_default():
    assert default_registry().default_complete() == []


def test_reserved_names_are_refused_and_not_registered():
    reg = default_registry()
    for slot, spec in SLOTS.items():
        for name in spec.reserved:
            with pytest.raises(SlotError, match="reserved"):
                reg.resolve(slot, name)
            assert name not in reg.names(slot)
    with pytest.raises(SlotError):
        SlotRegistry().register("anchor_solver", "fpfh_cross_check", lambda: None)


def test_unknown_slot_or_impl_is_an_error():
    reg = default_registry()
    with pytest.raises(SlotError):
        reg.resolve("nope")
    with pytest.raises(SlotError, match="no implementation"):
        reg.resolve("map_finalizer", "does_not_exist")


def test_a_replacement_implementation_can_be_selected_per_site(synth):
    """The whole point of slots: swap an implementation without touching any module."""
    from sts.adapters import register_all

    class FakeFinalizer:
        def finalize(self, site, read_only=True):
            return {"map_id": "feedfacecafe", "manifest_path": None, "already_locked": False}

    reg = SlotRegistry(); register_all(reg)
    reg.register("map_finalizer", "custom", FakeFinalizer)
    synth.site.slots["map_finalizer"] = "custom"
    chosen = reg.create("map_finalizer", synth.site.slots.get("map_finalizer"))
    assert chosen.finalize(synth.site)["map_id"] == "feedfacecafe"
    assert default_registry().names("map_finalizer") == ["manifest_lock", "noop"], "default registry untouched"


def test_watchdog_layer_specs_are_validated():
    from sts.adapters.runtime import DepthIcpLayers
    from sts.site import SiteError
    assert DepthIcpLayers().parse("depth+icp") == {"depth", "icp"}
    with pytest.raises(SiteError):
        DepthIcpLayers().parse("depth+magic")


def test_anchor_solver_builds_per_camera_commands(synth):
    from sts.adapters.offline import PyslamAnchorSolver
    s = PyslamAnchorSolver()
    c = s.command("capture", synth.site, "cam0", frames=50, warmup_s=5)
    assert c.needs_camera and "--realsense" in c.argv and "--capture-profile" in c.argv
    assert str(synth.site.capture_profile_path()) in c.argv
    assert s.command("solve", synth.site, "cam0").needs_camera is False
    assert not s.command("capture", synth.site, "cam0", bag="x.bag").needs_camera
    assert str(synth.site.anchor_dir("cam0")) in s.command("solve", synth.site, "cam0").argv


def test_map_builder_forces_callback_imu_and_records_by_default(synth):
    from sts.adapters.offline import PyslamLiveMapBuilder
    c = PyslamLiveMapBuilder().build(synth.site, "cam0")
    assert c.needs_camera and "--record" in c.argv
    assert c.argv[c.argv.index("--imu-mode") + 1] == "callback", "'polling' is rejected by RealSenseSource"
    assert "--record" not in PyslamLiveMapBuilder().build(synth.site, "cam0", record=False).argv
