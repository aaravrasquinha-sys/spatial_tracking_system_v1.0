"""R1 in Phase B on the real synthetic site: configgen points identity at the anchor bundle's empty-room references,
the engine loads them, tracking is unchanged with identity on, and the consistency gate only WARNS when they are missing."""
import json

from sts.configgen import write_configs
from sts.consistency import check_phase_b
from sts.runtime import build_chain, plan_run, run_frames


class Recorder:
    def __init__(self, inner):
        self.inner, self.frames = inner, []
        self.cam_id, self.frame_name, self.map_id = inner.cam_id, inner.frame_name, inner.map_id

    def submit(self, frame, records):
        self.frames.append([(r.id, tuple(round(float(x), 4) for x in r.p), getattr(r, "person", None)) for r in records])


def _run(synth, identity, n=40):
    from sts.cli import _synthetic_parts
    synth.site.watchdog.enabled = False
    if identity:
        synth.site.localization.overrides = {**synth.site.localization.overrides, "identity": {"enabled": True}}
        synth.site.slots["identity_resolver"] = "multicue_v1"
    plan = plan_run(synth.site, "cam0", write_run_manifest=False)
    src, be = _synthetic_parts(plan.site, "cam0", "two_crossing", n)
    chain = build_chain(plan, serve=False, source=src, backend=be, write_logs=False)
    chain.live_source = Recorder(chain.live_source)
    run_frames(chain, max_frames=n)
    return chain


def test_configgen_fills_identity_reference_paths_in_phase_b_but_leaves_it_disabled(synth):
    g = write_configs(synth.site, "cam0", phase="B", map_id=synth.map_id)
    ident = json.loads(g.m4_path.read_text())["identity"]
    assert ident["enabled"] is False
    assert ident["reference_depth_path"] == str(synth.site.reference_depth_path("cam0"))
    assert ident["static_capture_path"] == str(synth.site.static_capture_path("cam0"))
    assert synth.site.reference_depth_path("cam0").exists() and synth.site.static_capture_path("cam0").exists()


def test_phase_b_identity_loads_the_anchor_references_and_does_not_change_tracking(synth_master, tmp_path):
    import shutil
    from sts.site import SiteConfig
    from tests.synth.site_fixture import SynthSite

    def copy(name):
        dst = tmp_path / name
        shutil.copytree(synth_master.root, dst)
        (dst / "site.json").write_text((dst / "site.json").read_text().replace(str(synth_master.root), str(dst)))
        return SynthSite(dst, SiteConfig.load(dst / "site.json"), dst / "site.json", synth_master.world,
                         synth_master.T_map_room, True, synth_master.map_id)

    off = _run(copy("off"), identity=False)
    on = _run(copy("on"), identity=True)
    assert off.identity is None and on.identity is not None
    h = on.identity.health()
    assert h["scene"] is True and "no_reference_depth" not in h["degraded"], h
    assert h["extract_errors"] == 0
    strip = lambda frames: [[(i, p) for i, p, _ in fr] for fr in frames]
    assert strip(off.live_source.frames) == strip(on.live_source.frames)
    assert any(person for fr in on.live_source.frames for _, _, person in fr)
    assert all(person is None for fr in off.live_source.frames for _, _, person in fr)


def test_consistency_warns_but_does_not_block_when_identity_references_are_missing(synth):
    synth.site.localization.overrides = {**synth.site.localization.overrides, "identity": {"enabled": True}}
    synth.site.static_capture_path("cam0").unlink()
    g = write_configs(synth.site, "cam0", phase="B", map_id=synth.map_id)
    rep = check_phase_b(synth.site, "cam0", g.m4_path, g.present_path)
    assert rep.ok, rep.render()
    assert "identity static capture (colour reference) exists" in {c.name for c in rep.warnings}
