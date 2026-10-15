"""R1 Phase 1 end to end through the real chain (synthetic site): identity on must not change tracking,
must label every published track, and must be deterministic."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _run(name, frames, identity):
    from sts.cli import _synthetic_parts
    from sts.runtime import build_chain, plan_run, run_frames
    from sts.site import SiteConfig
    overrides = {"phase_a": {"fixed_transform": {"height_m": 2.3, "pitch_deg": 20.0, "yaw_deg": 0.0}},
                 "track": {"tentative_confirm_hits": 2}}
    raw = {"localization": {"phase": "A", "overrides": overrides}}
    with tempfile.TemporaryDirectory() as d:
        raw["data_dir"] = d
        if identity:
            overrides["identity"] = {"enabled": True}
            raw["slots"] = {"identity_resolver": "multicue_v1", "identity_gallery": "memory"}
        site = SiteConfig.from_dict(raw)
        plan = plan_run(site, "cam0", write_run_manifest=False)
        src, be = _synthetic_parts(site, "cam0", name, frames)
        chain = build_chain(plan, serve=False, source=src, backend=be, write_logs=False)
        rows = []

        class Rec:
            cam_id, frame_name, map_id = "cam0", "local", None

            def submit(self, frame, records):
                rows.append([(r.world_track_id, tuple(round(float(x), 4) for x in r.p_local), getattr(r, "person", None))
                             for r in records])
        chain.live_source = Rec()
        run_frames(chain, max_frames=frames)
        return rows, chain.identity


def test_identity_on_does_not_change_tracking_and_labels_every_track():
    off, eng_off = _run("two_crossing", 120, identity=False)
    on, eng_on = _run("two_crossing", 120, identity=True)
    assert eng_off is None and eng_on is not None
    assert [[(i, p) for i, p, _ in fr] for fr in off] == [[(i, p) for i, p, _ in fr] for fr in on]
    assert all(person is None for fr in off for _, _, person in fr)
    published = [x for fr in on for x in fr]
    assert published and all(person is not None and person["state"] in ("new", "resolving") for _, _, person in published)
    labels = {person["label"] for _, _, person in published if person["label"]}
    assert labels and all(l.startswith("P") for l in labels)


def test_identity_is_deterministic():
    a, _ = _run("two_crossing", 90, identity=True)
    b, _ = _run("two_crossing", 90, identity=True)
    assert a == b
