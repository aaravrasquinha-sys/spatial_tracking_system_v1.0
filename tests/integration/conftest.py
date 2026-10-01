import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synth_master(tmp_path_factory):
    """One real synthetic site (real M2 solve), built once per session (~20 s)."""
    from tests.synth.site_fixture import build_synth_site
    root = tmp_path_factory.mktemp("synth_master")
    s = build_synth_site(root)
    assert s.accepted, "the synthetic calibration must be ACCEPTED by M2's own gates"
    return s


@pytest.fixture()
def synth(synth_master, tmp_path):
    """A private COPY of the synthetic site for tests that mutate files."""
    from sts.site import SiteConfig
    from tests.synth.site_fixture import SynthSite
    dst = tmp_path / "site"
    shutil.copytree(synth_master.root, dst)
    old = str(synth_master.root)
    txt = (dst / "site.json").read_text().replace(old, str(dst))
    (dst / "site.json").write_text(txt)
    site = SiteConfig.load(dst / "site.json")
    return SynthSite(dst, site, dst / "site.json", synth_master.world, synth_master.T_map_room, True, synth_master.map_id)
