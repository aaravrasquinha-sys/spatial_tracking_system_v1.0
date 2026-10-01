"""Known defects in the uploaded legacy code, pinned so they cannot be forgotten OR silently worked around.
Each test documents a defect (docs/OPEN_ITEMS.md) and flips to a normal pass the day you fix it."""
import re
from pathlib import Path

import pytest

from sts.adapters.offline import PyslamLiveMapBuilder, _legacy_defines
from sts.site import SiteConfig, SiteError

ROOT = Path(__file__).resolve().parents[2]


def test_K1_lockstep_locking_helper_is_undefined():
    """run_live_map.py calls _lock_and_write() in --mode lockstep but defines it nowhere (NameError after mapping).
    When you fix it, this test and the guard in sts/adapters/offline.py both relax automatically."""
    src = (ROOT / "run_live_map.py").read_text()
    called = "_lock_and_write(" in src
    defined = _legacy_defines("run_live_map.py", "_lock_and_write")
    if called and not defined:
        pytest.xfail("K1: _lock_and_write is called but never defined (docs/OPEN_ITEMS.md)")
    assert defined or not called


def test_sts_map_refuses_lockstep_while_K1_is_open(tmp_path):
    site = SiteConfig.from_dict({"data_dir": str(tmp_path)})
    if _legacy_defines("run_live_map.py", "_lock_and_write"):
        PyslamLiveMapBuilder().build(site, "cam0", extra_args=["--mode", "lockstep"])      # fixed -> allowed
    else:
        with pytest.raises(SiteError, match="_lock_and_write"):
            PyslamLiveMapBuilder().build(site, "cam0", extra_args=["--mode", "lockstep"])
    PyslamLiveMapBuilder().build(site, "cam0", extra_args=["--mode", "live"])              # live mode always allowed


def test_K2_run_live_map_hardcodes_intrinsics_in_fork_mode():
    """In --mode live --mp-start-method fork the parent skips probing and uses fixed numbers. `sts doctor --camera`
    compares the DEVICE's intrinsics to site.json; this pins the numbers it must agree with."""
    src = (ROOT / "run_live_map.py").read_text()
    m = re.search(r"Intrinsics\(fx=([\d.]+), fy=([\d.]+), cx=([\d.]+), cy=([\d.]+)", src)
    assert m, "the hardcoded-intrinsics line moved; update docs/OPEN_ITEMS.md K2"
    assert tuple(map(float, m.groups())) == (606.75, 606.57, 320.19, 237.06)
    cam = SiteConfig().cameras[0]
    assert (cam.expected_fx, cam.expected_fy, cam.expected_cx, cam.expected_cy) == (606.75, 606.57, 320.19, 237.06)


def test_K3_default_mp_start_method_is_fork_not_spawn_as_the_legacy_docs_claim():
    src = (ROOT / "run_live_map.py").read_text()
    m = re.search(r'--mp-start-method".*?default="(\w+)"', src, re.S)
    assert m and m.group(1) == "fork", "default changed: update docs/RUNBOOK.md (M1) and OPEN_ITEMS K3"
