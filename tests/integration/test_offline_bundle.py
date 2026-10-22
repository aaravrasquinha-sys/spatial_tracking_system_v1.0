"""The offline mapper is Module 1: its bundle must be exactly what Module 2 reads and what map-lock hashes."""
import json
import os

import numpy as np
import pytest

from pyslam.core import lie


def _fake_run(tmp_path, first_x=0.5):
    from pyslam.offline.fuse import _write_ply_points
    from pyslam.offline.rundir import write_tum
    off = tmp_path / "run" / "offline"
    off.mkdir(parents=True)
    g = np.stack(np.meshgrid(np.arange(0, 2, 0.01), np.arange(0, 1, 0.01)), -1).reshape(-1, 2)
    pts = np.c_[g[:, 0], g[:, 1], np.full(len(g), 2.0)]
    _write_ply_points(str(off / "map_cloud.ply"), pts, np.full((len(pts), 3), 120, np.uint8))
    T0 = lie.make_T(np.eye(3), np.array([first_x, 0, 0]))
    write_tum(str(off / "trajectory_final_frames.tum"), [(0.0, T0), (1.0, T0)])
    (off / "report.json").write_text(json.dumps({"evidence": {"frames": 2, "keyframes": 1, "frames_lost": 0},
                                                 "loops": {"loops_final": 3},
                                                 "trajectory": {"start_to_end_gap_m": 0.1}}))
    return str(tmp_path / "run")


def test_bundle_is_in_w_cam0_and_map_id_survives_locking(tmp_path):
    from pyslam.anchor.map_io import load_map_bundle, resolve_map_id
    from pyslam.live.capture_profile import CaptureProfile
    from pyslam.mapping.lock import lock_bundle
    from pyslam.offline.bundle import write_bundle
    out = str(tmp_path / "bundle")
    summ = write_bundle(_fake_run(tmp_path), out, None)
    b = load_map_bundle(out)
    assert abs(b["pts_map"][:, 0].min() + 0.5) < 1e-3          # rebased so the first camera is the origin
    before = resolve_map_id(out)["map_id"]
    sha = CaptureProfile.load(os.path.join(out, "capture_profile.json")).content_hash()
    info = lock_bundle(out, ["dense/points.ply", "capture_profile.json"], sha, site_frame_summary={},
                       plane_summary=[], qa_snapshot=summ, builder_repo_root=".", make_read_only=False)
    after = resolve_map_id(out)
    assert before == info["map_id"] == after["map_id"] and after["source"] == "manifest.json"


def test_empty_map_writes_no_bundle(tmp_path):
    from pyslam.offline.bundle import write_bundle
    from pyslam.offline.fuse import _write_ply_points
    run = _fake_run(tmp_path)
    _write_ply_points(os.path.join(run, "offline", "map_cloud.ply"), np.zeros((0, 3)), np.zeros((0, 3), np.uint8))
    with pytest.raises(ValueError, match="empty"):
        write_bundle(run, str(tmp_path / "b"), None)


def test_offline_builder_is_the_default_and_needs_no_camera(tmp_path):
    from sts.adapters.offline import PyslamOfflineMapBuilder
    from sts.slots import SLOTS
    from sts.site import SiteError

    class _Site:
        def recordings_dir(self): return tmp_path / "rec"
        def capture_profile_path(self): return tmp_path / "p.json"

    assert SLOTS["map_builder"].default == "pyslam_offline"
    bag = tmp_path / "a.db3"
    bag.write_bytes(b"x")
    cmd = PyslamOfflineMapBuilder().build(_Site(), None, extra_args=["--bag", str(bag), "--top-k", "7"], out_dir=tmp_path / "m")
    assert not cmd.needs_camera and "run_map.py" in cmd.argv[1] and "--bundle" in cmd.argv and "7" in cmd.argv
    with pytest.raises(SiteError, match="no recording"):
        PyslamOfflineMapBuilder().build(_Site(), None, extra_args=[], out_dir=tmp_path / "m")
