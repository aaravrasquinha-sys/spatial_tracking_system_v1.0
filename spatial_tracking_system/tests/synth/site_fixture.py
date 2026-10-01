"""A complete synthetic SITE with a KNOWN answer, built through the real production code:

    analytic room --sample--> map bundle (maps/<site>/dense/points.ply + capture_profile.json, in an awkward W_cam0-like frame)
    analytic room --render--> static captures from a known camera pose
    pyslam.anchor.calibrate / write_anchor_bundle  --> anchors/<site>/cam0/ (calibration, walkable, room_frame, reference depth, viewer cloud)
    sts.site.SiteConfig                            --> configs pointing at all of it

No mocks of M1/M2 internals: the map hashes with M1's compute_map_id, the calibration is solved by M2's
solver and gated by M2's gates, and everything downstream (consistency gate, M4 Phase B loader, M5 scene)
reads the files M2 actually wrote. Building it takes ~1 min single-core, so tests share one session copy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pyslam.anchor.bundle import write_anchor_bundle
from pyslam.anchor.calibrate import calibrate
from pyslam.anchor.config import AnchorConfig
from pyslam.anchor.map_io import load_map_bundle
from pyslam.core import lie
from pyslam.live.capture_profile import CaptureProfile
from pyslam.mapping.export import write_ply
from sts.site import SiteConfig
from tests.synth.anchor_world import INTR, awkward_map_frame, cam_pose_room, default_world, render_capture

T_TRUE_ROOM_CAM = cam_pose_room(1.0, 3.4, 2.3, -25, 28)   # same pose test_g_anchor uses: floor + two walls + furniture in view


@dataclass
class SynthSite:
    root: Path
    site: SiteConfig
    site_path: Path
    world: object
    T_map_room: np.ndarray          # the awkward map frame
    accepted: bool
    map_id: str
    cam_id: str = "cam0"

    @property
    def anchor_dir(self) -> Path:
        return self.site.anchor_dir(self.cam_id)


def write_map_bundle(map_dir: Path, world, T_map_room: np.ndarray) -> None:
    P, C = world.sample_surface()
    pts_map = lie.transform_points(T_map_room, P)
    (map_dir / "dense").mkdir(parents=True, exist_ok=True)
    write_ply(str(map_dir / "dense" / "points.ply"), pts_map.astype(np.float32), np.clip(C, 0, 255).astype(np.uint8))
    CaptureProfile.load(str(Path(__file__).resolve().parents[2] / "configs" / "capture_profile.mapping.json")).save(
        str(map_dir / "capture_profile.json"))


def build_synth_site(root: Path, n_sessions: int = 2, site_name: str = "synth") -> SynthSite:
    root = Path(root)
    site = SiteConfig.from_dict({"site": site_name, "data_dir": str(root / "data"),
                                 "watchdog": {"depth_check_period_s": 0.0, "depth_window_frames": 10, "recheck_frames": 15,
                                              "recheck_period_s": 0.0, "recheck_idle_s": 0.0, "startup_check": True}})
    w = default_world()
    T_map_room = awkward_map_frame()
    write_map_bundle(site.map_dir(), w, T_map_room)
    mb = load_map_bundle(str(site.map_dir()))
    caps = [render_capture(w, T_TRUE_ROOM_CAM, n_frames=30, seed=30 + i) for i in range(n_sessions)]
    res = calibrate(mb, caps, AnchorConfig(), cam_id="cam0")
    anchor = site.anchor_dir("cam0")
    write_anchor_bundle(res, str(anchor), "cam0", mb.get("capture_profile_sha"))
    site.map.map_id = mb["map_id"]
    site_path = root / "site.json"
    site.save(site_path)
    return SynthSite(root, site, site_path, w, T_map_room, bool(res.accepted), mb["map_id"])


def frames_from_capture_depth(depth_m: np.ndarray, rgb: Optional[np.ndarray] = None):
    """uint16 raw depth (mm) from a metres image with NaN = invalid."""
    d = np.where(np.isfinite(depth_m), depth_m, 0.0) / INTR["depth_scale"]
    return np.clip(np.round(d), 0, 65535).astype(np.uint16)
