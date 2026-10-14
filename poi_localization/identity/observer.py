"""Per-track observation: gate -> foreground -> joints -> structure + appearance (R1 spec sections 6-8).

Pure function of one frame and one detection (no state). Never raises on bad pixels: returns an Observation
carrying the gate result and whatever could be extracted. Imports only identity modules and numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from poi_localization.identity.background import foreground_pixels
from poi_localization.identity.clean_gate import GateInput, GateResult, check_clean, isolation
from poi_localization.identity.config import IdentityConfig
from poi_localization.identity.extract_appearance import AppearanceSample, extract_appearance
from poi_localization.identity.extract_structure import extract_structure, to_features
from poi_localization.identity.joints3d import lift_measured_joints
from poi_localization.identity.photometric import estimate_gain, normalise
from poi_localization.identity.room_geometry import RoomGeometry
from poi_localization.identity.visibility import band_visibility, face_visible, viewpoint

BBox = Tuple[float, float, float, float]


@dataclass
class Scene:
    geom: RoomGeometry
    ref_depth: np.ndarray                       # (H, W) metres, NaN = invalid
    ref_rgb: Optional[np.ndarray] = None
    ref_stable: Optional[np.ndarray] = None


@dataclass
class OtherTrack:
    bbox: BBox
    xy: Tuple[float, float]
    z_cam: float


@dataclass
class Observation:
    gate: GateResult
    features: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    appearance: Optional[AppearanceSample] = None
    view: Optional[str] = None
    flags: List[str] = field(default_factory=list)


def _upright(kp, max_tilt_deg: float) -> bool:
    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(n not in kp for n in need):
        return False
    sh = (np.array(kp["left_shoulder"][:2]) + np.array(kp["right_shoulder"][:2])) / 2
    hp = (np.array(kp["left_hip"][:2]) + np.array(kp["right_hip"][:2])) / 2
    d = hp - sh
    if d[1] <= 1.0:
        return False
    return bool(np.degrees(np.arctan2(abs(d[0]), d[1])) <= max_tilt_deg)


def depth_to_metres(frame) -> np.ndarray:
    return frame.depth.astype(np.float32) * float(frame.intr.depth_scale)


def observe(scene: Scene, frame, depth_m: np.ndarray, det, meas, xy: Tuple[float, float], vel: Tuple[float, float],
            track_state: str, resolving: bool, others: Sequence[OtherTrack], cfg: IdentityConfig,
            s_since_2d_mismatch: Optional[float] = None, swap_flag: bool = False, calib_ok: bool = True) -> Observation:
    geom = scene.geom
    z_t = float(geom.project(np.array([[xy[0], xy[1], 1.2]]))[2][0])
    kp = det.keypoints
    iou, dist = isolation(det.bbox, xy, [(o.bbox, o.xy) for o in others])
    gi = GateInput(track_state=track_state, measured_this_frame=True, identity_resolving=resolving,
                   s_since_2d_mismatch=s_since_2d_mismatch, swap_flag=swap_flag, max_iou_with_others=iou,
                   min_dist_to_others_m=dist, torso_depth_m=z_t, det_conf=float(det.det_conf),
                   n_kp_conf=sum(1 for v in kp.values() if v[2] >= cfg.kp_conf_min), bbox=det.bbox,
                   img_w=geom.width, img_h=geom.height, upright=_upright(kp, cfg.upright_max_tilt_deg), calib_ok=calib_ok)
    gate = check_clean(gi, cfg)
    obs = Observation(gate=gate)
    if not gate.ok_for_evidence:
        return obs
    bg_others = [(o.bbox, o.z_cam) for o in others]
    fg = foreground_pixels(depth_m, scene.ref_depth, det.bbox, z_t, bg_others, cfg.background, stride=cfg.pixel_stride)
    if fg.n < 100:
        obs.flags.append("foreground_too_small")
        return obs
    joints = lift_measured_joints(geom, depth_m, kp, z_t, kp_conf_min=cfg.kp_conf_min,
                                  kp_pixel_sigma_px=cfg.kp_pixel_sigma_px, torso_xy=xy)
    ls, rs = joints.get("left_shoulder"), joints.get("right_shoulder")
    obs.view = viewpoint(None if ls is None else ls.p, None if rs is None else rs.p, geom.t[:2], np.asarray(xy, float),
                         face_visible(kp, cfg.kp_conf_min), np.asarray(vel, float))
    sample = extract_structure(geom, depth_m, joints, fg, kp_height_estimate=meas.height_sample_m, view=obs.view)
    obs.flags.extend(sample.flags)
    obs.features = to_features(sample)
    stature = sample.stature[0] if sample.stature else meas.height_sample_m
    if stature is None:
        return obs
    cols = frame.rgb[fg.vs, fg.us]
    quality = 0.0
    if scene.ref_rgb is not None:
        stable = scene.ref_stable if scene.ref_stable is not None else np.ones(scene.ref_depth.shape, bool)
        g = estimate_gain(frame.rgb, scene.ref_rgb, depth_m, scene.ref_depth, stable, det.bbox, cfg.photometric)
        cols, quality = normalise(cols, g.gain), g.quality
    else:
        cols = cols.astype(np.float32)
    states = band_visibility(geom, scene.ref_depth, xy[0], xy[1], float(stature), cfg.bands,
                             [(o.bbox, o.z_cam) for o in others], max_range_m=cfg.max_range_m)
    obs.appearance = extract_appearance(cols, geom.heights(depth_m, fg.vs, fg.us), float(stature), cfg.bands, states,
                                        cfg.hist, view=obs.view, photo_quality=quality)
    return obs
