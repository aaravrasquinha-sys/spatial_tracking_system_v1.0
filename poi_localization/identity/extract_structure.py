"""S1/S2 -- metric anthropometrics with propagated uncertainty (R1 spec section 7.2).

Stature: 98th percentile of foreground pixel heights near the head keypoint and above the shoulders.
Segments: length of a segment between two MEASURED joints, with variance u^T (S1 + S2) u, accepted only
when the segment makes >= 45 deg with the viewing ray (foreshortened segments are depth-noise dominated)
and sigma_L <= SEGMENT_SIGMA_MAX_M (5 cm; the spec's 3 cm drops legs beyond ~2.4 m). Left/right are averaged when they agree within 15%, else the lower-sigma side wins.
Features are decorrelated from stature as residuals r_seg = L_seg - beta_seg * H.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from poi_localization.identity.background import ForegroundResult
from poi_localization.identity.joints3d import Joint
from poi_localization.identity.room_geometry import RoomGeometry

# Segment-length / stature proportions (Drillis & Contini via Winter). The keypoint-defined segments are not
# exactly the anatomical landmarks these were defined on: Phase 0 refits them (and the spreads) on real recordings.
DEFAULT_BETA: Dict[str, float] = {
    "upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shank": 0.246,
    "torso": 0.288, "hip_width": 0.191, "shoulder_width": 0.259,
}
SOFT_FEATURES = ("shoulder_width",)          # a jacket changes these: their within-person variance is inflated
FEATURE_ORDER = ("stature", "upper_arm", "forearm", "thigh", "shank", "torso", "hip_width", "shoulder_width")

_PAIRS = {
    "upper_arm": (("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")),
    "forearm": (("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")),
    "thigh": (("left_hip", "left_knee"), ("right_hip", "right_knee")),
    "shank": (("left_knee", "left_ankle"), ("right_knee", "right_ankle")),
}
_COS_MAX = float(np.cos(np.pi / 4))          # >= 45 deg to the viewing ray
SEGMENT_SIGMA_MAX_M = 0.05     # spec says 3 cm; with the default (conservative, per-pixel) depth noise model that rejects
                               # every leg segment beyond ~2.4 m, so the default is 5 cm until Phase 0 fits the noise model.
                               # Noisy samples are down-weighted by their sigma in the LLR, so admitting them is harmless.
_HEAD_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


@dataclass
class StructureSample:
    stature: Optional[Tuple[float, float]] = None                 # (value m, sigma m)
    segments: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


def _segment_length(j1: Joint, j2: Joint, sigma_max: float = SEGMENT_SIGMA_MAX_M) -> Optional[Tuple[float, float]]:
    d = j2.p - j1.p
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    u = d / L
    ray = j1.ray + j2.ray
    ray = ray / np.linalg.norm(ray)
    c = float(u @ ray)
    if abs(c) > _COS_MAX:
        return None
    var = (j1.sigma_lat ** 2 + j2.sigma_lat ** 2) * (1 - c * c) + (j1.sigma_z ** 2 + j2.sigma_z ** 2) * c * c
    sigma = float(np.sqrt(var))
    return (L, sigma) if sigma <= sigma_max else None


def _mid(a: Joint, b: Joint) -> Joint:
    ray = a.ray + b.ray
    return Joint(p=(a.p + b.p) / 2, sigma_z=float(np.hypot(a.sigma_z, b.sigma_z) / 2),
                 sigma_lat=float(np.hypot(a.sigma_lat, b.sigma_lat) / 2), ray=ray / np.linalg.norm(ray))


def _combine_lr(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return a or b
    mean = (a[0] + b[0]) / 2
    if abs(a[0] - b[0]) / mean <= 0.15:
        wa, wb = 1 / a[1] ** 2, 1 / b[1] ** 2
        return ((wa * a[0] + wb * b[0]) / (wa + wb), float(1 / np.sqrt(wa + wb)))
    return a if a[1] <= b[1] else b


def _near_side(geom: RoomGeometry, joints: Dict[str, Joint]) -> Optional[str]:
    """Which body side faces the camera (smaller camera distance at the shoulder, else the hip)."""
    for part in ("shoulder", "hip"):
        l, r = joints.get(f"left_{part}"), joints.get(f"right_{part}")
        if l is not None and r is not None:
            return "left" if np.linalg.norm(l.p - geom.t) <= np.linalg.norm(r.p - geom.t) else "right"
    return None


def extract_structure(geom: RoomGeometry, depth_m: np.ndarray, joints: Dict[str, Joint], fg: ForegroundResult,
                      kp_height_estimate: Optional[float] = None, view: Optional[str] = "front",
                      sigma_max_m: float = SEGMENT_SIGMA_MAX_M) -> StructureSample:
    """`view` ('front' | 'back' | 'side' | None) gates which features are trustworthy. In a side or unknown view the
    FAR-side joints read the nearer body's depth (their keypoints are guessed behind it), which corrupts anything
    bilateral, and the near arm hangs in front of the hip and contaminates its depth. So in those views only stature
    and the NEAR-side upper arm, forearm and shank are used; width features, mid-point torso and thigh are skipped."""
    s = StructureSample()
    bilateral = view in ("front", "back")
    near = None if bilateral else _near_side(geom, joints)
    if not bilateral:
        s.flags.append(f"view_{view or 'unknown'}_near_side_only")

    # ---- stature (S1): the head is visible from every view
    head = [joints[n] for n in _HEAD_NAMES if n in joints]
    if head and fg.n >= 20:
        hp = np.mean([j.p for j in head], axis=0)
        sh = [joints[n].p[2] for n in ("left_shoulder", "right_shoulder") if n in joints]
        shoulder_z = float(np.mean(sh)) if sh else float(hp[2] - 0.30)
        P = geom.room_points(depth_m, fg.vs, fg.us)
        close = (np.hypot(P[:, 0] - hp[0], P[:, 1] - hp[1]) <= 0.15) & (P[:, 2] > shoulder_z)
        if int(close.sum()) >= 20:
            H = float(np.percentile(P[close, 2], 98))
            if kp_height_estimate is not None and abs(H - kp_height_estimate) > 0.08:
                s.flags.append("stature_disagrees_with_keypoint_estimate")
            else:
                s.stature = (H, float(np.hypot(min(j.sigma_z for j in head), 0.01)))
        else:
            s.flags.append("too_few_head_pixels")
    elif not head:
        s.flags.append("no_measured_head_joint")

    # ---- limb segments (S2)
    def seg_len(a, b):
        return _segment_length(joints[a], joints[b], sigma_max_m) if a in joints and b in joints else None

    for seg, (left, right) in _PAIRS.items():
        if not bilateral and seg == "thigh":
            continue
        if bilateral:
            v = _combine_lr(seg_len(*left), seg_len(*right))
        else:
            v = seg_len(*(left if near == "left" else right)) if near else None
        if v is not None:
            s.segments[seg] = v

    # ---- torso (S2)
    if bilateral:
        if all(n in joints for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
            v = _segment_length(_mid(joints["left_shoulder"], joints["right_shoulder"]),
                                _mid(joints["left_hip"], joints["right_hip"]), sigma_max_m)
            if v is not None:
                s.segments["torso"] = v

    # ---- widths (S2): bilateral only
    if bilateral:
        v = seg_len("left_hip", "right_hip")
        if v is not None:
            s.segments["hip_width"] = v
        v = seg_len("left_shoulder", "right_shoulder")
        if v is not None:
            s.segments["shoulder_width"] = v
    return s


def to_features(sample: StructureSample, H: Optional[float] = None, sigma_H: float = 0.0,
                beta: Optional[Dict[str, float]] = None) -> Dict[str, Tuple[float, float]]:
    """Feature vector {name: (value, sigma)}: stature plus residuals r = L - beta*H.
    `H` defaults to the sample's own stature; pass the person's running stature to decorrelate against it."""
    beta = beta or DEFAULT_BETA
    out: Dict[str, Tuple[float, float]] = {}
    if sample.stature is not None:
        out["stature"] = sample.stature
    if H is None and sample.stature is not None:
        H, sigma_H = sample.stature
    if H is None:
        return out
    for seg, (L, sig) in sample.segments.items():
        b = beta[seg]
        out[seg] = (L - b * H, float(np.hypot(sig, b * sigma_H)))
    return out
