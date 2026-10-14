"""Mannequin renderer for the identity (R1) tests.

A person = a sphere (head) + capsules (torso, shoulder/hip bars, limbs, feet) with KNOWN stature and segment
lengths and per-part colours, posed on the floor of the room frame, rendered into aligned colour + depth on
top of an analytic empty room (floor plane + back wall), with D435i-like depth noise and keypoint jitter.
Rendering is z-buffered point splatting: surfaces are sampled finer than a pixel and projected with an
independent pinhole implementation (deliberately NOT RoomGeometry, so a bug in RoomGeometry cannot cancel itself out).

Nothing here is imported by production code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COCO = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle")


@dataclass
class Body:
    stature: float = 1.75
    # segment lengths as fractions of stature (Winter) with per-body multiplicative deviations
    dev: Dict[str, float] = field(default_factory=dict)
    shirt: Tuple[int, int, int] = (200, 40, 40)
    trousers: Tuple[int, int, int] = (40, 50, 140)
    skin: Tuple[int, int, int] = (225, 170, 140)
    shoes: Tuple[int, int, int] = (30, 30, 30)

    def seg(self, name: str) -> float:
        base = {"upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shank": 0.246, "torso": 0.288,
                "hip_width": 0.191, "shoulder_width": 0.259}[name]
        return base * self.stature * self.dev.get(name, 1.0)


def pose(body: Body, x: float, y: float, heading: float) -> Tuple[Dict[str, np.ndarray], List[tuple]]:
    """Standing pose at floor (x, y) facing `heading` (radians, floor-plane). Returns (joints, parts).
    parts: ("sphere", centre, None, r, colour) | ("capsule", a, b, r, colour)."""
    H = body.stature
    f = np.array([np.cos(heading), np.sin(heading), 0.0])           # forward
    left = np.array([-np.sin(heading), np.cos(heading), 0.0])       # person's left (facing +x -> +y)
    o = np.array([x, y, 0.0])
    z = lambda h: np.array([0.0, 0.0, h])
    ankle_h = 0.039 * H
    knee = o + z(ankle_h + body.seg("shank"))
    hip_c = knee + z(body.seg("thigh"))
    sh_c = hip_c + z(body.seg("torso"))
    hw, sw = body.seg("hip_width") / 2, body.seg("shoulder_width") / 2
    J: Dict[str, np.ndarray] = {}
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        J[f"{side}_hip"] = hip_c + sgn * hw * left
        J[f"{side}_knee"] = J[f"{side}_hip"] - z(body.seg("thigh"))
        J[f"{side}_ankle"] = J[f"{side}_knee"] - z(body.seg("shank"))
        J[f"{side}_shoulder"] = sh_c + sgn * sw * left
        # arms hang beside the body, slightly out so they are not hidden by the torso
        arm_out = sgn * 0.06 * left
        J[f"{side}_elbow"] = J[f"{side}_shoulder"] + arm_out - z(body.seg("upper_arm"))
        J[f"{side}_wrist"] = J[f"{side}_elbow"] - z(body.seg("forearm"))
    head_r = 0.095
    head_c = o + z(H - head_r)
    J["nose"] = head_c + f * head_r + z(-0.01)
    J["left_eye"] = head_c + f * head_r * 0.9 + left * 0.035 + z(0.02)
    J["right_eye"] = head_c + f * head_r * 0.9 - left * 0.035 + z(0.02)
    J["left_ear"] = head_c + left * head_r - f * 0.01
    J["right_ear"] = head_c - left * head_r - f * 0.01
    parts = [("sphere", head_c, None, head_r, body.skin),
             ("capsule", sh_c, hip_c, 0.10, body.shirt),
             ("capsule", J["left_shoulder"], J["right_shoulder"], 0.05, body.shirt),
             ("capsule", J["left_hip"], J["right_hip"], 0.07, body.trousers)]
    for side in ("left", "right"):
        parts += [("capsule", J[f"{side}_shoulder"], J[f"{side}_elbow"], 0.04, body.shirt),
                  ("capsule", J[f"{side}_elbow"], J[f"{side}_wrist"], 0.035, body.skin),
                  ("capsule", J[f"{side}_hip"], J[f"{side}_knee"], 0.065, body.trousers),
                  ("capsule", J[f"{side}_knee"], J[f"{side}_ankle"], 0.045, body.trousers),
                  ("capsule", J[f"{side}_ankle"] - z(0.0), J[f"{side}_ankle"] + f * 0.12 - z(0.0), 0.04, body.shoes)]
    return J, parts


def _sphere_pts(c, r, s):
    n = max(int(np.ceil(4 * np.pi * r * r / (s * s))), 8)
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return c + r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def _capsule_pts(a, b, r, s):
    ax = b - a
    L = np.linalg.norm(ax)
    if L < 1e-9:
        return _sphere_pts(a, r, s)
    ax = ax / L
    e1 = np.cross(ax, [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    na, nr = max(int(np.ceil(L / s)), 1) + 1, max(int(np.ceil(2 * np.pi * r / s)), 8)
    tt, ph = np.meshgrid(np.linspace(0, L, na), np.linspace(0, 2 * np.pi, nr, endpoint=False), indexing="ij")
    cyl = a + tt[..., None] * ax + r * (np.cos(ph)[..., None] * e1 + np.sin(ph)[..., None] * e2)
    caps = []
    for c, sgn in ((a, -1.0), (b, 1.0)):
        sp = _sphere_pts(c, r, s)
        caps.append(sp[(sp - c) @ ax * sgn >= 0])
    return np.concatenate([cyl.reshape(-1, 3)] + caps)


class Room:
    """Analytic empty room: floor z=0 and a back wall at x = wall_x, seen by a pinhole camera (R, t, intr)."""

    def __init__(self, R, t, intr, wall_x: float = 6.0, seed: int = 0):
        self.R, self.t, self.intr, self.wall_x = np.asarray(R, float), np.asarray(t, float), intr, wall_x
        h, w = intr.height, intr.width
        vs, us = np.mgrid[0:h, 0:w]
        d = np.stack([(us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy, np.ones((h, w))], axis=-1) @ self.R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_f = np.where(d[..., 2] < -1e-9, -self.t[2] / d[..., 2], np.inf)
            lam_w = np.where(d[..., 0] > 1e-9, (wall_x - self.t[0]) / d[..., 0], np.inf)
        self.depth = np.minimum(lam_f, lam_w)
        self.depth[~np.isfinite(self.depth)] = np.nan
        rng = np.random.default_rng(seed)
        base = np.where((lam_f < lam_w)[..., None], np.array([120, 110, 90.0]), np.array([90, 120, 140.0]))
        tex = rng.normal(0, 12, (h, w, 1))
        self.rgb = np.clip(base + tex, 25, 230).astype(np.uint8)


def render(room: Room, people: Sequence[Tuple[Body, float, float, float]], rng: np.random.Generator,
           depth_noise: bool = True, spacing: float = 0.003, gain: Optional[Sequence[float]] = None):
    """Render people into the room. Returns (depth_m float32 with 0 = invalid, rgb uint8, joints_list).
    `gain` applies a per-channel colour gain to the whole live image (exposure / white-balance change)."""
    intr = room.intr
    h, w = intr.height, intr.width
    depth = np.where(np.isfinite(room.depth), room.depth, 0.0).astype(np.float64)
    rgb = room.rgb.astype(np.float64).copy()
    zbuf = np.where(depth > 0, depth, np.inf)
    joints_list = []
    for body, x, y, heading in people:
        J, parts = pose(body, x, y, heading)
        joints_list.append(J)
        pts, cols = [], []
        for kind, a, b, r, col in parts:
            p = _sphere_pts(a, r, spacing) if kind == "sphere" else _capsule_pts(a, b, r, spacing)
            pts.append(p)
            cols.append(np.tile(np.asarray(col, float), (len(p), 1)))
        P, C = np.concatenate(pts), np.concatenate(cols)
        pc = (P - room.t) @ room.R                                   # camera coordinates (independent pinhole)
        z = pc[:, 2]
        ok = z > 0.3
        u = np.round(intr.fx * pc[ok, 0] / z[ok] + intr.cx).astype(int)
        v = np.round(intr.fy * pc[ok, 1] / z[ok] + intr.cy).astype(int)
        zz, cc = z[ok], C[ok]
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, zz, cc = u[inb], v[inb], zz[inb], cc[inb]
        pix = v * w + u
        order = np.lexsort((zz, pix))
        pix_s = pix[order]
        first = np.concatenate([[True], pix_s[1:] != pix_s[:-1]])
        sel = order[first]
        pu, pv, pz, pcol = u[sel], v[sel], zz[sel], cc[sel]
        front = pz < zbuf[pv, pu]
        pu, pv, pz, pcol = pu[front], pv[front], pz[front], pcol[front]
        if depth_noise:
            from poi_localization.identity.room_geometry import depth_sigma_m
            sig = np.array([depth_sigma_m(float(q)) for q in pz])
            pz = pz + rng.normal(0, 1, pz.size) * sig
        depth[pv, pu] = pz
        zbuf[pv, pu] = pz
        rgb[pv, pu] = pcol
    if gain is not None:
        rgb = rgb * np.asarray(gain, float)
    return depth.astype(np.float32), np.clip(rgb, 0, 255).astype(np.uint8), joints_list


def keypoints(room: Room, J: Dict[str, np.ndarray], rng: np.random.Generator, sigma_px: float = 1.0,
              conf: float = 0.9) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    for n in COCO:
        pc = (J[n] - room.t) @ room.R
        u = room.intr.fx * pc[0] / pc[2] + room.intr.cx + rng.normal(0, sigma_px)
        v = room.intr.fy * pc[1] / pc[2] + room.intr.cy + rng.normal(0, sigma_px)
        out[n] = (float(u), float(v), conf)
    return out


def bbox_of(room: Room, J: Dict[str, np.ndarray], stature: float, pad: float = 0.12):
    """Pixel bbox from the projected extreme joints (+ the head top and feet), padded."""
    pts = np.array(list(J.values()) + [J["nose"] + [0, 0, 0.12], J["left_ankle"] * [1, 1, 0], J["right_ankle"] * [1, 1, 0]])
    pc = (pts - room.t) @ room.R
    u = room.intr.fx * pc[:, 0] / pc[:, 2] + room.intr.cx
    v = room.intr.fy * pc[:, 1] / pc[:, 2] + room.intr.cy
    w, h = u.max() - u.min(), v.max() - v.min()
    return (float(u.min() - pad * w), float(v.min() - pad * h), float(u.max() + pad * w), float(v.max() + pad * h))
