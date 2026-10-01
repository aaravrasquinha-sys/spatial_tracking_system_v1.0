"""
Synthetic room for the WP-ANCHOR oracle tests.

An analytic room (interior box) with axis-aligned furniture boxes, in the
ROOM frame (floor z=0, Z up). From it we can produce, with a KNOWN ground truth:
  * a "module 1" map: coloured surface points (2 cm spacing + noise) expressed in a
    deliberately awkward W_cam0-like MAP frame (y down, tilted, arbitrary origin),
    exactly the situation the real live map is in;
  * a static-camera capture: N noisy depth frames ray-cast from a known T_room_cam,
    pushed through the SAME build_capture() the hardware path uses.
Nothing here is shared with the code under test except build_capture (deliberately:
the capture->cloud path is part of what's being verified).
"""
from __future__ import annotations
import numpy as np

from pyslam.core import lie
from pyslam.anchor.capture import build_capture

INTR = {"fx": 606.75, "fy": 606.57, "cx": 320.19, "cy": 237.06, "width": 640, "height": 480,
        "depth_scale": 0.001, "baseline": 0.05}


class Box:
    def __init__(self, lo, hi, color=(160, 160, 160)):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.color = np.asarray(color, float)

    def moved(self, d):
        return Box(self.lo + d, self.hi + d, self.color)


class World:
    def __init__(self, room=(5.2, 4.1, 2.6), boxes=None, wall_colors=None, textured=True):
        self.room = np.asarray(room, float)
        self.boxes = list(boxes or [])
        self.wall_colors = wall_colors
        self.textured = textured

    # ---------------- ray casting ----------------
    def raycast(self, o: np.ndarray, d: np.ndarray) -> np.ndarray:
        """o (3,), d (N,3) un-normalised (t is then depth if d_z_cam == 1). -> t (N,) nearest hit or inf."""
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            # interior of the room: exit distance
            t1 = (0 - o) * inv
            t2 = (self.room - o) * inv
            t_exit = np.min(np.maximum(t1, t2), axis=1)
            t = np.where(t_exit > 0, t_exit, np.inf)
            for b in self.boxes:
                a1 = (b.lo - o) * inv
                a2 = (b.hi - o) * inv
                tn = np.max(np.minimum(a1, a2), axis=1)
                tf = np.min(np.maximum(a1, a2), axis=1)
                hit = (tn <= tf) & (tn > 1e-6)
                t = np.where(hit & (tn < t), tn, t)
        return t

    def color_at(self, P: np.ndarray) -> np.ndarray:
        """Ground-truth surface colour at points lying ON the geometry (same function sample_surface uses)."""
        R = self.room
        wc = self.wall_colors or [(200, 190, 170), (170, 190, 200), (190, 170, 190), (180, 200, 170),
                                  (120, 110, 100), (230, 230, 230)]
        base = np.zeros((P.shape[0], 3))
        assigned = np.zeros(P.shape[0], bool)
        tol = 2e-3
        for b in self.boxes:
            for ax in range(3):
                for val in (b.lo[ax], b.hi[ax]):
                    o = [i for i in range(3) if i != ax]
                    m = (np.abs(P[:, ax] - val) < tol) & np.all((P[:, o] >= b.lo[o] - tol) & (P[:, o] <= b.hi[o] + tol), axis=1) & ~assigned
                    base[m] = b.color
                    assigned |= m
        faces = [(0, 0.0), (0, R[0]), (1, 0.0), (1, R[1]), (2, 0.0), (2, R[2])]
        for k, (ax, val) in enumerate(faces):
            m = (np.abs(P[:, ax] - val) < tol) & ~assigned
            base[m] = wc[k]
            assigned |= m
        return self._color(P, base) if self.textured else base

    # ---------------- map surface sampling ----------------
    def _color(self, p, base):
        if not self.textured:
            return np.tile(base, (p.shape[0], 1))
        tex = 40 * np.sin(6.0 * p[:, 0]) * np.cos(5.0 * p[:, 1]) + 30 * np.sin(4.0 * p[:, 2] + 2 * p[:, 0])
        return np.clip(base[None] + tex[:, None], 0, 255)

    def sample_surface(self, spacing=0.02, noise=0.003, seed=0, drop_region=None):
        rng = np.random.default_rng(seed)
        pts, cols = [], []

        def face(axis, val, lo, hi, base):
            a1, a2 = [i for i in range(3) if i != axis]
            g1 = np.arange(lo[a1], hi[a1] + 1e-9, spacing)
            g2 = np.arange(lo[a2], hi[a2] + 1e-9, spacing)
            G1, G2 = np.meshgrid(g1, g2, indexing="ij")
            p = np.zeros((G1.size, 3))
            p[:, axis] = val
            p[:, a1] = G1.ravel()
            p[:, a2] = G2.ravel()
            pts.append(p)
            cols.append(self._color(p, np.asarray(base, float)))

        R = self.room
        wc = self.wall_colors or [(200, 190, 170), (170, 190, 200), (190, 170, 190), (180, 200, 170),
                                  (120, 110, 100), (230, 230, 230)]
        # x=0, x=R, y=0, y=R, floor, ceiling  (interior faces)
        face(0, 0.0, (0, 0, 0), R, wc[0]); face(0, R[0], (0, 0, 0), R, wc[1])
        face(1, 0.0, (0, 0, 0), R, wc[2]); face(1, R[1], (0, 0, 0), R, wc[3])
        face(2, 0.0, (0, 0, 0), R, wc[4]); face(2, R[2], (0, 0, 0), R, wc[5])
        for b in self.boxes:
            for ax in range(3):
                face(ax, b.lo[ax], b.lo, b.hi, b.color)
                if not (ax == 2):        # nothing under the box is ever seen; keep the top and the sides
                    face(ax, b.hi[ax], b.lo, b.hi, b.color)
                else:
                    face(2, b.hi[2], b.lo, b.hi, b.color)
        P = np.concatenate(pts)
        C = np.concatenate(cols)
        # drop points that are inside any box (walls/floor hidden by furniture)
        keep = np.ones(P.shape[0], bool)
        for b in self.boxes:
            inside = np.all((P > b.lo + 1e-6) & (P < b.hi - 1e-6), axis=1)
            keep &= ~inside
            # floor under a box: hidden
            under = (np.abs(P[:, 2]) < 1e-9) & np.all((P[:, :2] > b.lo[:2]) & (P[:, :2] < b.hi[:2]), axis=1)
            keep &= ~under
        P, C = P[keep], C[keep]
        if drop_region is not None:                       # simulate dropped keyframes -> holes in the map
            lo, hi = drop_region
            m = np.all((P > lo) & (P < hi), axis=1)
            P, C = P[~m], C[~m]
        P = P + rng.normal(0, noise, P.shape)
        return P, np.clip(C, 0, 255).astype(np.uint8)


def cam_pose_room(x, y, h, yaw_deg, pitch_deg, roll_deg=0.0) -> np.ndarray:
    """T_room_cam. pitch>0 tilts the camera DOWN. Camera axes: x right, y down, z forward."""
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=1)
    if roll_deg:
        R = R @ lie.so3_exp(np.array([0, 0, roll]))
    return lie.make_T(R, np.array([x, y, h]))


def awkward_map_frame(seed=1, yaw_deg=33.0, tilt_deg=7.0) -> np.ndarray:
    """T_map_room for a W_cam0-like frame: room z -> map -y (approximately), tilted and yawed,
    with an arbitrary origin -- so the room-frame derivation has real work to do."""
    R0 = np.stack([np.array([0, 0, 1.0]),      # room x -> map z (forward)
                   np.array([-1.0, 0, 0]),     # room y -> map -x
                   np.array([0, -1.0, 0])], axis=1)  # room z -> map -y (up)
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    Rt = lie.so3_exp(axis * np.radians(tilt_deg))
    Ry = lie.so3_exp(np.array([0, np.radians(yaw_deg), 0]))
    R = Ry @ Rt @ R0
    t = np.array([0.4, -1.5, 0.7])
    return lie.make_T(R, t)


def default_world(**kw) -> World:
    """Asymmetric on purpose (a symmetric room is a SEPARATE test): sofa, table, shelf, box."""
    boxes = [Box((0.3, 0.3, 0), (2.1, 1.1, 0.8), (90, 60, 60)),          # sofa
             Box((3.0, 2.4, 0), (3.9, 3.2, 0.75), (140, 110, 70)),       # table
             Box((4.9, 0.6, 0), (5.2, 1.9, 2.0), (70, 70, 110)),         # tall shelf on +x wall
             Box((2.6, 0.2, 0), (3.1, 0.7, 0.5), (200, 60, 60))]         # small box
    return World(boxes=boxes, **kw)


def render_capture(world: World, T_room_cam: np.ndarray, n_frames=40, seed=0, imu=True,
                   noise_k=0.0015, dropout=0.03, up_noise_deg=0.05, intr=None,
                   gravity_bias_deg=0.0, color=False):
    """Ray-cast + noise -> StaticCapture via the production build_capture()."""
    intr = intr or INTR
    rng = np.random.default_rng(seed)
    H, W = intr["height"], intr["width"]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"], np.ones_like(u, float)], -1).reshape(-1, 3)
    R, o = T_room_cam[:3, :3], T_room_cam[:3, 3]
    dirs_room = dirs_cam @ R.T
    t = world.raycast(o, dirs_room)                      # depth z (dir z_cam == 1)
    z = t.reshape(H, W)
    frames = np.zeros((n_frames, H, W), np.uint16)
    for i in range(n_frames):
        zz = z + rng.normal(0, 1, z.shape) * noise_k * z ** 2
        zz[rng.random(z.shape) < dropout] = 0
        zz[~np.isfinite(zz)] = 0
        frames[i] = np.clip(np.round(zz / intr["depth_scale"]), 0, 65535).astype(np.uint16)
    if color:
        hit = np.isfinite(t)
        P = np.zeros((t.size, 3))
        P[hit] = o[None] + dirs_room[hit] * t[hit, None]
        col = np.zeros((t.size, 3))
        col[hit] = world.color_at(P[hit])
        rgb = np.clip(col, 0, 255).astype(np.uint8).reshape(H, W, 3)
    else:
        rgb = np.tile(np.linspace(40, 200, W, dtype=np.uint8)[None, :, None], (H, 1, 3))
    rgbs = np.stack([rgb] * 3)
    up_cam = R.T @ np.array([0, 0, 1.0])
    imus = None
    Rbc = np.eye(3)
    if imu:
        if gravity_bias_deg:
            ax = np.cross(up_cam, [1.0, 0.3, 0.2]); ax /= np.linalg.norm(ax)
            up_cam = lie.so3_exp(ax * np.radians(gravity_bias_deg)) @ up_cam
        n = 400
        acc = 9.81 * up_cam[None] + rng.normal(0, 0.02, (n, 3))
        # small direction jitter of the MEAN (a real accel is never exactly repeatable)
        jit = rng.normal(0, np.radians(up_noise_deg), 3)
        acc = acc + 9.81 * np.cross(jit, up_cam)[None]
        gyro = rng.normal(0, 0.002, (n, 3))
        imus = [np.concatenate([np.arange(n)[:, None] * 0.0025, gyro, acc], axis=1)]
    return build_capture(frames, rgbs, intr, imus, Rbc if imu else None, source="synthetic")
