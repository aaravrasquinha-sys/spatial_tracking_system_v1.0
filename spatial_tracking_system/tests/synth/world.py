"""
Synthetic world generator for gate testing.

Produces analytic scenes (textured planar walls) so that ground-truth
projection and back-projection can be checked to near machine precision
when noise is disabled. This is deliberately simple geometry -- the goal
is a trustworthy oracle, not a photorealistic renderer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from pyslam.core.types import Intrinsics, Frame
from pyslam.core import lie


# ---------------------------------------------------------------- geometry

@dataclass
class Wall:
    """An axis-aligned rectangular wall patch with a precomputed random
    (non-periodic) texture, so that no two walls -- and no two patches
    within a wall -- look alike by construction. An earlier version used a
    deterministic sin/checker formula, which is periodic: local ORB
    patches from genuinely different physical locations could match each
    other by coincidence (verified on the corridor-loop fixture -- it
    caused both weak BoW discriminability and an outright false
    mid-corridor loop-closure hypothesis). Per-wall seeded noise has no
    such structure: two walls are statistically independent draws."""
    origin: np.ndarray      # (3,) one corner, world frame
    u_axis: np.ndarray      # (3,) unit vector, in-plane
    v_axis: np.ndarray      # (3,) unit vector, in-plane, orthogonal to u_axis
    normal: np.ndarray      # (3,) unit vector, outward
    u_len: float
    v_len: float
    texture_seed: int = 0
    texels_per_m: float = 24.0    # fine-detail resolution (corner richness for ORB)
    coarse_cells_per_wall: int = 10  # regional-distinctiveness resolution (for BoW)

    def __post_init__(self):
        rng = np.random.default_rng(self.texture_seed)
        fine_nu = max(4, int(self.u_len * self.texels_per_m))
        fine_nv = max(4, int(self.v_len * self.texels_per_m))
        coarse_nu = max(2, self.coarse_cells_per_wall)
        coarse_nv = max(2, self.coarse_cells_per_wall)

        # three independent RGB channels, each a mix of a coarse random
        # field (regional distinctiveness) and a fine random field (dense
        # corners for ORB); nearest-neighbour sampled, no interpolation
        # needed -- block noise gives sharp, corner-rich boundaries.
        self._fine = rng.integers(0, 256, size=(fine_nv, fine_nu, 3), dtype=np.uint16)
        self._coarse = rng.integers(0, 256, size=(coarse_nv, coarse_nu, 3), dtype=np.uint16)

    def sample_color(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """u, v pre-normalised to [0,1] (i.e. already divided by u_len/v_len
        by the caller)."""
        fv, fu = self._fine.shape[0], self._fine.shape[1]
        cv_, cu = self._coarse.shape[0], self._coarse.shape[1]
        fi = np.clip((v * fv).astype(np.int64), 0, fv - 1)
        fj = np.clip((u * fu).astype(np.int64), 0, fu - 1)
        ci = np.clip((v * cv_).astype(np.int64), 0, cv_ - 1)
        cj = np.clip((u * cu).astype(np.int64), 0, cu - 1)
        fine_col = self._fine[fi, fj].astype(np.float64)
        coarse_col = self._coarse[ci, cj].astype(np.float64)
        mixed = 0.45 * fine_col + 0.55 * coarse_col
        return np.clip(mixed, 0, 255).astype(np.uint8)


@dataclass
class World:
    walls: List[Wall] = field(default_factory=list)

    def add_box_room(self, center: np.ndarray, size: np.ndarray, seed0: int = 0):
        """Interior-facing walls of an axis-aligned box (a simple room)."""
        cx, cy, cz = center
        sx, sy, sz = size / 2.0
        specs = [
            # origin, u_axis, v_axis, normal, u_len, v_len
            (np.array([cx - sx, cy - sy, cz - sz]), np.array([0, 1, 0]), np.array([0, 0, 1]),
             np.array([1, 0, 0]), 2 * sy, 2 * sz),   # -X wall, normal +x
            (np.array([cx + sx, cy - sy, cz - sz]), np.array([0, 1, 0]), np.array([0, 0, 1]),
             np.array([-1, 0, 0]), 2 * sy, 2 * sz),  # +X wall, normal -x
            (np.array([cx - sx, cy - sy, cz - sz]), np.array([1, 0, 0]), np.array([0, 0, 1]),
             np.array([0, 1, 0]), 2 * sx, 2 * sz),   # -Y wall
            (np.array([cx - sx, cy + sy, cz - sz]), np.array([1, 0, 0]), np.array([0, 0, 1]),
             np.array([0, -1, 0]), 2 * sx, 2 * sz),  # +Y wall
            (np.array([cx - sx, cy - sy, cz - sz]), np.array([1, 0, 0]), np.array([0, 1, 0]),
             np.array([0, 0, 1]), 2 * sx, 2 * sy),   # floor
            (np.array([cx - sx, cy - sy, cz + sz]), np.array([1, 0, 0]), np.array([0, 1, 0]),
             np.array([0, 0, -1]), 2 * sx, 2 * sy),  # ceiling
        ]
        for i, (o, u, v, n, ul, vl) in enumerate(specs):
            self.walls.append(Wall(o.astype(float), u.astype(float), v.astype(float),
                                    n.astype(float), ul, vl, texture_seed=seed0 + i))

    def add_corridor_loop(self, seed0: int = 100):
        """A rectangular corridor (hollow box ring) guaranteeing a revisit."""
        # Four straight segments forming a loop, each modelled as a pair of
        # facing walls + floor/ceiling. Kept simple: reuse box rooms chained.
        segs = [
            (np.array([0, 0, 0]), np.array([6, 2, 2.4])),
            (np.array([6, 3, 0]), np.array([2, 4, 2.4])),
            (np.array([0, 6, 0]), np.array([6, 2, 2.4])),
            (np.array([-3, 3, 0]), np.array([2, 4, 2.4])),
        ]
        for i, (c, s) in enumerate(segs):
            self.add_box_room(c, s, seed0=seed0 + i * 10)

    def add_two_rooms(self, seed_a: int = 200, seed_b: int = 200):
        """Two rooms with IDENTICAL texture seeds -> deliberate perceptual
        aliasing, for the false-loop-closure gate."""
        self.add_box_room(np.array([0, 0, 0]), np.array([4, 4, 2.4]), seed0=seed_a)
        self.add_box_room(np.array([20, 0, 0]), np.array([4, 4, 2.4]), seed0=seed_b)


# ---------------------------------------------------------------- renderer

class Renderer:
    """Analytic ray-plane intersection renderer: pinhole projection with
    z-buffering across the wall set. Exact (no interpolation error) so it
    can serve as a ground-truth oracle."""

    def __init__(self, world: World, intr: Intrinsics):
        self.world = world
        self.intr = intr

    def render(self, T_world_body: np.ndarray, T_body_cam: np.ndarray = None,
               rgb_noise_std: float = 0.0, depth_noise_model=None,
               rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (rgb HxWx3 uint8, depth HxW uint16 in mm)."""
        if T_body_cam is None:
            T_body_cam = np.eye(4)
        T_world_cam = T_world_body @ T_body_cam
        T_cam_world = lie.se3_inverse(T_world_cam)

        W, H = self.intr.width, self.intr.height
        fx, fy, cx, cy = self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy

        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        dirs_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float64)],
                             axis=-1).reshape(-1, 3)
        R_wc = T_world_cam[:3, :3]
        origin_w = T_world_cam[:3, 3]
        # IMPORTANT: do NOT normalise dirs_w. dirs_cam has z==1 for every
        # pixel by construction ([(x-cx)/fx, (y-cy)/fy, 1]); a depth camera
        # reports Z-depth (camera-frame forward distance), not Euclidean
        # range along the ray. Solving the plane intersection with this
        # un-normalised, rotation-only-transformed direction makes the
        # resulting `t` exactly equal to camera-frame Z, matching how a
        # real RGB-D sensor (and our own back-projection: p = Kinv@[u,v,1]*Z)
        # defines depth. Normalising here was the bug: it silently returned
        # Euclidean range, which only coincides with Z-depth at the
        # principal point and diverges toward the image periphery.
        dirs_w = dirs_cam @ R_wc.T

        best_t = np.full(dirs_w.shape[0], np.inf)
        best_color = np.zeros((dirs_w.shape[0], 3), dtype=np.float64)
        hit_mask = np.zeros(dirs_w.shape[0], dtype=bool)

        for wall in self.world.walls:
            n = wall.normal
            denom = dirs_w @ n
            valid = np.abs(denom) > 1e-9
            # WP-A2 hardening (harmless in the original axis-aligned/
            # planar fixtures, but produces 0*inf=nan RuntimeWarnings
            # once 6-DoF motion puts a ray exactly parallel to a wall,
            # e.g. a pixel ray with a zero component while t is filled
            # with inf for invalid/parallel rays below): fill invalid
            # entries with 0.0 instead of inf. `valid` already excludes
            # these from `inb`/`closer`, so this changes nothing about
            # which points get selected -- it only keeps the unused
            # `pts`/`u`/`v` arithmetic finite for entries that are
            # discarded anyway.
            t = np.zeros(dirs_w.shape[0])
            t[valid] = ((wall.origin - origin_w) @ n) / denom[valid]
            pts = origin_w[None, :] + dirs_w * t[:, None]
            rel = pts - wall.origin[None, :]
            u = rel @ wall.u_axis
            v = rel @ wall.v_axis
            inb = valid & (t > 1e-4) & (u >= 0) & (u <= wall.u_len) & (v >= 0) & (v <= wall.v_len)
            closer = inb & (t < best_t)
            if not np.any(closer):
                continue
            colors = wall.sample_color(u[closer] / max(wall.u_len, 1e-6),
                                        v[closer] / max(wall.v_len, 1e-6))
            best_t[closer] = t[closer]
            best_color[closer] = colors
            hit_mask[closer] = True

        rgb = best_color.reshape(H, W, 3)
        if rgb_noise_std > 0 and rng is not None:
            rgb = rgb + rng.normal(scale=rgb_noise_std, size=rgb.shape)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        depth_m = best_t.reshape(H, W).copy()
        depth_m[~hit_mask.reshape(H, W)] = 0.0
        depth_m[depth_m > self.intr.depth_scale * 65535] = 0.0

        if depth_noise_model is not None and rng is not None:
            depth_m = depth_noise_model(depth_m, hit_mask.reshape(H, W), self.intr, rng)

        depth_raw = np.round(depth_m / self.intr.depth_scale).astype(np.uint16)
        return rgb, depth_raw


def stereo_depth_noise(depth_m: np.ndarray, valid: np.ndarray, intr: Intrinsics,
                        rng: np.random.Generator, sigma_d_px: float = 0.15) -> np.ndarray:
    """sigma_z = z^2 * sigma_d / (f * b) -- the stereo-depth noise model."""
    out = depth_m.copy()
    z = depth_m
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_z = (z ** 2) * sigma_d_px / (intr.fx * intr.baseline)
    noise = rng.normal(scale=np.where(valid, sigma_z, 0.0))
    out = np.where(valid, z + noise, 0.0)
    out[out < 0] = 0.0
    return out


# ---------------------------------------------------------------- trajectory

def spline_trajectory(waypoints: np.ndarray, n_samples: int, loop: bool = False) -> np.ndarray:
    """Catmull-Rom spline through waypoints (N,3) -> (n_samples,3), C1 continuous."""
    pts = waypoints
    n = len(pts)
    if loop:
        pts = np.vstack([pts[-1:], pts, pts[:2]])
    else:
        pts = np.vstack([pts[:1], pts, pts[-1:]])

    out = []
    n_segs = n if loop else n - 1
    per_seg = max(1, n_samples // max(n_segs, 1))
    for i in range(n_segs):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for s in range(per_seg):
            t = s / per_seg
            t2, t3 = t * t, t * t * t
            pt = 0.5 * ((2 * p1) + (-p0 + p2) * t +
                        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                        (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(pt)
    return np.array(out)


def poses_from_path(path_xyz: np.ndarray, up=np.array([0, 0, 1.0]),
                     look_ahead: int = 3, loop: bool = False) -> List[np.ndarray]:
    """Build world<-body poses that face along the direction of travel,
    body convention: x forward, y left, z up (then a fixed body->camera
    extrinsic converts to the optical frame).

    loop=False (Phase 0's only mode) clamps the look-ahead index at the
    end of the array via min(i+look_ahead, n-1): for the last
    `look_ahead` samples this makes the heading target the SAME fixed
    point the camera is approaching, so the heading vector collapses
    toward zero and falls back to a hardcoded [1,0,0] default -- a
    single-frame heading snap of whatever size separates the true final
    heading from world +x. This is the actual mechanism behind the
    Phase 1 plan's finding F3 (the flagship square-loop fixture's 38deg
    end-of-path discontinuity, which the odometry then fed a near-
    degenerate PnP solve). loop=True fixes this the correct way for a
    closed path: wrap the look-ahead index with modulo instead of
    clamping, so the heading at the seam continues smoothly into the
    next lap instead of snapping to a fixed direction.
    """
    n = len(path_xyz)
    poses = []
    for i in range(n):
        j = (i + look_ahead) % n if loop else min(i + look_ahead, n - 1)
        fwd = path_xyz[j] - path_xyz[i]
        if np.linalg.norm(fwd) < 1e-6:
            fwd = np.array([1.0, 0, 0])
        fwd = fwd / np.linalg.norm(fwd)
        # WP-T0 fix (found via det(R) audit during trajectory-export work):
        # the body convention is documented as x-forward, y-LEFT, z-up, but
        # this used to compute `right = fwd x up` and store it in the y
        # (left) column -- right != left, so every pose in every fixture
        # built through this function was a REFLECTION (det(R) = -1), not
        # a rotation. Confirmed on square6dof/corridor_v2 ground truth:
        # anchored ATE beat Umeyama ATE, which is impossible for two
        # proper rigid alignments -- the tell that flagged this. Fixed by
        # computing `left = up x fwd` directly (for x=fwd,y=left,z=up
        # right-handed, x cross y = z requires fwd cross left = up, i.e.
        # left = up cross fwd -- verified: det(R)=+1 for a canonical
        # fwd=[1,0,0], up=[0,0,1] case). room_orbit (which builds its own
        # rotation directly via lie.so3_exp, not through this function)
        # was never affected -- det(R)=+1 there already.
        left = np.cross(up, fwd)
        if np.linalg.norm(left) < 1e-6:
            left = np.array([0, 1.0, 0])
        left /= np.linalg.norm(left)
        true_up = np.cross(fwd, left)
        R = np.stack([fwd, left, true_up], axis=1)  # columns = body axes in world
        T = lie.make_T(R, path_xyz[i])
        poses.append(T)
    return poses


# ---------------------------------------------------------------- WP-A2: 6-DoF handheld motion


def handheld_poses(path_xyz: np.ndarray, dt: float, seed: int = 0,
                    roll_deg: float = 10.0, roll_hz: float = 0.5,
                    pitch_deg: float = 8.0, pitch_hz: float = 0.4,
                    bob_m: float = 0.05, bob_hz: float = 0.9,
                    jitter_deg: float = 3.0, jitter_tau_s: float = 0.3,
                    look_ahead: int = 3, loop: bool = False) -> List[np.ndarray]:
    """Perturb a forward-facing path (as poses_from_path would build it)
    into non-planar, non-yaw-only 6-DoF handheld motion: bounded roll and
    pitch oscillation, vertical bob, and smoothed small-amplitude random
    jitter. Phase 0's fixtures were flat and yaw-only, which is exactly
    the class of motion a mirrored/inverted odometry axis cannot be told
    apart from a correct one on (see the Phase 1 plan's finding F2) --
    the point of this function is specifically to break that symmetry.

    All perturbations are smooth, bounded-amplitude sinusoids plus an
    Ornstein-Uhlenbeck-filtered noise term (bounded by construction,
    unlike raw per-frame white noise, which would need every dt to be
    tiny to keep angular velocity in check). Nothing here directly
    enforces the validator's rate/discontinuity bounds -- call
    validate_scenario() on the result and tune amplitudes/rates if it
    fails, rather than assuming smoothness implies validity.
    """
    n = len(path_xyz)
    base = poses_from_path(path_xyz, look_ahead=look_ahead, loop=loop)
    rng = np.random.default_rng(seed)

    # OU process for smoothed jitter: dx = -x/tau*dt + noise, discretised.
    # Stationary std ~= sigma_step / sqrt(2*dt/tau) by construction; we
    # instead directly target a stationary amplitude and back out the
    # per-step noise scale, so `jitter_deg` means what it says regardless
    # of dt.
    alpha = dt / max(jitter_tau_s, 1e-6)
    alpha = min(alpha, 1.0)
    stationary_std = np.radians(jitter_deg) / 2.5  # ~2.5 sigma covers most of the range
    step_std = stationary_std * np.sqrt(2 * alpha) if alpha < 1.0 else stationary_std
    jitter_state = np.zeros(3)

    out = []
    for i in range(n):
        t = i * dt
        roll = np.radians(roll_deg) * np.sin(2 * np.pi * roll_hz * t)
        pitch = np.radians(pitch_deg) * np.sin(2 * np.pi * pitch_hz * t + 1.1)
        jitter_state = jitter_state * (1 - alpha) + rng.normal(scale=step_std, size=3)
        pert = lie.so3_exp(np.array([roll, pitch, 0.0]) + jitter_state)

        R = base[i][:3, :3] @ pert
        pos = base[i][:3, 3].copy()
        pos[2] += bob_m * np.sin(2 * np.pi * bob_hz * t)
        out.append(lie.make_T(R, pos))
    return out


def validate_scenario(world: "World", poses: List[np.ndarray], intr: Intrinsics,
                       dt: float, free_space_boxes: List[Tuple[np.ndarray, np.ndarray]],
                       min_valid_depth_frac: float = 0.5, min_clearance_m: float = 0.35,
                       max_ang_vel_deg_s: float = 150.0, max_frame_rot_ratio: float = 3.0,
                       body_cam=None, session_breaks: Tuple[int, ...] = ()) -> Tuple[bool, dict]:
    """WP-A2's scenario gate (Phase 1 plan G1A.3): before a fixture is
    allowed into the fixture set, render every frame noise-free and check
    the camera never leaves free space, never gets closer than
    min_clearance_m to a wall, always sees enough valid depth to be a
    fair test of the front end, and never takes a single-frame rotation
    step wildly larger than its neighbours (the exact mechanism behind
    Phase 0's flagship-gate 38deg end-of-path snap -- see the Phase 1
    plan section 0).

    free_space_boxes: list of (center(3,), size(3,)) AABBs; the camera
    position must fall inside at least one, with min_clearance_m margin
    from every one of that box's own faces. Explicit rather than derived
    from World.walls because the walls are arbitrary rectangles, not
    necessarily forming closed convex rooms the camera stays inside of.
    """
    if body_cam is None:
        body_cam = T_BODY_CAM
    renderer = Renderer(world, intr)
    n = len(poses)
    valid_fracs = np.zeros(n)
    inside_flags = np.zeros(n, dtype=bool)
    clearances = np.zeros(n)
    for i, T_wb in enumerate(poses):
        pos = T_wb[:3, 3]
        best_clear = -np.inf
        inside_any = False
        for c, s in free_space_boxes:
            c, s = np.asarray(c), np.asarray(s)
            half = s / 2.0
            d = half - np.abs(pos - c)   # distance to each face; positive if inside
            clear = float(np.min(d))
            best_clear = max(best_clear, clear)
            if clear >= min_clearance_m:
                inside_any = True
        inside_flags[i] = inside_any
        clearances[i] = best_clear

        _, depth = renderer.render(T_wb, body_cam)  # noise-free
        z = depth.astype(np.float64) * intr.depth_scale
        valid_fracs[i] = float(np.mean(z > 0))

    # WP-T0: every pose must be a PROPER rotation (det(R) = +1), not a
    # reflection (det(R) = -1). This is a distinct failure mode from
    # everything else this gate checks -- a reflected trajectory can
    # still stay inside free space, keep valid depth, and have smooth
    # frame-to-frame rotation steps, so none of the checks below would
    # ever catch it. Found the hard way: square6dof and corridor_v2
    # shipped with det(R)=-1 for their entire history (poses_from_path's
    # `right`-into-`left`-column bug) and only surfaced via an
    # after-the-fact anchored-vs-Umeyama ATE anomaly, not this gate.
    dets = np.array([np.linalg.det(T[:3, :3]) for T in poses])
    max_det_err = float(np.max(np.abs(dets - 1.0)))

    rot_steps_deg = []
    session_break_set = set(session_breaks)
    for i in range(1, n):
        if i in session_break_set:
            # An intentional teleport (aliasing_rooms' room-A-to-room-B
            # cut): there is no real motion between these two frames, so
            # a "rotation step" here is meaningless and must not be
            # scored as a snap or folded into the angular-velocity bound.
            continue
        xi = lie.se3_log(lie.se3_inverse(poses[i - 1]) @ poses[i])
        rot_steps_deg.append(np.degrees(np.linalg.norm(xi[3:])))
    rot_steps_deg = np.array(rot_steps_deg) if rot_steps_deg else np.zeros(0)
    ang_vel_deg_s = rot_steps_deg / dt if dt > 0 else rot_steps_deg
    median_step = float(np.median(rot_steps_deg)) if len(rot_steps_deg) else 0.0

    # A single-frame SNAP (Phase 1 plan F3: heading collapses to a
    # hardcoded default because a look-ahead index failed to wrap) is a
    # step wildly larger than its own immediate NEIGHBOURS. Comparing
    # against the whole-path median instead would also flag a
    # legitimately sharp-but-smooth corner in a fixture with mostly
    # straight segments (that corner's step is large relative to the
    # path's overall median, but grows and decays gradually over several
    # frames -- nothing a real odometry front end would call a
    # discontinuity). A local window catches the former without
    # penalising the latter.
    max_local_ratio = 0.0
    worst_rot_idx = -1
    if len(rot_steps_deg) >= 5:
        half_w = 4
        # A ratio-based snap check is meaningless when the WHOLE trajectory
        # is sub-degree per-frame (static_60s): relative noise between two
        # tiny numbers produces large ratios that carry no information
        # about a real discontinuity (a genuine F3-class snap was tens of
        # degrees). Only count a step toward the ratio check if its
        # absolute size clears a floor -- otherwise it's compared on
        # magnitude alone, which the max_single_frame_rot_deg check below
        # (interacting with max_ang_vel_deg_s) already covers.
        abs_floor_deg = 1.0
        floor = max(0.05, 0.3 * median_step)
        for i in range(len(rot_steps_deg)):
            if rot_steps_deg[i] < abs_floor_deg:
                continue
            lo, hi = max(0, i - half_w), min(len(rot_steps_deg), i + half_w + 1)
            neighbours = np.concatenate([rot_steps_deg[lo:i], rot_steps_deg[i + 1:hi]])
            local_med = float(np.median(neighbours)) if len(neighbours) else median_step
            ratio = rot_steps_deg[i] / max(local_med, floor)
            if ratio > max_local_ratio:
                max_local_ratio, worst_rot_idx = ratio, i

    report = {
        "n_frames": n,
        "frac_inside_freespace": float(np.mean(inside_flags)),
        "min_clearance_m": float(np.min(clearances)),
        "min_valid_depth_frac": float(np.min(valid_fracs)),
        "mean_valid_depth_frac": float(np.mean(valid_fracs)),
        "max_ang_vel_deg_s": float(np.max(ang_vel_deg_s)) if len(ang_vel_deg_s) else 0.0,
        "max_single_frame_rot_deg": float(np.max(rot_steps_deg)) if len(rot_steps_deg) else 0.0,
        "median_single_frame_rot_deg": median_step,
        "max_local_snap_ratio": max_local_ratio,
        "worst_rot_frame_idx": worst_rot_idx,
        "worst_frame_idx": int(np.argmin(valid_fracs)),
        "max_det_rotation_err": max_det_err,
    }
    ok = (
        np.all(inside_flags) and
        report["min_valid_depth_frac"] >= min_valid_depth_frac and
        report["max_ang_vel_deg_s"] <= max_ang_vel_deg_s and
        report["max_local_snap_ratio"] <= max_frame_rot_ratio and
        max_det_err < 1e-6
    )
    report["ok"] = ok
    if not ok:
        bad = [i for i in range(n) if not inside_flags[i]]
        report["frames_outside_freespace"] = bad[:20]
        bad_depth = [i for i in range(n) if valid_fracs[i] < min_valid_depth_frac]
        report["frames_low_valid_depth"] = bad_depth[:20]
    return ok, report


# body (x-fwd, y-left, z-up) -> camera optical (x-right, y-down, z-fwd).
# T_body_cam maps points in the camera frame into the body frame, so its
# rotation COLUMNS are the camera axes expressed in the body frame:
#   x_cam (right)   in body = -y_body   -> column 0 = [0,-1,0]
#   y_cam (down)    in body = -z_body   -> column 1 = [0, 0,-1]
#   z_cam (forward) in body =  x_body   -> column 2 = [1, 0, 0]
_R_BODY_CAM = np.column_stack([
    np.array([0, -1, 0.0]),
    np.array([0, 0, -1.0]),
    np.array([1, 0, 0.0]),
])
T_BODY_CAM = lie.make_T(_R_BODY_CAM, np.zeros(3))
