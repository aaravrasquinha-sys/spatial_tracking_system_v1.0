"""
Stage 0: derive the canonical "room" frame from the map ALONE.

M1 (live mode) leaves the dense cloud in W_cam0 -- the first mapping
camera's optical frame: arbitrary orientation, y down, not gravity-
aligned, no floor fit, no manifest. Everything downstream (M4's
walkable gate, ray-cast floor, M5's viewer) wants a frame where Z is up
and the floor IS z = 0. This module finds that frame:

  * floor  -- the largest near-horizontal plane that BOUNDS the cloud from
              below (almost nothing under it). Tabletops have lots of
              cloud beneath them; the floor doesn't. The up-prior
              (-y in W_cam0) is only used to pick a sign and reject
              non-horizontal planes; it may be off by tens of degrees.
  * yaw    -- dominant wall direction (histogram of horizontal normals),
              disambiguated among the four 90-degree choices by continuity
              with the mapping-start heading. No orthogonality is ASSUMED:
              if no wall direction dominates, falls back to start heading.
  * origin -- the floor point below the map origin (= first mapping camera).

Room Z = the FITTED FLOOR NORMAL, not gravity: the walking surface is
what ray-casting and the walkable mask care about (planning doc, sec 3).
The map's own accuracy (floor flatness / quadrant tilt) is reported and
later becomes a floor on the calibration's sigma.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from scipy.ndimage import gaussian_filter1d

from pyslam.core import lie
from pyslam.core.log import get_logger
from pyslam.anchor.config import AnchorConfig
from pyslam.anchor import geo

log = get_logger("anchor.room_frame")


class RoomFrameError(RuntimeError):
    pass


@dataclass
class RoomFrame:
    T_room_map: np.ndarray
    floor: dict
    yaw_source: str
    yaw_peak_frac: Optional[float]
    walls: List[dict]
    up_prior_angle_deg: float
    map_trans_sigma_m: float
    map_rot_sigma_deg: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "T_room_map": self.T_room_map.tolist(),
            "floor": self.floor,
            "yaw_source": self.yaw_source,
            "yaw_peak_frac": self.yaw_peak_frac,
            "walls": self.walls,
            "floor_normal_vs_up_prior_deg": self.up_prior_angle_deg,
            "map_accuracy_floor": {"trans_m": self.map_trans_sigma_m, "rot_deg": self.map_rot_sigma_deg},
            "warnings": self.warnings,
        }


def _find_horizontal_planes(pts5: np.ndarray, prior: np.ndarray, cfg: AnchorConfig, rng, max_planes=5):
    cos_tol = np.cos(np.radians(cfg.up_prior_tol_deg))
    ok = lambda n: abs(float(n @ prior)) >= cos_tol
    work = pts5
    out = []
    for _ in range(max_planes):
        r = geo.ransac_plane(work, cfg.floor_ransac_dist_m, cfg.floor_ransac_iters, rng, normal_ok=ok)
        if r is None:
            break
        n, off, mask = r
        if mask.sum() < cfg.floor_min_inliers:
            break
        out.append((n, off, int(mask.sum())))
        work = work[~mask]
        if work.shape[0] < cfg.floor_min_inliers:
            break
    return out


def _select_floor(planes, pts5, prior, cfg):
    """Return (n_up, offset, n_inliers, below_frac, diag_list)."""
    diag = []
    cands = []
    for n, off, cnt in planes:
        sd = pts5 @ n - off
        s = 1.0 if np.median(sd) >= 0 else -1.0     # orient INTO the room (toward the bulk of the cloud)
        n_in, off_in = s * n, s * off
        below = float(np.mean((pts5 @ n_in - off_in) < -0.05))
        cos_up = float(n_in @ prior)
        diag.append({"inliers": cnt, "below_frac": round(below, 4), "cos_with_up_prior": round(cos_up, 3)})
        if cos_up > 0:
            cands.append((n_in, off_in, cnt, below))
    if not cands:
        raise RoomFrameError(
            "no horizontal plane facing the up-prior found -- is the map's up direction very different "
            "from W_cam0 -y? Pass --up-prior-map x,y,z (a rough 'up' in the map frame).")
    biggest = max(c[2] for c in cands)
    strong = [c for c in cands if c[2] >= cfg.floor_lowest_frac * biggest]
    bounding = [c for c in strong if c[3] <= 0.05]
    if bounding:
        chosen = max(bounding, key=lambda c: c[2])
    else:
        # nothing bounds the cloud from below (floor barely mapped, or big holes / outliers below):
        # take the LOWEST strong candidate and say so.
        chosen = min(strong, key=lambda c: c[1] - 0.0)
        log.warning("no floor candidate bounds the cloud from below; picked the lowest large horizontal plane")
    return chosen, diag


def _quadrant_fits(floor_pts: np.ndarray, n: np.ndarray, off: float, u, v, cfg):
    c = floor_pts.mean(axis=0)
    a = (floor_pts - c) @ u
    b = (floor_pts - c) @ v
    quads = []
    for qa in (a >= 0, a < 0):
        for qb in (b >= 0, b < 0):
            m = qa & qb
            if m.sum() < 300:
                quads.append(None)
                continue
            qn, qo = geo.fit_plane_ls(floor_pts[m])
            if qn @ n < 0:
                qn, qo = -qn, -qo
            qc = floor_pts[m].mean(axis=0)
            tilt = geo.angle_between_deg(qn, n)
            # height of the quadrant's own plane vs the global plane, evaluated at the quadrant centroid
            hq = float((qn @ qc) - qo)      # ~0 (centroid lies on its own plane)
            hg = float((n @ qc) - off)      # global plane's signed distance to the quadrant centroid
            quads.append({"n_pts": int(m.sum()), "tilt_deg": round(tilt, 3),
                          "height_diff_m": round(abs(hg - hq), 4)})
    return [q for q in quads if q is not None]


def _wall_planes(pts_band5, up, cfg, rng):
    """Vertical planes (walls) via RANSAC restricted to horizontal normals."""
    ok = lambda n: abs(float(n @ up)) <= 0.10
    work = pts_band5
    walls = []
    for _ in range(cfg.n_walls):
        r = geo.ransac_plane(work, cfg.wall_ransac_dist_m, 400, rng, normal_ok=ok)
        if r is None:
            break
        n, off, mask = r
        if mask.sum() < cfg.wall_min_inliers:
            break
        walls.append((n, off, work[mask]))
        work = work[~mask]
    return walls


def derive_room_frame(pts_map: np.ndarray, cfg: Optional[AnchorConfig] = None, seed: int = 0,
                      map_forward_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> RoomFrame:
    cfg = cfg or AnchorConfig()
    rng = np.random.default_rng(seed)
    warnings: List[str] = []
    prior = np.asarray(cfg.up_prior_map, float)
    prior = prior / np.linalg.norm(prior)

    pts5, _ = geo.voxel_downsample(pts_map, cfg.floor_ransac_voxel_m)
    planes = _find_horizontal_planes(pts5, prior, cfg, rng)
    if not planes:
        raise RoomFrameError(
            f"no horizontal plane with >= {cfg.floor_min_inliers} inliers (5 cm cloud): the floor was not "
            f"mapped well enough. Re-map with the camera tilted down periodically (planning doc, M1 capture protocol).")
    (n_up, off, cnt, below), diag = _select_floor(planes, pts5, prior, cfg)

    # refine on the full-resolution cloud (looser gate: a slightly bowed floor must still be one plane)
    for _ in range(3):
        m = np.abs(pts_map @ n_up - off) < 1.5 * cfg.floor_ransac_dist_m
        if m.sum() < 50:
            break
        n_new, o_new = geo.fit_plane_ls(pts_map[m])
        if n_new @ n_up < 0:
            n_new, o_new = -n_new, -o_new
        n_up, off = n_new, o_new
    m_in = np.abs(pts_map @ n_up - off) < cfg.floor_ransac_dist_m
    floor_pts = pts_map[m_in]
    rms = float(np.sqrt(np.mean((floor_pts @ n_up - off) ** 2)))
    if rms > cfg.floor_flat_rms_m:
        warnings.append(f"floor inlier RMS {rms*1000:.1f} mm > {cfg.floor_flat_rms_m*1000:.0f} mm: map floor is not flat "
                        f"(drift or bowing) -- expect a larger calibration sigma / check M1 loop closure")

    u0, v0 = geo.horizontal_basis(n_up, map_forward_hint)
    quads = _quadrant_fits(floor_pts, n_up, off, u0, v0, cfg)
    max_tilt = max([q["tilt_deg"] for q in quads], default=0.0)
    max_h = max([q["height_diff_m"] for q in quads], default=0.0)
    if len(quads) < 4:
        warnings.append(f"only {len(quads)}/4 floor quadrants have enough points to fit -- floor coverage is uneven")
    if max_tilt > cfg.floor_quadrant_tilt_deg or max_h > cfg.floor_quadrant_height_m:
        warnings.append(f"floor quadrants disagree (max tilt {max_tilt:.2f} deg, max height {max_h*100:.1f} cm): "
                        f"the map is bowed; calibration sigma will be inflated accordingly")

    # ---- yaw from walls ----
    h_above = pts_map @ n_up - off
    band = (h_above > cfg.wall_min_h_m) & (h_above < cfg.wall_max_h_m)
    pts_band = pts_map[band]
    pts_band3, _ = geo.voxel_downsample(pts_band, 0.04) if pts_band.shape[0] else (pts_band, None)
    yaw_source = "start_heading"
    phi = 0.0
    peak_frac = None
    walls_out: List[dict] = []
    if pts_band3.shape[0] > 2000:
        nrm = geo.estimate_normals(pts_band3, k=16)
        horiz = np.abs(nrm @ n_up) < cfg.wall_normal_vertical_tol
        if horiz.sum() > 1000:
            ang = np.arctan2(nrm[horiz] @ v0, nrm[horiz] @ u0) % np.pi      # normal direction mod 180
            hist, _ = np.histogram(ang, bins=180, range=(0, np.pi))
            hs = gaussian_filter1d(hist.astype(float), 1.5, mode="wrap")
            k = int(np.argmax(hs))
            alpha = (k + 0.5) * np.pi / 180.0
            d = np.abs(((ang - alpha) + np.pi / 2) % np.pi - np.pi / 2)
            peak_frac = float(np.mean(d < np.radians(5.0)))
            if peak_frac >= cfg.wall_hist_min_peak_frac:
                # among alpha + k*pi/2 choose the one closest to the start heading (phi = 0)
                cands = [geo.wrap_pi(alpha + j * np.pi / 2) for j in range(4)]
                phi = min(cands, key=abs)
                yaw_source = "wall_histogram"
            else:
                warnings.append(f"no dominant wall direction (peak holds {peak_frac:.0%} of horizontal normals): "
                                f"room X falls back to the mapping-start heading")
    else:
        warnings.append("too few points between wall heights to find a wall direction; using start heading")

    # wall planes (map frame) -- used both to REFINE the yaw (histogram is 1-degree quantised) and reported
    pts_band5, _ = geo.voxel_downsample(pts_band, cfg.floor_ransac_voxel_m) if pts_band.shape[0] else (pts_band, None)
    walls_map = _wall_planes(pts_band5, n_up, cfg, rng)
    if yaw_source == "wall_histogram" and walls_map:
        ds, ws = [], []
        for wn, _, wp in walls_map:
            a = np.arctan2(wn @ v0, wn @ u0)
            d = ((a - phi + np.pi / 4) % (np.pi / 2)) - np.pi / 4      # deviation from the nearest axis-multiple of phi
            if abs(d) < np.radians(10.0):
                ds.append(d); ws.append(wp.shape[0])
        if ds:
            phi = phi + float(np.average(ds, weights=ws))

    x_dir = np.cos(phi) * u0 + np.sin(phi) * v0
    y_dir = np.cross(n_up, x_dir)
    R = np.stack([x_dir, y_dir, n_up], axis=0)                       # p_room = R (p_map - o)
    o = off * n_up                                                   # floor point below the map origin
    t = -R @ o
    T = lie.make_T(R, t)

    # walls (reported in the ROOM frame, inward-facing normals)
    centroid_room = R @ (floor_pts.mean(axis=0) - o)
    for wid, (wn, woff, wpts) in enumerate(walls_map):
        wn_r = R @ wn
        wo_r = float(woff - wn @ o)                                  # n_room . x_room = offset_room
        if wn_r @ centroid_room - wo_r < 0:
            wn_r, wo_r = -wn_r, -wo_r
        tang = np.cross(np.array([0, 0, 1.0]), wn_r)
        wpts_r = (R @ (wpts - o).T).T
        ext = float(np.ptp(wpts_r @ tang))
        walls_out.append({
            "id": wid, "normal_in_room": [round(float(x), 5) for x in wn_r], "offset_room": round(wo_r, 4),
            "n_inliers": int(wpts.shape[0]), "length_m": round(ext, 2),
            "yaw_deg": round(float(np.degrees(np.arctan2(wn_r[1], wn_r[0]))), 2),
        })

    map_rot = max(cfg.map_rot_floor_deg, max_tilt / 2.0)
    map_trans = max(cfg.map_trans_floor_m, max_h / 2.0, rms)
    floor_info = {
        "normal_map": [float(x) for x in n_up], "offset_map": float(off),
        "n_inliers": int(m_in.sum()), "rms_m": rms, "quadrants": quads,
        "max_quadrant_tilt_deg": max_tilt, "max_quadrant_height_diff_m": max_h,
        "candidate_planes": diag,
    }
    up_ang = geo.angle_between_deg(n_up, prior)
    log.info(f"Room frame: floor rms={rms*1000:.1f}mm, {int(m_in.sum())} pts, quad tilt<= {max_tilt:.2f} deg, "
             f"yaw from {yaw_source}, floor normal is {up_ang:.1f} deg from the up-prior")
    return RoomFrame(T_room_map=T, floor=floor_info, yaw_source=yaw_source, yaw_peak_frac=peak_frac,
                     walls=walls_out, up_prior_angle_deg=up_ang, map_trans_sigma_m=map_trans,
                     map_rot_sigma_deg=map_rot, warnings=warnings)
