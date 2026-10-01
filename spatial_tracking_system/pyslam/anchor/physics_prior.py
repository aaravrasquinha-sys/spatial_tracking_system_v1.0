"""
Stage 2: the physics prior -- 3 of the 6 DOF from the camera ITSELF, with no map involved.

  * roll/pitch  from the resting accelerometer (up_cam)
  * height + a SECOND, independent roll/pitch from RANSAC on the camera's own depth (the floor)

Same idea (and same sign convention) as M4 Phase A's local_floor_frame.py, re-implemented here
(the two repos are deliberately independent) so this module can also hand back the diagnostics
the verification stage needs. What remains unknown afterwards is a planar rigid motion: x, y, yaw.

Frames: 'L' = levelled camera frame: origin at the camera, z = up, x = camera-forward projected on
the horizontal. p_L = R_L_cam @ p_cam. The floor sits at z = -h in L.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.anchor import geo
from pyslam.anchor.config import AnchorConfig
from pyslam.core.log import get_logger

log = get_logger("anchor.physics")


class PhysicsPriorError(RuntimeError):
    pass


def level_rotation(up_cam: np.ndarray) -> np.ndarray:
    up = np.asarray(up_cam, float)
    up = up / np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up          # camera forward projected on the horizontal plane
    n = np.linalg.norm(fwd)
    if n < 0.05:
        raise PhysicsPriorError("camera is pointing (almost) straight up/down: the horizontal 'forward' axis is undefined")
    x = fwd / n
    y = np.cross(up, x)
    return np.stack([x, y, up], axis=0)


def fit_floor_in_camera(pts_cam: np.ndarray, up_hint: Optional[np.ndarray], cfg: AnchorConfig, seed: int = 0) -> dict:
    """RANSAC the floor in the camera's own cloud. With an up-hint: candidates = lowest points along it,
    normals within 25 deg of it. Without: assume a roughly upright camera (up ~ -y) and say so."""
    rng = np.random.default_rng(seed)
    hint = up_hint if up_hint is not None else np.array([0.0, -1.0, 0.0])
    hint = hint / np.linalg.norm(hint)
    proj = pts_cam @ (-hint)
    thr = np.percentile(proj, cfg.floor_fit_lowest_percentile)
    cand = pts_cam[proj >= thr]
    if cand.shape[0] < cfg.floor_fit_min_inliers:
        cand = pts_cam
    cos_tol = np.cos(np.radians(25.0 if up_hint is not None else 45.0))
    r = geo.ransac_plane(cand, cfg.floor_fit_dist_m, cfg.floor_fit_ransac_iters, rng,
                         normal_ok=lambda n: abs(float(n @ hint)) >= cos_tol)
    if r is None or r[2].sum() < cfg.floor_fit_min_inliers:
        raise PhysicsPriorError(
            f"could not fit the floor in the camera's own depth ({0 if r is None else int(r[2].sum())} inliers, need "
            f"{cfg.floor_fit_min_inliers}). The camera must see a reasonable patch of FLOOR (tilt it down / mount lower).")
    n, off, mask = r
    if n @ hint < 0:
        n, off = -n, -off
    inl = cand[mask]
    # extent is judged on EVERY point of the cloud lying on the plane (not just the lowest-percentile candidates,
    # which slice a large visible floor into a strip and made this check spuriously strict)
    on_plane = pts_cam[np.abs(pts_cam @ n - off) < cfg.floor_fit_dist_m]
    c = on_plane - on_plane.mean(axis=0)
    ev = np.linalg.eigvalsh(c.T @ c / max(len(c), 1))
    if np.sqrt(max(ev[1], 0.0)) < cfg.floor_fit_min_extent_m:
        raise PhysicsPriorError(
            f"the 'floor' RANSAC found only a thin strip (minor in-plane extent {np.sqrt(max(ev[1],0))*100:.0f} cm < "
            f"{cfg.floor_fit_min_extent_m*100:.0f} cm) -- typically a horizontal slice through a wall: NO real floor patch is visible. "
            f"Tilt the camera down / mount it further from the wall so a floor area is in view.")
    rms = float(np.sqrt(np.mean((inl @ n - off) ** 2)))
    h = float(-off)                    # camera origin's height above the plane (n up, plane n.x = off < 0)
    if h <= 0:
        raise PhysicsPriorError(f"floor fit put the camera {h:.2f} m 'above' the floor (negative): bad fit or wrong up-hint")
    return {"up_floor_cam": n, "height_m": h, "rms_m": rms, "n_inliers": int(mask.sum()),
            "tilt_vs_hint_deg": geo.angle_between_deg(n, hint)}


def physics_prior(cloud_pts_cam: np.ndarray, up_imu: Optional[np.ndarray], imu_ok: bool,
                  cfg: AnchorConfig, seed: int = 0) -> dict:
    """-> dict(R_L_cam, up_used_cam, up_source, height_m, floor, imu_vs_floor_tilt_deg, warnings)."""
    warnings = []
    fl = fit_floor_in_camera(cloud_pts_cam, up_imu if imu_ok else None, cfg, seed)
    if up_imu is not None and imu_ok:
        up, src = up_imu, "imu"
        tilt = geo.angle_between_deg(up_imu, fl["up_floor_cam"])
    else:
        up, src, tilt = fl["up_floor_cam"], "floor_fit", None
        msg = "no usable IMU up-vector" if up_imu is None else "IMU window was not usable (camera moving / bad |g|)"
        warnings.append(f"{msg}: levelling from the camera's own floor fit only -- the IMU cross-check is unavailable")
        if cfg.require_imu:
            warnings.append("require_imu=True: this calibration cannot pass acceptance without the IMU check")
    return {"R_L_cam": level_rotation(up), "up_used_cam": np.asarray(up, float) / np.linalg.norm(up),
            "up_source": src, "height_m": fl["height_m"], "floor": fl, "imu_vs_floor_tilt_deg": tilt,
            "warnings": warnings}
