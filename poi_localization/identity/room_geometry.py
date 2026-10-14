"""Room-geometry services (R1 spec section 6): G1 ray table + metric height, G6 measurement quality.

The camera is fixed and registered to the room frame (Z up, floor z = 0), so every pixel with depth
can be turned into a room point and a height above the floor with one multiply-add:

    p_room = d * A[v, u] + t          height = d * A_z[v, u] + t_z

where A[v, u] = R @ K^-1 [u_undist, v_undist, 1] is precomputed once (H x W x 3 float32, ~3.7 MB).
Imports only numpy + poi_perception contracts (M4 import boundary).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from poi_perception.contracts import Intrinsics


class RoomGeometry:
    def __init__(self, R: np.ndarray, t: np.ndarray, intr: Intrinsics, distortion_coeffs: Optional[Tuple] = None):
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)
        self.intr = intr
        self.width, self.height = int(intr.width), int(intr.height)
        self.distortion_coeffs = distortion_coeffs
        self.A = self._build_ray_table()                       # (H, W, 3) float32, room-frame directions (z_cam = 1)

    # ------------------------------------------------------------------ G1
    def _build_ray_table(self) -> np.ndarray:
        h, w, intr = self.height, self.width, self.intr
        vs, us = np.mgrid[0:h, 0:w]
        x = (us - intr.cx) / intr.fx
        y = (vs - intr.cy) / intr.fy
        coeffs = self.distortion_coeffs
        if coeffs and any(abs(c) > 1e-9 for c in coeffs):
            import cv2                                          # lazy: the common zero-distortion case needs no cv2
            pts = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, intr.K(), np.array(coeffs, dtype=np.float64))   # normalised coords
            x, y = und[:, 0, 0].reshape(h, w), und[:, 0, 1].reshape(h, w)
        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)       # (H, W, 3) camera-frame, z = 1
        return (dirs @ self.R.T).astype(np.float32)             # row-vector form of R @ d

    def matches_frame(self, rgb_or_depth: np.ndarray) -> bool:
        return rgb_or_depth.shape[0] == self.height and rgb_or_depth.shape[1] == self.width

    def room_points(self, depth_m: np.ndarray, vs: np.ndarray, us: np.ndarray) -> np.ndarray:
        """(N, 3) room points for pixels (vs, us) whose camera-z depth (metres) is depth_m[vs, us]."""
        d = depth_m[vs, us].astype(np.float64)
        return d[:, None] * self.A[vs, us].astype(np.float64) + self.t

    def heights(self, depth_m: np.ndarray, vs: np.ndarray, us: np.ndarray) -> np.ndarray:
        """Height above the floor (m) of pixels (vs, us): one multiply-add per pixel."""
        d = depth_m[vs, us].astype(np.float64)
        return d * self.A[vs, us, 2].astype(np.float64) + self.t[2]

    def project(self, p_room: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Room points (N, 3) -> (u, v, z_cam). Pinhole (distortion ignored; the D435i colour-aligned stream is ~zero)."""
        p = np.atleast_2d(np.asarray(p_room, dtype=np.float64))
        pc = (p - self.t) @ self.R                              # R^T (p - t) in row-vector form
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.intr.fx * pc[:, 0] / z + self.intr.cx
            v = self.intr.fy * pc[:, 1] / z + self.intr.cy
        return u, v, z

    def in_image(self, u: np.ndarray, v: np.ndarray, z: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        return (z > 0) & (u >= margin_px) & (u <= self.width - 1 - margin_px) & \
               (v >= margin_px) & (v <= self.height - 1 - margin_px)


# ---------------------------------------------------------------------- G6
def depth_sigma_m(z_m: float, stereo_fx_px: float = 390.0, baseline_m: float = 0.05,
                  sigma_disparity_px: float = 0.1, bias_frac: float = 0.008) -> float:
    """Expected depth noise at range z: sqrt((z^2 sigma_d / (f b))^2 + (bias z)^2) -- the M4 noise model."""
    stereo = (z_m ** 2) * sigma_disparity_px / (stereo_fx_px * baseline_m)
    return float(np.hypot(stereo, bias_frac * z_m))


def lateral_sigma_m(z_m: float, kp_pixel_sigma_px: float, fx_colour_px: float) -> float:
    """Lateral (image-plane) noise of a keypoint at range z."""
    return float(z_m * kp_pixel_sigma_px / fx_colour_px)
