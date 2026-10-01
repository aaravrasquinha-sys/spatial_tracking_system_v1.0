"""
Stage 1: static capture. The ONLY place in the anchor package that
touches hardware (via the project's existing RealSenseSource, so the
capture profile / depth alignment / IMU-extrinsic handling is exactly
what mapping used).

capture_static()  -> StaticCapture   (hardware or a recorded .bag)
build_capture()   -> StaticCapture   (pure numpy; what the gate tests call)

What is kept, and why:
  depth_med   per-pixel temporal MEDIAN depth (metres, NaN = invalid). Median, not mean:
              robust to flicker/flying pixels and to a person walking through.
  depth_mad   per-pixel temporal MAD -> a stability mask (unstable pixels are not calibration data)
  valid_frac  share of frames in which the pixel had depth
  sub_med     K interleaved sub-medians -> repeatability / empirical covariance without
              needing K separate capture sessions
  rgb_med     median colour image (overlay + colour tie-break)
  up_cam      unit 'up' in the COLOUR camera frame from the resting accelerometer
              (specific force points up; see local_floor_frame.py's note)
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from pyslam.core.log import get_logger
from pyslam.anchor.config import AnchorConfig

log = get_logger("anchor.capture")


@dataclass
class StaticCapture:
    depth_med: np.ndarray            # (H,W) float32 metres, NaN invalid
    depth_mad: np.ndarray            # (H,W) float32 metres
    valid_frac: np.ndarray           # (H,W) float32
    rgb_med: np.ndarray              # (H,W,3) uint8
    sub_med: np.ndarray              # (K,H,W) float32
    intr: dict                       # fx fy cx cy width height depth_scale baseline
    up_cam: Optional[np.ndarray]     # (3,) unit, or None if no usable IMU window
    imu_info: dict = field(default_factory=dict)
    n_frames: int = 0
    serial: Optional[str] = None
    capture_profile_sha: Optional[str] = None
    created: str = ""
    source: str = "unknown"

    def K(self) -> np.ndarray:
        i = self.intr
        return np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])

    def save(self, path: str) -> None:
        meta = {"intr": self.intr, "imu_info": self.imu_info, "n_frames": self.n_frames,
                "serial": self.serial, "capture_profile_sha": self.capture_profile_sha,
                "created": self.created, "source": self.source,
                "up_cam": None if self.up_cam is None else [float(x) for x in self.up_cam]}
        np.savez_compressed(path, depth_med=self.depth_med, depth_mad=self.depth_mad,
                            valid_frac=self.valid_frac, rgb_med=self.rgb_med, sub_med=self.sub_med,
                            meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str) -> "StaticCapture":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(depth_med=z["depth_med"], depth_mad=z["depth_mad"], valid_frac=z["valid_frac"],
                   rgb_med=z["rgb_med"], sub_med=z["sub_med"], intr=meta["intr"],
                   up_cam=None if meta["up_cam"] is None else np.array(meta["up_cam"]),
                   imu_info=meta.get("imu_info", {}), n_frames=meta["n_frames"], serial=meta.get("serial"),
                   capture_profile_sha=meta.get("capture_profile_sha"), created=meta.get("created", ""),
                   source=meta.get("source", ""))


def _median_and_mad(stack: np.ndarray, row_chunk: int = 60):
    """stack (N,H,W) float32 with NaN invalid -> (median, mad, valid_frac). Row-chunked so a
    150-frame 640x480 capture never needs a second full copy in memory."""
    n, h, w = stack.shape
    med = np.full((h, w), np.nan, np.float32)
    mad = np.full((h, w), np.nan, np.float32)
    vf = (~np.isnan(stack)).mean(axis=0).astype(np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN slices are expected
        for s in range(0, h, row_chunk):
            e = min(h, s + row_chunk)
            blk = stack[:, s:e]
            m = np.nanmedian(blk, axis=0)
            med[s:e] = m
            mad[s:e] = np.nanmedian(np.abs(blk - m[None]), axis=0) * 1.4826
    return med, mad, vf


def imu_to_up_cam(imu_windows: List[np.ndarray], R_body_cam: np.ndarray, cfg: AnchorConfig) -> tuple:
    """-> (up_cam or None, info dict). Each window is (N,7) [t,gx,gy,gz,ax,ay,az] in the IMU/body frame.
    v_body = R_body_cam @ v_cam  (same convention as RealSenseSource.imu_to_color_rotation), so
    up_cam = R_body_cam^T @ up_body."""
    ws = [w for w in imu_windows if w is not None and w.shape[0] > 0]
    if not ws:
        return None, {"ok": False, "reason": "no IMU samples captured"}
    imu = np.concatenate(ws, axis=0)
    if imu.shape[0] < 20:
        return None, {"ok": False, "reason": f"only {imu.shape[0]} IMU samples"}
    gyro, acc = imu[:, 1:4], imu[:, 4:7]
    gyro_std = float(np.max(np.std(gyro, axis=0)))
    gyro_mean = float(np.linalg.norm(np.mean(gyro, axis=0)))
    acc_std = float(np.max(np.std(acc, axis=0)))
    f_body = np.mean(acc, axis=0)
    norm = float(np.linalg.norm(f_body))
    info = {"n_samples": int(imu.shape[0]), "gyro_std_rad_s": gyro_std, "gyro_mean_rad_s": gyro_mean,
            "accel_std_mps2": acc_std, "accel_norm_mps2": norm}
    if norm < 1e-6:
        info.update(ok=False, reason="degenerate accelerometer reading")
        return None, info
    # quasi-static AND consistent with 1 g (an uncalibrated D435i accel can be a few % off; >15% = broken)
    static = (gyro_std <= cfg.imu_max_gyro_std_rad_s and gyro_mean <= 0.05
              and acc_std <= cfg.imu_max_accel_std_mps2)
    plausible_g = 8.3 <= norm <= 11.3
    up_body = f_body / norm
    up_cam = np.asarray(R_body_cam, float).T @ up_body
    up_cam = up_cam / np.linalg.norm(up_cam)
    info.update(ok=bool(static and plausible_g), static=bool(static), plausible_g=bool(plausible_g))
    if not info["ok"]:
        info["reason"] = ("camera not still during IMU window" if not static else
                          f"|accel|={norm:.2f} m/s^2 is not ~9.8 -- IMU uncalibrated/broken?")
    return up_cam, info


def build_capture(depth_raw: np.ndarray, rgb_frames: np.ndarray, intr: dict,
                  imu_windows: Optional[List[np.ndarray]], R_body_cam: Optional[np.ndarray],
                  cfg: Optional[AnchorConfig] = None, serial: Optional[str] = None,
                  capture_profile_sha: Optional[str] = None, source: str = "unknown",
                  up_cam_override: Optional[np.ndarray] = None) -> StaticCapture:
    """depth_raw: (N,H,W) uint16 raw units (0 = invalid); rgb_frames: (M,H,W,3) uint8 (M<=N is fine)."""
    cfg = cfg or AnchorConfig()
    n = depth_raw.shape[0]
    if n < 10:
        raise ValueError(f"need >= 10 frames for a meaningful temporal median, got {n}")
    scale = float(intr["depth_scale"])
    d = depth_raw.astype(np.float32) * scale
    d[depth_raw == 0] = np.nan
    med, mad, vf = _median_and_mad(d)
    k = max(2, cfg.capture_n_subsets)
    subs = np.stack([_median_and_mad(d[i::k])[0] for i in range(k)])
    rgb_med = np.median(rgb_frames, axis=0).astype(np.uint8) if rgb_frames is not None and len(rgb_frames) else \
        np.zeros(med.shape + (3,), np.uint8)
    if up_cam_override is not None:
        up_cam, info = np.asarray(up_cam_override, float), {"ok": True, "reason": "override"}
    elif imu_windows is not None and R_body_cam is not None:
        up_cam, info = imu_to_up_cam(imu_windows, R_body_cam, cfg)
    else:
        up_cam, info = None, {"ok": False, "reason": "IMU not provided"}
    return StaticCapture(depth_med=med, depth_mad=mad, valid_frac=vf, rgb_med=rgb_med, sub_med=subs,
                         intr=dict(intr), up_cam=up_cam, imu_info=info, n_frames=int(n), serial=serial,
                         capture_profile_sha=capture_profile_sha,
                         created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source=source)


def capture_static(source_factory, cfg: Optional[AnchorConfig] = None, n_frames: Optional[int] = None,
                   warmup_s: Optional[float] = None, capture_profile_sha: Optional[str] = None,
                   serial: Optional[str] = None, is_playback: bool = False) -> StaticCapture:
    """HARDWARE (or .bag playback). Opens the camera through `source_factory()` (a callable returning a
    pyslam RealSenseSource -- the same factory run_live_map.py builds), discards a warm-up period
    (D435i depth drifts slightly while the projector/sensor warm; auto-exposure also settles), then
    records n_frames aligned depth+RGB and the accelerometer, and closes the camera.
    Do NOT run this while another process owns the camera."""
    cfg = cfg or AnchorConfig()
    n_frames = n_frames or cfg.capture_frames
    warmup = 0.0 if is_playback else (cfg.capture_warmup_s if warmup_s is None else warmup_s)
    src = source_factory()
    try:
        intr_o = src.intrinsics()
        intr = {"fx": intr_o.fx, "fy": intr_o.fy, "cx": intr_o.cx, "cy": intr_o.cy, "width": intr_o.width,
                "height": intr_o.height, "depth_scale": intr_o.depth_scale, "baseline": intr_o.baseline}
        R_body_cam = src.imu_to_color_rotation()
        depth, rgbs, imus = [], [], []
        t0 = None
        log.info(f"Static capture: warm-up {warmup:.0f}s (+{cfg.capture_discard_s:.1f}s discard), then {n_frames} frames. "
                 f"Keep the room EMPTY and the camera UNTOUCHED.")
        for fr in src:
            if t0 is None:
                t0 = fr.t
            if fr.t - t0 < warmup + cfg.capture_discard_s:
                continue
            depth.append(fr.depth)
            if len(depth) % 10 == 0:
                rgbs.append(fr.rgb)
            if fr.imu is not None and fr.imu.shape[0]:
                imus.append(fr.imu)
            if len(depth) >= n_frames:
                break
        if len(depth) < min(n_frames, 30):
            raise RuntimeError(f"only captured {len(depth)} frames (wanted {n_frames}); camera stalled or bag too short")
        return build_capture(np.stack(depth), np.stack(rgbs) if rgbs else np.zeros((0,)), intr, imus,
                             R_body_cam, cfg, serial=serial, capture_profile_sha=capture_profile_sha,
                             source="bag" if is_playback else "live")
    finally:
        try:
            src.close()
        except Exception:
            pass
