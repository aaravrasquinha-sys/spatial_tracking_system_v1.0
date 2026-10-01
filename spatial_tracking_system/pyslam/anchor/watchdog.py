"""
Calibration watchdog (library; the M6 runtime daemon in the STS repo calls it, and
`run_anchor.py check` runs it once from the CLI). Three layers, cheapest first:

  1. TILT      accelerometer up-vector vs the calibrated one. A bump that changes tilt by > ~0.2 deg is
               caught in about a second at essentially zero cost. Blind to pure yaw / translation.
  2. DEPTH     live depth on the STABLE reference pixels vs the stored reference depth. If more than
               `wd_depth_frac` of them moved by > `wd_depth_move_m` for `wd_consecutive` consecutive checks
               => suspect. People occlude some pixels, hence a fraction + persistence, not a single threshold.
  3. RECHECK   (on demand, when 1 or 2 fire) fast ICP of a fresh median depth against the map from the
               calibrated pose; a pose shift > 1 cm / 0.3 deg confirms the camera moved.

It NEVER silently re-calibrates: it only reports 'ok' / 'suspect' (+ the measured shift). Adopting a new
pose requires re-running the full calibration and passing every gate.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional
import numpy as np

from pyslam.core import lie
from pyslam.anchor import geo
from pyslam.anchor.config import AnchorConfig


@dataclass
class WatchdogState:
    status: str = "ok"                 # ok | suspect
    reason: str = ""
    tilt_deg: Optional[float] = None
    moved_frac: Optional[float] = None
    consecutive: int = 0


class AnchorWatchdog:
    def __init__(self, calibration_path: str, reference_npz_path: str, cfg: Optional[AnchorConfig] = None):
        self.cfg = cfg or AnchorConfig()
        self.cal = json.load(open(calibration_path))
        z = np.load(reference_npz_path, allow_pickle=False)
        self.ref_depth = z["depth_med"]
        self.ref_stable = z["stable"].astype(bool)
        self.T = np.array(self.cal["T_room_cam"], float)
        up = self.cal.get("gravity_up_cam")
        self.ref_up = None if up is None else np.array(up, float)
        self.state = WatchdogState()

    def check_tilt(self, up_cam_now: Optional[np.ndarray]) -> Optional[float]:
        if up_cam_now is None or self.ref_up is None:
            return None
        a = geo.angle_between_deg(up_cam_now, self.ref_up)
        self.state.tilt_deg = a
        if a > self.cfg.wd_tilt_deg:
            self._raise(f"camera tilt changed by {a:.2f} deg (> {self.cfg.wd_tilt_deg} deg)")
        return a

    def check_depth(self, depth_now: np.ndarray) -> float:
        """depth_now: (H,W) float metres, NaN invalid -- ideally a short temporal median, not one frame."""
        both = self.ref_stable & np.isfinite(depth_now)
        n = int(self.ref_stable.sum())
        if n == 0 or both.sum() < 0.3 * n:
            # cannot see the reference structure at all (lens covered? camera turned away?)
            self.state.moved_frac = 1.0
            self.state.consecutive += 1
            if self.state.consecutive >= self.cfg.wd_consecutive:
                self._raise("reference structure no longer visible in the live depth")
            return 1.0
        moved = np.abs(depth_now - self.ref_depth)[both] > self.cfg.wd_depth_move_m
        frac = float(moved.mean())
        self.state.moved_frac = frac
        if frac > self.cfg.wd_depth_frac:
            self.state.consecutive += 1
            if self.state.consecutive >= self.cfg.wd_consecutive:
                self._raise(f"{frac:.0%} of stable reference pixels moved > {self.cfg.wd_depth_move_m*100:.0f} cm "
                            f"for {self.state.consecutive} consecutive checks")
        else:
            self.state.consecutive = 0     # (status stays 'suspect' until a recheck_pose() clears it -- never auto-clears)
        return frac

    def recheck_pose(self, cap, map_pts_room: np.ndarray, map_colors=None) -> dict:
        """Fast ICP from the calibrated pose. `cap` = a fresh StaticCapture (short). Returns the measured shift."""
        from pyslam.anchor.query_cloud import capture_to_cloud
        from pyslam.anchor.icp import MapTarget, icp
        cloud = capture_to_cloud(cap, self.cfg)
        tgt = MapTarget.crop(map_pts_room, map_colors, self.T[:3, 3], self.cfg.range_max_m + self.cfg.icp_crop_margin_m)
        res = icp(cloud["pts"], cloud["weights"], cloud["sigma"], tgt, self.T, self.cfg,
                  scales=(0.10, 0.05, 0.025), iters=(20, 20, 20))
        dt, dr = geo.pose_delta(self.T, res.T)
        moved = (dt > self.cfg.wd_pose_trans_m or dr > self.cfg.wd_pose_rot_deg)
        good = res.converged and res.fitness >= self.cfg.gate_fitness_min
        out = {"shift_m": dt, "shift_deg": dr, "fitness": res.fitness, "rmse_m": res.rmse_m, "converged": res.converged,
               "verdict": "moved" if (moved and good) else ("ok" if good else "inconclusive")}
        if out["verdict"] == "moved":
            self._raise(f"pose recheck: camera moved {dt*100:.1f} cm / {dr:.2f} deg since calibration")
        elif out["verdict"] == "ok":
            self.state = WatchdogState()
        out["T_measured"] = res.T
        return out

    def _raise(self, reason: str):
        self.state.status = "suspect"
        self.state.reason = reason

    def health_field(self) -> str:
        """Value for the M6 health message's `calib` field."""
        return "ok" if self.state.status == "ok" else "suspect"
