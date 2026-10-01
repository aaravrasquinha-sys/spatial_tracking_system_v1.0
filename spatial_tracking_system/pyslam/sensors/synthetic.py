from __future__ import annotations
from typing import Iterator, Optional, List
import numpy as np

from pyslam.core.types import Frame, Intrinsics
from pyslam.core import lie
from tests.synth.world import World, Renderer, stereo_depth_noise, T_BODY_CAM


class SyntheticSource:
    """Implements SensorSource. Wraps a World + a sequence of ground-truth
    body poses, producing Frame objects with rgb/depth/imu/gt_pose."""

    def __init__(self, world: World, intr: Intrinsics, poses_wb: List[np.ndarray],
                 dt: float = 1.0 / 30.0, rgb_noise_std: float = 2.0,
                 depth_noise_sigma_d_px: float = 0.15, imu_hz: float = 200.0,
                 seed: int = 0, add_noise: bool = True):
        self.world = world
        self.intr = intr
        self.poses = poses_wb
        self.dt = dt
        self.rgb_noise_std = rgb_noise_std if add_noise else 0.0
        self.depth_noise_sigma = depth_noise_sigma_d_px if add_noise else 0.0
        self.imu_hz = imu_hz
        self.rng = np.random.default_rng(seed)
        self.renderer = Renderer(world, intr)
        self._add_noise = add_noise

    def intrinsics(self) -> Intrinsics:
        return self.intr

    def _imu_between(self, T0: np.ndarray, T1: np.ndarray, t0: float, t1: float) -> np.ndarray:
        """Finite-difference angular velocity / specific force between two
        ground-truth body poses, sampled at imu_hz, with small noise."""
        n = max(1, int(round((t1 - t0) * self.imu_hz)))
        xi_rel = lie.se3_log(lie.se3_inverse(T0) @ T1)
        rows = []
        g_world = np.array([0, 0, -9.81])
        for i in range(n):
            t = t0 + (i + 0.5) * (t1 - t0) / n
            w = xi_rel[3:] / max(t1 - t0, 1e-9)              # body angular velocity (approx, const over interval)
            R = T0[:3, :3]
            a_specific = R.T @ (-g_world)                     # stationary-ish specific force (gravity reaction)
            if self._add_noise:
                w = w + self.rng.normal(scale=0.01, size=3)
                a_specific = a_specific + self.rng.normal(scale=0.05, size=3)
            rows.append([t, *w, *a_specific])
        return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 7))

    def __iter__(self) -> Iterator[Frame]:
        t = 0.0
        for i, T_wb in enumerate(self.poses):
            rgb, depth = self.renderer.render(
                T_wb, T_BODY_CAM,
                rgb_noise_std=self.rgb_noise_std,
                depth_noise_model=stereo_depth_noise if self._add_noise else None,
                rng=self.rng,
            )
            imu = None
            if i > 0:
                imu = self._imu_between(self.poses[i - 1], T_wb, t - self.dt, t)
            yield Frame(t=t, rgb=rgb, depth=depth, intr=self.intr, imu=imu,
                        frame_id=i, gt_pose=T_wb.copy())
            t += self.dt

    def close(self) -> None:
        pass
