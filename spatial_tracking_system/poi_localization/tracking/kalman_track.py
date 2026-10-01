"""
Section 6: "each tracked person is one constant-velocity Kalman filter
in the floor plane. State: [x, y, vx, vy]... No Z in the state."

Each available measurement this frame is applied as its own independent
2D position update (Section 6: "apply both as two sequential updates" if
both depth and raycast are available) -- so this class exposes predict()
and update() separately rather than a single combined-measurement step.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from poi_localization import geometry


class KalmanTrack:
    def __init__(self, x0: float, y0: float, process_accel_noise: float, initial_pos_std: float = 0.5, initial_vel_std: float = 2.0):
        self.dim_x = 4
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([initial_pos_std**2, initial_pos_std**2, initial_vel_std**2, initial_vel_std**2])
        self.process_accel_noise = process_accel_noise  # m/s^2 -- Section 6: "1-2 m/s^2" starting point
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

    def _process_noise(self, dt: float) -> np.ndarray:
        """Standard constant-velocity / white-noise-acceleration process
        noise model, discretized over dt, for each of the (independent)
        x and y axes."""
        q = self.process_accel_noise**2
        Qc = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Qc
        Q[np.ix_([1, 3], [1, 3])] = Qc
        return Q

    def predict(self, dt: float) -> None:
        dt = max(dt, 1e-6)
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)

    def predicted_position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def position_cov(self) -> np.ndarray:
        """2x2 covariance of just the position sub-state (top-left block of P)."""
        return self.P[:2, :2]

    def mahalanobis_distance_sq(self, z: np.ndarray, R: np.ndarray) -> float:
        """Chi-square gate input, computed against the CURRENT (pre-
        update) state -- used by the track manager to decide whether a
        measurement should be applied at all, before update() commits it."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        return geometry.mahalanobis_sq(y, S)

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_x) - K @ self.H
        # Joseph-form covariance update -- numerically more robust than
        # the textbook (I-KH)P form under repeated sequential updates
        # (Section 6 applies two updates per frame when both measurements
        # are available), since it stays symmetric/PSD even with small
        # numerical error accumulating over a long track.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
