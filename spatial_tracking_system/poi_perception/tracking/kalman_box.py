"""
A small, dependency-free constant-velocity Kalman filter over
[cx, cy, w, h] bounding-box state, used per-track inside ByteTrack.

Implemented manually (no filterpy) to keep M3's dependency surface small
and because this is genuinely the sum total of what's needed: an 8-dim
constant-velocity linear filter with a linear (identity-ish) observation
model. State: [cx, cy, w, h, vcx, vcy, vw, vh].
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class KalmanBoxTracker:
    def __init__(self, bbox_xyxy: Tuple[float, float, float, float]):
        x1, y1, x2, y2 = bbox_xyxy
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)

        self.dim_x = 8
        self.dim_z = 4

        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # constant velocity: position += velocity

        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(4):
            self.H[i, i] = 1.0

        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        # Moderate process noise on velocities (people accelerate walking,
        # not sprinting), tight on position (the box itself is directly
        # observed every frame it's detected).
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 2.0, 2.0])
        self.R = np.diag([4.0, 4.0, 4.0, 4.0])  # measurement noise, pixels^2
        self.P = np.eye(self.dim_x) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # box can't go negative-size; clamp softly after predict
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.bbox_xyxy()

    def update(self, bbox_xyxy: Tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
