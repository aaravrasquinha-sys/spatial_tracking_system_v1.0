"""
Section 2's core structural point: Phase A (local floor frame) and
Phase B (room frame) are "both... defined identically as a floor-
referenced right-handed frame with the camera at some known pose within
it." This dataclass IS that shared shape -- everything downstream of
frame_provider.py only ever sees a FloorFrameTransform and never needs
to know which phase produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from poi_localization import geometry


@dataclass
class FloorFrameTransform:
    R: np.ndarray  # 3x3: p_floor = R @ p_cam + t
    t: np.ndarray  # 3,  camera's own position in floor space
    frame_name: str  # "local" (Phase A) | "room" (Phase B)
    map_id: Optional[str] = None  # Phase B only -- for the M6-style map_id mismatch check
    # How much this transform itself is trusted -- NOT sensor noise, but
    # extrinsic/calibration error (Phase A: how good the IMU+depth fit
    # was; Phase B: M2's own reported sigma). Zero by default (exact),
    # matching prior behavior, but every real transform-producing path
    # (local_floor_frame.py, room_frame.py) fills these in; measurement
    # code adds their contribution via geometry.extrinsic_position_variance
    # rather than assuming the frame is exact. See that function's
    # docstring for why this matters and what the model does and doesn't
    # capture.
    sigma_rot_rad: float = 0.0
    sigma_trans_m: float = 0.0

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @property
    def camera_height_m(self) -> float:
        """The camera's own Z in floor space -- its height above the floor."""
        return float(self.t[2])

    def apply_point(self, p_cam: np.ndarray) -> np.ndarray:
        return geometry.transform_point(self.R, self.t, np.asarray(p_cam, dtype=np.float64))

    def apply_direction(self, d_cam: np.ndarray) -> np.ndarray:
        return geometry.transform_direction(self.R, np.asarray(d_cam, dtype=np.float64))

    def camera_origin_floor(self) -> np.ndarray:
        return self.t.copy()

    @classmethod
    def from_height_and_yaw(
        cls,
        height_m: float,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        frame_name: str = "local",
        sigma_rot_rad: float = 0.0,
        sigma_trans_m: float = 0.0,
    ) -> "FloorFrameTransform":
        """Build a canonical transform for a camera at a known height,
        level (or tilted down by pitch_deg), facing floor-frame +X --
        used for synthetic/test/demo setups (Section 2, Phase A note:
        "a real, physically grounded... frame" doesn't require this to
        come from hardware every time; see config M4Config.frame.fixed_transform).

        pitch_deg > 0 means tilted downward (looking more at the floor).
        """
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)

        # Camera-space axes expressed in floor space: camera forward
        # (+Z_cam) points along floor +X, tilted down by pitch. At
        # yaw=pitch=0 (level, unrolled, facing +X_floor), "camera down"
        # (+Y_cam) must point toward -Z_floor (physically toward the
        # floor) -- right_floor's sign below is chosen so that holds;
        # verified numerically in tests/test_frames.py.
        forward_floor = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
        right_floor = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        down_floor = np.cross(forward_floor, right_floor)  # always orthonormal: right x down = forward

        # R maps p_cam -> p_floor. Columns of R are where the camera's
        # own basis vectors (X_cam=right, Y_cam=down, Z_cam=forward) land
        # in floor space.
        R = np.stack([right_floor, down_floor, forward_floor], axis=1)
        t = np.array([0.0, 0.0, height_m])
        return cls(R=R, t=t, frame_name=frame_name, sigma_rot_rad=sigma_rot_rad, sigma_trans_m=sigma_trans_m)
