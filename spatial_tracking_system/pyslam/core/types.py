"""
Frozen contracts. Do not change these shapes casually -- every module
in the pipeline is written against them. If a later phase needs a
different shape, that is a design discussion, not a quick edit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Iterator, Protocol
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float  # metres per raw depth unit
    baseline: float = 0.05  # metres, for depth-noise modelling

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]], dtype=np.float64)


@dataclass
class Frame:
    """One synchronized sensor bundle."""
    t: float                      # device time, seconds, monotonic
    rgb: np.ndarray                # HxWx3 uint8
    depth: np.ndarray              # HxW uint16, raw units (metres = depth * depth_scale)
    intr: Intrinsics
    imu: Optional[np.ndarray] = None   # (N,7): t, gx,gy,gz, ax,ay,az since previous frame
    frame_id: int = -1
    gt_pose: Optional[np.ndarray] = None  # 4x4, world<-body, synthetic ground truth only


@dataclass
class Signature:
    """A node's appearance + associated 3D geometry, in the *camera* frame."""
    id: int
    t: float
    kp: np.ndarray          # (N,2) float32 pixel coords
    kp3d: np.ndarray        # (N,3) float32, camera frame, NaN row if invalid
    desc: np.ndarray        # (N,32) uint8 for ORB
    valid: np.ndarray       # (N,) bool, True where kp3d is usable
    word_ids: Optional[np.ndarray] = None  # (N,) int32, BoW vocabulary assignment
    global_desc: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None    # kept while resident in WM; dropped in LTM (future phase)
    depth: Optional[np.ndarray] = None


@dataclass
class Node:
    id: int
    sig: Signature
    pose_odom: np.ndarray     # 4x4 world<-body at creation time. IMMUTABLE after construction.
    pose_map: np.ndarray      # 4x4, mutated by the graph optimiser
    weight: int = 1
    session_id: int = 0


@dataclass
class Link:
    a: int
    b: int
    T_ab: np.ndarray          # 4x4, maps points in b's frame into a's frame
    info: np.ndarray          # 6x6 information matrix, order [rho(3), phi(3)]
    kind: str                 # 'odom' | 'loop' | 'proximity' | 'prior'
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    residual_rms: float = 0.0


@dataclass
class OdomResult:
    pose: np.ndarray          # 4x4 world<-body, current estimate
    T_rel: np.ndarray         # 4x4, relative motion since previous keyframe reference
    info: np.ndarray          # 6x6
    n_inliers: int
    status: str                # 'OK' | 'LOST'


@dataclass
class Hypothesis:
    node_id: int
    posterior: float
    n_consecutive: int


class SensorSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...
    def intrinsics(self) -> Intrinsics: ...
    def close(self) -> None: ...
