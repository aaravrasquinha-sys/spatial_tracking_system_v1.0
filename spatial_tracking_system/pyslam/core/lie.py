"""
SO(3)/SE(3) utilities.

Conventions (frozen, do not change):
  - Rotation storage at rest: 3x3 float64 matrix, or unit quaternion [w,x,y,z].
  - Pose storage: 4x4 float64 homogeneous matrix.
  - T_a_b maps points from frame b into frame a:  p_a = T_a_b @ p_b_h
  - Camera optical convention: x right, y down, z forward.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-10


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y],
                      [z, 0.0, -x],
                      [-y, x, 0.0]], dtype=np.float64)


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: so(3) vector -> SO(3) matrix."""
    theta = np.linalg.norm(w)
    if theta < _EPS:
        return np.eye(3) + skew(w)
    k = w / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) matrix -> so(3) vector.

    Implemented via quaternion + atan2 rather than a direct trace/sin(theta)
    branch: the direct formula is ill-conditioned near theta=pi (division by
    a near-zero sin(theta)), while Shepperd's method (used in rot_to_quat)
    plus atan2(|v|, w) stays well-conditioned across the full range,
    including theta close to pi.
    """
    q = rot_to_quat(R)  # [w,x,y,z], w >= 0 (hemisphere-canonicalised)
    w, v = q[0], q[1:]
    vnorm = np.linalg.norm(v)
    theta = 2.0 * np.arctan2(vnorm, w)
    if vnorm < _EPS:
        # theta ~ 0: axis undefined, but so is the rotation -- return ~0
        return v * 2.0  # first-order: phi ~ 2*v for small angles
    axis = v / vnorm
    return axis * theta


def so3_left_jacobian(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    K = skew(w)
    if theta < _EPS:
        return np.eye(3) + 0.5 * K
    a = np.sin(theta) / theta
    b = (1 - np.cos(theta)) / (theta ** 2)
    c = (1 - a) / (theta ** 2)
    return np.eye(3) + b * K + c * (K @ K)


def so3_left_jacobian_inv(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    K = skew(w)
    if theta < _EPS:
        return np.eye(3) - 0.5 * K
    half = theta / 2.0
    cot = np.cos(half) / np.sin(half)
    coeff = (1.0 / (theta ** 2)) * (1.0 - (half * cot))
    return np.eye(3) - 0.5 * K + coeff * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """xi = [rho(3), phi(3)] in R^6 -> SE(3) 4x4 matrix."""
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi)
    J = so3_left_jacobian(phi)
    t = J @ rho
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    phi = so3_log(R)
    Jinv = so3_left_jacobian_inv(phi)
    rho = Jinv @ t
    return np.concatenate([rho, phi])


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Adj = np.zeros((6, 6))
    Adj[:3, :3] = R
    Adj[:3, 3:] = skew(t) @ R
    Adj[3:, 3:] = R
    return Adj


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """q = [w,x,y,z], assumed normalised."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,3) -> (N,3), applies T_a_b to points in b."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]
