"""
WP-P5.2: loose coupling -- a gravity-aligned roll/pitch (tilt-only)
prior, built as `Link(kind="prior", ...)` between the graph's gauge
node (node 0, orientation-fixed at identity -- see pipeline.py's own
node-0 initialisation) and any later node. Deliberately the SIMPLER
half of Phase 5's coupling spectrum (architecture doc section 5, P5):
no velocity/bias states, no new GraphBackend machinery -- `Link`'s
existing 6x6-info-matrix contract already supports an ANISOTROPIC
prior (strong on tilt, ~zero on yaw and position) with zero changes to
`backend_native.py` (confirmed: only `kind="loop"` gets robust-kernel
treatment there; "prior" passes through the same as "odom").

Why tilt-only, not a full 6-DoF prior: gravity direction alone cannot
observe yaw (rotation about the vertical axis) or position -- an
accelerometer reading is fundamentally blind to both. Encoding this
honestly (near-zero info on those components, not just "small") is
what makes this a correct partial constraint rather than a
mis-specified full one. See `check_tilt_correction_oracle` in
tests/gates/test_g5.py for why get this right matters: an anisotropic
info matrix built in the wrong basis, or a yaw-corrupting correction,
would silently bias the graph's YAW estimate using gravity, which is
information it fundamentally does not have.

QUASI-STATIC ASSUMPTION: specific force approximates -gravity (in body
frame) only when linear acceleration is small relative to g -- i.e.
while the camera is nearly stationary or at ~constant velocity. This
module does NOT distinguish "genuinely static" from "moving with
canceling real acceleration" (a rare but real false-quasi-static case);
it only checks gyro rate, the same first-order heuristic real VIO
initialisation routines use. `is_quasi_static`'s threshold is
deliberately conservative (see its own docstring) -- the cost of
missing a genuinely-static interval is one fewer prior link, cheap;
the cost of a false positive is a WRONG prior pulling the graph away
from a correct vision estimate, expensive. No prior beats a wrong one,
same philosophy `local_ba.py`'s conservative landmark-freeze already
established for this codebase.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core import lie
from pyslam.core.types import Link


def is_quasi_static(imu: np.ndarray, gyro_thresh_rad_s: float = 0.08,
                     accel_std_thresh_mps2: float = 0.5) -> bool:
    """Conservative heuristic: every gyro sample's magnitude must stay
    under `gyro_thresh_rad_s` (~4.6deg/s -- comfortably above ordinary
    sensor noise, comfortably below any deliberate motion) AND the
    accelerometer readings' own sample std must stay under
    `accel_std_thresh_mps2` (catches translational jostling that a low
    gyro rate alone wouldn't -- e.g. a camera sliding smoothly, no
    rotation, but still accelerating). Both conditions, not either --
    see module docstring on why a false positive here is the expensive
    failure mode."""
    if imu is None or imu.shape[0] < 3:
        return False
    gyro = imu[:, 1:4]
    accel = imu[:, 4:7]
    if np.max(np.linalg.norm(gyro, axis=1)) > gyro_thresh_rad_s:
        return False
    if np.max(np.std(accel, axis=0)) > accel_std_thresh_mps2:
        return False
    return True


def gravity_direction_body(imu: np.ndarray) -> np.ndarray:
    """Unit vector: the direction gravity points, AS OBSERVED IN THE
    BODY FRAME, from a quasi-static accelerometer window. specific
    force f_body = R^T(a_world - g_world); a_world~=0 when static, so
    f_body ~= -R^T g_world = -g_body, i.e. g_body = -f_body =
    -mean(accel). Caller is responsible for checking `is_quasi_static`
    first -- this function does not check it itself and will silently
    return a wrong answer if fed a dynamic window."""
    f_body = np.mean(imu[:, 4:7], axis=0)
    g_body = -f_body
    n = np.linalg.norm(g_body)
    if n < 1e-6:
        raise ValueError("degenerate (near-zero) accelerometer reading -- cannot determine gravity direction")
    return g_body / n


def rotation_aligning(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Minimal rotation R such that R @ p = q, for unit vectors p, q.
    Standard Rodrigues construction from the rotation axis (p x q) and
    angle (via atan2 of the cross/dot norms, robust near both 0 and
    pi -- the same atan2-based-not-arccos-based conditioning choice
    `lie.so3_log` already uses elsewhere in this codebase, for the same
    reason)."""
    axis = np.cross(p, q)
    s = np.linalg.norm(axis)
    c = float(np.dot(p, q))
    angle = np.arctan2(s, c)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)  # already aligned
        # exactly anti-parallel: any perpendicular axis works
        perp = np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(p, perp)
        axis = axis / np.linalg.norm(axis)
        return lie.so3_exp(axis * np.pi)
    return lie.so3_exp((axis / s) * angle)


def tilt_correct_rotation(R_vision: np.ndarray, u_world: np.ndarray, u_body_observed: np.ndarray) -> np.ndarray:
    """Returns R_corrected = R_vision @ dR such that
    R_corrected^T @ u_world == u_body_observed exactly, while
    disturbing R_vision's own YAW (rotation about u_world) as little as
    possible -- this is the tilt-only correction. Derivation: we need
    dR with dR^T @ (R_vision^T @ u_world) == u_body_observed; since dR
    is a pure rotation, dR^T == dR^-1, so dR = rotation_aligning(
    u_body_observed, R_vision^T @ u_world) satisfies this exactly (see
    rotation_aligning's own contract: dR @ u_body_observed ==
    R_vision^T @ u_world, therefore dR^T @ (R_vision^T @ u_world) ==
    u_body_observed). Applying dR on the RIGHT (body-frame side) rather
    than composing a world-frame correction on the left is what keeps
    the disturbance confined to tilt: a right-multiplication by a
    rotation whose own axis lies in the plane spanned by two
    near-parallel vectors (u_body_observed and R_vision's own predicted
    up direction) has no component about the world-vertical axis
    itself when the two input directions are close -- checked directly,
    not just asserted, by `check_tilt_correction_oracle`."""
    u_body_predicted = R_vision.T @ u_world
    dR = rotation_aligning(u_body_observed, u_body_predicted)
    return R_vision @ dR


def build_gravity_prior_link(node_a_id: int, node_b_id: int, T_a_world: np.ndarray,
                              T_b_vision: np.ndarray, u_world: np.ndarray,
                              imu_at_b: np.ndarray, sigma_tilt_rad: float,
                              gyro_thresh_rad_s: float = 0.08,
                              accel_std_thresh_mps2: float = 0.5,
                              R_body_cam: np.ndarray = None) -> Optional[Link]:
    """Returns a tilt-only `Link(kind="prior")` between node_a (the
    reference whose own orientation defines `u_world`'s coordinate
    frame -- pipeline.py always calls this with node_a = node 0, the
    gauge-fixed node, so the link's own tangent frame IS the graph's
    world frame, no extra rotation needed -- see module docstring) and
    node_b, or None if `imu_at_b` isn't quasi-static enough to trust
    (see `is_quasi_static`'s own docstring on why a missing prior is
    the safe failure mode here, not a wrong one).

    `R_body_cam`: rotation mapping a vector expressed in the CAMERA
    frame into the physical BODY frame the IMU actually measures in
    (i.e. `v_body = R_body_cam @ v_cam` -- same convention as
    `tests/synth/world.py::T_BODY_CAM`'s own rotation block). REQUIRED
    to be non-identity for this to be correct on this project's own
    synthetic fixtures and on the real D435i: `pose_map`/`pose_odom`
    are CAMERA-frame poses (confirmed via `wpb1_eval.py`'s own
    `node_gt[i] @ T_BODY_CAM` convention), while `Frame.imu` measures
    the physical BODY frame -- these are DIFFERENT coordinate systems
    related by this fixed rotation, not the same frame under a
    different name. Defaults to `None` (treated as identity, i.e.
    "assume camera and IMU share axes") ONLY because the frozen
    `Intrinsics`/`Config` contract has no field carrying a real
    extrinsic yet (see WP_P5_Findings.md's own note on this) --
    identity is a known-wrong default for THIS project's own D435i
    rig, not a safe assumption, and callers with a real R_body_cam
    (including every synthetic-fixture caller, which has
    `tests.synth.world.T_BODY_CAM` available) should always pass it
    explicitly. This gap was found (not designed in) via
    `gravity_init_square6dof`'s own end-to-end validation -- ignoring
    it produced a spurious ~7deg systematic tilt-correction magnitude
    on a fixture where visual odometry alone is already accurate to a
    fraction of a degree.
    """
    if not is_quasi_static(imu_at_b, gyro_thresh_rad_s, accel_std_thresh_mps2):
        return None
    if R_body_cam is None:
        R_body_cam = np.eye(3)
    u_body_observed_physical = gravity_direction_body(imu_at_b)
    # convert the PHYSICAL body-frame observation into the CAMERA
    # frame's own coordinate convention (the same one T_b_vision and
    # u_world are expressed in) before comparing them -- see this
    # function's own docstring.
    u_body_observed = R_body_cam.T @ u_body_observed_physical
    R_corrected = tilt_correct_rotation(T_b_vision[:3, :3], u_world, u_body_observed)

    T_ab_vision = lie.se3_inverse(T_a_world) @ T_b_vision
    T_ab_target = T_ab_vision.copy()
    T_ab_target[:3, :3] = lie.se3_inverse(T_a_world)[:3, :3] @ R_corrected

    u = u_world / np.linalg.norm(u_world)
    proj_perp = np.eye(3) - np.outer(u, u)
    w_tilt = 1.0 / max(sigma_tilt_rad, 1e-6) ** 2
    eps = 1e-6  # near-zero, not exactly zero -- see module docstring
        # on why (cholesky in backend_native.py needs positive
        # DEFINITE, not merely PSD)
    info = np.zeros((6, 6))
    info[:3, :3] = np.eye(3) * eps          # ~no position information
    info[3:, 3:] = w_tilt * proj_perp + eps * np.outer(u, u)  # tilt strong, yaw ~eps

    return Link(a=node_a_id, b=node_b_id, T_ab=T_ab_target, info=info, kind="prior")
