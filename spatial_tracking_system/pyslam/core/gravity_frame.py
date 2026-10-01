"""
WP-T3: build a gravity-aligned world frame (W_grav) for TRAJECTORY EXPORT
-- bird's-eye view and height plots -- as a one-shot, offline transform
applied to already-optimised poses. This is deliberately separate from
pyslam.imu.gravity's gravity_prior_enabled machinery (WP-P5.2), which
feeds a tilt-only correction INTO the graph optimisation itself: that
requires gravity_prior_enabled=True and a quasi-static window at every
keyframe, and is still opt-in/off by default. This module needs neither
-- it only needs ONE quasi-static window near the start of the run
(PipelineResult.startup_imu_window, collected unconditionally by
pipeline.py's _run_loop for cfg.gravity_align_hold_s seconds) and never
touches the graph, so a run with gravity_prior_enabled=False (today's
default) still gets a sensible W_grav for its plots.

Frame definitions (see the graph's own frozen convention in
pyslam.core.lie: T_a_b maps points in b's frame into a's frame; camera
optical convention x-right/y-down/z-forward):

  W_cam0:  the graph's native world frame. Origin and orientation of the
           FIRST keyframe's camera-optical frame. This is what pose_map
           already is for every node -- no transform needed to get here.

  W_grav:  a gravity-aligned frame with the SAME ORIGIN as W_cam0 (the
           first keyframe's camera centre) but re-oriented so:
             z = up            (opposite the gravity direction)
             x = the first keyframe's own forward-looking direction,
                 projected onto the horizontal plane (so "forward" at
                 t=0 defines "bird's-eye view +x", not an arbitrary
                 world axis)
             y = z cross x     (right-handed)
           A node's pose in W_grav is T_grav_cam0 @ node.pose_map.

Reuses pyslam.imu.gravity.gravity_direction_body/is_quasi_static rather
than re-deriving the accelerometer sign convention -- see that module's
own docstring for why "specific force ~= -gravity when static" is the
correct physical relationship, and pipeline.py's _try_gravity_prior for
why the R_body_cam rotation (camera-frame IMU-measurement convention,
NOT identity by default on this project's own D435i rig -- see
Config/Pipeline docstrings) must be applied before this is usable.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core import lie
from pyslam.imu.gravity import gravity_direction_body, is_quasi_static


class GravityAlignmentResult:
    def __init__(self, T_grav_cam0: np.ndarray, aligned: bool, reason: str,
                 up_cam0: Optional[np.ndarray] = None):
        self.T_grav_cam0 = T_grav_cam0
        self.aligned = aligned    # False => T_grav_cam0 is identity (no usable IMU window)
        self.reason = reason      # human-readable, goes straight into trajectory_report.json
        self.up_cam0 = up_cam0    # unit "up" direction in W_cam0, for diagnostics/plots


def estimate_gravity_alignment(startup_imu_window: list[np.ndarray], R_body_cam: np.ndarray,
                                gyro_thresh_rad_s: float = 0.08,
                                accel_std_thresh_mps2: float = 0.5,
                                cam0_forward_cam: np.ndarray = np.array([0.0, 0.0, 1.0]),
                                ) -> GravityAlignmentResult:
    """startup_imu_window: PipelineResult.startup_imu_window (list of
    (Ni,7) arrays, [t,gx,gy,gz,ax,ay,az]); concatenated and checked for
    quasi-staticness with the SAME thresholds gravity_prior uses, so
    "was this actually a still hold" is judged consistently across both
    mechanisms. cam0_forward_cam: the camera's own forward axis in its
    own (cam0) frame -- [0,0,1] by the frozen optical convention, i.e.
    "the direction the camera was pointed at t=0", used to seed W_grav's
    +x axis. Returns identity (aligned=False) rather than raising if no
    usable window is found -- a run without IMU, or without a still
    start, should still export SOMETHING (un-aligned W_cam0 plots), not
    crash the exporter. Callers MUST check .aligned and surface .reason
    to the user (trajectory_report.json does this) rather than silently
    presenting an un-aligned "bird's-eye view" as if it were gravity-true.
    """
    if not startup_imu_window:
        return GravityAlignmentResult(np.eye(4), False, "no IMU data collected at startup")
    imu = np.concatenate(startup_imu_window, axis=0)
    if imu.shape[0] < 3:
        return GravityAlignmentResult(np.eye(4), False, "fewer than 3 IMU samples at startup")
    if not is_quasi_static(imu, gyro_thresh_rad_s, accel_std_thresh_mps2):
        return GravityAlignmentResult(
            np.eye(4), False,
            "startup IMU window is not quasi-static (camera was moving/rotating at t=0) -- "
            "a still hold at the start of the run is required for gravity alignment")
    try:
        g_body = gravity_direction_body(imu)  # unit vector, DOWN (direction gravity points)
    except ValueError as e:
        return GravityAlignmentResult(np.eye(4), False, f"degenerate accelerometer reading: {e}")

    # WP-T3 bug found by this module's own oracle test
    # (test_g_traj.py::check_gravity_alignment_oracle): gravity_direction_body
    # returns the direction gravity POINTS (down), per its own explicit
    # docstring ("g_body = -f_body = -mean(accel)") -- an early version of
    # this function used that value directly as "up_cam0", which is
    # backwards by exactly 180 degrees. pipeline.py's own self._g_world
    # (WP-P5.2) happens to be self-consistent regardless of this sign
    # (it only ever compares u_world against another u computed the SAME
    # way, via tilt_correct_rotation -- see gravity.py's
    # build_gravity_prior_link), which is why that code path never
    # surfaced this. This module needs the correct physical UP direction
    # specifically (it builds z_grav = up), so the negation below is
    # required, not optional.
    up_body = -g_body
    up_cam0 = R_body_cam.T @ up_body
    up_cam0 = up_cam0 / max(np.linalg.norm(up_cam0), 1e-12)

    fwd = np.asarray(cam0_forward_cam, dtype=np.float64)
    fwd = fwd - np.dot(fwd, up_cam0) * up_cam0   # project onto the horizontal plane
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-3:
        # Degenerate: the camera was pointed almost straight up/down at
        # t=0, so its own forward axis carries no horizontal information.
        # Fall back to the camera's "right" axis instead (still just a
        # convention choice for where W_grav's +x points -- it does not
        # affect z/up correctness, only the arbitrary in-plane rotation).
        fallback = np.array([1.0, 0.0, 0.0])
        fwd = fallback - np.dot(fallback, up_cam0) * up_cam0
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm < 1e-3:
            fallback = np.array([0.0, 1.0, 0.0])
            fwd = fallback - np.dot(fallback, up_cam0) * up_cam0
            fwd_norm = np.linalg.norm(fwd)
    x_grav = fwd / fwd_norm
    z_grav = up_cam0
    y_grav = np.cross(z_grav, x_grav)   # right-handed: z cross x = y
    y_grav = y_grav / max(np.linalg.norm(y_grav), 1e-12)
    # re-orthogonalise x against y,z (already unit/orthogonal by construction,
    # this just guards floating-point drift)
    x_grav = np.cross(y_grav, z_grav)

    # Columns of R_cam0_world are the WORLD axes expressed in cam0
    # coordinates -- i.e. R_cam0_world @ [1,0,0]^T = x_grav (in cam0
    # coords), etc. That means R_cam0_world maps a WORLD-frame vector
    # into cam0 coordinates. We want the inverse direction (T_ab
    # convention: T_grav_cam0 maps a CAM0-frame point into W_grav), which
    # for an orthonormal matrix is just the transpose.
    R_cam0_world = np.stack([x_grav, y_grav, z_grav], axis=1)
    R_grav_cam0 = R_cam0_world.T
    T_grav_cam0 = lie.make_T(R_grav_cam0, np.zeros(3))
    return GravityAlignmentResult(T_grav_cam0, True, "aligned from startup static-hold IMU window",
                                   up_cam0=up_cam0)


def pose_to_grav(T_cam0_x: np.ndarray, T_grav_cam0: np.ndarray) -> np.ndarray:
    """Convenience: a node's pose_map (world_cam0<-node) into W_grav."""
    return T_grav_cam0 @ T_cam0_x
