"""
WP-B2: information matrix for a PnP-estimated relative pose, derived from
the PnP reprojection Hessian instead of the heuristic
`eye(6) * inlier_ratio * 100` used throughout odometry.py/odometry_f2m.py.

Convention: `solvePnPRansac(objectPoints, imagePoints, ...)` returns
(R, t) mapping object-frame points into the camera that observed
imagePoints: p_cam = R @ p_obj + t, i.e. T_cam_obj (this is the SAME raw
PnP convention both odometry.py and odometry_f2m.py already use -- see
their own convention-warning comments). What both trackers actually need
an information matrix FOR is not T_cam_obj itself but its inverse,
T_obj_cam (odometry.py's T_ref_cur; odometry_f2m.py's T_world_cam) --
the pose actually used as the emitted Link's relative transform, in that
pose's own RIGHT-tangent perturbation convention (T_obj_cam @ exp(xi)),
matching how backend_native.py parameterises every pose in the graph.

Derivation (validated against a Monte Carlo oracle before being trusted
-- see test_pnp_info_matrix_oracle in tests/gates/test_g0.py and
WP_B2_Findings.md, mirroring this engagement's now-established pattern
of oracle-first validation for every piece of new numerical code):

For a 3D point X (object frame) with camera-frame image p_cam = R@X+t,
a RIGHT perturbation of T_cam_obj (T_cam_obj @ exp(xi), xi=[rho,phi])
gives, to first order:
    d(p_cam)/d(rho) = R
    d(p_cam)/d(phi) = -R @ skew(X)
(same identity used in local_ba.py's residual Jacobian -- see that
module for the algebra). Composing with the pinhole projection Jacobian
d(pi)/d(p_cam) gives each point's 2x6 Jacobian block; stacking over all
inliers and forming J^T @ J gives the Gauss-Newton Hessian for T_cam_obj's
OWN right-tangent perturbation. Under i.i.d. Gaussian pixel noise with
std sigma_px, Cov(xi) ~= sigma_px^2 * H^-1 (standard Gauss-Newton/Laplace
approximation), so Info(xi) = H / sigma_px^2.

This describes uncertainty in T_cam_obj's tangent space, not T_obj_cam's.
Using inverse(T @ exp(xi)) = exp(-xi) @ inverse(T), and the standard
adjoint identity T^-1 @ exp(v) @ T = exp(Adj(T^-1) @ v), a right-
perturbation of T_cam_obj with tangent covariance Cov(xi) induces a
right-perturbation of T_obj_cam = inverse(T_cam_obj) with tangent vector
xi' = -Adj(T_cam_obj) @ xi (derived by setting T := T_obj_cam in the
identity above and solving for xi'; NOTE an earlier version of this
derivation got the adjoint's placement wrong -- used Adj(T_obj_cam) @ H
@ Adj(T_obj_cam)^T, caught by the Monte Carlo oracle below giving
NEES~=24 instead of the expected ~6, not by inspection. Every one of 4
plausible sign/ordering/which-pose variants was tested numerically
against empirical samples rather than re-derived by hand a third time --
see WP_B2_Findings.md for the comparison table):
    Info(T_obj_cam right-tangent) = Adj(T_obj_cam)^T @ H @ Adj(T_obj_cam) / sigma_px^2

This is directly usable as `Link.info` for BOTH trackers with no further
transform: odometry.py's T_rel IS T_ref_cur = T_obj_cam directly (obj =
reference keyframe's own camera frame); odometry_f2m.py's link is
T_rel_since_kf = inverse(prev_kf_pose) @ cur_pose, and since prev_kf_pose
is treated as a fixed (already-certain, from the previous cycle) LEFT
constant in that composition, a right-perturbation of cur_pose (=
T_world_cam = T_obj_cam here, with 'obj' being the tracker's own
persistent world frame) induces the IDENTICAL right-perturbation of
T_rel_since_kf with no Adjoint needed (left-multiplying by a constant
doesn't change the tangent-space perturbation).

sigma_px is a single free scale parameter, calibrated via NEES on
static_60s (see calibrate_sigma_px below and WP_B2_Findings.md) rather
than guessed.
"""
from __future__ import annotations
import numpy as np

from pyslam.core import lie


def pnp_info_matrix(obj_pts: np.ndarray, R_cam_obj: np.ndarray, t_cam_obj: np.ndarray,
                     K: np.ndarray, sigma_px: float) -> np.ndarray:
    """obj_pts: (N,3) inlier points in the object/reference frame.
    R_cam_obj, t_cam_obj: PnP's raw output, p_cam = R@p_obj + t.
    K: 3x3 intrinsics. sigma_px: assumed per-axis pixel noise std.
    Returns a 6x6 information matrix for T_obj_cam = inverse(T_cam_obj)'s
    own right-tangent perturbation, [rho(3), phi(3)] order. See module
    docstring for the full derivation.
    """
    H = _pnp_hessian(obj_pts, R_cam_obj, t_cam_obj, K)
    T_cam_obj = lie.make_T(R_cam_obj, t_cam_obj)
    T_obj_cam = lie.se3_inverse(T_cam_obj)
    Adj = lie.se3_adjoint(T_obj_cam)
    info = (Adj.T @ H @ Adj) / (sigma_px ** 2)
    return info


def pnp_info_matrix_direct_frame(obj_pts: np.ndarray, R_cam_obj: np.ndarray,
                                  t_cam_obj: np.ndarray, K: np.ndarray,
                                  sigma_px: float) -> np.ndarray:
    """WP-P4: the other half of pnp_info_matrix's Adjoint transform.

    verify.py's bidirectional check calls _solve_pnp() in BOTH
    directions and keeps whichever gave more inliers. For direction 1
    (a's 3D vs b's image), the raw PnP output's INVERSE is the emitted
    Link.T_ab -- pnp_info_matrix (above) is exactly right for that case.
    For direction 2 (b's 3D vs a's image), the raw PnP output IS
    Link.T_ab directly, with no inversion -- using pnp_info_matrix there
    would silently apply an Adjoint transform for a pose that was never
    inverted, exactly the class of bug WP-B2's own Monte Carlo oracle
    was built to catch (see that module's docstring: "an earlier version
    of this derivation got the adjoint's placement wrong... caught by
    the Monte Carlo oracle, not by inspection"). This function returns
    info for T_cam_obj's OWN right-tangent perturbation -- i.e. the raw
    Hessian, scaled, with no Adjoint step at all.

    Both functions are oracle-tested together in
    tests/gates/test_g4.py::check_pnp_info_direction_oracle so this
    distinction is checked, not just asserted in a docstring.
    """
    H = _pnp_hessian(obj_pts, R_cam_obj, t_cam_obj, K)
    return H / (sigma_px ** 2)


def _pnp_hessian(obj_pts: np.ndarray, R_cam_obj: np.ndarray, t_cam_obj: np.ndarray,
                  K: np.ndarray) -> np.ndarray:
    """Gauss-Newton Hessian J^T@J for T_cam_obj's own right-tangent
    perturbation, shared by both pnp_info_matrix and
    pnp_info_matrix_direct_frame -- see their docstrings for which pose
    each one's final answer describes."""
    n = obj_pts.shape[0]
    if n == 0:
        return np.eye(6) * 1e-9  # degenerate: no information at all
    p_cam = (R_cam_obj @ obj_pts.T).T + t_cam_obj  # (N,3)
    z = np.clip(p_cam[:, 2], 1e-6, None)
    fx, fy = K[0, 0], K[1, 1]

    # d(pi)/d(p_cam), one 2x3 block per point, vectorised.
    dpi_dpcam = np.zeros((n, 2, 3))
    dpi_dpcam[:, 0, 0] = fx / z
    dpi_dpcam[:, 0, 2] = -fx * p_cam[:, 0] / z ** 2
    dpi_dpcam[:, 1, 1] = fy / z
    dpi_dpcam[:, 1, 2] = -fy * p_cam[:, 1] / z ** 2

    # d(p_cam)/d(rho) = R (same for every point); d(p_cam)/d(phi) = -R @ skew(X_i).
    dpcam_drho = np.broadcast_to(R_cam_obj, (n, 3, 3))
    skews = np.zeros((n, 3, 3))
    skews[:, 0, 1] = -obj_pts[:, 2]; skews[:, 0, 2] = obj_pts[:, 1]
    skews[:, 1, 0] = obj_pts[:, 2];  skews[:, 1, 2] = -obj_pts[:, 0]
    skews[:, 2, 0] = -obj_pts[:, 1]; skews[:, 2, 1] = obj_pts[:, 0]
    dpcam_dphi = -np.einsum('ij,njk->nik', R_cam_obj, skews)

    dpcam_dxi = np.concatenate([dpcam_drho, dpcam_dphi], axis=2)  # (N,3,6)
    J = np.einsum('nij,njk->nik', dpi_dpcam, dpcam_dxi)  # (N,2,6)
    J = J.reshape(-1, 6)  # (2N,6)
    return J.T @ J


def calibrate_sigma_px(nees_at_sigma1: list[float], dof: int = 6) -> float:
    """NEES scales as 1/sigma_px^2 (since Info scales that way), so given
    NEES values computed with sigma_px=1 (i.e. e^T @ H_unit @ e for the
    unscaled Hessian-only info matrix), the correctly-calibrated sigma_px
    has a closed form: mean(NEES_at_sigma1) / dof should equal 1/sigma_px^2
    at calibration (since a properly-calibrated info matrix gives
    E[NEES] = dof, the chi-square(dof) mean). No grid search needed.
    """
    mean_nees_1 = float(np.mean(nees_at_sigma1))
    if mean_nees_1 <= 0:
        return 1.0
    return float(np.sqrt(mean_nees_1 / dof))
