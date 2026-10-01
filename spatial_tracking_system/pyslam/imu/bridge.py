"""
WP-M: bridge-link replacement after LOST.

This is the item both WP-K and WP-L Findings independently flagged as the
biggest open lever, ahead of the IMU tilt prior:

  * WP_K_Findings.md section 2: the identity bridge asserts ZERO motion
    across a LOST gap. On corridor_v2 (seed 1, native backend) the 15
    bridge links account for essentially all of the run's 3.78m missing
    path length (3.84m of real, asserted-away motion) and inject a
    measured mean 7.7deg (max 13.1deg) of rotation error per LOST event.
  * WP_L_Findings.md section 7: "Bridge replacement after LOST is still
    the biggest open lever, ahead of the IMU tilt prior."

Phase 0's bridge (pipeline.py, still the DEFAULT -- cfg.bridge_mode=
"identity", unchanged behaviour) inserts Link(T_ab=eye(4),
info=eye(6)*bridge_link_info_scale) whenever force_new_keyframe() leaves
a fresh keyframe with no real odometry measurement against the previous
one. That is not "no information" -- it is a WRONG, confidently-
asserted-zero claim, only harmless because bridge_link_info_scale is
tiny. This module builds a BETTER estimate of the true node_a -> node_b
relative transform from two things that stay available even though
visual tracking failed:

  * ROTATION, from the gyroscope. Integrating angular velocity needs no
    visual features or depth and does not degrade just because PnP lost
    track -- the same guarantee real VIO systems lean on. Reuses
    `pyslam.imu.preintegration.preintegrate` UNCHANGED, zero bias
    assumed (this project has no bias ESTIMATE yet -- P5.4 is not
    started; a wrong assumed bias is a small, honestly-bounded error
    over a single bridge gap, not the multi-second drift a bias
    estimate would need to worry about -- see WP_M_Findings.md for the
    measured size on the fixtures this shipped with).
  * TRANSLATION, from constant-velocity extrapolation of the last
    ODOMETRY-MEASURED velocity (finite-differenced from the last two
    good `OdomResult.T_rel` samples BEFORE tracking was lost, in the
    pre-LOST reference keyframe's own frame -- see pipeline.py's
    `_last_two_ok_frames`) times the elapsed gap time. Deliberately NOT
    double-integrated accelerometer: accel bias and (pre-P5.4) the lack
    of a trustworthy gravity subtraction both grow with dt^2, and unlike
    the gyro there is no oracle-validated component in this codebase yet
    trustworthy enough to use blind. Constant velocity is honestly a
    WEAKER assumption during exactly the kind of motion that tends to
    cause LOST (fast/erratic) -- which is why its own info-matrix weight
    is deliberately WIDENED (weakened) with elapsed gap time rather than
    held fixed, unlike the rotation info -- see `build_bridge_link`.

Both halves fail open to the Phase-0 identity behaviour, per component,
independently: a missing/degenerate gyro window degrades ONLY the
rotation block to identity/weak, a missing pre-LOST velocity degrades
ONLY the translation block, and both missing reproduces Phase-0's bridge
bit-for-bit. A caller can never get back something less trustworthy than
the identity bridge already was -- see `check_full_fallback_matches_identity_bridge`
in tests/gates/test_gm.py.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core.types import Link
from pyslam.imu.preintegration import preintegrate


def gyro_bridge_rotation(imu_window: np.ndarray, R_body_cam: np.ndarray) -> Optional[np.ndarray]:
    """3x3 rotation R_ab, in the CAMERA convention Link.T_ab's own
    contract uses ("maps points in b's frame into a's frame"), from pure
    gyro integration over imu_window ((N,7) rows [t, gx,gy,gz, ax,ay,az]
    spanning (t_a, t_b] -- same layout/window convention as
    `Frame.imu`/`preintegrate`'s own contract; the accel columns are
    present only because that shape is frozen, `preintegrate` never
    reads them for delta_R). Returns None for an empty/too-short window
    -- the CALLER decides the fallback (see `build_bridge_link`), this
    function only ever answers "what does the gyro say", never guesses.

    Frame conversion: preintegrate's delta_R is R_body(t_a)_body(t_b)
    (see preintegration.py's own docstring -- same right-multiplication,
    "T_a_b" convention this whole codebase uses). R_body_cam is the
    FIXED rotation relating a vector's camera-frame representation to
    its body-frame representation (v_body = R_body_cam @ v_cam, same
    rigid mounting at both t_a and t_b). Conjugating delta_R by it
    converts the body-frame relative rotation into the same convention
    in the camera frame: R_ab_cam = R_body_cam^T @ delta_R_body @
    R_body_cam (derivation in WP_M_Findings.md section 2 -- the same
    conjugate-transform relation `pyslam/imu/gravity.py` already uses
    for the tilt prior, applied here to a relative rotation instead of a
    single direction vector).
    """
    if imu_window is None or imu_window.shape[0] == 0:
        return None
    pre = preintegrate(imu_window, bg=np.zeros(3), ba=np.zeros(3), compute_jacobians=False)
    if pre.n_samples == 0 or pre.dt <= 0.0:
        return None
    delta_R_body = pre.delta_R
    return R_body_cam.T @ delta_R_body @ R_body_cam


def constant_velocity_translation(velocity_a_frame: Optional[np.ndarray],
                                   dt: Optional[float]) -> Optional[np.ndarray]:
    """Straight-line extrapolation p_b_in_a = velocity_a_frame * dt.
    velocity_a_frame: (3,) m/s in node_a's own camera frame, or None if
    no pre-LOST velocity estimate was ever established (e.g. LOST on
    the very first tracked frame after a keyframe, before two good
    samples existed). dt<=0/None (bad clock, zero-length gap) also
    degrades to None -- multiplying a real velocity by a nonsense dt
    would be worse than admitting no estimate."""
    if velocity_a_frame is None or dt is None or dt <= 0.0:
        return None
    return velocity_a_frame * dt


def build_bridge_link(node_a_id: int, node_b_id: int,
                       imu_window: np.ndarray, R_body_cam: np.ndarray,
                       velocity_a_frame: Optional[np.ndarray], dt: Optional[float],
                       min_gyro_samples: int, identity_info_scale: float,
                       rotation_sigma_rad: float,
                       velocity_sigma_base_mps: float,
                       velocity_sigma_growth_mps_per_s: float) -> Link:
    """Always returns a Link (never None) -- pipeline.py needs SOME
    bridge whenever a LOST recovery happens; an unanchored node is what
    crashed GTSAM on corridor_v2's hardware run in the first place (see
    Config.bridge_link_info_scale's own docstring). Degrades PER
    COMPONENT rather than all-or-nothing: a trustworthy gyro window
    upgrades rotation; a trustworthy pre-LOST velocity upgrades
    translation; either or both missing/degenerate leaves that block at
    exactly Phase 0's identity-bridge value (T=identity on that block,
    info=identity_info_scale) -- see the module docstring's fallback
    guarantee.

    Info-matrix construction mirrors `pyslam/imu/gravity.py`'s own
    anisotropic-prior pattern (build_gravity_prior_link): weight =
    1/sigma^2 on each upgraded block, left at identity_info_scale
    (Phase 0's existing "trust this hardly at all" constant) elsewhere.
    Rotation's sigma is a FIXED per-run calibration constant (gyro
    measurement quality does not degrade with elapsed gap time the way
    the constant-velocity assumption does); translation's sigma GROWS
    linearly with dt, so a bridge across a long LOST gap is honestly
    less trusted on position than a short one, even though both use the
    same velocity estimate -- see the module docstring.
    """
    min_gyro_samples = max(int(min_gyro_samples), 1)
    R_ab = None
    if imu_window is not None and imu_window.shape[0] >= min_gyro_samples:
        R_ab = gyro_bridge_rotation(imu_window, R_body_cam)
    p_ab = constant_velocity_translation(velocity_a_frame, dt)

    T_ab = np.eye(4)
    info = np.eye(6) * identity_info_scale

    if R_ab is not None:
        T_ab[:3, :3] = R_ab
        w_rot = 1.0 / max(rotation_sigma_rad, 1e-6) ** 2
        info[3:, 3:] = np.eye(3) * w_rot

    if p_ab is not None:
        T_ab[:3, 3] = p_ab
        sigma_p = velocity_sigma_base_mps + velocity_sigma_growth_mps_per_s * max(dt, 0.0)
        w_pos = 1.0 / max(sigma_p, 1e-6) ** 2
        info[:3, :3] = np.eye(3) * w_pos

    return Link(a=node_a_id, b=node_b_id, T_ab=T_ab, info=info, kind="bridge", n_inliers=0)
