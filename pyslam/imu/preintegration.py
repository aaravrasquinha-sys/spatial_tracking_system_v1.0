"""
WP-P5.0: IMU preintegration -- the blocking prerequisite for all of
Phase 5 (architecture doc section 5, P5). Built and oracle-validated as
a PURE function first, no dependency on Frame/Signature/Pipeline, same
discipline `local_ba.py` (WP-B1) and `pnp_info.py` (WP-B2) were built
under: get a ground-truth-checkable oracle before trusting this against
anything real, not after.

Standard on-manifold discrete preintegration between two keyframe times
t0 < t1, following Forster et al. ("On-Manifold Preintegration for
Real-Time Visual-Inertial Odometry", IMU factor as used in GTSAM), with
the Euler (not midpoint/RK4) discretisation -- the same "simplest thing
that's still correct" choice this project made for Phase 0's odometry,
appropriate at this codebase's IMU rates (200Hz synthetic, ~200-400Hz
real BMI055/BMI085 -- see env_probe.py) and revisitable later if a gate
needs tighter discretisation error than Euler gives.

Convention (matches lie.py's own T_a_b / base_pose @ exp(xi) pattern
used throughout this codebase -- backend_native.py, local_ba.py):
gyro measures BODY-frame angular velocity w_body such that
dR/dt = R @ skew(w_body), so a rotation increment over one IMU sample
is a RIGHT multiplication: R(t+dt) = R(t) @ so3_exp(w_body * dt).
World frame is gravity-aligned (P5.2's job to establish on real data);
g_world is passed in explicitly here, not assumed.

State propagated between t0 and t1, biases held CONSTANT over the
interval (first-order bias correction via preint_bias_jacobians):
  delta_R: SO(3), body(t0) -> body(t1) rotation, bias-corrected gyro only
  delta_v: R^3, velocity contribution due to specific force alone
           (gravity is added separately at composition time -- see
           compose_prediction)
  delta_p: R^3, position contribution due to specific force alone
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from pyslam.core import lie


@dataclass
class PreintegratedImu:
    delta_R: np.ndarray            # 3x3
    delta_v: np.ndarray            # (3,)
    delta_p: np.ndarray            # (3,)
    dt: float                      # total elapsed time, t1 - t0
    bg: np.ndarray                 # (3,) gyro bias this was linearised at
    ba: np.ndarray                 # (3,) accel bias this was linearised at
    n_samples: int = 0
    # First-order bias-correction Jacobians (Forster et al. eq. 44),
    # populated by preintegrate() when compute_jacobians=True. Lets a
    # later re-estimated bias (bg', ba') correct delta_R/delta_v/delta_p
    # via a first-order Taylor step instead of re-integrating from raw
    # samples -- see correct_for_bias().
    dR_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dv_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dv_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dp_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dp_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


def preintegrate(imu: np.ndarray, bg: np.ndarray, ba: np.ndarray,
                  compute_jacobians: bool = True) -> PreintegratedImu:
    """imu: (N,7) rows [t, gx,gy,gz, ax,ay,az] -- SAME layout as
    `Frame.imu` (core/types.py's frozen contract), sorted by t, spanning
    (t0, t1] (samples strictly after the previous keyframe, up to and
    including this one -- matches SyntheticSource/realsense.py's own
    "imu since previous frame" convention). Each row's dt is taken as
    the gap to the NEXT row (or, for the last row, assumed equal to the
    previous gap -- see the loop below); a single-row/empty input
    returns an identity (zero-motion) result at dt=0.

    Bias Jacobians computed via CENTRAL FINITE DIFFERENCE on bg/ba,
    not hand-derived analytically. Deliberate choice, not a shortcut
    taken to save time: WP-B2's own findings document a hand-derived
    Adjoint-placement bug that passed inspection and was only caught by
    a Monte Carlo oracle; the equivalent risk here (getting the
    recursive analytic bias-Jacobian propagation wrong) is exactly the
    class of bug this project's own established practice says to avoid
    by construction rather than catch after the fact. Numerical
    Jacobians cost more CPU per preintegration call (12 extra
    integration passes) but are correct by construction, given
    `preintegrate` itself (the thing being differentiated) is correct --
    which is exactly what `check_preintegration_oracle` verifies
    independently. An analytic-Jacobian upgrade, if the CPU cost proves
    to matter once wired into P5.4, is the same class of deferred
    optimisation F5/WP-B1-continued's BA sparsity fix both went
    through: correctness first, measured before optimising.
    """
    if imu is None or imu.shape[0] == 0:
        return PreintegratedImu(delta_R=np.eye(3), delta_v=np.zeros(3),
                                 delta_p=np.zeros(3), dt=0.0, bg=bg.copy(), ba=ba.copy())

    delta_R, delta_v, delta_p, dt_total = _integrate(imu, bg, ba)
    result = PreintegratedImu(delta_R=delta_R, delta_v=delta_v, delta_p=delta_p,
                               dt=dt_total, bg=bg.copy(), ba=ba.copy(), n_samples=imu.shape[0])
    if compute_jacobians:
        eps = 1e-5
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            Rp, vp, pp, _ = _integrate(imu, bg + d, ba)
            Rm, vm, pm, _ = _integrate(imu, bg - d, ba)
            result.dR_dbg[:, k] = lie.so3_log(Rm.T @ Rp) / (2 * eps)
            result.dv_dbg[:, k] = (vp - vm) / (2 * eps)
            result.dp_dbg[:, k] = (pp - pm) / (2 * eps)
            Rp, vp, pp, _ = _integrate(imu, bg, ba + d)
            Rm, vm, pm, _ = _integrate(imu, bg, ba - d)
            result.dv_dba[:, k] = (vp - vm) / (2 * eps)
            result.dp_dba[:, k] = (pp - pm) / (2 * eps)
    return result


def _integrate(imu: np.ndarray, bg: np.ndarray, ba: np.ndarray):
    """Euler preintegration core, shared by preintegrate() and its own
    finite-difference Jacobian evaluations above (so the Jacobians are,
    by construction, differences of the SAME code path being
    differentiated, not a separately-maintained approximation of it)."""
    n = imu.shape[0]
    delta_R = np.eye(3)
    delta_v = np.zeros(3)
    delta_p = np.zeros(3)
    ts = imu[:, 0]
    dt_prev = None
    for k in range(n):
        w = imu[k, 1:4] - bg
        a = imu[k, 4:7] - ba
        if k + 1 < n:
            dt_k = ts[k + 1] - ts[k]
        elif dt_prev is not None:
            dt_k = dt_prev  # last sample: reuse the previous gap (see docstring)
        else:
            dt_k = 0.0
        dt_prev = dt_k
        # position/velocity accel term uses delta_R BEFORE this sample's
        # rotation update (standard Euler preintegration -- the specific
        # force over [t_k, t_k+dt_k) is expressed in the body(t0) frame
        # via the rotation accumulated up to t_k, not t_k+dt_k).
        delta_p = delta_p + delta_v * dt_k + 0.5 * (delta_R @ a) * dt_k ** 2
        delta_v = delta_v + (delta_R @ a) * dt_k
        delta_R = delta_R @ lie.so3_exp(w * dt_k)
    return delta_R, delta_v, delta_p, float(ts[-1] - ts[0]) if n > 1 else 0.0


def correct_for_bias(pre: PreintegratedImu, bg_new: np.ndarray, ba_new: np.ndarray) -> PreintegratedImu:
    """First-order re-linearisation at a new bias estimate, without
    re-integrating from raw samples -- see PreintegratedImu's own
    Jacobian fields."""
    dbg, dba = bg_new - pre.bg, ba_new - pre.ba
    delta_R = pre.delta_R @ lie.so3_exp(pre.dR_dbg @ dbg)
    delta_v = pre.delta_v + pre.dv_dbg @ dbg + pre.dv_dba @ dba
    delta_p = pre.delta_p + pre.dp_dbg @ dbg + pre.dp_dba @ dba
    return PreintegratedImu(delta_R=delta_R, delta_v=delta_v, delta_p=delta_p, dt=pre.dt,
                             bg=bg_new.copy(), ba=ba_new.copy(), n_samples=pre.n_samples,
                             dR_dbg=pre.dR_dbg, dv_dbg=pre.dv_dbg, dv_dba=pre.dv_dba,
                             dp_dbg=pre.dp_dbg, dp_dba=pre.dp_dba)


def compose_prediction(T_wb0: np.ndarray, v0: np.ndarray, pre: PreintegratedImu,
                        g_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard IMU-factor composition: predict (T_wb1, v1) from
    (T_wb0, v0) and a preintegrated measurement. `g_world` passed in
    explicitly (not hardcoded) -- P5.2's own job is establishing the
    gravity-aligned world frame; this function takes it as a given
    constant, does not estimate it."""
    R0, p0 = T_wb0[:3, :3], T_wb0[:3, 3]
    dt = pre.dt
    R1 = R0 @ pre.delta_R
    v1 = v0 + g_world * dt + R0 @ pre.delta_v
    p1 = p0 + v0 * dt + 0.5 * g_world * dt ** 2 + R0 @ pre.delta_p
    T_wb1 = lie.make_T(R1, p1)
    return T_wb1, v1
