"""
Gate G5 (WP-P5.0: IMU preintegration oracle) -- the blocking
prerequisite for the rest of Phase 5 (see architecture doc section 5,
P5, and SYSTEM_SUMMARY.md's Phase 5 planning notes: SyntheticSource's
existing IMU generator is gravity-reaction-only and cannot validate
anything here).

Standalone, same pattern as G2/G3/G4 -- NOT yet wired into
pyslam.selftest (see SYSTEM_SUMMARY.md's own open item about G2/G3/G4
already not being covered there; G5 inherits that same gap rather than
being a new one).

    python -m tests.gates.test_g5
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")

from pyslam.core import lie
from pyslam.imu.preintegration import preintegrate, compose_prediction, correct_for_bias
from tests.synth.imu_world import make_imu_trajectory


def check_preintegration_convergence():
    """Noise-free preintegration against the RK4 ground-truth
    trajectory, at increasing IMU sample rates. Asserts (a) the error
    is small at a realistic 200Hz rate and (b) it converges at
    approximately the O(1/hz) rate Euler preintegration should have --
    the discriminating check: a real bug (wrong sign, swapped gyro/
    accel columns, wrong composition order) would NOT show clean
    halving behaviour as the rate increases; it would show a
    non-shrinking or inconsistent error instead."""
    traj = make_imu_trajectory(seed=0)
    t0, t1 = 0.3, 0.8
    g = traj.g_world
    T0, v0 = traj.pose(t0), traj.vel(t0)
    T1_true, v1_true = traj.pose(t1), traj.vel(t1)

    errs = []
    for hz in (100.0, 200.0, 400.0, 800.0, 1600.0):
        imu = traj.sample_imu(t0, t1, hz)
        pre = preintegrate(imu, np.zeros(3), np.zeros(3), compute_jacobians=False)
        T1_pred, v1_pred = compose_prediction(T0, v0, pre, g)
        xi = lie.se3_log(lie.se3_inverse(T1_true) @ T1_pred)
        errs.append((hz, np.linalg.norm(xi[:3]), np.linalg.norm(xi[3:]), np.linalg.norm(v1_pred - v1_true)))

    hz200_trans = [e[1] for e in errs if e[0] == 200.0][0]
    assert hz200_trans < 0.05, f"200Hz translation error {hz200_trans*100:.2f}cm, expected <5cm"

    # halving the sample interval should roughly halve the error (first-
    # order method); check consecutive ratios land in [1.5, 2.5] --
    # loose enough to tolerate real discretisation-error irregularity,
    # tight enough that a genuine bug (no convergence, or convergence at
    # the wrong order) fails it.
    for i in range(len(errs) - 1):
        ratio = errs[i][1] / max(errs[i + 1][1], 1e-12)
        assert 1.5 < ratio < 2.5, (
            f"translation error ratio between {errs[i][0]:.0f}Hz and {errs[i+1][0]:.0f}Hz "
            f"was {ratio:.2f}, expected ~2.0 (first-order convergence)")
    print(f"  [ok] Preintegration convergence: 200Hz trans_err={hz200_trans*100:.2f}cm; "
          f"error ratio doubling-to-doubling stays in [1.5,2.5] across 100->1600Hz "
          f"(consistent with first-order Euler discretisation, not a bug)")


def check_bias_correction_first_order():
    """First-order bias re-linearisation (correct_for_bias) against
    directly re-integrating with the new bias baked in from the start.
    Error should scale ~quadratically with bias magnitude (Taylor
    remainder) and stay sub-millimetre at realistic MEMS bias
    magnitudes (~0.01-0.05 rad/s gyro, ~0.05-0.3 m/s^2 accel -- see
    env_probe.py's own BMI055/BMI085 context)."""
    traj = make_imu_trajectory(seed=0)
    imu = traj.sample_imu(0.3, 0.8, 200.0)
    pre0 = preintegrate(imu, np.zeros(3), np.zeros(3), compute_jacobians=True)

    bg, ba = np.array([0.02, -0.01, 0.015]), np.array([0.1, -0.05, 0.08])
    corrected = correct_for_bias(pre0, bg, ba)
    direct = preintegrate(imu, bg, ba, compute_jacobians=False)

    rot_err = np.linalg.norm(lie.so3_log(direct.delta_R.T @ corrected.delta_R))
    v_err = np.linalg.norm(corrected.delta_v - direct.delta_v)
    p_err = np.linalg.norm(corrected.delta_p - direct.delta_p)
    assert rot_err < np.radians(0.01), f"rot err {np.degrees(rot_err):.4f}deg too large"
    assert v_err < 0.01, f"velocity err {v_err*1000:.2f}mm/s too large"
    assert p_err < 0.002, f"position err {p_err*1000:.2f}mm too large"
    print(f"  [ok] Bias-Jacobian first-order correction vs direct re-integration: "
          f"rot_err={np.degrees(rot_err):.5f}deg v_err={v_err*1000:.4f}mm/s p_err={p_err*1000:.4f}mm "
          f"(realistic MEMS bias magnitudes: |bg|={np.linalg.norm(bg):.3f}rad/s |ba|={np.linalg.norm(ba):.3f}m/s2)")


def check_zero_motion_identity():
    """Degenerate-input sanity: empty/single-sample IMU input returns
    an identity (zero-motion) preintegration rather than crashing or
    silently fabricating motion -- the kind of boundary case that's
    easy to skip when building the oracle around a rich trajectory."""
    pre_empty = preintegrate(np.zeros((0, 7)), np.zeros(3), np.zeros(3))
    assert np.allclose(pre_empty.delta_R, np.eye(3))
    assert np.allclose(pre_empty.delta_v, 0.0) and np.allclose(pre_empty.delta_p, 0.0)
    assert pre_empty.dt == 0.0

    pre_single = preintegrate(np.array([[0.5, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0]]), np.zeros(3), np.zeros(3))
    assert np.allclose(pre_single.delta_R, np.eye(3)), "single sample has no dt, should be a no-op"
    print("  [ok] Zero/single-sample IMU input degenerates to identity, not a crash or fabricated motion")


def check_gravity_tilt_correction_oracle():
    """WP-P5.2: `pyslam/imu/gravity.py`'s tilt-only correction, checked
    against a KNOWN injected error, decomposed into its tilt (should be
    corrected to ~0) and yaw (should be PRESERVED exactly, since
    gravity alone cannot observe yaw and the correction must not
    silently invent information it doesn't have) components -- the
    property that actually matters for this to be a safe partial prior,
    not just "produces a plausible-looking rotation"."""
    from pyslam.imu.gravity import tilt_correct_rotation
    from tests.synth.imu_world import make_imu_trajectory

    traj = make_imu_trajectory(seed=2)
    t = 0.6
    R_true = traj.rotation(t)
    u_world = traj.g_world / np.linalg.norm(traj.g_world)
    u_body_true = R_true.T @ u_world

    # orthonormal basis with u_body_true as one axis, for injecting a
    # controlled tilt-plane error vs a controlled yaw-axis error
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u_body_true[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u_body_true, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_body_true, e1)

    tilt_angle, yaw_angle = np.radians(5.0), np.radians(15.0)
    injected_xi = tilt_angle * e1 + yaw_angle * u_body_true
    R_vision = R_true @ lie.so3_exp(injected_xi)

    R_corrected = tilt_correct_rotation(R_vision, u_world, u_body_true)

    # (a) exact algebraic guarantee: corrected pose's implied "up" must
    # match the accelerometer's own observation exactly
    up_err = np.linalg.norm(R_corrected.T @ u_world - u_body_true)
    assert up_err < 1e-9, f"tilt_correct_rotation's own contract violated: up_err={up_err:.2e}"

    # (b) tilt removed, yaw preserved
    xi = lie.so3_log(R_true.T @ R_corrected)
    xi_yaw_mag = float(np.dot(xi, u_body_true))
    xi_tilt = xi - xi_yaw_mag * u_body_true
    assert np.linalg.norm(xi_tilt) < np.radians(0.1), (
        f"MUTATION-LIKE FAILURE: tilt not corrected, residual tilt={np.degrees(np.linalg.norm(xi_tilt)):.3f}deg")
    assert abs(xi_yaw_mag - yaw_angle) < np.radians(0.1), (
        f"yaw was disturbed: injected {np.degrees(yaw_angle):.2f}deg, "
        f"after correction {np.degrees(xi_yaw_mag):.2f}deg -- gravity prior must not touch yaw")
    print(f"  [ok] Gravity tilt-only correction: injected tilt={np.degrees(tilt_angle):.1f}deg "
          f"corrected to {np.degrees(np.linalg.norm(xi_tilt)):.4f}deg residual; "
          f"injected yaw={np.degrees(yaw_angle):.1f}deg preserved at {np.degrees(xi_yaw_mag):.2f}deg")


def check_gravity_prior_end_to_end():
    """WP-P5.2: the only END-TO-END check for the gravity prior --
    everything else in this gate tests components in isolation.
    Runs the real `Pipeline` (not a hand-crafted IMU window) on
    `gravity_init_square6dof` (the fixture built specifically because
    none of the other 5 can ever trigger this -- see
    WP_P5_Findings.md), with the correct `R_body_cam` supplied, and
    checks (a) the prior actually fires at least once and (b) the
    correction it applies is SMALL -- this fixture's vision-only
    tracking is already accurate, so a correctly-wired prior should
    only nudge the pose a small amount, not a spurious multi-degree
    correction. A regression in the camera/body frame conversion this
    check exists specifically to catch (found via this exact fixture,
    not designed in -- see pyslam/imu/gravity.py's own docstring) would
    show up here as a correction magnitude several times larger than
    this bound."""
    from pyslam.core.config import Config
    from pyslam.pipeline import Pipeline
    from pyslam.sensors.synthetic import SyntheticSource
    from tests.synth.scenarios import build
    from tests.synth.world import T_BODY_CAM

    scen = build("gravity_init_square6dof", seed=6, validate=False)
    source = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=6)
    cfg = Config(gravity_prior_enabled=True)
    pipe = Pipeline(cfg, backend_prefer="native", R_body_cam=T_BODY_CAM[:3, :3])
    result = pipe.run(source, max_frames=185, verbose=False)

    assert len(result.gravity_prior_events) >= 1, (
        "no gravity prior fired at all on gravity_init_square6dof -- the one fixture "
        "built specifically so this feature CAN fire end-to-end")

    prior_links = [l for l in pipe.graph.links if l.kind == "prior"]
    link = prior_links[0]
    node_b = pipe.memory.get(link.b)
    node0 = pipe.memory.get(link.a)
    R_target_b = node0.pose_map[:3, :3] @ link.T_ab[:3, :3]
    correction_deg = np.degrees(np.linalg.norm(lie.so3_log(node_b.pose_map[:3, :3].T @ R_target_b)))

    assert correction_deg < 2.0, (
        f"gravity-prior correction magnitude {correction_deg:.2f}deg is implausibly large "
        f"for a fixture where vision alone is already accurate -- likely the camera/body "
        f"frame conversion (R_body_cam) is wrong or missing")
    print(f"  [ok] Gravity prior end-to-end (gravity_init_square6dof): fired at node "
          f"{result.gravity_prior_events[0][0]}, correction={correction_deg:.2f}deg "
          f"(<2deg bound -- a wrong R_body_cam would blow well past this)")


ALL_G5_CHECKS = [
    check_preintegration_convergence,
    check_bias_correction_first_order,
    check_zero_motion_identity,
    check_gravity_tilt_correction_oracle,
    check_gravity_prior_end_to_end,
]


def main() -> int:
    print(f"Gate G5 (WP-P5.0 IMU preintegration oracle) -- {len(ALL_G5_CHECKS)} checks\n")
    n_pass = 0
    for check in ALL_G5_CHECKS:
        try:
            check()
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {check.__name__}: {e}")
    print(f"\n{n_pass}/{len(ALL_G5_CHECKS)} G5 checks passed")
    return 0 if n_pass == len(ALL_G5_CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
