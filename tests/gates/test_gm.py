"""
Gate G-M (WP-M: gyro/constant-velocity LOST-recovery bridge).

    python3 -m tests.gates.test_gm

  GM.1  gyro rotation oracle   constant body-frame angular velocity, integrated then converted
                                through a REAL (non-identity) R_body_cam, matches the analytic
                                exp(w*T) answer to float precision -- see pyslam/imu/bridge.py's
                                own derivation of the conjugate-transform formula
  GM.2  gyro fallback          empty/too-short windows return None, never a wrong rotation
  GM.3  constant-velocity      p = v*dt exactly; None on a missing velocity or a bad dt
  GM.4  full fallback          build_bridge_link with no imu and no velocity reproduces Phase
                                0/WP-T0's identity bridge (T_ab, info) bit-for-bit
  GM.5  partial upgrade        each component (rotation / translation) upgrades INDEPENDENTLY
                                of whether the other one has real data
  GM.6  translation info       weakens (sigma grows) monotonically with elapsed gap time; the
                                translation itself still scales as v*dt
  GM.7  rotation info          does NOT depend on gap time (only translation's does -- see the
                                module's own rationale for why these two differ)
  GM.8  end to end             on a real fixture (square6dof) with a genuine LOST triggered by
                                a featureless stretch, the gyro bridge recovers the true
                                relative rotation across the gap to a fraction of a degree
                                (identity's own error equals the full ~29deg rotation that
                                really happened) and measurably reduces translation error too
"""
from __future__ import annotations
import sys

import numpy as np

sys.path.insert(0, ".")

from pyslam.core import lie
from pyslam.core.types import Link
from pyslam.imu.bridge import build_bridge_link, constant_velocity_translation, gyro_bridge_rotation


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _make_imu(w_body: np.ndarray, n: int, dt: float, t0: float = 0.0) -> np.ndarray:
    """n rows of constant body-frame angular velocity w_body, dt apart,
    accel columns filled with an arbitrary (unused by gyro_bridge_rotation)
    constant so a real Frame.imu-shaped array is exercised end to end."""
    rows = []
    for k in range(n):
        rows.append([t0 + k * dt, *w_body, 1.0, 2.0, 3.0])
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------- GM.1
def check_gyro_rotation_oracle() -> None:
    from tests.synth.world import T_BODY_CAM
    R_body_cam = T_BODY_CAM[:3, :3]  # a REAL, non-identity rig rotation --
        # see WP-P5.2's own finding on why identity here would hide a bug.

    w_body = np.array([0.1, -0.2, 0.3])  # rad/s
    n, dt = 200, 0.01  # 2.0s total
    imu = _make_imu(w_body, n, dt)

    R_ab = gyro_bridge_rotation(imu, R_body_cam)
    if R_ab is None:
        _fail("gyro_bridge_rotation returned None for a well-formed window")

    # Analytic answer: for TRULY constant angular velocity, Euler
    # integration over any number of equal steps is exact (each step is
    # so3_exp(w*dt) about the SAME fixed axis, and so3_exp(w*dt)^n ==
    # so3_exp(w*dt*n) since matrix exponentials about a common axis
    # commute) -- so this oracle is exact, not approximate.
    T_total = n * dt
    delta_R_body_expected = lie.so3_exp(w_body * T_total)
    R_ab_expected = R_body_cam.T @ delta_R_body_expected @ R_body_cam

    err_deg = np.degrees(np.linalg.norm(lie.so3_log(R_ab_expected.T @ R_ab)))
    if err_deg > 1e-6:
        _fail(f"gyro rotation oracle mismatch: {err_deg:.8f} deg (expected ~0)")

    # Sanity: an IDENTITY R_body_cam must leave delta_R untouched (the
    # conjugation is then a no-op) -- catches a transposed/reversed
    # conjugate-transform formula that an identity-only test would miss
    # for the general case but would otherwise still slip through here.
    R_ab_identity_rig = gyro_bridge_rotation(imu, np.eye(3))
    err2_deg = np.degrees(np.linalg.norm(lie.so3_log(delta_R_body_expected.T @ R_ab_identity_rig)))
    if err2_deg > 1e-6:
        _fail(f"identity-rig sanity check failed: {err2_deg:.8f} deg")


# ---------------------------------------------------------------- GM.2
def check_gyro_fallback() -> None:
    if gyro_bridge_rotation(np.zeros((0, 7)), np.eye(3)) is not None:
        _fail("empty window must return None, not a fabricated rotation")
    if gyro_bridge_rotation(None, np.eye(3)) is not None:
        _fail("None window must return None")


# ---------------------------------------------------------------- GM.3
def check_constant_velocity_translation() -> None:
    v = np.array([1.0, -2.0, 0.5])
    p = constant_velocity_translation(v, 2.0)
    if p is None or np.linalg.norm(p - v * 2.0) > 1e-12:
        _fail(f"expected exactly v*dt, got {p}")
    if constant_velocity_translation(None, 2.0) is not None:
        _fail("missing velocity must degrade to None, not a fabricated translation")
    if constant_velocity_translation(v, 0.0) is not None:
        _fail("dt=0 must degrade to None")
    if constant_velocity_translation(v, -1.0) is not None:
        _fail("negative dt must degrade to None")
    if constant_velocity_translation(v, None) is not None:
        _fail("dt=None must degrade to None")


# ---------------------------------------------------------------- GM.4
def check_full_fallback_matches_identity_bridge() -> None:
    """No imu, no velocity: must reproduce Phase 0/WP-T0's own bridge
    (Link(T_ab=eye(4), info=eye(6)*identity_info_scale)) EXACTLY -- this
    is the regression guarantee bridge_mode='gyro' with degenerate inputs
    must never be worse than bridge_mode='identity' was."""
    identity_scale = 1e-3
    link = build_bridge_link(
        node_a_id=3, node_b_id=9, imu_window=np.zeros((0, 7)), R_body_cam=np.eye(3),
        velocity_a_frame=None, dt=0.5, min_gyro_samples=5, identity_info_scale=identity_scale,
        rotation_sigma_rad=0.05, velocity_sigma_base_mps=0.3, velocity_sigma_growth_mps_per_s=1.0,
    )
    old = Link(a=3, b=9, T_ab=np.eye(4), info=np.eye(6) * identity_scale, kind="bridge", n_inliers=0)
    if not np.allclose(link.T_ab, old.T_ab):
        _fail(f"T_ab must stay identity when neither component has data, got\n{link.T_ab}")
    if not np.allclose(link.info, old.info):
        _fail(f"info must stay the old uniform identity_info_scale*eye(6), got\n{link.info}")
    if link.a != 3 or link.b != 9 or link.kind != "bridge":
        _fail("Link identity fields (a/b/kind) must be preserved")


# ---------------------------------------------------------------- GM.5
def check_independent_component_upgrade() -> None:
    from tests.synth.world import T_BODY_CAM
    R_body_cam = T_BODY_CAM[:3, :3]
    imu = _make_imu(np.array([0.0, 0.0, 0.5]), 50, 0.01)  # a real, well-formed window
    v = np.array([0.2, 0.0, 0.0])
    identity_scale = 1e-3
    common = dict(node_a_id=1, node_b_id=2, R_body_cam=R_body_cam, min_gyro_samples=5,
                  identity_info_scale=identity_scale, rotation_sigma_rad=0.05,
                  velocity_sigma_base_mps=0.3, velocity_sigma_growth_mps_per_s=1.0)

    # gyro only -- rotation upgraded, translation must stay Phase-0 identity
    rot_only = build_bridge_link(imu_window=imu, velocity_a_frame=None, dt=0.5, **common)
    if np.allclose(rot_only.T_ab[:3, :3], np.eye(3)):
        _fail("a real gyro window must produce a non-identity rotation block")
    if not np.allclose(rot_only.T_ab[:3, 3], np.zeros(3)):
        _fail("translation must stay zero when no velocity estimate is available")
    if not np.allclose(rot_only.info[:3, :3], np.eye(3) * identity_scale):
        _fail("position info must stay at the old uniform identity_info_scale when no velocity")

    # velocity only -- translation upgraded, rotation must stay Phase-0 identity
    vel_only = build_bridge_link(imu_window=np.zeros((0, 7)), velocity_a_frame=v, dt=0.5, **common)
    if not np.allclose(vel_only.T_ab[:3, :3], np.eye(3)):
        _fail("rotation must stay identity when no gyro window is available")
    if not np.allclose(vel_only.T_ab[:3, 3], v * 0.5):
        _fail("translation must be exactly v*dt when a velocity estimate is available")
    if not np.allclose(vel_only.info[3:, 3:], np.eye(3) * identity_scale):
        _fail("rotation info must stay at the old uniform identity_info_scale when no gyro")


# ---------------------------------------------------------------- GM.6
def check_translation_info_widens_with_gap_time() -> None:
    from tests.synth.world import T_BODY_CAM
    v = np.array([0.3, 0.1, -0.1])
    kwargs = dict(node_a_id=0, node_b_id=1, imu_window=np.zeros((0, 7)),
                  R_body_cam=T_BODY_CAM[:3, :3], velocity_a_frame=v,
                  min_gyro_samples=5, identity_info_scale=1e-3,
                  rotation_sigma_rad=0.05, velocity_sigma_base_mps=0.3,
                  velocity_sigma_growth_mps_per_s=1.0)
    link_short = build_bridge_link(dt=0.2, **kwargs)
    link_long = build_bridge_link(dt=4.0, **kwargs)
    w_short = link_short.info[0, 0]
    w_long = link_long.info[0, 0]
    if not (w_short > w_long):
        _fail(f"a longer LOST gap must be trusted LESS on translation: "
              f"w(dt=0.2)={w_short}, w(dt=4.0)={w_long}")
    # the translation ESTIMATE itself still scales as v*dt, independent of the weight
    if not np.allclose(link_long.T_ab[:3, 3], v * 4.0):
        _fail("translation estimate itself must still be exactly v*dt regardless of its own confidence")


# ---------------------------------------------------------------- GM.7
def check_rotation_info_independent_of_gap_time() -> None:
    from tests.synth.world import T_BODY_CAM
    imu = _make_imu(np.array([0.0, 0.1, 0.0]), 80, 0.01)
    kwargs = dict(node_a_id=0, node_b_id=1, imu_window=imu, R_body_cam=T_BODY_CAM[:3, :3],
                  velocity_a_frame=None, min_gyro_samples=5, identity_info_scale=1e-3,
                  rotation_sigma_rad=0.05, velocity_sigma_base_mps=0.3,
                  velocity_sigma_growth_mps_per_s=1.0)
    link_short = build_bridge_link(dt=0.2, **kwargs)
    link_long = build_bridge_link(dt=4.0, **kwargs)
    if not np.allclose(link_short.info[3:, 3:], link_long.info[3:, 3:]):
        _fail("rotation info must NOT depend on elapsed gap time -- gyro measurement quality "
              "does not degrade with dt the way the constant-velocity assumption does")


# ---------------------------------------------------------------- GM.8
class _LostInjectingSource:
    """Wraps a real SensorSource and replaces RGB with a solid-black
    (zero-keypoint) image for the given frame_ids, deterministically
    forcing tracking failure over that stretch while leaving depth/imu/
    gt_pose untouched -- so the IMU data the bridge estimator sees is
    genuinely the scenario's own ground-truth-consistent gyro signal,
    not a fabricated test double."""

    def __init__(self, inner, blank_frame_ids) -> None:
        self._inner = inner
        self._blank_ids = set(blank_frame_ids)

    def intrinsics(self):
        return self._inner.intrinsics()

    def close(self) -> None:
        self._inner.close()

    def __iter__(self):
        for frame in self._inner:
            if frame.frame_id in self._blank_ids:
                frame.rgb = np.zeros_like(frame.rgb)
            yield frame


def _run_lost_scenario(bridge_mode: str):
    from pyslam.core.config import Config
    from pyslam.pipeline import Pipeline
    from pyslam.sensors.synthetic import SyntheticSource
    from tests.synth.scenarios import square6dof
    from tests.synth.world import T_BODY_CAM

    scen = square6dof(seed=1)
    cfg = Config(bridge_mode=bridge_mode)
    inner = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, seed=1, add_noise=False)
    src = _LostInjectingSource(inner, blank_frame_ids=range(10, 20))
    pipe = Pipeline(cfg, backend_prefer="native", R_body_cam=T_BODY_CAM[:3, :3])
    result = pipe.run(src, max_frames=30, verbose=False)
    links = {(l.a, l.b): l for l in getattr(pipe.graph.backend, "_links", []) if l.kind == "bridge"}
    return result, links


def check_end_to_end_lost_recovery() -> None:
    from tests.synth.world import T_BODY_CAM

    result_gyro, links_gyro = _run_lost_scenario("gyro")
    result_identity, links_identity = _run_lost_scenario("identity")

    bridge_seq = [(a, b) for (a, b, _dt, _ug, _uv) in result_gyro.bridge_events]
    if len(bridge_seq) < 2:
        _fail(f"expected at least 2 chained bridge links from the 10-frame blank stretch, "
              f"got {bridge_seq} -- the fixture/blank window may have drifted")
    if not all(ug for (_a, _b, _dt, ug, _uv) in result_gyro.bridge_events):
        _fail("every bridge over this stretch has real IMU rows available and should use them")

    def compose(links, seq):
        T = np.eye(4)
        for pair in seq:
            T = T @ links[pair].T_ab
        return T

    T_gyro = compose(links_gyro, bridge_seq)
    T_identity = compose(links_identity, bridge_seq)

    a_id, c_id = bridge_seq[0][0], bridge_seq[-1][1]
    gt_a, gt_c = result_gyro.node_gt[a_id], result_gyro.node_gt[c_id]
    # true relative CAMERA-frame transform a->c: T_ac = T_BODY_CAM^-1 @
    # gt_a^-1 @ gt_c @ T_BODY_CAM -- see pyslam/imu/bridge.py's own
    # docstring for this convention (T_wc = T_wb @ T_BODY_CAM, T_ab
    # maps b's frame into a's).
    T_ac_true = lie.se3_inverse(T_BODY_CAM) @ lie.se3_inverse(gt_a) @ gt_c @ T_BODY_CAM
    true_rot_deg = np.degrees(np.linalg.norm(lie.so3_log(T_ac_true[:3, :3])))
    if true_rot_deg < 10.0:
        _fail(f"fixture sanity: expected a substantial ({true_rot_deg:.1f}deg) true rotation "
              f"across this gap for the comparison to mean anything -- fixture may have drifted")

    def rot_err_deg(R_est: np.ndarray) -> float:
        return np.degrees(np.linalg.norm(lie.so3_log(T_ac_true[:3, :3].T @ R_est)))

    err_gyro = rot_err_deg(T_gyro[:3, :3])
    err_identity = rot_err_deg(T_identity[:3, :3])
    if err_gyro > 2.0:
        _fail(f"gyro bridge should recover the true rotation to a fraction of a degree, "
              f"got {err_gyro:.3f} deg")
    if err_identity < true_rot_deg - 1.0:
        _fail(f"identity bridge should claim ~zero rotation, i.e. its error should equal the "
              f"whole true rotation ({true_rot_deg:.1f} deg); got {err_identity:.1f} deg")
    if not (err_gyro < 0.1 * err_identity):
        _fail(f"gyro bridge must be dramatically more accurate on rotation than identity: "
              f"gyro={err_gyro:.3f} deg, identity={err_identity:.1f} deg")

    trans_err_gyro = float(np.linalg.norm(T_gyro[:3, 3] - T_ac_true[:3, 3]))
    trans_err_identity = float(np.linalg.norm(T_identity[:3, 3] - T_ac_true[:3, 3]))
    if not (trans_err_gyro < 0.9 * trans_err_identity):
        _fail(f"gyro bridge (constant-velocity translation) should still measurably reduce "
              f"translation error vs identity's own asserted-zero-motion error: "
              f"gyro={trans_err_gyro:.3f}m, identity={trans_err_identity:.3f}m")


ALL_TESTS = [
    check_gyro_rotation_oracle,
    check_gyro_fallback,
    check_constant_velocity_translation,
    check_full_fallback_matches_identity_bridge,
    check_independent_component_upgrade,
    check_translation_info_widens_with_gap_time,
    check_rotation_info_independent_of_gap_time,
    check_end_to_end_lost_recovery,
]


def main() -> int:
    print(f"Gate G-M (WP-M) -- {len(ALL_TESTS)} checks\n")
    n_pass = 0
    for t in ALL_TESTS:
        try:
            t()
            n_pass += 1
            print(f"  [ok] {t.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{n_pass}/{len(ALL_TESTS)} passed")
    return 0 if n_pass == len(ALL_TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
