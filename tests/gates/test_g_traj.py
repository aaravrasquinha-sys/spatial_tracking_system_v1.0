"""
Gate G-TRAJ (WP-T0/T1/T2/T3): trajectory-export and LOST-recovery
correctness checks. Run via `python -m tests.gates.test_g_traj`, and
included in `python -m pyslam.selftest` alongside G0's own checks (see
pyslam/selftest.py).

These exist because WP-T0 through WP-T3 fixed real, previously-uncaught
defects (see git history / WP_T_Findings.md):
  - a LOST recovery could leave a graph node with zero factors, which
    crashes GTSAM's elimination ordering (the exact 'inconsistent
    arguments' RuntimeError seen on corridor_v2's hardware run) and is
    at best silently under-constrained on the native backend;
  - the TUM/g2o exporters reorder this codebase's [w,x,y,z] quaternion
    convention into each format's own (w-last) convention, a classic
    place for a silent sign/order bug to hide;
  - gravity_frame.py's up-vector/axis construction has enough vector-
    algebra surface area (cross products, sign conventions, degenerate-
    input fallbacks) to be worth an oracle check independent of any
    real IMU data.
None of G0-G5's existing checks touch any of this.
"""
from __future__ import annotations
from dataclasses import replace
import sys
import numpy as np

sys.path.insert(0, ".")

from pyslam.core import lie
from pyslam.core.config import Config
from pyslam.core.gravity_frame import estimate_gravity_alignment
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from pyslam.tools.trajectory_export import write_tum, read_tum
from tests.synth.scenarios import build
from tests.synth.world import T_BODY_CAM


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def check_no_orphan_node_after_repeated_lost() -> None:
    """WP-T0's core invariant: force odometry LOST repeatedly (via an
    impossibly high odom_min_inliers threshold -- every frame after
    initialisation fails to match enough inliers, so pipeline._run_loop's
    LOST branch and force_new_keyframe() fire over and over) and confirm
    (a) the graph never splits into more than one connected component,
    (b) pipeline.finalize() -- the exact full-graph optimize() call a
    real GTSAM run would make at shutdown -- completes without raising,
    and (c) every node the graph ever saw gets a pose back from it. This
    is a synthetic stand-in for corridor_v2's hardware crash: the native
    backend (available in this environment) doesn't raise GTSAM's
    specific 'inconsistent arguments' error even on an orphan node (it's
    a different linear-algebra implementation), so this test's real
    value is (a) -- confirming the graph-connectivity invariant itself,
    which is what GTSAM's crash was actually a symptom of.
    """
    cfg = replace(Config(), odom_min_inliers=100000)  # impossible to satisfy
    scen = build("square6dof", seed=1, validate=True)
    src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=1)
    p = Pipeline(cfg, backend_prefer="native", R_body_cam=T_BODY_CAM[:3, :3])
    r = p.run(src, verbose=False)

    if r.n_keyframes < 3:
        _fail(f"expected several forced keyframes from repeated LOST, got only {r.n_keyframes}")
    n_bridge = sum(1 for s in r.status_log if "bridge link" in s)
    if n_bridge < 1:
        _fail("expected at least one WP-T0 bridge link to have fired under an impossible "
              "odom_min_inliers threshold -- either LOST never triggered (threshold too "
              "weak for this fixture) or the bridge-link code path regressed")

    roots = {p._uf_find(nid) for nid in p.graph.node_ids}
    if len(roots) != 1:
        _fail(f"graph split into {len(roots)} connected components after repeated LOST -- "
              f"WP-T0's bridge link is not keeping the graph connected")

    final_poses = p.finalize(r)
    missing = [nid for nid in p.graph.node_ids if nid not in final_poses]
    if missing:
        _fail(f"finalize() (full-graph optimize()) did not return a pose for node id(s) "
              f"{missing} -- these would be exactly the nodes a real GTSAM elimination-"
              f"ordering crash reports as inconsistent")
    for nid, T in final_poses.items():
        if not np.all(np.isfinite(T)):
            _fail(f"finalize() returned a non-finite pose for node {nid}: {T}")

    print(f"  [ok] G-TRAJ.1 no-orphan-after-LOST: {r.n_keyframes} keyframes, "
          f"{n_bridge} bridge link(s) fired, graph stayed 1 connected component, "
          f"finalize() returned {len(final_poses)}/{len(p.graph.node_ids)} finite poses")


def check_tum_round_trip() -> None:
    """WP-T2: write_tum/read_tum must round-trip poses to numerical
    precision, INCLUDING the quaternion reorder -- this codebase's own
    convention is [w,x,y,z] (pyslam.core.lie), TUM's is [x,y,z,w]. A
    silent mixup here would still produce a file that LOOKS well-formed
    (8 numbers per line) and would only show up as a wrong ORIENTATION
    in every downstream consumer, which is exactly the kind of bug this
    gate exists to catch before it ships."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi, np.pi)
        R = lie.so3_exp(axis * angle)
        t = rng.normal(size=3) * 2.0
        T = lie.make_T(R, t)
        rows.append((float(i) * 0.1, T))

    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "test_g_traj_roundtrip.tum")
    write_tum(path, rows)
    rows_back = read_tum(path)
    os.remove(path)

    if len(rows_back) != len(rows):
        _fail(f"round-trip changed row count: {len(rows)} -> {len(rows_back)}")
    rows_sorted = sorted(rows, key=lambda r: r[0])
    max_t_err, max_R_err = 0.0, 0.0
    for (t0, T0), (t1, T1) in zip(rows_sorted, rows_back):
        if abs(t0 - t1) > 1e-6:
            _fail(f"timestamp round-trip mismatch: {t0} -> {t1}")
        max_t_err = max(max_t_err, float(np.linalg.norm(T0[:3, 3] - T1[:3, 3])))
        max_R_err = max(max_R_err, float(np.linalg.norm(T0[:3, :3] - T1[:3, :3])))
    if max_t_err > 1e-6 or max_R_err > 1e-6:
        _fail(f"round-trip pose error too large: max_t_err={max_t_err:.2e}m "
              f"max_R_err={max_R_err:.2e} (quaternion w-first/w-last reorder likely wrong)")
    print(f"  [ok] G-TRAJ.2 TUM round-trip: {len(rows)} poses, "
          f"max_t_err={max_t_err:.2e}m max_R_err={max_R_err:.2e}")


def check_gravity_alignment_oracle() -> None:
    """WP-T3 oracle: feed a hand-built quasi-static IMU window whose
    accelerometer reads a KNOWN 'up' direction (in body frame), with a
    KNOWN R_body_cam, and confirm estimate_gravity_alignment recovers a
    T_grav_cam0 that maps that direction to world +z -- independent of
    any real hardware/synthetic-scenario data, so this isolates the
    module's own vector algebra from everything else that could be
    wrong upstream of it (accelerometer sign convention, R_body_cam
    extraction, etc., all covered by other gates/oracles already)."""
    rng = np.random.default_rng(1)
    # A stationary accelerometer reads +g opposite the direction gravity
    # pulls (see pyslam.imu.gravity's own docstring) -- pick an arbitrary
    # known body-frame "up" and build synthetic accel samples around it.
    true_up_body = np.array([0.1, 0.2, 0.97])
    true_up_body /= np.linalg.norm(true_up_body)
    g_mag = 9.81
    n = 50
    gyro = rng.normal(scale=0.005, size=(n, 3))  # well under is_quasi_static's threshold
    accel = true_up_body[None, :] * g_mag + rng.normal(scale=0.02, size=(n, 3))
    t = np.arange(n) * 0.01
    imu = np.concatenate([t[:, None], gyro, accel], axis=1)

    # An arbitrary (non-identity) R_body_cam, same convention pipeline.py
    # uses: v_body = R_body_cam @ v_cam.
    R_body_cam = lie.so3_exp(np.array([0.3, -0.2, 0.1]))

    align = estimate_gravity_alignment([imu], R_body_cam)
    if not align.aligned:
        _fail(f"oracle window should have been judged quasi-static, but alignment failed: {align.reason}")

    # up_cam0 should equal R_body_cam.T @ true_up_body (same relation
    # pipeline.py's own _try_gravity_prior uses for self._g_world).
    expected_up_cam0 = R_body_cam.T @ true_up_body
    err = float(np.linalg.norm(align.up_cam0 - expected_up_cam0))
    if err > 0.05:
        _fail(f"up_cam0 does not match the expected R_body_cam.T @ true_up_body relation "
              f"(err={err:.4f}) -- check gravity_frame.py's rotation convention")

    # And T_grav_cam0 must map up_cam0 to world +z.
    up_in_grav = align.T_grav_cam0[:3, :3] @ align.up_cam0
    z_err = float(np.linalg.norm(up_in_grav - np.array([0, 0, 1.0])))
    if z_err > 1e-6:
        _fail(f"T_grav_cam0 does not map the estimated up direction to world +z "
              f"(residual={z_err:.2e}) -- gravity_frame.py's R_grav_cam0 construction is wrong")

    # T_grav_cam0's rotation must be proper (det=+1), not a reflection --
    # the exact class of bug WP-T0 found and fixed in the synthetic
    # fixtures (tests/synth/world.py::poses_from_path).
    det = float(np.linalg.det(align.T_grav_cam0[:3, :3]))
    if abs(det - 1.0) > 1e-9:
        _fail(f"T_grav_cam0's rotation is not proper: det={det:.9f} (expected +1)")

    print(f"  [ok] G-TRAJ.3 gravity alignment oracle: up_cam0 err={err:.4f}, "
          f"world-z residual={z_err:.2e}, det(R)={det:.9f}")


ALL_TESTS = [
    check_no_orphan_node_after_repeated_lost,
    check_tum_round_trip,
    check_gravity_alignment_oracle,
]


def main() -> int:
    print(f"Gate G-TRAJ (WP-T0/T1/T2/T3) -- {len(ALL_TESTS)} checks\n")
    n_pass, n_fail = 0, 0
    for test in ALL_TESTS:
        try:
            test()
            n_pass += 1
        except AssertionError as e:
            n_fail += 1
            print(f"  [FAIL] {test.__name__}: {e}")
        except Exception:
            import traceback
            n_fail += 1
            print(f"  [ERROR] {test.__name__}:")
            traceback.print_exc()
    print(f"\n{n_pass}/{len(ALL_TESTS)} G-TRAJ checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
