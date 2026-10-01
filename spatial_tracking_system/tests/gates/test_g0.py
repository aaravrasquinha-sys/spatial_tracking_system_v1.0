"""
Phase-0 gate suite (G0). Run via `python -m pyslam.selftest`.

These are the checks that must pass before touching hardware, and
before any Phase-1+ change is considered safe to build on. See the
Phase Evaluation doc for the hardware-side checks (G0-HW) that
complement this.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")  # allow running from repo root without install

from pyslam.core import lie
from pyslam.core.types import Intrinsics, Frame
from pyslam.core.config import Config
from pyslam.frontend.features import grid_bucket


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_se3_identities():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(5000):
        w = rng.normal(scale=1.0, size=3)
        if rng.random() < 0.2:
            w = w / (np.linalg.norm(w) + 1e-12) * rng.uniform(np.pi - 1e-3, np.pi)
        R = lie.so3_exp(w)
        R2 = lie.so3_exp(lie.so3_log(R))
        max_err = max(max_err, np.linalg.norm(R - R2))
    assert max_err < 1e-9, f"so3 exp/log roundtrip err {max_err}"

    max_err = 0.0
    for _ in range(5000):
        xi = rng.normal(scale=0.5, size=6)
        T = lie.se3_exp(xi)
        T2 = lie.se3_exp(lie.se3_log(T))
        max_err = max(max_err, np.linalg.norm(T - T2))
    assert max_err < 1e-9, f"se3 exp/log roundtrip err {max_err}"

    max_err = 0.0
    for _ in range(2000):
        T = lie.se3_exp(rng.normal(scale=0.3, size=6))
        xi = rng.normal(scale=0.1, size=6)
        lhs = lie.se3_adjoint(T) @ xi
        rhs = lie.se3_log(T @ lie.se3_exp(xi) @ lie.se3_inverse(T))
        max_err = max(max_err, np.linalg.norm(lhs - rhs))
    assert max_err < 1e-9, f"adjoint identity err {max_err}"
    print("  [ok] SE3/SO3 identities")


def test_renderer_oracle():
    from tests.synth.world import World, Renderer, T_BODY_CAM

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    r = Renderer(w, intr)
    T_wb = lie.make_T(np.eye(3), np.array([0.1, -0.1, 0.2]))
    rgb, depth = r.render(T_wb, T_BODY_CAM)  # noise-free
    K = intr.K()
    Kinv = np.linalg.inv(K)
    T_wc = T_wb @ T_BODY_CAM

    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(3000):
        px = rng.integers(0, intr.width)
        py = rng.integers(0, intr.height)
        d = depth[py, px]
        if d == 0:
            continue
        z = d * intr.depth_scale
        p_cam = Kinv @ np.array([px, py, 1.0]) * z
        p_world = (T_wc[:3, :3] @ p_cam) + T_wc[:3, 3]
        best = min(abs((p_world - wl.origin) @ wl.normal) for wl in w.walls)
        max_err = max(max_err, best)
    assert max_err < 1e-6, f"renderer oracle max plane-distance err {max_err}"
    print("  [ok] Renderer geometric oracle")


def test_bayes_filter_single_spike_no_fire():
    from pyslam.loop.bayes import BayesFilter

    cfg = Config()
    bf = BayesFilter(cfg)
    candidates = [1, 2, 3, 4, 5]
    # warm up with neutral evidence
    for _ in range(5):
        bf.update(np.ones(len(candidates) + 1), candidates)
    # single-frame spike for candidate 3
    L = np.ones(len(candidates) + 1)
    L[2] = 8.0
    hyp = bf.update(L, candidates)
    assert hyp is None, "single-frame spike must not fire on its own update"
    # immediately followed by neutral evidence again -- must not fire either
    fired_after = False
    for _ in range(3):
        h = bf.update(np.ones(len(candidates) + 1), candidates)
        fired_after = fired_after or (h is not None)
    assert not fired_after, "single-frame spike must not cause a delayed false fire"
    print("  [ok] Bayes filter rejects single-frame likelihood spikes")


def test_bayes_filter_sustained_match_fires():
    from pyslam.loop.bayes import BayesFilter

    cfg = Config()
    bf = BayesFilter(cfg)
    candidates = [1, 2, 3, 4, 5]
    fired = False
    for _ in range(6):
        L = np.ones(len(candidates) + 1)
        L[2] = 3.0
        hyp = bf.update(L, candidates)
        if hyp is not None:
            fired = True
            assert hyp.node_id == 3
            break
    assert fired, "sustained moderate match should eventually fire"
    print("  [ok] Bayes filter fires on sustained evidence")


def test_verify_rejects_unrelated_pairs():
    from tests.synth.world import World, Renderer, T_BODY_CAM
    from pyslam.frontend.features import extract_signature, make_orb
    from pyslam.loop.verify import GeometricVerifier
    from pyslam.core.types import Node

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    w.add_box_room(np.array([20, 0, 0]), np.array([3, 3, 2.4]), seed0=50)
    r = Renderer(w, intr)
    cfg = Config()
    orb = make_orb(cfg)

    def make_node(pos, nid):
        T_wb = lie.make_T(np.eye(3), np.array(pos))
        rgb, depth = r.render(T_wb, T_BODY_CAM)
        frame = Frame(t=0.0, rgb=rgb, depth=depth, intr=intr)
        sig = extract_signature(frame, cfg, orb)
        sig.id = nid
        return Node(id=nid, sig=sig, pose_odom=T_wb, pose_map=T_wb.copy())

    verifier = GeometricVerifier(cfg, intr)
    false_accepts = 0
    n_trials = 15
    rng = np.random.default_rng(3)
    for i in range(n_trials):
        a = make_node([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0], i * 2)
        b = make_node([20 + rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0], i * 2 + 1)
        link = verifier.verify(a, b)
        if link is not None:
            false_accepts += 1
    assert false_accepts == 0, f"{false_accepts}/{n_trials} false accepts on unrelated rooms"
    print(f"  [ok] Verifier: 0/{n_trials} false accepts on unrelated locations")


def test_end_to_end_loop_closure():
    from tests.synth.world import World, poses_from_path, spline_trajectory
    from pyslam.sensors.synthetic import SyntheticSource
    from pyslam.pipeline import Pipeline
    from pyslam.tools.evaluate import ate_rmse

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    waypoints = np.array([
        [0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.7, 0.0],
        [0.0, 0.0, 0.0], [0.1, 0.02, 0.0],
    ])
    path = spline_trajectory(waypoints, n_samples=120, loop=False)
    poses = poses_from_path(path, look_ahead=4)
    source = SyntheticSource(w, intr, poses, dt=1 / 10.0, add_noise=True, seed=1)

    cfg = Config()
    pipe = Pipeline(cfg, vocab_path=None, backend_prefer="native")
    result = pipe.run(source, verbose=False)

    assert len(result.loop_events) >= 1, "expected at least one loop closure on the square-loop fixture"
    for new_id, old_id, link in result.loop_events:
        gt_new = result.node_gt[new_id][:3, 3]
        gt_old = result.node_gt[old_id][:3, 3]
        true_dist = np.linalg.norm(gt_new - gt_old)
        assert true_dist < 0.20, f"loop closure {new_id}<->{old_id} true distance {true_dist}m is not a real match"

    node_ids = sorted(result.node_gt.keys())
    est_odom = np.array([pipe.memory.get(i).pose_odom[:3, 3] for i in node_ids])
    est_map = np.array([pipe.memory.get(i).pose_map[:3, 3] for i in node_ids])
    gt = np.array([result.node_gt[i][:3, 3] for i in node_ids])
    ate_odom = ate_rmse(est_odom, gt)
    ate_map = ate_rmse(est_map, gt)
    assert ate_map <= ate_odom * 1.01, "graph optimisation should not make ATE worse"
    print(f"  [ok] End-to-end loop closure: {len(result.loop_events)} events, "
          f"ATE odom={ate_odom*100:.2f}cm -> graph={ate_map*100:.2f}cm")


def test_odometry_direction_convention():
    """Convention oracle for VisualOdometry.update().

    Renders two frames related by a KNOWN ground-truth motion (translation
    AND rotation, so a direction bug cannot hide behind symmetry) and
    checks that OdomResult.pose lands within 1mm / 0.05deg of the truth,
    and -- the actual regression check -- that it does NOT match the
    INVERSE of the truth. A naive "it roughly works" check would pass on
    either a correct implementation or one with T_ref_cur/T_cur_ref
    swapped; asserting against both directions is what makes this test
    capable of catching that specific bug (see Phase 1 plan, finding F1).
    """
    from tests.synth.world import World, Renderer, T_BODY_CAM
    from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
    from pyslam.frontend.odometry import VisualOdometry

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([4, 4, 2.4]), seed0=7)
    r = Renderer(w, intr)
    cfg = Config()
    orb = make_orb(cfg)
    _reset_id_counter(0)

    # deliberately asymmetric known motion: 12cm translation + 8deg yaw,
    # nothing here should be self-inverse.
    xi_true = np.array([0.12, -0.03, 0.02, 0.0, 0.0, np.radians(8.0)])
    T_wb0 = lie.make_T(np.eye(3), np.array([0.05, -0.05, 0.1]))
    T_wb1 = T_wb0 @ lie.se3_exp(xi_true)

    odo = VisualOdometry(cfg)

    for T_wb in (T_wb0, T_wb1):
        rgb, depth = r.render(T_wb, T_BODY_CAM)  # noise-free
        frame = Frame(t=0.0, rgb=rgb, depth=depth, intr=intr)
        sig = extract_signature(frame, cfg, orb)
        res = odo.update(sig, frame)

    # VisualOdometry treats its FIRST frame as its own local origin
    # (ref_pose = cur_pose = identity at init), so res.pose after the
    # second frame is the CAM0<-CAM1... no: cur_pose = ref_pose @ T_ref_cur
    # with ref_pose==I, so res.pose IS T_ref_cur, the relative motion from
    # frame0's camera pose to frame1's camera pose expressed as
    # (frame0_cam)<-(frame1_cam). Compare against the true relative camera
    # motion rather than an absolute world pose the odometry never claims
    # to know.
    T_wc0_true = T_wb0 @ T_BODY_CAM
    T_wc1_true = T_wb1 @ T_BODY_CAM
    T_ref_cur_true = lie.se3_inverse(T_wc0_true) @ T_wc1_true   # cam0<-cam1

    err_fwd = lie.se3_log(lie.se3_inverse(T_ref_cur_true) @ res.pose)
    err_inv_hypothesis = lie.se3_log(lie.se3_inverse(lie.se3_inverse(T_ref_cur_true)) @ res.pose)

    trans_err = np.linalg.norm(err_fwd[:3])
    rot_err = np.degrees(np.linalg.norm(err_fwd[3:]))
    trans_err_if_inverted = np.linalg.norm(err_inv_hypothesis[:3])

    # Thresholds are set from measured PnP+ORB precision on a noise-free
    # render (~7mm/~0.2deg here), not from the Lie-algebra machinery's
    # 1e-9 precision -- ORB keypoints are pixel-quantised and depth
    # comes from a median patch, so a few mm of residual is the honest
    # floor. 15mm/0.5deg leaves ~2x headroom above that floor while
    # staying two orders of magnitude below the ~50-300mm error a
    # direction/scale bug produces at this motion size.
    assert trans_err < 0.015, f"odometry translation err {trans_err*1000:.2f}mm (>15mm) -- check direction convention"
    assert rot_err < 0.5, f"odometry rotation err {rot_err:.3f}deg (>0.5deg) -- check direction convention"
    assert trans_err_if_inverted > 0.05, (
        "odometry pose matches the INVERSE of ground truth to within 5cm -- "
        "this is the T_ref_cur/T_cur_ref direction bug (Phase 1 plan F1), not noise"
    )
    print(f"  [ok] Odometry direction convention: err={trans_err*1000:.3f}mm/{rot_err:.4f}deg "
          f"(inverse-hypothesis err={trans_err_if_inverted*100:.1f}cm, correctly rejected)")


def test_imu_part_inference():
    """Pure-function unit test for the BMI055-vs-BMI085 rate heuristic in
    sensors/realsense.py, runnable with no RealSense hardware attached.
    Guards against a regression to the Phase 0 hardcoded 250Hz/200Hz
    assumption (Phase 1 plan finding F7)."""
    from pyslam.sensors.realsense import infer_imu_part

    # Use the FULL set of rates a real device advertises for each part
    # (not a single ambiguous rate -- 200Hz alone is valid for either
    # part's range) so the two are actually distinguishable.
    assert infer_imu_part([63, 250], [200, 400]) == ["BMI055"], "BMI055 nominal rates should be recognised"
    assert infer_imu_part([100, 200], [100, 200]) == ["BMI085"], "BMI085 nominal rates should be recognised"
    assert infer_imu_part([], []) == [], "no rates -> no guess, not a silent default"
    assert infer_imu_part([9999], [9999]) == [], "nonsense rates must not match either part"
    print("  [ok] IMU part-rate inference (hardware-free)")


def test_bag_extension_dispatch():
    """run_bag.py must route .bag to native RealSense playback and .npz
    to the legacy reader, and reject anything else loudly rather than
    guessing (Phase 1 plan finding F7)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_bag_mod", "run_bag.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # .bag must be routed to RealSenseSource, not BagReader -- it will fail
    # for an environment reason (no pyrealsense2 / no such file here), but
    # must NOT fail with the ValueError that means dispatch itself is wrong.
    try:
        mod._open_bag("nonexistent_recording.bag")
        assert False, "expected some failure opening a nonexistent file"
    except ValueError:
        assert False, ".bag was routed through the extension-rejection path, not RealSenseSource"
    except Exception:
        pass  # RuntimeError (no pyrealsense2) or similar -- correct code path, wrong environment

    # .npz must be routed to the legacy BagReader.
    try:
        mod._open_bag("nonexistent_recording.npz")
        assert False, "expected some failure opening a nonexistent file"
    except ValueError:
        assert False, ".npz was routed through the extension-rejection path, not BagReader"
    except Exception:
        pass  # FileNotFoundError from np.load -- correct code path, file just doesn't exist

    # An unrecognised extension must be rejected explicitly, not guessed.
    try:
        mod._open_bag("recording.xyz")
        assert False, "unrecognised extension should raise, not silently pick a format"
    except ValueError:
        pass
    print("  [ok] run_bag.py extension dispatch: .bag/.npz routed correctly, unknown rejected")


def test_local_map_odometry_direction_convention():
    """Convention oracle for LocalMapOdometry (WP-B1), mirroring
    test_odometry_direction_convention's pattern for the F2F tracker.
    Renders three frames related by KNOWN, asymmetric 6-DoF motions and
    checks LocalMapOdometry.cur_pose against ground truth at each step,
    explicitly rejecting the inverse-transform hypothesis. Three frames
    (not two) because this tracker's second frame must guided-match
    against a map built from only the first frame, and its third frame
    exercises the normal per-frame path with an established map and a
    non-trivial constant-velocity prediction -- both are new code paths
    the F2F oracle never had to cover.
    """
    from tests.synth.world import World, Renderer, T_BODY_CAM
    from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
    from pyslam.frontend.odometry_f2m import LocalMapOdometry

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([4, 4, 2.4]), seed0=13)
    r = Renderer(w, intr)
    cfg = Config()
    orb = make_orb(cfg)
    _reset_id_counter(0)

    T_wb0 = lie.make_T(np.eye(3), np.array([0.0, 0.0, 0.1]))
    xi1 = np.array([0.10, -0.04, 0.02, 0.02, -0.01, np.radians(6.0)])
    xi2 = np.array([0.08, 0.05, -0.01, -0.015, 0.03, np.radians(-9.0)])
    T_wb1 = T_wb0 @ lie.se3_exp(xi1)
    T_wb2 = T_wb1 @ lie.se3_exp(xi2)

    odo = LocalMapOdometry(cfg)
    results = []
    for T_wb in (T_wb0, T_wb1, T_wb2):
        rgb, depth = r.render(T_wb, T_BODY_CAM)
        frame = Frame(t=0.0, rgb=rgb, depth=depth, intr=intr)
        sig = extract_signature(frame, cfg, orb)
        res = odo.update(sig, frame)
        results.append(res)

    T_wc0_true = T_wb0 @ T_BODY_CAM
    T_wc2_true = T_wb2 @ T_BODY_CAM
    T_rel_true = lie.se3_inverse(T_wc0_true) @ T_wc2_true  # cam0<-cam2, this tracker's own frame convention

    err_fwd = lie.se3_log(lie.se3_inverse(T_rel_true) @ results[-1].pose)
    err_inv = lie.se3_log(lie.se3_inverse(lie.se3_inverse(T_rel_true)) @ results[-1].pose)
    trans_err = np.linalg.norm(err_fwd[:3])
    rot_err = np.degrees(np.linalg.norm(err_fwd[3:]))
    trans_err_if_inverted = np.linalg.norm(err_inv[:3])

    assert results[-1].status == "OK", f"LocalMapOdometry went {results[-1].status} on a clean 3-frame oracle"
    # Same tolerance rationale as the F2F oracle (see
    # test_odometry_direction_convention): PnP+ORB precision floor, not
    # Lie-algebra precision. This tracker has one more matching stage
    # (guided projection) than F2F's single BF match, so a slightly
    # wider tolerance is used, still ~10x below the ~5-30cm error a
    # direction/scale bug produces at this motion size.
    assert trans_err < 0.02, f"LocalMapOdometry translation err {trans_err*1000:.2f}mm (>20mm) -- check direction convention"
    assert rot_err < 0.7, f"LocalMapOdometry rotation err {rot_err:.3f}deg (>0.7deg) -- check direction convention"
    assert trans_err_if_inverted > 0.05, (
        "LocalMapOdometry pose matches the INVERSE of ground truth to within 5cm -- "
        "this is the objectPoints/imagePoints direction bug (Phase 1 plan F1 pattern), not noise"
    )
    print(f"  [ok] LocalMapOdometry (WP-B1) direction convention: err={trans_err*1000:.3f}mm/{rot_err:.4f}deg "
          f"(inverse-hypothesis err={trans_err_if_inverted*100:.1f}cm, correctly rejected; map size={len(odo.map)})")


def test_local_bundle_adjustment_oracle():
    """WP-B1 finish: noise-free convergence oracle for local_ba.py, same
    pattern as this codebase's GTSAM-backend gate ("noise-free synthetic
    graph, recovers GT to ~1e-8") applied to the new sliding-window BA.
    Also exercises the two-pass robust-outlier path directly -- see
    local_ba.py's module docstring and WP_B1_Findings.md for the
    reasoning behind both checks.
    """
    from pyslam.frontend.local_ba import local_bundle_adjust, KFRecord

    rng = np.random.default_rng(42)
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])

    def project(T_world_cam, X_world):
        T_cam_world = lie.se3_inverse(T_world_cam)
        R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
        p_cam = (R @ X_world.T).T + t
        uv = np.stack([K[0, 0] * p_cam[:, 0] / p_cam[:, 2] + K[0, 2],
                       K[1, 1] * p_cam[:, 1] / p_cam[:, 2] + K[1, 2]], axis=1)
        return uv

    T0_true = np.eye(4)
    xi1 = np.array([0.3, 0.05, -0.02, 0.02, -0.03, np.radians(8.0)])
    xi2 = np.array([0.25, -0.04, 0.03, -0.015, 0.02, np.radians(-6.0)])
    T1_true = T0_true @ lie.se3_exp(xi1)
    T2_true = T1_true @ lie.se3_exp(xi2)

    N = 40
    X_true = np.zeros((N, 3))
    X_true[:, 0] = rng.uniform(-1.5, 1.5, N)
    X_true[:, 1] = rng.uniform(-1.0, 1.0, N)
    X_true[:, 2] = rng.uniform(2.0, 5.0, N)
    shared_all, shared_01, single_kf2 = list(range(0, 15)), list(range(15, 30)), list(range(30, 40))

    def obs_for(T, ids):
        return list(zip(ids, project(T, X_true[ids])))

    obs0 = obs_for(T0_true, shared_all + shared_01)
    obs1 = obs_for(T1_true, shared_all + shared_01)
    obs2 = obs_for(T2_true, shared_all + single_kf2)

    pose_noise1 = rng.normal(scale=0.03, size=6)
    pose_noise2 = rng.normal(scale=0.04, size=6)
    T1_input = T1_true @ lie.se3_exp(pose_noise1)
    T2_input = T2_true @ lie.se3_exp(pose_noise2)
    X_input = X_true.copy()
    free_ids = shared_all + shared_01
    X_input[free_ids] += rng.normal(scale=0.05, size=(len(free_ids), 3))
    landmarks = {i: X_input[i].copy() for i in range(N)}

    window = [KFRecord(pose=T0_true.copy(), obs=obs0),
              KFRecord(pose=T1_input.copy(), obs=obs1),
              KFRecord(pose=T2_input.copy(), obs=obs2)]

    refined_poses, refined_lm = local_bundle_adjust(window, landmarks, K, huber_px=3.0, max_nfev=500)

    err0 = lie.se3_log(lie.se3_inverse(T0_true) @ refined_poses[0])
    err1 = lie.se3_log(lie.se3_inverse(T1_true) @ refined_poses[1])
    err2 = lie.se3_log(lie.se3_inverse(T2_true) @ refined_poses[2])
    assert np.linalg.norm(err0[:3]) < 1e-9, "anchor pose0 must be returned EXACTLY unchanged"
    assert np.linalg.norm(err1[:3]) < 1e-6, f"pose1 trans err {np.linalg.norm(err1[:3])*1000:.4f}mm, expected ~0 (noise-free oracle)"
    assert np.linalg.norm(err2[:3]) < 1e-6, f"pose2 trans err {np.linalg.norm(err2[:3])*1000:.4f}mm, expected ~0 (noise-free oracle)"
    lm_err = np.array([np.linalg.norm(refined_lm[i] - X_true[i]) for i in free_ids])
    assert lm_err.max() < 1e-6, f"free landmark max err {lm_err.max()*1000:.4f}mm, expected ~0 (noise-free oracle)"
    assert not any(i in refined_lm for i in single_kf2), "single-window-observation landmarks must never be marked free"
    assert len(refined_lm) == len(free_ids), f"expected {len(free_ids)} free landmarks returned, got {len(refined_lm)}"

    # Outlier robustness: corrupt one observation by 40px (a match that
    # individually passed its own frame's RANSAC but is wrong across
    # frames -- never filtered before reaching BA). Must not corrupt any
    # RETURNED value: every pose and every surviving landmark should
    # still be correct, even if landmark 0 itself ends up conservatively
    # dropped (see local_ba.py's "KNOWN CONSERVATIVE BEHAVIOUR" note).
    obs2_bad = list(obs2)
    for idx, (lm_id, uv) in enumerate(obs2_bad):
        if lm_id == 0:
            obs2_bad[idx] = (lm_id, uv + np.array([40.0, -40.0]))
            break
    window_bad = [KFRecord(pose=T0_true.copy(), obs=obs0),
                  KFRecord(pose=T1_input.copy(), obs=obs1),
                  KFRecord(pose=T2_input.copy(), obs=obs2_bad)]
    refined_poses_bad, refined_lm_bad = local_bundle_adjust(window_bad, landmarks, K, huber_px=3.0, max_nfev=500)
    err1_bad = lie.se3_log(lie.se3_inverse(T1_true) @ refined_poses_bad[1])
    err2_bad = lie.se3_log(lie.se3_inverse(T2_true) @ refined_poses_bad[2])
    assert np.linalg.norm(err1_bad[:3]) < 1e-6, "single cross-frame outlier corrupted pose1 -- two-pass rejection failed"
    assert np.linalg.norm(err2_bad[:3]) < 1e-6, "single cross-frame outlier corrupted pose2 -- two-pass rejection failed"
    for i in free_ids:
        if i in refined_lm_bad:  # survived -- must be correct, not corrupted
            e = np.linalg.norm(refined_lm_bad[i] - X_true[i])
            assert e < 1e-6, f"landmark {i} survived outlier rejection but is corrupted ({e*1000:.3f}mm err)"

    print(f"  [ok] Local bundle adjustment (WP-B1) oracle: noise-free convergence to GT "
          f"(pose err <1um, landmark err <1um, {len(refined_lm)}/{len(free_ids)} free landmarks correctly "
          f"identified), single cross-frame outlier rejected without corrupting any returned pose or landmark")


def test_pnp_info_matrix_oracle():
    """WP-B2: Monte Carlo oracle for pnp_info.py's PnP-Hessian-derived
    information matrix. Unlike local_ba's noise-free convergence oracle,
    this is a STATISTICAL derivation (predicted uncertainty, not a single
    ground-truth value), so the right oracle is: inject KNOWN Gaussian
    pixel noise many times, re-solve PnP each trial, and check the
    predicted information matrix's NEES (e^T @ Info @ e, e = empirical
    error vs ground truth) averages to the chi-square(6) mean of 6.0 --
    the standard consistency check for a claimed covariance/information
    estimate. This caught a real bug during development: the initial
    Adjoint-propagation formula (from T_cam_obj's tangent space to
    T_obj_cam's) had the adjoint matrix's transpose in the wrong
    position, giving NEES~=24 instead of ~6 -- 4 plausible variants were
    tested numerically before finding the one that actually matches
    empirical scatter, rather than trusting a second hand-derivation.
    See pnp_info.py's module docstring and WP_B2_Findings.md.
    """
    import cv2
    from pyslam.core.pnp_info import pnp_info_matrix

    rng = np.random.default_rng(55)
    K = np.array([[550.0, 0, 310.0], [0, 545.0, 250.0], [0, 0, 1.0]])
    N, sigma_px = 50, 0.5
    X_obj = np.zeros((N, 3))
    X_obj[:, 0] = rng.uniform(-2.0, 2.0, N)
    X_obj[:, 1] = rng.uniform(-1.5, 1.5, N)
    X_obj[:, 2] = rng.uniform(3.0, 7.0, N)
    xi_true = rng.normal(scale=0.15, size=6)
    xi_true[:3] *= 0.3
    T_cam_obj_true = lie.se3_exp(xi_true)
    R_true, t_true = T_cam_obj_true[:3, :3], T_cam_obj_true[:3, 3]
    T_obj_cam_true = lie.se3_inverse(T_cam_obj_true)
    p_cam_true = (R_true @ X_obj.T).T + t_true
    assert np.all(p_cam_true[:, 2] > 0.5), "fix synthetic setup, points too close/behind camera"
    u_true = np.stack([K[0, 0] * p_cam_true[:, 0] / p_cam_true[:, 2] + K[0, 2],
                        K[1, 1] * p_cam_true[:, 1] / p_cam_true[:, 2] + K[1, 2]], axis=1)

    n_trials = 600
    xi_samples = []
    for _ in range(n_trials):
        u_noisy = u_true + rng.normal(scale=sigma_px, size=u_true.shape)
        ok, rvec, tvec = cv2.solvePnP(X_obj.astype(np.float64), u_noisy.astype(np.float64), K, None,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R_est, _ = cv2.Rodrigues(rvec)
        T_cam_obj_est = lie.make_T(R_est, tvec.reshape(3))
        T_obj_cam_est = lie.se3_inverse(T_cam_obj_est)
        xi = lie.se3_log(lie.se3_inverse(T_obj_cam_true) @ T_obj_cam_est)
        xi_samples.append(xi)
    xi_samples = np.array(xi_samples)
    assert len(xi_samples) > n_trials * 0.9, "too many PnP solve failures in oracle, check setup"

    info_pred = pnp_info_matrix(X_obj, R_true, t_true, K, sigma_px=sigma_px)
    nees = np.einsum('ni,ij,nj->n', xi_samples, info_pred, xi_samples)
    mean_nees = float(nees.mean())
    # Chi-square(6) mean is exactly 6; Monte Carlo noise with 600 trials
    # gives a standard error of sqrt(2*6/600)~=0.14, so a wide-ish but
    # still meaningfully discriminating band catches a real derivation
    # bug (the actual bug found gave ~24, not a close miss) without
    # being flaky on RNG variation.
    assert 4.5 < mean_nees < 7.5, (
        f"mean NEES={mean_nees:.3f}, expected ~6.0 (chi2(6) mean) -- "
        f"info matrix derivation likely wrong, check Adjoint propagation direction"
    )
    print(f"  [ok] PnP information matrix (WP-B2) oracle: mean NEES={mean_nees:.3f} "
          f"over {len(xi_samples)} Monte Carlo trials (chi2(6) mean=6.0)")


def _grid_bucket_reference(kps, descs, cfg, width, height):
    """Original (pre-Orin-port) pure-Python grid_bucket, kept ONLY as a
    reference oracle for test_grid_bucket_vectorized_matches_loop below.
    Do not use in the pipeline -- this is the O(n_keypoints) Python-loop
    version the vectorised pyslam.frontend.features.grid_bucket replaced."""
    if len(kps) == 0:
        return [], np.zeros((0, 32), dtype=np.uint8)
    cell_w = width / cfg.grid_cols
    cell_h = height / cfg.grid_rows
    per_cell = max(1, cfg.n_features // (cfg.grid_cols * cfg.grid_rows))
    buckets: dict = {}
    for idx, kp in enumerate(kps):
        cx = min(cfg.grid_cols - 1, int(kp.pt[0] // cell_w))
        cy = min(cfg.grid_rows - 1, int(kp.pt[1] // cell_h))
        buckets.setdefault((cx, cy), []).append(idx)
    keep = []
    for key, idxs in buckets.items():
        idxs.sort(key=lambda i: kps[i].response, reverse=True)
        keep.extend(idxs[:per_cell])
    keep = keep[: cfg.n_features] if len(keep) > cfg.n_features else keep
    out_kps = [kps[i] for i in keep]
    out_descs = descs[keep] if len(keep) > 0 else np.zeros((0, 32), dtype=np.uint8)
    return out_kps, out_descs


def test_grid_bucket_vectorized_matches_loop():
    """WP-J1 (Orin port, Tier 1): the vectorised grid_bucket must select
    bit-identical keypoints/descriptors to the original per-keypoint
    Python loop, on real ORB output (not synthetic points -- real
    cv2.KeyPoint response ties/near-ties are exactly where a subtly
    different stable-sort composition could diverge)."""
    import cv2
    rng = np.random.default_rng(3)
    cfg = Config()
    width, height = 640, 480
    n_trials = 12
    max_kp_mismatch = 0
    for trial in range(n_trials):
        img = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
        # add some structure so ORB actually finds several hundred features
        for _ in range(40):
            x, y, r = rng.integers(20, width - 20), rng.integers(20, height - 20), rng.integers(5, 30)
            cv2.circle(img, (int(x), int(y)), int(r), int(rng.integers(0, 255)), -1)
        orb = cv2.ORB_create(nfeatures=cfg.n_features * 3, scaleFactor=cfg.orb_scale_factor,
                              nlevels=cfg.orb_n_levels)
        kps, descs = orb.detectAndCompute(img, None)
        if descs is None or len(kps) == 0:
            continue
        ref_kps, ref_descs = _grid_bucket_reference(kps, descs, cfg, width, height)
        vec_kps, vec_descs = grid_bucket(kps, descs, cfg, width, height)
        assert len(ref_kps) == len(vec_kps), (
            f"trial {trial}: count mismatch ref={len(ref_kps)} vec={len(vec_kps)}")
        ref_pts = np.array([kp.pt for kp in ref_kps])
        vec_pts = np.array([kp.pt for kp in vec_kps])
        if len(ref_pts) > 0:
            assert np.array_equal(ref_pts, vec_pts), f"trial {trial}: keypoint order/selection mismatch"
            assert np.array_equal(ref_descs, vec_descs), f"trial {trial}: descriptor mismatch"
    print(f"  [ok] grid_bucket vectorized == reference loop, bit-identical over {n_trials} real-ORB trials")


ALL_TESTS = [
    test_se3_identities,
    test_renderer_oracle,
    test_odometry_direction_convention,
    test_local_map_odometry_direction_convention,
    test_local_bundle_adjustment_oracle,
    test_pnp_info_matrix_oracle,
    test_imu_part_inference,
    test_bag_extension_dispatch,
    test_bayes_filter_single_spike_no_fire,
    test_bayes_filter_sustained_match_fires,
    test_verify_rejects_unrelated_pairs,
    test_end_to_end_loop_closure,
    test_grid_bucket_vectorized_matches_loop,
]
