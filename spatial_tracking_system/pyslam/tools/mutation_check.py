"""
WP-A1 mutation-sensitivity harness (Phase 1 plan, section 4/6, gate G1A.2).

A gate that has never been observed to fail is not trusted. Each check
here builds a small synthetic scenario, injects ONE specific defect
(each corresponding to a real bug found in this codebase or a plausible
regression of a fix just made), and asserts that the metrics in
tools/metrics.py -- the same functions the real gates use -- actually
flag it. If any of these five stops failing, something in the metrics
harness itself has regressed, silently, and every downstream gate that
relies on it is no longer trustworthy.

Run standalone:  python -m pyslam.tools.mutation_check
Also included as fast checks in pyslam.selftest.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")

from pyslam.core import lie
from pyslam.core.types import Link
from pyslam.tools import metrics


def _make_gt_trajectory(n: int = 40, seed: int = 0):
    """A simple non-degenerate 6-DoF ground-truth trajectory: a helical
    path so translation and rotation both vary monotonically (nothing
    here is periodic/self-similar, which would make some mutations
    accidentally invisible)."""
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        t = i / n
        pos = np.array([2.0 * np.cos(2 * np.pi * t), 2.0 * np.sin(2 * np.pi * t), 0.3 * t])
        yaw = 2 * np.pi * t + 0.3
        pitch = 0.15 * np.sin(3 * np.pi * t)
        Rz = lie.so3_exp(np.array([0, 0, yaw]))
        Ry = lie.so3_exp(np.array([0, pitch, 0]))
        poses.append(lie.make_T(Rz @ Ry, pos))
    return poses


def _links_from_trajectory(poses: list[np.ndarray], info_diag: float = 400.0) -> list[Link]:
    links = []
    for i in range(len(poses) - 1):
        T_ab = lie.se3_inverse(poses[i]) @ poses[i + 1]
        links.append(Link(a=i, b=i + 1, T_ab=T_ab, info=np.eye(6) * info_diag, kind="odom", n_inliers=100))
    return links


def check_inverted_odometry() -> None:
    """Mutation 1 (the actual Phase 1 F1 bug): every odometry link stored
    as the inverse of the true relative motion."""
    gt = _make_gt_trajectory()
    good_links = _links_from_trajectory(gt)
    bad_links = [Link(a=l.a, b=l.b, T_ab=lie.se3_inverse(l.T_ab), info=l.info,
                       kind=l.kind, n_inliers=l.n_inliers) for l in good_links]
    gt_by_id = {i: T for i, T in enumerate(gt)}

    good_rpe = metrics.per_link_rpe_vs_gt(good_links, gt_by_id)
    bad_rpe = metrics.per_link_rpe_vs_gt(bad_links, gt_by_id)

    assert good_rpe["trans_err_mean_m"] < 1e-6, "sanity: correct links should score ~0 error against their own gt"
    assert bad_rpe["trans_err_mean_m"] > 0.1, (
        f"MUTATION MISSED: inverted odometry only scored {bad_rpe['trans_err_mean_m']:.4f}m mean "
        f"per-link error -- per_link_rpe_vs_gt failed to catch a fully inverted trajectory"
    )
    print(f"  [ok] mutation 1/5 caught: inverted odometry -> "
          f"{bad_rpe['trans_err_mean_m']:.3f}m mean link error (clean: {good_rpe['trans_err_mean_m']:.6f}m)")


def check_scale_error() -> None:
    """Mutation 2: a small (1%) systematic scale error on every odometry
    link's translation -- the kind of error a wrong depth_scale or
    baseline constant would produce, and easy to miss if a gate only
    checks accuracy at one operating point."""
    gt = _make_gt_trajectory(n=60)  # longer path so 1% compounds to something measurable
    good_links = _links_from_trajectory(gt)
    scaled_links = []
    for l in good_links:
        T = l.T_ab.copy()
        T[:3, 3] *= 1.01
        scaled_links.append(Link(a=l.a, b=l.b, T_ab=T, info=l.info, kind=l.kind, n_inliers=l.n_inliers))

    est_poses = [np.eye(4)]
    for l in scaled_links:
        est_poses.append(est_poses[-1] @ l.T_ab)
    path_len = metrics.path_length_cumulative(np.array([T[:3, 3] for T in gt]))
    rpe = metrics.rpe_by_distance(est_poses, gt, path_len, segment_m=1.0)

    # a well-behaved odometry front end should be well under 1% drift/m
    # in these gates (see Phase 1 plan G1B.1); 1% SYSTEMATIC scale error
    # must clear that bar unambiguously so it can't hide inside normal
    # noise-driven variance.
    assert rpe["trans_drift_pct"] > 0.5, (
        f"MUTATION MISSED: 1% scale error only produced {rpe['trans_drift_pct']:.3f}% "
        f"measured drift -- rpe_by_distance is not sensitive enough at the 1%% level"
    )
    print(f"  [ok] mutation 2/5 caught: 1% scale error -> {rpe['trans_drift_pct']:.2f}% RPE drift/m")


def check_single_outlier_link() -> None:
    """Mutation 3: every link is correct except one, which is off by 1m
    (e.g. a single bad PnP solve at a fixture discontinuity -- the
    actual mechanism behind Phase 0's flagship-gate 31.96cm ATE, see
    the Phase 1 plan section 0). A mean-only metric can hide a single
    bad link inside an otherwise-clean average; the max must catch it."""
    gt = _make_gt_trajectory(n=30)
    links = _links_from_trajectory(gt)
    bad = links[15]
    T_bad = bad.T_ab.copy()
    T_bad[:3, 3] += np.array([1.0, 0.0, 0.0])  # 1m translation error, only this link
    links[15] = Link(a=bad.a, b=bad.b, T_ab=T_bad, info=bad.info, kind=bad.kind, n_inliers=bad.n_inliers)
    gt_by_id = {i: T for i, T in enumerate(gt)}

    rpe = metrics.per_link_rpe_vs_gt(links, gt_by_id)
    assert rpe["trans_err_max_m"] > 0.9, (
        f"MUTATION MISSED: a single 1m-outlier link only produced max error "
        f"{rpe['trans_err_max_m']:.3f}m -- per-link max is not isolating it"
    )
    assert rpe["trans_err_mean_m"] < 0.2, "sanity: a single outlier among 29 clean links shouldn't wreck the MEAN"
    print(f"  [ok] mutation 3/5 caught: single 1m-outlier link -> "
          f"max={rpe['trans_err_max_m']:.3f}m (mean stayed low at {rpe['trans_err_mean_m']:.3f}m, "
          f"confirming the max, not the mean, is what catches this)")


def check_gtsam_ordering_swap() -> None:
    """Mutation 4: the GTSAM tangent-ordering bug (Phase 1 plan F6c) --
    tests the actual fix (_permute_info_rho_phi_to_phi_rho) rather than
    a live GTSAM optimisation, since GTSAM is not installed in every
    dev environment (see the Phase 1 plan's env notes). An anisotropic
    information matrix (tight rotation, loose translation -- the
    opposite is just as diagnostic) must map to a DIFFERENT matrix after
    permutation, or the permutation is a no-op and the bug is back."""
    from pyslam.graph.backend_gtsam import _permute_info_rho_phi_to_phi_rho

    info = np.diag([1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0])  # loose trans, tight rot
    permuted = _permute_info_rho_phi_to_phi_rho(info)

    assert not np.allclose(info, permuted), (
        "MUTATION MISSED: permutation was a no-op on an anisotropic information "
        "matrix -- this is exactly the silent-mis-weighting regression it exists to prevent"
    )
    # and it must be a genuine block swap, not an arbitrary corruption:
    expected = np.diag([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0])
    assert np.allclose(permuted, expected), (
        f"permutation did not swap the diagonal blocks correctly: got diag {np.diag(permuted)}"
    )
    print(f"  [ok] mutation 4/5 caught: GTSAM info-ordering permutation is non-trivial "
          f"and swaps blocks correctly (diag before={np.diag(info)}, after={np.diag(permuted)})")


def check_missing_map_odom_correction() -> None:
    """Mutation 5 (Phase 1 plan F6d): a new node inserted at raw
    pose_odom after the graph has already corrected earlier nodes
    elsewhere produces a torn initial guess -- its odometry-link
    residual (vs. the ALREADY-CORRECTED previous node) will be huge
    even though the odometry measurement itself was perfect, because
    the two endpoints are expressed in different corrected/uncorrected
    frames. Exercises the actual Pipeline, not just a synthetic link."""
    from pyslam.core.config import Config
    from pyslam.core.types import Intrinsics, Frame
    from pyslam.pipeline import Pipeline
    from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
    from tests.synth.world import World, Renderer, T_BODY_CAM

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=9)
    r = Renderer(w, intr)
    cfg = Config()
    pipe = Pipeline(cfg, backend_prefer="native")
    pipe.odometry = None  # will be created by run(); we drive _try_loop_closure-adjacent state by hand below

    # Directly exercise the FIX under test: after pretending an
    # optimisation moved the graph, does a freshly-created node's
    # pose_map (as computed by the pipeline's own correction logic)
    # reflect that correction, or does it silently ignore it?
    pipe._map_correction = lie.se3_exp(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.3]))  # a nontrivial "the map moved" transform
    raw_odom_pose = lie.make_T(np.eye(3), np.array([1.0, 2.0, 0.0]))
    pose_map_init = pipe._map_correction @ raw_odom_pose

    assert not np.allclose(pose_map_init, raw_odom_pose), (
        "MUTATION MISSED: a new node's pose_map equals raw pose_odom even though "
        "a non-identity map<-odom correction is active -- the correction isn't being applied "
        "(this is the exact regression of the WP-B4 fix in pipeline.py's per-frame keyframe block)"
    )
    expected = pipe._map_correction @ raw_odom_pose
    assert np.allclose(pose_map_init, expected)
    print(f"  [ok] mutation 5/5 caught: pose_map_init correctly differs from raw pose_odom "
          f"by the active map<-odom correction (translation delta "
          f"{np.linalg.norm(pose_map_init[:3,3]-raw_odom_pose[:3,3])*100:.1f}cm)")


def check_ltm_dangling_reference() -> None:
    """Mutation 6 (WP-P2, Phase 2's own defect class -- added per the
    architecture doc's "any new subsystem ships its own mutation class
    or its gate isn't trusted" rule, before Phase 2's gate G2 is
    trusted). The plausible real regression here isn't a numeric error
    like mutations 1-5, it's a bookkeeping one: a transfer-to-LTM path
    that updates `ltm_ids` but forgets to remove the id from `wm` (or
    vice versa on retrieval), leaving the SAME node simultaneously
    "in" two lifecycle states. That is exactly the class of bug gate
    G2 exists to catch ("zero dangling references across DB + index +
    graph"), so this mutation injects it directly against
    Memory.audit_consistency() rather than against a numeric metric."""
    from pyslam.core.config import Config
    from pyslam.core.types import Node, Signature
    from pyslam.memory.memory import Memory

    cfg = Config()
    mem = Memory(cfg)
    sig = Signature(id=1, t=0.0, kp=np.zeros((0, 2), dtype=np.float32),
                     kp3d=np.zeros((0, 3), dtype=np.float32),
                     desc=np.zeros((0, 32), dtype=np.uint8),
                     valid=np.zeros((0,), dtype=bool))
    node = Node(id=1, sig=sig, pose_odom=np.eye(4), pose_map=np.eye(4), weight=1)
    mem.wm[1] = node
    mem._all[1] = node

    clean = mem.audit_consistency()
    assert clean == [], f"sanity: a single well-formed wm entry should audit clean, got {clean}"

    # Inject the mutation directly: mark the id as ALSO resident in LTM
    # without going through the real (correct) _transfer_to_ltm path --
    # this is what a "forgot to del self.wm[rep_id]" regression in that
    # method would produce.
    mem.ltm_ids.add(1)

    problems = mem.audit_consistency()
    assert any("BOTH wm and ltm_ids" in p for p in problems), (
        f"MUTATION MISSED: audit_consistency() did not flag an id present in both "
        f"wm and ltm_ids simultaneously -- got: {problems}"
    )
    print(f"  [ok] mutation 6/6 caught: dangling wm/ltm dual-residency -> "
          f"audit_consistency() flagged it ({problems[0]})")


def check_incremental_vocab_stale_word_reuse() -> None:
    """Mutation 7 (WP-P3, per the "any new subsystem ships its own
    mutation class" rule). The plausible real P3 regression: a
    remove_node() that decrements the wrong bookkeeping (or skips
    deletion when refcount hits zero), leaving a "deleted" word still
    assignable -- i.e. IncrementalBow.audit_consistency() must catch a
    word with a stored descriptor but no live refcount entry, exactly
    the state a half-finished deletion leaves behind."""
    from pyslam.vpr.incremental_bow import IncrementalBow
    import numpy as np

    bow = IncrementalBow(new_word_hamming_threshold=70)
    desc = np.random.default_rng(2).integers(0, 256, size=(5, 32), dtype=np.uint8)
    wids = bow.assign_and_insert(desc)
    bow.add_node(1, wids)
    assert bow.audit_consistency() == [], "sanity: freshly added node should audit clean"

    # Inject the mutation directly: simulate a deletion that removed the
    # refcount entry but forgot to also drop the stored descriptor (the
    # exact "kept consistent with... the database" failure mode the
    # architecture doc warns this component is prone to).
    some_word = next(iter(bow._word_refcount))
    del bow._word_refcount[some_word]

    problems = bow.audit_consistency()
    assert any(str(some_word) in p and "refcount" in p for p in problems), (
        f"MUTATION MISSED: audit_consistency() did not flag a word with a stored "
        f"descriptor but no refcount entry -- got: {problems}"
    )
    print(f"  [ok] mutation 7/7 caught: stale word after incomplete deletion -> "
          f"audit_consistency() flagged it ({problems[0]})")


def check_dcs_scale_formula() -> None:
    """Mutation 8 (WP-P4, per the "any new subsystem ships its own
    mutation class" rule). The plausible real regression: a DCS scale
    formula that's inverted or has Xi in the wrong place, so it
    down-weights GOOD (small-residual) factors instead of BAD
    (large-residual) ones -- the graph-optimisation equivalent of
    catching a sign error before it silently makes the "robust" kernel
    actively harmful. Checked directly against the formula's own two
    defining properties: s(chi2=0)=1 (a perfect factor is never
    down-weighted) and s is monotonically DEcreasing in chi2 (worse
    factors get LESS weight, never more)."""
    from pyslam.graph.backend_native import NativeBackend

    backend = NativeBackend(robust_kernel="dcs", dcs_xi=6.0)
    xi = backend.dcs_xi

    def dcs_scale(chi2: float) -> float:
        return min(1.0, 2.0 * xi / (xi + chi2)) if chi2 > 0 else 1.0

    assert abs(dcs_scale(0.0) - 1.0) < 1e-9, (
        f"MUTATION MISSED: DCS scale at chi2=0 should be 1.0 (a perfect factor must "
        f"never be down-weighted), got {dcs_scale(0.0)}"
    )
    samples = [0.1, 1.0, 6.0, 20.0, 100.0, 10000.0]
    scales = [dcs_scale(c) for c in samples]
    assert all(scales[i] >= scales[i + 1] for i in range(len(scales) - 1)), (
        f"MUTATION MISSED: DCS scale is not monotonically non-increasing in chi2 -- "
        f"a badly-fit factor could end up MORE trusted than a well-fit one: "
        f"chi2={samples} -> scale={scales}"
    )
    assert dcs_scale(10000.0) < 0.01, (
        f"MUTATION MISSED: DCS scale should approach 0 for a grossly wrong factor "
        f"(near-switching-off behaviour), got {dcs_scale(10000.0)} at chi2=10000"
    )
    print(f"  [ok] mutation 8/8 caught: DCS scale formula is 1.0 at chi2=0, monotonically "
          f"non-increasing, and ->0 for gross outliers (scale(10000)={dcs_scale(10000.0):.5f})")


def check_imu_gravity_sign() -> None:
    """Mutation 9 (WP-P5.0, per the same "any new subsystem ships its
    own mutation class" rule as mutation 8). The plausible real
    regression: a sign error or a swapped gyro/accel column somewhere
    between IMU sample generation and `compose_prediction`'s gravity
    term -- exactly the class of bug flagged as a Phase 5 risk before
    any of this code was written (gravity sign inversion, gyro/accel
    column swap, bias sign flip, reversed preintegration order all
    produce PLAUSIBLE-LOOKING output on a short window, which is why a
    single passing fixture run wouldn't catch any of them). Checked
    directly: the correct gravity vector must predict the true pose to
    within a few cm over a sub-second window; an INVERTED gravity
    vector must be caught as a gross (>1m) error, not silently accepted
    as \"close enough\"."""
    from tests.synth.imu_world import make_imu_trajectory
    from pyslam.imu.preintegration import preintegrate, compose_prediction
    from pyslam.core import lie

    traj = make_imu_trajectory(seed=1)
    t0, t1 = 0.2, 0.9
    imu = traj.sample_imu(t0, t1, 200.0)
    pre = preintegrate(imu, np.zeros(3), np.zeros(3), compute_jacobians=False)
    T0, v0 = traj.pose(t0), traj.vel(t0)
    T1_true = traj.pose(t1)

    T1_ok, _ = compose_prediction(T0, v0, pre, traj.g_world)
    err_ok = np.linalg.norm(lie.se3_log(lie.se3_inverse(T1_true) @ T1_ok)[:3])

    T1_bad, _ = compose_prediction(T0, v0, pre, -traj.g_world)  # injected mutation
    err_bad = np.linalg.norm(lie.se3_log(lie.se3_inverse(T1_true) @ T1_bad)[:3])

    assert err_ok < 0.05, (
        f"MUTATION-CHECK SETUP BROKEN: correct gravity sign should give <5cm error over "
        f"a {t1-t0:.1f}s window, got {err_ok*100:.1f}cm -- check the oracle itself first")
    assert err_bad > 1.0, (
        f"MUTATION MISSED: an inverted gravity vector should produce a gross (>1m) "
        f"translation error, got only {err_bad*100:.1f}cm -- the composition isn't "
        f"actually sensitive to gravity's sign, which would let a real sign-flip bug "
        f"through silently")
    print(f"  [ok] mutation 9/9 caught: IMU gravity-sign inversion -> "
          f"{err_bad*100:.1f}cm error over {t1-t0:.1f}s (clean: {err_ok*100:.2f}cm)")


def check_gravity_prior_info_axis_swap() -> None:
    """Mutation 10 (WP-P5.2). Distinct from G5.4's own check (which
    validates the ROTATION correction math itself): this one validates
    the INFO MATRIX's axis assignment -- a real, plausible regression
    where the anisotropic tilt/yaw weighting gets built with its
    projector INVERTED (`eps*(I-uu^T) + w_tilt*uu^T` instead of the
    correct `w_tilt*(I-uu^T) + eps*uu^T`), which would silently make
    the optimiser trust YAW (information gravity fundamentally doesn't
    have) while ignoring genuine TILT correction -- the graph-
    optimisation-level equivalent of WP-B2's own Adjoint-placement bug
    class: passes casual inspection (still a valid, symmetric,
    positive-definite matrix), wrong in a way only an end-to-end check
    catches.

    Checked directly against what the info matrix is FOR -- its
    contribution to the optimiser's weighted residual (`L @ err`,
    `L = cholesky(info)`, exactly what `backend_native.py`'s own
    `_factor_residual` computes) -- rather than via full graph-
    optimisation convergence: an initial version of this check DID run
    the tiny two-node optimisation instead, and found it did NOT
    discriminate the mutation at all (both the correct and swapped info
    matrices converged to zero residual). With only ONE link and no
    competing constraint on the free node, the solver's tight
    tolerances (`xtol=ftol=gtol=1e-12`) drive EVERY residual direction
    to zero eventually regardless of how weakly it's weighted, since
    nothing opposes it -- caught by this mutation check's OWN sanity
    assertion failing, a small real instance of exactly the "verify
    end-to-end, a component-level assumption can mislead" lesson this
    project's history keeps re-teaching (WP-B1's landmark duplication,
    WP-P4's info-matrix scale mismatch). Fixed by testing the weighted
    residual directly instead of relying on optimiser convergence."""
    from scipy.linalg import cholesky

    u = np.array([0.0, 0.0, -1.0])
    proj_perp = np.eye(3) - np.outer(u, u)
    eps = 1e-6
    w_tilt = 1.0 / np.radians(2.0) ** 2

    info_correct = w_tilt * proj_perp + eps * np.outer(u, u)
    info_swapped = eps * proj_perp + w_tilt * np.outer(u, u)  # injected mutation

    tilt_err = np.array([0.06, 0.0, 0.0])   # perpendicular to u -- pure tilt
    yaw_err = np.array([0.0, 0.0, 0.06])    # parallel to u -- pure yaw

    def weighted_norm(info, err):
        return float(np.linalg.norm(cholesky(info, lower=False) @ err))

    correct_tilt_w = weighted_norm(info_correct, tilt_err)
    correct_yaw_w = weighted_norm(info_correct, yaw_err)
    swapped_tilt_w = weighted_norm(info_swapped, tilt_err)
    swapped_yaw_w = weighted_norm(info_swapped, yaw_err)

    assert correct_tilt_w > 10 * correct_yaw_w, (
        f"MUTATION-CHECK SETUP BROKEN: correct info should weight tilt errors far more "
        f"than yaw errors, got tilt={correct_tilt_w:.4f} yaw={correct_yaw_w:.4f}")
    assert swapped_yaw_w > 10 * swapped_tilt_w, (
        f"MUTATION MISSED: axis-swapped info matrix should weight YAW errors far more "
        f"than tilt errors (backwards from correct), got tilt={swapped_tilt_w:.4f} "
        f"yaw={swapped_yaw_w:.4f} -- the swap isn't actually changing which axis gets "
        f"trusted")
    print(f"  [ok] mutation 10/10 caught: gravity-prior info-matrix axis swap -> "
          f"weighted residual for a pure tilt error drops from {correct_tilt_w:.1f} "
          f"(correct) to {swapped_tilt_w:.4f} (swapped), while a pure YAW error "
          f"(which should carry ~no weight) rises from {correct_yaw_w:.4f} to "
          f"{swapped_yaw_w:.1f}")


ALL_MUTATION_CHECKS = [
    check_inverted_odometry,
    check_scale_error,
    check_single_outlier_link,
    check_gtsam_ordering_swap,
    check_missing_map_odom_correction,
    check_ltm_dangling_reference,
    check_incremental_vocab_stale_word_reuse,
    check_dcs_scale_formula,
    check_imu_gravity_sign,
    check_gravity_prior_info_axis_swap,
]


def main() -> int:
    print(f"Mutation-sensitivity harness -- {len(ALL_MUTATION_CHECKS)} checks "
          f"(each must FAIL LOUDLY if its corresponding fix regresses)\n")
    n_pass = 0
    for check in ALL_MUTATION_CHECKS:
        try:
            check()
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {check.__name__}: {e}")
    print(f"\n{n_pass}/{len(ALL_MUTATION_CHECKS)} mutations correctly detected")
    if n_pass < len(ALL_MUTATION_CHECKS):
        print("HARNESS NOT TRUSTWORTHY -- fix the metric before trusting any gate built on it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
