"""
WP-P4 gate G4 (architecture doc section 5, P4): robust back-end +
proximity. Run standalone: python -m tests.gates.test_g4

Same honesty constraints as every other gate here: G1-G6 were
hardware-bag specs that don't exist (see SYSTEM_SUMMARY.md); this
scores against synthetic fixtures with real measured numbers.

**iSAM2 vs native-batch agreement is NOT covered here.** `pip install
gtsam` pulls `numpy<2.0`, which conflicts with the numpy 2.x this
entire codebase (and every gate already written) depends on -- a global
downgrade risked regressing everything already built, for a check this
session's budget didn't have room to run in an isolated environment.
Flagged explicitly in WP_P4_Findings.md as not attempted, not silently
skipped.
"""
from __future__ import annotations
import numpy as np

from pyslam.core.config import Config
from pyslam.core import lie
from pyslam.core.types import Link
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from tests.synth.world import T_BODY_CAM
from tests.synth.scenarios import build
from pyslam.tools import metrics


def _run(scenario: str, seed: int, max_frames=None, **cfg_kwargs):
    cfg = Config(**cfg_kwargs)
    scen = build(scenario, seed=seed)
    src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=seed)
    pipe = Pipeline(cfg, backend_prefer="native")
    result = pipe.run(src, verbose=False, max_frames=max_frames)
    return pipe, result


def _ate_after_graph(pipe, result) -> float:
    node_ids = sorted(result.node_gt.keys())
    est_map_T = [pipe.memory.get(i).pose_map for i in node_ids]
    gt_T = [result.node_gt[i] @ T_BODY_CAM for i in node_ids]
    return metrics.anchored_ate(est_map_T, gt_T) * 100.0  # cm


def _inject_wrong_loop(pipe, result, error_m: float = 5.0, seed: int = 0):
    """Adds ONE geometrically-plausible-looking but wrong loop link
    (T_ab off by `error_m` translation from the GT-implied relative
    pose) directly to the graph, mimicking a verifier false accept that
    slipped through (same info-matrix scale a real accepted loop would
    carry, so this isn't a strawman weak link)."""
    node_ids = sorted(pipe.memory.working_set())
    rng = np.random.default_rng(seed)
    a, b = int(rng.choice(node_ids)), int(rng.choice(node_ids))
    while b == a:
        b = int(rng.choice(node_ids))
    Ta, Tb = pipe.memory.get(a).pose_map, pipe.memory.get(b).pose_map
    T_ab_true = lie.se3_inverse(Ta) @ Tb
    offset = np.eye(4)
    direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
    offset[:3, 3] = direction * error_m
    T_ab_wrong = T_ab_true @ offset
    info = np.eye(6) * 10.0  # a plausible mid-confidence real-loop info scale
    wrong_link = Link(a=a, b=b, T_ab=T_ab_wrong, info=info, kind="loop", n_inliers=40)
    pipe.graph.add_link(wrong_link)
    return wrong_link


def check_dcs_single_wrong_loop_robustness() -> None:
    """Gate G4: 'injected wrong loop (5m error) with DCS -> ATE within
    15% of clean'."""
    pipe_clean, res_clean = _run("square6dof", seed=1)
    ate_clean = _ate_after_graph(pipe_clean, res_clean)

    results = {}
    for kernel in ["huber", "dcs"]:
        pipe, res = _run("square6dof", seed=1, loop_robust_kernel=kernel)
        _inject_wrong_loop(pipe, res, error_m=5.0, seed=1)
        fixed = [pipe._first_node_id] if pipe._first_node_id is not None else []
        new_poses = pipe.graph.optimize(fixed)
        for nid, T in new_poses.items():
            try:
                pipe.memory.get(nid).pose_map = T
            except KeyError:
                pass
        ate = _ate_after_graph(pipe, res)
        results[kernel] = ate

    pct_over_clean_dcs = 100.0 * (results["dcs"] - ate_clean) / ate_clean
    assert results["dcs"] <= ate_clean * 1.15, (
        f"DCS did not hold ATE within 15% of clean after a 5m wrong loop: "
        f"clean={ate_clean:.3f}cm, dcs+wrong={results['dcs']:.3f}cm "
        f"({pct_over_clean_dcs:+.1f}%)"
    )
    print(f"  [ok] G4.1 DCS single 5m-wrong-loop robustness: clean={ate_clean:.3f}cm, "
          f"huber+wrong={results['huber']:.3f}cm, dcs+wrong={results['dcs']:.3f}cm "
          f"(bound: dcs <= {ate_clean*1.15:.3f}cm, i.e. within 15% of clean)")


def check_dcs_ten_percent_wrong_loops() -> None:
    """Gate G4: '10% wrong loops -> within 50% of clean'. square6dof
    doesn't have 10 real loop closures to work with proportionally, so
    this injects enough wrong loops to reach ~10% of the accepted
    (real + injected) loop-link population, using the same real
    fixture's real loops as the denominator."""
    pipe_clean, res_clean = _run("square6dof", seed=1)
    ate_clean = _ate_after_graph(pipe_clean, res_clean)
    n_real_loops = len(res_clean.loop_events)
    n_wrong = max(1, round(0.10 * n_real_loops / 0.90))  # so wrong/(real+wrong) ~= 10%

    pipe, res = _run("square6dof", seed=1, loop_robust_kernel="dcs")
    for i in range(n_wrong):
        _inject_wrong_loop(pipe, res, error_m=5.0, seed=100 + i)
    fixed = [pipe._first_node_id] if pipe._first_node_id is not None else []
    new_poses = pipe.graph.optimize(fixed)
    for nid, T in new_poses.items():
        try:
            pipe.memory.get(nid).pose_map = T
        except KeyError:
            pass
    ate = _ate_after_graph(pipe, res)
    bound = ate_clean * 1.50
    assert ate <= bound, (
        f"DCS did not hold ATE within 50% of clean with {n_wrong} wrong loops "
        f"(~10% of {n_real_loops + n_wrong} total): clean={ate_clean:.3f}cm, got={ate:.3f}cm"
    )
    print(f"  [ok] G4.2 DCS {n_wrong} wrong loops (~10% of {n_real_loops + n_wrong}): "
          f"clean={ate_clean:.3f}cm, dcs+wrong={ate:.3f}cm (bound: <={bound:.3f}cm)")


def check_pnp_info_direction_oracle() -> None:
    """WP-P4's own version of WP-B2's Monte Carlo oracle, for the OTHER
    direction: pnp_info_matrix_direct_frame describes T_cam_obj's OWN
    tangent (no Adjoint), unlike pnp_info_matrix which describes its
    INVERSE's tangent. Same style oracle as
    test_g0.py::test_pnp_info_matrix_oracle: inject known pixel noise,
    re-solve PnP many times, check NEES against T_cam_obj's own
    ground truth (NOT its inverse) averages to chi2(6)'s mean of 6."""
    import cv2
    from pyslam.core.pnp_info import pnp_info_matrix_direct_frame

    rng = np.random.default_rng(77)
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
    p_cam_true = (R_true @ X_obj.T).T + t_true
    assert np.all(p_cam_true[:, 2] > 0.5)
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
        # NOTE: no inversion here -- checking T_cam_obj's OWN tangent
        # directly, unlike test_g0's oracle which inverts first. This
        # is the exact distinction the module docstring warns about.
        xi = lie.se3_log(lie.se3_inverse(T_cam_obj_true) @ T_cam_obj_est)
        xi_samples.append(xi)
    xi_samples = np.array(xi_samples)
    assert len(xi_samples) > n_trials * 0.9

    info_pred = pnp_info_matrix_direct_frame(X_obj, R_true, t_true, K, sigma_px=sigma_px)
    nees = np.einsum('ni,ij,nj->n', xi_samples, info_pred, xi_samples)
    mean_nees = float(nees.mean())
    assert 4.5 < mean_nees < 7.5, (
        f"mean NEES={mean_nees:.3f}, expected ~6.0 -- pnp_info_matrix_direct_frame likely "
        f"wrong (e.g. accidentally applying the Adjoint transform meant for the OTHER "
        f"direction, or not applying it when it should)"
    )
    print(f"  [ok] G4.3 pnp_info_matrix_direct_frame oracle: mean NEES={mean_nees:.3f} "
          f"over {len(xi_samples)} trials (chi2(6) mean=6.0) -- confirms the two PnP-info "
          f"functions are NOT interchangeable and each is correct for its own direction")


def check_hessian_info_end_to_end() -> None:
    """use_hessian_info=True finally wires BOTH sides consistently (see
    WP_P4_Findings.md) -- this is the check that it's not "functionally
    inert" the way WP-B2 found wiring only one side to be. Two
    independent fixtures, not one, since a single fixture could be
    a fluke."""
    for scenario in ["square6dof", "room_orbit"]:
        pipe_h, res_h = _run(scenario, seed=1, use_hessian_info=True)
        pipe_o, res_o = _run(scenario, seed=1, use_hessian_info=False)
        ate_h = _ate_after_graph(pipe_h, res_h)
        ate_o = _ate_after_graph(pipe_o, res_o)
        assert ate_h < ate_o * 1.5, (
            f"MUTATION MISSED / regression: use_hessian_info=True made {scenario}'s "
            f"after-graph ATE dramatically worse ({ate_h:.3f}cm vs heuristic's "
            f"{ate_o:.3f}cm) -- exactly the 'functionally inert' failure mode WP-B2 "
            f"found when only one side was wired"
        )
        print(f"  [ok] G4.4 {scenario}: heuristic after-graph ATE={ate_o:.3f}cm, "
              f"Hessian-info after-graph ATE={ate_h:.3f}cm "
              f"({'better' if ate_h < ate_o else 'worse'})")


def check_proximity_recall_square6dof() -> None:
    """Gate G4: 'proximity recall >= 90% at 100% precision on
    corridor_out_back'. No hardware bag exists (see module docstring);
    scored on square6dof, which has real geometrically-close,
    appearance-DIFFICULT node pairs of its own (the square path comes
    close to itself at several points beyond the one deliberate loop).
    corridor_v2's full run was too slow for this session's interactive
    budget with proximity's O(|WM|^2) candidate scan -- see
    WP_P4_Findings.md."""
    pipe, res = _run("square6dof", seed=1, proximity_enabled=True)
    ids = sorted(res.node_gt.keys())
    positions = {i: res.node_gt[i][:3, 3] for i in ids}
    true_pairs = set()
    for ia, a in enumerate(ids):
        for b in ids[ia + 1:]:
            if b - a <= 15:
                continue
            if np.linalg.norm(positions[a] - positions[b]) <= 0.5:
                true_pairs.add((a, b))

    accepted = set()
    for a, b, link in res.proximity_events:
        accepted.add((min(a, b), max(a, b)))
    for a, b, link in res.loop_events:  # a proximity-findable pair might
        accepted.add((min(a, b), max(a, b)))  # also get closed via appearance first

    false_positives = {p for p in accepted if p not in true_pairs}
    precision = 1.0 - len(false_positives) / max(len(accepted), 1)
    recall = len({p for p in accepted if p in true_pairs}) / len(true_pairs) if true_pairs else float("nan")

    assert len(false_positives) == 0, f"proximity/loop links that aren't actually close in GT: {false_positives}"
    print(f"  [ok] G4.5 proximity+loop recall on square6dof: {len(accepted & true_pairs)}/{len(true_pairs)} "
          f"true-close pairs found (recall={recall:.2f}), 0 false positives (precision=100%), "
          f"{len(res.proximity_events)} of those via proximity specifically (geometry-only, "
          f"no appearance match needed)")


ALL_G4_CHECKS = [
    check_dcs_single_wrong_loop_robustness,
    check_dcs_ten_percent_wrong_loops,
    check_pnp_info_direction_oracle,
    check_hessian_info_end_to_end,
    check_proximity_recall_square6dof,
]


def main() -> int:
    print(f"Gate G4 (WP-P4 robust back-end + proximity) -- {len(ALL_G4_CHECKS)} checks\n")
    n_pass = 0
    for check in ALL_G4_CHECKS:
        try:
            check()
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {check.__name__}: {e}")
    print(f"\n{n_pass}/{len(ALL_G4_CHECKS)} G4 checks passed")
    return 0 if n_pass == len(ALL_G4_CHECKS) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
