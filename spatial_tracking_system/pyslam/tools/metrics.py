"""
WP-A1 metrics: built specifically to catch what the Phase 0 gate suite
could not (see Phase 1 plan, section 0 and finding F2).

`ate_rmse` in evaluate.py Umeyama-aligns the estimate to ground truth
before scoring, which can hide a mirrored/inverted trajectory whenever
the true motion is planar (a 180deg out-of-plane rotation is a valid
rigid alignment). Everything here is either anchored (no realignment
freedom beyond the first pose) or checks per-link/per-frame quantities
where a global alignment can't average the problem away.
"""
from __future__ import annotations
import numpy as np

from pyslam.core import lie
from pyslam.tools.evaluate import umeyama_align, ate_rmse


def anchored_ate(est_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> float:
    """RMSE of position error after aligning ONLY by the first pose
    (est_poses[0]^-1 applied, then mapped into gt's frame by gt_poses[0]),
    i.e. exactly the alignment odometry itself claims to have -- no
    least-squares freedom to rotate/reflect the whole trajectory into
    agreement. Drift shows up as drift, not as a smaller RMSE from a
    better-fitting rotation."""
    assert len(est_poses) == len(gt_poses) and len(est_poses) >= 1
    T_align = gt_poses[0] @ lie.se3_inverse(est_poses[0])
    err = 0.0
    for Te, Tg in zip(est_poses, gt_poses):
        p_est_in_gt = (T_align @ Te)[:3, 3]
        err += np.sum((p_est_in_gt - Tg[:3, 3]) ** 2)
    return float(np.sqrt(err / len(est_poses)))


def mirror_check(est_positions: np.ndarray, gt_positions: np.ndarray,
                  anchored_ate_value: float, umeyama_ate_value: float,
                  ratio_threshold: float = 3.0) -> dict:
    """Flags a mirrored/inverted trajectory OR a reflected ground-truth
    fixture: Umeyama alignment has enough freedom (rotation +
    reflection-corrected via SVD) to make a mirrored trajectory score
    well, while the anchored metric (no such freedom) will not -- ratio
    >> 1 is that case, as originally implemented.

    WP-T0 fix: the original check was one-sided (ratio > threshold
    only), which is why it never caught square6dof/corridor_v2's
    ACTUAL defect -- a reflected (det(R)=-1) ground-truth fixture, where
    anchored ATE was *smaller* than Umeyama ATE (ratio << 1). That
    direction is exactly as diagnostic: two PROPER rigid alignments can
    never have anchored beat Umeyama by a wide margin either (Umeyama is
    the least-squares-OPTIMAL proper alignment, so it should be <=
    anchored whenever both are honest proper-rotation comparisons; if
    anchored comes in much smaller, Umeyama's extra reflection freedom
    is doing something the anchored (no-freedom) alignment structurally
    can't, which only happens when the estimate and ground truth differ
    by a reflection somewhere in the pipeline). Both directions are
    flagged the same way, symmetrically in log-ratio space so the same
    `ratio_threshold` applies to both.
    """
    ratio = anchored_ate_value / max(umeyama_ate_value, 1e-6)
    inv_ratio = umeyama_ate_value / max(anchored_ate_value, 1e-6)
    return {
        "anchored_ate_m": anchored_ate_value,
        "umeyama_ate_m": umeyama_ate_value,
        "ratio": ratio,
        "mirror_suspected": (ratio > ratio_threshold) or (inv_ratio > ratio_threshold),
    }


def per_link_rpe_vs_gt(links: list, gt_poses_by_id: dict) -> dict:
    """Per-odometry-link error against ground truth, in the link's own
    declared frame (a<-b). This is where a direction/convention bug
    (Phase 1 F1) shows up directly, rather than being averaged into a
    single trajectory-level number: a systematic 2x-the-true-step error
    on EVERY link is the fingerprint of an inverted transform, and this
    function reports exactly that per link rather than only in aggregate."""
    trans_errs, rot_errs = [], []
    per_link = []
    for l in links:
        if l.a not in gt_poses_by_id or l.b not in gt_poses_by_id:
            continue
        gt_ab = lie.se3_inverse(gt_poses_by_id[l.a]) @ gt_poses_by_id[l.b]
        d = lie.se3_log(lie.se3_inverse(gt_ab) @ l.T_ab)
        te, re = float(np.linalg.norm(d[:3])), float(np.degrees(np.linalg.norm(d[3:])))
        trans_errs.append(te)
        rot_errs.append(re)
        per_link.append({"a": l.a, "b": l.b, "trans_err_m": te, "rot_err_deg": re,
                          "n_inliers": l.n_inliers})
    if not trans_errs:
        return {"n_links": 0}
    return {
        "n_links": len(trans_errs),
        "trans_err_mean_m": float(np.mean(trans_errs)),
        "trans_err_max_m": float(np.max(trans_errs)),
        "rot_err_mean_deg": float(np.mean(rot_errs)),
        "rot_err_max_deg": float(np.max(rot_errs)),
        "per_link": per_link,
    }


def rpe_by_distance(est_poses: list[np.ndarray], gt_poses: list[np.ndarray],
                     path_lengths: np.ndarray, segment_m: float = 1.0,
                     session_ids: list[int] = None) -> dict:
    """KITTI-style RPE: relative pose error accumulated over ~segment_m of
    travelled distance (using cumulative path length, not frame index),
    reported as % translation drift and deg/m rotation drift. Distance-
    normalised so it's comparable across fixtures of different scale.

    session_ids (WP-B3, Phase 1 plan): if provided, a segment [i,j] is
    skipped whenever session_ids[i] != session_ids[j]. A LOST recovery
    starts a new local odometry frame with no measured transform back to
    the old one (see pipeline.py's force_new_keyframe handling); scoring
    est_poses[i] against est_poses[j] across such a boundary compares two
    UNRELATED coordinate frames and produces a meaningless number, not a
    real drift measurement -- this is exactly what made corridor_v2's
    pre-WP-B3 baseline (18%/m) misleading despite genuinely good per-link
    tracking accuracy (0.63cm mean) on the links that did form. See
    WP_A3_Findings.md for the traced root cause."""
    n = len(est_poses)
    assert len(gt_poses) == n and len(path_lengths) == n
    trans_pct, rot_per_m = [], []
    skipped_cross_session = 0
    for i in range(n):
        target = path_lengths[i] + segment_m
        j = i
        while j + 1 < n and path_lengths[j + 1] <= target:
            j += 1
        if j <= i or path_lengths[j] - path_lengths[i] < 0.5 * segment_m:
            continue
        if session_ids is not None and session_ids[i] != session_ids[j]:
            skipped_cross_session += 1
            continue
        dist = path_lengths[j] - path_lengths[i]
        rel_est = lie.se3_inverse(est_poses[i]) @ est_poses[j]
        rel_gt = lie.se3_inverse(gt_poses[i]) @ gt_poses[j]
        diff = lie.se3_inverse(rel_gt) @ rel_est
        xi = lie.se3_log(diff)
        trans_pct.append(100.0 * np.linalg.norm(xi[:3]) / dist)
        rot_per_m.append(np.degrees(np.linalg.norm(xi[3:])) / dist)
    if not trans_pct:
        return {"trans_drift_pct": 0.0, "rot_drift_deg_per_m": 0.0, "n_segments": 0,
                "n_skipped_cross_session": skipped_cross_session}
    return {
        "trans_drift_pct": float(np.mean(trans_pct)),
        "trans_drift_pct_max": float(np.max(trans_pct)),
        "rot_drift_deg_per_m": float(np.mean(rot_per_m)),
        "rot_drift_deg_per_m_max": float(np.max(rot_per_m)),
        "n_segments": len(trans_pct),
        "n_skipped_cross_session": skipped_cross_session,
    }


def link_nees(link, gt_poses_by_id: dict) -> float | None:
    """Normalised Estimation Error Squared for one link: err^T @ info @ err
    where info is the link's own information (inverse-covariance) matrix
    and err is its tangent-space error against ground truth. For a
    correctly-calibrated Gaussian estimator this should average to 6
    (the DOF) across many links -- that's the basis of the WP-B2
    covariance-consistency gate. Returns None if ground truth isn't
    available for this link's endpoints."""
    if link.a not in gt_poses_by_id or link.b not in gt_poses_by_id:
        return None
    gt_ab = lie.se3_inverse(gt_poses_by_id[link.a]) @ gt_poses_by_id[link.b]
    err = lie.se3_log(lie.se3_inverse(link.T_ab) @ gt_ab)
    return float(err @ link.info @ err)


def nees_summary(links: list, gt_poses_by_id: dict, dof: int = 6) -> dict:
    vals = [v for l in links if (v := link_nees(l, gt_poses_by_id)) is not None]
    if not vals:
        return {"n": 0}
    vals = np.array(vals)
    chi2_95 = 12.592  # scipy.stats.chi2.ppf(0.95, df=6)
    return {
        "n": len(vals),
        "mean_nees": float(np.mean(vals)),
        "mean_nees_over_dof": float(np.mean(vals) / dof),
        "frac_above_chi2_95": float(np.mean(vals > chi2_95)),
    }


def path_length_cumulative(positions: np.ndarray) -> np.ndarray:
    """Cumulative travelled distance at each pose, position[0] -> 0."""
    d = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])
