"""
WP-B1 finish: sliding-window local bundle adjustment for LocalMapOdometry
(odometry_f2m.py). See that module's docstring, section "A structural
gap (NOT fixed, scoped out)": landmarks are inserted at insertion-time
pose and never corrected, so early pose error compounds permanently into
the map (PnP inlier ratio collapsed to 17% by frame 28/140 on
square6dof, purely from this). This is the fix: jointly re-optimise the
last few keyframe poses AND the landmark positions they share, using the
actual 2D pixel observations, instead of trusting a single per-frame PnP
solve forever.

Deliberately built and tested as a PURE function first (no dependency on
LocalMapOdometry/Signature/Frame), validated against a synthetic oracle
with known ground truth BEFORE being wired into the tracker -- see
test_local_bundle_adjustment_oracle in tests/gates/test_g0.py and
WP_B1_Findings.md. This mirrors the F5 lesson from earlier in this
engagement (a first implementation that looked right on a clean synthetic
benchmark was actually wrong on real data) applied pre-emptively: get an
independent, ground-truth-checkable oracle before trusting this against
the real pipeline at all, rather than after.

Convention: SAME as backend_native.py's pose-graph optimiser --
base_pose @ exp(xi) manifold perturbation for free poses, re-linearised
each `least_squares` call. Landmark positions are plain R^3.

Gauge: window[0] (the OLDEST keyframe currently in the window) is held
FIXED as the anchor -- letting every window pose float simultaneously
would leave a 6-DoF rigid transform ungrounded. Cheap approximation to a
proper marginalised prior at the window boundary (no Schur-complement
marginalisation is performed when a keyframe slides out; it is simply
dropped) -- documented simplification, appropriate for a
"time-permitting" scope item per the Phase 1 plan's own framing, not a
claim of RTAB-Map parity.

Landmark freedom: a landmark is only a free variable if it has >=2
observations WITHIN the window. A landmark observed by exactly one
window keyframe is held FIXED: with one observation, reprojection error
is invariant to moving the point along that camera's viewing ray
(rank-deficient) -- it still constrains the observing POSE, it just
isn't refined itself until a second keyframe re-observes it.

TWO-PASS ROBUST SOLVE (added after direct testing found the single-pass
version wasn't robust enough -- see WP_B1_Findings.md): a single
`loss="huber"` solve down-weights outlier residuals but never fully
rejects them, so a landmark with very few (2-3) window observations and
one genuine cross-frame-only outlier among them (one that individually
passed its own frame's RANSAC, so was never filtered before reaching
here) can still converge to a badly wrong position -- confirmed directly:
a single 40px-corrupted observation on an otherwise noise-free
3-observation landmark pulled it 3.76m from ground truth with huber
active, AND measurably degraded several OTHER, uncorrupted landmarks
through the shared pose estimates (mean error across survivors was
comparable to or worse than doing no BA at all) before a landmark-level
post-hoc rejection alone could catch it. The fix is the standard
two-pass pattern: solve once (soft-robust), inspect residuals per
INDIVIDUAL OBSERVATION (not aggregated per landmark) against that first
solve, physically remove any observation whose residual is still large,
and re-solve cleanly on the reduced observation set. A final
landmark-level residual check (same shape as backend_native.py's
post-optimisation NEES rollback for loop links: optimise, then verify
the result, then possibly reject that specific piece of it) still runs
after pass 2 as a last-resort safety net, in case a structural conflict
survives even with the flagged observation removed.

KNOWN CONSERVATIVE BEHAVIOUR (found via the same outlier test, worth
understanding rather than "fixing" further): pass 1's contaminated
estimate for an outlier-affected landmark can make even that landmark's
OWN genuinely clean observations look like residual outliers too (their
residual is measured against pass 1's already-wrong position, not
ground truth), so pass 2 can end up stripping every observation of that
landmark down to below the >=2-observations-free threshold -- the
landmark then simply isn't refined this cycle (stays at its pre-BA
position) rather than being refined from its remaining good data. This
is intentionally conservative, not a bug: the alternative (a more
sophisticated iterative one-outlier-at-a-time removal to try to recover
the clean observations) was judged not worth the added complexity, since
the current behaviour's worst case is "misses an improvement this
cycle" -- never "applies a corrupted one" -- and the same landmark gets
another chance once the window slides and fresh observations arrive.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from pyslam.core import lie

# WP-B1 continued -- same F5 lesson (backend_native.py), applied here:
# giving least_squares a structural sparsity pattern for its
# finite-difference Jacobian lets scipy batch non-conflicting columns
# via graph colouring instead of perturbing every variable independently
# every iteration. Became necessary here specifically because the
# landmark-fusion fix (odometry_f2m.py) roughly DOUBLED the median
# number of free landmarks reaching a BA call (66 -> 138 on
# square6dof) -- profiling a real pipeline run found local_bundle_adjust
# responsible for 74% of total wall time (28s/38s over 50 frames),
# almost entirely inside scipy's dense finite-difference Jacobian
# machinery. Each observation's 2-row residual block depends on exactly
# one window pose's 6 columns (none, for the fixed anchor) plus one
# landmark's 3 columns (if free) -- >95% structural zero once a window
# has more than a couple of free landmarks, the same shape of sparsity
# F5 exploited.
#
# Gated by a size threshold rather than applied unconditionally, same
# reasoning as F5's own _SPARSE_THRESHOLD_POSES: measured on 8 real
# problem instances captured from a live square6dof/f2m run (see
# WP_B1_Findings.md), sparse won by 2.7-4.0x at n_vars in
# {126,213,333,417} but LOST (0.71-0.80x) at three mid-range instances
# (n_vars 285-306) -- not cleanly monotonic in n_vars alone, the same
# conditioning-dependent variance F5 found for the pose graph (there,
# widely-varying info-matrix weights; here, huber's own per-iteration
# reweighting presumably interacts with lsmr similarly on some problem
# shapes). Net aggregate across all 8: 20.2s dense vs 9.9s sparse, a
# real ~2x win despite 3/8 individual losses -- threshold set to route
# every one of these captured instances (and, by extension, every
# comparably-sized real BA call) onto the winning-in-aggregate path,
# not a claim that sparse is faster on every single call.
_BA_SPARSE_THRESHOLD_VARS = 120


def _build_sparsity(n_kf: int, n_pose_vars: int, n_vars: int, group_uv: list,
                     group_free_mask: list, group_free_idx: list):
    """Row layout matches resfun's exactly: window index i (skipped if
    empty) contributes 2*n_obs_i rows, obs-major, (u,v) interleaved --
    the same order `(uv_pred - group_uv[i]).reshape(-1)` produces.
    Column layout matches x's own layout: pose vars first (6 per window
    index 1..n_kf-1; index 0 is the fixed anchor, no columns), then 3
    columns per free landmark."""
    row_offsets, total_rows = [], 0
    for i in range(n_kf):
        row_offsets.append(total_rows)
        total_rows += 2 * group_uv[i].shape[0]
    if total_rows == 0:
        return None
    sp = lil_matrix((total_rows, n_vars), dtype=np.int8)
    for i in range(n_kf):
        n_obs_i = group_uv[i].shape[0]
        if n_obs_i == 0:
            continue
        r0 = row_offsets[i]
        if i >= 1:
            c0 = 6 * (i - 1)
            sp[r0:r0 + 2 * n_obs_i, c0:c0 + 6] = 1
        mask, fidx = group_free_mask[i], group_free_idx[i]
        free_rows = np.where(mask)[0]
        for j in free_rows:
            c0 = n_pose_vars + 3 * int(fidx[j])
            rr = r0 + 2 * int(j)
            sp[rr:rr + 2, c0:c0 + 3] = 1
    return sp.tocsr()


@dataclass
class KFRecord:
    pose: np.ndarray                       # 4x4, world<-cam, current estimate
    obs: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (landmark_id, uv pixel (2,))


def _solve(window: list, valid_obs: list, landmarks: dict, K: np.ndarray,
           huber_px: float, xtol: float, ftol: float, max_nfev: int):
    """One robustified solve. Returns (refined_poses, refined_lm,
    per_obs_residuals) where per_obs_residuals is a list (one entry per
    window index) of (lm_id, residual_px) pairs computed against the
    REFINED result -- used both for the final safety filter and, by the
    caller, to decide what to strip before a second pass."""
    n_kf = len(window)
    obs_count: dict[int, int] = {}
    for kept in valid_obs:
        for lm_id, _ in kept:
            obs_count[lm_id] = obs_count.get(lm_id, 0) + 1
    free_ids = sorted([lm_id for lm_id, c in obs_count.items() if c >= 2])
    free_idx_of = {lm_id: k for k, lm_id in enumerate(free_ids)}
    n_free_lm = len(free_ids)
    n_pose_vars = (n_kf - 1) * 6
    n_vars = n_pose_vars + n_free_lm * 3

    base_poses = [kf.pose.copy() for kf in window]
    if n_vars == 0:
        return base_poses, {}, [[] for _ in range(n_kf)]

    group_uv, group_free_mask, group_free_idx, group_fixed_pos, group_lm_ids = [], [], [], [], []
    for kept in valid_obs:
        if not kept:
            group_uv.append(np.zeros((0, 2)))
            group_free_mask.append(np.zeros(0, dtype=bool))
            group_free_idx.append(np.zeros(0, dtype=np.int64))
            group_fixed_pos.append(np.zeros((0, 3)))
            group_lm_ids.append([])
            continue
        uv = np.array([o[1] for o in kept], dtype=np.float64)
        mask = np.array([o[0] in free_idx_of for o in kept], dtype=bool)
        fidx = np.array([free_idx_of.get(o[0], 0) for o in kept], dtype=np.int64)
        fpos = np.array([landmarks.get(o[0], np.zeros(3)) for o in kept], dtype=np.float64)
        group_uv.append(uv)
        group_free_mask.append(mask)
        group_free_idx.append(fidx)
        group_fixed_pos.append(fpos)
        group_lm_ids.append([o[0] for o in kept])

    free_lm_x0 = np.array([landmarks[lm_id] for lm_id in free_ids], dtype=np.float64).reshape(-1)

    def unpack_poses(x_pose):
        poses = [base_poses[0]]
        for i in range(1, n_kf):
            xi = x_pose[6 * (i - 1): 6 * (i - 1) + 6]
            poses.append(base_poses[i] @ lie.se3_exp(xi))
        return poses

    def project(poses, x_lm, i):
        T_cam_world = lie.se3_inverse(poses[i])
        R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
        X = np.where(group_free_mask[i][:, None], x_lm[group_free_idx[i]], group_fixed_pos[i])
        p_cam = (R @ X.T).T + t
        z = np.clip(p_cam[:, 2], 1e-6, None)
        uv_pred = np.stack([K[0, 0] * p_cam[:, 0] / z + K[0, 2],
                             K[1, 1] * p_cam[:, 1] / z + K[1, 2]], axis=1)
        return uv_pred

    def resfun(x):
        x_pose = x[:n_pose_vars]
        x_lm = x[n_pose_vars:].reshape(n_free_lm, 3) if n_free_lm else np.zeros((0, 3))
        poses = unpack_poses(x_pose)
        out = []
        for i in range(n_kf):
            if group_uv[i].shape[0] == 0:
                continue
            uv_pred = project(poses, x_lm, i)
            out.append((uv_pred - group_uv[i]).reshape(-1))
        if not out:
            return np.zeros(1)
        return np.concatenate(out)

    x0 = np.concatenate([np.zeros(n_pose_vars), free_lm_x0])

    # WP-B1 continued: see module-level _BA_SPARSE_THRESHOLD_VARS note.
    use_sparse = n_vars >= _BA_SPARSE_THRESHOLD_VARS
    sparsity = (_build_sparsity(n_kf, n_pose_vars, n_vars, group_uv, group_free_mask, group_free_idx)
                if use_sparse else None)

    result = least_squares(resfun, x0, method="trf", xtol=xtol, ftol=ftol, gtol=xtol,
                            max_nfev=max_nfev, loss="huber", f_scale=huber_px,
                            jac_sparsity=sparsity)

    refined_poses = unpack_poses(result.x[:n_pose_vars])
    x_lm_final = result.x[n_pose_vars:].reshape(n_free_lm, 3) if n_free_lm else np.zeros((0, 3))
    refined_lm = {lm_id: x_lm_final[k] for k, lm_id in enumerate(free_ids)}

    per_obs_residuals = []
    for i in range(n_kf):
        if group_uv[i].shape[0] == 0:
            per_obs_residuals.append([])
            continue
        uv_pred = project(refined_poses, x_lm_final, i)
        resid = np.linalg.norm(uv_pred - group_uv[i], axis=1)
        per_obs_residuals.append(list(zip(group_lm_ids[i], resid.tolist())))

    return refined_poses, refined_lm, per_obs_residuals


def local_bundle_adjust(window: list[KFRecord], landmarks: dict[int, np.ndarray],
                         K: np.ndarray, huber_px: float = 3.0,
                         xtol: float = 1e-10, ftol: float = 1e-10,
                         max_nfev: int = 200,
                         reject_obs_residual_px: float = 12.0,
                         reject_landmark_residual_px: float = 15.0,
                         reject_landmark_displacement_m: float = 0.5
                         ) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """Jointly refines window[1:]'s poses and shared landmarks' positions
    against window[0] (fixed anchor). Returns (refined_poses,
    refined_landmark_positions): refined_poses has the SAME length as
    `window`; refined_landmark_positions has an entry for every free
    landmark that survived both robustness passes (see module docstring).
    Landmarks with only one window observation, or ids missing from
    `landmarks`, are omitted. `landmarks` is read-only to this function.

    Returns immediately (poses unchanged, empty landmark dict) if
    len(window) < 2.
    """
    n_kf = len(window)
    if n_kf < 2:
        return [kf.pose.copy() for kf in window], {}

    valid_obs = [[(lm_id, uv) for lm_id, uv in kf.obs if lm_id in landmarks] for kf in window]

    refined_poses, refined_lm, per_obs = _solve(window, valid_obs, landmarks, K,
                                                 huber_px, xtol, ftol, max_nfev)

    # Pass 2: strip any individual observation whose residual against
    # pass 1's result is still large, then re-solve on the reduced set.
    # See module docstring for why this catches what huber loss alone
    # (pass 1) doesn't.
    bad = set()
    for i, obs_r in enumerate(per_obs):
        for lm_id, r in obs_r:
            if r > reject_obs_residual_px:
                bad.add((i, lm_id))
    if bad:
        valid_obs2 = [[(lm_id, uv) for lm_id, uv in kept if (i, lm_id) not in bad]
                      for i, kept in enumerate(valid_obs)]
        refined_poses, refined_lm, per_obs = _solve(window, valid_obs2, landmarks, K,
                                                      huber_px, xtol, ftol, max_nfev)

    # Final safety net: per-landmark worst residual, same idea as
    # backend_native.py's post-optimisation NEES rollback -- reject
    # specific results after the fact rather than trusting the solver
    # unconditionally.
    if refined_lm:
        max_resid: dict[int, float] = {}
        for obs_r in per_obs:
            for lm_id, r in obs_r:
                if lm_id in refined_lm:
                    max_resid[lm_id] = max(max_resid.get(lm_id, 0.0), r)
        for lm_id, r in max_resid.items():
            if r > reject_landmark_residual_px:
                del refined_lm[lm_id]

    # WP-B1 continued -- SECOND, physically-grounded safety net, found
    # necessary while validating the sparse-Jacobian path above against
    # the dense one on real captured problem instances: a landmark with
    # only weak/near-degenerate triangulation support can have its
    # position AND its observing pose's correction drift together along
    # an under-constrained direction, keeping REPROJECTION residual
    # (what the check above measures) near zero even as the landmark's
    # ABSOLUTE position runs away to something physically absurd for
    # this project's fixtures (all room/corridor scale, a few metres
    # across) -- directly observed: a landmark moved from a reasonable
    # ~2m estimate to [5485, -2372, 13628] (>15km from its pre-BA
    # position) in 3 of 8 captured real instances, on BOTH the dense
    # and sparse Jacobian paths equally (confirming this is a pre-
    # existing gap in the residual-only check, not something the
    # sparsity change introduced). The residual-based check above
    # cannot catch this BY CONSTRUCTION (it's exactly the failure mode
    # where residual stays low); a bound on how far a single local-BA
    # call is allowed to move a landmark from where it started is the
    # right complementary check, mirroring the same
    # physically-grounded-plausibility pattern already used elsewhere
    # in this codebase (verify.py's verify_max_translation_m,
    # odometry_f2m.py's own _plausibility_gate motion-gate check) rather
    # than trusting the solver's own internal residual accounting alone.
    if refined_lm:
        for lm_id in list(refined_lm.keys()):
            if lm_id not in landmarks:
                continue  # shouldn't happen (free_ids is built from ids
                    # present in `landmarks`), but a landmark this
                    # function can't compare a displacement for is not
                    # one it should confidently accept either.
            moved = float(np.linalg.norm(refined_lm[lm_id] - landmarks[lm_id]))
            if moved > reject_landmark_displacement_m:
                del refined_lm[lm_id]

    return refined_poses, refined_lm
