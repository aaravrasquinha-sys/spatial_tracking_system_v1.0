"""
WP-B1: frame-to-local-map odometry (Phase 1 plan section 5).

Phase 0's VisualOdometry (odometry.py) tracks frame-to-KEYFRAME: every
frame is matched against only the single most recent keyframe's 3D
points, and the camera pose is built by composing a chain of relative
deltas (world<-cur = world<-ref @ ref<-cur). This upgrade instead
maintains a persistent LOCAL MAP of 3D landmarks in a stable frame (this
tracker's own odometry origin -- the same "first frame is identity"
convention as the F2F tracker, so it drops into the pipeline exactly
the same way) and solves PnP directly against that map every frame.

Two structural differences from F2F, both expected (not yet proven on
hardware) to help specifically with what WP-A3's baseline runs found:
  1. Landmarks persist across many keyframes, not just one hop, so a
     frame with weak overlap against the IMMEDIATELY PREVIOUS keyframe
     (a long straight corridor is the textbook case -- see
     WP_A3_Findings.md) can still track against older-but-still-visible
     landmarks.
  2. PnP against the local map gives an ABSOLUTE pose within this
     tracker's frame directly (world<-cam = inverse(T_cam_world) from a
     single PnP solve), not a running composition of per-hop deltas --
     one fewer place for error to compound between keyframes.

CONVENTION WARNING (read before touching the PnP call below): this
module's entire reason for existing is more accuracy, and the Phase 1
plan's finding F1 was a silently-inverted PnP result that passed every
existing test because the fixtures were flat and yaw-only. Before this
file's pose convention is trusted, it is checked by
test_local_map_odometry_direction_convention in tests/gates/test_g0.py,
which uses the same non-planar, non-yaw-only oracle pattern as the F2F
convention check and is verified RED (>10cm) against an early version
of this file that had objectPoints/imagePoints swapped, GREEN (<15mm)
against the version below.

Deliberate simplifications vs a full RTAB-Map-parity F2M tracker
(documented, not hidden -- see the Phase 1 plan's WP-B1 section for
what a fuller version would add):
  - No motion-only Gauss-Newton refinement after RANSAC; the RANSAC
    inlier solution is used directly. A refinement pass would tighten
    the pose further but the convention risk and testing cost of a
    hand-rolled GN solver did not fit this session's budget.
  - No sliding-window local bundle adjustment.
  - Landmark de-duplication is a simple "map already has points here"
    density skip, not a proper occupancy grid.

VALIDATION STATUS (read before enabling odometry_backend="f2m"):
the PnP direction convention is verified correct (see the oracle gate
above, and re-verified after the bug described below). END-TO-END
PERFORMANCE IS NOT YET AT PARITY WITH F2F and this is NOT the default
(cfg.odometry_backend="f2f" remains default in pipeline.py). Two things
found while validating this against the frozen WP-A3 baseline, both
worth reading before extending this module:

  1. A real bug (fixed): _match_to_map's guided-vs-unguided fallback
     treated "_match_guided ran but found zero matches" the same as
     "guided matching succeeded", so the unguided fallback never fired
     when guided matching starved. Confirmed via direct trace: frame 1
     of square6dof went from 0 guided matches (bug) to 397 unguided
     matches (correct) once fixed.
  2. A structural gap (NOT fixed, scoped out): landmarks are inserted
     into the map using pose_world_cam AT INSERTION TIME and never
     corrected afterward. Any pose error present when a keyframe adds
     landmarks is baked into their world-frame position permanently.
     Traced directly on square6dof: PnP inlier ratio collapsed to 17%
     by frame 28 (of 140) purely from accumulated landmark-placement
     error, well under the f2m_min_inlier_ratio=0.35 plausibility gate
     -- the gate is doing its job; the map itself has degraded. F2F's
     simpler "reset the reference every keyframe" approach doesn't have
     this failure mode (each hop's error doesn't compound into a
     standing map), which is why F2F currently outperforms this file
     end-to-end despite F2M's better theoretical resistance to
     short-range aliasing. The standard fix is local bundle adjustment
     (jointly re-optimising recent keyframe poses AND landmark
     positions over a sliding window) -- exactly the "local BA, kept
     only if it improves RPE by >=20%" item the Phase 1 plan's WP-B1
     section already flagged as optional/time-permitting. It didn't fit
     this session's remaining budget. Until it's added, prefer f2f.
"""
from __future__ import annotations
from typing import Optional
from collections import deque
import numpy as np
import cv2

from pyslam.core.types import Signature, Frame, OdomResult
from pyslam.core.config import Config
from pyslam.core import lie
from pyslam.core.pnp import robust_pnp_flag
from pyslam.core.pnp_info import pnp_info_matrix
from pyslam.frontend.local_ba import KFRecord, local_bundle_adjust

# WP-B1 continued: vectorised Hamming distance for landmark fusion (see
# _add_landmarks). Precomputed popcount-per-byte table, same trick as
# any bitwise-Hamming implementation; avoids the much slower
# np.unpackbits-then-sum path used for one-off diagnostics.
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_pairs(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """desc_a, desc_b: (N,32) uint8, ALREADY PAIRED (row i vs row i, not
    all-pairs) -- caller is responsible for building the pair list, since
    the whole point of this function's caller is to only compute Hamming
    on a short-listed set of pairs, not the full n_map x n_cand cross
    product. Returns (N,) int distances in bits."""
    x = np.bitwise_xor(desc_a, desc_b)
    return _POPCOUNT_TABLE[x].sum(axis=1).astype(np.int32)


class _Landmark:
    __slots__ = ("pos", "desc", "last_seen", "n_obs")

    def __init__(self, pos: np.ndarray, desc: np.ndarray, frame_idx: int):
        self.pos = pos            # (3,) float64, in THIS TRACKER'S world frame
        self.desc = desc          # (32,) uint8
        self.last_seen = frame_idx
        self.n_obs = 1


class LocalMapOdometry:
    """Drop-in replacement for VisualOdometry with the same public
    interface (update/force_new_keyframe/status/cur_pose/
    last_keyframe_*), so pipeline.py can select either via config
    without other changes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.cur_pose: np.ndarray = np.eye(4)   # world<-cam, world = this tracker's own origin
        self._prev_pose: np.ndarray = np.eye(4)
        self._velocity: np.ndarray = np.zeros(6)  # constant-velocity tangent-space prediction

        self.map: dict[int, _Landmark] = {}
        self._next_lm_id = 0
        self._frame_idx = 0

        # WP-B1 finish: sliding-window local bundle adjustment (see
        # local_ba.py). Each entry is one keyframe's (pose, observations
        # against landmarks that already existed in the map BEFORE that
        # keyframe added its own new ones) -- see the comment at its
        # append site in update() for why only pre-existing-landmark
        # observations are recorded.
        self._window: deque = deque(maxlen=cfg.f2m_ba_window)

        self.last_keyframe_created: bool = False
        self.last_keyframe_signature: Optional[Signature] = None
        self.last_keyframe_link_T: Optional[np.ndarray] = None
        self.last_keyframe_link_info: Optional[np.ndarray] = None
        self.last_keyframe_link_inliers: int = 0

        self._last_kf_pose: np.ndarray = np.eye(4)
        self._lost_streak = 0
        self.status = "OK"
        self._initialised = False

    # ------------------------------------------------------------------ matching
    def _match_unguided(self, sig: Signature, lm_ids: list, lm_pos: np.ndarray,
                         lm_desc: np.ndarray) -> tuple[np.ndarray, np.ndarray, list, list]:
        """Brute-force descriptor match against the WHOLE local map --
        used on the first tracked frame after a reset, or as a fallback
        when guided matching starves."""
        knn = self.matcher.knnMatch(sig.desc, lm_desc, k=2)
        q_idx, m_idx = [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.8 * n.distance:
                q_idx.append(m.queryIdx)
                m_idx.append(m.trainIdx)
        if not q_idx:
            return np.zeros((0, 3)), np.zeros((0, 2)), [], []
        obj_pts = lm_pos[np.array(m_idx)]
        img_pts = sig.kp[np.array(q_idx)]
        matched_ids = [lm_ids[i] for i in m_idx]
        return obj_pts, img_pts, matched_ids, list(q_idx)

    def _match_guided(self, sig: Signature, predicted_pose: np.ndarray, K: np.ndarray,
                       lm_ids: list, lm_pos: np.ndarray, lm_desc: np.ndarray) -> Optional[tuple]:
        """Project the local map into `predicted_pose` and only match
        descriptors within a pixel window of their projection. Cheaper
        than brute-force once the map is large, and rejects far more
        false matches by construction (two visually-similar-but-wrong
        points rarely project to the same few-pixel window). Returns
        None if too few landmarks project into the frame at all, so the
        caller can fall back to unguided matching."""
        T_cam_world = lie.se3_inverse(predicted_pose)
        R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
        p_cam = (R @ lm_pos.T).T + t
        in_front = p_cam[:, 2] > 1e-3
        if not np.any(in_front):
            return None
        px = (K @ p_cam[in_front].T).T
        uv_front = px[:, :2] / px[:, 2:3]
        h, w = (sig.rgb.shape[0], sig.rgb.shape[1]) if sig.rgb is not None else (480, 640)
        in_frame = (uv_front[:, 0] >= 0) & (uv_front[:, 0] < w) & (uv_front[:, 1] >= 0) & (uv_front[:, 1] < h)
        sel = np.where(in_front)[0][in_frame]           # indices into lm_ids/lm_pos/lm_desc
        sel_uv = uv_front[in_frame]                      # projected pixel for each selected landmark
        if len(sel) < self.cfg.odom_min_inliers:
            return None

        window_px = self.cfg.f2m_guided_window_px
        sub_desc = lm_desc[sel]
        knn = self.matcher.knnMatch(sig.desc, sub_desc, k=2)
        q_idx, m_idx = [], []
        for qi, pair in enumerate(knn):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= 0.8 * n.distance:
                continue
            cand_uv = sel_uv[m.trainIdx]
            query_uv = sig.kp[qi]
            if np.linalg.norm(cand_uv - query_uv) <= window_px:
                q_idx.append(qi)
                m_idx.append(sel[m.trainIdx])
        if not q_idx:
            return np.zeros((0, 3)), np.zeros((0, 2)), [], []
        obj_pts = lm_pos[np.array(m_idx)]
        img_pts = sig.kp[np.array(q_idx)]
        matched_ids = [lm_ids[i] for i in m_idx]
        return obj_pts, img_pts, matched_ids, list(q_idx)

    def _match_to_map(self, sig: Signature, predicted_pose: np.ndarray, K: np.ndarray,
                       guided: bool) -> tuple[np.ndarray, np.ndarray, list, list]:
        if not self.map or sig.desc.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros((0, 2)), [], []
        lm_ids = list(self.map.keys())
        lm_pos = np.array([self.map[i].pos for i in lm_ids])
        lm_desc = np.array([self.map[i].desc for i in lm_ids])

        if guided:
            result = self._match_guided(sig, predicted_pose, K, lm_ids, lm_pos, lm_desc)
            # Bug found while validating this against the frozen WP-A3
            # baseline (square6dof went to 17/140 LOST with 0 inliers on
            # nearly every frame before this fix): _match_guided can
            # legitimately return a non-None but EMPTY result (its own
            # "if not q_idx: return zeros" branch), which is different
            # from "too few landmarks project into frame" (which returns
            # None). Treating "guided ran but found nothing" as success
            # meant unguided fallback below never fired. Fall back
            # whenever the guided count is below the usable floor,
            # not only when guided returned None outright.
            if result is not None and result[0].shape[0] >= self.cfg.odom_min_inliers:
                return result
        return self._match_unguided(sig, lm_ids, lm_pos, lm_desc)

    def _pnp(self, obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, reproj_px: float):
        if obj_pts.shape[0] < 6:
            return None
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts.astype(np.float64), img_pts.astype(np.float64), K, None,
                reprojectionError=reproj_px, confidence=0.999, iterationsCount=300,
                flags=robust_pnp_flag(),
            )
        except cv2.error:
            return None
        if not ok or inliers is None or len(inliers) < self.cfg.odom_min_inliers:
            return None
        R, _ = cv2.Rodrigues(rvec)
        # solvePnPRansac(objectPoints=WORLD-frame landmarks, imagePoints=
        # current frame pixels) returns the pose that maps object (world)
        # points into the camera that observed imagePoints -- OpenCV's
        # documented convention, p_cam = R @ p_world + t. That is
        # T_cam_world (camera<-world), NOT world<-camera. This is the
        # exact class of error the Phase 1 plan's F1 finding was about;
        # see this module's docstring for the convention-oracle gate
        # that specifically checks this line hasn't been silently
        # inverted.
        T_cam_world = lie.make_T(R, tvec.reshape(3))
        T_world_cam = lie.se3_inverse(T_cam_world)
        return T_world_cam, inliers.reshape(-1)

    def _plausibility_gate(self, T_world_cam: np.ndarray, n_inliers: int, n_matches: int,
                            img_pts_inlier: np.ndarray, img_shape: tuple) -> bool:
        """WP-B3 plausibility gates: reject an accepted-by-RANSAC pose
        that is nonetheless implausible, rather than trusting RANSAC's
        inlier count alone (the Phase 1 plan's F3 finding: a
        near-degenerate 17-inlier solve at a fixture discontinuity was
        accepted by Phase 0's F2F tracker with no further scrutiny)."""
        inlier_ratio = n_inliers / max(n_matches, 1)
        if inlier_ratio < self.cfg.f2m_min_inlier_ratio:
            return False
        h, w = img_shape[0], img_shape[1]
        spread_x = (img_pts_inlier[:, 0].max() - img_pts_inlier[:, 0].min()) / max(w, 1)
        spread_y = (img_pts_inlier[:, 1].max() - img_pts_inlier[:, 1].min()) / max(h, 1)
        if max(spread_x, spread_y) < self.cfg.f2m_min_spatial_spread:
            return False
        if self._frame_idx > 3:
            # WP-B1 continued: use self.cur_pose (this call's "last known
            # good pose", not yet overwritten -- see update()'s own fix
            # below for why self._prev_pose is the wrong anchor here).
            predicted = self.cur_pose @ lie.se3_exp(self._velocity)
            err = lie.se3_log(lie.se3_inverse(predicted) @ T_world_cam)
            recent_speed = np.linalg.norm(self._velocity[:3]) + 1e-3
            if np.linalg.norm(err[:3]) > self.cfg.f2m_motion_gate_factor * max(recent_speed, 0.02):
                return False
        return True

    # ------------------------------------------------------------------
    def update(self, sig: Signature, frame: Frame) -> OdomResult:
        self.last_keyframe_created = False
        K = frame.intr.K()
        self._frame_idx += 1

        if not self._initialised:
            self.cur_pose = np.eye(4)
            self._prev_pose = np.eye(4)
            self._add_landmarks(sig, np.eye(4))
            self._initialised = True
            self.last_keyframe_created = True
            self.last_keyframe_signature = sig
            self.last_keyframe_link_T = None
            self._last_kf_pose = np.eye(4)
            self.status = "OK"
            self._lost_streak = 0
            return OdomResult(pose=self.cur_pose.copy(), T_rel=np.eye(4),
                               info=np.eye(6) * 1e6, n_inliers=0, status="OK")

        predicted_pose = self.cur_pose @ lie.se3_exp(self._velocity)
        obj_pts, img_pts, matched_ids, kp_idx = self._match_to_map(sig, predicted_pose, K, guided=True)
        if obj_pts.shape[0] < self.cfg.odom_min_inliers:
            return self._handle_lost()

        result = self._pnp(obj_pts, img_pts, K, self.cfg.odom_reproj_px)
        if result is None:
            return self._handle_lost()
        T_world_cam, inlier_idx = result
        n_inliers = len(inlier_idx)
        n_matches = obj_pts.shape[0]
        img_shape = sig.rgb.shape if sig.rgb is not None else (480, 640, 3)

        if not self._plausibility_gate(T_world_cam, n_inliers, n_matches, img_pts[inlier_idx], img_shape):
            return self._handle_lost()

        self._lost_streak = 0
        self.status = "OK"

        # WP-B1 continued -- velocity off-by-one-frame bug (found by
        # direct instrumentation: median |velocity|/|true per-frame
        # step| = 2.057 across square6dof, i.e. velocity was
        # consistently spanning TWO frame intervals, not one).
        #
        # Root cause: self.cur_pose still holds THIS call's "previous
        # frame" pose at this point (it isn't reassigned until the line
        # below) -- so it's exactly the right anchor for "relative
        # motion since the last frame". self._prev_pose, however, was
        # last written at the END of the PREVIOUS call, to THAT call's
        # cur_pose-before-reassignment -- i.e. it lags self.cur_pose by
        # one additional frame. Using it here silently computed the
        # relative motion across two frame intervals every time, which
        # is why every downstream consumer of self._velocity (the
        # constant-velocity guided-matching prediction above, and
        # _plausibility_gate's own motion-gate check) was working from
        # a prediction roughly 2x too large -- directly explaining the
        # 68/140 unguided-matching-fallback rate measured before this
        # fix (guided matching's 25px window was, in effect, always
        # aimed a full frame-step past where the landmarks actually
        # projected).
        self._velocity = lie.se3_log(lie.se3_inverse(self.cur_pose) @ T_world_cam)
        self._prev_pose = self.cur_pose.copy()  # kept in sync for any
            # external reader; no longer used for velocity/prediction
            # math above, both of which now anchor on self.cur_pose
            # directly (see _plausibility_gate's matching fix).
        self.cur_pose = T_world_cam

        for i in inlier_idx:
            lm_id = matched_ids[i]
            if lm_id in self.map:
                self.map[lm_id].last_seen = self._frame_idx
                self.map[lm_id].n_obs += 1

        info = np.eye(6) * (n_inliers / max(n_matches, 1)) * 100.0
        if self.cfg.use_hessian_info:
            # WP-P4: see odometry.py's equivalent note. T_world_cam here
            # is ALREADY inverted (obj=world) -- _pnp() does the
            # inversion internally (see its own convention comment) --
            # so pnp_info_matrix needs the RAW (pre-inversion) R,t back;
            # recovered by inverting T_world_cam once more rather than
            # threading a second return value through _pnp().
            T_cam_world = lie.se3_inverse(T_world_cam)
            info = pnp_info_matrix(obj_pts[inlier_idx], T_cam_world[:3, :3], T_cam_world[:3, 3],
                                    K, self.cfg.pnp_sigma_px)

        T_rel_since_kf = lie.se3_inverse(self._last_kf_pose) @ self.cur_pose
        xi = lie.se3_log(T_rel_since_kf)
        trans = np.linalg.norm(xi[:3])
        rot_deg = np.degrees(np.linalg.norm(xi[3:]))
        inlier_ratio = n_inliers / max(n_matches, 1)

        need_new_kf = (
            trans > self.cfg.keyframe_trans_m or
            rot_deg > self.cfg.keyframe_rot_deg or
            n_inliers < self.cfg.keyframe_min_inliers or
            inlier_ratio < self.cfg.f2m_keyframe_coverage_floor
        )
        if need_new_kf:
            # WP-B1 finish: record this keyframe's re-observations of
            # EXISTING map landmarks (matched_ids/img_pts/inlier_idx were
            # already computed above, before this frame added anything
            # new to the map) as this window entry's observations, then
            # run sliding-window local BA once at least 2 keyframes are
            # in the window. This is what fixes the structural gap this
            # module's docstring describes: landmark positions get
            # jointly re-optimised against real 2D observations instead
            # of being frozen forever at their insertion-time estimate.
            prev_kf_pose = self._last_kf_pose  # anchor for this keyframe's relative link
            obs_list = [(matched_ids[i], img_pts[i].copy()) for i in inlier_idx
                        if matched_ids[i] in self.map]
            self._window.append(KFRecord(pose=self.cur_pose.copy(), obs=obs_list))

            self.last_keyframe_created = True
            self.last_keyframe_signature = sig
            self.last_keyframe_link_inliers = n_inliers
            matched_kp_idx = {kp_idx[i] for i in inlier_idx}
            self._add_landmarks(sig, self.cur_pose, exclude_kp_idx=matched_kp_idx)

            if len(self._window) >= 2:
                K3 = K if K.shape == (3, 3) else K[:3, :3]
                landmark_pos = {lm_id: lm.pos for lm_id, lm in self.map.items()}
                refined_poses, refined_lm = local_bundle_adjust(
                    list(self._window), landmark_pos, K3,
                    huber_px=self.cfg.odom_reproj_px,
                )
                for i, kf in enumerate(self._window):
                    kf.pose = refined_poses[i]
                for lm_id, pos in refined_lm.items():
                    if lm_id in self.map:
                        self.map[lm_id].pos = pos
                self.cur_pose = self._window[-1].pose.copy()

            self._last_kf_pose = self.cur_pose.copy()
            T_rel_since_kf = lie.se3_inverse(prev_kf_pose) @ self.cur_pose
            self.last_keyframe_link_T = T_rel_since_kf.copy()
            self.last_keyframe_link_info = info.copy()
            self._prune_map()

        return OdomResult(pose=self.cur_pose.copy(), T_rel=T_rel_since_kf, info=info,
                           n_inliers=n_inliers, status="OK")

    def _add_landmarks(self, sig: Signature, pose_world_cam: np.ndarray,
                        exclude_kp_idx: Optional[set] = None) -> None:
        """Adds depth-valid keypoints NOT already matched to an existing
        landmark this frame (see `exclude_kp_idx`). Relies on
        `_prune_map()` (always called right after this, by every caller)
        to bring the total back down to `cfg.f2m_max_landmarks` --
        insertion here is NOT itself gated on current map size. See the
        "PERMANENT-FREEZE BUG" note below for why that matters.

        `exclude_kp_idx` fix (found while validating WP-B1's local BA
        against the real pipeline): this module's docstring already
        claimed "a simple 'map already has points here' density skip"
        existed, but no such check was actually implemented -- every
        valid keypoint got inserted as a brand-new landmark at every
        keyframe, including keypoints that had just been matched to an
        existing landmark that same frame. A real bug, fixed here by
        skipping the caller-supplied matched keypoint indices. NOTE this
        fix's ORIGINAL commit message overclaimed it as the explanation
        for square6dof's collapse -- it measurably was NOT (see the
        PERMANENT-FREEZE BUG note below, found afterward by more precise
        measurement); this fix is real and worth keeping on its own
        merits (it stops wasteful duplicate insertion) but is not what
        was causing the collapse.

        PERMANENT-FREEZE BUG (the actual cause of square6dof's collapse,
        found by direct instrumentation after eviction-order, this
        de-dup fix, and cap-size sweeps all independently failed to
        explain or resolve it -- see WP_B1_Findings.md for the full
        trail): this method used to gate its own insertion loop with
        `if len(self.map) >= cap: break`. Once the map first reached
        EXACTLY the cap (which happens fast -- square6dof hit it by
        frame 10 of 140), that condition is true on the very FIRST
        candidate keypoint, so NOTHING more ever gets inserted, ever
        again, for the rest of the run -- and since `_prune_map()` only
        evicts when OVER budget, and insertion here now never pushes the
        map over budget (it stops exactly at it), prune never fires
        either. The map doesn't churn too aggressively, as originally
        (wrongly) suspected; it does the opposite -- it freezes
        completely, permanently, the instant it first fills up, however
        far the camera still has left to travel. Confirmed directly:
        `_next_lm_id` (the monotonic landmark-creation counter) stopped
        incrementing entirely at frame 10 and never moved again through
        frame 140. This also fully explains why the three earlier
        targeted fixes each had no effect: eviction-order tuning changes
        nothing if eviction never runs; de-duplication changes nothing
        if insertion had already permanently stopped regardless; and
        raising the cap only delays exactly this same one-time freeze
        rather than preventing it (consistent with the observed "bigger
        cap survives a bit longer, still eventually collapses" pattern).
        The fix: don't gate insertion on current occupancy at all --
        `_prune_map()`, which every caller already invokes immediately
        after this method, is what's supposed to enforce the budget, by
        evicting the worst EXISTING landmarks to make room for
        legitimately new ones. That's now what actually happens.

        WP-B1 CONTINUED -- landmark fusion (the actual accuracy-gap fix,
        see WP_B1_Findings.md's diagnostic trace): even with the freeze
        bug and the de-dup-by-kp-index fix above both in place, a
        directly-instrumented trace of this method still found 30,817
        landmarks created over a 140-frame run against a 2000 cap --
        i.e. the map was turning over roughly 15 TIMES per run, holding
        only 2-3 keyframes' worth of history at any moment, which
        defeats F2M's entire reason for existing (persistent landmarks
        across MANY keyframes, not just the immediately-preceding one --
        see this module's top docstring). A sampled census of the final
        map found 100% of landmarks had at least one near-duplicate
        (same physical point within 3cm, descriptor Hamming <40)
        already resident. The cause: `exclude_kp_idx` only excludes
        keypoints that were successfully MATCHED to an existing
        landmark this frame via `_match_guided`/`_match_unguided`'s
        Lowe-ratio (0.8) test -- and that test itself starves when the
        map already contains near-duplicates of the same point (their
        distances to a query descriptor are nearly equal, so the ratio
        test correctly refuses to pick one), which is exactly the
        failure this fix is closing -- a self-reinforcing loop: failed
        matching produces more duplicate insertion, which produces more
        failed matching.

        The fix: before inserting a candidate keypoint as a brand-new
        landmark, check whether it corresponds to an EXISTING map
        landmark that this frame's own matching simply failed to
        associate, rather than assuming "not matched this frame" means
        "not in the map". Spatial pre-filter first (cheap: candidate
        3D positions vs existing landmark positions, `f2m_fusion_radius_m`
        default 5cm) narrows to plausible pairs; only those get the
        (per-pair) Hamming check (`f2m_fusion_hamming_max`, default 50
        bits, deliberately TIGHTER than P3's cross-frame vocabulary
        threshold of 70 -- fusion is the stronger claim "this IS that
        point", not "these should share a vocabulary word"). Matched
        pairs are resolved greedily, best (lowest Hamming, then
        distance) first, each map landmark and each candidate used at
        most once per call.

        On a fuse (not a fresh insert): position is updated via a
        running mean weighted by the landmark's existing `n_obs` (so an
        established, many-times-observed landmark's position moves
        less per new observation than a newly-created one's would --
        the standard incremental-mean behaviour, not a separate
        weighting scheme). `n_obs` and `last_seen` are updated exactly
        as a successful MATCH would update them (see the `for i in
        inlier_idx` loop in `update()`), so a fused landmark is
        indistinguishable downstream from one that was matched
        directly. The descriptor is DELIBERATELY NOT updated on fuse --
        kept from the landmark's original creation. Averaging binary
        ORB descriptors isn't well-defined, and always keeping the
        newest descriptor risks drifting away from whichever appearance
        BA's shared-landmark constraints were built against; documented
        simplification, not a claim that a smarter update policy
        couldn't do better (candidate follow-up, not attempted here).

        This is a real behaviour change, not just a bookkeeping tweak --
        it changes what `self.map` contains after every keyframe, so it
        is validated by re-running the same duplicate-census diagnostic
        this docstring cites (before: 100% of sampled landmarks had
        >=1 duplicate; after: see WP_B1_Findings.md's continued section)
        rather than trusted from the reasoning above alone.
        """
        R, t = pose_world_cam[:3, :3], pose_world_cam[:3, 3]
        valid_idx = np.where(sig.valid)[0]
        exclude_kp_idx = exclude_kp_idx or set()
        cand_idx = np.array([i for i in valid_idx if i not in exclude_kp_idx], dtype=np.int64)
        if cand_idx.size == 0:
            return
        p_world = (R @ sig.kp3d[cand_idx].T).T + t   # (M,3)
        cand_desc = sig.desc[cand_idx]                # (M,32) uint8

        fused_cand = set()
        if self.map:
            lm_ids = np.array(list(self.map.keys()))
            lm_pos = np.array([self.map[i].pos for i in lm_ids])
            lm_desc = np.array([self.map[i].desc for i in lm_ids])

            # Spatial pre-filter: O(n_map x M) distances, cheap relative
            # to a full n_map x M Hamming check (which this filter
            # exists specifically to avoid computing in full).
            d = np.linalg.norm(lm_pos[:, None, :] - p_world[None, :, :], axis=2)
            map_rows, cand_cols = np.where(d < self.cfg.f2m_fusion_radius_m)

            if map_rows.size > 0:
                ham = _hamming_pairs(lm_desc[map_rows], cand_desc[cand_cols])
                keep = ham <= self.cfg.f2m_fusion_hamming_max
                map_rows, cand_cols, ham = map_rows[keep], cand_cols[keep], ham[keep]
                dist = d[map_rows, cand_cols]

                # Greedy assignment: best (tightest Hamming, then
                # closest) pair wins first; each landmark and each
                # candidate can only be consumed once. A landmark
                # already fused this call cannot also absorb a second
                # candidate (and vice versa) -- prevents two distinct
                # nearby keypoints both silently collapsing onto one
                # landmark id.
                order = np.lexsort((dist, ham))
                used_map, used_cand = set(), set()
                for k in order:
                    mrow, ccol = int(map_rows[k]), int(cand_cols[k])
                    if mrow in used_map or ccol in used_cand:
                        continue
                    used_map.add(mrow)
                    used_cand.add(ccol)
                    lm = self.map[lm_ids[mrow]]
                    n = lm.n_obs
                    lm.pos = (lm.pos * n + p_world[ccol]) / (n + 1)
                    lm.n_obs = n + 1
                    lm.last_seen = self._frame_idx
                    fused_cand.add(ccol)

        # WP-B1 continued -- intra-batch dedup: fusion above only checks
        # new candidates against the EXISTING map, not against each
        # OTHER. A single frame with a dense feature set (863 valid
        # keypoints on the very first frame, before the map has
        # anything to fuse against) can itself contain two grid-bucketed
        # ORB detections that land within a few cm of each other on the
        # same physical surface -- re-measured directly after the
        # existing-map fusion above alone: total landmarks created did
        # drop substantially (30,817 -> 15,813 over the same 140-frame
        # run) but a sampled duplicate census still found ~100% of final
        # landmarks had >=1 near-duplicate, just fewer per landmark on
        # average (1.5 -> 1.2) -- consistent with most remaining
        # duplication now coming from WITHIN single insertion batches
        # rather than across them. Same radius/Hamming thresholds,
        # applied within `remaining` (the candidates NOT already fused
        # into an existing landmark above) before they're inserted.
        remaining = [j for j in range(cand_idx.shape[0]) if j not in fused_cand]
        skip_local: set = set()
        if len(remaining) > 1:
            ridx = np.array(remaining)
            rp = p_world[ridx].copy()
            rd = cand_desc[ridx]
            dd = np.linalg.norm(rp[:, None, :] - rp[None, :, :], axis=2)
            iu, ju = np.triu_indices(len(ridx), k=1)
            close = dd[iu, ju] < self.cfg.f2m_fusion_radius_m
            ci, cj = iu[close], ju[close]
            if ci.size:
                ham = _hamming_pairs(rd[ci], rd[cj])
                keep = ham <= self.cfg.f2m_fusion_hamming_max
                ci, cj, ham = ci[keep], cj[keep], ham[keep]
                dist = dd[ci, cj]
                order = np.lexsort((dist, ham))
                for k in order:
                    a, b = int(ci[k]), int(cj[k])
                    if a in skip_local or b in skip_local:
                        continue
                    rp[a] = (rp[a] + rp[b]) / 2.0  # absorb b into representative a
                    skip_local.add(b)
            for local_i, j in enumerate(remaining):
                if local_i in skip_local:
                    continue
                self.map[self._next_lm_id] = _Landmark(rp[local_i], rd[local_i].copy(), self._frame_idx)
                self._next_lm_id += 1
        else:
            for j in remaining:
                self.map[self._next_lm_id] = _Landmark(p_world[j], cand_desc[j].copy(), self._frame_idx)
                self._next_lm_id += 1

    def _prune_map(self) -> None:
        """Cap total landmark count (RTAB-Map's own F2M default is
        ~2000): drop the least-observed, least-recently-seen points
        first when over budget.

        Sort key: (last_seen, n_obs) -- recency PRIMARY. Re-tested
        directly after fixing the permanent-freeze bug in
        _add_landmarks (see that method's docstring): with the freeze
        bug active, eviction essentially never ran at all (insertion
        stopped before the map could ever exceed budget), so an earlier
        test of (n_obs, last_seen) vs (last_seen, n_obs) ordering showed
        no difference -- it wasn't really exercising eviction at all.
        With the freeze bug fixed, eviction now genuinely runs every
        keyframe once at cap, and ordering matters: a single large prune
        event (hundreds of landmarks evicted in one shot, immediately
        after a keyframe's own hundreds of new insertions) was observed
        to zero out next-frame matching outright when sorted
        (n_obs, last_seen) -- a landmark's last_seen is set to the
        CURRENT frame at insertion (see _Landmark.__init__), but n_obs
        as the PRIMARY key ignores that entirely, so brand-new,
        currently-still-visible landmarks (n_obs=1, maximally recent)
        get evicted on equal footing with genuinely stale ones (n_obs=1,
        long-unseen) whenever both exist in the same low-n_obs tier --
        which is most of the map, most of the time, since re-observation
        takes several frames to accumulate. Sorting by last_seen first
        fixes this: anything just inserted is automatically among the
        most-recently-seen and survives; only landmarks that have
        genuinely stopped being re-matched become eviction-eligible.
        n_obs remains a tiebreaker among equally-stale entries.
        """
        if len(self.map) <= self.cfg.f2m_max_landmarks:
            return
        items = sorted(self.map.items(), key=lambda kv: (kv[1].last_seen, kv[1].n_obs))
        n_drop = len(self.map) - self.cfg.f2m_max_landmarks
        for lm_id, _ in items[:n_drop]:
            del self.map[lm_id]

    def _handle_lost(self) -> OdomResult:
        self._lost_streak += 1
        if self._lost_streak >= self.cfg.lost_consecutive_frames:
            self.status = "LOST"
        return OdomResult(pose=self.cur_pose.copy(), T_rel=np.eye(4),
                           info=np.eye(6) * 1e-6, n_inliers=0, status=self.status)

    def force_new_keyframe(self, sig: Signature) -> None:
        """LOST recovery: reset the local map to whatever's visible right
        now, at the current (unreliable, but the only estimate we have)
        pose. Matches VisualOdometry's Phase 0 behaviour -- global
        relocalisation against WM stays a later-phase upgrade."""
        self.map.clear()
        self._window.clear()  # WP-B1: don't bundle pre-LOST keyframes
        # (whose landmarks are all gone) with the fresh post-recovery map.
        self._add_landmarks(sig, self.cur_pose)
        self._prune_map()  # safety: map is empty beforehand so this is
        # normally a no-op, but keeps every _add_landmarks call site
        # consistent regardless of n_features/cap configuration.
        self._velocity = np.zeros(6)
        self._prev_pose = self.cur_pose.copy()
        self._lost_streak = 0
        self.status = "OK"
        self.last_keyframe_created = True
        self.last_keyframe_signature = sig
        self.last_keyframe_link_T = None
        self._last_kf_pose = self.cur_pose.copy()
