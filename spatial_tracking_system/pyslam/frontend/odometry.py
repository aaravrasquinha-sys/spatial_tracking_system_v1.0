"""
Phase-0 odometry: frame-to-KEYFRAME PnP tracking (not full frame-to-map;
that upgrade is Phase 1). Every incoming frame is matched against the most
recent keyframe's 3D points via PnP+RANSAC. When the tracked motion or
match quality crosses a threshold, the current frame is promoted to a new
keyframe and an odometry Link is emitted between consecutive keyframes.

This is deliberately the simplest thing that (a) doesn't drift
catastrophically over short sequences and (b) has an honest failure mode
(LOST) rather than silently producing garbage.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

from pyslam.core.types import Signature, Frame, OdomResult
from pyslam.core.config import Config
from pyslam.core import lie
from pyslam.core.pnp import robust_pnp_flag
from pyslam.core.pnp_info import pnp_info_matrix


class VisualOdometry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.ref_sig: Optional[Signature] = None
        self.ref_pose: np.ndarray = np.eye(4)   # world<-cam pose of the reference keyframe
        self.cur_pose: np.ndarray = np.eye(4)   # world<-cam pose of the latest tracked frame

        self.last_keyframe_created: bool = False
        self.last_keyframe_reason: Optional[str] = None  # WP-L3: why the newest keyframe was created
        self.last_keyframe_signature: Optional[Signature] = None
        self.last_keyframe_link_T: Optional[np.ndarray] = None   # T_prevkf_newkf
        self.last_keyframe_link_info: Optional[np.ndarray] = None
        self.last_keyframe_link_inliers: int = 0

        self._lost_streak = 0
        self.status = "OK"
        self._initialised = False

    # ------------------------------------------------------------------
    def _match(self, sig_query: Signature, sig_train: Signature, T_train_query_predicted=None, K=None):
        """WP-LIVE 4.7: optional prediction-gated matching. When a
        predicted relative pose is available (constant-velocity or
        gyro-integrated, same sources pyslam/imu/bridge.py already
        uses), matches whose train-side 3D point reprojects far from
        its actual query-side 2D location under that prediction are
        discarded BEFORE being handed to PnP+RANSAC -- raising the
        inlier ratio RANSAC sees, which WP-L's own findings tie
        directly to fewer inlier-triggered keyframes (102/179 on
        corridor_v2) and fewer LOST events. Purely additive: with
        T_train_query_predicted=None (the default, unchanged call sites)
        this is bit-identical to the original ratio-test-only matcher.
        """
        if sig_query.desc.shape[0] == 0 or sig_train.desc.shape[0] == 0:
            return []
        train_valid_idx = np.where(sig_train.valid)[0]
        if train_valid_idx.size == 0:
            return []
        train_desc = sig_train.desc[train_valid_idx]
        knn = self.matcher.knnMatch(sig_query.desc, train_desc, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.8 * n.distance:
                good.append((m.queryIdx, train_valid_idx[m.trainIdx]))
        if T_train_query_predicted is None or K is None or len(good) == 0:
            return good
        return self._prediction_gate(good, sig_query, sig_train, T_train_query_predicted, K)

    def _prediction_gate(self, matches, sig_query, sig_train, T_train_query, K,
                          max_reproj_px: float = None):
        """Reprojects each train-side 3D point (train camera frame)
        into the QUERY image using the predicted train<-query transform
        inverted, and drops matches whose predicted pixel is far from
        the actual matched query pixel. max_reproj_px defaults to
        4x this Config's own odom_reproj_px -- generous on purpose:
        this is a PRE-filter to help RANSAC, not a second RANSAC: a
        genuinely bad prediction must never be able to silently starve
        the match set down to nothing, so the gate degrades to a no-op
        rather than making things worse when it can't help (see the
        fallback below)."""
        if max_reproj_px is None:
            max_reproj_px = 4.0 * self.cfg.odom_reproj_px
        T_query_train = lie.se3_inverse(T_train_query)
        q_idx = np.array([m[0] for m in matches])
        t_idx = np.array([m[1] for m in matches])
        obj_pts_train = sig_train.kp3d[t_idx]
        pred_pts_query = lie.transform_points(T_query_train, obj_pts_train)
        z = pred_pts_query[:, 2]
        in_front = z > 1e-3
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        px = np.zeros_like(z)
        py = np.zeros_like(z)
        px[in_front] = pred_pts_query[in_front, 0] / z[in_front] * fx + cx
        py[in_front] = pred_pts_query[in_front, 1] / z[in_front] * fy + cy
        actual = sig_query.kp[q_idx]
        err = np.full(len(matches), np.inf)
        err[in_front] = np.hypot(px[in_front] - actual[in_front, 0], py[in_front] - actual[in_front, 1])
        keep = err < max_reproj_px
        filtered = [m for m, k in zip(matches, keep) if k]
        # Fail open: if the prediction gate would leave too few matches
        # to even attempt PnP, trust the ORIGINAL (ungated) match set
        # instead -- a bad prediction (e.g. right after a LOST recovery,
        # before any velocity estimate exists) must degrade to Phase-0
        # behaviour, never make tracking worse than not gating at all.
        if len(filtered) < self.cfg.odom_min_inliers:
            return matches
        return filtered

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
        # WP-LIVE 4.7: refine on the full inlier set with LM. RANSAC's
        # own output is the best MINIMAL/scored sample, never refined
        # against every inlier it found -- solvePnPRefineLM does exactly
        # that (Gauss-Newton/LM over the full inlier set, seeded from
        # the RANSAC result), a few tenths of a millisecond at typical
        # inlier counts. Falls back to the unrefined RANSAC result on
        # any failure (never raises, never makes the pose WORSE than
        # not refining -- same fail-open discipline as the prediction
        # gate above).
        inlier_idx = inliers.reshape(-1)
        try:
            rvec_r, tvec_r = cv2.solvePnPRefineLM(
                obj_pts[inlier_idx].astype(np.float64), img_pts[inlier_idx].astype(np.float64),
                K, None, rvec, tvec)
            rvec, tvec = rvec_r, tvec_r
        except cv2.error:
            pass
        R, _ = cv2.Rodrigues(rvec)
        T_train_query = lie.make_T(R, tvec.reshape(3))  # maps ref-frame pts into query(cam) frame
        return T_train_query, inlier_idx

    # ------------------------------------------------------------------
    def update(self, sig: Signature, frame: Frame, T_ref_query_predicted: Optional[np.ndarray] = None
               ) -> OdomResult:
        """T_ref_query_predicted (WP-LIVE 4.7): an optional predicted
        ref<-query transform (constant-velocity extrapolation or gyro
        integration -- pyslam/imu/bridge.py's own building blocks,
        reused here by the caller, not duplicated) used ONLY to
        prefilter matches before PnP+RANSAC (see _prediction_gate).
        None (the default) reproduces the original, ungated matcher
        exactly -- existing call sites need no change."""
        self.last_keyframe_created = False
        K = frame.intr.K()

        if not self._initialised:
            self.ref_sig = sig
            self.ref_pose = np.eye(4)
            self.cur_pose = np.eye(4)
            self._initialised = True
            self.last_keyframe_created = True
            self.last_keyframe_signature = sig
            self.last_keyframe_link_T = None
            self.status = "OK"
            self._lost_streak = 0
            return OdomResult(pose=self.cur_pose.copy(), T_rel=np.eye(4),
                               info=np.eye(6) * 1e6, n_inliers=0, status="OK")

        matches = self._match(sig, self.ref_sig, T_train_query_predicted=T_ref_query_predicted, K=K)
        if len(matches) < self.cfg.odom_min_inliers:
            return self._handle_lost()

        q_idx = np.array([m[0] for m in matches])
        t_idx = np.array([m[1] for m in matches])
        obj_pts = self.ref_sig.kp3d[t_idx]          # 3D points in reference keyframe's camera frame
        img_pts = sig.kp[q_idx]

        result = self._pnp(obj_pts, img_pts, K, self.cfg.odom_reproj_px)
        if result is None:
            return self._handle_lost()
        # solvePnPRansac(objectPoints=ref-frame 3D, imagePoints=cur-frame 2D)
        # returns the pose of the OBJECT (ref) in the CAMERA (cur) frame,
        # i.e. it maps points FROM ref INTO cur: p_cur = T_cur_ref @ p_ref.
        # This is opencv's documented convention (rvec/tvec map object
        # points into the camera frame that imagePoints were observed in).
        # An earlier version of this file labelled the raw PnP output
        # T_ref_cur and composed cur_pose = ref_pose @ T_ref_cur, which
        # silently integrated the INVERSE of every motion -- see the
        # Phase 1 plan's finding F1 and the
        # test_odometry_direction_convention gate, which fails loudly
        # (>30cm err at typical keyframe spacing) whenever this ordering
        # regresses. Fixed here: invert once, at the source, and keep
        # every downstream line unchanged so it operates on the correct
        # ref<-cur transform it always assumed it had.
        T_cur_ref, inlier_idx = result               # p_cur = T_cur_ref @ p_ref  (cur<-ref)
        n_inliers = len(inlier_idx)

        self._lost_streak = 0
        self.status = "OK"

        T_ref_cur = lie.se3_inverse(T_cur_ref)      # ref<-cur, maps cur-frame pts to ref frame
        # world<-cur = world<-ref @ ref<-cur = self.ref_pose @ T_ref_cur.
        self.cur_pose = self.ref_pose @ T_ref_cur
        T_rel = T_ref_cur

        info = np.eye(6) * (n_inliers / max(len(matches), 1)) * 100.0
        if self.cfg.use_hessian_info:
            # WP-P4: real Hessian-derived info, replacing the heuristic
            # above. T_cur_ref is the raw (un-inverted) PnP output
            # (T_cam_obj, cam=cur, obj=ref); T_rel=T_ref_cur is its
            # inverse -- exactly the case pnp_info_matrix is built for
            # (info for T_obj_cam's own tangent). See pnp_info.py and
            # WP_P4_Findings.md for why this is finally being wired in
            # now (it wasn't safe to wire alone -- see WP-B2 -- until
            # verify.py's loop-link side got the matching treatment
            # below, so both sides of every optimize() call are
            # consistently scaled).
            info = pnp_info_matrix(obj_pts[inlier_idx], T_cur_ref[:3, :3], T_cur_ref[:3, 3],
                                    K, self.cfg.pnp_sigma_px)

        # -------- keyframe decision --------
        xi = lie.se3_log(T_ref_cur)
        trans = np.linalg.norm(xi[:3])
        rot_deg = np.degrees(np.linalg.norm(xi[3:]))
        inlier_ratio = n_inliers / max(len(matches), 1)

        reasons = [name for name, hit in (("trans", trans > self.cfg.keyframe_trans_m),
                                           ("rot", rot_deg > self.cfg.keyframe_rot_deg),
                                           ("inliers", n_inliers < self.cfg.keyframe_min_inliers)) if hit]
        need_new_kf = bool(reasons)
        if need_new_kf:
            self.last_keyframe_reason = "+".join(reasons)  # WP-L3 (telemetry only)
            self.last_keyframe_created = True
            self.last_keyframe_signature = sig
            self.last_keyframe_link_T = T_ref_cur.copy()
            self.last_keyframe_link_info = info.copy()
            self.last_keyframe_link_inliers = n_inliers
            self.ref_sig = sig
            self.ref_pose = self.cur_pose.copy()

        return OdomResult(pose=self.cur_pose.copy(), T_rel=T_rel, info=info,
                           n_inliers=n_inliers, status="OK")

    def _handle_lost(self) -> OdomResult:
        self._lost_streak += 1
        if self._lost_streak >= self.cfg.lost_consecutive_frames:
            self.status = "LOST"
        return OdomResult(pose=self.cur_pose.copy(), T_rel=np.eye(4),
                           info=np.eye(6) * 1e-6, n_inliers=0, status=self.status)

    def force_new_keyframe(self, sig: Signature) -> None:
        """Used to bootstrap tracking again after LOST (Phase 0: simply
        restarts local tracking from the current frame; global relocalisation
        against WM is a later-phase upgrade)."""
        self.ref_sig = sig
        self.ref_pose = self.cur_pose.copy()
        self._lost_streak = 0
        self.status = "OK"
        self.last_keyframe_created = True
        self.last_keyframe_signature = sig
        self.last_keyframe_link_T = None
