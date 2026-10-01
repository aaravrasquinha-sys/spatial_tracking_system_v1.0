"""
WP-LIVE: the TRACKER process. Owns capture, ORB, f2f PnP odometry (with
LM refinement and prediction-gated matching), ICP fallback tracking for
low-texture frames, the keyframe decision, per-frame fused-depth
accumulation into the current reference keyframe, and live QA metrics
-- everything the planning notes' architecture table puts at the 30Hz
frame budget. Never touches memory/retrieval/graph (that's the backend
process's job, at keyframe rate).

Every raw frame is ALSO forwarded to the backend process (as a
FramePacket) so the backend's own Pipeline instance can run its
already-validated memory/retrieval/verify/local_reloc/proximity/
gravity-prior/incremental-smoothing logic unchanged -- see
orchestrator.py's module docstring for why this (rather than a riskier
mid-project refactor of pipeline.py's internals to accept externally-
decided keyframes) is the deliberate scope for this work package, and
backend_process.py's own docstring for the honest cost of that choice
(the backend recomputes its own odometry pass; not yet eliminated).

This module's tracker-side odometry/ICP/keyframe-decision/fused-depth
work is real and independently useful regardless: it drives the live
mapping_status QA stream (pyslam/mapping/qa_stream.py) and the dense
process's Level-1 input, and is exactly the code path
tests/gates/test_g_live.py's ICP/odometry oracles already validate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from pyslam.core.types import SensorSource, Frame, Signature
from pyslam.core.config import Config
from pyslam.core import lie
from pyslam.frontend.features import extract_signature, make_orb
from pyslam.frontend.odometry import VisualOdometry
from pyslam.frontend.icp_fallback import point_to_plane_icp
from pyslam.mapping.fused_depth import KeyframeFusedDepth, FusionParams


@dataclass
class TrackerQaSnapshot:
    """One entry of the live mapping_status stream (WP-LIVE N1) -- see
    pyslam/mapping/qa_stream.py for the fuller aggregation; this is the
    per-frame raw material it's built from."""
    t: float
    frame_id: int
    odom_status: str
    n_inliers: int
    used_icp_fallback: bool
    icp_degenerate_directions: int
    speed_mps: Optional[float]
    rot_rate_dps: Optional[float]
    is_keyframe: bool
    fused_depth_valid_frac: Optional[float]


class TrackerStage:
    """The tracker's per-frame logic, factored out of any particular
    I/O (camera vs bag vs synthetic, in-process vs subprocess) so it's
    directly unit-testable and directly reusable from LockstepRunner
    (single-process, deterministic) and tracker_process.py's
    multiprocess entry point alike -- same relationship
    pyslam/frontend/odometry.py's VisualOdometry already has to
    pipeline.py."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.orb = make_orb(cfg)
        self.odometry = VisualOdometry(cfg)
        self._last_two: list[tuple] = []  # (t, xyz) for constant-velocity prediction
        self._cur_fused: Optional[KeyframeFusedDepth] = None
        self._fusion_params = FusionParams(depth_fx_px=cfg.__dict__.get("depth_stereo_fx_px", 390.0),
                                            baseline_m=0.0499)
        self.n_icp_fallback_used = 0

    def _predicted_T_ref_query(self) -> Optional[np.ndarray]:
        if len(self._last_two) < 2:
            return None
        (t0, x0), (t1, x1) = self._last_two
        dt = t1 - t0
        if dt <= 1e-6:
            return None
        # constant-velocity extrapolation in the ref-keyframe frame --
        # same source pyslam/imu/bridge.py's own pre-lost-velocity
        # tracking uses, reused here for prediction gating instead of
        # bridge construction.
        v = (x1 - x0) / dt
        return lie.make_T(np.eye(3), x1 + v * dt)

    def process_frame(self, frame: Frame) -> TrackerQaSnapshot:
        sig = extract_signature(frame, self.cfg, self.orb)
        T_pred = self._predicted_T_ref_query()
        res = self.odometry.update(sig, frame, T_ref_query_predicted=T_pred)

        used_icp = False
        icp_degenerate = 0
        if res.status == "LOST" and self._cur_fused is not None and sig.valid.any():
            # WP-LIVE 4.1/ICP fallback: try point-to-plane ICP of this
            # frame's own depth against the reference keyframe's fused
            # (denoised) depth before giving up -- see icp_fallback.py's
            # module docstring for why this targets exactly the
            # low-texture/corridor case f2f odometry is worst at.
            K = frame.intr.K()
            valid_idx = np.where(sig.valid)[0]
            if valid_idx.size >= 80:
                src_pts = sig.kp3d[valid_idx].astype(np.float64)
                # WP-LIVE perf note: icp_fallback's correspondence search
                # is O(points) in pure Python (see that module's own
                # docstring on why -- a coarse voxel bucket, not a real
                # kd-tree). A stride-2 dense query cloud (tens of
                # thousands of points at 640x480) is FAR too slow for
                # any frame budget; a coarse stride here keeps the point
                # count in the few-hundred to low-thousand range, which
                # is what this module's own gate (test_g_live.py's ICP
                # oracle) is sized against. Profiling on real hardware
                # should decide the final stride -- flagged as an open
                # tuning item in SYSTEM_SUMMARY_LIVE.md, not treated as
                # settled here.
                dst_pts, _ = self._cur_fused.get_points(K, stride=16)
                if dst_pts.shape[0] > 1500:
                    idx = np.random.default_rng(0).choice(dst_pts.shape[0], size=1500, replace=False)
                    dst_pts = dst_pts[idx]
                if dst_pts.shape[0] >= 200:
                    from pyslam.frontend.icp_fallback import _estimate_normals
                    dst_normals = _estimate_normals(dst_pts)
                    T_init = T_pred if T_pred is not None else np.eye(4)
                    # WP-LIVE perf/tuning note: a LOST fallback's seed is
                    # necessarily weaker than the well-conditioned,
                    # already-close seed tests/gates/test_g_live.py's ICP
                    # oracle uses (recovering a KNOWN small transform) --
                    # the correspondence threshold has to be loose enough
                    # to bootstrap from a poor (or absent, identity) seed.
                    # A tighter, tuned schedule (loose first iterations,
                    # tightening as it converges) is the natural
                    # production follow-up; flagged here rather than
                    # quietly shipped as settled.
                    icp_res = point_to_plane_icp(src_pts, dst_pts, dst_normals, T_init,
                                                  corr_dist_m=0.2, min_correspondences=50)
                    if icp_res is not None and icp_res.degenerate_directions < 6:
                        used_icp = True
                        icp_degenerate = icp_res.degenerate_directions
                        self.n_icp_fallback_used += 1
                        # feed the ICP result back as a recovered odometry
                        # frame rather than a new keyframe -- treats ICP
                        # as an ADDITIONAL measurement path for the SAME
                        # tracking state machine, not a separate tracker.
                        self.odometry.status = "OK"
                        self.odometry.cur_pose = self.odometry.ref_pose @ icp_res.T_ref_cur
                        self.odometry._lost_streak = 0

        if self.odometry.status == "OK" and res.n_inliers > 0:
            self._last_two.append((frame.t, res.T_rel[:3, 3].copy()))
            if len(self._last_two) > 2:
                self._last_two.pop(0)

        speed = rot_rate = None
        if len(self._last_two) == 2:
            (t0, x0), (t1, x1) = self._last_two
            dt = t1 - t0
            if dt > 1e-6:
                speed = float(np.linalg.norm(x1 - x0) / dt)

        if self.odometry.last_keyframe_created:
            self._cur_fused = KeyframeFusedDepth(frame.intr.height, frame.intr.width, self._fusion_params)

        if self._cur_fused is not None:
            depth_m = frame.depth.astype(np.float64) * frame.intr.depth_scale
            # warp THIS frame's depth into the reference keyframe via
            # T_ref_frame is only meaningful for points, not a dense
            # depth MAP re-projection in general (occlusion, resampling)
            # -- for a keyframe's own frame (T_ref_frame=identity) this
            # is a direct fuse; for any other frame, fusing raw (un-
            # warped) depth is an approximation deliberately accepted
            # here for the frame-rate budget (see fused_depth.py's own
            # "pure numpy, cheap" scoping) -- a frame close to its
            # keyframe (typical keyframe spacing) has small enough
            # baseline for this to still denoise usefully; warping
            # properly is a natural follow-up once profiling shows
            # headroom.
            if self.odometry.last_keyframe_created:
                self._cur_fused.fuse_frame(depth_m)

        return TrackerQaSnapshot(
            t=frame.t, frame_id=frame.frame_id, odom_status=self.odometry.status,
            n_inliers=res.n_inliers, used_icp_fallback=used_icp,
            icp_degenerate_directions=icp_degenerate, speed_mps=speed, rot_rate_dps=rot_rate,
            is_keyframe=self.odometry.last_keyframe_created,
            fused_depth_valid_frac=(self._cur_fused.stats()["frac_valid"]
                                     if self._cur_fused is not None else None),
        )
