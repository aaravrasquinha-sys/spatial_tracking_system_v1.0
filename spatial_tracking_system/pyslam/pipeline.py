"""
The Phase-0 pipeline: a single-threaded sequential loop wiring every
subsystem together. Deliberately no threads (see architecture brief,
section 7 rule #1) -- this eliminates the hardest bug class while
correctness of the algorithm itself is established.

    capture -> features -> odometry -> memory -> retrieval -> bayes
             -> verify -> graph -> map

This file should stay short. If it grows past ~250 lines, logic has
leaked in that belongs in one of the subsystem modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from pyslam.core.types import Frame, Node, Link, SensorSource, Intrinsics
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.core import lie
from pyslam.tools import metrics

from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
from pyslam.frontend.odometry import VisualOdometry
from pyslam.memory.memory import Memory
from pyslam.vpr.vocab import load_vocabulary, assign_words
from pyslam.vpr.bow import BowIndex
from pyslam.vpr.raw_match import score_candidates
from pyslam.vpr.likelihood import normalize_likelihood
from pyslam.vpr.place_recognizer import PlaceRecognizer
from pyslam.loop.bayes import BayesFilter
from pyslam.loop.verify import GeometricVerifier
from pyslam.graph.posegraph import PoseGraph
from pyslam.proximity.detect import find_proximity_candidates
from pyslam.imu.gravity import build_gravity_prior_link, is_quasi_static, gravity_direction_body
from pyslam.imu.bridge import build_bridge_link  # WP-M
from pyslam.loop.local_reloc import try_local_relocalization, build_local_reloc_link  # WP-LIVE N2

log = get_logger("pipeline")


@dataclass
class PipelineResult:
    memory: Memory
    graph: PoseGraph
    node_gt: dict = field(default_factory=dict)          # node_id -> gt 4x4, synthetic only
    loop_events: list = field(default_factory=list)
    proximity_events: list = field(default_factory=list)  # WP-P4, same
        # (new_id, other_id, Link) shape as loop_events
    bridge_events: list = field(default_factory=list)  # WP-M: (node_a_id,
        # node_b_id, dt_s, used_gyro, used_velocity) for every LOST-
        # recovery bridge link actually created, regardless of
        # cfg.bridge_mode -- empty unless bridge_mode="gyro" (the
        # "identity" default appends nothing here, same telemetry-only-
        # when-relevant pattern gravity_prior_events already uses).
        # used_gyro/used_velocity say which HALF of the estimate was real
        # (vs falling back to Phase 0's identity behaviour) -- see
        # pyslam/imu/bridge.py's per-component fallback guarantee.
    gravity_prior_events: list = field(default_factory=list)  # WP-P5.2:
        # (node_id, sigma_tilt_rad) for every keyframe a gravity/tilt
        # prior was actually added at -- most runs will have FEW of
        # these even with cfg.gravity_prior_enabled=True, since each
        # one requires that keyframe's own IMU window to be quasi-static
        # (see pyslam/imu/gravity.py). An empty list is not itself a
        # failure -- see that module's docstring on why a missing prior
        # is the safe outcome, not a wrong one.
    cross_session_merges: list = field(default_factory=list)  # see pipeline._try_loop_closure's
        # union-find check: loops that merged two previously-disconnected
        # graph components (e.g. across a LOST/session break) get logged
        # here explicitly, whether or not they were correct -- see the
        # Phase 1 plan's aliasing_rooms finding (WP-A3): this class of
        # merge has NO continuous odometry chain to cross-check against,
        # so the post-optimisation NEES rollback (which works well for
        # intra-component loops) cannot catch a geometrically coherent
        # false match. This list exists so such merges are auditable,
        # not silently accepted.        # (new_id, old_id, Link)
    telemetry: list = field(default_factory=list)           # list of dict
    odom_trajectory: list = field(default_factory=list)     # (t, 4x4) every processed frame
    status_log: list = field(default_factory=list)
    n_frames: int = 0
    n_keyframes: int = 0

    # WP-T1: per-frame pose bookkeeping for full-trajectory export (not
    # just keyframes). Each entry is a dict:
    #   frame_id, t, ref_kf_id, T_ref_frame (4x4, ref-keyframe<-frame,
    #   i.e. what odometry.update()'s own T_rel already computes, so no
    #   new tracking math is introduced here), status, session_id,
    #   is_keyframe.
    # A frame's FINAL pose (after any loop-closure optimisation, incl.
    # the closing optimisation at shutdown) is reconstructed as
    # final_pose(ref_kf_id) @ T_ref_frame -- this is why ref_kf_id/
    # T_ref_frame are stored instead of a resolved world pose: it lets
    # every ordinary frame inherit corrections made to its keyframe
    # AFTER the frame itself was processed, without re-running odometry.
    frame_records: list = field(default_factory=list)

    # WP-T3: raw IMU rows (each an (Ni,7) array) from the first
    # cfg.gravity_align_hold_s seconds of the run, for gravity-frame
    # alignment at export time -- see pyslam/core/gravity_frame.py.
    # Deliberately independent of gravity_prior_enabled/_try_gravity_prior
    # (which only look at windows BETWEEN keyframes once the pipeline is
    # already running): this captures the run's very first samples,
    # intended to be a still hold before any motion, regardless of
    # keyframe timing.
    startup_imu_window: list = field(default_factory=list)

    # WP-T1: set by Pipeline.finalize() -- a full-graph closing
    # optimisation (fixed=[first_node] only, NOT the WM-only LTM-anchored
    # optimisation _accept_link uses during the run) run once at
    # shutdown, over every node the graph has ever seen. This is the
    # "final" trajectory trajectory_export.py writes; odom_trajectory
    # and frame_records' online reconstruction remain the "online" one.
    final_poses: Optional[dict] = None


class Pipeline:
    def __init__(self, cfg: Config, vocab_path: Optional[str] = None,
                 backend_prefer: str = "auto", ltm_path: Optional[str] = None,
                 R_body_cam: Optional[np.ndarray] = None,
                 imagery_cache_dir: Optional[str] = None):
        self.cfg = cfg
        _reset_id_counter(0)
        self.orb = None  # created lazily once we know we're extracting features
        self.vocab = load_vocabulary(vocab_path) if vocab_path else None
        self.memory = Memory(cfg, ltm_path=ltm_path,
                              imagery_cache_dir=imagery_cache_dir)  # WP-P2 / WP-K3
        self.bow_index = BowIndex(cfg.vocab_size) if self.vocab is not None else None
        # WP-P3: cfg.retrieval_backend selects the retrieval mechanism.
        # "raw" (Phase 0's direct-Hamming matching) remains the default
        # for the same reason odometry_backend defaults to "f2f" -- see
        # config.py's note. self.place_recognizer is None on "raw" and
        # the old raw_match/bow_index path (unchanged) still drives
        # _try_loop_closure.
        self.place_recognizer = (PlaceRecognizer(cfg) if cfg.retrieval_backend != "raw" else None)
        if self.place_recognizer is not None:
            self.memory.on_transfer_out = self.place_recognizer.remove
            self.memory.on_transfer_in = lambda nid: self.place_recognizer.add(self.memory.get(nid))
        self.bayes = BayesFilter(cfg)
        self.graph = PoseGraph(cfg, prefer_backend=backend_prefer)
        self.odometry: Optional[VisualOdometry] = None
        self.verifier: Optional[GeometricVerifier] = None
        self._prev_kf_id: Optional[int] = None
        self._first_node_id: Optional[int] = None
        self._map_correction: np.ndarray = np.eye(4)  # world_map <- world_odom, see WP-B4 note below
        self._uf_parent: dict[int, int] = {}  # union-find over node ids, tracks graph connectivity
        self._kf_path: dict[int, float] = {}  # WP-L1: node id -> cumulative ODOMETRY path
            # length (m) at the moment that keyframe was created
        self._path_len: float = 0.0
        self._last_kf_pos: Optional[np.ndarray] = None
        self._session_id: int = 0  # WP-B3: incremented on every LOST/force_new_keyframe recovery
        self._pending_local_reloc = None  # WP-LIVE N2: set right before force_new_keyframe()
            # when try_local_relocalization() succeeds this LOST event; consumed (and cleared)
            # by the node-creation block below instead of building a bridge link.
        self._proximity_attempted: set = set()  # WP-P4: (a,b) pairs already
            # tried (accepted or not) -- never re-verify the same pair twice
        self._imu_since_last_kf: list = []  # WP-P5.2: accumulated Frame.imu
            # rows since the last keyframe, reset on every new keyframe --
            # see _run_loop's own accumulation site.
        self._prev_kf_t: Optional[float] = None  # WP-M: frame.t at which
            # self._prev_kf_id was created -- gives the bridge estimator
            # the true elapsed gap time (frame.t - this) without assuming
            # anything about frame rate during the LOST stretch.
        self._last_two_ok_frames: list = []  # WP-M: up to 2 most recent
            # (t, translation) samples of OdomResult.T_rel -- i.e. real,
            # SUCCESSFUL tracking against the CURRENT reference keyframe
            # (self._prev_kf_id; VisualOdometry.ref_sig always IS that
            # keyframe's own signature -- see update()/force_new_keyframe()).
            # Reset every time a new keyframe is created (the reference
            # changes). Only ever appended on a real match+PnP success
            # (n_inliers>0), never on a LOST/_handle_lost frame (whose
            # T_rel is a fabricated identity, not a measurement) or the
            # very first (bootstrap) frame -- see the appending site below.
        self._pre_lost_velocity: Optional[np.ndarray] = None  # WP-M:
            # finite-differenced from self._last_two_ok_frames, FROZEN the
            # first time LOST fires after a reset (not kept updating during
            # the LOST streak itself, since no real samples arrive then).
            # None until two good frames exist, or on the very first
            # keyframe -- pyslam/imu/bridge.py treats that as "no velocity
            # estimate available" and degrades the bridge's translation
            # block to Phase 0's identity value, not a crash.
        self._g_world: Optional[np.ndarray] = None  # WP-P5.2: established
            # once, from the first quasi-static keyframe-to-keyframe
            # window (see _try_gravity_prior); stays None (feature
            # silently inert) for the whole run if that never happens.
        self._R_body_cam: np.ndarray = (R_body_cam.copy() if R_body_cam is not None
                                         else np.eye(3))  # WP-P5.2: rotation mapping a
            # CAMERA-frame vector into the physical BODY frame Frame.imu
            # actually measures in (v_body = R_body_cam @ v_cam), same
            # convention as tests/synth/world.py's own T_BODY_CAM.
            # Defaults to identity ONLY because the frozen Intrinsics/
            # Config contract has no field for a real extrinsic yet --
            # see pyslam/imu/gravity.py's own docstring: this default is
            # KNOWN WRONG for this project's own D435i rig (found via
            # gravity_init_square6dof's end-to-end validation, not
            # designed in), kept only so gravity_prior_enabled doesn't
            # crash when a caller has no better value to pass; every
            # synthetic-fixture caller should pass the real one.

    def run(self, source: SensorSource, max_frames: Optional[int] = None,
            verbose: bool = True) -> PipelineResult:
        intr = source.intrinsics()
        # WP-B1: cfg.odometry_backend selects the tracker. "f2f" (Phase 0)
        # remains the default until f2m has been validated against the
        # frozen WP-A3 baseline on hardware, not just synthetic seeds.
        if self.cfg.odometry_backend == "f2m":
            from pyslam.frontend.odometry_f2m import LocalMapOdometry
            self.odometry = LocalMapOdometry(self.cfg)
        else:
            self.odometry = VisualOdometry(self.cfg)
        self.verifier = GeometricVerifier(self.cfg, intr)
        self.orb = make_orb(self.cfg)

        result = PipelineResult(memory=self.memory, graph=self.graph)

        n = 0
        try:
            self._run_loop(source, max_frames, verbose, result)
        except KeyboardInterrupt:
            # Live hardware runs (run_slam.py) are normally stopped this
            # way. Return whatever was captured rather than losing the
            # whole run -- see the Phase 1 plan's F7: a run that dies
            # without summary.json/map.ply defeats the run-directory
            # bookkeeping this codebase is otherwise careful about.
            log.warning(f"Interrupted after {result.n_frames} frames -- "
                        f"returning partial result.")
        return result

    def finalize(self, result: "PipelineResult") -> dict[int, np.ndarray]:
        """WP-T1: one full-graph closing optimisation at shutdown, over
        EVERY node the graph has ever seen (fixed=[first_node] only --
        deliberately NOT the WM-only, LTM-nodes-frozen optimisation
        _accept_link uses during the run for bounded online latency).
        The graph backend already holds every node regardless of
        Memory's WM/LTM residency (add_node is called once per keyframe,
        unconditionally), so this needs no extra bookkeeping -- it's the
        same optimize() call with a smaller fixed set.

        This is what makes a loop closure able to correct drift in a
        part of the trajectory that had already been evicted to LTM by
        the time the loop fired, which the online optimisation
        (correctly, for latency reasons) cannot do. Call this once,
        after run() returns (including after a KeyboardInterrupt/partial
        result -- there is no reason a partial graph can't still be
        closed), before exporting any trajectory. Idempotent: safe to
        call more than once (e.g. if a caller wants a pre- and post-
        finalize comparison), though ordinary use is exactly once.
        """
        if not self.graph.node_ids:
            result.final_poses = {}
            return {}
        fixed = [self._first_node_id] if self._first_node_id is not None else [self.graph.node_ids[0]]
        final_poses = self.graph.optimize(fixed)
        for nid, T in final_poses.items():
            try:
                self.memory.get(nid).pose_map = T
            except KeyError:
                pass
        result.final_poses = final_poses
        return final_poses

    def _run_loop(self, source: SensorSource, max_frames: Optional[int],
                   verbose: bool, result: "PipelineResult") -> None:
        n = 0
        t_start: Optional[float] = None
        for frame in source:
            t0 = time.perf_counter()
            if t_start is None:
                t_start = frame.t
            # WP-T1: capture the reference keyframe this FRAME (not the
            # eventual new keyframe, if any) is being tracked against,
            # BEFORE odometry.update() below potentially reassigns
            # ref_sig to a brand-new keyframe. This is exactly the
            # keyframe frame_records' T_ref_frame will be relative to.
            ref_kf_id_before = (self.odometry.ref_sig.id
                                 if self.odometry is not None and self.odometry.ref_sig is not None
                                 else None)

            sig = extract_signature(frame, self.cfg, self.orb)
            if self.vocab is not None:
                sig.word_ids = assign_words(sig.desc, self.vocab)

            odom_res = self.odometry.update(sig, frame)
            result.odom_trajectory.append((frame.t, odom_res.pose.copy()))

            # WP-M: maintain a short window of recent successful-tracking
            # samples, relative to the CURRENT reference keyframe, so a
            # constant-velocity estimate is available the instant LOST
            # fires -- see pyslam/imu/bridge.py and self._last_two_ok_frames'
            # own docstring above. Excludes the bootstrap frame and any
            # frame that came back from _handle_lost (n_inliers==0 marks
            # both, cheaply, without duplicating VisualOdometry's own
            # internal state machine here).
            if self.odometry.status == "OK" and odom_res.n_inliers > 0:
                self._last_two_ok_frames.append((frame.t, odom_res.T_rel[:3, 3].copy()))
                if len(self._last_two_ok_frames) > 2:
                    self._last_two_ok_frames.pop(0)
            elif self.odometry.status == "LOST" and self._pre_lost_velocity is None \
                    and len(self._last_two_ok_frames) == 2:
                (t0, p0), (t1, p1) = self._last_two_ok_frames
                dt_v = t1 - t0
                if dt_v > 1e-6:
                    self._pre_lost_velocity = (p1 - p0) / dt_v

            if frame.imu is not None and frame.imu.shape[0] > 0:
                self._imu_since_last_kf.append(frame.imu)  # WP-P5.2
                # WP-T3: also feed the startup window, independent of
                # keyframe timing -- see PipelineResult.startup_imu_window.
                if frame.t - t_start <= self.cfg.gravity_align_hold_s:
                    result.startup_imu_window.append(frame.imu)

            if self.odometry.status == "LOST":
                result.status_log.append(f"[t={frame.t:.2f}] LOST, attempting recovery")
                # WP-LIVE N2: try a real, measured RGB-D relocalization
                # against recent working memory BEFORE giving up to a
                # session break + guessed bridge link. On success this
                # keeps tracking CONTINUOUS in the map frame (no session
                # increment, no bridge) -- see local_reloc.py's module
                # docstring for why this is well-conditioned even from a
                # single candidate, unlike the frozen-map relocalizer.
                self._pending_local_reloc = None
                if self.cfg.local_reloc_enabled and sig.valid.any():
                    candidate_ids = self.memory.working_set()[-self.cfg.local_reloc_max_candidates:]
                    candidate_ids = list(reversed(candidate_ids))  # most-recently-added first
                    candidates = [self.memory.get(nid) for nid in candidate_ids if nid != getattr(sig, "id", None)]
                    reloc_result = try_local_relocalization(
                        sig, candidates, self.cfg,
                        min_inliers=self.cfg.local_reloc_min_inliers,
                        min_inlier_ratio=self.cfg.local_reloc_min_inlier_ratio,
                        max_rms_m=self.cfg.local_reloc_max_rms_m,
                        dist_thresh_m=self.cfg.local_reloc_dist_thresh_m,
                    )
                    if reloc_result is not None:
                        matched_node = self.memory.get(reloc_result.node_id)
                        T_world_query = matched_node.pose_map @ lie.se3_inverse(reloc_result.T_node_query)
                        # odometry.cur_pose is in the ODOMETRY (world_odom)
                        # frame; pose_map_init is later computed as
                        # self._map_correction @ odometry.cur_pose, so we
                        # store the INVERSE-corrected value here to keep
                        # that composition producing the right registered
                        # pose without touching the node-creation code
                        # below at all.
                        self.odometry.cur_pose = lie.se3_inverse(self._map_correction) @ T_world_query
                        self._pending_local_reloc = (reloc_result.node_id, reloc_result)
                        result.status_log.append(
                            f"[t={frame.t:.2f}] local_reloc SUCCESS vs node {reloc_result.node_id} "
                            f"(inliers={reloc_result.n_inliers}, ratio={reloc_result.inlier_ratio:.2f}) "
                            f"-- session NOT broken, no bridge link needed")
                self.odometry.force_new_keyframe(sig)
                # WP-B3 (Phase 1 plan finding, WP-A3): a LOST recovery
                # breaks tracking CONTINUITY -- the new reference frame's
                # pose_odom is not related to the old one by any measured
                # transform, so downstream pose_odom comparisons across
                # this point (drift metrics, RPE-by-distance) are
                # comparing two effectively-unrelated coordinate frames.
                # session_id makes this explicit rather than silent: see
                # metrics.rpe_by_distance's session_ids parameter, which
                # now refuses to score a segment that crosses one of
                # these boundaries instead of reporting a meaningless
                # "drift" number, which is what produced corridor_v2's
                # misleading 18%/m baseline (see WP_A3_Findings.md).
                # WP-LIVE N2: this reasoning no longer applies when
                # local_reloc succeeded above -- there IS a real measured
                # transform (reloc_result.T_node_query) tying the new
                # reference back to the map, so the session stays
                # continuous.
                if self._pending_local_reloc is None:
                    self._session_id += 1

            if self.odometry.last_keyframe_created:
                # WP-B4 fix (Phase 1 plan finding F6d): new nodes used to
                # be inserted at their raw pose_odom even after earlier
                # nodes had been dragged elsewhere by graph optimisation,
                # tearing both the initial guess handed to the NEXT
                # optimisation and the point cloud for every node added
                # between loop closures. self._map_correction is the
                # rigid transform (updated below, right after each
                # optimize() call) that maps the odometry frame into the
                # graph's current best estimate of the world; applying it
                # here keeps every new node's initial pose_map consistent
                # with wherever the map was last corrected to.
                pose_map_init = self._map_correction @ self.odometry.cur_pose
                node = Node(id=sig.id, sig=sig,
                            pose_odom=self.odometry.cur_pose.copy(),
                            pose_map=pose_map_init,
                            weight=1,
                            session_id=self._session_id)
                t_mem0 = time.perf_counter()  # WP-J2: see enforce_budget call below --
                    # everything from here through the end of this keyframe
                    # block is memory/retrieval/graph work that actually
                    # SCALES with WM size (so shrinking WM via transfer can
                    # reduce it); feature extraction and odometry, above,
                    # do not scale with WM size and must not drive eviction.
                self.memory.add(node)
                if self.bow_index is not None:
                    self.bow_index.add(node.id, sig.word_ids)
                if self.place_recognizer is not None:
                    self.place_recognizer.add(node)  # WP-P3
                # seed the optimiser with the corrected pose (pose_map),
                # not raw pose_odom -- see the map<-odom correction note
                # above. pose_odom itself is never touched: it stays the
                # frozen odometry-only estimate other code (e.g. the
                # convention oracles, RPE-vs-odom metrics) depends on.
                self.graph.add_node(node.id, node.pose_map)
                self._uf_parent[node.id] = node.id
                if self._first_node_id is None:
                    self._first_node_id = node.id

                if frame.gt_pose is not None:
                    result.node_gt[node.id] = frame.gt_pose.copy()

                if self._prev_kf_id is not None and self.odometry.last_keyframe_link_T is not None:
                    odom_link = Link(
                        a=self._prev_kf_id, b=node.id,
                        T_ab=self.odometry.last_keyframe_link_T,
                        info=self.odometry.last_keyframe_link_info,
                        kind="odom",
                        n_inliers=self.odometry.last_keyframe_link_inliers,
                    )
                    self.graph.add_link(odom_link)
                    self._uf_union(self._prev_kf_id, node.id)
                    self.memory.record_adjacency(self._prev_kf_id, node.id)  # WP-P2
                elif self._prev_kf_id is not None:
                    # WP-T0/WP-M: this keyframe was created via
                    # force_new_keyframe() after a LOST recovery (the only
                    # path that leaves last_keyframe_link_T None with a
                    # previous keyframe existing). Some link is always
                    # required here -- see Config.bridge_link_info_scale's
                    # docstring for why an orphan/unanchored node is what
                    # crashed GTSAM on corridor_v2's hardware run.
                    if self._pending_local_reloc is not None:
                        matched_id, reloc_result = self._pending_local_reloc
                        local_reloc_link = build_local_reloc_link(matched_id, node, reloc_result)
                        self.graph.add_link(local_reloc_link)
                        self._uf_union(matched_id, node.id)
                        self.memory.record_adjacency(matched_id, node.id)
                        result.status_log.append(
                            f"[t={frame.t:.2f}] local_reloc link node {matched_id} <-> node {node.id} "
                            f"(replaces bridge_mode={self.cfg.bridge_mode!r})")
                        self._pending_local_reloc = None
                    else:
                        # WP-LIVE N2 fix: the bridge construction AND its
                        # shared add_link/uf_union/record_adjacency tail
                        # must live inside this else -- an earlier version
                        # of this wiring left add_link/_uf_union/
                        # record_adjacency OUTSIDE the local_reloc-vs-
                        # bridge branch entirely, so a SUCCESSFUL
                        # local_reloc (which already added its own link
                        # above) fell through and re-added whatever
                        # `bridge_link` a PRIOR iteration had left bound
                        # (a stale Python local, reused unconditionally --
                        # confirmed via tests/gates/test_g_live.py's
                        # end-to-end LOST-recovery check, which caught
                        # this by asserting bridge-link COUNT rather than
                        # just presence: enabling local_reloc produced the
                        # SAME bridge count as leaving it off, plus extra
                        # local_reloc links, instead of trading one for
                        # the other). See that gate for the regression
                        # test.
                        # cfg.bridge_mode selects WHICH estimate: "identity"
                        # (Phase 0/WP-T0, unchanged default) or "gyro" (WP-M --
                        # see pyslam/imu/bridge.py). A real loop closure across
                        # this gap, once verified, will dominate either one.
                        if self.cfg.bridge_mode == "gyro":
                            imu_window = (np.concatenate(self._imu_since_last_kf, axis=0)
                                          if self._imu_since_last_kf else np.zeros((0, 7)))
                            dt_bridge = (frame.t - self._prev_kf_t) if self._prev_kf_t is not None else 0.0
                            bridge_link = build_bridge_link(
                                self._prev_kf_id, node.id, imu_window, self._R_body_cam,
                                self._pre_lost_velocity, dt_bridge,
                                min_gyro_samples=self.cfg.bridge_gyro_min_samples,
                                identity_info_scale=self.cfg.bridge_link_info_scale,
                                rotation_sigma_rad=self.cfg.bridge_rotation_sigma_rad,
                                velocity_sigma_base_mps=self.cfg.bridge_velocity_sigma_base_mps,
                                velocity_sigma_growth_mps_per_s=self.cfg.bridge_velocity_sigma_growth_mps_per_s,
                            )
                            used_gyro = imu_window.shape[0] >= self.cfg.bridge_gyro_min_samples
                            used_vel = self._pre_lost_velocity is not None and dt_bridge > 0.0
                            result.bridge_events.append(
                                (self._prev_kf_id, node.id, dt_bridge, used_gyro, used_vel))
                            result.status_log.append(
                                f"[t={frame.t:.2f}] bridge link node {self._prev_kf_id} <-> "
                                f"node {node.id} (post-LOST, gyro={'y' if used_gyro else 'n'} "
                                f"vel={'y' if used_vel else 'n'}, dt={dt_bridge:.2f}s)")
                        else:
                            bridge_info = np.eye(6) * self.cfg.bridge_link_info_scale
                            bridge_link = Link(
                                a=self._prev_kf_id, b=node.id,
                                T_ab=np.eye(4), info=bridge_info,
                                kind="bridge", n_inliers=0,
                            )
                            result.status_log.append(
                                f"[t={frame.t:.2f}] bridge link node {self._prev_kf_id} <-> "
                                f"node {node.id} (post-LOST, no real odometry measurement)")
                        self.graph.add_link(bridge_link)
                        self._uf_union(self._prev_kf_id, node.id)
                        self.memory.record_adjacency(self._prev_kf_id, node.id)
                self._prev_kf_id = node.id
                self._prev_kf_t = frame.t  # WP-M
                self._last_two_ok_frames = []  # WP-M: reset -- the reference just changed
                self._pre_lost_velocity = None  # WP-M
                result.n_keyframes += 1
                # WP-L1: odometry path length at this keyframe (see loop_min_path_m)
                pos = node.pose_odom[:3, 3].copy()
                if self._last_kf_pos is not None:
                    self._path_len += float(np.linalg.norm(pos - self._last_kf_pos))
                self._last_kf_pos = pos
                self._kf_path[node.id] = self._path_len

                self._try_loop_closure(node, result)
                if self.cfg.proximity_enabled:
                    self._try_proximity(node, result)
                if self.cfg.gravity_prior_enabled:
                    self._try_gravity_prior(node, result)
                mem_duration_ms = (time.perf_counter() - t_mem0) * 1000.0
                self._imu_since_last_kf = []  # WP-P5.2: reset for the next window,
                    # regardless of whether this keyframe's window was used
            else:
                mem_duration_ms = 0.0  # WP-J2: no memory-management work happens
                    # on a non-keyframe frame (see enforce_budget call below)

            # WP-T1: resolve this frame's (ref_kf_id, T_ref_frame) pair --
            # see PipelineResult.frame_records' docstring. A keyframe is
            # its own reference (T=identity); an ordinary frame is
            # referenced to whatever keyframe odometry was tracking
            # against BEFORE this update() call (captured above, since
            # update() may have just switched ref_sig to a brand-new
            # keyframe -- node.id below, for a keyframe frame).
            if self.odometry.last_keyframe_created:
                frame_ref_kf_id = node.id
                frame_T_ref_frame = np.eye(4)
            else:
                frame_ref_kf_id = ref_kf_id_before
                frame_T_ref_frame = odom_res.T_rel.copy()
            result.frame_records.append({
                "frame_id": frame.frame_id, "t": frame.t,
                "ref_kf_id": frame_ref_kf_id, "T_ref_frame": frame_T_ref_frame,
                "status": self.odometry.status, "session_id": self._session_id,
                "is_keyframe": bool(self.odometry.last_keyframe_created),
            })

            n += 1
            duration_ms = (time.perf_counter() - t0) * 1000.0
            # WP-J2 (Orin port, Tier 1): enforce_budget used to receive the
            # WHOLE frame's duration_ms -- capture, ORB extraction, PnP
            # odometry, memory ops, retrieval, verify, graph optimise, all
            # of it. That silently couples WM eviction pressure to
            # frontend/backend speed, neither of which eviction can fix
            # (a slower ORB implementation doesn't get faster by shrinking
            # WM). On a platform where the frontend is proportionally
            # slower or faster than the reference machine -- exactly the
            # case when porting to different hardware -- that mis-coupling
            # changes how aggressively nodes get evicted to LTM, and
            # therefore changes loop-closure recall, even though nothing
            # about the memory subsystem itself changed. mem_duration_ms
            # (0.0 on non-keyframe frames; real elapsed time over
            # memory.add + retrieval + verify + graph.optimize on keyframe
            # frames -- see the t_mem0 marker above) only reflects the part
            # of the pipeline whose cost actually scales with WM size,
            # which eviction can actually reduce.
            t_enf0 = time.perf_counter()
            self.memory.enforce_budget(mem_duration_ms)  # WP-P2/WP-J2
                # a no-op every frame that stays under cfg.wm_budget_ms
            enforce_ms = (time.perf_counter() - t_enf0) * 1000.0  # WP-K3: an eviction now also
                # writes the victim's rgb+depth to the imagery side-cache (two PNG encodes).
                # It happens AFTER duration_ms/mem_duration_ms were taken, so without this
                # field its cost would be invisible in telemetry -- measure it, don't assume it.
            result.n_frames = n
            result.telemetry.append({
                "t": frame.t, "frame_id": frame.frame_id,
                "duration_ms": duration_ms,
                "mem_duration_ms": mem_duration_ms,
                "enforce_budget_ms": enforce_ms,
                "odom_status": self.odometry.status,
                "n_inliers": odom_res.n_inliers,
                "wm_size": len(self.memory.working_set()),
                "keyframe": self.odometry.last_keyframe_created,
                "kf_reason": (getattr(self.odometry, "last_keyframe_reason", None)
                               if self.odometry.last_keyframe_created else None),  # WP-L3
            })
            if verbose and n % 30 == 0:
                log.info(f"frame {n}: t={frame.t:.2f}s status={self.odometry.status} "
                         f"kf={result.n_keyframes} wm={len(self.memory.working_set())}")

            if max_frames is not None and n >= max_frames:
                break

    def _uf_find(self, x: int) -> int:
        root = x
        while self._uf_parent.get(root, root) != root:
            root = self._uf_parent[root]
        while self._uf_parent.get(x, x) != root:
            self._uf_parent[x], x = root, self._uf_parent.get(x, x)
        return root

    def _uf_union(self, a: int, b: int) -> None:
        ra, rb = self._uf_find(a), self._uf_find(b)
        if ra != rb:
            self._uf_parent[ra] = rb

    def _loop_candidates(self, node: Node) -> list:
        """WM nodes eligible as loop-closure candidates for `node`. With the
        WP-L1 bounds off (default) this is exactly memory.working_set(). With
        loop_min_path_m / loop_min_time_s set, nodes that are too recent
        (odometry path or capture time since they were created below the
        bound) are dropped BEFORE retrieval scoring -- they never enter the
        likelihood population, the Bayes belief, or retrieval's cost. A node
        only ever moves from 'too recent' to 'eligible' (path and time only
        grow), so no stale belief is left behind."""
        cands = self.memory.working_set()
        min_path, min_time = self.cfg.loop_min_path_m, self.cfg.loop_min_time_s
        if not cands or (min_path <= 0.0 and min_time <= 0.0):
            return cands
        cur_path = self._kf_path.get(node.id)
        keep = []
        for c in cands:
            if min_path > 0.0 and cur_path is not None:
                cp = self._kf_path.get(c)
                if cp is not None and (cur_path - cp) < min_path:
                    continue
            if min_time > 0.0 and (node.sig.t - self.memory.get(c).sig.t) < min_time:
                continue
            keep.append(c)
        return keep

    def _try_loop_closure(self, node: Node, result: PipelineResult) -> None:
        candidate_ids = self._loop_candidates(node)
        if not candidate_ids:
            return

        if self.place_recognizer is not None:
            # WP-P3: incremental BoW / learned-descriptor / both, per
            # cfg.retrieval_backend. See place_recognizer.py.
            likelihood = self.place_recognizer.likelihood(node.sig, candidate_ids)
        else:
            candidate_nodes = [self.memory.get(cid) for cid in candidate_ids]
            # See vpr/raw_match.py docstring for why Phase 0 scores retrieval
            # by direct descriptor matching rather than through the fixed
            # BoW vocabulary (bow_index is still populated above for future
            # phases, just not used to drive this decision yet).
            raw_scores = score_candidates(node.sig, candidate_nodes)
            likelihood = normalize_likelihood(raw_scores)

        hyp = self.bayes.update(likelihood, candidate_ids,
                                 neighbours=self.memory.neighbours if self.cfg.bayes_diffusion_enabled else None)
        if hyp is None:
            return

        old_node = self.memory.get(hyp.node_id)
        link = self.verifier.verify(old_node, node)
        if link is None:
            return
        self._accept_link(link, hyp.node_id, node, result, kind_label="LOOP CLOSURE",
                           events_attr="loop_events", extra_log=f"posterior={hyp.posterior:.3f}, ",
                           cross_session_score=hyp.posterior)

    def _try_proximity(self, node: Node, result: PipelineResult) -> None:
        """WP-P4: geometry-only candidate generation (proximity/detect.py)
        alongside appearance-triggered _try_loop_closure. Every candidate
        still goes through the SAME GeometricVerifier -- see that
        module's docstring for why this is a complement to retrieval,
        not a relaxation of what counts as a verified link."""
        existing = set()
        # NativeBackend keeps its own link list; GTSAM's backend doesn't
        # expose an equivalent (not reachable in this environment --
        # GTSAM isn't installed here, see WP_P4_Findings.md -- so this
        # is a real but currently-inert gap, not exercised).
        for l in getattr(self.graph.backend, "_links", []):
            existing.add((min(l.a, l.b), max(l.a, l.b)))
        for pair in self._proximity_attempted:
            existing.add(pair)
        candidates = find_proximity_candidates(
            self.memory, existing, self.cfg.proximity_radius_m, self.cfg.proximity_min_index_gap)
        for a_id, b_id in candidates:
            self._proximity_attempted.add((a_id, b_id))
            a_node, b_node = self.memory.get(a_id), self.memory.get(b_id)
            link = self.verifier.verify(a_node, b_node)
            if link is None:
                continue
            link.kind = "proximity"
            self._accept_link(link, a_id, b_node, result, kind_label="PROXIMITY LINK",
                               events_attr="proximity_events", extra_log="")

    def _try_gravity_prior(self, node: Node, result: PipelineResult) -> None:
        """WP-P5.2: opt-in tilt-only prior at every keyframe once a
        gravity-aligned reference direction has been established (from
        the first quasi-static keyframe-to-keyframe IMU window
        encountered -- see pyslam/imu/gravity.py). Silently a no-op for
        any keyframe whose own window isn't quasi-static enough to
        trust, INCLUDING the very attempt to establish the reference
        direction in the first place -- see that module's own docstring
        on why a missing prior is the intended safe outcome here, not a
        bug to work around."""
        imu_window = (np.concatenate(self._imu_since_last_kf, axis=0)
                      if self._imu_since_last_kf else np.zeros((0, 7)))
        if self._g_world is None:
            if is_quasi_static(imu_window, self.cfg.gravity_prior_gyro_thresh_rad_s,
                                self.cfg.gravity_prior_accel_std_thresh_mps2):
                # Unit vector, established once, in the GRAPH's own
                # (camera) coordinate convention -- gravity_direction_body
                # returns the PHYSICAL body-frame direction, which must be
                # rotated through R_body_cam into camera convention before
                # being stored as u_world (see pyslam/imu/gravity.py's own
                # docstring on why this conversion is required, not
                # optional, found via end-to-end fixture testing).
                g_body_physical = gravity_direction_body(imu_window)
                self._g_world = self._R_body_cam.T @ g_body_physical
                log.info(f"WP-P5.2: gravity direction established at node {node.id} "
                         f"(u_world={self._g_world.round(3)})")
            return  # the establishing window is never also used for a
                # prior -- there is nothing yet to compare it against.

        node0 = None
        if self._first_node_id is not None:
            try:
                node0 = self.memory.get(self._first_node_id)
            except KeyError:
                node0 = None
        if node0 is None:
            return
        link = build_gravity_prior_link(
            node0.id, node.id, node0.pose_map, node.pose_map, self._g_world, imu_window,
            sigma_tilt_rad=self.cfg.gravity_prior_sigma_tilt_rad,
            gyro_thresh_rad_s=self.cfg.gravity_prior_gyro_thresh_rad_s,
            accel_std_thresh_mps2=self.cfg.gravity_prior_accel_std_thresh_mps2,
            R_body_cam=self._R_body_cam,
        )
        if link is None:
            return
        self.graph.add_link(link)
        result.gravity_prior_events.append((node.id, self.cfg.gravity_prior_sigma_tilt_rad))

    def _accept_link(self, link: Link, other_id: int, node: Node, result: PipelineResult,
                      kind_label: str, events_attr: str, extra_log: str,
                      cross_session_score: float = 0.0) -> None:
        """Shared accept/optimise/NEES-rollback path for both appearance-
        triggered loop closure and WP-P4's geometry-triggered proximity
        links -- extracted so proximity gets the exact same post-
        optimisation acceptance rigour (WP-B4's rollback rule) as loop
        closure always has, not a lighter-weight variant."""
        self.graph.add_link(link)
        self.memory.record_adjacency(other_id, node.id)  # WP-P2
        self.memory.on_loop(other_id)  # WP-P2: rehydrate other_id's
            # LTM-resident graph neighbours into WM before we optimise,
            # so verification/optimisation have local context, not just
            # the single matched node.

        # WP-A3 finding (aliasing_rooms): loops that merge two currently-
        # disconnected graph components (e.g. a real LOST/session break)
        # have no continuous odometry chain to check the new link's
        # implied pose against, which is what makes the NEES rollback
        # below effective for ordinary loops. HOWEVER: on aliasing_rooms
        # (two rooms with IDENTICAL texture seeds), this check turned out
        # not to fire at all -- because the frame-to-keyframe ODOMETRY
        # ITSELF was fooled by the aliasing across the room-A/room-B cut:
        # status_log was empty (no LOST ever declared) and the odom link
        # spanning the physical 20m teleport reported a normal-looking
        # 3.9cm step with 160 inliers, so the graph stayed one connected
        # component the whole time. Three subsequent cross-room loop
        # closures were then accepted with high inlier counts (60-412)
        # and NEES~0, and were mutually GEOMETRICALLY CONSISTENT with
        # each other (0.6-3.2cm agreement) -- not noisy, so no threshold
        # on inliers/posterior/NEES catches it. This is a strictly
        # harder problem than "verify a loop candidate": appearance-only
        # tracking cannot distinguish "genuinely revisiting this scene"
        # from "looking at an identical but different scene" at ANY
        # stage of the pipeline, frame-to-frame or loop closure, when the
        # two scenes are truly indistinguishable in the sensor's own
        # modality. Fixing this needs a signal outside pure appearance+
        # geometry -- e.g. Phase 5's IMU fusion would at least notice
        # that zero real acceleration was measured across an apparent
        # 20m visual jump, which vision alone cannot. Not fixed here.
        # The cross-component check below still has value for the
        # ordinary case (genuine tracking failure, e.g. textureless
        # segments or hard_case.bag) where LOST correctly fires.
        was_cross_component = self._uf_find(other_id) != self._uf_find(node.id)

        # WP-B4 post-optimisation loop acceptance (Phase 1 plan finding
        # F6b/F6d + the graph plan's "optimise, check, maybe roll back"
        # rule): a bad or aliased loop can slip past the verifier's
        # bidirectional geometric check (its inlier/reprojection test has
        # no knowledge of the REST of the graph); the only thing that can
        # catch "this loop is geometrically plausible on its own but
        # contradicts the accumulated odometry chain" is optimising with
        # it in and checking whether the graph now believes it.
        pre_snapshot = self.graph.snapshot()
        fixed = [self._first_node_id] if self._first_node_id is not None else []
        # WP-P2: "WM-only optimisation with frozen-LTM anchor priors"
        # (architecture doc section 5, P2). The graph backend already
        # supports this exactly -- `fixed` holds a node's pose constant
        # during optimize() -- so every LTM-resident id still present
        # in the graph (a node can be evicted from Memory's WM long
        # after the graph itself has it) is added to `fixed` alongside
        # the usual gauge-fixed first node. WM nodes still optimise
        # normally; LTM nodes act as anchor priors, not free variables.
        ltm_in_graph = [nid for nid in self.memory.ltm_ids_expanded() if nid in self._uf_parent]
        fixed = fixed + ltm_in_graph
        new_poses = self.graph.optimize(fixed)

        gt_poses_by_id = {nid: T for nid, T in new_poses.items()}
        nees = metrics.link_nees(link, gt_poses_by_id)
        # link_nees compares against a "ground truth" dict positionally;
        # here we deliberately reuse it against the OPTIMISED poses
        # (not ground truth) to ask "does the graph's own solution agree
        # with this link", which is exactly what post-hoc loop
        # acceptance needs and does not require any ground truth at all.
        if nees is not None and nees > self.cfg.loop_accept_chi2_995:
            log.warning(f"{kind_label} REJECTED (post-optimisation): node {node.id} <-> node "
                        f"{other_id} NEES={nees:.1f} > {self.cfg.loop_accept_chi2_995:.1f} -- "
                        f"optimisation disagrees with this link; rolling back.")
            self.graph.remove_last_link()
            self.graph.restore(pre_snapshot)
            return

        getattr(result, events_attr).append((node.id, other_id, link))
        self._uf_union(other_id, node.id)
        if was_cross_component:
            result.cross_session_merges.append((node.id, other_id, link.n_inliers, cross_session_score))
            log.warning(f"CROSS-SESSION MERGE ({kind_label}): node {node.id} <-> node {other_id} "
                        f"(inliers={link.n_inliers}) -- this link connects two previously-"
                        f"disconnected parts of the graph with no continuous odometry to "
                        f"cross-check against. See pipeline.py's WP-A3 note.")
        nees_str = f"{nees:.2f}" if nees is not None else "n/a"
        log.info(f"{kind_label}: node {node.id} <-> node {other_id} "
                 f"({extra_log}inliers={link.n_inliers}, nees={nees_str})")

        for nid, T in new_poses.items():
            try:
                self.memory.get(nid).pose_map = T
            except KeyError:
                pass
        # Update the running odom->map correction from the node that
        # triggered this optimisation (freshest estimate available) so
        # every subsequently-created node is seeded consistently -- see
        # the pose_map_init comment above in the per-frame loop.
        self._map_correction = new_poses[node.id] @ lie.se3_inverse(node.pose_odom)

