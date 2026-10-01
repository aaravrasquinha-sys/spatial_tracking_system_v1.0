"""
WP-LIVE: the BACKEND process. Owns memory/retrieval, loop verification,
local RGB-D relocalization, proximity, gravity prior, and incremental
(iSAM2) smoothing -- everything the planning notes' architecture table
puts at keyframe rate, asynchronous to the tracker's 30Hz loop.

Deliberate scope decision, stated honestly rather than silently:
this process reuses pyslam.pipeline.Pipeline UNCHANGED as its brain,
fed by a QueueSensorSource that pulls FramePackets off the tracker's
IPC queue instead of opening a camera. That means Pipeline's own
VisualOdometry runs a SECOND TIME here (the tracker already ran its
own odometry pass for keyframe decision/ICP-fallback/QA purposes) --
this is real, measurable duplicated frontend cost, not eliminated by
this work package. It was the right tradeoff for this scope: reusing
Pipeline verbatim means the entire, already-gated (54/54 selftest,
every WP_*_Findings.md's validated behaviour) memory/retrieval/verify/
proximity/gravity-prior/local_reloc/graph logic carries over with ZERO
risk of introducing a NEW correctness bug in code that already runs
correctly today, instead of a hasty mid-project refactor of
pipeline.py's internals to accept externally-decided keyframes from
the tracker. The natural WP-LIVE follow-up (tracked, not done here) is
giving Pipeline an injectable "keyframe decision already made
upstream" mode so the backend can skip its own frontend pass entirely.

Runs Pipeline.run() in a background thread (Pipeline itself is a
plain, non-async, single-consumer loop with no periodic-yield hook) so
this process's main thread can poll live state -- current pose
estimate, QA aggregation, dense hand-off -- at a controlled cadence
without any change to pipeline.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import queue
import threading
import time
import numpy as np

from pyslam.core.types import SensorSource, Intrinsics
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.pipeline import Pipeline
from pyslam.live.ipc import unpack_frame, pack_pose_update

log = get_logger("live.backend_process")

# WP-LIVE fix: a plain `object()` sentinel does NOT survive a real
# multiprocessing.Queue round-trip -- pickling it across the process
# boundary produces a NEW, non-identical instance, so an `item is
# _SENTINEL` check silently fails the moment this actually crosses a
# process boundary (it passed every in-process manual smoke test in
# this module's own development, since no pickling occurred there --
# caught only by tests/gates/test_g_live.py's REAL 3-process
# orchestrator check, which is exactly why that gate exists). None is
# a true CPython singleton and is guaranteed to unpickle to the same
# object every time, so it's used as the sole end-of-stream sentinel
# instead.
_SENTINEL = None


class QueueSensorSource:
    """SensorSource protocol adapter: pulls FramePackets off an
    mp.Queue (or any queue-like object with .get()) and unpacks them
    into ordinary Frame objects for Pipeline.run() to consume, exactly
    as if it were reading a camera. Blocks on .get() until a frame
    arrives or the sentinel/stop signals end of stream -- this is what
    lets Pipeline.run() work completely unmodified in a live,
    continuously-fed setting, not just batch/bag replay."""

    def __init__(self, frame_queue, intrinsics: Intrinsics, stop_event=None, get_timeout_s: float = 1.0):
        self._q = frame_queue
        self._intr = intrinsics
        self._stop_event = stop_event
        self._timeout = get_timeout_s

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def close(self) -> None:
        pass

    def __iter__(self):
        while True:
            if self._stop_event is not None and self._stop_event.is_set():
                return
            try:
                item = self._q.get(timeout=self._timeout)
            except queue.Empty:
                continue
            if item is None:
                return
            # free_after=True: the backend is the LAST consumer of this
            # frame's imagery in the current (Phase-1) architecture --
            # dense reads fused-depth POINTS from the tracker directly
            # (KeyframePointsPacket, a separate, smaller channel), never
            # the raw frame imagery -- see ipc.py's ownership-rule note.
            yield unpack_frame(item, free_after=True)


@dataclass
class BackendStatus:
    n_nodes: int
    n_links: int
    n_keyframes: int
    last_pose_generation: int
    wall_time_s: float


class BackendProcessState:
    """Holds the Pipeline + background thread + polling loop, factored
    out of the bare function target so it's usable both from
    backend_process.run() (the real multiprocessing entry point) and
    directly in-process from a test (no subprocess needed to exercise
    the polling/pose-broadcast logic)."""

    def __init__(self, cfg: Config, intrinsics: Intrinsics, frame_queue, pose_out_queues: list,
                 R_body_cam: Optional[np.ndarray] = None, backend_prefer: str = "auto",
                 stop_event=None, dense_pose_queues: Optional[list] = None):
        self.cfg = cfg
        self.source = QueueSensorSource(frame_queue, intrinsics, stop_event=stop_event)
        self.pipeline = Pipeline(cfg, backend_prefer=backend_prefer, R_body_cam=R_body_cam)
        self.pose_out_queues = pose_out_queues
        # WP-LIVE dense hookup fix: the DENSE process keys its stored
        # keyframe point clouds by frame_id (the TRACKER's own stable,
        # deterministic numbering -- see tracker_stage.py), while this
        # backend's own graph node ids come from a SEPARATE, independent
        # global counter (pyslam.frontend.features._alloc_id, shared
        # across every extract_signature() call this PROCESS makes --
        # in the live multi-process architecture the tracker and backend
        # are different processes and therefore have their OWN
        # independent counters too, so "same numeric id" was never a
        # safe assumption regardless). Sending dense a node_id-keyed
        # pose dict, as if node_id meant anything to it, silently
        # integrates nothing (a real bug this caught -- see
        # tests/gates/test_g_live.py's lockstep end-to-end check).
        # Fix: translate node_id -> frame_id using
        # PipelineResult.frame_records (which already records, for
        # every keyframe frame, exactly which node.id it became) before
        # broadcasting to dense_pose_queues specifically; tracker's own
        # pose_out_queues keep the ordinary node_id-keyed packet, since
        # tracker only needs a corrective transform, not per-keyframe
        # correlation.
        self.dense_pose_queues = dense_pose_queues or []
        self.stop_event = stop_event
        self._thread: Optional[threading.Thread] = None
        self._result = None
        self._generation = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._thread.start()

    def _run_pipeline(self) -> None:
        # WP-LIVE: replicates Pipeline.run()'s own setup (odometry/
        # verifier/orb init + result construction) rather than calling
        # run() directly, for exactly one reason: run() only returns
        # `result` once the WHOLE loop finishes, but this backend is
        # LIVE (the loop runs for the duration of the mapping session) --
        # poll_and_broadcast() needs to read result.frame_records
        # INCREMENTALLY, while _run_loop is still appending to it, to
        # build the frame_id->node_id map dense needs every poll cycle,
        # not just once at shutdown. self._result is assigned BEFORE
        # _run_loop runs (not after, the way run() does it), so the
        # polling thread always sees the live, growing object. No
        # change to pipeline.py itself -- see that module's own frozen-
        # interface discipline.
        from pyslam.core.types import Frame as _Frame  # noqa: F401 (typing aid only)
        from pyslam.pipeline import PipelineResult
        intr = self.source.intrinsics()
        if self.cfg.odometry_backend == "f2m":
            from pyslam.frontend.odometry_f2m import LocalMapOdometry
            self.pipeline.odometry = LocalMapOdometry(self.cfg)
        else:
            from pyslam.frontend.odometry import VisualOdometry
            self.pipeline.odometry = VisualOdometry(self.cfg)
        from pyslam.loop.verify import GeometricVerifier
        from pyslam.frontend.features import make_orb
        self.pipeline.verifier = GeometricVerifier(self.cfg, intr)
        self.pipeline.orb = make_orb(self.cfg)
        result = PipelineResult(memory=self.pipeline.memory, graph=self.pipeline.graph)
        self._result = result  # visible to poll_and_broadcast() from this point on
        try:
            self.pipeline._run_loop(self.source, None, False, result)
        except KeyboardInterrupt:
            log.warning(f"Backend interrupted after {result.n_frames} frames -- "
                        f"returning partial result.")

    def _frame_id_to_node_id_map(self) -> dict:
        if self._result is None:
            return {}
        return {rec["frame_id"]: rec["ref_kf_id"] for rec in self._result.frame_records
                if rec["is_keyframe"] and rec["ref_kf_id"] is not None}

    def poll_and_broadcast(self, min_interval_s: float = 0.2) -> BackendStatus:
        """Call periodically from the process's main thread (or a
        lockstep-mode caller) between frame arrivals: snapshots the
        live graph's current pose estimate and pushes it to every
        downstream consumer queue (tracker, for map_correction; dense,
        for Level-2 re-integration -- re-keyed by frame_id, see
        __init__'s docstring). Cheap: pipeline.graph.get_poses() just
        reads the backend's own already-computed dict."""
        poses = self.pipeline.graph.get_poses()
        self._generation += 1
        if poses:
            packet = pack_pose_update(poses, self._generation)
            for q in self.pose_out_queues:
                try:
                    q.put_nowait(packet)
                except Exception:
                    pass  # a full/closed downstream queue must never stall the backend

            if self.dense_pose_queues:
                fid_map = self._frame_id_to_node_id_map()
                dense_poses = {fid: poses[nid] for fid, nid in fid_map.items() if nid in poses}
                if dense_poses:
                    dense_packet = pack_pose_update(dense_poses, self._generation)
                    for q in self.dense_pose_queues:
                        try:
                            q.put_nowait(dense_packet)
                        except Exception:
                            pass
        return BackendStatus(
            n_nodes=len(self.pipeline.graph.node_ids), n_links=len(self.pipeline.graph.links),
            n_keyframes=(self._result.n_keyframes if self._result is not None else 0),
            last_pose_generation=self._generation, wall_time_s=time.time(),
        )

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)


def run_backend_process(cfg: Config, intrinsics: Intrinsics, frame_queue, pose_out_queues: list,
                         stop_event, status_queue=None, R_body_cam: Optional[np.ndarray] = None,
                         backend_prefer: str = "auto", poll_interval_s: float = 0.2,
                         dense_pose_queues: Optional[list] = None) -> None:
    """The actual multiprocessing.Process target. Spawns Pipeline.run()
    on a background thread, then polls/broadcasts poses at
    poll_interval_s cadence until stop_event fires, at which point it
    pushes the sentinel onto frame_queue (so QueueSensorSource's
    __iter__ returns and Pipeline.run() finishes cleanly) and joins."""
    state = BackendProcessState(cfg, intrinsics, frame_queue, pose_out_queues,
                                 R_body_cam=R_body_cam, backend_prefer=backend_prefer,
                                 stop_event=stop_event, dense_pose_queues=dense_pose_queues)
    state.start()
    try:
        while not stop_event.is_set():
            status = state.poll_and_broadcast()
            if status_queue is not None:
                try:
                    status_queue.put_nowait(status)
                except Exception:
                    pass
            time.sleep(poll_interval_s)
    finally:
        try:
            frame_queue.put_nowait(_SENTINEL)
        except Exception:
            pass
        state.join(timeout=30.0)
        state.poll_and_broadcast()  # final snapshot after the pipeline thread has drained
        log.info(f"Backend process stopped: {len(state.pipeline.graph.node_ids)} nodes, "
                 f"{len(state.pipeline.graph.links)} links.")
