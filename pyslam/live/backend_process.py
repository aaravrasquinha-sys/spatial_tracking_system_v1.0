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
from pyslam.live.ipc import unpack_frame, pack_pose_update, free_pose_packet, free_frame_packet
from pyslam.live.dense_select import select_dense_pose

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

    def __init__(self, frame_queue, intrinsics: Intrinsics, stop_event=None, get_timeout_s: float = 1.0,
                 catchup_queue_frames: int = 0):
        self._q = frame_queue
        self._intr = intrinsics
        self._stop_event = stop_event      # ABORT only: a normal stop ends via the sentinel so the
                                           # queue is drained first (Batch-1 N8)
        self._timeout = get_timeout_s
        # Batch-B catch-up: if at least this many frames are waiting, skip every second ORDINARY frame
        # (never a tracker keyframe) until the queue is back to half of it. 0 = off.
        self._catchup_hi = int(catchup_queue_frames)
        self._catching_up = False
        self._skip_toggle = False
        self.n_catchup_skipped = 0
        self.tracker_kf_ids: list = []     # Batch-1 N6: frame_ids the tracker flagged as keyframes
                                           # (list.append is atomic; read from another thread via list())

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def close(self) -> None:
        pass

    def _should_skip(self) -> bool:
        try:
            backlog = int(self._q.qsize())
        except Exception:       # qsize is not implemented on every platform
            return False
        if backlog >= self._catchup_hi and not self._catching_up:
            self._catching_up = True
            log.warning(f"Backend is behind ({backlog} frames waiting) -- skipping every second ordinary "
                        f"frame until it catches up (keyframe frames are never skipped).")
        elif backlog <= self._catchup_hi // 2 and self._catching_up:
            self._catching_up = False
            log.info(f"Backend caught up ({self.n_catchup_skipped} frames skipped in total).")
        if not self._catching_up:
            return False
        self._skip_toggle = not self._skip_toggle
        return self._skip_toggle

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
            if getattr(item, "tracker_keyframe", False):
                self.tracker_kf_ids.append(item.frame_id)
            elif self._catchup_hi > 0 and self._should_skip():
                free_frame_packet(item)        # give the shared memory back, the frame is not used
                self.n_catchup_skipped += 1
                continue
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
        self.source = QueueSensorSource(frame_queue, intrinsics, stop_event=stop_event,
                                        catchup_queue_frames=int(getattr(cfg, "backend_catchup_queue_frames", 0)))
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
        self._finalized = False
        self.error: Optional[str] = None
        self._rec_index: dict = {}      # frame_id -> index into result.frame_records (built incrementally)
        self._rec_scanned = 0

    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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
        except Exception as e:  # noqa: BLE001
            # Batch-1: a crash used to kill this thread silently while the process kept "running".
            import traceback
            self.error = f"{type(e).__name__}: {e}"
            log.error(f"Backend pipeline thread crashed after {result.n_frames} frames: {self.error}\n"
                      f"{traceback.format_exc()}")

    def finalize(self) -> bool:
        """Batch-1: the full-graph closing optimisation (Pipeline.finalize) -- the live backend never
        called it, so a loop closure could not correct drift in nodes already evicted from working
        memory. Call ONLY after the pipeline thread has finished."""
        if self._result is None or self._finalized:
            return self._finalized
        try:
            self.pipeline.finalize(self._result)
            self._finalized = True
            log.info(f"Backend finalize(): full-graph optimisation done over "
                     f"{len(self.pipeline.graph.node_ids)} nodes.")
        except Exception as e:  # noqa: BLE001
            log.warning(f"Backend finalize() failed ({type(e).__name__}: {e}); "
                        f"the map keeps the ONLINE poses.")
        return self._finalized

    def _frame_id_to_node_id_map(self) -> dict:
        if self._result is None:
            return {}
        return {rec["frame_id"]: rec["ref_kf_id"] for rec in self._result.frame_records
                if rec["is_keyframe"] and rec["ref_kf_id"] is not None}

    def _dense_frame_poses(self, poses: dict) -> dict:
        """Batch-1 (N6): {tracker_frame_id: world<-cam pose} for every frame the TRACKER flagged as a
        keyframe. pose = pose[ref keyframe] @ T_ref_frame, so it works even when the backend's own
        odometry chose different keyframes (those used to get no pose and silently never reached
        the dense map). Frames the backend itself marked LOST are skipped: their T_ref_frame is a
        fabricated identity, not a measurement."""
        res = self._result
        if res is None:
            return {}
        recs = res.frame_records
        n = len(recs)
        for i in range(self._rec_scanned, n):
            self._rec_index[recs[i]["frame_id"]] = i
        self._rec_scanned = n
        out = {}
        for fid in set(list(self.source.tracker_kf_ids)):
            i = self._rec_index.get(fid)
            if i is None:
                continue
            T = select_dense_pose(recs[i], poses, self.cfg)     # Batch-C: dense_pose_mode
            if T is not None:
                out[fid] = T
        return out

    def _put(self, q, packet, block_timeout_s: float = 0.0) -> bool:
        """Put a pose packet on a queue; if it cannot be delivered, FREE its shared memory
        (it used to leak one block pair per failed put)."""
        try:
            if block_timeout_s > 0:
                q.put(packet, timeout=block_timeout_s)
            else:
                q.put_nowait(packet)
            return True
        except Exception:
            free_pose_packet(packet)
            return False

    def poll_and_broadcast(self, min_interval_s: float = 0.2, final: bool = False) -> BackendStatus:
        """Call periodically from the process's main thread (or a
        lockstep-mode caller) between frame arrivals: snapshots the
        live graph's current pose estimate and pushes it to every
        downstream consumer queue (tracker, for map_correction; dense,
        for Level-2 re-integration -- re-keyed by frame_id, see
        __init__'s docstring). Cheap: pipeline.graph.get_poses() just
        reads the backend's own already-computed dict.

        final=True (Batch-1): the shutdown broadcast. Uses the finalized poses when finalize()
        succeeded, and BLOCKS (up to 5 s per queue) instead of dropping the packet."""
        poses = None
        if final and self._finalized and self._result is not None and getattr(self._result, "final_poses", None):
            poses = self._result.final_poses
        if poses is None:
            poses = self.pipeline.graph.get_poses()
        self._generation += 1
        wait = 5.0 if final else 0.0
        if poses:
            for q in self.pose_out_queues:      # one packet PER queue: each consumer frees its own copy
                self._put(q, pack_pose_update(poses, self._generation))
            if self.dense_pose_queues:
                dense_poses = self._dense_frame_poses(poses)
                if dense_poses:
                    for q in self.dense_pose_queues:
                        self._put(q, pack_pose_update(dense_poses, self._generation), wait)
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
                         dense_pose_queues: Optional[list] = None, drain_timeout_s: float = 120.0) -> None:
    """The actual multiprocessing.Process target. Spawns Pipeline.run()
    on a background thread, then polls/broadcasts poses at
    poll_interval_s cadence.

    Batch-1 shutdown contract (N8): stop_event is now an ABORT signal only. A normal stop is the
    tracker sending the end-of-stream sentinel after its last frame; this loop then keeps running
    until the pipeline thread has consumed every queued frame, runs the full-graph finalize(), and
    makes one last (blocking) pose broadcast to dense, BEFORE the process exits."""
    state = BackendProcessState(cfg, intrinsics, frame_queue, pose_out_queues,
                                 R_body_cam=R_body_cam, backend_prefer=backend_prefer,
                                 stop_event=stop_event, dense_pose_queues=dense_pose_queues)
    state.start()
    try:
        while not stop_event.is_set() and state.thread_alive():
            status = state.poll_and_broadcast()
            if status_queue is not None:
                try:
                    status_queue.put_nowait(status)
                except Exception:
                    pass
            time.sleep(poll_interval_s)
    finally:
        try:
            frame_queue.put_nowait(_SENTINEL)   # harmless if the tracker already sent one
        except Exception:
            pass
        state.join(timeout=drain_timeout_s)
        if state.thread_alive():
            log.warning(f"Backend pipeline still busy after {drain_timeout_s:.0f}s -- skipping finalize(); "
                        f"the map keeps the ONLINE poses.")
        else:
            state.finalize()
        state.poll_and_broadcast(final=True)
        log.info(f"Backend process stopped: {len(state.pipeline.graph.node_ids)} nodes, "
                 f"{len(state.pipeline.graph.links)} links, finalized={state._finalized}, "
                 f"pauses={state.pipeline.n_pauses} ({state.pipeline.paused_s_total:.1f}s paused), "
                 f"catch-up skipped={state.source.n_catchup_skipped} frames.")
