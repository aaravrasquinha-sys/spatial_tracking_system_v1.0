"""
WP-LIVE: the orchestrator. Spawns tracker/backend/dense as three real
`multiprocessing.Process` instances (approved: multi-process live mode,
"no threads before P6" relaxed ONLY for this kind of process
isolation), wires the IPC queues between them (pyslam/live/ipc.py), and
owns clean shutdown (Ctrl-C safe, same discipline run_slam.py already
uses for the single-process pipeline).

Process graph:

    tracker  --frame_queue-->        backend
    tracker  --points_queue-->                dense
             <--pose_queue(tracker)--  backend
                                       backend --dense_pose_queue--> dense

Uses spawn (not fork) as the start method explicitly -- fork silently
duplicates CUDA/GPU contexts and open device handles (the RealSense
pipeline handle in particular) into the child, which is a well-known
source of hard-to-debug crashes; spawn re-imports cleanly in each
child, at the cost of needing every process target to be a real
top-level function (satisfied here: tracker_entry/backend_entry/
dense_entry are all module-level).

NOT exercised against real hardware or GTSAM in this development
sandbox (no D435i, no pyrealsense2, no gtsam importable here -- see
every other WP_*_Findings.md's own "not validated" sections for the
same honest caveat). What IS exercised here, and gated
(tests/gates/test_g_live.py), is the SYNTHETIC end-to-end path: real
OS processes, real shared memory, a real synthetic camera source,
proving the wiring itself is correct independent of hardware
availability.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import multiprocessing as mp
import os
import signal
import time

import numpy as np

from pyslam.core.config import Config
from pyslam.core.log import get_logger

log = get_logger("live.orchestrator")


def _ignore_sigint() -> None:
    """Batch-1: Ctrl-C in a terminal goes to EVERY process in the foreground group, not just the
    parent. Children used to die on it (the dense process never got to write its final export).
    Now only the parent reacts to Ctrl-C and runs the staged stop; children are stopped through
    their events/sentinel (and terminate() as a last resort, which is SIGTERM, not SIGINT)."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def _harden_child() -> None:
    """Every child process starts with this. (1) ignore Ctrl-C (see _ignore_sigint). (2) Linux only: ask
    the kernel to send this child SIGTERM if the PARENT dies for any reason, even a SIGKILL. Without (2),
    a force-killed run_live_map.py would leave tracker/backend/dense running forever with the camera held
    (children that ignore Ctrl-C no longer die from it the way they used to)."""
    _ignore_sigint()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)     # PR_SET_PDEATHSIG = 1
        if os.getppid() == 1:                           # parent already gone before the call took effect
            os._exit(0)
    except Exception:
        pass            # non-Linux / no libc: the staged stop and terminate() still apply


def _send_stream_meta(source, meta_queue) -> None:
    """Batch-A: the tracker owns the real camera, so it is the one process that knows the DEVICE's own
    intrinsics and the camera<->IMU rotation. Hand both to the backend BEFORE the first frame, because the
    backend builds its Pipeline (and its loop-closure verifier) before any frame arrives and used to get a
    hardcoded intrinsics guess and no rotation at all (R_body_cam=None -> identity)."""
    if meta_queue is None:
        return
    meta = {"intr": None, "R_body_cam": None}
    try:
        meta["intr"] = source.intrinsics()
    except Exception as e:  # noqa: BLE001
        log.warning(f"Tracker could not read the source intrinsics ({e}); the backend keeps its default.")
    for name in ("imu_to_color_rotation", "R_body_cam"):
        val = getattr(source, name, None)
        try:
            val = val() if callable(val) else val
            if val is not None:
                meta["R_body_cam"] = np.asarray(val, dtype=np.float64).reshape(3, 3)
                break
        except Exception as e:  # noqa: BLE001
            log.warning(f"Tracker could not read {name} from the source ({e}).")
    try:
        meta_queue.put_nowait(meta)
    except Exception:
        log.warning("Tracker could not hand the stream metadata to the backend.")


def _wait_stream_meta(meta_queue, intrinsics, R_body_cam, timeout_s: float = 60.0):
    """Backend side of _send_stream_meta. Falls back to the values it was started with (the old behaviour)
    if the tracker never delivers (e.g. the camera failed to open), so a missing message cannot hang it."""
    if meta_queue is None:
        return intrinsics, R_body_cam
    try:
        meta = meta_queue.get(timeout=timeout_s)
    except Exception:
        log.warning(f"Backend got no stream metadata from the tracker within {timeout_s:.0f}s -- "
                    f"using the default intrinsics and NO camera<->IMU rotation.")
        return intrinsics, R_body_cam
    intr = meta.get("intr") or intrinsics
    R = meta.get("R_body_cam")
    R = R if R is not None else R_body_cam
    log.info(f"Backend uses the DEVICE intrinsics fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.cx:.2f} "
             f"cy={intr.cy:.2f} {intr.width}x{intr.height}; camera<->IMU rotation "
             f"{'received from the device' if meta.get('R_body_cam') is not None else 'NOT available (identity)'}.")
    return intr, R


def _apply_device_depth_options(source, cfg: Config, capture_profile) -> None:
    """Batch-C: on a real camera, optionally apply the capture profile's depth preset / laser power and the depth
    post-processing filters. A no-op for any source that does not support it (synthetic, bag playback)."""
    want_preset = bool(getattr(cfg, "device_depth_preset", False))
    filters = str(getattr(cfg, "device_depth_filters", "off"))
    if not want_preset and filters == "off":
        return
    fn = getattr(source, "apply_capture_profile", None)
    if fn is None:
        log.info("device_depth_* options set, but this source is not a live camera -- ignored.")
        return
    try:
        if capture_profile is None:
            from pyslam.live.capture_profile import CaptureProfile
            capture_profile = CaptureProfile()
        fn(capture_profile, apply_preset=want_preset, filters=filters)
    except Exception as e:  # noqa: BLE001
        log.warning(f"Could not apply the camera depth options ({e}); continuing with the camera defaults.")


def _send_keyframe_points(kf, cfg: Config, points_queue, qa) -> None:
    """Turn a finished tracker keyframe into a dense point packet and queue it for the dense process."""
    from pyslam.live.ipc import alloc_shm_array, KeyframePointsPacket, free_points_packet
    pts, pixel_yx = kf.fused.get_points(kf.K, stride=4,
                                        max_depth_m=float(getattr(cfg, "dense_max_depth_m", 0.0)),
                                        edge_rel=float(getattr(cfg, "dense_edge_rel", 0.0)))
    if pts.shape[0] == 0:
        return
    colors = kf.rgb[pixel_yx[:, 0], pixel_yx[:, 1]]
    ppkt = KeyframePointsPacket(
        keyframe_id=kf.frame_id,
        points_handle=alloc_shm_array(pts.astype(np.float32)),
        colors_handle=alloc_shm_array(colors.astype(np.uint8)),
    )
    try:
        points_queue.put_nowait(ppkt)
    except Exception:
        free_points_packet(ppkt)      # used to leak the shared memory
        qa.add_drop(points=1)


def _tracker_entry(cfg: Config, source_factory, capture_profile, frame_queue, points_queue,
                    pose_in_queue, stop_event, qa_queue, meta_queue=None) -> None:
    """Module-level (spawn-picklable) tracker process target.
    source_factory: a zero-arg callable returning a SensorSource (e.g.
    a closure/partial built from RealSenseSource(...) or
    SyntheticSource(...) -- kept as a factory rather than a live object
    so it's constructed INSIDE the child process, where it belongs, not
    pickled from the parent (a RealSense pipeline handle in particular
    must never cross a process boundary this way)."""
    _harden_child()
    from pyslam.live.tracker_stage import TrackerStage
    from pyslam.live.ipc import (pack_frame, alloc_shm_array, KeyframePointsPacket, free_frame_packet,
                                 free_points_packet, unpack_pose_update)
    from pyslam.mapping.qa_stream import MappingStatusAggregator
    import numpy as np

    source = source_factory()
    _apply_device_depth_options(source, cfg, capture_profile)
    _send_stream_meta(source, meta_queue)
    stage = TrackerStage(cfg)
    qa = MappingStatusAggregator()
    last_qa_push = 0.0

    try:
        for frame in source:
            if stop_event.is_set():
                break
            snap = stage.process_frame(frame)
            qa.add_frame(snap)

            pkt = pack_frame(frame, tracker_keyframe=snap.is_keyframe)
            try:
                if snap.is_keyframe:
                    # Batch-A: the frame of a KEYFRAME is the one the dense map needs a backend pose for;
                    # losing it leaves a hole in the map. Wait (bounded) for room instead of dropping.
                    frame_queue.put(pkt, timeout=1.0)
                else:
                    frame_queue.put_nowait(pkt)
            except Exception:
                # a full backend queue must never block capture for long -- count the drop (the
                # backend's odometry sees a gap, which means bigger jumps and more LOST events) and
                # free the shared memory (it used to leak).
                free_frame_packet(pkt)
                qa.add_drop(frames=1)
                if snap.is_keyframe:
                    log.warning(f"Backend queue stayed full for 1 s: dropped the frame of KEYFRAME "
                                f"{frame.frame_id} (a hole in the map). The backend cannot keep up.")

            for kf in stage.take_finished_keyframes():      # Batch-C: one per keyframe once its depth is ready
                _send_keyframe_points(kf, cfg, points_queue, qa)

            # Drain (and free the shared memory of) pose updates from the backend. They are not used
            # here: the old `map_correction` variable kept an arbitrary dict entry and was never read.
            try:
                while True:
                    unpack_pose_update(pose_in_queue.get_nowait(), free_after=True)
            except Exception:
                pass

            now = time.time()
            if now - last_qa_push > 0.5:
                try:
                    qa_queue.put_nowait(qa.snapshot())
                except Exception:
                    pass
                last_qa_push = now
    finally:
        try:
            for kf in stage.flush():                        # Batch-C: the keyframe still being built
                _send_keyframe_points(kf, cfg, points_queue, qa)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Could not hand out the last keyframe's points ({e}).")
        try:
            source.close()
        except Exception:
            pass
        # Batch-1 (N8): tell the backend "no more frames" AFTER the last one, so it drains its queue,
        # runs finalize() and exits by itself (the old shutdown stopped all three processes at once).
        try:
            frame_queue.put(None, timeout=10.0)
        except Exception:
            log.warning("Tracker could not enqueue the end-of-stream sentinel (backend queue full).")
        try:
            qa_queue.put_nowait(qa.snapshot())     # one last QA snapshot, includes drop counters
        except Exception:
            pass
        try:
            while True:
                unpack_pose_update(pose_in_queue.get_nowait(), free_after=True)
        except Exception:
            pass
        log.info(f"Tracker process stopped: {qa.n_frames} frames, {qa.n_keyframes} keyframes, "
                 f"{stage.n_icp_fallback_used} ICP-fallback recoveries, "
                 f"{qa.n_frames_dropped} frames / {qa.n_points_dropped} point clouds dropped.")


def _backend_entry(cfg: Config, intrinsics, frame_queue, tracker_pose_queue, dense_pose_queue,
                    abort_event, backend_prefer: str, R_body_cam, meta_queue=None) -> None:
    _harden_child()
    from pyslam.live.backend_process import run_backend_process
    intrinsics, R_body_cam = _wait_stream_meta(meta_queue, intrinsics, R_body_cam)
    run_backend_process(cfg, intrinsics, frame_queue, pose_out_queues=[tracker_pose_queue],
                         stop_event=abort_event, R_body_cam=R_body_cam, backend_prefer=backend_prefer,
                         dense_pose_queues=[dense_pose_queue])


def _dense_entry(points_queue, dense_pose_queue, stop_event, out_ply_path, voxel_size_m, min_neighbors=0) -> None:
    _harden_child()
    from pyslam.live.dense_process import run_dense_process
    run_dense_process(points_queue, dense_pose_queue, stop_event, out_ply_path=out_ply_path,
                       voxel_size_m=voxel_size_m, export_min_neighbors=min_neighbors)


@dataclass
class LiveMappingSession:
    tracker_proc: object
    backend_proc: object
    dense_proc: object
    stop_event: object          # tracker stop (kept under the old name for existing callers)
    qa_queue: object
    abort_event: object = None       # backend hard-abort (only set if the clean drain times out)
    dense_stop_event: object = None  # dense stop (set only AFTER the backend has finished)
    tracker_pose_queue: object = None
    dense_pose_queue: object = None
    points_queue: object = None

    def _terminate(self, p, why: str) -> None:
        if p.is_alive():
            log.warning(f"Process {p.name} {why} -- terminating.")
            p.terminate()
            p.join(timeout=5.0)

    def stop(self, timeout: float = 15.0, backend_timeout: float = 150.0) -> None:
        """Batch-1 (N8) staged shutdown: tracker first -> backend drains + finalize()s + does a last
        pose broadcast -> only then dense is told to stop and export. Previously all three were
        stopped at once, so the final poses (and any finalize()) were lost."""
        log.info("Stopping live mapping session (tracker -> backend -> dense)...")
        try:
            self.stop_event.set()
            self.tracker_proc.join(timeout=timeout)
            self._terminate(self.tracker_proc, f"did not stop within {timeout}s")
            self.backend_proc.join(timeout=backend_timeout)   # drains queued frames + full-graph optimisation
            if self.backend_proc.is_alive() and self.abort_event is not None:
                log.warning(f"Backend still busy after {backend_timeout}s -- aborting it.")
                self.abort_event.set()
                self.backend_proc.join(timeout=10.0)
            self._terminate(self.backend_proc, f"did not stop within {backend_timeout}s")
            (self.dense_stop_event or self.stop_event).set()
            self.dense_proc.join(timeout=timeout)
            self._terminate(self.dense_proc, f"did not stop within {timeout}s")
        except KeyboardInterrupt:
            # children ignore Ctrl-C now, so a SECOND Ctrl-C must still be able to end everything
            log.warning("Second interrupt -- forcing all processes down now (the map may be incomplete).")
            for ev in (self.stop_event, self.abort_event, self.dense_stop_event):
                if ev is not None:
                    ev.set()
            for p in (self.tracker_proc, self.backend_proc, self.dense_proc):
                self._terminate(p, "forced stop")
        self._drain_shared_memory()

    def _drain_shared_memory(self) -> None:
        """Free any shared-memory packets still sitting in queues after every process is gone."""
        from pyslam.live.ipc import free_pose_packet, free_points_packet
        for q, freer in ((self.tracker_pose_queue, free_pose_packet), (self.dense_pose_queue, free_pose_packet),
                         (self.points_queue, free_points_packet)):
            if q is None:
                continue
            try:
                while True:
                    freer(q.get_nowait())
            except Exception:
                pass

    def poll_qa(self) -> Optional[dict]:
        try:
            latest = None
            while True:
                latest = self.qa_queue.get_nowait()
        except Exception:
            pass
        return latest


def start_live_mapping(cfg: Config, source_factory, intrinsics, capture_profile=None,
                        out_ply_path: Optional[str] = None, backend_prefer: str = "auto",
                        R_body_cam=None, voxel_size_m: float = 0.02,
                        mp_start_method: str = "spawn") -> LiveMappingSession:
    """The real entry point for live (multi-process) mapping. Spawns
    all three processes and returns a handle the caller (run_live_map.py)
    polls/stops -- never blocks itself, since a live mapping session is
    interactive (the operator watches QA output and decides when to
    lock the map, per the "no offline stage" approved scope: locking is
    just a snapshot, not a trigger for more computation).

    mp_start_method defaults to "spawn" for the reason in this module's
    own docstring (no duplicated device/GPU handles across the fork
    boundary) and is what the target Orin should use. NOT VALIDATED
    end-to-end with "spawn" in this project's development sandbox: this
    container's multiprocessing.Queue/Event objects fail to rebuild
    their POSIX semaphore state inside a spawned child
    (FileNotFoundError from SemLock._rebuild) -- an environment/
    container restriction on this sandbox specifically, not a defect in
    this module's own logic. The full 3-process wiring WAS validated
    end-to-end in this sandbox using mp_start_method="fork" (see
    tests/gates/test_g_live.py's orchestrator check) -- confirm "spawn"
    works on the real target hardware before trusting it there; fall
    back to "fork" (accepting the device-handle-duplication risk that
    motivated preferring spawn) only if "spawn" turns out to have the
    same restriction on the Orin, which would be surprising but is not
    yet ruled out by anything tested here.
    """
    ctx = mp.get_context(mp_start_method)
    frame_queue = ctx.Queue(maxsize=64)
    points_queue = ctx.Queue(maxsize=64)
    tracker_pose_queue = ctx.Queue(maxsize=4)
    dense_pose_queue = ctx.Queue(maxsize=4)
    qa_queue = ctx.Queue(maxsize=8)
    stop_event = ctx.Event()         # tracker
    abort_event = ctx.Event()        # backend hard-abort (clean stop is the end-of-stream sentinel)
    dense_stop_event = ctx.Event()   # dense (set only after the backend has finished)
    meta_queue = ctx.Queue(maxsize=1)    # Batch-A: tracker -> backend, ONE message: device intrinsics + R_body_cam

    backend_proc = ctx.Process(
        name="pyslam-backend", target=_backend_entry,
        args=(cfg, intrinsics, frame_queue, tracker_pose_queue, dense_pose_queue,
              abort_event, backend_prefer, R_body_cam, meta_queue))
    dense_proc = ctx.Process(
        name="pyslam-dense", target=_dense_entry,
        args=(points_queue, dense_pose_queue, dense_stop_event, out_ply_path, voxel_size_m,
              int(getattr(cfg, "dense_export_min_neighbors", 0))))
    tracker_proc = ctx.Process(
        name="pyslam-tracker", target=_tracker_entry,
        args=(cfg, source_factory, capture_profile, frame_queue, points_queue,
              tracker_pose_queue, stop_event, qa_queue, meta_queue))

    backend_proc.start()
    dense_proc.start()
    tracker_proc.start()
    log.info(f"Live mapping session started: tracker pid={tracker_proc.pid}, "
             f"backend pid={backend_proc.pid}, dense pid={dense_proc.pid}")
    return LiveMappingSession(tracker_proc=tracker_proc, backend_proc=backend_proc,
                               dense_proc=dense_proc, stop_event=stop_event, qa_queue=qa_queue,
                               abort_event=abort_event, dense_stop_event=dense_stop_event,
                               tracker_pose_queue=tracker_pose_queue, dense_pose_queue=dense_pose_queue,
                               points_queue=points_queue)
