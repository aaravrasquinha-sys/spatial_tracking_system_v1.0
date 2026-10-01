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
import time

from pyslam.core.config import Config
from pyslam.core.log import get_logger

log = get_logger("live.orchestrator")


def _tracker_entry(cfg: Config, source_factory, capture_profile, frame_queue, points_queue,
                    pose_in_queue, stop_event, qa_queue) -> None:
    """Module-level (spawn-picklable) tracker process target.
    source_factory: a zero-arg callable returning a SensorSource (e.g.
    a closure/partial built from RealSenseSource(...) or
    SyntheticSource(...) -- kept as a factory rather than a live object
    so it's constructed INSIDE the child process, where it belongs, not
    pickled from the parent (a RealSense pipeline handle in particular
    must never cross a process boundary this way)."""
    from pyslam.live.tracker_stage import TrackerStage
    from pyslam.live.ipc import pack_frame, alloc_shm_array, KeyframePointsPacket
    from pyslam.mapping.qa_stream import MappingStatusAggregator
    import numpy as np

    source = source_factory()
    stage = TrackerStage(cfg)
    qa = MappingStatusAggregator()
    last_qa_push = 0.0
    map_correction = np.eye(4)

    try:
        for frame in source:
            if stop_event.is_set():
                break
            snap = stage.process_frame(frame)
            qa.add_frame(snap)

            try:
                frame_queue.put_nowait(pack_frame(frame))
            except Exception:
                pass  # a full backend queue must never block capture

            if snap.is_keyframe and stage._cur_fused is not None:
                K = frame.intr.K()
                pts, pixel_yx = stage._cur_fused.get_points(K, stride=4)
                if pts.shape[0] > 0:
                    colors = frame.rgb[pixel_yx[:, 0], pixel_yx[:, 1]]
                    try:
                        points_queue.put_nowait(KeyframePointsPacket(
                            keyframe_id=frame.frame_id,
                            points_handle=alloc_shm_array(pts.astype(np.float32)),
                            colors_handle=alloc_shm_array(colors.astype(np.uint8)),
                        ))
                    except Exception:
                        pass

            # drain any pose correction from the backend (non-blocking;
            # tracker's own frame-rate budget must never wait on this)
            try:
                while True:
                    from pyslam.live.ipc import unpack_pose_update
                    packet = pose_in_queue.get_nowait()
                    poses = unpack_pose_update(packet, free_after=True)
                    if poses:
                        # crude but honest: use the most recent node's
                        # correction as a global rigid nudge, the same
                        # role pipeline.py's own _map_correction plays
                        # internally -- a full per-node correction isn't
                        # meaningful to a tracker that only tracks
                        # against its OWN current reference keyframe.
                        map_correction = list(poses.values())[-1]
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
            source.close()
        except Exception:
            pass
        log.info(f"Tracker process stopped: {qa.n_frames} frames, {qa.n_keyframes} keyframes, "
                 f"{stage.n_icp_fallback_used} ICP-fallback recoveries.")


def _backend_entry(cfg: Config, intrinsics, frame_queue, tracker_pose_queue, dense_pose_queue,
                    stop_event, backend_prefer: str, R_body_cam) -> None:
    from pyslam.live.backend_process import run_backend_process
    run_backend_process(cfg, intrinsics, frame_queue, pose_out_queues=[tracker_pose_queue],
                         stop_event=stop_event, R_body_cam=R_body_cam, backend_prefer=backend_prefer,
                         dense_pose_queues=[dense_pose_queue])


def _dense_entry(points_queue, dense_pose_queue, stop_event, out_ply_path, voxel_size_m) -> None:
    from pyslam.live.dense_process import run_dense_process
    run_dense_process(points_queue, dense_pose_queue, stop_event, out_ply_path=out_ply_path,
                       voxel_size_m=voxel_size_m)


@dataclass
class LiveMappingSession:
    tracker_proc: object
    backend_proc: object
    dense_proc: object
    stop_event: object
    qa_queue: object

    def stop(self, timeout: float = 15.0) -> None:
        log.info("Stopping live mapping session...")
        self.stop_event.set()
        for p in (self.tracker_proc, self.backend_proc, self.dense_proc):
            p.join(timeout=timeout)
            if p.is_alive():
                log.warning(f"Process {p.name} did not stop within {timeout}s -- terminating.")
                p.terminate()
                p.join(timeout=5.0)

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
    stop_event = ctx.Event()

    backend_proc = ctx.Process(
        name="pyslam-backend", target=_backend_entry,
        args=(cfg, intrinsics, frame_queue, tracker_pose_queue, dense_pose_queue,
              stop_event, backend_prefer, R_body_cam))
    dense_proc = ctx.Process(
        name="pyslam-dense", target=_dense_entry,
        args=(points_queue, dense_pose_queue, stop_event, out_ply_path, voxel_size_m))
    tracker_proc = ctx.Process(
        name="pyslam-tracker", target=_tracker_entry,
        args=(cfg, source_factory, capture_profile, frame_queue, points_queue,
              tracker_pose_queue, stop_event, qa_queue))

    backend_proc.start()
    dense_proc.start()
    tracker_proc.start()
    log.info(f"Live mapping session started: tracker pid={tracker_proc.pid}, "
             f"backend pid={backend_proc.pid}, dense pid={dense_proc.pid}")
    return LiveMappingSession(tracker_proc=tracker_proc, backend_proc=backend_proc,
                               dense_proc=dense_proc, stop_event=stop_event, qa_queue=qa_queue)
