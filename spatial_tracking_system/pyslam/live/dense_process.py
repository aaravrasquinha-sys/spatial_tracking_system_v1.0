"""
WP-LIVE: the DENSE process. Consumes keyframe fused-depth point clouds
from the tracker (KeyframePointsPacket, each keyframe's own camera
frame) and pose updates from the backend (PoseUpdatePacket, world-frame
poses per node id), transforms each keyframe's points into the world
frame using its CURRENT pose, and integrates them into the Level-2
store (pyslam.mapping.dense_voxel -- the CPU VoxelHashFuser by default;
see pyslam/tools/nvblox_eval.py for the bounded evaluation deciding
whether nvblox_torch replaces it, behind the same DenseBackend
Protocol).

Re-fusion on loop closure: each keyframe is tracked as one "submap" of
exactly one keyframe (the simplest possible submap-to-keyframe mapping
-- the planning notes' Level-2 "group consecutive keyframes into
submaps of a few metres" grouping is a natural follow-up for voxel-
hash memory efficiency, not required for correctness). When a pose
update packet reports a NEW pose for a keyframe whose points are
already integrated, this process re-integrates that keyframe's stored
point cloud at the new pose with an incremented generation counter --
VoxelHashFuser.integrate_submap's own generation-tagged overwrite
handles the "replace, don't duplicate" semantics (see dense_voxel.py).
This is what makes a loop closure correction show up in the live dense
map without any offline recomputation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import queue
import time
import numpy as np

from pyslam.core.log import get_logger
from pyslam.core import lie
from pyslam.mapping.dense_voxel import VoxelHashFuser
from pyslam.live.ipc import unpack_pose_update, ShmHandle

log = get_logger("live.dense_process")


@dataclass
class _StoredKeyframePoints:
    points_cam: np.ndarray   # (N,3), keyframe's own camera frame
    colors: np.ndarray       # (N,3) uint8
    last_pose: Optional[np.ndarray] = None  # world<-cam, last pose this was integrated at


class DenseProcessState:
    def __init__(self, voxel_size_m: float = 0.02, reintegrate_pose_delta_m: float = 0.03,
                 reintegrate_pose_delta_deg: float = 2.0, max_stored_keyframes: Optional[int] = None):
        self.fuser = VoxelHashFuser(voxel_size_m=voxel_size_m)
        self._stored: dict[int, _StoredKeyframePoints] = {}
        self._generation = 0
        self._pose_delta_m = reintegrate_pose_delta_m
        self._pose_delta_deg = reintegrate_pose_delta_deg
        self._max_stored = max_stored_keyframes  # None = unbounded (fine for a room-scale
            # capture; a facility-scale run would want this set, paging Level-1 points to disk
            # the same way memory/ltm_store.py already does for the SLAM graph -- flagged as a
            # WP-LIVE follow-up for the "any indoor space" scaling case, not required here)

    def on_keyframe_points(self, keyframe_id: int, points_cam: np.ndarray, colors: np.ndarray) -> None:
        self._stored[keyframe_id] = _StoredKeyframePoints(points_cam=points_cam, colors=colors)
        # integrate immediately at IDENTITY if no pose has arrived yet --
        # better to show something (even if it moves once a real pose
        # arrives) than to show nothing; the next on_pose_update call
        # will re-integrate at the corrected pose as soon as it exists.

    def on_pose_update(self, poses: dict) -> int:
        """Returns how many keyframes were (re)integrated this call."""
        n_updated = 0
        for kf_id, T_world_cam in poses.items():
            stored = self._stored.get(kf_id)
            if stored is None:
                continue
            if stored.last_pose is not None:
                delta = lie.se3_log(lie.se3_inverse(stored.last_pose) @ T_world_cam)
                moved_m = float(np.linalg.norm(delta[:3]))
                moved_deg = float(np.degrees(np.linalg.norm(delta[3:])))
                if moved_m < self._pose_delta_m and moved_deg < self._pose_delta_deg:
                    continue  # pose hasn't moved enough to be worth re-integrating
            pts_world = lie.transform_points(T_world_cam, stored.points_cam)
            self._generation += 1
            self.fuser.drop_submap(kf_id)
            self.fuser.integrate_submap(kf_id, pts_world, stored.colors, generation=self._generation)
            stored.last_pose = T_world_cam.copy()
            n_updated += 1
        return n_updated

    def export_ply_points(self, max_points: Optional[int] = 2_000_000):
        return self.fuser.export_points(max_points=max_points)


def run_dense_process(points_queue, pose_queue, stop_event, out_ply_path: Optional[str] = None,
                       export_interval_s: float = 5.0, voxel_size_m: float = 0.02,
                       get_timeout_s: float = 0.5) -> None:
    """The multiprocessing.Process target. Two independent input
    channels (points from tracker, poses from backend) are drained in
    one loop with short timeouts on each so neither one can starve the
    other; a periodic PLY export gives the live viewer (M5a-equivalent
    for this repo, or a debug tool) something to read without needing
    its own IPC channel into this process."""
    from pyslam.live.ipc import KeyframePointsPacket
    state = DenseProcessState(voxel_size_m=voxel_size_m)
    last_export = time.time()
    while not stop_event.is_set():
        drained_any = False
        try:
            pp: KeyframePointsPacket = points_queue.get(timeout=get_timeout_s)
            pts = pp.points_handle.read(copy=True)
            colors = pp.colors_handle.read(copy=True)
            from pyslam.live.ipc import free_shm_array
            free_shm_array(pp.points_handle)
            free_shm_array(pp.colors_handle)
            state.on_keyframe_points(pp.keyframe_id, pts, colors)
            drained_any = True
        except queue.Empty:
            pass

        try:
            posepkt = pose_queue.get(timeout=0.01)
            poses = unpack_pose_update(posepkt, free_after=True)
            state.on_pose_update(poses)
            drained_any = True
        except queue.Empty:
            pass

        if out_ply_path is not None and (time.time() - last_export) > export_interval_s:
            pts, colors = state.export_ply_points()
            if pts.shape[0] > 0:
                from pyslam.mapping.export import write_ply
                write_ply(out_ply_path, pts, colors)
            last_export = time.time()

        if not drained_any:
            time.sleep(0.01)

    # WP-LIVE fix: after stop_event fires, drain any last-moment
    # messages for a short grace window -- the backend process pushes
    # ONE final pose broadcast only AFTER its own Pipeline thread fully
    # drains (backend_process.py's own shutdown sequence), which can
    # arrive slightly after this process's own stop_event check exits
    # the main loop above. Without this grace drain, the final export
    # below could race ahead of that last correction and miss it.
    grace_deadline = time.time() + 2.0
    while time.time() < grace_deadline:
        drained = False
        try:
            pp = points_queue.get(timeout=0.1)
            pts = pp.points_handle.read(copy=True)
            colors = pp.colors_handle.read(copy=True)
            from pyslam.live.ipc import free_shm_array
            free_shm_array(pp.points_handle)
            free_shm_array(pp.colors_handle)
            state.on_keyframe_points(pp.keyframe_id, pts, colors)
            drained = True
        except queue.Empty:
            pass
        try:
            posepkt = pose_queue.get(timeout=0.1)
            poses = unpack_pose_update(posepkt, free_after=True)
            state.on_pose_update(poses)
            drained = True
        except queue.Empty:
            pass
        if not drained:
            break  # nothing arrived in this pass -- no point waiting out the full grace window

    # WP-LIVE fix: always write one final export on shutdown, regardless
    # of export_interval_s timing -- a session stopped shortly after
    # starting (exactly what a short test run, or an operator's quick
    # "lock now" does) could otherwise exit before the periodic export
    # above ever fires even once, silently producing no output at all.
    if out_ply_path is not None:
        pts, colors = state.export_ply_points()
        if pts.shape[0] > 0:
            from pyslam.mapping.export import write_ply
            write_ply(out_ply_path, pts, colors)

    log.info(f"Dense process stopped: {state.fuser.n_voxels()} voxels.")
