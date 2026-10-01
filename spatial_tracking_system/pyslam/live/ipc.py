"""
WP-LIVE: shared-memory IPC primitives for the multi-process live
architecture (approved: process isolation, "no threads before P6"
relaxed ONLY for this). Big arrays (rgb, depth, fused-depth point
clouds) cross process boundaries via multiprocessing.shared_memory --
zero-copy on the same machine (the Orin), never pickled through a
Queue -- while small metadata (pose, ids, timestamps) goes through an
ordinary multiprocessing.Queue as a plain picklable dataclass carrying
only the shared-memory block NAMES plus shape/dtype, never the array
data itself.

Ownership rule, stated once here because it's easy to get wrong: the
PRODUCER allocates a shared-memory block and hands its handle
(ShmHandle) downstream; the LAST consumer in the pipeline that no
longer needs the block is responsible for calling free_shm_array() on
it. In this architecture that's always the dense process for frame
rgb/depth (tracker -> backend AND tracker -> dense both read the same
block; dense frees it) -- see orchestrator.py for the actual reference-
counting discipline (a simple "both consumers must ack" scheme, since
this is a small, fixed 2-consumer fan-out, not a general pub/sub
system).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import shared_memory
import numpy as np


def _untrack(shm) -> None:
    """Known CPython multiprocessing.shared_memory gotcha, worked
    around the standard way: EVERY process that so much as opens a
    SharedMemory block (even read-only, even just to attach) registers
    it with THAT PROCESS's own local resource_tracker for automatic
    cleanup-on-exit. In this architecture a block is deliberately
    opened by multiple processes across its lifetime (producer creates
    it, one or more consumers attach to read it) -- with automatic
    tracking left on, ANY of those processes exiting (or even just
    garbage-collecting its local SharedMemory handle) can unlink the
    block out from under a consumer that hasn't read it yet, producing
    exactly the "FileNotFoundError: [Errno 2] No such file or
    directory" race this function exists to prevent. Explicitly
    unregistering from the CURRENT process's tracker immediately after
    create/attach makes ownership explicit instead: nobody auto-unlinks
    anything, and free_shm_array() (called by whichever consumer this
    module's own ownership-rule docstring designates as "last") is the
    ONLY thing that ever unlinks a block. This is the standard
    workaround for this well-documented Python issue (multiple
    independent bug threads for Python 3.8-3.12 describe the same
    failure mode); the alternative of leaving automatic tracking on and
    hoping process exit ordering happens to work out was tried first in
    this work package and DID fail exactly this way under a real
    3-process run (see SYSTEM_SUMMARY_LIVE.md's own note on this)."""
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass  # best-effort: an already-unregistered or platform-without-
              # resource_tracker (unlikely on Linux/the Orin target) case
              # must never crash a live process over bookkeeping hygiene

from pyslam.core.types import Frame, Intrinsics


@dataclass
class ShmHandle:
    name: str
    shape: tuple
    dtype: str  # np.dtype.name, e.g. "uint8", "float32"

    def read(self, copy: bool = True) -> np.ndarray:
        shm = shared_memory.SharedMemory(name=self.name)
        _untrack(shm)
        try:
            arr = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=shm.buf)
            return arr.copy() if copy else arr
        finally:
            # always close OUR handle to the mapping (does not unlink the
            # underlying block -- that's a separate, explicit step, see
            # free_shm_array). Closing here even when copy=False is safe:
            # numpy's .copy() above already detached from the buffer, and
            # a caller passing copy=False is expected to have its own
            # reason to keep the mapping open (rare; not used elsewhere
            # in this codebase) and should manage shm.close() itself in
            # that case by calling shared_memory.SharedMemory(name=...)
            # again rather than relying on this method's returned view.
            if copy:
                shm.close()


def alloc_shm_array(arr: np.ndarray) -> ShmHandle:
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    _untrack(shm)
    dst = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    dst[:] = arr[:]
    handle = ShmHandle(name=shm.name, shape=tuple(arr.shape), dtype=arr.dtype.name)
    shm.close()  # detach from THIS process's mapping; the block itself persists
                 # (backed by the OS) until free_shm_array() unlinks it
    return handle


def free_shm_array(handle: ShmHandle) -> None:
    try:
        shm = shared_memory.SharedMemory(name=handle.name)
        _untrack(shm)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass  # already freed -- double-free must not crash a live process


@dataclass
class FramePacket:
    """IPC-safe stand-in for pyslam.core.types.Frame. rgb/depth cross
    via shared memory (handles only, in this packet); imu is small
    enough (a handful of rows per frame at most) to pickle directly
    through the Queue rather than allocating a shm block per frame for
    it."""
    t: float
    frame_id: int
    rgb_handle: ShmHandle
    depth_handle: ShmHandle
    imu: Optional[np.ndarray]
    gt_pose: Optional[np.ndarray]
    # Intrinsics fields flattened (Intrinsics itself is a frozen
    # dataclass of plain floats/ints, trivially picklable -- carried
    # here rather than re-sent every frame from a separate channel,
    # since a Frame is meaningless without its own intrinsics and this
    # keeps FramePacket self-contained).
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float
    baseline: float


def pack_frame(frame: Frame) -> FramePacket:
    return FramePacket(
        t=frame.t, frame_id=frame.frame_id,
        rgb_handle=alloc_shm_array(frame.rgb), depth_handle=alloc_shm_array(frame.depth),
        imu=(frame.imu.copy() if frame.imu is not None else None),
        gt_pose=(frame.gt_pose.copy() if frame.gt_pose is not None else None),
        fx=frame.intr.fx, fy=frame.intr.fy, cx=frame.intr.cx, cy=frame.intr.cy,
        width=frame.intr.width, height=frame.intr.height,
        depth_scale=frame.intr.depth_scale, baseline=frame.intr.baseline,
    )


def unpack_frame(packet: FramePacket, free_after: bool = False) -> Frame:
    """free_after=True unlinks the shared-memory blocks once read --
    set by whichever consumer is the LAST to need this frame's imagery
    (see module docstring's ownership rule)."""
    rgb = packet.rgb_handle.read(copy=True)
    depth = packet.depth_handle.read(copy=True)
    if free_after:
        free_shm_array(packet.rgb_handle)
        free_shm_array(packet.depth_handle)
    intr = Intrinsics(fx=packet.fx, fy=packet.fy, cx=packet.cx, cy=packet.cy,
                       width=packet.width, height=packet.height,
                       depth_scale=packet.depth_scale, baseline=packet.baseline)
    return Frame(t=packet.t, rgb=rgb, depth=depth, intr=intr, imu=packet.imu,
                 frame_id=packet.frame_id, gt_pose=packet.gt_pose)


@dataclass
class KeyframePointsPacket:
    """Tracker -> dense: one keyframe's fused-depth point cloud (its
    own camera frame, from pyslam.mapping.fused_depth.KeyframeFusedDepth.
    get_points()), plus the colour sampled at the same pixels. Sent once
    per keyframe when the tracker moves on to a new reference (i.e. this
    keyframe's fusion is considered "settled" for now -- WP-LIVE dense
    Level 1's own re-fusion-on-demand note in fused_depth.py's docstring
    covers what happens if a loop closure later revises this keyframe's
    pose, which is purely a DENSE-side (Level 2) re-integration using
    the SAME point cloud, no re-send needed)."""
    keyframe_id: int
    points_handle: ShmHandle   # (N,3) float32, keyframe camera frame
    colors_handle: ShmHandle   # (N,3) uint8


@dataclass
class PoseUpdatePacket:
    """Backend -> tracker AND backend -> dense: current best pose
    estimate(s), pushed after every incremental smoothing update.
    poses: {node_id: 4x4 flattened as a (N,4,4) float64 array} sent as
    two parallel small arrays (ids, poses) through shared memory when N
    is large, or directly pickled when small -- this implementation
    always uses shared memory for uniformity and because pose updates
    can grow to hundreds of nodes on a long mapping run."""
    ids_handle: ShmHandle      # (N,) int64
    poses_handle: ShmHandle    # (N,4,4) float64
    generation: int            # monotonically increasing, so a stale
                                # update received out of order can be
                                # detected and dropped by the consumer


def pack_pose_update(poses: dict, generation: int) -> PoseUpdatePacket:
    ids = np.array(list(poses.keys()), dtype=np.int64)
    mats = np.stack([poses[i] for i in ids], axis=0).astype(np.float64) if ids.size else np.zeros((0, 4, 4))
    return PoseUpdatePacket(ids_handle=alloc_shm_array(ids), poses_handle=alloc_shm_array(mats),
                             generation=generation)


def unpack_pose_update(packet: PoseUpdatePacket, free_after: bool = True) -> dict:
    ids = packet.ids_handle.read(copy=True)
    mats = packet.poses_handle.read(copy=True)
    if free_after:
        free_shm_array(packet.ids_handle)
        free_shm_array(packet.poses_handle)
    return {int(ids[i]): mats[i] for i in range(ids.shape[0])}
