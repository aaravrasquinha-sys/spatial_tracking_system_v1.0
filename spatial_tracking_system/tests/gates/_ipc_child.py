"""
Helper module for tests/gates/test_g_live_ipc.py -- the child-process
target function must be importable (module-level, not a closure/lambda)
for multiprocessing's 'spawn' start method, which is the default on
some platforms this project may run gates on (macOS, and available on
Linux) -- keeping it here rather than inline in the test module avoids
a pickling failure that would otherwise depend on which start method
happened to be active.
"""
from __future__ import annotations
import numpy as np


def _child_read_frame_packet(packet, result_queue) -> None:
    from pyslam.live.ipc import unpack_frame
    frame = unpack_frame(packet, free_after=True)
    result_queue.put({
        "rgb_sum": int(frame.rgb.sum()),
        "depth_sum": int(frame.depth.sum()),
        "t": frame.t,
        "frame_id": frame.frame_id,
        "imu_shape": None if frame.imu is None else tuple(frame.imu.shape),
    })


def _child_read_pose_update(packet, result_queue) -> None:
    from pyslam.live.ipc import unpack_pose_update
    poses = unpack_pose_update(packet, free_after=True)
    result_queue.put({k: v.tolist() for k, v in poses.items()})
