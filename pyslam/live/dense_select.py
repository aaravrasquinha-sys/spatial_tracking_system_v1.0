"""
Batch-C: which tracker keyframes go into the dense map, and at which pose (cfg.dense_pose_mode).

A tracker keyframe is not always a backend graph node. When it is not, its pose is built as
(pose of its reference node) x (the frame-to-keyframe offset measured by odometry). That offset is a raw
measurement, never corrected by the graph, so a wall seen from several such keyframes can land a few cm apart
in the voxel map and show up as a thick wall. The modes trade coverage against that risk:
    "all"   every tracker keyframe (previous behaviour)
    "nodes" only keyframes that ARE graph nodes: their pose is an optimised graph pose (the original repo's behaviour)
    "near"  every keyframe whose offset from its reference node is small (dense_near_max_offset_m / _deg)
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core import lie


def select_dense_pose(rec: Optional[dict], poses: dict, cfg) -> Optional[np.ndarray]:
    """rec: one entry of PipelineResult.frame_records for the tracker keyframe's frame (or None). Returns the
    world<-cam pose to draw it at, or None if it should not be drawn."""
    if rec is None or rec["status"] != "OK":      # a LOST frame's T_ref_frame is a fabricated identity
        return None
    ref = rec["ref_kf_id"]
    if ref is None or ref not in poses:
        return None
    mode = getattr(cfg, "dense_pose_mode", "all") if cfg is not None else "all"
    T = rec["T_ref_frame"]
    if mode == "nodes":
        if not rec["is_keyframe"]:
            return None
    elif mode == "near":
        xi = lie.se3_log(T)
        if (float(np.linalg.norm(xi[:3])) > float(cfg.dense_near_max_offset_m)
                or float(np.degrees(np.linalg.norm(xi[3:]))) > float(cfg.dense_near_max_offset_deg)):
            return None
    return poses[ref] @ T
