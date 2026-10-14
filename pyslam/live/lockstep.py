"""
WP-LIVE: lockstep mode. Runs the SAME logic the multi-process live
architecture uses, but synchronously in one process/one thread --
deterministic, debuggable, and what selftest/bag-replay/gates should
use (mirrors the rest of this codebase's own established split between
a deterministic single-threaded gate-suite path and a faster/riskier
live path, e.g. pipeline.py's single-threaded design vs. this
package's multi-process one).

For the TRACKER-side logic (odometry/ICP-fallback/keyframe-decision/
fused-depth/QA), lockstep mode calls pyslam.live.tracker_stage.TrackerStage
directly, frame by frame -- no IPC, no queues, just Python objects.

For the BACKEND-side logic (memory/retrieval/verify/local_reloc/
proximity/gravity-prior/graph), lockstep mode uses
pyslam.pipeline.Pipeline directly (pipeline.run(source)) -- exactly
today's existing, already-gated single-process behaviour, unmodified.
This means lockstep mode does NOT exercise pyslam.live.backend_process's
QueueSensorSource/threading wrapper at all (that machinery exists
specifically for the multi-process case) -- it exercises the SAME
Pipeline class both modes ultimately rely on, which is the important
invariant: a bag replayed in lockstep mode and the live multi-process
run of the same bag should produce the same graph, because both are
"Pipeline.run() over the same frame sequence" underneath, just wired
differently.

Dense-side (Level 1/2) is run inline too, via DenseProcessState fed
directly by TrackerStage's fused-depth output and Pipeline's pose
updates after each keyframe -- same reasoning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from pyslam.core.types import SensorSource
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.pipeline import Pipeline
from pyslam.live.tracker_stage import TrackerStage
from pyslam.live.dense_process import DenseProcessState
from pyslam.mapping.qa_stream import MappingStatusAggregator
from pyslam.live.dense_select import select_dense_pose

log = get_logger("live.lockstep")


def dense_frame_poses(poses: dict, frame_records: list, frame_ids, cfg=None) -> dict:
    """Batch-1 (N6): {frame_id: world<-cam} for the given TRACKER keyframe ids, computed as
    pose[ref keyframe] @ T_ref_frame. Works even when the backend's own keyframes differ from the
    tracker's; frames the backend marked LOST are skipped (their T_ref_frame is a fabricated identity)."""
    rec_by_fid = {rec["frame_id"]: rec for rec in frame_records}
    out = {}
    for fid in frame_ids:
        T = select_dense_pose(rec_by_fid.get(fid), poses, cfg)      # Batch-C: dense_pose_mode (cfg=None -> "all")
        if T is not None:
            out[fid] = T
    return out


@dataclass
class LockstepResult:
    pipeline_result: object   # PipelineResult
    pipeline: Pipeline
    tracker: TrackerStage
    dense: DenseProcessState
    qa: MappingStatusAggregator


def run_lockstep(cfg: Config, source: SensorSource, max_frames: Optional[int] = None,
                  R_body_cam: Optional[np.ndarray] = None, backend_prefer: str = "auto",
                  voxel_size_m: float = 0.02, verbose: bool = False) -> LockstepResult:
    """Runs tracker-side processing (for QA and dense Level 1) and the
    full Pipeline (for the graph) over the SAME frame sequence. The
    tracker and the Pipeline's own internal odometry both run frame by
    frame here -- same duplicated-frontend-cost tradeoff
    backend_process.py's module docstring documents for the live case,
    accepted for the same reason (reuse Pipeline unmodified). Since
    this is single-process and deterministic, the duplication only
    costs wall-clock time, never correctness."""
    tracker = TrackerStage(cfg)
    pipeline = Pipeline(cfg, backend_prefer=backend_prefer, R_body_cam=R_body_cam)
    dense = DenseProcessState(voxel_size_m=voxel_size_m)
    qa = MappingStatusAggregator()

    frames = []
    n = 0
    for frame in source:
        frames.append(frame)
        n += 1
        if max_frames is not None and n >= max_frames:
            break

    # tracker + dense pass (frame rate)
    def _feed_dense(kf):
        pts, pixel_yx = kf.fused.get_points(kf.K, stride=4,
                                            max_depth_m=float(getattr(cfg, "dense_max_depth_m", 0.0)),
                                            edge_rel=float(getattr(cfg, "dense_edge_rel", 0.0)))
        if pts.shape[0] > 0:
            # Batch-1 fix: colour is sampled at the EXACT pixels get_points returned.
            dense.on_keyframe_points(kf.frame_id, pts, kf.rgb[pixel_yx[:, 0], pixel_yx[:, 1]])

    for frame in frames:
        snap = tracker.process_frame(frame)
        qa.add_frame(snap)
        for kf in tracker.take_finished_keyframes():        # Batch-C: a keyframe is ready once its depth is done
            _feed_dense(kf)
    for kf in tracker.flush():
        _feed_dense(kf)

    # backend pass (keyframe rate, via the existing validated Pipeline)
    class _ReplaySource:
        def intrinsics(self):
            return frames[0].intr if frames else None

        def close(self):
            pass

        def __iter__(self):
            return iter(frames)

    result = pipeline.run(_ReplaySource(), verbose=verbose)
    # WP-LIVE dense hookup fix (see backend_process.py's own note): dense
    # stores keyframe points by frame_id (tracker's numbering); graph
    # node ids come from a separate counter. Translate via
    # frame_records before handing poses to dense.
    poses = pipeline.graph.get_poses()
    dense.on_pose_update(dense_frame_poses(poses, result.frame_records, list(dense._stored.keys()), cfg))

    return LockstepResult(pipeline_result=result, pipeline=pipeline, tracker=tracker,
                           dense=dense, qa=qa)
