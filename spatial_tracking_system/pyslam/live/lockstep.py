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

log = get_logger("live.lockstep")


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
    for frame in frames:
        snap = tracker.process_frame(frame)
        qa.add_frame(snap)
        if snap.is_keyframe and tracker._cur_fused is not None:
            K = frame.intr.K()
            pts, _ = tracker._cur_fused.get_points(K, stride=4)
            if pts.shape[0] > 0:
                rgb = frame.rgb
                # colour sampled at the SAME stride/pixels get_points used
                colors = rgb[::4, ::4].reshape(-1, 3)[:pts.shape[0]]
                dense.on_keyframe_points(frame.frame_id, pts, colors)

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
    fid_to_nid = {rec["frame_id"]: rec["ref_kf_id"] for rec in result.frame_records
                  if rec["is_keyframe"] and rec["ref_kf_id"] is not None}
    poses = pipeline.graph.get_poses()
    dense_poses = {fid: poses[nid] for fid, nid in fid_to_nid.items() if nid in poses}
    dense.on_pose_update(dense_poses)

    return LockstepResult(pipeline_result=result, pipeline=pipeline, tracker=tracker,
                           dense=dense, qa=qa)
