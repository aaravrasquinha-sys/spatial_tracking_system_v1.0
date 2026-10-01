#!/usr/bin/env python3
"""
Run the full M3+M4 pipeline end-to-end with no hardware: SyntheticSource
+ MockPoseBackend (M3's Section 9 scenarios) through the real M3 tracker/
footpoint code, then through the real M4 measurement/tracking code using
a fixed (non-hardware) floor-frame transform.

Note: M3's mock scenarios place people at plausible pixel positions but
don't know about M4's assumed camera height/pitch, and SyntheticSource's
depth image is a flat constant plane -- so the resulting world positions
here are NOT geometrically self-consistent ground truth (unlike
tests/test_m4_pipeline_e2e.py, which constructs pixel/depth data that
IS consistent with a known transform). This script is for exercising the
full pipeline's wiring and control flow end to end, not for judging
measurement accuracy.

    python3 scripts/demo_synthetic_m4.py --scenario two_crossing --frames 90
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_localization.config import M4Config
from poi_localization.frames.frame_provider import FixedFrameProvider
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.runtime.m4_pipeline import M4Pipeline
from poi_perception.capture.synthetic_source import SyntheticSource
from poi_perception.config import M3Config
from poi_perception.inference import mock_infer
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.runtime.m3_daemon import M3Pipeline
from poi_perception.runtime.masks import empty_mask_set

SCENARIOS = {
    "single_loop": mock_infer.scenario_single_loop,
    "two_crossing": mock_infer.scenario_two_crossing,
    "exit_reenter": mock_infer.scenario_exit_reenter,
    "sitting": mock_infer.scenario_sitting,
    "partial_occlusion": mock_infer.scenario_partial_occlusion,
    "near_mirror": mock_infer.scenario_near_mirror,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="single_loop")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--height", type=float, default=2.3)
    ap.add_argument("--pitch", type=float, default=20.0)
    args = ap.parse_args()

    m3_cfg = M3Config.default()
    m4_cfg = M4Config.default()
    m4_cfg.track.tentative_confirm_hits = 2

    source = SyntheticSource(m3_cfg.camera, n_frames=args.frames)
    scenario_fn = SCENARIOS[args.scenario](frame_w=m3_cfg.camera.width, frame_h=m3_cfg.camera.height, n_frames=args.frames)
    estimator = PoseEstimator(mock_infer.MockPoseBackend(scenario_fn))
    m3_pipeline = M3Pipeline(m3_cfg, mask_set=empty_mask_set())

    floor_transform = FloorFrameTransform.from_height_and_yaw(args.height, pitch_deg=args.pitch, yaw_deg=0.0)
    provider = FixedFrameProvider(floor_transform)
    m4_pipeline = M4Pipeline(m4_cfg, provider)

    for frame in source:
        raw_dets = estimator.infer(frame)
        m3_dets = m3_pipeline.process(frame, raw_dets)
        world_tracks = m4_pipeline.process(frame, m3_dets)
        summary = ", ".join(
            f"id={t.world_track_id}:{t.state}:src={t.src}:p=({t.p_local[0]:.2f},{t.p_local[1]:.2f}):h={t.height_m}"
            for t in world_tracks
        )
        print(f"frame {frame.frame_id:4d}  t={frame.t:6.2f}  n={len(world_tracks)}  [{summary}]")


if __name__ == "__main__":
    main()
