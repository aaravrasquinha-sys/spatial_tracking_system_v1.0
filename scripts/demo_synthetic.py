#!/usr/bin/env python3
"""
Run the full M3 pipeline end-to-end with no hardware at all: a
SyntheticSource feeding a MockPoseBackend running one of Section 9's
scripted scenarios, through the real tracker/footpoint/mask code.

    python3 scripts/demo_synthetic.py --scenario two_crossing --frames 90
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="single_loop")
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()

    cfg = M3Config.default()
    source = SyntheticSource(cfg.camera, n_frames=args.frames)
    scenario_fn = SCENARIOS[args.scenario](frame_w=cfg.camera.width, frame_h=cfg.camera.height, n_frames=args.frames)
    estimator = PoseEstimator(mock_infer.MockPoseBackend(scenario_fn))
    pipeline = M3Pipeline(cfg, mask_set=empty_mask_set())

    for frame in source:
        raw_dets = estimator.infer(frame)
        dets = pipeline.process(frame, raw_dets)
        summary = ", ".join(
            f"id={d.track_id_2d}:{d.track_state}:{d.footpoint_source}" for d in dets
        )
        print(f"frame {frame.frame_id:4d}  t={frame.t:6.2f}  n={len(dets)}  [{summary}]")


if __name__ == "__main__":
    main()
