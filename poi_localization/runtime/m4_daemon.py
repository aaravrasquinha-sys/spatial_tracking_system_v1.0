"""
Runs M3 and M4 together, live. Section 13: "Module 6 will eventually own
the thread wiring that feeds Module 4 from Module 3's queue" -- until
Module 6 exists, this daemon is the simplest correct wiring: it reuses
poi_perception's M3Daemon (capture + inference threads, unmodified) and
chains M4Pipeline.process() through M3Pipeline's existing on_detections
hook, so M3's code needed zero changes to support this.

    [M3Daemon: capture thread] -> [M3Daemon: inference thread]
        -> [M3Pipeline.process() on the daemon's main thread]
            -> on_detections callback -> [M4Pipeline.process()]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from poi_localization.config import M4Config
from poi_localization.frames.frame_provider import build_frame_provider
from poi_localization.io.world_track import JsonlWorldTrackWriter
from poi_localization.log import get_logger
from poi_localization.runtime.m4_pipeline import M4Pipeline
from poi_localization.tracking.gates import load_walkable_grid
from poi_perception.config import M3Config
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.io.detection2d import JsonlDetectionWriter
from poi_perception.runtime.m3_daemon import M3Daemon, M3Pipeline, build_backend, build_source
from poi_perception.runtime.masks import empty_mask_set, load_masks

log = get_logger("runtime.m4_daemon")


def build_m4_pipeline(m4_cfg: M4Config, cam_cfg=None, writer: Optional[JsonlWorldTrackWriter] = None) -> M4Pipeline:
    provider = build_frame_provider(m4_cfg, cam_cfg=cam_cfg)
    walkable_grid = None
    if m4_cfg.phase == "B" and m4_cfg.phase_b.walkable_grid_path:
        walkable_grid = load_walkable_grid(m4_cfg.phase_b.walkable_grid_path)
    return M4Pipeline(m4_cfg, provider, walkable_grid=walkable_grid, writer=writer)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m3-config", required=True, help="path to an M3Config JSON file")
    ap.add_argument("--m4-config", required=True, help="path to an M4Config JSON file")
    ap.add_argument("--source", choices=["realsense", "synthetic"], default="realsense")
    ap.add_argument("--playback", default=None, help="replay a recorded .bag instead of a live camera")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-m3-log", action="store_true", help="skip writing M3's own Detection2D JSONL log")
    args = ap.parse_args(argv)

    m3_cfg = M3Config.load(args.m3_config)
    m4_cfg = M4Config.load(args.m4_config)

    # Phase A's live calibration capture (if not using a fixed_transform
    # override) needs its own short-lived RealSense pipeline, opened and
    # closed BEFORE the main capture source starts -- see
    # frames/local_floor_frame.py's capture_calibration_data docstring.
    m4_writer_path = Path(m4_cfg.output.jsonl_dir) / f"{m3_cfg.camera.cam_id}_{int(time.time())}.jsonl"
    m4_writer = JsonlWorldTrackWriter(m4_writer_path, phase=m4_cfg.phase)
    m4_pipeline = build_m4_pipeline(m4_cfg, cam_cfg=m3_cfg.camera, writer=m4_writer)
    # If Phase B, the loaded calibration's map_id is now known -- attach
    # it to the writer for the JSONL header discipline (Section 6's M6
    # map_id check, applied here at the writer level).
    m4_writer.map_id = m4_pipeline.floor_transform.map_id

    source = build_source(m3_cfg, args.source, args.playback)
    backend = build_backend(m3_cfg)
    estimator = PoseEstimator(backend)

    mask_set = None
    if m3_cfg.mask.masks_path:
        mask_set = load_masks(m3_cfg.mask.masks_path).get(m3_cfg.camera.cam_id, empty_mask_set())

    m3_writer = None
    if not args.no_m3_log:
        m3_jsonl_path = Path(m3_cfg.output.jsonl_dir) / f"{m3_cfg.camera.cam_id}_{int(time.time())}.jsonl"
        m3_writer = JsonlDetectionWriter(m3_jsonl_path)

    m3_pipeline = M3Pipeline(
        m3_cfg,
        mask_set=mask_set,
        writer=m3_writer,
        on_detections=lambda frame, dets: m4_pipeline.process(frame, dets),
    )
    daemon = M3Daemon(m3_cfg, source, estimator, m3_pipeline)

    log.info(f"M4 phase={m4_cfg.phase}, floor_transform camera_height="
             f"{m4_pipeline.floor_transform.camera_height_m:.3f}m")
    try:
        daemon.run(max_frames=args.max_frames)
    finally:
        m4_writer.close()
        if m3_writer is not None:
            m3_writer.close()


if __name__ == "__main__":
    main()
