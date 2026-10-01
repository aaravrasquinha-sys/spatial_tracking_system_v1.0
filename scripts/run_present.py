#!/usr/bin/env python3
"""
Run Module 5 (dashboard + VR publisher).

    # Develop the viewers with no hardware at all:
    python3 scripts/run_present.py --source synthetic --synthetic-mode scripted \
        --config configs/present.example.json

    # Develop against a real recorded session:
    python3 scripts/run_present.py --source replay --replay-path logs/world_tracks/cam0_....jsonl \
        --config configs/present.example.json

    # Live, on the Orin, alongside M3+M4 (recommended once you want the
    # real thing on a phone or the headset):
    python3 scripts/run_present.py --source live \
        --m3-config configs/m3.example.json --m4-config configs/m4.phaseA.example.json \
        --present-config configs/present.example.json --m4-source realsense

`--source live` is the one mode that touches M4 at all, and it changes
NOTHING in poi_localization: it builds the exact same M4Pipeline
run_m4.py builds, and additionally passes LiveSource.submit as
M4Pipeline's on_tracks callback -- a parameter that already existed
and was simply unused. See poi_present/sources/live.py's docstring.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_present.config import PresentConfig
from poi_present.log import get_logger
from poi_present.server.app import PresentServer, install_signal_handlers
from poi_present.sources.replay import ReplaySource
from poi_present.sources.synthetic import build_synthetic_source

log = get_logger("run_present")


def _build_live(args, present_cfg: PresentConfig, server: PresentServer):
    """Wires LiveSource into a real M3+M4 daemon, the same way
    run_m4.py wires M4 into M3 -- see the module docstring above."""
    from poi_localization.config import M4Config
    from poi_localization.frames.frame_provider import build_frame_provider
    from poi_localization.io.world_track import JsonlWorldTrackWriter
    from poi_localization.runtime.m4_pipeline import M4Pipeline
    from poi_localization.tracking.gates import load_walkable_grid
    from poi_perception.config import M3Config
    from poi_perception.inference.pose_infer import PoseEstimator
    from poi_perception.io.detection2d import JsonlDetectionWriter
    from poi_perception.runtime.m3_daemon import M3Daemon, M3Pipeline, build_backend, build_source
    from poi_perception.runtime.masks import empty_mask_set, load_masks

    from poi_present.sources.live import LiveSource

    m3_cfg = M3Config.load(args.m3_config)
    m4_cfg = M4Config.load(args.m4_config)

    provider = build_frame_provider(m4_cfg, cam_cfg=m3_cfg.camera)
    walkable_grid = None
    if m4_cfg.phase == "B" and m4_cfg.phase_b.walkable_grid_path:
        walkable_grid = load_walkable_grid(m4_cfg.phase_b.walkable_grid_path)

    frame_name = "room" if m4_cfg.phase == "B" else "local"
    live_source = LiveSource(cam_id=m3_cfg.camera.cam_id, frame_name=frame_name, map_id=None)

    m4_writer = None
    if not args.no_m4_log:
        path = Path(m4_cfg.output.jsonl_dir) / f"{m3_cfg.camera.cam_id}_{int(time.time())}.jsonl"
        m4_writer = JsonlWorldTrackWriter(path, phase=m4_cfg.phase)

    m4_pipeline = M4Pipeline(m4_cfg, provider, walkable_grid=walkable_grid, writer=m4_writer, on_tracks=live_source.submit)

    T = m4_pipeline.floor_transform
    server.scene.update_camera(T.R.tolist(), T.t.tolist(), T.camera_height_m, frame_name, T.map_id)
    log.info(f"M4 phase={m4_cfg.phase}, camera_height={T.camera_height_m:.3f}m -- scene updated")

    source_kind = args.m4_source
    m3_source = build_source(m3_cfg, source_kind, args.m4_playback)
    backend = build_backend(m3_cfg)
    estimator = PoseEstimator(backend)

    mask_set = None
    if m3_cfg.mask.masks_path:
        mask_set = load_masks(m3_cfg.mask.masks_path).get(m3_cfg.camera.cam_id, empty_mask_set())

    m3_writer = None
    if not args.no_m3_log:
        m3_path = Path(m3_cfg.output.jsonl_dir) / f"{m3_cfg.camera.cam_id}_{int(time.time())}.jsonl"
        m3_writer = JsonlDetectionWriter(m3_path)

    m3_pipeline = M3Pipeline(
        m3_cfg, mask_set=mask_set, writer=m3_writer,
        on_detections=lambda frame, dets: m4_pipeline.process(frame, dets),
    )
    daemon = M3Daemon(m3_cfg, m3_source, estimator, m3_pipeline)

    return live_source, daemon, (m3_writer, m4_writer)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "--present-config", dest="present_config", required=True)
    ap.add_argument("--source", choices=["live", "replay", "synthetic"], default="synthetic")

    ap.add_argument("--replay-path", default=None)
    ap.add_argument("--replay-speed", type=float, default=None)
    ap.add_argument("--replay-loop", action="store_true", default=None)

    ap.add_argument("--synthetic-mode", choices=["scripted", "realistic"], default=None)
    ap.add_argument("--synthetic-n", type=int, default=None)

    ap.add_argument("--m3-config", default=None)
    ap.add_argument("--m4-config", default=None)
    ap.add_argument("--m4-source", choices=["realsense", "synthetic"], default="realsense")
    ap.add_argument("--m4-playback", default=None)
    ap.add_argument("--no-m3-log", action="store_true")
    ap.add_argument("--no-m4-log", action="store_true")
    args = ap.parse_args(argv)

    present_cfg = PresentConfig.load(args.present_config)
    if args.replay_path:
        present_cfg.source.replay_path = args.replay_path
    if args.replay_speed is not None:
        present_cfg.source.replay_speed = args.replay_speed
    if args.replay_loop is not None:
        present_cfg.source.replay_loop = args.replay_loop
    if args.synthetic_mode is not None:
        present_cfg.source.synthetic_mode = args.synthetic_mode
    if args.synthetic_n is not None:
        present_cfg.source.synthetic_n_walkers = args.synthetic_n
    present_cfg.source.kind = args.source

    daemon = None
    if args.source == "live":
        if not (args.m3_config and args.m4_config):
            ap.error("--source live requires --m3-config and --m4-config")
        # Built after the server object exists (needs server.scene) --
        # see below.
        server = PresentServer(present_cfg, source=None)  # placeholder, replaced next line
        live_source, daemon, _writers = _build_live(args, present_cfg, server)
        server.source = live_source
    elif args.source == "replay":
        if not present_cfg.source.replay_path:
            ap.error("--source replay requires --replay-path (or source.replay_path in the config)")
        server = PresentServer(present_cfg, source=ReplaySource(present_cfg.source, cam_id=present_cfg.scene.cam_id))
    else:
        server = PresentServer(present_cfg, source=build_synthetic_source(present_cfg.source))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    install_signal_handlers(loop, server)

    async def _run():
        server_task = asyncio.create_task(server.serve_forever())
        daemon_task = None
        if daemon is not None:
            daemon_task = loop.run_in_executor(None, daemon.run)
        try:
            await server_task
        finally:
            if daemon is not None:
                daemon.stop()
            if daemon_task is not None:
                await daemon_task

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
