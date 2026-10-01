#!/usr/bin/env python3
"""
Section 11's floor-marker test, the capture half: run the live M3+M4
pipeline, stand on a taped marker, type its name and press Enter, and
this records that marker's position as the median of the next ~1s of
filtered track positions. Repeat for all 10+ markers, then compare
against your tape-measured ground truth with
poi_localization.eval.floor_markers.compare.

    python3 scripts/floor_marker_capture.py --m3-config configs/m3.example.json \\
        --m4-config configs/m4.phaseA.example.json --out logs/floor_markers/captured.json

Only one person (you) should be in frame while capturing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_localization.config import M4Config
from poi_localization.runtime.m4_daemon import build_m4_pipeline
from poi_perception.config import M3Config
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.runtime.m3_daemon import M3Daemon, M3Pipeline, build_backend, build_source
from poi_perception.runtime.masks import empty_mask_set, load_masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m3-config", required=True)
    ap.add_argument("--m4-config", required=True)
    ap.add_argument("--source", choices=["realsense", "synthetic"], default="realsense")
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture-seconds", type=float, default=1.0)
    args = ap.parse_args()

    m3_cfg = M3Config.load(args.m3_config)
    m4_cfg = M4Config.load(args.m4_config)

    recent_positions = []
    lock = threading.Lock()

    def on_tracks(frame, records):
        with lock:
            if len(records) == 1:
                p = records[0].p_local if m4_cfg.phase == "A" else records[0].p
                recent_positions.append((time.time(), p[0], p[1]))
            # keep only the last few seconds
            cutoff = time.time() - 3.0
            while recent_positions and recent_positions[0][0] < cutoff:
                recent_positions.pop(0)

    m4_pipeline = build_m4_pipeline(m4_cfg, cam_cfg=m3_cfg.camera)
    m4_pipeline.on_tracks = on_tracks

    source = build_source(m3_cfg, args.source, playback_path=None)
    backend = build_backend(m3_cfg)
    estimator = PoseEstimator(backend)
    mask_set = load_masks(m3_cfg.mask.masks_path).get(m3_cfg.camera.cam_id, empty_mask_set()) if m3_cfg.mask.masks_path else None

    m3_pipeline = M3Pipeline(m3_cfg, mask_set=mask_set, on_detections=lambda f, d: m4_pipeline.process(f, d))
    daemon = M3Daemon(m3_cfg, source, estimator, m3_pipeline)

    markers = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    daemon_thread = threading.Thread(target=daemon.run, daemon=True)
    daemon_thread.start()
    time.sleep(1.0)  # let the pipeline warm up and start tracking

    print("Stand on a marker, type its name, and press Enter (blank line to finish).")
    try:
        while True:
            name = input("marker name> ").strip()
            if not name:
                break
            print(f"Recording '{name}' for {args.capture_seconds:.1f}s -- hold still...")
            time.sleep(args.capture_seconds)
            with lock:
                cutoff = time.time() - args.capture_seconds
                sample = [(x, y) for (t, x, y) in recent_positions if t >= cutoff]
            if not sample:
                print("  No track seen during that window -- not recorded. Check you're in frame and confirmed.")
                continue
            mx = statistics.median(x for x, y in sample)
            my = statistics.median(y for x, y in sample)
            markers.append({"name": name, "x": mx, "y": my})
            print(f"  Recorded {name}: ({mx:.3f}, {my:.3f}) from {len(sample)} samples")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        daemon.stop()

    out_path.write_text(json.dumps({"markers": markers}, indent=2))
    print(f"\nWrote {len(markers)} markers to {out_path}")


if __name__ == "__main__":
    main()
