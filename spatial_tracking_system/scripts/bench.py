#!/usr/bin/env python3
"""
Section 3: "Expect... a yolov8n-class model in the few-hundred-FPS range
for GPU-only inference alone... That number is not your achievable
end-to-end frame rate... Treat every number in this section as a
placeholder until you've run --bench on your own board."

This script measures the real thing: capture, inference, and
track+footpoint, timed separately, on your own hardware.

Before running on the Orin: lock power mode and clocks, exactly as the
doc requires --

    sudo nvpmodel -m 0     # verify what mode 0 actually is on this kit
    sudo jetson_clocks

and watch GPU utilization/temperature with jtop in a second terminal --
this script does not attempt to shell out to jtop itself, since a number
measured at a throttled/low-power state is not comparable to one at
MAXN_SUPER and conflating them is worse than not measuring GPU load at
all.

Usage:
    python3 scripts/bench.py --config configs/m3.example.json \\
        --source synthetic --frames 300
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_perception.config import M3Config
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.runtime.m3_daemon import M3Pipeline, build_backend, build_source
from poi_perception.runtime.masks import empty_mask_set


def _percentiles(values: list, ps=(50, 90, 99)) -> dict:
    if not values:
        return {p: float("nan") for p in ps}
    s = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
        out[p] = s[idx]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", choices=["realsense", "synthetic"], default="synthetic")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args(argv)

    cfg = M3Config.load(args.config)
    source = build_source(cfg, args.source, playback_path=None)
    backend = build_backend(cfg)
    estimator = PoseEstimator(backend)
    pipeline = M3Pipeline(cfg, mask_set=empty_mask_set())

    print(f"Warming up ({args.warmup} inferences)...")
    estimator.warm_up(args.warmup)

    cap_ms, infer_ms, track_ms, total_ms = [], [], [], []

    it = iter(source)
    n = 0
    t_start = time.perf_counter()
    while n < args.frames:
        t0 = time.perf_counter()
        frame = next(it)
        t1 = time.perf_counter()

        raw_dets = estimator.infer(frame)
        t2 = time.perf_counter()

        pipeline.process(frame, raw_dets)
        t3 = time.perf_counter()

        cap_ms.append((t1 - t0) * 1000)
        infer_ms.append((t2 - t1) * 1000)
        track_ms.append((t3 - t2) * 1000)
        total_ms.append((t3 - t0) * 1000)
        n += 1
    wall_s = time.perf_counter() - t_start
    source.close()

    def report(name: str, values: list) -> None:
        p = _percentiles(values)
        print(
            f"{name:<10} mean={statistics.mean(values):7.2f}ms  "
            f"p50={p[50]:7.2f}ms  p90={p[90]:7.2f}ms  p99={p[99]:7.2f}ms"
        )

    print(f"\n{n} frames, backend={cfg.model.backend}, source={args.source}")
    report("capture", cap_ms)
    report("inference", infer_ms)
    report("track", track_ms)
    report("total/frame", total_ms)
    print(f"\nend-to-end wall-clock rate: {n / wall_s:.1f} fps (sequential; the live "
          f"daemon overlaps capture/inference on separate threads, so expect this "
          f"to undercount the real threaded rate -- see runtime/m3_daemon.py)")


if __name__ == "__main__":
    main()
