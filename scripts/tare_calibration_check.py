"""
WP-LIVE: guides the D435i's own Tare Calibration (the Intel tool meant
for depth-MEASUREMENT accuracy, distinct from On-Chip Calibration which
only reduces noise -- see the planning notes' section 3.2) and then
verifies the result using the approved single tape-distance scale check
(pyslam.mapping.scale_check) against a live depth reading, BEFORE any
mapping run. This script does not perform tare calibration itself
(that's realsense-viewer's job, or librealsense's rs-depth-quality-tool
example) -- it's the pre-flight check that confirms tare calibration
actually took, quantitatively, rather than trusting the tool's own
health-check number alone.

Usage:
    python3 scripts/tare_calibration_check.py --distance-m 1.000

Point the camera at a flat, featureless wall at EXACTLY the given
distance (measure with a tape, same discipline the approved scale-check
feature uses everywhere else in this project) and this reads the
median depth at the image centre, compares it against the tape
measurement, and reports pass/fail using the same
pyslam.mapping.scale_check.check_scale function the live mapping
scale-verification feature uses -- one shared notion of "how much scale
error is acceptable," not a second ad-hoc threshold invented here.

NOT RUN against real hardware in this development sandbox (no D435i,
no pyrealsense2 -- same honest caveat as pyslam/sensors/realsense.py's
own WP-LIVE 3.5 fix). The scale_check call this script wraps IS
oracle-tested (tests/gates/test_g_live.py::check_scale_check_ratio_and_tolerance);
what's unvalidated here is only the RealSense capture glue around it.
"""
from __future__ import annotations
import argparse
import sys

import numpy as np

from pyslam.mapping.scale_check import check_scale


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distance-m", type=float, required=True,
                     help="tape-measured distance from the camera to the flat wall/target, metres")
    ap.add_argument("--tol-pct", type=float, default=1.5,
                     help="acceptable depth-scale error percent (tighter than the live mapping "
                          "scale check's own 3%% default, since this is a controlled, close-range, "
                          "flat-target measurement -- see Intel's own Tare Calibration accuracy "
                          "claims for why 1-1.5%% is a reasonable bar here)")
    ap.add_argument("--roi", type=int, default=20,
                     help="half-width in pixels of the centre region to median over")
    args = ap.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not importable -- run this on the target machine "
              "(Orin Nano or the dev workstation with librealsense installed).")
        sys.exit(2)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print("Point the camera at a flat, featureless wall at EXACTLY "
          f"{args.distance_m:.3f}m (tape-measured, perpendicular to the sensor). "
          "Capturing in 3 seconds to let auto-exposure settle...")
    import time
    time.sleep(3.0)

    samples = []
    for _ in range(30):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth = np.asanyarray(depth_frame.get_data())
        h, w = depth.shape
        cy, cx = h // 2, w // 2
        roi = depth[cy - args.roi:cy + args.roi, cx - args.roi:cx + args.roi].astype(np.float64)
        roi = roi[roi > 0] * depth_scale
        if roi.size > 0:
            samples.append(float(np.median(roi)))
    pipeline.stop()

    if not samples:
        print("No valid depth samples captured -- check the target is within range and textured "
              "enough for the ROI to have valid pixels (a perfectly flat, feature-less wall at "
              "close range should still return valid stereo depth; if not, check laser_power "
              "and lighting).")
        sys.exit(1)

    measured_m = float(np.median(samples))
    result = check_scale(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, measured_m]),
                          tape_distance_m=args.distance_m, tol_pct=args.tol_pct)
    print(f"\nMeasured depth (median of {len(samples)} frames, {args.roi*2}x{args.roi*2}px ROI): "
          f"{measured_m:.4f}m")
    print(f"Tape distance: {args.distance_m:.4f}m")
    print(f"Ratio: {result.ratio:.4f}  Error: {result.error_pct:.2f}%  "
          f"{'PASS' if result.passed else 'FAIL'} (tol={args.tol_pct}%)")
    if not result.passed:
        print("\nRun Tare Calibration (realsense-viewer -> More -> Tare Calibration, or "
              "librealsense's rs-depth-quality-tool) against this same target/distance, "
              "then re-run this script to confirm.")
        sys.exit(1)
    print("\nTare calibration verified -- safe to proceed with mapping.")


if __name__ == "__main__":
    main()
