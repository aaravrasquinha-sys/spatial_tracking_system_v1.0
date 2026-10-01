"""
Section 11's floor-marker test: "Tape 10+ crosses on the floor... at
known, tape-measured positions... Stand on each, record the filtered
p_local, compare to the tape-measured position... Track median error at
short range (< ~4m) and long range separately."

Both files are the same simple schema: {"markers": [{"name": "m1", "x":
1.20, "y": 0.35}, ...]}, in the Phase A local frame (origin = the floor
point directly below the camera, per Section 2 -- so "range" is just
hypot(x, y), no separate camera-position bookkeeping needed).

Ground truth: measure with a tape measure from the point on the floor
directly below the camera (mark it before you start), along the local
frame's X axis (wherever the camera happened to be facing at t=0) and
perpendicular to it for Y. scripts/floor_marker_capture.py writes the
"captured" file in this exact schema by recording live filtered
positions while you stand on each marker.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SHORT_RANGE_MAX_M = 4.0
SHORT_RANGE_ERROR_MAX_M = 0.10
LONG_RANGE_ERROR_MAX_M = 0.20


def _load_markers(path: str | Path) -> Dict[str, Tuple[float, float]]:
    raw = json.loads(Path(path).read_text())
    return {m["name"]: (float(m["x"]), float(m["y"])) for m in raw["markers"]}


@dataclass
class MarkerError:
    name: str
    range_m: float
    error_m: float


def compare(captured_path: str | Path, ground_truth_path: str | Path) -> List[MarkerError]:
    captured = _load_markers(captured_path)
    ground_truth = _load_markers(ground_truth_path)

    missing_gt = set(captured) - set(ground_truth)
    missing_cap = set(ground_truth) - set(captured)
    if missing_gt:
        print(f"Warning: captured markers with no ground truth entry, skipped: {sorted(missing_gt)}", file=sys.stderr)
    if missing_cap:
        print(f"Warning: ground-truth markers never captured, skipped: {sorted(missing_cap)}", file=sys.stderr)

    results = []
    for name in sorted(set(captured) & set(ground_truth)):
        gx, gy = ground_truth[name]
        cx, cy = captured[name]
        range_m = float(np.hypot(gx, gy))
        error_m = float(np.hypot(cx - gx, cy - gy))
        results.append(MarkerError(name, range_m, error_m))
    return results


def report(results: List[MarkerError]) -> bool:
    """Prints the Section 11 done-when table and returns whether it passes."""
    if not results:
        print("No matched markers to grade.")
        return False

    short = [r for r in results if r.range_m < SHORT_RANGE_MAX_M]
    long = [r for r in results if r.range_m >= SHORT_RANGE_MAX_M]

    print(f"{'marker':<12}{'range_m':>10}{'error_m':>10}")
    for r in results:
        print(f"{r.name:<12}{r.range_m:>10.2f}{r.error_m:>10.3f}")

    passed = True
    print()
    if short:
        med = float(np.median([r.error_m for r in short]))
        ok = med <= SHORT_RANGE_ERROR_MAX_M
        passed = passed and ok
        print(f"Short range (<{SHORT_RANGE_MAX_M}m, n={len(short)}): median error = {med*100:.1f}cm "
              f"(limit {SHORT_RANGE_ERROR_MAX_M*100:.0f}cm) -- {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Short range (<{SHORT_RANGE_MAX_M}m): no markers in this range")

    if long:
        med = float(np.median([r.error_m for r in long]))
        ok = med <= LONG_RANGE_ERROR_MAX_M
        passed = passed and ok
        print(f"Long range (>={SHORT_RANGE_MAX_M}m, n={len(long)}): median error = {med*100:.1f}cm "
              f"(limit {LONG_RANGE_ERROR_MAX_M*100:.0f}cm) -- {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Long range (>={SHORT_RANGE_MAX_M}m): no markers in this range")

    if len(results) < 10:
        print(f"\nNote: Section 11 asks for 10+ markers with a real spread of distances/angles; "
              f"only {len(results)} matched here.")

    return passed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captured", required=True)
    ap.add_argument("--ground-truth", required=True)
    args = ap.parse_args(argv)

    results = compare(args.captured, args.ground_truth)
    passed = report(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
