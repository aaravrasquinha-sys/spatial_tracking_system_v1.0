# Floor markers (Section 11's primary Phase A acceptance test)

Empty in the repo -- your tape-measured ground truth and captured runs
are room-specific. This is the quantitative "done when" check:
**median error <= 10cm within 4m of the camera, <= 20cm beyond**.

## 1. Tape down 10+ markers

Mark the floor point directly below the camera first (this is the local
frame's origin, Section 2). Then tape 10+ crosses around the room with a
real spread of distances and angles -- not just a grid directly in
front. Measure each one with a tape measure from the origin mark: X
distance along wherever the camera happened to be facing at calibration
time, Y perpendicular to that.

Write `ground_truth.json`:

```json
{
  "markers": [
    {"name": "m1", "x": 1.20, "y": 0.35},
    {"name": "m2", "x": 2.80, "y": -0.60},
    {"name": "m3", "x": 5.10, "y": 1.40}
  ]
}
```

## 2. Capture

```bash
python3 scripts/floor_marker_capture.py \
    --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json \
    --out poi_localization/eval/floor_markers/<room_name>/captured.json
```

Stand on each marker, type its name (matching `ground_truth.json`
exactly), press Enter, hold still for ~1s while it records. Only one
person should be in frame during capture -- the tool ignores frames with
more than one confirmed track.

## 3. Compare

```bash
python3 -m poi_localization.eval.floor_markers.compare \
    --captured poi_localization/eval/floor_markers/<room_name>/captured.json \
    --ground-truth poi_localization/eval/floor_markers/<room_name>/ground_truth.json
```

Prints per-marker error, then the short-range/long-range median error
against Section 11's thresholds, exiting non-zero if either fails.

## Phase B

After Module 2's calibration exists, re-tape (or re-measure the same
tape marks in the room frame instead of the local frame) and re-run both
steps against `configs/m4.phaseB.example.json` -- Section 11's Phase B
exit criterion is passing this same test again after the frame swap.
