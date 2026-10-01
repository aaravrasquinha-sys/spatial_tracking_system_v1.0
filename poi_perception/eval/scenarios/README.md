# Scenario clips and baselines

This directory is where the Section 9 evaluation protocol's recorded
clips and hand-graded counts live once you're running on real hardware.
It's intentionally empty in the repo — these are per-room, per-rig
artifacts, not code.

## The seven scenarios (Section 9)

1. One person walking a simple loop.
2. Two people crossing paths.
3. Person exits frame and re-enters.
4. Person sits down.
5. Partial occlusion behind furniture for ~1-2s.
6. Someone standing near a mirror or TV (masking check).
7. Low light / backlit near a window, if relevant to your space.

Scenarios 1-6 have direct automated equivalents in
`inference/mock_infer.py` (`scenario_single_loop`,
`scenario_two_crossing`, `scenario_exit_reenter`, `scenario_sitting`,
`scenario_partial_occlusion`, `scenario_near_mirror`) exercised in
`tests/test_pipeline_mock_e2e.py`. Those are useful regression coverage
for the tracking/footpoint/masking *logic*, but they are not a
substitute for recording the real thing — real ankle-keypoint
reliability at your actual mount height/angle, real furniture geometry,
real lighting, is exactly what Section 9's "done when" table is checking
for. Scenario 7 (low light) has no synthetic equivalent at all and is
hardware-only.

## Recording a scenario

1. `python3 scripts/run_m3.py --config configs/m3.example.json --source realsense`
   with `output.jsonl_dir` and `output.clips_dir` pointed here, e.g.
   `poi_perception/eval/scenarios/<scenario_name>/`.
2. Act out the scenario in front of the camera.
3. Stop the daemon. You now have a JSONL log and (if any trigger fired)
   auto-saved clips under `output.clips_dir` (Section 8).

## Grading

```bash
python3 -m poi_perception.eval.grade poi_perception/eval/scenarios/<scenario_name>/detections.jsonl \
    --out poi_perception/eval/scenarios/<scenario_name>/baseline_summary.json
```

Watch the overlay video (or the raw clip, if you don't yet have overlay
rendering) and, by hand, per Section 9:

- Count ID switches. `grade.py`'s `new_track_events` is a *candidate*
  count (every time a track starts fresh) — confirm each one by eye
  against the video; a genuine new person entering frame is not an ID
  switch.
- Count false persons from masked regions — should be zero. If it isn't,
  the mask polygon in `configs/masks.<cam_id>.json` needs redrawing, not
  the detector re-tuned.
- Note any footpoint that's visibly wrong on the overlay.

Write those hand counts down next to `baseline_summary.json` (a small
`notes.md` in the scenario's folder is fine). That combination — the
machine-countable summary plus your hand-graded notes — is the baseline
a later change (different model, different tracker tuning) gets compared
against:

```bash
python3 -m poi_perception.eval.replay_diff \
    --baseline-summary poi_perception/eval/scenarios/<scenario_name>/baseline_summary.json \
    --current poi_perception/eval/scenarios/<scenario_name>/new_run.jsonl
```
