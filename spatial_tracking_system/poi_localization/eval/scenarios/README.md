# M4 scenario tests (Section 11)

Recorded runs go here, one folder per scenario, e.g.
`poi_localization/eval/scenarios/crossing_01/`. Empty in the repo --
these are per-room, per-rig artifacts, not code.

## The scenarios (Phase A, runnable today, no map needed)

1. **Floor markers** -- see `../floor_markers/README.md` instead; it's
   the primary quantitative acceptance test (Section 11's "done when"
   thresholds) and has its own capture/compare tooling.
2. **Straight-line walk.** Tape a straight line, walk it at normal pace.
   Grade by hand: lateral (perpendicular) deviation of the filtered
   track from the taped line should have RMS <= 5cm.
3. **Depth dropout.** Wear dark/IR-absorbing clothing, or stand near/
   past the D435i's reliable depth range (~3m). Confirm tracking
   continues via ray-cast -- check `src` in the JSONL log reads
   `"raycast"` (not `"depth"` or `"fused"`) for that stretch.
4. **Occlusion.** Walk behind furniture for ~1s. Confirm the same
   `world_track_id` survives via `"coasting"` rather than ending and a
   new ID starting -- `eval/grade.py`'s `distinct_track_ids` should stay
   at 1 for a single-person occlusion test.
5. **Crossing.** Two people crossing paths at different depths. Confirm
   IDs don't swap -- specifically confirm this holds even when M3's own
   2D IDs *did* swap during the crossing (check the M3 JSONL log's
   `track_id_2d` alongside M4's `world_track_id` for the same time
   window; a `2d_id_mismatch` event in M4's log with the same
   `world_track_id` both before and after is exactly the success case,
   per Section 9).
6. **False-source rejection.** Point the camera at a mirror or a screen
   showing a person (the same masked-region case M3's ROI masks
   handle). Confirm no track reaches `"confirmed"` from it. In Phase A
   there's no walkable-grid gate yet (Section 8) -- this specifically
   tests whether M3's masking plus M4's chi-square/plausibility gates
   are enough on their own; expect this to be a known-weaker case until
   Phase B's walkable-grid gate exists.

## Recording

```bash
python3 scripts/run_m4.py --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json --source realsense
```

Point `output.jsonl_dir` in `configs/m4.phaseA.example.json` at this
scenario's folder before recording, or move the log there afterward.

## Grading

```bash
python3 -m poi_localization.eval.grade poi_localization/eval/scenarios/<name>/world_tracks.jsonl \
    --out poi_localization/eval/scenarios/<name>/baseline_summary.json
```

Write your hand-graded notes (lateral RMS for the straight-line test,
ID-swap count for crossing, confirmed-or-not for false-source) next to
`baseline_summary.json` -- that combination is the baseline a later
change (different measurement constants, different gate thresholds)
gets compared against:

```bash
python3 -m poi_localization.eval.replay_diff \
    --baseline-summary poi_localization/eval/scenarios/<name>/baseline_summary.json \
    --current poi_localization/eval/scenarios/<name>/new_run.jsonl
```

## Phase B additions

Once Module 2's calibration exists, re-run the floor-marker and
straight-line tests in the room frame (Section 11's Phase B exit
criterion: confirms the Phase A -> B swap didn't introduce an error) and
add the false-source test again with a real `walkable_grid_path`
configured -- this is the scenario Phase A could only partially defend
against, and Phase B's gate should now reject it outright.
