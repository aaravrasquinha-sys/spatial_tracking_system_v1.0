# Map bundle (M1 -> M2)

```
data/maps/<site>/
  dense/points.ply          REQUIRED  dense cloud in M1's W_cam0 frame (the first mapping camera's optical frame:
                                      x right, y DOWN, z forward; NOT gravity-aligned, NOT floor-referenced)
  capture_profile.json      REQUIRED  the capture profile used while mapping (hash recorded by M2 at capture time)
  manifest.json             written by `sts map-lock` (or M1's own lock): map_id, created, capture_profile_sha,
                            builder_git_commit, qa snapshot. Schema: schemas/map_manifest.schema.json
  session_summary.json      written by run_live_map.py on stop (QA counters)
  trajectory.json, reloc_map/, layers/   present only if the mapper produced them; nothing downstream requires them
```

After acceptance the directory is read-only (`sts map-lock` does `chmod a-w`). **Re-mapping makes a new bundle with a new
`map_id`; a locked bundle is never edited.**

## The `map_id` rule

```
map_id = first 12 hex chars of SHA-256 over the sorted "<relative path>:<sha256(file)>" lines of
         ["dense/points.ply", "capture_profile.json"]          (pyslam.mapping.lock.compute_map_id)
```

* M2 computes exactly this for an unlocked bundle and **prefers `manifest.json`'s `map_id` when one exists**
  (`pyslam.anchor.map_io.resolve_map_id`), warning if the manifest disagrees with the recomputed hash.
* Therefore **locking must not change the id**: any finalizer must hash the *same two files*. If it hashed more (a mesh, a
  manifest), every existing calibration's `map_id` would stop matching and the Phase-B gate would block the run.
  `test_lock_keeps_the_map_id_that_anchors_were_solved_against` enforces this.
* If a future map needs different content files, that is a **new map**: new `map_id`, re-anchor.
* Editing `dense/points.ply` or `capture_profile.json` after locking changes the recomputed hash; the gate reports
  "map bundle unchanged since locking" and refuses to run.

## Known limits (docs/OPEN_ITEMS.md)

* `run_live_map.py --mode live` writes no manifest -> use `sts map-lock`. `--mode lockstep` is broken in the uploaded
  script (K1).
* No `room.glb` mesh exists; the viewer uses the point cloud M2 writes into the anchor bundle.
