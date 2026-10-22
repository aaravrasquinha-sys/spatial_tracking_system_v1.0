# Offline mapping (recorded walks -> a finished map)

    python run_map.py ~/bags/room2.db3

Runs tracking over the whole recording (no dropped frames), then `run_offline.py`:

1. loads every keyframe, the odometry/bridge links and the per-frame poses of the run;
2. lost-tracking gaps (`bridge` links): translation from the walking speed measured just before and
   after the gap, rotation from the gyro, weight loose enough for loop closures to move them;
3. loop closing over ALL keyframe pairs: bag-of-words candidates (vocabulary trained on this run), the
   pipeline's bidirectional PnP verification, a consistency check between neighbouring loops, a robust
   (Cauchy, annealed) sparse pose-graph solve, then proximity rounds on the corrected poses;
4. fuses the depth of every 2nd tracked frame into one surface: `map_mesh.ply` (needs open3d) and
   `map_cloud.ply`.

Outputs: `runs/run_*_bag/offline/` -- `trajectory_final_frames.tum`, `trajectory_final_kf.tum`,
`loops.csv`, `graph.g2o`, `report.json`, `map_mesh.ply`, `map_cloud.ply`.

Why not the live loop closing: it only searches keyframes in working memory and its loop links were
weighted ~100x weaker than odometry (`loop link sigma 1.07 m / 61 deg` in the room2 run), so on a
two-lap walk it linked frames 0.5 s apart and never joined lap 2 to lap 1.

Settings: `pyslam/offline/profile.py` (the settings of the best room2 tracking run). `Config()` is
unchanged so the live chain and the test gates behave as before. Redo only the mapping step with
`python run_map.py BAG --map-only RUN_DIR`.

Known limits: needs loops (a walk that never revisits a place gets no loop correction); repeated
texture can still produce wrong loops; long bare-wall stretches still lose tracking; memory/time grow
with keyframe count (use `--voxel 0.03 --stride 3` for very long walks).

## As Module 1 (`sts map`)

The offline mapper is the default `map_builder` slot (`pyslam_offline`, `sts/adapters/offline.py`). `sts map --bag FILE`
runs `run_map.py FILE --bundle <map folder>`; `pyslam/offline/bundle.py` then writes the bundle Module 2 reads:
`dense/points.ply` (the fused cloud, rebased so the first camera frame is the origin = W_cam0, isolated voxels removed),
`capture_profile.json`, `session_summary.json` (QA numbers + warnings). Only the first two decide `map_id`, so locking and
anchoring behave exactly as with the live builder. An empty map writes no bundle and nothing is activated.

## The map points Module 2 anchors to

`dense/points.ply` is built from the TSDF surface (`map_tsdf_points.ply`). Before it is fused, each depth frame loses
its flying pixels: pixels whose neighbour 2 px away differs by more than 4% of the depth. Earlier bundles used the raw
voxel average. That kept two kinds of junk that broke anchoring on real data:
- smeared fans of points inside the room, from flying pixels and moving people;
- walls 10-15 cm thick, where laps did not line up perfectly.

The voxel average (`map_cloud.ply`) is still written. It is used only if open3d is missing.

## Module 2 changes made for these maps (pyslam/anchor)

- **Global search uses two wall pictures** (`csm_map_mode = "both"`). One counts points (the original rule; keeps
  low furniture). The other keeps only "tall" cells, whose points span at least half of the 0.3-2.0 m slice. Every
  distinct peak from both goes to ICP, and fitness, colour and the gates decide. On the room2 map, the point count
  alone made the whole room look like wall and never proposed the real pose.
- **A camera that sees no floor.** A plane only counts as the floor if:
  - it is within 5 deg of gravity, and
  - less than 2% of the camera's points lie more than 6 cm below it.

  A wall-aimed camera otherwise latches onto a sill or cabinet top. In that case the solve takes roll/pitch from the
  IMU and searches the height against the map, above the lowest visible point. The floor cross-check gates are then
  reported as unavailable. They are replaced by two stricter checks: no camera point may end up below the map floor,
  and observability must be at least 0.004.
- Covered by `tests/gates/test_g_anchor.py::check_camera_that_sees_no_floor_is_solved_from_the_map`. On the room2
  capture aimed at the window wall, the solve is accepted with no hint at 1.93 m height, fitness 0.996, rmse 4.1 mm.
