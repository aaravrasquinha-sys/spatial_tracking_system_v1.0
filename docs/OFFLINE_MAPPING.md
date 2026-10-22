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
