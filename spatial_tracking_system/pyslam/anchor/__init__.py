"""
Module 2 (WP-ANCHOR): static-camera anchor.

Given the map bundle module 1 produced (dense/points.ply, in the SLAM
graph's native W_cam0 frame) and a short capture from a D435i sitting
in its final mount, find T_room_cam -- the camera's 6-DOF pose in a
gravity/floor-referenced "room" frame derived from the map itself --
with an honest uncertainty, and refuse to emit anything if independent
checks disagree.

Pipeline (see SYSTEM_SUMMARY_ANCHOR.md for the full design/rationale):
  map_io / room_frame / walkable   Stage 0: room frame + walkable grid from the map
  capture / query_cloud            Stage 1: static capture -> weighted query cloud
  physics_prior                    Stage 2: IMU gravity + own-floor fit (3 of 6 DOF, no map)
  global_search                    Stage 3: correlative x/y/yaw search (exhaustive, FFT)
  icp                              Stage 4: robust multi-scale point-to-plane ICP + degeneracy
  uncertainty / verify             Stage 5/6: covariance, gates
  bundle / calibrate               Stage 7: outputs (calibration.cam0.json etc.)
  watchdog                         runtime drift detection (tilt + depth + fast ICP recheck)

Hardware is touched in exactly one place (capture.capture_static, via the
project's existing RealSenseSource); everything else is pure numpy/scipy/
opencv and gate-tested against a synthetic room with a known camera pose.
"""
