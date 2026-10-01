# poi_perception — M3 (2D perception)

**Never imports** `pyslam`, `poi_localization`, `poi_present` or `sts`. Knows nothing about the map or calibration.

* **Consumes:** `Frame` (RGB + aligned depth, `poi_perception/contracts.py`).
* **Produces:** `Detection2D` per person per frame (time, 2D track id, box, confidence, 17 COCO keypoints, **footpoint + its
  source**, torso polygon, quality flags) and `data/logs/<site>/<cam>/detections/*.jsonl` (every frame, even empty) plus
  auto-saved clips around ID switches / lost tracks / low-confidence footpoints.
* **Pipeline:** RealSense capture (global time, High-Accuracy preset, laser 360, spatial+temporal+hole-filling filters) →
  TensorRT FP16 pose model (`yolo11n-pose`) → decode + NMS → ByteTrack → footpoint (mean of confident ankles, else one ankle,
  else bbox bottom flagged low-confidence) + torso polygon (shoulder–hip quad shrunk 15%, else a central strip) → ROI mask drop.
* **Threads:** capture and inference on separate threads; size-1 drop-stale queues (`runtime/queues.py`); track+footpoint on the
  calling thread (`runtime/m3_daemon.py`).
* **Entry points:** `scripts/run_m3.py`, `scripts/bench.py`, `scripts/export_model.py`, `scripts/debug_visualize.py`,
  `scripts/demo_synthetic.py`; through `sts`: `python -m sts engine`, and M3 is part of `sts run`.
* **Config:** `poi_perception.config.M3Config` (strict loader: unknown keys raise). Generated per camera by `sts configgen`.
* **Backends:** `trt` (runtime; imports only `tensorrt` + `pycuda`), `ultralytics` (export/dev only), mock (tests).
* **Tests:** `tests/unit/test_{decode,footpoint,footpoint_polygon_winding,bytetrack,masks,queues,detection2d_io,eval,pipeline_mock_e2e,realsense_post_processing}.py`.
* **Eval:** `poi_perception/eval/` — seven scripted scenarios (six automated with the mock backend; low-light is graded by hand),
  `grade.py`, `replay_diff.py`.

**Limits:** engines are device-specific (rebuild after any JetPack change; `sts doctor` compares the engine manifest to
`/etc/nv_tegra_release` and to the camera size) · the dataclass default model size is 640x640 while the camera is 640x480 —
`sts` always generates 640x480 · low-light RGB detection degrades (plan lighting, or later try the IR stream — `capture_source`
slot `ir_stream`, reserved) · ID switches when people cross are accepted in v1 (M4 re-checks association in the world frame).
