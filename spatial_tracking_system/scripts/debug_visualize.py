#!/usr/bin/env python3
"""
Debug-only live visualizer. Opens a window showing the camera feed with
bounding boxes, keypoints, the footpoint, and the torso polygon drawn on
top of whatever M3 is currently tracking, so you can eyeball tracking
quality in real time (e.g. someone walking in front of the D435i).

Usage (real camera, stream to workstation via web/mjpeg):
    python3 scripts/debug_visualize.py --config configs/m3.example.json --source realsense --host-stream --no-window
"""
from __future__ import annotations

import argparse
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from poi_perception.config import M3Config
from poi_perception.contracts import COCO_KEYPOINT_NAMES
from poi_perception.inference import mock_infer
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.io.detection2d import Detection2D
from poi_perception.runtime.m3_daemon import M3Pipeline, build_backend, build_source
from poi_perception.runtime.masks import empty_mask_set, load_masks

# --- HTTP MJPEG STREAMING SERVER (No Disk I/O) ---
_stream_frame = None
_stream_cond = threading.Condition()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _stream_cond:
                        _stream_cond.wait()
                        if _stream_frame is None:
                            continue
                        jpeg = _stream_frame.tobytes()
                    
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(jpeg))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
            except Exception:
                pass

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    pass

def start_stream_server(port=8080):
    server = ThreadedHTTPServer(('', port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"\n[STREAMING] Live feed hosted successfully!")
    print(f"  View on Workstation browser: http://<jetson_orin_ip>:{port}")
    print(f"  View via Workstation terminal: ffplay http://<jetson_orin_ip>:{port}\n")

# --- SKELETON & DRAWING CONFIG ---
_SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

_STATE_COLOR = {
    "new": (0, 255, 255),  # yellow (BGR)
    "tracked": (0, 220, 0),  # green
    "lost_buffer": (0, 0, 255),  # red
}


def _track_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973 + 17)
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


def _draw_detection(img: np.ndarray, d: Detection2D, kp_conf_thresh: float = 0.3) -> None:
    color = _track_color(d.track_id_2d)
    x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"id={d.track_id_2d} {d.track_state} conf={d.det_conf:.2f}"
    label2 = f"fp={d.footpoint_source} torso={d.torso_quality}"
    state_color = _STATE_COLOR.get(d.track_state, color)
    cv2.rectangle(img, (x1, max(0, y1 - 34)), (x1 + max(180, 9 * len(label)), y1), state_color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, label2, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    if d.torso_polygon_px:
        pts = np.array([[int(px), int(py)] for px, py in d.torso_polygon_px], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 180, 0), thickness=1, lineType=cv2.LINE_AA)

    kpts = d.keypoints
    for name_a, name_b in _SKELETON_EDGES:
        a = kpts.get(name_a)
        b = kpts.get(name_b)
        if a and b and a[2] >= kp_conf_thresh and b[2] >= kp_conf_thresh:
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, cv2.LINE_AA)
    for name, (px, py, pc) in kpts.items():
        if pc >= kp_conf_thresh:
            r = 4 if "ankle" in name else 2
            cv2.circle(img, (int(px), int(py)), r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(px), int(py)), r, (0, 0, 0), 1, cv2.LINE_AA)

    fx, fy = d.footpoint_px
    marker_color = (0, 0, 255) if d.low_confidence else (0, 255, 0)
    cv2.drawMarker(img, (int(fx), int(fy)), marker_color, cv2.MARKER_STAR, 16, 2)


def _annotate(rgb: np.ndarray, detections: List[Detection2D], fps: float) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        _draw_detection(bgr, d)
    cv2.putText(bgr, f"{fps:5.1f} fps  n={len(detections)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, f"{fps:5.1f} fps  n={len(detections)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return bgr


def _build_estimator(cfg: M3Config, args) -> PoseEstimator:
    if args.mock_scenario:
        fn = getattr(mock_infer, f"scenario_{args.mock_scenario}")(
            frame_w=cfg.camera.width, frame_h=cfg.camera.height
        )
        return PoseEstimator(mock_infer.MockPoseBackend(fn))
    return PoseEstimator(build_backend(cfg))


def main() -> None:
    global _stream_frame
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", choices=["realsense", "synthetic"], default="realsense")
    ap.add_argument("--playback", default=None, help="replay a recorded .bag instead of a live camera")
    ap.add_argument("--mock-scenario", default=None)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="skip cv2.imshow locally")
    ap.add_argument("--host-stream", action="store_true", help="host a live web stream on port 8080")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    if args.host_stream:
        start_stream_server(port=8080)

    cfg = M3Config.load(args.config)

    mask_set = empty_mask_set()
    if cfg.mask.masks_path and not args.no_mask:
        mask_set = load_masks(cfg.mask.masks_path).get(cfg.camera.cam_id, empty_mask_set())

    source = build_source(cfg, args.source, args.playback)
    estimator = _build_estimator(cfg, args)
    estimator.warm_up()
    pipeline = M3Pipeline(cfg, mask_set=mask_set)

    show_window = not args.no_window
    window_broken = False

    print("Press Ctrl+C to stop.")
    t_last = time.perf_counter()
    n = 0
    try:
        for frame in source:
            raw_dets = estimator.infer(frame)
            detections = pipeline.process(frame, raw_dets)

            now = time.perf_counter()
            fps = 1.0 / max(now - t_last, 1e-6)
            t_last = now

            annotated = _annotate(frame.rgb, detections, fps)

            # Push frame to the network stream buffer if enabled (no disk writing)
            if args.host_stream:
                ret, jpeg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    with _stream_cond:
                        _stream_frame = jpeg
                        _stream_cond.notify_all()

            if show_window and not window_broken:
                try:
                    cv2.imshow("M3 debug visualizer", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    window_broken = True

            n += 1
            if args.max_frames is not None and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if show_window and not window_broken:
            cv2.destroyAllWindows()
        print(f"Processed {n} frames.")


if __name__ == "__main__":
    main()
