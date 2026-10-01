#!/usr/bin/env python3
"""
Debug-only live visualizer for M4 (not part of the pipeline, same spirit
as poi_perception's scripts/debug_visualize.py): runs M3+M4 together and
shows two panels side by side --

  left:  the camera feed with M3's bbox/skeleton/footpoint overlay
  right: a bird's-eye (top-down) view of the floor frame, with each
         world track drawn as a colored dot + fading trail + 2-sigma
         uncertainty ellipse, the camera drawn at the origin with its
         FOV wedge, and (optionally) your tape-measured floor markers
         overlaid as crosses so you can stand on a known point and see,
         at a glance, how far the dot is from where it should be.

This is exactly the "walk in front of the camera and watch it track you
in a top-down view" ground-truth sanity check for Phase A -- it is NOT a
substitute for eval/floor_markers' actual numeric comparison, which is
the real Section 11 acceptance test.

Usage (real camera, Phase A, live calibration):
    python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
        --m4-config configs/m4.phaseA.example.json --source realsense

Usage (real camera, Phase A, skip live calibration, known rig geometry):
    python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
        --m4-config configs/m4.phaseA.fixed.example.json --source realsense

Usage (no hardware, sanity-check the drawing code / a scripted scenario):
    python3 scripts/debug_visualize_world.py --m3-config configs/m3.example.json \
        --m4-config configs/m4.phaseA.fixed.example.json --source synthetic --mock-scenario two_crossing

With tape-measured ground truth overlaid (see eval/floor_markers/README.md
for the file schema):
    python3 scripts/debug_visualize_world.py ... --ground-truth poi_localization/eval/floor_markers/<room>/ground_truth.json

Press 'q' in the window to quit.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from poi_localization.config import M4Config
from poi_localization.frames.frame_provider import build_frame_provider
from poi_localization.runtime.m4_pipeline import M4Pipeline
from poi_localization.tracking.gates import load_walkable_grid
from poi_perception.config import M3Config
from poi_perception.inference import mock_infer
from poi_perception.inference.pose_infer import PoseEstimator
from poi_perception.io.detection2d import Detection2D
from poi_perception.runtime.m3_daemon import M3Pipeline, build_backend, build_source
from poi_perception.runtime.masks import empty_mask_set, load_masks


def _track_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973 + 17)
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


# ---------------------------------------------------------------------------
# Left panel: plain camera-feed overlay (a lighter version of M3's own
# debug_visualize.py drawing, kept self-contained here rather than shared,
# since that script lives outside the poi_perception package).
# ---------------------------------------------------------------------------
def _annotate_camera_panel(rgb: np.ndarray, detections: List[Detection2D]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        color = _track_color(d.track_id_2d)
        x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        fx, fy = d.footpoint_px
        marker_color = (0, 0, 255) if d.low_confidence else (0, 255, 0)
        cv2.drawMarker(bgr, (int(fx), int(fy)), marker_color, cv2.MARKER_STAR, 14, 2)
        cv2.putText(bgr, f"2d id={d.track_id_2d}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return bgr


# ---------------------------------------------------------------------------
# Right panel: the bird's-eye view.
# ---------------------------------------------------------------------------
class BirdEyeView:
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 640),
        x_range: Tuple[float, float] = (-0.5, 6.0),
        y_range: Tuple[float, float] = (-3.5, 3.5),
        trail_len: int = 60,
        fov_deg: float = 69.0,  # D435i color FOV, horizontal
    ):
        self.w, self.h = canvas_size
        self.x_range = x_range
        self.y_range = y_range
        self.fov_deg = fov_deg
        self.trails: Dict[int, Deque[Tuple[float, float]]] = {}
        self.trail_len = trail_len

    def world_to_px(self, x: float, y: float) -> Tuple[int, int]:
        # Forward (+X) points up the image; lateral (+Y) points right.
        col = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.w
        row = self.h - (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.h
        return int(round(col)), int(round(row))

    def _draw_grid(self, canvas: np.ndarray) -> None:
        grid_color = (50, 50, 50)
        text_color = (110, 110, 110)
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        for xm in range(int(np.ceil(x0)), int(np.floor(x1)) + 1):
            p0 = self.world_to_px(xm, y0)
            p1 = self.world_to_px(xm, y1)
            cv2.line(canvas, p0, p1, grid_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{xm}m", (5, self.world_to_px(xm, y0)[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
        for ym in range(int(np.ceil(y0)), int(np.floor(y1)) + 1):
            p0 = self.world_to_px(x0, ym)
            p1 = self.world_to_px(x1, ym)
            cv2.line(canvas, p0, p1, grid_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{ym}m", (self.world_to_px(x0, ym)[0] + 2, self.h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)

    def _draw_camera(self, canvas: np.ndarray) -> None:
        origin_px = self.world_to_px(0.0, 0.0)
        half_fov = np.radians(self.fov_deg / 2)
        far = self.x_range[1]
        left = self.world_to_px(far * np.cos(half_fov), -far * np.sin(half_fov))
        right = self.world_to_px(far * np.cos(half_fov), far * np.sin(half_fov))
        wedge_color = (70, 70, 30)
        pts = np.array([origin_px, left, right], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], wedge_color)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
        cv2.drawMarker(canvas, origin_px, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
        cv2.putText(canvas, "camera", (origin_px[0] + 8, origin_px[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_ground_truth(self, canvas: np.ndarray, markers: List[dict]) -> None:
        for m in markers:
            px = self.world_to_px(m["x"], m["y"])
            color = (0, 200, 255)
            size = 8
            cv2.line(canvas, (px[0] - size, px[1] - size), (px[0] + size, px[1] + size), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (px[0] - size, px[1] + size), (px[0] + size, px[1] - size), color, 2, cv2.LINE_AA)
            cv2.putText(canvas, m["name"], (px[0] + 10, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _draw_track(self, canvas: np.ndarray, track_id: int, x: float, y: float, cov_xy, state: str, height, src: str) -> None:
        color = _track_color(track_id)
        trail = self.trails.setdefault(track_id, collections.deque(maxlen=self.trail_len))
        trail.append((x, y))

        pts = [self.world_to_px(px, py) for px, py in trail]
        for i in range(1, len(pts)):
            fade = i / len(pts)
            faded = tuple(int(c * fade) for c in color)
            cv2.line(canvas, pts[i - 1], pts[i], faded, 2, cv2.LINE_AA)

        sxx, sxy, syy = cov_xy
        cov = np.array([[sxx, sxy], [sxy, syy]])
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0, None)
            angle_deg = float(np.degrees(np.arctan2(eigvecs[0, 1], eigvecs[0, 0])))
            scale_x = self.h / (self.x_range[1] - self.x_range[0])
            scale_y = self.w / (self.y_range[1] - self.y_range[0])
            scale = (scale_x + scale_y) / 2
            axes = (max(1, int(2 * np.sqrt(eigvals[1]) * scale)), max(1, int(2 * np.sqrt(eigvals[0]) * scale)))
            center = self.world_to_px(x, y)
            cv2.ellipse(canvas, center, axes, angle_deg, 0, 360, color, 1, cv2.LINE_AA)
        except np.linalg.LinAlgError:
            pass

        center = self.world_to_px(x, y)
        state_color = {"confirmed": (0, 220, 0), "coasting": (0, 220, 220), "lost": (0, 0, 255)}.get(state, color)
        cv2.circle(canvas, center, 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 7, state_color, 2, cv2.LINE_AA)
        h_str = f" h={height:.2f}m" if height is not None else ""
        cv2.putText(canvas, f"id={track_id} {state} {src}{h_str}", (center[0] + 10, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def render(self, tracks: List, phase: str, ground_truth_markers: Optional[List[dict]] = None) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_grid(canvas)
        self._draw_camera(canvas)
        if ground_truth_markers:
            self._draw_ground_truth(canvas, ground_truth_markers)
        for t in tracks:
            if phase == "A":
                x, y = t.p_local[0], t.p_local[1]
                height = t.height_m
            else:
                x, y = t.p[0], t.p[1]
                height = t.height
            self._draw_track(canvas, t.world_track_id if phase == "A" else t.id, x, y, t.cov_xy, t.state, height, t.src)
        return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m3-config", required=True)
    ap.add_argument("--m4-config", required=True)
    ap.add_argument("--source", choices=["realsense", "synthetic"], default="realsense")
    ap.add_argument("--playback", default=None)
    ap.add_argument("--mock-scenario", default=None, choices=sorted(
        n[len("scenario_"):] for n in dir(mock_infer) if n.startswith("scenario_")
    ), help="use one of M3's scripted mock scenarios instead of a real model (--source synthetic only)")
    ap.add_argument("--ground-truth", default=None, help="a floor_markers-schema JSON to overlay for visual accuracy checking")
    ap.add_argument("--x-max", type=float, default=6.0)
    ap.add_argument("--y-half", type=float, default=3.5)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    m3_cfg = M3Config.load(args.m3_config)
    m4_cfg = M4Config.load(args.m4_config)

    ground_truth_markers = None
    if args.ground_truth:
        ground_truth_markers = json.loads(Path(args.ground_truth).read_text())["markers"]

    mask_set = None
    if m3_cfg.mask.masks_path:
        mask_set = load_masks(m3_cfg.mask.masks_path).get(m3_cfg.camera.cam_id, empty_mask_set())

    if args.mock_scenario:
        from poi_perception.capture.synthetic_source import SyntheticSource

        source = SyntheticSource(m3_cfg.camera, n_frames=args.max_frames)
        scenario_fn = getattr(mock_infer, f"scenario_{args.mock_scenario}")(
            frame_w=m3_cfg.camera.width, frame_h=m3_cfg.camera.height
        )
        estimator = PoseEstimator(mock_infer.MockPoseBackend(scenario_fn))
    else:
        source = build_source(m3_cfg, args.source, args.playback)
        estimator = PoseEstimator(build_backend(m3_cfg))
    estimator.warm_up()

    m3_pipeline = M3Pipeline(m3_cfg, mask_set=mask_set or empty_mask_set())

    walkable_grid = None
    if m4_cfg.phase == "B" and m4_cfg.phase_b.walkable_grid_path:
        walkable_grid = load_walkable_grid(m4_cfg.phase_b.walkable_grid_path)
    provider = build_frame_provider(m4_cfg, cam_cfg=m3_cfg.camera)
    m4_pipeline = M4Pipeline(m4_cfg, provider, walkable_grid=walkable_grid)

    bev = BirdEyeView(x_range=(-0.5, args.x_max), y_range=(-args.y_half, args.y_half))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    show_window = not args.no_window
    window_broken = False

    print("Press 'q' in the window to quit (Ctrl+C also works).")
    n = 0
    try:
        for frame in source:
            raw_dets = estimator.infer(frame)
            m3_dets = m3_pipeline.process(frame, raw_dets)
            world_tracks = m4_pipeline.process(frame, m3_dets)

            cam_panel = _annotate_camera_panel(frame.rgb, m3_dets)
            bev_panel = bev.render(world_tracks, m4_cfg.phase, ground_truth_markers)

            cam_h, cam_w = cam_panel.shape[:2]
            bev_h, bev_w = bev_panel.shape[:2]
            target_h = max(cam_h, bev_h)
            if cam_h != target_h:
                cam_panel = cv2.resize(cam_panel, (int(cam_w * target_h / cam_h), target_h))
            if bev_h != target_h:
                bev_panel = cv2.resize(bev_panel, (int(bev_w * target_h / bev_h), target_h))
            composite = np.hstack([cam_panel, bev_panel])

            if show_window and not window_broken:
                try:
                    cv2.imshow("M4 debug visualizer (camera | bird's-eye)", composite)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    print(f"cv2.imshow unavailable ({e}); falling back to --save-dir only.")
                    window_broken = True
                    if save_dir is None:
                        save_dir = Path("debug_frames_world")
                        save_dir.mkdir(parents=True, exist_ok=True)
                        print(f"Saving annotated frames to {save_dir}/ instead.")

            if save_dir:
                cv2.imwrite(str(save_dir / f"frame_{frame.frame_id:06d}.jpg"), composite)

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