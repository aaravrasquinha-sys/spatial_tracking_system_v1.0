"""
Section 4: runtime architecture.

    [Capture] --frame(t)--> [Inference: TensorRT pose + decode]
        --raw kpts/boxes(t)--> [Track + footpoint] --Detection2D(t)-->
        [out: queue to M4 / JSONL logger]

Capture and inference run on separate threads (a capture stall, e.g. a
USB hiccup, must not block the last inference from finishing and being
published) even though it's a single GPU context. Track + footpoint is
stateful (ByteTrack + the footpoint/torso logic) and runs as the "main"
loop here; everything upstream is stateless and already independently
tested (see inference/pose_infer.py, tracking/footpoint.py docstrings).

Section 6 (M6, the full runtime daemon) will eventually own this
thread/queue wiring as part of its single long-running process -- this
is written to lift into that later with the queues reused, not rewritten
(Section 11).
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from poi_perception.config import M3Config
from poi_perception.contracts import Frame
from poi_perception.inference.pose_infer import PoseEstimator, RawDetection
from poi_perception.io.clips import ClipRecorder
from poi_perception.io.detection2d import Detection2D, JsonlDetectionWriter
from poi_perception.log import get_logger
from poi_perception.runtime.masks import MaskSet, empty_mask_set
from poi_perception.runtime.queues import DropStaleQueue
from poi_perception.tracking.bytetrack_wrap import ByteTrackWrapper
from poi_perception.tracking.footpoint import compute_footpoint, compute_torso_polygon

log = get_logger("runtime.m3_daemon")

DetectionCallback = Callable[[Frame, List[Detection2D]], None]


class M3Pipeline:
    """The stateful "track + footpoint" stage, plus the glue to build a
    full Detection2D record. Exposed as a plain synchronous method
    (process) so it's testable deterministically (see tests/), and
    driven either directly (bench/eval tooling) or from the threaded
    daemon below (live operation)."""

    def __init__(
        self,
        cfg: M3Config,
        mask_set: Optional[MaskSet] = None,
        writer: Optional[JsonlDetectionWriter] = None,
        clip_recorder: Optional[ClipRecorder] = None,
        on_detections: Optional[DetectionCallback] = None,
    ):
        self.cfg = cfg
        self.mask_set = mask_set or empty_mask_set()
        self.tracker = ByteTrackWrapper(cfg.track)
        self.writer = writer
        self.clip_recorder = clip_recorder
        self.on_detections = on_detections
        self._dropped_masked_count = 0

    def process(self, frame: Frame, raw_dets: List[RawDetection]) -> List[Detection2D]:
        # Section 7: drop masked detections before they reach the tracker.
        kept = []
        for d in raw_dets:
            if self.mask_set.is_masked(d.bbox):
                self._dropped_masked_count += 1
                continue
            kept.append(d)

        tracked = self.tracker.update(kept)

        fp_cfg = self.cfg.footpoint
        out: List[Detection2D] = []
        for tp in tracked:
            kpts = tp.det.keypoints if tp.det is not None else {}
            det_conf = tp.det.det_conf if tp.det is not None else 0.0

            footpoint_px, footpoint_source, low_conf_fp = compute_footpoint(
                kpts, tp.bbox, fp_cfg.ankle_conf_thresh
            )
            torso_poly, torso_quality = compute_torso_polygon(
                kpts,
                tp.bbox,
                kp_conf_thresh=fp_cfg.ankle_conf_thresh,
                shrink_frac=fp_cfg.torso_shrink_frac,
                fallback_strip_frac=fp_cfg.fallback_strip_frac,
            )
            low_confidence = low_conf_fp or (tp.det is None)

            out.append(
                Detection2D(
                    t_capture=frame.t,
                    frame_id=frame.frame_id,
                    cam_id=frame.cam_id,
                    track_id_2d=tp.track_id_2d,
                    bbox=tp.bbox,
                    det_conf=det_conf,
                    keypoints=kpts,
                    footpoint_px=footpoint_px,
                    footpoint_source=footpoint_source,
                    torso_polygon_px=torso_poly,
                    torso_quality=torso_quality,
                    low_confidence=low_confidence,
                    track_state=tp.track_state,
                )
            )

        if self.writer is not None:
            self.writer.write_frame(frame.t, frame.frame_id, out)
        if self.clip_recorder is not None:
            self.clip_recorder.on_frame(frame.t, frame.frame_id, frame.rgb, out)
        if self.on_detections is not None:
            self.on_detections(frame, out)
        return out


class M3Daemon:
    """Owns the capture and inference threads plus the size-1 drop-stale
    queues between stages (Section 4). Runs process()'s track+footpoint
    stage on the calling thread."""

    def __init__(self, cfg: M3Config, source, pose_estimator: PoseEstimator, pipeline: M3Pipeline):
        self.cfg = cfg
        self.source = source
        self.pose_estimator = pose_estimator
        self.pipeline = pipeline
        self._stop = threading.Event()
        self._cap_queue: DropStaleQueue[Frame] = DropStaleQueue()
        self._infer_queue: DropStaleQueue[tuple] = DropStaleQueue()

    def _capture_loop(self) -> None:
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                self._cap_queue.put(frame)
        except Exception:
            log.exception("Capture thread crashed")
        finally:
            self._cap_queue.close()

    def _inference_loop(self) -> None:
        try:
            log.info("Warming up inference engine (Section 3: first-call latency is not representative)...")
            self.pose_estimator.warm_up()

            while not self._stop.is_set():
                frame = self._cap_queue.get(timeout=1.0)
                if frame is None:
                    if self._cap_queue.closed:
                        break
                    continue
                raw_dets = self.pose_estimator.infer(frame)
                self._infer_queue.put((frame, raw_dets))
        except Exception:
            log.exception("Inference thread crashed")
        finally:
            self._infer_queue.close()

    def run(self, max_frames: Optional[int] = None) -> None:
        cap_thread = threading.Thread(target=self._capture_loop, name="m3-capture", daemon=True)
        infer_thread = threading.Thread(target=self._inference_loop, name="m3-inference", daemon=True)
        cap_thread.start()
        infer_thread.start()

        n_processed = 0
        try:
            while True:
                item = self._infer_queue.get(timeout=1.0)
                if item is None:
                    if self._infer_queue.closed:
                        log.info("Upstream stages ended; stopping.")
                        break
                    continue
                frame, raw_dets = item
                self.pipeline.process(frame, raw_dets)
                n_processed += 1
                if max_frames is not None and n_processed >= max_frames:
                    log.info(f"Reached max_frames={max_frames}; stopping.")
                    break
        finally:
            self.stop()
            cap_thread.join(timeout=5.0)
            infer_thread.join(timeout=5.0)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.source.close()
        except Exception:
            pass
        self._cap_queue.close()
        self._infer_queue.close()


def build_source(cfg: M3Config, source_kind: str, playback_path: Optional[str]):
    if source_kind == "realsense":
        from poi_perception.capture.realsense_source import RealSenseSource

        return RealSenseSource(cfg.camera, playback_path=playback_path)
    if source_kind == "synthetic":
        from poi_perception.capture.synthetic_source import SyntheticSource

        return SyntheticSource(cfg.camera)
    raise ValueError(f"Unknown source kind: {source_kind}")


def build_backend(cfg: M3Config):
    if cfg.model.backend == "trt":
        from poi_perception.models.trt_engine import TRTPoseBackend

        if not cfg.model.engine_path:
            raise ValueError("model.engine_path is required for backend='trt'")
        return TRTPoseBackend(
            cfg.model.engine_path,
            cfg.model.input_width,
            cfg.model.input_height,
            det_conf_thresh=cfg.model.det_conf_thresh,
            nms_iou_thresh=cfg.model.nms_iou_thresh,
            keypoint_conf_thresh_for_decode=cfg.model.keypoint_conf_thresh_for_decode,
        )
    if cfg.model.backend == "ultralytics":
        from poi_perception.inference.pose_infer import UltralyticsPoseBackend

        if not cfg.model.weights_path:
            raise ValueError("model.weights_path is required for backend='ultralytics'")
        return UltralyticsPoseBackend(cfg.model.weights_path, det_conf_thresh=cfg.model.det_conf_thresh)
    raise ValueError(f"backend='{cfg.model.backend}' has no live implementation; use mock_infer directly for tests")


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to an M3Config JSON file")
    ap.add_argument("--source", choices=["realsense", "synthetic"], default="realsense")
    ap.add_argument("--playback", default=None, help="replay a recorded .bag instead of a live camera")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = M3Config.load(args.config)
    source = build_source(cfg, args.source, args.playback)
    backend = build_backend(cfg)
    estimator = PoseEstimator(backend)

    mask_set = None
    if cfg.mask.masks_path:
        from poi_perception.runtime.masks import load_masks

        all_masks = load_masks(cfg.mask.masks_path)
        mask_set = all_masks.get(cfg.camera.cam_id, empty_mask_set())

    jsonl_path = Path(cfg.output.jsonl_dir) / f"{cfg.camera.cam_id}_{int(time.time())}.jsonl"
    writer = JsonlDetectionWriter(jsonl_path)
    clip_recorder = ClipRecorder(
        Path(cfg.output.clips_dir) / cfg.camera.cam_id,
        fps=cfg.camera.fps,
        low_confidence_streak_s=cfg.output.low_confidence_streak_s,
    )

    pipeline = M3Pipeline(cfg, mask_set=mask_set, writer=writer, clip_recorder=clip_recorder)
    daemon = M3Daemon(cfg, source, estimator, pipeline)
    try:
        daemon.run(max_frames=args.max_frames)
    finally:
        writer.close()


if __name__ == "__main__":
    main()