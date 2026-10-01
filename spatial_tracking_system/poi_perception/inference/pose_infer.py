"""
Section 4 "Inference" stage: single TensorRT context, batch size 1, FP16.
Preprocess -> engine forward -> decode boxes+keypoints+scores -> NMS.

Kept stateless and side-effect-free (Section 4) so it's independently
testable and swappable -- e.g. to benchmark yolo11n-pose against
yolov8n-pose head to head (Section 1), or to run CPU-only via ultralytics
during development on a machine with no TensorRT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple

import numpy as np

from poi_perception.contracts import COCO_KEYPOINT_NAMES, Frame
from poi_perception.log import get_logger

log = get_logger("inference.pose")


@dataclass
class RawDetection:
    """One person detection for one frame, before tracking. Pixel space,
    matches the input frame's resolution (not the model's letterboxed
    input -- see decode.py for the un-letterbox step)."""

    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    det_conf: float
    keypoints: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # name -> (x, y, conf)


class PoseBackend(Protocol):
    """Anything that turns one RGB frame into a list of RawDetection.
    Implementations: models.trt_engine.TRTPoseBackend (deployment),
    UltralyticsPoseBackend (dev/CPU), inference.mock_infer.MockPoseBackend
    (hardware-free tests)."""

    def infer(self, frame: Frame) -> List[RawDetection]: ...

    def warm_up(self, n: int = 5) -> None: ...


class PoseEstimator:
    """Thin, stateless wrapper around whatever backend is configured.
    This is the only object the runtime daemon talks to for inference,
    so swapping models/backends never touches runtime code."""

    def __init__(self, backend: PoseBackend):
        self.backend = backend

    def warm_up(self, n: int = 5) -> None:
        """First-call latency on a freshly loaded TensorRT engine is not
        representative (Section 3) -- always warm up before timing."""
        self.backend.warm_up(n)

    def infer(self, frame: Frame) -> List[RawDetection]:
        return self.backend.infer(frame)


class UltralyticsPoseBackend:
    """Runs a .pt (or an ultralytics-exported .engine) directly through
    the ultralytics package. Convenient for development on a machine
    without a hand-built TensorRT pipeline; on the Orin, prefer
    models.trt_engine.TRTPoseBackend for the manual preprocess/decode
    control Section 4 asks for. Import is lazy so this module doesn't
    require ultralytics unless this backend is actually used."""

    def __init__(self, weights_path: str, det_conf_thresh: float = 0.1, device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not importable. `pip install ultralytics`, or use "
                "MockPoseBackend for hardware-free development (see scripts/demo_synthetic.py)."
            ) from e
        self._model = YOLO(weights_path)
        self.det_conf_thresh = det_conf_thresh
        self.device = device

    def warm_up(self, n: int = 5) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self._model.predict(dummy, verbose=False, device=self.device)

    def infer(self, frame: Frame) -> List[RawDetection]:
        results = self._model.predict(
            frame.rgb, verbose=False, device=self.device, conf=self.det_conf_thresh
        )
        out: List[RawDetection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or r.keypoints is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, conf
        for i in range(len(boxes)):
            kp_dict = {
                name: (float(kpts[i, j, 0]), float(kpts[i, j, 1]), float(kpts[i, j, 2]))
                for j, name in enumerate(COCO_KEYPOINT_NAMES)
            }
            out.append(
                RawDetection(
                    bbox=tuple(float(v) for v in boxes[i]),
                    det_conf=float(confs[i]),
                    keypoints=kp_dict,
                )
            )
        return out
