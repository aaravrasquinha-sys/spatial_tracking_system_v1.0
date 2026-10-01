"""
Preprocess and postprocess for a raw yolo11n-pose / yolov8n-pose
TensorRT output. Pure functions, no TensorRT/CUDA dependency, so they're
unit-testable without an engine or GPU.

Output layout assumed (standard ultralytics pose export, single class
"person"): shape (1, 56, N) where the 56 channels are
[cx, cy, w, h, box_conf, (kp_x, kp_y, kp_conf) x 17], in the *model
input* pixel space (e.g. 640x640 if letterboxed). decode_yolo_pose_output
un-letterboxes back to the original frame's pixel space before returning.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from poi_perception.contracts import COCO_KEYPOINT_NAMES
from poi_perception.inference.pose_infer import RawDetection

N_KEYPOINTS = 17
CHANNELS = 5 + 3 * N_KEYPOINTS  # 56


class LetterboxParams:
    """Pad offsets and scale needed to invert a letterbox resize exactly
    (Section 1: "keep the letterbox parameters around explicitly so you
    can invert them exactly when mapping keypoints back to the original
    640x480 depth-aligned frame"). If the exporter doesn't force a square
    input, scale_x/scale_y differ and pad is zero -- this class covers
    both the plain-resize and the letterboxed case with one formula."""

    __slots__ = ("scale_x", "scale_y", "pad_x", "pad_y")

    def __init__(self, scale_x: float, scale_y: float, pad_x: float, pad_y: float):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.pad_x = pad_x
        self.pad_y = pad_y

    @classmethod
    def identity(cls) -> "LetterboxParams":
        return cls(1.0, 1.0, 0.0, 0.0)

    @classmethod
    def compute(
        cls, src_w: int, src_h: int, dst_w: int, dst_h: int, letterbox: bool
    ) -> "LetterboxParams":
        """If not letterboxing (Section 1's preference: don't upscale or
        letterbox to a square unless export tooling forces it), and
        src/dst differ, this still records the plain per-axis scale so
        inversion is correct either way."""
        if not letterbox:
            return cls(dst_w / src_w, dst_h / src_h, 0.0, 0.0)
        scale = min(dst_w / src_w, dst_h / src_h)
        new_w, new_h = src_w * scale, src_h * scale
        pad_x = (dst_w - new_w) / 2.0
        pad_y = (dst_h - new_h) / 2.0
        return cls(scale, scale, pad_x, pad_y)

    def undo_point(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.pad_x) / self.scale_x, (y - self.pad_y) / self.scale_y


def preprocess(rgb: np.ndarray, dst_w: int, dst_h: int, letterbox: bool) -> Tuple[np.ndarray, LetterboxParams]:
    """rgb: HxWx3 uint8 -> (1, 3, dst_h, dst_w) float32 in [0, 1], plus the
    params needed to invert the resize on the way out."""
    src_h, src_w = rgb.shape[:2]
    params = LetterboxParams.compute(src_w, src_h, dst_w, dst_h, letterbox)

    # Minimal, dependency-free resize (nearest) is enough for a placeholder;
    # swap in cv2.resize/cv2.warpAffine for production-quality interpolation.
    import cv2

    if letterbox:
        new_w = int(round(src_w * params.scale_x))
        new_h = int(round(src_h * params.scale_y))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((dst_h, dst_w, 3), 114, dtype=np.uint8)
        ox, oy = int(round(params.pad_x)), int(round(params.pad_y))
        canvas[oy : oy + new_h, ox : ox + new_w] = resized
        img = canvas
    else:
        img = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    chw = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[None, ...], params


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) -> (N,M) IoU matrix."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Class-agnostic NMS (single class: "person"). Returns kept indices,
    highest score first. Pure numpy -- no torchvision dependency."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_xyxy(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def decode_yolo_pose_output(
    raw_output: np.ndarray,
    letterbox_params: LetterboxParams,
    src_w: int,
    src_h: int,
    det_conf_thresh: float,
    nms_iou_thresh: float,
    keypoint_conf_thresh_for_decode: float,
) -> List[RawDetection]:
    """raw_output: (1, 56, N) or (56, N) model output -> decoded, NMS'd,
    un-letterboxed detections in the original frame's pixel space."""
    arr = raw_output
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] == CHANNELS:
        arr = arr.T  # -> (N, 56)
    elif arr.shape[1] != CHANNELS:
        raise ValueError(f"Unexpected pose output shape {raw_output.shape}, expected 56 channels")

    conf = arr[:, 4]
    mask = conf >= det_conf_thresh
    arr = arr[mask]
    if arr.shape[0] == 0:
        return []

    cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = arr[:, 4]

    keep = nms(boxes_xyxy, scores, nms_iou_thresh)
    arr = arr[keep]
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    out: List[RawDetection] = []
    for row_box, row_score, row in zip(boxes_xyxy, scores, arr):
        x1, y1 = letterbox_params.undo_point(float(row_box[0]), float(row_box[1]))
        x2, y2 = letterbox_params.undo_point(float(row_box[2]), float(row_box[3]))
        x1, x2 = np.clip([x1, x2], 0, src_w - 1)
        y1, y2 = np.clip([y1, y2], 0, src_h - 1)

        kp_raw = row[5:].reshape(N_KEYPOINTS, 3)
        kp_dict: Dict[str, Tuple[float, float, float]] = {}
        for j, name in enumerate(COCO_KEYPOINT_NAMES):
            kx, ky, kc = float(kp_raw[j, 0]), float(kp_raw[j, 1]), float(kp_raw[j, 2])
            if kc < keypoint_conf_thresh_for_decode:
                kp_dict[name] = (0.0, 0.0, 0.0)
                continue
            ux, uy = letterbox_params.undo_point(kx, ky)
            kp_dict[name] = (float(ux), float(uy), kc)

        out.append(RawDetection(bbox=(float(x1), float(y1), float(x2), float(y2)), det_conf=float(row_score), keypoints=kp_dict))
    return out
