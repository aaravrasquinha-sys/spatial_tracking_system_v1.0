"""
Section 6: ByteTrack on top of the raw per-frame pose detections, with
two customizations beyond a stock ByteTrack integration:

  1. Matching uses keypoint confidence in the loop, not just box IoU +
     score -- a detection with confident, anatomically-plausible
     keypoints gets a small matching-score bonus on top of IoU, so a
     partially-occluded person (low box confidence, but a couple of
     good keypoints) is a little more likely to keep its identity
     through the low-confidence second pass.
  2. The lost-track buffer (how many frames a track survives with no
     matching detection before being dropped) is a config value meant
     to be tuned against this room's actual furniture-occlusion
     durations (Section 9), not a COCO-video default.

This module's ByteTrackWrapper is intentionally self-contained (numpy +
scipy only) rather than wrapping an external bytetrack/yolox package --
the two customizations above touch the matching cost function directly,
which is easier to keep correct and testable in ~200 lines here than to
bolt onto a third-party implementation's internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from poi_perception.config import TrackConfig
from poi_perception.inference.pose_infer import RawDetection
from poi_perception.tracking.kalman_box import KalmanBoxTracker

BBox = Tuple[float, float, float, float]

# Second-stage (low-confidence) association is deliberately looser than
# the first stage and IoU-only (Section 6: no appearance/keypoint signal
# here -- a blurry, low-score detection's keypoints aren't trustworthy
# either), matching stock ByteTrack's own design.
_SECOND_STAGE_IOU_THRESH = 0.5
_DISALLOWED_COST = 1e6


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mean_keypoint_conf(det: RawDetection) -> float:
    if not det.keypoints:
        return 0.0
    confs = [c for (_, _, c) in det.keypoints.values()]
    return float(np.mean(confs)) if confs else 0.0


def _match_score(track_bbox: BBox, det: RawDetection, kpt_weight: float) -> Tuple[float, float]:
    """Returns (raw_iou, combined_score). combined_score is used to rank
    candidate matches; raw_iou is what actually gates whether a match is
    allowed at all, so the keypoint bonus can break ties/prefer a better
    detection but can never manufacture a match out of a poor box
    overlap."""
    iou = _iou(track_bbox, det.bbox)
    bonus = kpt_weight * _mean_keypoint_conf(det) * iou
    return iou, iou + bonus


@dataclass
class TrackedPerson:
    track_id_2d: int
    bbox: BBox  # matched detection's bbox this frame, or the KF prediction while coasting
    det: Optional[RawDetection]  # None while coasting (lost_buffer) -- no fresh keypoints available
    track_state: str  # "new" | "tracked" | "lost_buffer"
    time_since_update: int
    hits: int


class _Track:
    __slots__ = ("id", "kf", "hits", "time_since_update", "confirmed")

    def __init__(self, track_id: int, det: RawDetection):
        self.id = track_id
        self.kf = KalmanBoxTracker(det.bbox)
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False


class ByteTrackWrapper:
    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections: List[RawDetection]) -> List[TrackedPerson]:
        cfg = self.cfg

        # 1. Predict every live track forward one frame.
        predicted_bbox: Dict[int, BBox] = {tid: t.kf.predict() for tid, t in self._tracks.items()}

        high = [d for d in detections if d.det_conf >= cfg.track_high_thresh]
        low = [d for d in detections if cfg.track_low_thresh <= d.det_conf < cfg.track_high_thresh]

        unmatched_track_ids = set(self._tracks.keys())
        matched_this_frame: Dict[int, RawDetection] = {}

        # 2. First stage: all live tracks vs. high-confidence detections,
        # IoU-gated, keypoint-confidence-weighted for ranking.
        if unmatched_track_ids and high:
            track_ids = list(unmatched_track_ids)
            cost = np.full((len(track_ids), len(high)), _DISALLOWED_COST)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(high):
                    iou, score = _match_score(predicted_bbox[tid], det, cfg.keypoint_conf_weight)
                    if iou >= cfg.match_iou_thresh:
                        cost[i, j] = -score
            row_idx, col_idx = linear_sum_assignment(cost)
            used_dets = set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] >= _DISALLOWED_COST:
                    continue
                tid = track_ids[r]
                matched_this_frame[tid] = high[c]
                used_dets.add(c)
                unmatched_track_ids.discard(tid)
            high = [d for j, d in enumerate(high) if j not in used_dets]

        # 3. Second stage: remaining unmatched tracks vs. low-confidence
        # detections, IoU-only, looser threshold (Section 6's "low-
        # confidence second pass" -- keeps an ID alive through partial
        # occlusion instead of ending the track).
        if unmatched_track_ids and low:
            track_ids = list(unmatched_track_ids)
            cost = np.full((len(track_ids), len(low)), _DISALLOWED_COST)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(low):
                    iou = _iou(predicted_bbox[tid], det.bbox)
                    if iou >= _SECOND_STAGE_IOU_THRESH:
                        cost[i, j] = -iou
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] >= _DISALLOWED_COST:
                    continue
                tid = track_ids[r]
                matched_this_frame[tid] = low[c]
                unmatched_track_ids.discard(tid)
            # low-confidence detections are never used to spawn new tracks
            # (Section 6) -- whether matched or not, they're done here.

        # 4. Apply matches / advance coasting for existing tracks.
        results: List[TrackedPerson] = []
        for tid, track in list(self._tracks.items()):
            if tid in matched_this_frame:
                det = matched_this_frame[tid]
                track.kf.update(det.bbox)
                track.time_since_update = 0
                track.hits += 1
                if track.hits >= cfg.min_hits_to_confirm:
                    track.confirmed = True
                state = "tracked" if track.confirmed or track.hits > 1 else "new"
                results.append(
                    TrackedPerson(tid, det.bbox, det, state, 0, track.hits)
                )
            else:
                track.time_since_update += 1
                if track.time_since_update > cfg.lost_buffer_frames:
                    del self._tracks[tid]
                    continue
                results.append(
                    TrackedPerson(tid, predicted_bbox[tid], None, "lost_buffer", track.time_since_update, track.hits)
                )

        # 5. Spawn new tracks from unmatched high-confidence detections.
        for det in high:
            if det.det_conf < cfg.new_track_thresh:
                continue
            tid = self._new_id()
            self._tracks[tid] = _Track(tid, det)
            results.append(TrackedPerson(tid, det.bbox, det, "new", 0, 1))

        return results

    def active_track_count(self) -> int:
        return len(self._tracks)
