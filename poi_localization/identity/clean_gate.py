"""Clean-sample gate: learn only when sure (R1 spec section 8). Pure logic, no pixels.

`ok_for_training` = all eight rules. `ok_for_evidence` = the reduced set (1, 4, 5, 8): an open case may
still use a partially occluded newcomer at reduced weight, but such an observation never trains anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from poi_localization.identity.config import IdentityConfig

BBox = Tuple[float, float, float, float]


@dataclass
class GateInput:
    track_state: str                              # "confirmed" | "coasting" | ...
    measured_this_frame: bool                     # src != predicted
    identity_resolving: bool                      # a resolving track never updates a gallery profile
    s_since_2d_mismatch: Optional[float]          # None = never
    swap_flag: bool
    max_iou_with_others: float
    min_dist_to_others_m: float                   # inf when alone
    torso_depth_m: float
    det_conf: float
    n_kp_conf: int                                # keypoints with conf >= kp_conf_min
    bbox: BBox
    img_w: int
    img_h: int
    upright: bool
    calib_ok: bool


@dataclass
class GateResult:
    ok_for_training: bool
    ok_for_evidence: bool
    failed: List[int] = field(default_factory=list)   # failed rule numbers (1..8)


def bbox_iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def isolation(bbox: BBox, xy: Sequence[float], others: Sequence[Tuple[BBox, Sequence[float]]]) -> Tuple[float, float]:
    """(max IoU with any other bbox, min floor distance to any other track). (0, inf) when alone."""
    max_iou, min_d = 0.0, float("inf")
    for ob, oxy in others:
        max_iou = max(max_iou, bbox_iou(bbox, ob))
        min_d = min(min_d, float(np.hypot(xy[0] - oxy[0], xy[1] - oxy[1])))
    return max_iou, min_d


def check_clean(g: GateInput, cfg: IdentityConfig) -> GateResult:
    failed: List[int] = []
    # 1. confirmed, measured this frame, and not a still-resolving case
    if not (g.track_state == "confirmed" and g.measured_this_frame):
        failed.append(1)
    elif g.identity_resolving:
        failed.append(1)          # only fails TRAINING; evidence (below) tolerates a resolving track
    # 2. no recent 2D-id mismatch, no swap-monitor conflict
    if g.swap_flag or (g.s_since_2d_mismatch is not None and g.s_since_2d_mismatch < 2.0):
        failed.append(2)
    # 3. isolation
    if g.max_iou_with_others > cfg.isolation_iou_max or g.min_dist_to_others_m < cfg.isolation_min_dist_m:
        failed.append(3)
    # 4. range
    if not (cfg.min_range_m <= g.torso_depth_m <= cfg.max_range_m):
        failed.append(4)
    # 5. detection quality
    if g.det_conf < 0.5 or g.n_kp_conf < 10:
        failed.append(5)
    # 6. bbox not touching the image border (>= 4 px margin)
    x1, y1, x2, y2 = g.bbox
    if x1 < 4 or y1 < 4 or x2 > g.img_w - 4 or y2 > g.img_h - 4:
        failed.append(6)
    # 7. upright
    if not g.upright:
        failed.append(7)
    # 8. calibration ok (engine not frozen)
    if not g.calib_ok:
        failed.append(8)

    training = not failed
    # evidence: rules 1 (measured + confirmed; a resolving track is fine), 4, 5, 8
    ev_ok_1 = g.track_state == "confirmed" and g.measured_this_frame
    evidence = ev_ok_1 and not ({4, 5, 8} & set(failed))
    return GateResult(training, evidence, failed)
