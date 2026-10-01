"""
Section 5: per detected person, per frame, compute the ground-contact
pixel (footpoint) and the depth-sampling region (torso polygon).

Both functions are pure and unit-testable on a single frame's keypoints
+ bbox -- no tracker state involved (Section 4: "Everything upstream of
this stage can be unit-tested on single frames").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]
KeypointDict = Dict[str, Tuple[float, float, float]]  # name -> (x, y, conf)


def _confident(kpts: KeypointDict, name: str, thresh: float) -> Optional[Point]:
    if name not in kpts:
        return None
    x, y, c = kpts[name]
    if c < thresh:
        return None
    return (x, y)


def compute_footpoint(
    kpts: KeypointDict,
    bbox: Tuple[float, float, float, float],
    ankle_conf_thresh: float = 0.5,
) -> Tuple[Point, str, bool]:
    """Returns (footpoint_px, footpoint_source, low_confidence).

    Priority order (Section 5):
      1. both ankles confident -> midpoint                  -> "ankles"
      2. one ankle confident   -> that ankle's pixel          -> "ankle_single"
      3. neither confident     -> bottom-center of the box    -> "bbox_fallback", low_confidence=True
    """
    left = _confident(kpts, "left_ankle", ankle_conf_thresh)
    right = _confident(kpts, "right_ankle", ankle_conf_thresh)

    if left is not None and right is not None:
        return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2), "ankles", False
    if left is not None or right is not None:
        p = left if left is not None else right
        return p, "ankle_single", False

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2), "bbox_fallback", True


def _shrink_toward_centroid(poly: List[Point], frac: float) -> List[Point]:
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return [(cx + (1 - frac) * (px - cx), cy + (1 - frac) * (py - cy)) for px, py in poly]


def compute_torso_polygon(
    kpts: KeypointDict,
    bbox: Tuple[float, float, float, float],
    kp_conf_thresh: float = 0.5,
    shrink_frac: float = 0.15,
    fallback_strip_frac: Tuple[float, float] = (0.20, 0.60),
) -> Tuple[List[Point], str]:
    """Returns (torso_polygon_px, torso_quality).

    Priority order (Section 5):
      1. both shoulders + both hips confident -> quad, shrunk toward its
         own centroid                                         -> "full"
      2. partial keypoints -> best available subset extended to a
         reasonable polygon (shoulders + box width at hip height) -> "partial"
      3. no usable keypoints -> a vertical strip through the box center,
         narrower than the full box width                     -> "fallback"
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    l_sh = _confident(kpts, "left_shoulder", kp_conf_thresh)
    r_sh = _confident(kpts, "right_shoulder", kp_conf_thresh)
    l_hip = _confident(kpts, "left_hip", kp_conf_thresh)
    r_hip = _confident(kpts, "right_hip", kp_conf_thresh)

    if l_sh and r_sh and l_hip and r_hip:
        # Order the quad by actual pixel x, not by anatomical L/R label.
        # COCO's "left_shoulder" is the person's own left, which for a
        # camera-facing person is on the image's RIGHT side -- so
        # l_sh.x > r_sh.x is the *common* case, not an edge case. The
        # previous [l_sh, r_sh, r_hip, l_hip] ordering silently assumed
        # l_sh.x < r_sh.x and produced a self-intersecting (bowtie) quad
        # whenever a person faced the camera with only a partial keypoint
        # set below (see the "partial" branch, same bug, fixed the same
        # way). Sorting by x guarantees a simple (non-self-intersecting)
        # quadrilateral regardless of which way the person is facing.
        top = sorted((l_sh, r_sh), key=lambda p: p[0])
        bottom = sorted((l_hip, r_hip), key=lambda p: p[0])
        quad = [top[0], top[1], bottom[1], bottom[0]]
        return _shrink_toward_centroid(quad, shrink_frac), "full"

    have_any_shoulder = l_sh is not None or r_sh is not None
    have_any_hip = l_hip is not None or r_hip is not None
    if have_any_shoulder or have_any_hip:
        # "shoulders + box width at hip height": use whichever shoulder
        # y's are available for the top edge (falling back to a chest
        # estimate at 20% box height if neither shoulder is confident but
        # a hip is, e.g. arms-occluded-but-hip-visible), and the full box
        # width (or the available hip x's) at an estimated (or measured)
        # hip height for the bottom edge.
        shoulder_pts = [p for p in (l_sh, r_sh) if p is not None]
        hip_pts = [p for p in (l_hip, r_hip) if p is not None]

        shoulder_ys = [p[1] for p in shoulder_pts]
        top_y = sum(shoulder_ys) / len(shoulder_ys) if shoulder_ys else y1 + 0.20 * h

        hip_ys = [p[1] for p in hip_pts]
        hip_y = sum(hip_ys) / len(hip_ys) if hip_ys else y1 + 0.55 * h

        # Build each edge's (left_x, right_x) from whatever keypoints are
        # actually available, by pixel position -- never by anatomical
        # label -- and fall back to the box edges only for a side with no
        # keypoint at all. This is the same fix as the "full" branch:
        # deciding left/right from the label (l_sh vs r_sh) rather than
        # from x assumed a facing direction and produced a bowtie for a
        # camera-facing person. Using x directly (with a single-keypoint
        # side falling back to whichever box edge is on the *other* side
        # of that point) keeps every case convex.
        if len(shoulder_pts) == 2:
            top_left_x, top_right_x = sorted(p[0] for p in shoulder_pts)
        elif len(shoulder_pts) == 1:
            sx = shoulder_pts[0][0]
            top_left_x, top_right_x = (sx, x2) if sx <= (x1 + x2) / 2 else (x1, sx)
        else:
            top_left_x, top_right_x = x1, x2

        if len(hip_pts) == 2:
            bottom_left_x, bottom_right_x = sorted(p[0] for p in hip_pts)
        elif len(hip_pts) == 1:
            hx = hip_pts[0][0]
            bottom_left_x, bottom_right_x = (hx, x2) if hx <= (x1 + x2) / 2 else (x1, hx)
        else:
            bottom_left_x, bottom_right_x = x1, x2

        quad = [
            (top_left_x, top_y),
            (top_right_x, top_y),
            (bottom_right_x, hip_y),
            (bottom_left_x, hip_y),
        ]
        return _shrink_toward_centroid(quad, shrink_frac), "partial"

    top_frac, bottom_frac = fallback_strip_frac
    strip_y1 = y1 + top_frac * h
    strip_y2 = y1 + bottom_frac * h
    # narrower than the full box width -- roughly chest-to-hip width
    strip_half_w = w * 0.20
    cx = (x1 + x2) / 2
    quad = [
        (cx - strip_half_w, strip_y1),
        (cx + strip_half_w, strip_y1),
        (cx + strip_half_w, strip_y2),
        (cx - strip_half_w, strip_y2),
    ]
    return quad, "fallback"
