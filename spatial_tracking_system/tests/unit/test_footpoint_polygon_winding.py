"""
Regression tests for a self-intersecting ("bowtie") torso polygon.

COCO's `left_shoulder` is the person's own left side, which for a person
facing the camera lands on the image's *right* (larger x). The previous
implementation of compute_torso_polygon's "partial" branch (and the
"full" branch before it) assigned "top_left_x"/"top_right_x" by
anatomical label rather than by actual pixel position, so a
camera-facing person with a fully-visible shoulder pair but a
low-confidence hip pair produced a self-intersecting quad -- cv2.fillPoly
silently draws that as two small triangles instead of one trapezoid,
roughly halving the sampled area exactly when samples are scarcest
(partial-occlusion frames).

These tests pin down: (1) the "full" and "partial" branches produce a
simple (non-self-intersecting) polygon regardless of facing direction,
and (2) the filled pixel area is the same whichever way the person faces
the camera, since the room doesn't care about anatomical labels.
"""
import cv2
import numpy as np

from poi_perception.tracking.footpoint import compute_torso_polygon

BBOX = (280.0, 150.0, 380.0, 450.0)


def _filled_area(poly):
    mask = np.zeros((480, 640), np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return int(mask.sum())


def _is_simple_quad(poly):
    """A quad is simple (non-self-intersecting) iff its two diagonals'
    intersection test agrees with a non-bowtie shoelace-area check: the
    shoelace formula's absolute value should equal the sum of the two
    triangle areas the quad decomposes into when it is NOT self-
    intersecting. Easiest robust check here: a simple quad's filled
    raster area is non-trivially larger than a bowtie's (a bowtie's
    raster area is only the two small opposing triangles). We assert
    against a concrete geometric lower bound instead of reproducing a
    general polygon-simplicity test.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return _filled_area(poly) > 0.5 * bbox_area  # a bowtie fills well under half its bbox


def test_partial_torso_facing_camera_is_not_a_bowtie():
    # left_shoulder.x > right_shoulder.x: the common case for someone
    # facing the camera. Hips present but below confidence threshold.
    kpts = {
        "left_shoulder": (360.0, 200.0, 0.9),
        "right_shoulder": (300.0, 200.0, 0.9),
        "left_hip": (355.0, 300.0, 0.1),
        "right_hip": (305.0, 300.0, 0.1),
    }
    poly, quality = compute_torso_polygon(kpts, BBOX)
    assert quality == "partial"
    assert _is_simple_quad(poly)


def test_partial_torso_area_is_facing_direction_invariant():
    facing_camera = {
        "left_shoulder": (360.0, 200.0, 0.9),
        "right_shoulder": (300.0, 200.0, 0.9),
        "left_hip": (355.0, 300.0, 0.1),
        "right_hip": (305.0, 300.0, 0.1),
    }
    back_to_camera = {
        "left_shoulder": (300.0, 200.0, 0.9),
        "right_shoulder": (360.0, 200.0, 0.9),
        "left_hip": (305.0, 300.0, 0.1),
        "right_hip": (355.0, 300.0, 0.1),
    }
    poly_a, _ = compute_torso_polygon(facing_camera, BBOX)
    poly_b, _ = compute_torso_polygon(back_to_camera, BBOX)
    assert _filled_area(poly_a) == _filled_area(poly_b)


def test_full_torso_facing_camera_is_not_a_bowtie():
    kpts = {
        "left_shoulder": (360.0, 200.0, 0.9),
        "right_shoulder": (300.0, 200.0, 0.9),
        "left_hip": (355.0, 300.0, 0.9),
        "right_hip": (305.0, 300.0, 0.9),
    }
    poly, quality = compute_torso_polygon(kpts, BBOX)
    assert quality == "full"
    assert _is_simple_quad(poly)


def test_partial_single_shoulder_only_is_convex_and_on_the_correct_side():
    # A single left_shoulder at image-right: the box edge used for the
    # opposite (unseen) side must be the far edge (x1), not x2, or the
    # polygon would be reduced to a sliver instead of using the space
    # actually available.
    kpts = {"left_shoulder": (360.0, 200.0, 0.9)}
    poly, quality = compute_torso_polygon(kpts, BBOX, shrink_frac=0.0)
    assert quality == "partial"
    top_y = min(p[1] for p in poly)
    top_xs = sorted(p[0] for p in poly if p[1] == top_y)
    x1, _, x2, _ = BBOX
    # top edge (shoulders): the seen shoulder (360) paired with the far
    # box edge (x1=280) on the unseen side -- not the near edge (x2),
    # which would shrink the polygon to a sliver instead of using the
    # space actually available.
    assert top_xs == [x1, 360.0]
    assert _is_simple_quad(poly)
