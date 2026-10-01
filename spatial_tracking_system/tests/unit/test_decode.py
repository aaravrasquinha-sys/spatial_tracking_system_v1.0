import numpy as np

from poi_perception.inference.decode import LetterboxParams, nms


def test_letterbox_identity_roundtrip():
    p = LetterboxParams.identity()
    assert p.undo_point(10, 20) == (10, 20)


def test_letterbox_plain_resize_inversion():
    # 640x480 -> 320x320 plain resize (no padding, Section 1's preferred
    # path when the exporter doesn't force a square input).
    p = LetterboxParams.compute(src_w=640, src_h=480, dst_w=320, dst_h=320, letterbox=False)
    # a point at the model-input center should map back near source center
    ux, uy = p.undo_point(160, 160)
    assert abs(ux - 320) < 1e-6
    assert abs(uy - 240) < 1e-6


def test_letterbox_square_padding_inversion():
    # 640x480 -> 640x640 letterboxed: scale = min(640/640, 640/480) = 1.0,
    # pad_x = 0, pad_y = (640 - 480)/2 = 80
    p = LetterboxParams.compute(src_w=640, src_h=480, dst_w=640, dst_h=640, letterbox=True)
    assert abs(p.pad_y - 80.0) < 1e-6
    assert abs(p.pad_x - 0.0) < 1e-6
    ux, uy = p.undo_point(100, 80 + 50)  # 50px into the un-padded region
    assert abs(ux - 100.0) < 1e-6
    assert abs(uy - 50.0) < 1e-6


def test_nms_drops_overlapping_lower_score_box():
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],  # heavily overlapping with box 0
            [300, 300, 400, 400],  # far away, independent
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert 0 in keep
    assert 1 not in keep
    assert 2 in keep


def test_nms_empty_input():
    keep = nms(np.zeros((0, 4)), np.zeros((0,)), 0.5)
    assert len(keep) == 0
