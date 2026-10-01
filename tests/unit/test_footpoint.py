from poi_perception.tracking.footpoint import compute_footpoint, compute_torso_polygon

BBOX = (100.0, 50.0, 160.0, 220.0)


def _kpts(**overrides):
    base = {}
    base.update(overrides)
    return base


def test_footpoint_both_ankles_confident_uses_midpoint():
    kpts = {
        "left_ankle": (110.0, 210.0, 0.9),
        "right_ankle": (130.0, 212.0, 0.85),
    }
    px, source, low_conf = compute_footpoint(kpts, BBOX, ankle_conf_thresh=0.5)
    assert source == "ankles"
    assert not low_conf
    assert px == ((110.0 + 130.0) / 2, (210.0 + 212.0) / 2)


def test_footpoint_one_ankle_confident_uses_that_ankle():
    kpts = {
        "left_ankle": (110.0, 210.0, 0.9),
        "right_ankle": (130.0, 212.0, 0.2),  # below threshold
    }
    px, source, low_conf = compute_footpoint(kpts, BBOX, ankle_conf_thresh=0.5)
    assert source == "ankle_single"
    assert not low_conf
    assert px == (110.0, 210.0)


def test_footpoint_right_ankle_only():
    kpts = {
        "left_ankle": (110.0, 210.0, 0.1),
        "right_ankle": (130.0, 212.0, 0.8),
    }
    px, source, low_conf = compute_footpoint(kpts, BBOX, ankle_conf_thresh=0.5)
    assert source == "ankle_single"
    assert px == (130.0, 212.0)


def test_footpoint_no_confident_ankles_falls_back_to_bbox_bottom_center():
    kpts = {
        "left_ankle": (110.0, 210.0, 0.1),
        "right_ankle": (130.0, 212.0, 0.05),
    }
    px, source, low_conf = compute_footpoint(kpts, BBOX, ankle_conf_thresh=0.5)
    assert source == "bbox_fallback"
    assert low_conf
    x1, y1, x2, y2 = BBOX
    assert px == ((x1 + x2) / 2, y2)


def test_footpoint_missing_keypoints_dict_falls_back():
    px, source, low_conf = compute_footpoint({}, BBOX, ankle_conf_thresh=0.5)
    assert source == "bbox_fallback"
    assert low_conf


def test_footpoint_threshold_is_configurable():
    kpts = {"left_ankle": (110.0, 210.0, 0.4), "right_ankle": (130.0, 212.0, 0.4)}
    # below default 0.5 -> fallback
    _, source_default, _ = compute_footpoint(kpts, BBOX, ankle_conf_thresh=0.5)
    assert source_default == "bbox_fallback"
    # a looser threshold accepts them
    _, source_loose, _ = compute_footpoint(kpts, BBOX, ankle_conf_thresh=0.3)
    assert source_loose == "ankles"


def test_torso_full_quad_shrinks_toward_centroid():
    kpts = {
        "left_shoulder": (110.0, 70.0, 0.9),
        "right_shoulder": (150.0, 70.0, 0.9),
        "left_hip": (115.0, 140.0, 0.9),
        "right_hip": (145.0, 140.0, 0.9),
    }
    poly, quality = compute_torso_polygon(kpts, BBOX, kp_conf_thresh=0.5, shrink_frac=0.15)
    assert quality == "full"
    assert len(poly) == 4
    cx = sum(p[0] for p in poly) / 4
    cy = sum(p[1] for p in poly) / 4
    raw_cx = (110.0 + 150.0 + 145.0 + 115.0) / 4
    raw_cy = (70.0 + 70.0 + 140.0 + 140.0) / 4
    assert abs(cx - raw_cx) < 1e-6
    assert abs(cy - raw_cy) < 1e-6
    # shrunk points must be strictly inside the original quad's extent
    raw_xs = [110.0, 150.0, 145.0, 115.0]
    for p in poly:
        assert min(raw_xs) < p[0] < max(raw_xs)


def test_torso_partial_when_only_shoulders_confident():
    kpts = {
        "left_shoulder": (110.0, 70.0, 0.9),
        "right_shoulder": (150.0, 70.0, 0.9),
        "left_hip": (115.0, 140.0, 0.1),
        "right_hip": (145.0, 140.0, 0.1),
    }
    poly, quality = compute_torso_polygon(kpts, BBOX, kp_conf_thresh=0.5)
    assert quality == "partial"
    assert len(poly) == 4


def test_torso_fallback_strip_when_no_keypoints():
    poly, quality = compute_torso_polygon({}, BBOX, kp_conf_thresh=0.5, fallback_strip_frac=(0.2, 0.6))
    assert quality == "fallback"
    x1, y1, x2, y2 = BBOX
    h = y2 - y1
    w = x2 - x1
    ys = [p[1] for p in poly]
    assert min(ys) == y1 + 0.2 * h
    assert max(ys) == y1 + 0.6 * h
    xs = [p[0] for p in poly]
    # strip must be narrower than the full box width
    assert (max(xs) - min(xs)) < w
