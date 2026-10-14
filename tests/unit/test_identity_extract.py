"""R1 Phase 2/3: G4 visibility, G5 viewpoint, clean gate, joint lifting, structure + appearance extraction on mannequins,
and the profile accumulators."""
import numpy as np
import pytest

from poi_localization.config import M4Config
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.identity.background import foreground_pixels
from poi_localization.identity.clean_gate import GateInput, bbox_iou, check_clean, isolation
from poi_localization.identity.config import HistConfig, IdentityConfig, PhotometricConfig
from poi_localization.identity.extract_appearance import (LOWER_BANDS, UPPER_BANDS, bhattacharyya, extract_appearance,
                                                          histogram52, rgb_to_hsv)
from poi_localization.identity.extract_structure import DEFAULT_BETA, extract_structure, to_features
from poi_localization.identity.joints3d import lift_measured_joints
from poi_localization.identity.photometric import estimate_gain, normalise
from poi_localization.identity.profile import AppearanceAccumulator, RobustStat, StructureModel
from poi_localization.identity.room_geometry import RoomGeometry
from poi_localization.identity.visibility import (DYNAMIC_OCCLUDED, OUT_OF_VIEW, STATIC_OCCLUDED, VISIBLE,
                                                  band_visibility, face_visible, viewpoint)
from poi_perception.contracts import Intrinsics
from tests.synth.mannequin import Body, Room, bbox_of, keypoints, render

INTR = Intrinsics(fx=606.8, fy=606.8, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001)
TF = FloorFrameTransform.from_height_and_yaw(height_m=2.3, pitch_deg=20.0, yaw_deg=0.0, frame_name="room")
EDGES = IdentityConfig().bands
ROOM = Room(TF.R, TF.t, INTR)
GEOM = RoomGeometry(TF.R, TF.t, INTR)
REF = ROOM.depth.astype(np.float32)


def _observe(body, x, y, heading, seed, kp_sigma=1.0, gain=None):
    rng = np.random.default_rng(seed)
    depth, rgb, Js = render(ROOM, [(body, x, y, heading)], rng, gain=gain)
    J = Js[0]
    kp = keypoints(ROOM, J, rng, kp_sigma)
    u, v = [int(round(q)) for q in (np.array(kp["left_shoulder"][:2]) + np.array(kp["right_hip"][:2])) / 2]
    d_t = float(np.median(depth[v - 3:v + 4, u - 3:u + 4]))
    bb = bbox_of(ROOM, J, body.stature)
    fg = foreground_pixels(depth, REF, bb, d_t)
    joints = lift_measured_joints(GEOM, depth, kp, d_t, kp_pixel_sigma_px=max(kp_sigma, 1.0), torso_xy=(x, y))
    return depth, rgb, J, kp, d_t, bb, fg, joints


# ---------------------------------------------------------------- foreground on a rendered person
def _dilate3(m):
    o = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o


def test_foreground_mask_covers_the_mannequin_with_no_background_leak():
    body = Body(1.75)
    depth, rgb, J, kp, d_t, bb, fg, joints = _observe(body, 3.0, 0.2, np.pi, 0)
    person = (depth > 0) & (np.abs(depth - np.nan_to_num(REF)) > 0.03)
    leak = 1.0 - person[fg.vs, fg.us].mean()
    fgm = np.zeros(person.shape, bool)
    fgm[fg.vs, fg.us] = True
    vs, us = np.nonzero(person)
    covered = _dilate3(fgm)[vs, us]
    heights = GEOM.heights(np.where(depth > 0, depth, np.nan), vs, us)
    band = np.clip(np.digitize(heights / 1.75, EDGES) - 1, 0, 7)
    assert leak < 0.02, f"background leaked into the mask: {leak:.3f}"
    # Measured on the mannequin: ~0.86 overall, ~0.88 per upper/body band, ~0.33 at the shoes (feet touch the floor,
    # so they are not >= margin in front of it). The erosion ring on thin limbs is the cost of silhouette safety;
    # the spec's 0.95 target is re-measured on real data in P0.
    assert covered.mean() > 0.80
    for b in range(2, 8):
        assert covered[band == b].mean() > 0.80, b
    assert covered[band == 0].mean() < 0.6                 # shoe band: documented limitation


# ---------------------------------------------------------------- joints
def test_ankles_are_measured_with_the_pitch_invariant_gate_and_foreign_joints_rejected():
    body = Body(1.75)
    depth, rgb, J, kp, d_t, bb, fg, joints = _observe(body, 3.0, 0.3, np.pi, 1)
    assert "left_ankle" in joints and "right_ankle" in joints and len(joints) == 17
    far = lift_measured_joints(GEOM, depth, kp, d_t, torso_xy=(3.0 + 1.5, 0.3))      # track somewhere else
    assert far == {}
    low = dict(kp)
    low["left_wrist"] = (kp["left_wrist"][0], kp["left_wrist"][1], 0.2)               # low confidence
    assert "left_wrist" not in lift_measured_joints(GEOM, depth, low, d_t, torso_xy=(3.0, 0.3))


# ---------------------------------------------------------------- structure
def _mean_structure(body, x, y, heading, view, n=6):
    samples = []
    for seed in range(n):
        depth, rgb, J, kp, d_t, bb, fg, joints = _observe(body, x, y, heading, seed)
        samples.append(extract_structure(GEOM, depth, joints, fg, view=view))
    return samples


def test_front_view_structure_is_accurate_on_the_mannequin():
    body = Body(1.75)
    samples = _mean_structure(body, 3.0, 0.3, np.pi, "front")
    truth = {"stature": body.stature, **{k: body.seg(k) for k in DEFAULT_BETA}}
    got = {}
    for s in samples:
        if s.stature:
            got.setdefault("stature", []).append(s.stature[0])
        for k, (L, _) in s.segments.items():
            got.setdefault(k, []).append(L)
    assert {"stature", "upper_arm", "forearm", "thigh", "torso", "hip_width", "shoulder_width"} <= set(got)
    # measured bias on the noise-free-pose mannequin: stature ~ -1.0 cm (98th percentile of a noisy top cap), segments
    # within ~2 cm (joint depth is read on the clothed SURFACE, not the joint centre): the P0 priors absorb these.
    assert abs(np.mean(got["stature"]) - truth["stature"]) < 0.015
    for k in ("upper_arm", "forearm", "thigh", "torso", "hip_width", "shoulder_width"):
        assert abs(np.mean(got[k]) - truth[k]) < 0.025, k


def test_side_view_keeps_only_trustworthy_features():
    s = _mean_structure(Body(1.75), 3.0, 0.0, np.pi / 2, "side", n=3)[0]
    assert "view_side_near_side_only" in s.flags
    assert not ({"hip_width", "shoulder_width", "torso", "thigh"} & set(s.segments))
    assert s.stature is not None


def test_different_bodies_have_different_residual_features():
    a, b = Body(1.75), Body(1.75, dev={"thigh": 1.12, "forearm": 0.88})
    fa = to_features(_mean_structure(a, 3.0, 0.3, np.pi, "front", n=1)[0])
    fb = to_features(_mean_structure(b, 3.0, 0.3, np.pi, "front", n=1)[0])
    assert fb["thigh"][0] - fa["thigh"][0] > 0.015          # +12% of 0.43 m = +5 cm, minus noise/bias
    assert fb["forearm"][0] - fa["forearm"][0] < -0.01


def test_stature_cross_check_discards_a_disagreeing_estimate():
    depth, rgb, J, kp, d_t, bb, fg, joints = _observe(Body(1.75), 3.0, 0.3, np.pi, 2)
    s = extract_structure(GEOM, depth, joints, fg, kp_height_estimate=1.55, view="front")
    assert s.stature is None and "stature_disagrees_with_keypoint_estimate" in s.flags
    assert extract_structure(GEOM, depth, joints, fg, kp_height_estimate=1.76, view="front").stature is not None


def test_to_features_decorrelates_against_a_running_stature():
    s = _mean_structure(Body(1.75), 3.0, 0.3, np.pi, "front", n=1)[0]
    f_own = to_features(s)
    f_ref = to_features(s, H=1.75, sigma_H=0.005)
    assert set(f_own) == set(f_ref) and f_own["stature"] == f_ref["stature"]
    k = "thigh"
    assert abs(f_ref[k][0] - (s.segments[k][0] - DEFAULT_BETA[k] * 1.75)) < 1e-12
    assert f_ref[k][1] >= s.segments[k][1]                   # H uncertainty only adds variance


# ---------------------------------------------------------------- appearance
def _appearance(body, x=3.0, y=0.2, heading=np.pi, seed=0, gain=None, normalise_colour=True):
    depth, rgb, J, kp, d_t, bb, fg, joints = _observe(body, x, y, heading, seed, gain=gain)
    gres = estimate_gain(rgb, ROOM.rgb, depth.astype(np.float64), REF.astype(np.float64), np.ones(REF.shape, bool), bb)
    cols = rgb[fg.vs, fg.us]
    cols = normalise(cols, gres.gain) if normalise_colour else cols.astype(np.float32)
    H = body.stature
    heights = GEOM.heights(depth, fg.vs, fg.us)
    states = band_visibility(GEOM, REF, x, y, H, EDGES)
    return extract_appearance(cols, heights, H, EDGES, states, HistConfig(), view="front", photo_quality=gres.quality), gres


def _upper_lower_score(a, b):
    def region(bands):
        shared = [i for i in bands if i in a.hists and i in b.hists]
        w = np.array([min(a.mass[i], b.mass[i]) for i in shared], float)
        return float(np.average([bhattacharyya(a.hists[i], b.hists[i]) for i in shared], weights=w)) if shared else None
    return region(UPPER_BANDS), region(LOWER_BANDS)


def test_hsv_and_histogram_basics():
    hsv = rgb_to_hsv(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0.5], [0, 0, 0]]))
    assert np.allclose(hsv[:3, 0], [0, 1 / 3, 2 / 3]) and np.allclose(hsv[:3, 1:], 1.0)
    assert hsv[3, 1] == 0 and hsv[4, 2] == 0
    h = histogram52(np.array([[255, 0, 0]] * 50 + [[10, 10, 10]] * 50), HistConfig())
    assert h.shape == (52,) and abs(h.sum() - 1) < 1e-6
    assert abs(h[48:].sum() - 0.5) < 1e-6                    # the dark pixels land in the achromatic (value) bins
    assert histogram52(np.zeros((0, 3)), HistConfig()).sum() == 0
    assert bhattacharyya(h, h) > 0.999


def test_same_clothes_match_and_different_clothes_do_not():
    red = Body(1.75, shirt=(200, 40, 40), trousers=(40, 50, 140))
    blue = Body(1.75, shirt=(40, 60, 200), trousers=(200, 180, 40))
    a1, _ = _appearance(red, seed=0)
    a2, _ = _appearance(red, x=3.4, y=-0.2, seed=1)
    b, _ = _appearance(blue, seed=2)
    assert a1.hists and set(a1.hists) <= set(range(8))
    same_u, same_l = _upper_lower_score(a1, a2)
    diff_u, diff_l = _upper_lower_score(a1, b)
    assert same_u > 0.9 and same_l > 0.9
    assert diff_u < 0.6 and diff_l < 0.6


def test_photometric_normalisation_restores_the_histogram_under_an_exposure_change():
    body = Body(1.75, shirt=(180, 90, 60), trousers=(60, 90, 160))
    base, _ = _appearance(body, seed=0)
    gain = (1.35, 1.0, 0.75)
    raw, _ = _appearance(body, seed=0, gain=gain, normalise_colour=False)
    fixed, g = _appearance(body, seed=0, gain=gain, normalise_colour=True)
    assert g.mode == "ring" and np.all(np.abs(g.gain / np.array(gain) - 1) < 0.05)
    s_raw = np.mean([x for x in _upper_lower_score(base, raw) if x is not None])
    s_fix = np.mean([x for x in _upper_lower_score(base, fixed) if x is not None])
    assert s_fix > 0.9 and s_fix > s_raw + 0.05
    assert fixed.quality < base.quality + 1e-9 or fixed.quality < 1.0   # a large correction lowers the observation weight


def test_band_states_gate_what_is_accumulated():
    body = Body(1.75)
    depth, rgb, J, kp, d_t, bb, fg, joints = _observe(body, 3.0, 0.2, np.pi, 0)
    heights = GEOM.heights(depth, fg.vs, fg.us)
    states = [VISIBLE] * 8
    states[0] = STATIC_OCCLUDED
    states[1] = DYNAMIC_OCCLUDED
    a = extract_appearance(rgb[fg.vs, fg.us].astype(np.float32), heights, 1.75, EDGES, states, HistConfig())
    assert 0 not in a.hists and 1 not in a.hists and 7 in a.hists
    assert all(m >= 40 for m in a.mass.values())


# ---------------------------------------------------------------- G4 visibility, G5 viewpoint
def test_band_visibility_static_dynamic_and_out_of_view():
    base = band_visibility(GEOM, REF, 3.0, 0.2, 1.75, EDGES)
    assert base == [VISIBLE] * 8
    # a desk/pillar in the empty-room reference in front of the legs: reference depth much nearer than the body
    ref = REF.copy()
    ref[330:480, :] = 2.0
    st = band_visibility(GEOM, ref, 3.0, 0.2, 1.75, EDGES)
    assert st[0] == STATIC_OCCLUDED and st[7] == VISIBLE
    # someone nearer covers the upper body
    others = [((250.0, 100.0, 400.0, 300.0), 1.8)]
    dyn = band_visibility(GEOM, REF, 3.0, 0.2, 1.75, EDGES, others=others)
    assert dyn[7] == DYNAMIC_OCCLUDED and dyn[0] == VISIBLE
    # too close: the feet fall below the image / too far
    assert band_visibility(GEOM, REF, 1.6, 0.0, 1.75, EDGES)[0] == OUT_OF_VIEW
    assert band_visibility(GEOM, REF, 3.0, 0.2, 1.75, EDGES, max_range_m=2.0) == [OUT_OF_VIEW] * 8


@pytest.mark.parametrize("heading,face,vel,expected", [
    (np.pi, True, None, "front"),          # facing the camera
    (0.0, False, (1.0, 0.0), "back"),      # walking away from the camera
    (np.pi / 2, True, None, "side"),
    (0.0, False, None, None),              # unknown: no face, not moving
    (np.pi, False, (0.1, 0.0), None),      # too slow to infer the heading
])
def test_viewpoint(heading, face, vel, expected):
    from tests.synth.mannequin import pose
    J, _ = pose(Body(1.75), 3.0, 0.0, heading)
    v = viewpoint(J["left_shoulder"], J["right_shoulder"], TF.t[:2], np.array([3.0, 0.0]), face,
                  None if vel is None else np.array(vel))
    assert v == expected


def test_face_visible_needs_nose_and_an_eye():
    kp = {"nose": (0, 0, 0.9), "left_eye": (0, 0, 0.9), "right_eye": (0, 0, 0.1)}
    assert face_visible(kp)
    assert not face_visible({"nose": (0, 0, 0.9), "left_eye": (0, 0, 0.1), "right_eye": (0, 0, 0.1)})
    assert not face_visible({"left_eye": (0, 0, 0.9), "right_eye": (0, 0, 0.9)})


# ---------------------------------------------------------------- clean gate (truth table)
def _gi(**kw):
    base = dict(track_state="confirmed", measured_this_frame=True, identity_resolving=False, s_since_2d_mismatch=None,
                swap_flag=False, max_iou_with_others=0.0, min_dist_to_others_m=float("inf"), torso_depth_m=3.0,
                det_conf=0.9, n_kp_conf=15, bbox=(100.0, 50.0, 200.0, 400.0), img_w=640, img_h=480, upright=True,
                calib_ok=True)
    base.update(kw)
    return GateInput(**base)


CFG = IdentityConfig()


def test_clean_gate_truth_table():
    assert check_clean(_gi(), CFG).ok_for_training
    cases = {
        1: dict(track_state="coasting"), 2: dict(s_since_2d_mismatch=1.0), 3: dict(max_iou_with_others=0.2),
        4: dict(torso_depth_m=5.0), 5: dict(det_conf=0.3), 6: dict(bbox=(2.0, 50.0, 200.0, 400.0)),
        7: dict(upright=False), 8: dict(calib_ok=False),
    }
    for rule, kw in cases.items():
        r = check_clean(_gi(**kw), CFG)
        assert not r.ok_for_training and rule in r.failed, rule
    assert not check_clean(_gi(measured_this_frame=False), CFG).ok_for_training
    assert not check_clean(_gi(swap_flag=True), CFG).ok_for_training
    assert not check_clean(_gi(min_dist_to_others_m=0.4), CFG).ok_for_training
    assert not check_clean(_gi(n_kp_conf=9), CFG).ok_for_training
    assert not check_clean(_gi(torso_depth_m=0.8), CFG).ok_for_training


def test_resolving_track_may_give_evidence_but_never_trains():
    r = check_clean(_gi(identity_resolving=True), CFG)
    assert not r.ok_for_training and r.ok_for_evidence
    # a partially occluded newcomer near the door: fails isolation / border / upright, still usable as evidence
    r = check_clean(_gi(max_iou_with_others=0.3, bbox=(0.0, 50.0, 120.0, 400.0), upright=False), CFG)
    assert not r.ok_for_training and r.ok_for_evidence
    # but not out of range, low confidence, or while the calibration is suspect
    assert not check_clean(_gi(torso_depth_m=6.0), CFG).ok_for_evidence
    assert not check_clean(_gi(det_conf=0.2), CFG).ok_for_evidence
    assert not check_clean(_gi(calib_ok=False), CFG).ok_for_evidence
    assert not check_clean(_gi(track_state="coasting"), CFG).ok_for_evidence


def test_isolation_helpers():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert isolation((0, 0, 10, 10), (0.0, 0.0), []) == (0.0, float("inf"))
    iou, d = isolation((0, 0, 10, 10), (0.0, 0.0), [((5, 5, 15, 15), (0.3, 0.4)), ((50, 50, 60, 60), (3.0, 4.0))])
    assert 0 < iou < 1 and abs(d - 0.5) < 1e-12


# ---------------------------------------------------------------- profile accumulators
def test_robust_stat_mean_clip_and_throttle():
    st = RobustStat()
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(40):
        assert st.add(1.75 + rng.normal(0, 0.01), 0.01, t)
        t += 0.4
    assert abs(st.mean - 1.75) < 0.01 and st.n == 40
    assert abs(st.spread - 0.01) < 0.004
    before = st.mean
    assert st.add(2.60, 0.01, t)                       # an outlier is clipped to mean + 3 sigma, not absorbed
    assert abs(st.mean - before) < 0.003
    assert not st.add(1.75, 0.01, t + 0.1)             # throttled: faster than 3 samples / s
    assert st.sigma_mean(0.02) >= st.sigma_sensor_mean
    again = RobustStat.from_dict(st.to_dict())
    assert again.mean == st.mean and again.n == st.n


def test_inverse_variance_weighting_prefers_precise_samples():
    st = RobustStat()
    st.add(1.70, 0.05, 0.0)
    st.add(1.80, 0.005, 1.0)
    assert abs(st.mean - 1.80) < 0.002


def test_structure_model_roundtrip_and_throttle():
    m = StructureModel(max_rate_hz=2.0)
    assert m.add({"stature": (1.75, 0.01), "thigh": (0.0, 0.02)}, 0.0) == 2
    assert m.add({"stature": (1.76, 0.01)}, 0.1) == 0
    assert m.add({"stature": (1.76, 0.01)}, 0.6) == 1
    m2 = StructureModel.from_dict(m.to_dict(), 2.0)
    assert m2.mean("stature") == m.mean("stature") and m2.mean("nope") is None


def test_appearance_accumulator_ema_views_and_roundtrip():
    acc = AppearanceAccumulator(halflife_s=10.0)
    h1, h2 = np.zeros(52, np.float32), np.zeros(52, np.float32)
    h1[0], h2[1] = 1.0, 1.0
    acc.add({4: h1}, {4: 100}, "front", 0.0)
    acc.add({4: h2}, {4: 100}, "front", 10.0)          # one half-life later: weights 50/50
    h = acc.get("front", 4)
    assert abs(h[0] - 0.5) < 1e-6 and abs(h[1] - 0.5) < 1e-6 and acc.mass[("front", 4)] == 200
    assert acc.get("side", 4) is None and acc.get("any", 4) is not None
    acc.add({3: h1}, {3: 50}, None, 11.0)               # unknown view -> only the "any" aggregate
    assert acc.get("front", 3) is None and acc.get("any", 3) is not None
    r = AppearanceAccumulator.from_dict(acc.to_dict())
    assert np.allclose(r.get("front", 4), acc.get("front", 4)) and r.mass == acc.mass
