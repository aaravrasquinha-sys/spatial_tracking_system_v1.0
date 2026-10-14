"""R1 Phase 2 (part 1): G1 ray table/heights, G2 background mask, G3 photometric gain, G6 noise model."""
import numpy as np

from poi_localization import geometry
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.identity.background import foreground_pixels
from poi_localization.identity.config import BackgroundConfig, PhotometricConfig
from poi_localization.identity.photometric import estimate_gain, estimate_global_gain, normalise
from poi_localization.identity.room_geometry import RoomGeometry, depth_sigma_m, lateral_sigma_m
from poi_perception.contracts import Intrinsics

INTR = Intrinsics(fx=606.8, fy=606.8, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001)
TF = FloorFrameTransform.from_height_and_yaw(height_m=2.3, pitch_deg=20.0, yaw_deg=30.0, frame_name="room")


def test_heights_match_the_reference_transform_within_1mm():
    g = RoomGeometry(TF.R, TF.t, INTR)
    rng = np.random.default_rng(0)
    vs = rng.integers(0, 480, 2000)
    us = rng.integers(0, 640, 2000)
    depth = rng.uniform(1.0, 5.0, (480, 640))
    pts = g.room_points(depth, vs, us)
    ref = np.array([TF.apply_point(geometry.deproject_pixel(u, v, depth[v, u], INTR)) for v, u in zip(vs, us)])
    assert np.abs(pts - ref).max() < 1e-3
    assert np.abs(g.heights(depth, vs, us) - ref[:, 2]).max() < 1e-3


def test_project_is_the_inverse_of_room_points():
    g = RoomGeometry(TF.R, TF.t, INTR)
    depth = np.full((480, 640), 3.0)
    vs, us = np.array([100, 240, 400]), np.array([50, 320, 600])
    u, v, z = g.project(g.room_points(depth, vs, us))
    assert np.allclose(u, us, atol=1e-2) and np.allclose(v, vs, atol=1e-2) and np.allclose(z, 3.0, atol=1e-4)
    assert g.in_image(u, v, z).all()
    assert not g.in_image(np.array([700.0]), np.array([10.0]), np.array([2.0])).any()


def test_distortion_coeffs_change_the_table_but_zero_coeffs_do_not():
    a = RoomGeometry(TF.R, TF.t, INTR).A
    z = RoomGeometry(TF.R, TF.t, INTR, distortion_coeffs=(0, 0, 0, 0, 0)).A
    assert np.array_equal(a, z)
    d = RoomGeometry(TF.R, TF.t, INTR, distortion_coeffs=(0.1, -0.05, 0.0, 0.0, 0.0)).A
    assert np.abs(d - a).max() > 1e-3


def test_noise_model_grows_with_range():
    assert depth_sigma_m(1.0) < depth_sigma_m(3.0) < depth_sigma_m(5.0)
    assert 0.001 < depth_sigma_m(3.0) < 0.08
    assert abs(lateral_sigma_m(3.0, 3.0, 606.8) - 3.0 * 3.0 / 606.8) < 1e-12


# ---------------------------------------------------------------- G2
def _scene():
    ref = np.full((480, 640), 4.0)
    ref[:, :20] = np.nan                                  # unknown reference region
    live = ref.copy()
    live[100:400, 200:300] = 2.5                          # person A (nearer than the wall)
    return ref, live


def test_foreground_covers_the_person_and_excludes_background():
    ref, live = _scene()
    r = foreground_pixels(live, ref, (200, 100, 300, 400), 2.5, stride=2)
    assert r.flags == []
    inside = ((r.us >= 200) & (r.us < 300) & (r.vs >= 100) & (r.vs < 400)).mean()
    assert inside == 1.0                                    # zero background leaks in
    full = (300 // 2) * (100 // 2)                         # 300 x 100 px block sampled at stride 2 = 7500
    assert 0.9 * full <= r.n <= full                       # only a thin erosion ring is dropped


def test_person_far_behind_band_is_not_foreground_and_no_reference_is_flagged():
    ref, live = _scene()
    live[100:400, 200:300] = 3.4                          # 0.9 m from the torso depth below -> rule 3 rejects
    assert foreground_pixels(live, ref, (200, 100, 300, 400), 2.5).n == 0
    r = foreground_pixels(live, None, (200, 100, 300, 400), 3.4)
    assert "no_reference_background" in r.flags and r.n > 0


def test_pixels_claimed_by_a_nearer_person_are_excluded():
    ref, live = _scene()
    live[150:350, 250:350] = 1.8                          # person B, nearer, overlapping A's right half
    r = foreground_pixels(live, ref, (200, 100, 300, 400), 2.5, others=[((250, 150, 350, 350), 1.8)])
    in_b = (r.us >= 250) & (r.us <= 350) & (r.vs >= 150) & (r.vs <= 350)
    assert not in_b.any() and r.n > 0


def test_tiny_bbox_returns_empty():
    ref, live = _scene()
    assert foreground_pixels(live, ref, (10, 10, 12, 12), 2.5).n == 0


# ---------------------------------------------------------------- G3
def _room_images(seed=1):
    rng = np.random.default_rng(seed)
    ref_rgb = rng.integers(60, 200, (480, 640, 3)).astype(np.uint8)
    depth = np.full((480, 640), 4.0)
    return ref_rgb, depth


def test_gain_recovered_from_the_ring_within_3_percent():
    ref_rgb, depth = _room_images()
    gain = np.array([1.25, 1.0, 0.8])
    live = np.clip(ref_rgb * gain, 0, 255).astype(np.uint8)
    live_depth = depth.copy()
    live_depth[100:400, 250:350] = 2.5                    # the person occludes the middle
    stable = np.ones((480, 640), bool)
    r = estimate_gain(live, ref_rgb, live_depth, depth, stable, (250, 100, 350, 400))
    assert r.mode == "ring" and r.n_px >= 300
    assert np.all(np.abs(r.gain / gain - 1) < 0.03)
    assert r.quality > 0.1
    # normalising live colours brings them back to the reference
    px = live[50:60, 50:60].reshape(-1, 3)
    back = normalise(px, r.gain)
    assert np.abs(back - ref_rgb[50:60, 50:60].reshape(-1, 3)).mean() < 5


def test_gain_falls_back_to_global_then_none():
    ref_rgb, depth = _room_images()
    live = np.clip(ref_rgb * np.array([1.2, 1.2, 1.2]), 0, 255).astype(np.uint8)
    stable = np.zeros((480, 640), bool)                   # nothing stable near the person -> ring fails
    cfg = PhotometricConfig()
    gg = np.array([1.2, 1.2, 1.2])
    r = estimate_gain(live, ref_rgb, depth, depth, stable, (250, 100, 350, 400), cfg, global_gain=gg)
    assert r.mode == "global" and np.allclose(r.gain, gg)
    r = estimate_gain(live, ref_rgb, depth, depth, stable, (250, 100, 350, 400), cfg)
    assert r.mode == "none" and np.allclose(r.gain, 1.0) and r.quality == 0.0
    off = estimate_gain(live, ref_rgb, depth, depth, np.ones((480, 640), bool), (250, 100, 350, 400),
                        PhotometricConfig(enabled=False))
    assert off.mode == "none"


def test_global_gain_and_clamp():
    ref_rgb, depth = _room_images()
    live = np.clip(ref_rgb.astype(float) * 3.0, 0, 255).astype(np.uint8)
    stable = np.ones((480, 640), bool)
    g = estimate_global_gain(live, ref_rgb, depth, depth, stable, PhotometricConfig())
    assert g is not None and np.all(g <= 2.0 + 1e-9)       # clamped to gain_clamp
    assert estimate_global_gain(live, ref_rgb, depth, depth, np.zeros((480, 640), bool), PhotometricConfig()) is None
