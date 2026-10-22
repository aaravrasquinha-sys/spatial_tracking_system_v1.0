"""R1 Phase 4: the engine learns people from clean observations and re-identifies them after they leave (mannequins)."""
from types import SimpleNamespace

import numpy as np

from poi_localization.frames.types import FloorFrameTransform
from poi_localization.identity.config import IdentityConfig
from poi_localization.identity.engine import IdentityEngine
from poi_localization.identity.observer import Scene
from poi_localization.identity.room_geometry import RoomGeometry
from poi_perception.contracts import Intrinsics
from tests.synth.mannequin import Body, Room, bbox_of, keypoints, render

INTR = Intrinsics(fx=606.8, fy=606.8, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001)
TF = FloorFrameTransform.from_height_and_yaw(height_m=2.3, pitch_deg=20.0, yaw_deg=0.0, frame_name="room")
ROOM = Room(TF.R, TF.t, INTR)
SCENE = Scene(RoomGeometry(TF.R, TF.t, INTR), ROOM.depth.astype(np.float32), ROOM.rgb)

ALICE = Body(1.75, dev={"thigh": 1.10, "forearm": 0.90}, shirt=(200, 40, 40), trousers=(40, 50, 140))
BOB = Body(1.80, dev={"shank": 0.92, "upper_arm": 1.10}, shirt=(40, 160, 60), trousers=(200, 180, 40))


def _engine():
    eng = IdentityEngine(IdentityConfig(enabled=True, sample_hz_per_track=10.0, max_extractions_per_frame=4))
    eng.set_scene(SCENE)
    return eng


def _run(eng, people, t0, n, rng, dt=0.2, jitter=0.0):
    """people: [(track_id, Body, x, y, heading)]. Feeds n frames; returns the last views and the event list."""
    views = {}
    for i in range(n):
        t = t0 + i * dt
        depth, rgb, Js = render(ROOM, [(b, x, y + jitter * np.sin(i), h) for _, b, x, y, h in people], rng, spacing=0.004)
        assign, outs = {}, []
        for (tid, b, x, y, h), J in zip(people, Js):
            kp = keypoints(ROOM, J, rng)
            det = SimpleNamespace(bbox=bbox_of(ROOM, J, b.stature), det_conf=0.9, keypoints=kp, track_id_2d=tid)
            assign[tid] = (det, SimpleNamespace(height_sample_m=b.stature))
            outs.append(SimpleNamespace(world_track_id=tid, state="confirmed", position_xy=(x, y), velocity_xy=(0.0, 0.0),
                                        height_m=b.stature))
        frame = SimpleNamespace(t=t, rgb=rgb, depth=np.round(depth / 0.001).astype(np.uint16), intr=INTR)
        views = eng.step(frame, t, assign, {}, [], outs)
    return views


def _end(eng, tid, t):
    eng.step(SimpleNamespace(t=t, rgb=ROOM.rgb, depth=np.zeros((480, 640), np.uint16), intr=INTR), t, {}, {},
             [SimpleNamespace(kind="track_ended", world_track_id=tid, detail="")], [])


def test_returning_person_is_matched_and_a_stranger_is_new():
    rng = np.random.default_rng(0)
    eng = _engine()
    # Alice alone: no candidates -> new immediately, then she trains her profile
    v = _run(eng, [(1, ALICE, 3.0, 0.1, np.pi)], 0.0, 14, rng)
    assert v[1].state == "new" and v[1].id == 1
    assert eng.counters["train_updates"] >= 5
    _end(eng, 1, 3.0)
    assert eng.gallery.get(1).state == "missing" and eng.gallery.get(1).structure   # profile synced on exit
    # Alice returns as a new track beside Bob (also new): Alice must match P1, Bob must not
    v = _run(eng, [(2, ALICE, 3.0, -0.9, np.pi), (3, BOB, 3.2, 0.9, np.pi)], 4.0, 30, rng)
    assert v[2].state == "matched" and v[2].id == 1, (v[2], eng.health())
    assert v[3].id == 2 and v[3].state in ("new", "resolving")
    assert eng.counters["commits_match"] == 1


def test_changed_shirt_still_matches_on_body_structure():
    rng = np.random.default_rng(1)
    eng = _engine()
    _run(eng, [(1, ALICE, 3.0, 0.1, np.pi)], 0.0, 14, rng)
    _end(eng, 1, 3.0)
    swapped = Body(ALICE.stature, dev=ALICE.dev, shirt=(30, 30, 160), trousers=ALICE.trousers)   # new upper-body clothes
    v = _run(eng, [(2, swapped, 3.0, 0.0, np.pi)], 4.0, 30, rng)
    assert v[2].state == "matched" and v[2].id == 1, (v[2], eng.health())


def test_stranger_alone_is_not_matched_to_a_missing_person():
    rng = np.random.default_rng(2)
    eng = _engine()
    _run(eng, [(1, ALICE, 3.0, 0.1, np.pi)], 0.0, 14, rng)
    _end(eng, 1, 3.0)
    v = _run(eng, [(2, BOB, 3.0, 0.0, np.pi)], 4.0, 40, rng)
    assert v[2].state in ("new", "ambiguous") and v[2].id != 1
    assert eng.counters["commits_match"] == 0


def test_frozen_engine_neither_commits_nor_learns():
    rng = np.random.default_rng(3)
    eng = _engine()
    eng.freeze("calibration drift")
    v = _run(eng, [(1, ALICE, 3.0, 0.1, np.pi)], 0.0, 8, rng)
    assert v[1].state == "resolving" and eng.counters["train_updates"] == 0 and eng.counters["extractions"] == 0
    eng.unfreeze()
    v = _run(eng, [(1, ALICE, 3.0, 0.1, np.pi)], 2.0, 3, rng)
    assert v[1].state == "new"


def test_no_scene_keeps_phase1_behaviour():
    eng = IdentityEngine(IdentityConfig(enabled=True))
    rng = np.random.default_rng(4)
    v = _run(eng, [(1, ALICE, 3.0, 0.1, np.pi)], 0.0, 2, rng)
    assert v[1].state == "new" and eng.counters["extractions"] == 0


# ---------------------------------------------------------------- loading the M2 anchor-bundle references
def test_load_scene_reads_the_anchor_bundle_formats(tmp_path):
    from poi_localization.identity.observer import load_scene
    geom = SCENE.geom
    ref = ROOM.depth.astype(np.float32)
    rp, sp = tmp_path / "cam0_reference_depth.npz", tmp_path / "static_capture.npz"
    np.savez_compressed(rp, depth_med=ref, stable=np.ones(ref.shape, bool))
    np.savez_compressed(sp, depth_med=ref, depth_mad=np.full(ref.shape, 0.004, np.float32), rgb_med=ROOM.rgb,
                        valid_frac=np.ones(ref.shape, np.float32))
    deg = []
    s = load_scene(geom, str(rp), str(sp), deg)
    assert s is not None and s.ref_rgb is not None and s.ref_mad is not None and s.ref_stable.all() and deg == []
    # colour reference captured without colour (all zeros) -> no photometric correction, still usable
    np.savez_compressed(sp, depth_med=ref, depth_mad=np.zeros(ref.shape, np.float32), rgb_med=np.zeros_like(ROOM.rgb))
    deg = []
    s = load_scene(geom, str(rp), str(sp), deg)
    assert s is not None and s.ref_rgb is None and "no_photometric_reference" in deg
    # wrong resolution / missing file -> no scene, with the reason
    np.savez_compressed(rp, depth_med=ref[::2, ::2], stable=np.ones((240, 320), bool))
    deg = []
    assert load_scene(geom, str(rp), str(sp), deg) is None and deg[0].startswith("reference_depth_shape_mismatch")
    deg = []
    assert load_scene(geom, str(tmp_path / "nope.npz"), "", deg) is None and deg == ["no_reference_depth"]
