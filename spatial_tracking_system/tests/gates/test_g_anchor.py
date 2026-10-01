"""
G-ANCHOR gate suite: oracle tests for Module 2 (static-camera anchor). Same discipline as the rest of the
project's gates -- build a case with a KNOWN answer, assert the code recovers it, and for every safety
property (ambiguity, degeneracy, map holes, bad IMU, moved camera) assert the code REFUSES rather than
guesses. Hardware-free by construction (numpy/scipy/opencv only); the synthetic room is analytic
(tests/synth/anchor_world.py) and the map is deliberately expressed in an awkward W_cam0-like frame.

Run:  python3 -m tests.gates.test_g_anchor
"""
from __future__ import annotations
import functools
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

from pyslam.core import lie
from pyslam.anchor.config import AnchorConfig
from tests.synth.anchor_world import (World, Box, default_world, awkward_map_frame, cam_pose_room, render_capture, INTR)

CFG = AnchorConfig()


# ------------------------------------------------------------------ fixtures
@functools.lru_cache(maxsize=None)
def _default_map():
    w = default_world()
    P, C = w.sample_surface()
    Tmr = awkward_map_frame()
    return w, P, C, Tmr, lie.transform_points(Tmr, P)


def _mb(pts_map, colors, map_id="synthetic0001", sha=None):
    return {"pts_map": pts_map, "colors": colors, "map_id": map_id, "capture_profile_sha": sha}


def _err(res, Tmr, T_true):
    T_true_d = res.room_frame.T_room_map @ Tmr @ T_true
    d = lie.se3_inverse(T_true_d) @ res.T_room_cam
    return float(np.linalg.norm(d[:3, 3])), float(np.degrees(np.linalg.norm(lie.so3_log(d[:3, :3]))))


def _failed(res):
    return {c.name for c in res.checks if c.required and not c.passed}


T_TRUE = cam_pose_room(1.0, 3.4, 2.3, -25, 28)


# ------------------------------------------------------------------ Stage 0
def check_ply_round_trip_and_map_id():
    from pyslam.mapping.export import write_ply
    from pyslam.anchor.map_io import read_ply, load_map_bundle, resolve_map_id
    from pyslam.mapping.lock import compute_map_id
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)).astype(np.float32)
    col = rng.integers(0, 255, (500, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "dense"))
        write_ply(os.path.join(d, "dense", "points.ply"), pts, col)
        p2, c2 = read_ply(os.path.join(d, "dense", "points.ply"))
        assert np.allclose(p2, pts, atol=1e-6) and np.array_equal(c2, col), "PLY round-trip mismatch"
        open(os.path.join(d, "capture_profile.json"), "w").write("{}")
        r = resolve_map_id(d)
        assert r["map_id"] == compute_map_id(d, ["dense/points.ply", "capture_profile.json"]) and r["source"].startswith("computed")
        json.dump({"map_id": "abc123"}, open(os.path.join(d, "manifest.json"), "w"))
        r2 = resolve_map_id(d)
        assert r2["map_id"] == "abc123" and r2["source"] == "manifest.json", r2
    print("  PLY round trip exact; map_id = M1's compute_map_id, manifest wins when present -- OK")


def check_room_frame_recovers_floor_up_and_walls():
    from pyslam.anchor.room_frame import derive_room_frame
    w, P, C, Tmr, pts_map = _default_map()
    rf = derive_room_frame(pts_map, CFG)
    T = rf.T_room_map
    pts_room = lie.transform_points(T, pts_map)
    floor = pts_room[np.abs(pts_room[:, 2]) < 0.03]
    assert floor.shape[0] > 40000 and abs(np.median(pts_room[np.isin(np.arange(len(P)), np.nonzero(P[:, 2] < 1e-9)[0]), 2])) < 0.005, "floor not at z=0"
    # true room z axis expressed in derived room frame must be +Z within 0.1 deg
    R_true_in_derived = T[:3, :3] @ Tmr[:3, :3]
    up_err = np.degrees(np.arccos(np.clip(R_true_in_derived[2, 2], -1, 1)))
    assert up_err < 0.15, f"room Z off gravity-of-floor by {up_err:.3f} deg"
    yaw = np.degrees(np.arctan2(R_true_in_derived[1, 0], R_true_in_derived[0, 0]))
    assert min(abs(((yaw + 45) % 90) - 45), 1e9) < 0.3, f"room X not wall-aligned: residual yaw {yaw:.2f} deg"
    assert rf.yaw_source == "wall_histogram"
    normals = np.array([wl["normal_in_room"] for wl in rf.walls[:4]])
    assert np.all(np.minimum(np.abs(normals[:, 0]), np.abs(normals[:, 1])) < 0.01), "walls not axis aligned in room frame"
    assert not rf.warnings, rf.warnings
    print(f"  floor at z=0 (rms {rf.floor['rms_m']*1000:.1f} mm), up err {up_err:.3f} deg, walls axis-aligned, "
          f"yaw source={rf.yaw_source}; a tabletop (0.75 m) was NOT mistaken for the floor -- OK")


def check_room_frame_honest_about_a_bowed_floor():
    from pyslam.anchor.room_frame import derive_room_frame
    w, P, C, Tmr, _ = _default_map()
    Pb = P.copy()
    low = Pb[:, 2] < 0.3
    Pb[low, 2] += 0.10 * ((Pb[low, 0] - 2.6) / 2.6) ** 2                 # 10 cm bowl-shaped drift of the floor
    Pb[low, 2] += 0.03 * (Pb[low, 1] / 4.1)                              # + a small tilt component
    rf = derive_room_frame(lie.transform_points(Tmr, Pb), CFG)
    assert rf.floor["max_quadrant_tilt_deg"] > 0.3 or rf.floor["rms_m"] > 0.005, "bowed floor not reflected in floor stats"
    assert rf.map_rot_sigma_deg > CFG.map_rot_floor_deg or rf.map_trans_sigma_m > CFG.map_trans_floor_m, \
        "map accuracy floor did not grow for a distorted map"
    print(f"  bowed floor: quad tilt {rf.floor['max_quadrant_tilt_deg']:.2f} deg, rms {rf.floor['rms_m']*1000:.1f} mm "
          f"=> map accuracy floor {rf.map_trans_sigma_m*1000:.1f} mm / {rf.map_rot_sigma_deg:.2f} deg -- OK")


def check_walkable_grid_schema_and_semantics():
    from pyslam.anchor.room_frame import derive_room_frame
    from pyslam.anchor.walkable import build_walkable, walkable_to_json
    w, P, C, Tmr, pts_map = _default_map()
    rf = derive_room_frame(pts_map, CFG)
    pr = lie.transform_points(rf.T_room_map, pts_map)
    walk = build_walkable(pr, CFG)
    j = json.loads(json.dumps(walkable_to_json(walk)))
    ox, oy = j["origin"]; res = j["resolution"]; grid = np.array(j["grid"], bool)
    T_true_room = lie.se3_inverse(rf.T_room_map @ Tmr)                    # derived-room -> true-room  (true = T @ derived)

    def is_walkable(x_true, y_true, z=0.0):                               # query in TRUE room coords, like M4 would in derived
        p = lie.transform_points(rf.T_room_map @ Tmr, np.array([[x_true, y_true, z]]))[0]
        col, row = int((p[0] - ox) / res), int((p[1] - oy) / res)
        return bool(0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and grid[row, col])
    assert is_walkable(2.5, 2.0), "open floor must be walkable"
    assert is_walkable(1.2, 0.7), "cell on top of the sofa (sittable) must be walkable"
    assert not is_walkable(5.05, 1.2), "cell inside the tall shelf must not be walkable"
    assert not is_walkable(-0.6, 2.0), "outside the room (behind the wall: where mirror reflections land) must not be walkable"
    assert not is_walkable(2.5, 4.6), "outside the room (other side) must not be walkable"
    assert abs(j["stats"]["walkable_area_m2"] - (5.2 * 4.1 - 1.3 * 0.3 * 4)) < 5.0
    print(f"  walkable.json in STS schema (row=y, col=x); floor/sofa True, shelf/outside-the-walls False; "
          f"{j['stats']['walkable_area_m2']:.1f} m^2 -- OK")


# ------------------------------------------------------------------ Stage 1/2
def check_capture_median_imu_sign_and_edge_filter():
    from pyslam.anchor.query_cloud import capture_to_cloud, stable_mask
    w = default_world()
    cap = render_capture(w, T_TRUE, n_frames=40, seed=1)
    up_true = T_TRUE[:3, :3].T @ np.array([0, 0, 1.0])
    ang = np.degrees(np.arccos(np.clip(cap.up_cam @ up_true, -1, 1)))
    assert cap.imu_info["ok"] and ang < 0.3, f"IMU up-vector off by {ang:.3f} deg (sign/frame convention?)"
    # temporal median beats single-frame noise: compare to ray-cast truth
    H, W = INTR["height"], INTR["width"]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(u - INTR["cx"]) / INTR["fx"], (v - INTR["cy"]) / INTR["fy"], np.ones_like(u, float)], -1).reshape(-1, 3)
    zt = w.raycast(T_TRUE[:3, 3], dirs @ T_TRUE[:3, :3].T).reshape(H, W)
    m = np.isfinite(cap.depth_med) & (zt < 4)
    e = np.abs(cap.depth_med[m] - zt[m])
    assert np.median(e) < 0.003, f"median-of-frames depth error {np.median(e)*1000:.1f} mm too large"
    # edges: flying pixels must not survive
    # flying pixels: a 1-px column of intermediate depth between a near object (1 m) and the far wall (3 m),
    # plus an isolated speck and a hole -- none of them (nor their neighbours) may reach the calibration cloud
    d = np.full((60, 80), 3.0, np.float32); d[:, :40] = 1.0; d[:, 40] = 2.0
    d[10, 10] = 0.6; d[30, 60] = np.nan
    ones = np.ones_like(d)
    st_d = stable_mask(d, np.zeros_like(d), ones, CFG)
    assert not st_d[:, 39:42].any() and not st_d[10, 10] and not st_d[29:32, 59:62].any(), "flying/edge/hole-adjacent pixels leaked"
    assert st_d[14:, :38].all() and st_d[:, 43:].mean() > 0.97, "edge filter ate real surface"
    cl = capture_to_cloud(render_capture(w, T_TRUE, n_frames=15, seed=2), CFG)
    assert cl["pts"].shape[0] > 30000 and np.all(cl["pts"][:, 2] < CFG.range_max_m + 1e-6)
    print(f"  IMU up-vector within {ang:.2f} deg of truth (sign convention OK), median depth err {np.median(e)*1000:.1f} mm, "
          f"flying pixels rejected -- OK")


def check_physics_prior_height_tilt_and_failure_modes():
    from pyslam.anchor.query_cloud import capture_to_cloud
    from pyslam.anchor.physics_prior import physics_prior, PhysicsPriorError, level_rotation
    w = default_world()
    cap = render_capture(w, T_TRUE, n_frames=20, seed=3)
    cl = capture_to_cloud(cap, CFG)
    pp = physics_prior(cl["pts"], cap.up_cam, True, CFG)
    assert abs(pp["height_m"] - 2.3) < 0.015, f"floor-fit height {pp['height_m']:.3f} vs 2.300"
    assert pp["imu_vs_floor_tilt_deg"] < 0.3, f"IMU vs floor tilt {pp['imu_vs_floor_tilt_deg']:.3f} deg"
    R_L = pp["R_L_cam"]
    assert np.allclose(R_L @ cap.up_cam, [0, 0, 1], atol=1e-6) and abs(np.linalg.det(R_L) - 1) < 1e-9
    try:
        level_rotation(np.array([0.0, 0.0, 1.0]))
        raise AssertionError("looking straight up/down must raise")
    except PhysicsPriorError:
        pass
    # camera that sees no floor at all: must refuse rather than invent a height
    cap2 = render_capture(w, cam_pose_room(2.6, 2.0, 1.0, 180, -60), n_frames=15, seed=4)   # looks UP at the ceiling
    try:
        physics_prior(capture_to_cloud(cap2, CFG)["pts"], cap2.up_cam, True, CFG)
        raise AssertionError("ceiling-only view should not yield a floor fit")
    except PhysicsPriorError:
        pass
    print(f"  height {pp['height_m']:.3f} m (truth 2.300), IMU-vs-floor tilt {pp['imu_vs_floor_tilt_deg']:.3f} deg; "
          f"straight-up / no-floor views refuse -- OK")


# ------------------------------------------------------------------ Stage 3/4
def check_global_search_finds_true_pose_without_a_guess():
    from pyslam.anchor.room_frame import derive_room_frame
    from pyslam.anchor.query_cloud import capture_to_cloud
    from pyslam.anchor.physics_prior import physics_prior
    from pyslam.anchor.global_search import (build_map_raster, fine_config, query_slice, csm_search, distinct_peaks, refine_local)
    w, P, C, Tmr, pts_map = _default_map()
    cap = render_capture(w, T_TRUE, n_frames=25, seed=5)
    cl = capture_to_cloud(cap, CFG)
    pp = physics_prior(cl["pts"], cap.up_cam, True, CFG)
    q, qw = query_slice((pp["R_L_cam"] @ cl["pts"].T).T, pp["height_m"], CFG, cl["weights"])
    ras = build_map_raster(P, CFG)
    fras = build_map_raster(P, fine_config(CFG), pad_m=CFG.csm_range_m + 0.5)
    pk = distinct_peaks(csm_search(ras, q, qw, CFG), CFG)
    T_room_L = T_TRUE @ lie.se3_inverse(lie.make_T(pp["R_L_cam"], np.zeros(3)))
    yaw_true = np.arctan2(T_room_L[1, 0], T_room_L[0, 0])
    best = min((refine_local(fras, q, qw, c) for c in pk[:3]),
               key=lambda c: -c.refined_score)
    dxy = np.hypot(best.x - T_TRUE[0, 3], best.y - T_TRUE[1, 3])
    dyaw = abs(np.degrees((best.yaw - yaw_true + np.pi) % (2 * np.pi) - np.pi))
    assert dxy < 0.03 and dyaw < 0.7, f"global search+refine off by {dxy*100:.1f} cm / {dyaw:.2f} deg"
    print(f"  exhaustive x/y/yaw search (no guess) + local refine: {dxy*100:.1f} cm / {dyaw:.2f} deg from truth -- OK")


def check_icp_recovers_known_transform_and_jacobian_convention():
    from pyslam.anchor.query_cloud import capture_to_cloud
    from pyslam.anchor.icp import MapTarget, icp
    w, P, C, Tmr, pts_map = _default_map()
    cap = render_capture(w, T_TRUE, n_frames=25, seed=6, noise_k=0.0, dropout=0.0)      # NOISE-FREE => sub-mm oracle
    cl = capture_to_cloud(cap, CFG)
    Pclean, _ = w.sample_surface(noise=0.0, seed=1)
    tgt = MapTarget.crop(Pclean, None, T_TRUE[:3, 3], 4.8)
    for xi in (np.array([0.06, -0.05, 0.03, 0.02, -0.03, 0.04]), np.array([-0.05, 0.06, -0.04, -0.03, 0.02, -0.05])):
        T0 = lie.se3_exp(xi) @ T_TRUE
        r = icp(cl["pts"], cl["weights"], cl["sigma"], tgt, T0, CFG)
        d = lie.se3_inverse(T_TRUE) @ r.T
        et, er = np.linalg.norm(d[:3, 3]) * 1000, np.degrees(np.linalg.norm(lie.so3_log(d[:3, :3])))
        assert et < 1.0 and er < 0.03, f"ICP from a {np.linalg.norm(xi[:3])*100:.0f} cm / {np.degrees(np.linalg.norm(xi[3:])):.1f} deg seed "\
                                        f"ended {et:.2f} mm / {er:.3f} deg off"
    print(f"  ICP (noise-free oracle) recovers the pose to {et:.2f} mm / {er:.4f} deg from 8 cm / 5 deg seeds -- OK")


def check_icp_covariance_is_not_overconfident_nees():
    from pyslam.anchor.query_cloud import capture_to_cloud
    from pyslam.anchor.icp import MapTarget, icp
    w, P, C, Tmr, pts_map = _default_map()
    tgt = MapTarget.crop(P, None, T_TRUE[:3, 3], 4.8)
    nees = []
    errs_t = []
    for s in range(14):
        cap = render_capture(w, T_TRUE, n_frames=12, seed=100 + s)
        cl = capture_to_cloud(cap, CFG, stride=3)
        T0 = lie.se3_exp(np.array([0.02, -0.02, 0.01, 0.01, -0.01, 0.01])) @ T_TRUE
        r = icp(cl["pts"], cl["weights"], cl["sigma"], tgt, T0, CFG, scales=(0.05, 0.03, 0.02), iters=(15, 20, 20))
        e = np.concatenate([r.T[:3, 3] - T_TRUE[:3, 3], lie.so3_log(r.T[:3, :3] @ T_TRUE[:3, :3].T)])
        nees.append(float(e @ np.linalg.solve(r.cov, e)))
        errs_t.append(np.linalg.norm(e[:3]))
    m = float(np.mean(nees))
    assert m < 6.0, f"mean NEES {m:.2f} > 6 (dof): reported covariance is OVERCONFIDENT"
    print(f"  ICP covariance vs actual error over 14 noise draws: mean NEES {m:.2f} (<6 = not overconfident; "
          f"actual translation error mean {np.mean(errs_t)*1000:.1f} mm) -- OK")


# ------------------------------------------------------------------ full pipeline
def check_full_calibration_accuracy_with_a_moved_chair():
    from pyslam.anchor.calibrate import calibrate
    w, P, C, Tmr, pts_map = _default_map()
    w2 = default_world()
    w2.boxes[0] = w2.boxes[0].moved(np.array([0.0, 0.4, 0.0]))                  # the sofa moved 40 cm since mapping
    w2.boxes.append(Box((1.5, 2.2, 0), (2.0, 2.7, 0.5), (60, 160, 60)))         # + a NEW stool that was never mapped
    caps = [render_capture(w2, T_TRUE, n_frames=40, seed=s) for s in (10, 11, 12)]
    res = calibrate(_mb(pts_map, C), caps, CFG)
    et, er = _err(res, Tmr, T_TRUE)
    assert res.accepted, f"should be accepted; failed gates: {_failed(res)}"
    assert et < 0.005 and er < 0.1, f"pose error {et*1000:.2f} mm / {er:.3f} deg (spec < 5 mm / 0.1 deg)"
    assert res.sigma["trans_m"] >= et and res.sigma["rot_deg"] >= er, "reported sigma smaller than the ACTUAL error"
    assert res.sigma["trans_m"] < CFG.gate_sigma_trans_m
    T = res.T_room_cam
    assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-9 and np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert res.report["ambiguity"]["ambiguous"] is False
    print(f"  full calibration (map frame awkward; sofa moved 40 cm; unmapped stool): error {et*1000:.2f} mm / {er:.3f} deg, "
          f"sigma {res.sigma['trans_m']*1000:.1f} mm / {res.sigma['rot_deg']:.2f} deg (>= actual), ACCEPTED -- OK")


@functools.lru_cache(maxsize=None)
def _symmetric_case():
    w = World(room=(4.0, 4.0, 2.6), boxes=[], textured=False,
              wall_colors=[(190, 190, 190)] * 6)
    P, C = w.sample_surface()
    return w, P, C


def check_symmetric_room_is_flagged_ambiguous_and_a_hint_resolves_it():
    from pyslam.anchor.calibrate import calibrate
    w, P, C = _symmetric_case()
    Tmr = awkward_map_frame(seed=3, yaw_deg=71.0)
    pts_map = lie.transform_points(Tmr, P)
    T = cam_pose_room(2.0, 2.0, 2.2, 30, 40)                                     # exactly centred: 4-fold symmetric view
    cap = render_capture(w, T, n_frames=30, seed=20)
    res = calibrate(_mb(pts_map, C), [cap], CFG)
    assert not res.accepted and "global_uniqueness" in _failed(res), \
        f"symmetric room must be REJECTED as ambiguous; failed={_failed(res)} amb={res.report['ambiguity'].get('note')}"
    assert res.report["ambiguity"]["second_fitness_ratio"] > 0.9
    # an operator hint (room frame of THIS map: convert the true pose through the derived frame)
    Td = res.room_frame.T_room_map @ Tmr @ T
    yaw_fwd = np.arctan2((Td[:3, :3] @ [0, 0, 1])[1], (Td[:3, :3] @ [0, 0, 1])[0])
    res2 = calibrate(_mb(pts_map, C), [cap], CFG, hint=(Td[0, 3], Td[1, 3], 0.6, yaw_fwd))
    et, er = _err(res2, Tmr, T)
    assert res2.accepted and "operator_hint" in res2.method, f"hint should resolve it; failed={_failed(res2)}"
    assert et < 0.01 and er < 0.2, f"hinted pose off by {et*1000:.1f} mm / {er:.2f} deg"
    print(f"  symmetric room: REJECTED as ambiguous (2nd-best pose fitness ratio {res.report['ambiguity']['second_fitness_ratio']:.2f}); "
          f"with a hint: accepted, {et*1000:.2f} mm / {er:.3f} deg -- OK")


def check_colour_breaks_a_geometric_tie():
    from pyslam.anchor.calibrate import calibrate
    wc = [(200, 60, 60), (60, 200, 60), (60, 60, 200), (200, 200, 60), (120, 110, 100), (230, 230, 230)]
    w = World(room=(4.0, 4.0, 2.6), boxes=[], textured=False, wall_colors=wc)      # geometry symmetric, walls coloured differently
    P, C = w.sample_surface()
    Tmr = awkward_map_frame(seed=4, yaw_deg=12.0)
    T = cam_pose_room(2.0, 2.0, 2.2, 30, 40)
    cap = render_capture(w, T, n_frames=30, seed=21, color=True)
    res = calibrate(_mb(lie.transform_points(Tmr, P), C), [cap], CFG)
    et, er = _err(res, Tmr, T)
    assert res.accepted and res.report["ambiguity"]["tiebreak"] == "colour", \
        f"colour should have broken the tie: accepted={res.accepted} amb={res.report['ambiguity']}"
    assert et < 0.01 and er < 0.2, f"colour-tiebroken pose is the WRONG symmetric image ({et:.2f} m / {er:.1f} deg)"
    print(f"  geometrically symmetric room with distinct wall colours: tie broken by colour to the right pose "
          f"({et*1000:.2f} mm / {er:.3f} deg) -- OK")


def check_map_holes_reject_when_in_view_and_are_harmless_when_not():
    from pyslam.anchor.calibrate import calibrate
    w = default_world()
    Tmr = awkward_map_frame()
    cap = render_capture(w, T_TRUE, n_frames=30, seed=22)
    # (a) dropped keyframes leave a hole BEHIND the camera: irrelevant to this view -> must still calibrate correctly
    P, C = w.sample_surface(drop_region=((-1, 3.7, -1), (0.7, 5, 3)))
    res_a = calibrate(_mb(lie.transform_points(Tmr, P), C), [cap], CFG)
    et, er = _err(res_a, Tmr, T_TRUE)
    assert res_a.accepted and et < 0.005 and er < 0.1, f"an out-of-view hole must not hurt: failed={_failed(res_a)} err={et*1000:.1f} mm"
    # (b) holes covering much of what the camera sees: must be REJECTED (by whichever independent gate trips first)
    outcomes = []
    for dr in (((3.4, -1, -1), (6, 5, 3)), ((-1, 3.0, -1), (6, 5, 3))):
        P, C = w.sample_surface(drop_region=dr)
        res_b = calibrate(_mb(lie.transform_points(Tmr, P), C), [cap], CFG)
        assert not res_b.accepted, f"map with a hole across the camera's view was ACCEPTED (err {_err(res_b, Tmr, T_TRUE)})"
        outcomes.append(sorted(_failed(res_b))[:3])
    print(f"  out-of-view hole: accepted, {et*1000:.2f} mm; holes across the view: REJECTED (first failing gates {outcomes}) -- OK")


def check_bad_imu_and_missing_imu_are_caught():
    from pyslam.anchor.calibrate import calibrate
    w, P, C, Tmr, pts_map = _default_map()
    cap = render_capture(w, T_TRUE, n_frames=30, seed=23, gravity_bias_deg=2.0)  # accelerometer 2 deg off
    res = calibrate(_mb(pts_map, C), [cap], CFG)
    assert not res.accepted and "roll_pitch_vs_imu_deg" in _failed(res), f"2-deg IMU error must trip the cross-check; failed={_failed(res)}"
    cap0 = render_capture(w, T_TRUE, n_frames=30, seed=24, imu=False)
    res0 = calibrate(_mb(pts_map, C), [cap0], CFG)
    assert not res0.accepted and "roll_pitch_vs_imu_deg" in _failed(res0), "missing IMU must fail when require_imu=True"
    from pyslam.anchor.config import apply_overrides
    res1 = calibrate(_mb(pts_map, C), [cap0], apply_overrides(CFG, ["require_imu=false"]))
    et, er = _err(res1, Tmr, T_TRUE)
    assert res1.accepted and et < 0.01 and any("IMU" in x for x in res1.report["warnings"]), \
        f"require_imu=false should accept with a warning; failed={_failed(res1)}"
    print(f"  2 deg accelerometer error -> REJECTED by roll/pitch cross-check; no IMU -> rejected (require_imu) / "
          f"accepted with a warning when waived ({et*1000:.1f} mm) -- OK")


def check_degenerate_view_of_one_flat_wall_is_rejected():
    from pyslam.anchor.calibrate import calibrate
    w = World(room=(5.2, 4.1, 2.6), boxes=[], textured=False)
    P, C = w.sample_surface()
    Tmr = awkward_map_frame(seed=5)
    T = cam_pose_room(2.2, 2.05, 2.0, 180, 40)                                    # 2.2 m from the -x wall, steeply down: floor + ONE wall
    cap = render_capture(w, T, n_frames=30, seed=25)
    res = calibrate(_mb(lie.transform_points(Tmr, P), C), [cap], CFG)
    bad = _failed(res)
    assert not res.accepted and bad & {"observability_min_eig", "global_uniqueness", "sigma_trans_m", "sigma_rot_deg"}, \
        f"a view of one flat wall + floor cannot fix x/y/yaw and must be rejected; failed={bad}"
    print(f"  view of a single flat wall + floor: REJECTED ({sorted(bad)}; weakest direction: {res.report['icp']['obs_worst_direction']}) -- OK")


# ------------------------------------------------------------------ outputs / verification
@functools.lru_cache(maxsize=None)
def _accepted_result():
    from pyslam.anchor.calibrate import calibrate
    w, P, C, Tmr, pts_map = _default_map()
    caps = [render_capture(w, T_TRUE, n_frames=30, seed=s) for s in (30, 31)]
    return calibrate(_mb(pts_map, C), caps, CFG), Tmr


def check_bundle_calibration_json_matches_m4_contract_and_rejected_name():
    from pyslam.anchor.bundle import write_anchor_bundle
    res, Tmr = _accepted_result()
    assert res.accepted, _failed(res)
    with tempfile.TemporaryDirectory() as d:
        info = write_anchor_bundle(res, d)
        names = set(os.listdir(d))
        for f in ("calibration.cam0.json", "room_frame.json", "walkable.json", "report.json", "overlay.png",
                  "depth_residual.png", "top_down.png", "frustum_coverage.png", "cam0_reference_depth.npz", "static_capture.npz"):
            assert f in names, f"missing {f}"
        assert os.path.exists(os.path.join(d, "viewer", "points.ply"))
        cal = json.load(open(info["calibration"]))
        T = np.array(cal["T_room_cam"])
        assert T.shape == (4, 4) and np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-3) and np.linalg.det(T[:3, :3]) > 0
        assert cal["map_id"] == "synthetic0001" and set(cal["sigma"]) == {"trans_m", "rot_deg"} and cal["accepted"] is True
        w = json.load(open(os.path.join(d, "walkable.json")))
        assert set(["origin", "resolution", "grid"]) <= set(w) and isinstance(w["grid"][0][0], bool)
        from pyslam.anchor.map_io import read_ply
        vp, _ = read_ply(os.path.join(d, "viewer", "points.ply"))
        assert abs(np.median(vp[:, 2][vp[:, 2] < 0.05])) < 0.01, "viewer cloud not in the floor-referenced room frame"
        # M4's own loader, if the STS repo is on the path (it is a separate repo -> optional)
        try:
            from poi_localization.frames.room_frame import load_room_frame
            tf = load_room_frame(info["calibration"], expected_map_id="synthetic0001")
            assert np.allclose(tf.R, T[:3, :3]) and abs(tf.camera_height_m - T[2, 3]) < 1e-9
            m4 = "loads through M4's load_room_frame"
        except ModuleNotFoundError:
            m4 = "M4 repo not on PYTHONPATH -> contract checked inline"
        # a rejected calibration must never land under the name M4 loads
        res.accepted = False
        d2 = os.path.join(d, "rej")
        info2 = write_anchor_bundle(res, d2)
        res.accepted = True
        assert os.path.basename(info2["calibration"]) == "calibration.cam0.REJECTED.json"
        assert "calibration.cam0.json" not in os.listdir(d2)
    print(f"  anchor bundle complete; calibration.json satisfies M4's contract ({m4}); rejected => .REJECTED.json -- OK")


def check_frustum_coverage_and_depth_residual_and_overlay():
    res, Tmr = _accepted_result()
    f = res.frustum
    assert 0.15 < f["frac_visible"] < 0.95, f"implausible visible fraction {f['frac_visible']:.2f}"
    assert abs(f["frac_visible"] + f["frac_in_frustum_unverified"] + f["frac_dead"] - 1.0) < 1e-6
    # cells directly behind the camera must be dead
    T = res.T_room_cam
    fwd = (T[:3, :3] @ [0, 0, 1])[:2]; fwd /= np.linalg.norm(fwd)
    w = res.walkable
    r = w["resolution"]; ox, oy = w["origin"]
    behind = T[:2, 3] - 1.0 * fwd
    row, col = int((behind[1] - oy) / r), int((behind[0] - ox) / r)
    if 0 <= row < w["grid"].shape[0] and 0 <= col < w["grid"].shape[1]:
        assert not f["visible_grid"][row, col], "a cell behind the camera was marked visible"
    dr = res.report["depth_residual"]
    assert dr["frac_within_tol"] > 0.9
    ov = res.images["overlay"]
    assert ov.shape == (480, 640, 3) and ov.dtype == np.uint8
    print(f"  frustum: {f['frac_visible']:.0%} of walkable floor visible, {f['frac_dead']:.0%} dead; "
          f"depth-residual {dr['frac_within_tol']:.0%} within 3 cm; overlay rendered -- OK")


def check_marker_check_passes_true_pose_and_fails_a_wrong_one():
    from pyslam.anchor.verify import check_markers
    res, Tmr = _accepted_result()
    walls = res.room_frame.walls
    T = res.T_room_cam
    K = res.capture.intr
    # floor markers at known room positions, seen at known pixels (from the TRUE pose, expressed in the derived frame)
    T_true_d = res.room_frame.T_room_map @ Tmr @ T_TRUE
    mk = []
    for name, xy_true in (("m1", (3.25, 1.8)), ("m2", (3.5, 1.3)), ("m3", (3.75, 2.05))):      # floor points this camera actually sees
        pd = (res.room_frame.T_room_map @ Tmr @ np.array([*xy_true, 0.0, 1.0]))[:3]
        pc = T_true_d[:3, :3].T @ (pd - T_true_d[:3, 3])
        u, v = K["fx"] * pc[0] / pc[2] + K["cx"], K["fy"] * pc[1] / pc[2] + K["cy"]
        assert 0 <= u < 640 and 0 <= v < 480, f"marker {name} not in view"
        # 'measured' by tape from two walls: distances to the wall planes (inward normals) in the derived frame
        best = []
        for wl in walls[:4]:
            dist = float(np.array(wl["normal_in_room"]) @ pd - wl["offset_room"])
            best.append((wl["id"], dist, np.array(wl["normal_in_room"][:2])))
        a = best[0]
        b = next(x for x in best[1:] if abs(np.cross(a[2], x[2])) > 0.5)
        mk.append({"name": name, "pixel": [u, v], "wall_offsets": [{"wall": a[0], "d": a[1]}, {"wall": b[0], "d": b[1]}]})
    ok = check_markers(mk, T, K, walls, 0.03)
    assert ok["pass"] and ok["max_error_m"] < 0.02, f"true-pose markers: {ok['max_error_m']}"
    Twrong = T.copy()
    Twrong[:3, :3] = lie.so3_exp(np.array([0, 0, np.radians(1.5)])) @ T[:3, :3]     # 1.5 deg yaw error
    bad = check_markers(mk, Twrong, K, walls, 0.03)
    assert not bad["pass"], f"a 1.5 deg yaw error must fail the 3 cm marker check (got {bad['max_error_m']:.3f} m)"
    print(f"  wall-referenced marker check: true pose max err {ok['max_error_m']*1000:.1f} mm (pass); "
          f"1.5 deg yaw error -> {bad['max_error_m']*100:.1f} cm (FAIL) -- OK")


def check_watchdog_detects_tilt_yaw_nudge_but_not_a_person():
    from pyslam.anchor.watchdog import AnchorWatchdog
    from pyslam.anchor.bundle import write_anchor_bundle
    from pyslam.anchor.query_cloud import stable_mask
    res, Tmr = _accepted_result()
    w = default_world()
    with tempfile.TemporaryDirectory() as d:
        write_anchor_bundle(res, d)
        mk = lambda: AnchorWatchdog(os.path.join(d, "calibration.cam0.json"), os.path.join(d, "cam0_reference_depth.npz"))
        # 1. unchanged + a person-sized occluder over ~8% of the image -> stays ok
        wd = mk()
        same = render_capture(w, T_TRUE, n_frames=15, seed=40)
        dm = same.depth_med.copy()
        dm[150:330, 250:330] = 1.5
        for _ in range(4):
            wd.check_depth(dm)
        assert wd.health_field() == "ok", f"a person in view must not trip the watchdog ({wd.state})"
        assert wd.check_tilt(same.up_cam) < wd.cfg.wd_tilt_deg
        # 2. tilt nudge of 0.6 deg (pitch) -> tilt layer fires within ONE check
        wd = mk()
        Tn = T_TRUE.copy(); Tn[:3, :3] = Tn[:3, :3] @ lie.so3_exp(np.array([np.radians(0.6), 0, 0]))
        nud = render_capture(w, Tn, n_frames=15, seed=41)
        wd.check_tilt(nud.up_cam)
        assert wd.health_field() == "suspect" and "tilt" in wd.state.reason
        # 3. pure yaw nudge: the IMU is BLIND to it. A 4 deg nudge must be caught by the depth layer; a 1.5 deg one is
        #    (honestly) below what the depth layer can see and is left to the ICP recheck in step 4.
        wd = mk()
        T4 = T_TRUE.copy(); T4[:3, :3] = lie.so3_exp(np.array([0, 0, np.radians(4.0)])) @ T4[:3, :3]
        big = render_capture(w, T4, n_frames=15, seed=43)
        assert wd.check_tilt(big.up_cam) < wd.cfg.wd_tilt_deg, "yaw nudge should be invisible to the accelerometer"
        for _ in range(wd.cfg.wd_consecutive):
            fr = wd.check_depth(big.depth_med)
        assert wd.health_field() == "suspect", f"4 deg yaw nudge not caught by depth layer (moved_frac={fr:.2f})"
        Ty = T_TRUE.copy(); Ty[:3, :3] = lie.so3_exp(np.array([0, 0, np.radians(1.5)])) @ Ty[:3, :3]
        yaw = render_capture(w, Ty, n_frames=15, seed=42)
        wd = mk()
        small = wd.check_depth(yaw.depth_med)
        # 4. fast ICP recheck measures the shift, and clears an unmoved camera
        pts_room = res.pts_room
        wd = mk()
        out = wd.recheck_pose(yaw, pts_room, res.colors)
        assert out["verdict"] == "moved" and 1.0 < out["shift_deg"] < 2.0, f"recheck: {out['verdict']} {out['shift_deg']:.2f} deg"
        wd = mk(); wd.state.status = "suspect"
        ok = wd.recheck_pose(same, pts_room, res.colors)
        assert ok["verdict"] == "ok" and wd.health_field() == "ok", f"unmoved camera flagged: {ok}"
    print(f"  watchdog: person occluder ignored; 0.6 deg tilt caught by IMU layer; 4 deg YAW (IMU-blind) caught by depth layer; "
          f"1.5 deg yaw moves only {small:.0%} of pixels (below the depth layer) but the ICP recheck measured it ({out['shift_deg']:.2f} deg) "
          f"and cleared the unmoved camera -- OK")


def check_cli_prepare_solve_markers_end_to_end():
    from pyslam.mapping.export import write_ply
    w, P, C, Tmr, pts_map = _default_map()
    cap = render_capture(w, T_TRUE, n_frames=30, seed=50)
    with tempfile.TemporaryDirectory() as d:
        mdir = os.path.join(d, "map"); os.makedirs(os.path.join(mdir, "dense"))
        write_ply(os.path.join(mdir, "dense", "points.ply"), pts_map.astype(np.float32), C)
        json.dump({"width": 640}, open(os.path.join(mdir, "capture_profile.json"), "w"))     # unused fields tolerated
        cpath = os.path.join(d, "c0.npz"); cap.save(cpath)
        cap.save(os.path.join(d, "c1.npz"))
        out = os.path.join(d, "anchor")
        env = dict(os.environ); env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run([sys.executable, "run_anchor.py", "prepare", "--map", mdir, "--out", out],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0 and os.path.exists(os.path.join(out, "top_down.png")), r.stdout[-400:] + r.stderr[-400:]
        r = subprocess.run([sys.executable, "run_anchor.py", "solve", "--map", mdir, "--capture", os.path.join(d, "c*.npz"), "--out", out],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, "solve failed:\n" + r.stdout[-800:] + r.stderr[-800:]
        assert os.path.exists(os.path.join(out, "calibration.cam0.json"))
    print("  CLI: prepare -> solve (glob of captures) exit 0, bundle written -- OK")


CHECKS = [
    check_ply_round_trip_and_map_id,
    check_room_frame_recovers_floor_up_and_walls,
    check_room_frame_honest_about_a_bowed_floor,
    check_walkable_grid_schema_and_semantics,
    check_capture_median_imu_sign_and_edge_filter,
    check_physics_prior_height_tilt_and_failure_modes,
    check_global_search_finds_true_pose_without_a_guess,
    check_icp_recovers_known_transform_and_jacobian_convention,
    check_icp_covariance_is_not_overconfident_nees,
    check_full_calibration_accuracy_with_a_moved_chair,
    check_symmetric_room_is_flagged_ambiguous_and_a_hint_resolves_it,
    check_colour_breaks_a_geometric_tie,
    check_map_holes_reject_when_in_view_and_are_harmless_when_not,
    check_bad_imu_and_missing_imu_are_caught,
    check_degenerate_view_of_one_flat_wall_is_rejected,
    check_bundle_calibration_json_matches_m4_contract_and_rejected_name,
    check_frustum_coverage_and_depth_residual_and_overlay,
    check_marker_check_passes_true_pose_and_fails_a_wrong_one,
    check_watchdog_detects_tilt_yaw_nudge_but_not_a_person,
    check_cli_prepare_solve_markers_end_to_end,
]


def main():
    only = sys.argv[1:]
    passed = 0
    todo = [fn for fn in CHECKS if not only or any(o in fn.__name__ for o in only)]
    for fn in todo:
        print(f"[{fn.__name__}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\nG-ANCHOR: {passed}/{len(todo)} passed")
    if passed != len(todo):
        sys.exit(1)


if __name__ == "__main__":
    main()
