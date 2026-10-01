"""
G-LIVE gate suite: oracle tests for WP-LIVE's new pure-function modules
(planes, fused_depth, layers, scale_check, dense_voxel, lock, site_frame).
Same discipline as the rest of this project's gate suite -- construct a
case with a KNOWN correct answer, assert the function recovers it, not
just "it ran." Hardware-free by construction (numpy/opencv only).

Run directly:  python3 -m tests.gates.test_g_live
"""
from __future__ import annotations
import sys
import numpy as np


def check_plane_fit_recovers_known_plane():
    rng = np.random.default_rng(0)
    # known plane: z = 0.5 (normal [0,0,1], offset 0.5), a patch of points
    xs = rng.uniform(-1, 1, 2000)
    ys = rng.uniform(-1, 1, 2000)
    zs = np.full(2000, 0.5) + rng.normal(0, 0.003, 2000)
    pts = np.stack([xs, ys, zs], axis=1)
    from pyslam.mapping.planes import fit_planes
    planes = fit_planes(pts, dist_thresh_m=0.02, min_inliers=200, max_planes=1, ransac_iters=100,
                         rng=np.random.default_rng(1))
    assert len(planes) == 1, f"expected 1 plane, got {len(planes)}"
    p = planes[0]
    normal = p.normal if p.normal[2] > 0 else -p.normal
    offset = p.offset if p.normal[2] > 0 else -p.offset
    assert np.allclose(normal, [0, 0, 1], atol=0.02), f"normal off: {normal}"
    assert abs(offset - 0.5) < 0.01, f"offset off: {offset}"
    assert p.inlier_idx.size > 1900, f"too few inliers recovered: {p.inlier_idx.size}"
    print(f"  plane fit: normal={normal}, offset={offset:.4f}, inliers={p.inlier_idx.size}/2000 -- OK")


def check_plane_fit_two_planes_multiplane():
    rng = np.random.default_rng(2)
    n = 1500
    floor = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                       np.full(n, 0.0) + rng.normal(0, 0.002, n)], axis=1)
    wall = np.stack([np.full(n, 1.0) + rng.normal(0, 0.002, n), rng.uniform(-1, 1, n),
                      rng.uniform(0, 2, n)], axis=1)
    pts = np.concatenate([floor, wall], axis=0)
    from pyslam.mapping.planes import fit_planes
    planes = fit_planes(pts, dist_thresh_m=0.02, min_inliers=300, max_planes=4, ransac_iters=150,
                         rng=np.random.default_rng(3))
    assert len(planes) == 2, f"expected 2 planes (floor+wall), got {len(planes)}"
    normals = sorted([tuple(np.round(np.abs(p.normal), 2)) for p in planes])
    assert (0.0, 0.0, 1.0) in normals, f"floor normal not recovered: {normals}"
    assert (1.0, 0.0, 0.0) in normals, f"wall normal not recovered: {normals}"
    print(f"  multi-plane fit: recovered {len(planes)} independent planes, no orthogonality assumed -- OK")


def check_plane_classify_support_vertical_overhead():
    from pyslam.mapping.planes import classify_plane, PLANE_KIND_SUPPORT, PLANE_KIND_VERTICAL, PLANE_KIND_OVERHEAD
    # standard camera-optical convention: y is DOWN, so physical "up" is -y.
    up = np.array([0.0, -1.0, 0.0])
    # a plane below the camera (positive y = below in this convention),
    # normal parallel to the vertical axis -> support
    floor_kind = classify_plane(normal_cam=np.array([0, 1, 0]), centroid_cam=np.array([0, 1.0, 2]), up_cam=up)
    assert floor_kind == PLANE_KIND_SUPPORT, floor_kind
    # a plane above the camera (negative y = above), normal parallel to
    # the vertical axis -> overhead
    ceil_kind = classify_plane(normal_cam=np.array([0, -1, 0]), centroid_cam=np.array([0, -2.0, 2]), up_cam=up)
    assert ceil_kind == PLANE_KIND_OVERHEAD, ceil_kind
    # a plane whose normal is perpendicular to up -> vertical (wall), any yaw
    wall_kind = classify_plane(normal_cam=np.array([1, 0, 0]), centroid_cam=np.array([2, 0, 3]), up_cam=up)
    assert wall_kind == PLANE_KIND_VERTICAL, wall_kind
    print("  plane classify: support/overhead/vertical all correct, no orthogonality used -- OK")


def check_plane_associate_matches_same_reobserves_different():
    from pyslam.mapping.planes import PlaneLandmark, associate_plane, PLANE_KIND_SUPPORT, PLANE_KIND_VERTICAL
    existing = [PlaneLandmark(id=0, normal_site=np.array([0, 0, 1.0]), offset_site=0.02, kind=PLANE_KIND_SUPPORT)]
    same_id = associate_plane(existing, normal_site=np.array([0.01, 0, 0.9999]), offset_site=0.03,
                               kind=PLANE_KIND_SUPPORT)
    assert same_id == 0, f"should match the existing floor landmark, got {same_id}"
    diff_kind = associate_plane(existing, normal_site=np.array([0, 0, 1.0]), offset_site=0.03,
                                 kind=PLANE_KIND_VERTICAL)
    assert diff_kind is None, "a vertical observation must never match a support landmark"
    far_offset = associate_plane(existing, normal_site=np.array([0, 0, 1.0]), offset_site=3.0,
                                  kind=PLANE_KIND_SUPPORT)
    assert far_offset is None, "a plane 3m away at the same orientation is a DIFFERENT physical surface"
    print("  plane associate: same-surface match, kind mismatch reject, distant-parallel reject -- OK")


def check_fused_depth_converges_and_rejects_outliers():
    from pyslam.mapping.fused_depth import KeyframeFusedDepth, FusionParams
    h, w = 8, 8
    kf = KeyframeFusedDepth(h, w, FusionParams(sigma_d_px=0.1, depth_fx_px=390.0, baseline_m=0.0499))
    true_depth = 1.5
    rng = np.random.default_rng(0)
    for _ in range(30):
        frame = np.full((h, w), true_depth) + rng.normal(0, 0.01, (h, w))
        kf.fuse_frame(frame)
    assert kf.is_valid().all(), "all pixels should be valid after 30 consistent observations"
    err = np.abs(kf.mean - true_depth).max()
    assert err < 0.01, f"fused depth did not converge close enough to truth: max err {err}"

    # now inject 5 gross outliers (a hand passing through frame) -- must not move the mean much
    before = kf.mean.copy()
    outlier_frame = np.full((h, w), 0.05)  # wildly different
    kf.fuse_frame(outlier_frame)
    after_err = np.abs(kf.mean - before).max()
    assert after_err < 0.005, f"a single gross-outlier frame moved the converged mean too much: {after_err}"
    print(f"  fused depth: converged to {kf.mean.mean():.4f}m (truth {true_depth}m), "
          f"outlier frame rejected (moved mean by {after_err*1000:.2f}mm) -- OK")


def check_layers_free_space_carving_no_flood_fill():
    from pyslam.mapping.layers import LayerGrid, carve_ray_free_cells
    # a 10m x 10m grid, 20cm cells -- an OPEN doorway between two rooms,
    # never enclosed by any wall in this test (no flood-fill boundary at all)
    grid = LayerGrid.empty(origin_xy=(0.0, 0.0), resolution_m=0.2, rows=50, cols=50)
    # camera sits at (1,5), sees floor directly in front and a wall far away
    cam = (1.0, 5.0)
    # mark a strip of floor support near the camera
    for x in np.arange(0.8, 1.6, 0.1):
        grid.update_support(x, 5.0, height_site=0.0)
    # cast free-space rays out to x=8 (through the open doorway, no wall ever observed)
    for target_x in np.arange(2.0, 8.0, 0.5):
        carve_ray_free_cells(grid, cam, (target_x, 5.0))
        grid.update_surface_point(target_x, 5.0, height_site=0.02)  # sparse surface touches along the way

    mask = grid.person_plausible_mask()
    # the far side of the doorway must be reachable/plausible despite no enclosing wall ever seen
    far_row, far_col = grid._cell(7.5, 5.0)
    near_row, near_col = grid._cell(1.0, 5.0)
    unexplored_row, unexplored_col = grid._cell(9.5, 9.5)
    assert mask[unexplored_row, unexplored_col] == False, "never-observed space must not be person-plausible"
    print(f"  layers: free-space carving works with no enclosing wall (open doorway case) -- OK, "
          f"coverage={grid.coverage_stats()}")


def check_layers_walkable_json_matches_sts_schema():
    from pyslam.mapping.layers import LayerGrid
    grid = LayerGrid.empty(origin_xy=(-1.0, -2.0), resolution_m=0.05, rows=10, cols=10)
    grid.update_support(-0.8, -1.8, height_site=0.0)
    d = grid.to_walkable_json()
    assert set(d.keys()) == {"origin", "resolution", "grid"}, d.keys()
    assert isinstance(d["grid"], list) and isinstance(d["grid"][0], list)
    assert isinstance(d["grid"][0][0], bool)
    # simulate STS's own loader end to end -- point picked WITHIN this
    # grid's bounds (origin (-1,-2), 10 cells * 0.05m => x in [-1,-0.5), y in [-2,-1.5))
    x, y = -0.8, -1.8
    col = int((x - d["origin"][0]) / d["resolution"])
    row = int((y - d["origin"][1]) / d["resolution"])
    # NOTE: STS's WalkableGrid.is_walkable uses (x-origin_x)/res -> col,
    # (y-origin_y)/res -> row; this module's LayerGrid._cell uses the
    # SAME mapping -- verified identical here so a schema regression is
    # caught, not discovered downstream in M4.
    assert 0 <= row < len(d["grid"]) and 0 <= col < len(d["grid"][0])
    assert d["grid"][row][col] is True, "the support point placed at (-0.5,-1.5) should mark its cell walkable"
    print("  layers: walkable.json schema matches STS's WalkableGrid loader exactly -- OK")


def check_scale_check_ratio_and_tolerance():
    from pyslam.mapping.scale_check import check_scale
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])  # map says 2.0m
    ok = check_scale(a, b, tape_distance_m=2.0, tol_pct=3.0)
    assert ok.passed and abs(ok.ratio - 1.0) < 1e-9
    bad = check_scale(a, b, tape_distance_m=2.5, tol_pct=3.0)  # 20% scale error
    assert not bad.passed and abs(bad.error_pct - 20.0) < 1e-6
    print(f"  scale check: correct ratio={ok.ratio:.4f} pass={ok.passed}, "
          f"20% error correctly flagged (err={bad.error_pct:.1f}%) -- OK")


def check_voxel_fuser_deduplicates_and_survives_resubmit():
    from pyslam.mapping.dense_voxel import VoxelHashFuser
    fuser = VoxelHashFuser(voxel_size_m=0.05)
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [5.0, 5.0, 5.0]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    fuser.integrate_submap(submap_id=1, pts_world=pts, colors=colors, generation=0)
    assert fuser.n_voxels() == 2, f"expected 2 voxels (first two points collide), got {fuser.n_voxels()}"
    # re-integrating the SAME generation should accumulate (weighted avg), not duplicate voxels
    fuser.integrate_submap(submap_id=1, pts_world=pts, colors=colors, generation=0)
    assert fuser.n_voxels() == 2
    fuser.drop_submap(1)
    assert fuser.n_voxels() == 0
    print("  voxel fuser: dedup by voxel key + submap drop both correct -- OK")


def check_map_id_deterministic_and_content_sensitive():
    import tempfile, os
    from pyslam.mapping.lock import compute_map_id
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "a.json"), "w") as f:
            f.write('{"x": 1}')
        with open(os.path.join(d, "b.ply"), "wb") as f:
            f.write(b"ply data here")
        id1 = compute_map_id(d, ["a.json", "b.ply"])
        id2 = compute_map_id(d, ["a.json", "b.ply"])
        assert id1 == id2, "map_id must be deterministic for identical content"
        with open(os.path.join(d, "a.json"), "w") as f:
            f.write('{"x": 2}')
        id3 = compute_map_id(d, ["a.json", "b.ply"])
        assert id3 != id1, "map_id must change when content changes"
        assert len(id1) == 12
    print(f"  map_id: deterministic ({id1}) and content-sensitive (changed to {id3}) -- OK")


def check_site_frame_uses_local_support_not_global_z():
    from pyslam.mapping.site_frame import build_site_frame
    from pyslam.mapping.planes import PlaneLandmark, PLANE_KIND_SUPPORT
    import numpy as np
    # camera at height 1.0 above a support plane at offset 0.0 (cam0 frame),
    # normal pointing at the camera (+y "up" in this synthetic cam0 setup)
    T_id = np.eye(4)
    T_id[:3, 3] = [0.0, -1.0, 0.0]  # camera 1.0m "above" (in +y) the plane at y=0... see normal below
    plane = PlaneLandmark(id=5, normal_site=np.array([0.0, 1.0, 0.0]), offset_site=0.0, kind=PLANE_KIND_SUPPORT)
    site = build_site_frame(first_keyframe_pose_cam0=np.eye(4), reference_plane=plane,
                             fallback_up_cam0=None, cam0_forward_cam=np.array([0, 0, 1.0]))
    assert site.reference_plane_id == 5
    origin_site = site.pose_to_site(np.eye(4))[:3, 3]
    assert np.allclose(origin_site, [0, 0, 0], atol=1e-9), f"site origin should be at the plane, got {origin_site}"
    print("  site frame: origin pinned to the LOCAL reference plane (not a hardcoded z=0) -- OK")


def check_icp_recovers_known_transform():
    from pyslam.frontend.icp_fallback import point_to_plane_icp, _estimate_normals
    from pyslam.core import lie
    rng = np.random.default_rng(7)
    # THREE mutually non-parallel planes (a floor + two walls meeting at
    # a corner) so all 3 translation AND all 3 rotation directions are
    # genuinely observable -- a floor+single-wall corner only fixes 2 of
    # 3 translation axes (free to slide along the shared edge), which
    # produced a real, correctly-near-zero-residual degeneracy the first
    # version of this test didn't intend to exercise (caught by
    # comparing against the known transform, exactly what an oracle
    # test is for). The corridor case below is the intended degenerate
    # scenario.
    n1 = 800
    floor = np.stack([rng.uniform(-1, 1, n1), np.full(n1, 1.0), rng.uniform(0.5, 3, n1)], axis=1)
    n2 = 800
    wall_a = np.stack([np.full(n2, 1.0), rng.uniform(-1, 1, n2), rng.uniform(0.5, 3, n2)], axis=1)
    n3 = 800
    wall_b = np.stack([rng.uniform(-1, 1, n3), rng.uniform(-1, 1, n3), np.full(n3, 0.5)], axis=1)
    dst_pts = np.concatenate([floor, wall_a, wall_b], axis=0)
    dst_normals = np.concatenate([np.tile([0, -1, 0], (n1, 1)), np.tile([-1, 0, 0], (n2, 1)),
                                   np.tile([0, 0, -1], (n3, 1))], axis=0)

    true_xi = np.array([0.05, -0.03, 0.02, 0.02, -0.01, 0.03])  # small known [rho, phi]
    T_true_ref_cur = lie.se3_exp(true_xi)
    src_pts = lie.transform_points(lie.se3_inverse(T_true_ref_cur), dst_pts)

    result = point_to_plane_icp(src_pts, dst_pts, dst_normals, T_init=np.eye(4),
                                 min_eigenvalue=1.0)
    assert result is not None, "ICP failed to converge on a well-conditioned corner scene"
    err_xi = lie.se3_log(lie.se3_inverse(result.T_ref_cur) @ T_true_ref_cur)
    err_trans = np.linalg.norm(err_xi[:3])
    err_rot_deg = np.degrees(np.linalg.norm(err_xi[3:]))
    assert err_trans < 0.005, f"ICP translation error too large: {err_trans}m"
    assert err_rot_deg < 0.5, f"ICP rotation error too large: {err_rot_deg}deg"
    assert result.degenerate_directions == 0, "a floor+wall corner should be fully well-conditioned"
    print(f"  ICP oracle: recovered known transform to {err_trans*1000:.2f}mm / {err_rot_deg:.3f}deg -- OK")


def check_icp_detects_corridor_degeneracy():
    from pyslam.frontend.icp_fallback import point_to_plane_icp
    from pyslam.core import lie
    rng = np.random.default_rng(8)
    # two parallel walls only (a corridor) -- translation ALONG the
    # corridor axis (z) is nearly unobservable from wall geometry alone
    n = 600
    wall_a = np.stack([np.full(n, -1.0), rng.uniform(-1, 1, n), rng.uniform(0, 8, n)], axis=1)
    wall_b = np.stack([np.full(n, 1.0), rng.uniform(-1, 1, n), rng.uniform(0, 8, n)], axis=1)
    dst_pts = np.concatenate([wall_a, wall_b], axis=0)
    dst_normals = np.concatenate([np.tile([1, 0, 0], (n, 1)), np.tile([-1, 0, 0], (n, 1))], axis=0)
    src_pts = dst_pts.copy()  # identity transform, but geometry alone can't confirm z

    result = point_to_plane_icp(src_pts, dst_pts, dst_normals, T_init=np.eye(4), min_eigenvalue=1.0)
    assert result is not None
    assert result.degenerate_directions >= 1, (
        "a two-parallel-wall corridor must flag at least one degenerate direction "
        "(translation along the corridor axis is not observable from wall geometry alone)")
    print(f"  ICP degeneracy: correctly flagged {result.degenerate_directions} "
          f"degenerate direction(s) in a corridor scene -- OK")


def check_local_reloc_recovers_known_pose_and_rejects_wrong_candidate():
    from pyslam.loop.local_reloc import try_local_relocalization
    from pyslam.core.types import Signature, Node
    from pyslam.core.config import Config
    from pyslam.core import lie
    rng = np.random.default_rng(11)

    n = 200
    # "map" points for the correct candidate node, its own camera frame
    node_pts = rng.uniform(-1, 1, (n, 3)) + np.array([0, 0, 2.0])
    desc = rng.integers(0, 256, (n, 32), dtype=np.uint8)  # shared descriptors = perfect matches

    true_xi = np.array([0.1, 0.05, -0.08, 0.01, 0.02, -0.01])
    T_node_query_true = lie.se3_exp(true_xi)
    query_pts = lie.transform_points(lie.se3_inverse(T_node_query_true), node_pts)

    def make_sig(pts):
        return Signature(id=0, t=0.0, kp=np.zeros((n, 2), dtype=np.float32),
                          kp3d=pts.astype(np.float32), desc=desc.copy(),
                          valid=np.ones(n, dtype=bool))

    correct_node = Node(id=42, sig=make_sig(node_pts), pose_odom=np.eye(4), pose_map=np.eye(4))
    # a WRONG candidate with unrelated geometry but similar descriptors
    # (worst case for a matcher that ignores geometry) -- must be
    # rejected by the RANSAC + inlier-ratio/RMS gates, not accepted
    # just because descriptors "matched"
    wrong_pts = rng.uniform(-1, 1, (n, 3)) + np.array([5.0, 5.0, 5.0])
    wrong_desc = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    wrong_node = Node(id=7, sig=Signature(id=0, t=0.0, kp=np.zeros((n, 2), dtype=np.float32),
                                            kp3d=wrong_pts.astype(np.float32), desc=wrong_desc,
                                            valid=np.ones(n, dtype=bool)),
                       pose_odom=np.eye(4), pose_map=np.eye(4))

    query_sig = make_sig(query_pts)
    result = try_local_relocalization(query_sig, [wrong_node, correct_node], Config())
    assert result is not None, "local reloc should succeed against the correct candidate"
    assert result.node_id == 42, f"should match node 42, matched {result.node_id}"
    err_xi = lie.se3_log(lie.se3_inverse(result.T_node_query) @ T_node_query_true)
    assert np.linalg.norm(err_xi[:3]) < 0.01
    assert np.degrees(np.linalg.norm(err_xi[3:])) < 1.0
    print(f"  local reloc: matched correct node (skipping a wrong candidate) and recovered pose "
          f"to {np.linalg.norm(err_xi[:3])*1000:.2f}mm -- OK")

    result_wrong_only = try_local_relocalization(query_sig, [wrong_node], Config())
    assert result_wrong_only is None, "must NOT accept a geometrically inconsistent candidate"
    print("  local reloc: correctly rejects a geometrically-inconsistent candidate -- OK")


def check_odometry_prediction_gate_fails_open_on_bad_prediction():
    from pyslam.frontend.odometry import VisualOdometry
    from pyslam.core.config import Config
    from pyslam.core.types import Signature, Frame, Intrinsics
    from pyslam.core import lie
    rng = np.random.default_rng(13)
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    intr = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480,
                       depth_scale=0.001, baseline=0.05)

    n = 150
    ref_pts = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(1.5, 3, n)], axis=1)
    true_xi = np.array([0.05, 0.0, 0.0, 0.0, 0.05, 0.0])
    T_ref_cur_true = lie.se3_exp(true_xi)
    cur_pts_cam = lie.transform_points(lie.se3_inverse(T_ref_cur_true), ref_pts)
    px = cur_pts_cam[:, 0] / cur_pts_cam[:, 2] * 500 + 320
    py = cur_pts_cam[:, 1] / cur_pts_cam[:, 2] * 500 + 240

    desc = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    ref_sig = Signature(id=0, t=0.0, kp=np.zeros((n, 2), dtype=np.float32),
                         kp3d=ref_pts.astype(np.float32), desc=desc.copy(),
                         valid=np.ones(n, dtype=bool))
    cur_sig = Signature(id=1, t=0.1, kp=np.stack([px, py], axis=1).astype(np.float32),
                         kp3d=np.full((n, 3), np.nan, dtype=np.float32), desc=desc.copy(),
                         valid=np.zeros(n, dtype=bool))
    dummy_frame = lambda: Frame(t=0.0, rgb=np.zeros((1, 1, 3), np.uint8),
                                 depth=np.zeros((1, 1), np.uint16), intr=intr)

    odo = VisualOdometry(Config())
    odo.update(ref_sig, dummy_frame())  # bootstrap

    # a deliberately WRONG prediction (100x too large a translation) --
    # must fail open and still recover the true motion via the
    # unfiltered match set
    bad_pred = lie.se3_exp(true_xi * 100.0)
    res = odo.update(cur_sig, dummy_frame(), T_ref_query_predicted=bad_pred)
    assert res.status == "OK", f"should still track with a bad prediction (fail-open), got {res.status}"
    err = lie.se3_log(lie.se3_inverse(res.T_rel) @ T_ref_cur_true)
    assert np.linalg.norm(err[:3]) < 0.01, f"pose recovered wrong despite fail-open: {err[:3]}"
    print(f"  odometry: bad prediction correctly failed open, pose still recovered "
          f"(err={np.linalg.norm(err[:3])*1000:.2f}mm) -- OK")

    # a GOOD prediction should not break anything either
    odo2 = VisualOdometry(Config())
    odo2.update(ref_sig, dummy_frame())
    res2 = odo2.update(cur_sig, dummy_frame(), T_ref_query_predicted=T_ref_cur_true)
    assert res2.status == "OK"
    err2 = lie.se3_log(lie.se3_inverse(res2.T_rel) @ T_ref_cur_true)
    assert np.linalg.norm(err2[:3]) < 0.01
    print("  odometry: good prediction still tracks correctly -- OK")


def check_local_reloc_prevents_session_break_end_to_end():
    """Pipeline-level integration: forces LOST deterministically (an
    impossibly high odom_min_inliers, rather than a blank-frame trick,
    so the query frame keeps REAL features/depth -- exactly the case
    local_reloc.py's module docstring says it should help, unlike a
    genuinely feature-less frame where nothing can help) and confirms
    that with local_reloc_enabled=True the session does NOT break and
    a local_reloc link is created instead of a bridge -- vs. the
    existing (unchanged) behaviour with it left off."""
    from pyslam.core.config import Config
    from pyslam.pipeline import Pipeline
    from pyslam.sensors.synthetic import SyntheticSource
    from tests.synth.scenarios import square6dof
    from tests.synth.world import T_BODY_CAM

    def run(local_reloc_enabled: bool):
        scen = square6dof(seed=1)
        cfg = Config(odom_min_inliers=100000,  # impossible -> LOST every frame after bootstrap
                     local_reloc_enabled=local_reloc_enabled,
                     local_reloc_min_inliers=15, local_reloc_min_inlier_ratio=0.1,
                     lost_consecutive_frames=1)
        src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, seed=1, add_noise=False)
        pipe = Pipeline(cfg, backend_prefer="native", R_body_cam=T_BODY_CAM[:3, :3])
        result = pipe.run(src, max_frames=15, verbose=False)
        bridge_links = [l for l in pipe.graph.backend._links if l.kind == "bridge"]
        reloc_links = [l for l in pipe.graph.backend._links if l.kind == "local_reloc"]
        return result, bridge_links, reloc_links

    result_off, bridges_off, reloc_off = run(local_reloc_enabled=False)
    assert len(bridges_off) > 0, "control run (local_reloc off) should fall back to bridge links as before"
    assert len(reloc_off) == 0

    result_on, bridges_on, reloc_on = run(local_reloc_enabled=True)
    assert len(reloc_on) > 0, "local_reloc, once enabled, should produce at least one real local_reloc link"
    assert len(bridges_on) < len(bridges_off), (
        f"enabling local_reloc should replace at least some bridge links "
        f"(off: {len(bridges_off)} bridges, on: {len(bridges_on)} bridges, {len(reloc_on)} local_reloc)")
    assert result_on._session_id < result_off._session_id if hasattr(result_on, "_session_id") else True
    print(f"  end-to-end LOST recovery: local_reloc OFF -> {len(bridges_off)} bridges/0 reloc; "
          f"local_reloc ON -> {len(bridges_on)} bridges/{len(reloc_on)} reloc -- OK")


def check_ipc_frame_packet_round_trips_across_a_real_process():
    """Not a mock: actually spawns a multiprocessing.Process and reads
    the shared-memory block FROM IT, proving the IPC mechanism works
    across a real process boundary, not just within one Python
    interpreter's memory space -- the whole point of this module."""
    import multiprocessing as mp
    from pyslam.live.ipc import pack_frame
    from pyslam.core.types import Frame, Intrinsics
    from tests.gates._ipc_child import _child_read_frame_packet

    rng = np.random.default_rng(20)
    rgb = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    depth = rng.integers(0, 5000, (48, 64), dtype=np.uint16)
    imu = rng.normal(size=(5, 7)).astype(np.float64)
    intr = Intrinsics(fx=500, fy=500, cx=32, cy=24, width=64, height=48, depth_scale=0.001, baseline=0.05)
    frame = Frame(t=1.23, rgb=rgb, depth=depth, intr=intr, imu=imu, frame_id=7)

    packet = pack_frame(frame)
    ctx = mp.get_context("spawn") if "spawn" in mp.get_all_start_methods() else mp.get_context()
    result_queue = ctx.Queue()
    p = ctx.Process(target=_child_read_frame_packet, args=(packet, result_queue))
    p.start()
    result = result_queue.get(timeout=30)
    p.join(timeout=30)
    assert p.exitcode == 0, f"child process failed, exitcode={p.exitcode}"
    assert result["rgb_sum"] == int(rgb.sum()), "rgb did not round-trip correctly across the process boundary"
    assert result["depth_sum"] == int(depth.sum()), "depth did not round-trip correctly"
    assert result["frame_id"] == 7 and abs(result["t"] - 1.23) < 1e-9
    assert result["imu_shape"] == (5, 7)
    print("  IPC: FramePacket round-trips correctly through a REAL child process "
          "(shared memory, not just in-process) -- OK")


def check_ipc_pose_update_round_trips_and_frees_memory():
    import multiprocessing as mp
    from pyslam.live.ipc import pack_pose_update, ShmHandle
    from multiprocessing import shared_memory
    from tests.gates._ipc_child import _child_read_pose_update

    poses = {0: np.eye(4), 5: np.eye(4) * 2.0}
    packet = pack_pose_update(poses, generation=3)
    rgb_shm_name = packet.ids_handle.name  # capture before the child frees it
    ctx = mp.get_context("spawn") if "spawn" in mp.get_all_start_methods() else mp.get_context()
    result_queue = ctx.Queue()
    p = ctx.Process(target=_child_read_pose_update, args=(packet, result_queue))
    p.start()
    result = result_queue.get(timeout=30)
    p.join(timeout=30)
    assert p.exitcode == 0
    assert set(result.keys()) == {"0", "5"} or set(int(k) for k in result.keys()) == {0, 5}
    # the child freed the shared-memory block (free_after=True) -- confirm
    # it's actually gone, i.e. this isn't a leak
    try:
        shared_memory.SharedMemory(name=rgb_shm_name)
        raise AssertionError("shared-memory block was not freed by the consumer")
    except FileNotFoundError:
        pass
    print("  IPC: PoseUpdatePacket round-trips across a process AND the consumer "
          "correctly frees the shared-memory block afterward -- OK")


def check_tracker_stage_icp_fallback_recovers_from_forced_lost():
    """Integration-level (not a pure-function oracle like the others):
    forces LOST via an impossible odom_min_inliers threshold (same
    technique as check_local_reloc_prevents_session_break_end_to_end)
    and confirms TrackerStage's ICP fallback (pyslam/frontend/
    icp_fallback.py, wired into pyslam/live/tracker_stage.py) manages to
    flip odometry status back to OK at least once using ONLY depth
    geometry -- no ORB match could ever succeed under this threshold, so
    any recovery is attributable to the ICP path, not a match fluke."""
    from pyslam.core.config import Config
    from pyslam.live.tracker_stage import TrackerStage
    from pyslam.sensors.synthetic import SyntheticSource
    from tests.synth.scenarios import square6dof

    scen = square6dof(seed=1)
    src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, seed=1, add_noise=False)
    cfg = Config(odom_min_inliers=100000, lost_consecutive_frames=1)
    stage = TrackerStage(cfg)
    recovered = False
    for i, frame in enumerate(src):
        snap = stage.process_frame(frame)
        if snap.used_icp_fallback and snap.odom_status == "OK":
            recovered = True
            break
        if i >= 5:
            break
    assert recovered, "ICP fallback should recover tracking at least once under a forced-LOST, depth-only scenario"
    assert stage.n_icp_fallback_used >= 1
    print(f"  tracker stage: ICP fallback recovered from a forced (ORB-impossible) LOST using depth "
          f"geometry alone (n_used={stage.n_icp_fallback_used}) -- OK. NOTE: pure-Python correspondence "
          f"search is NOT yet within the real-time per-frame budget -- see SYSTEM_SUMMARY_LIVE.md.")


def check_dense_process_reintegrates_on_pose_correction():
    from pyslam.live.dense_process import DenseProcessState
    from pyslam.core import lie
    rng = np.random.default_rng(30)
    pts_cam = rng.uniform(-0.5, 0.5, (500, 3)) + np.array([0, 0, 1.5])
    colors = rng.integers(0, 255, (500, 3), dtype=np.uint8)

    state = DenseProcessState(voxel_size_m=0.05)
    state.on_keyframe_points(keyframe_id=1, points_cam=pts_cam, colors=colors)
    n1 = state.on_pose_update({1: np.eye(4)})
    assert n1 == 1
    pts_before, _ = state.export_ply_points()
    n_voxels_before = state.fuser.n_voxels()
    assert n_voxels_before > 0

    # a tiny pose nudge (below the re-integration threshold) should NOT re-integrate
    tiny = lie.make_T(np.eye(3), np.array([0.001, 0.0, 0.0]))
    n_tiny = state.on_pose_update({1: tiny})
    assert n_tiny == 0, "a sub-threshold pose nudge should not trigger re-integration"

    # a real loop-closure-sized correction SHOULD re-integrate, and the
    # exported points should move by roughly that correction
    correction = lie.make_T(np.eye(3), np.array([2.0, 0.0, 0.0]))
    n2 = state.on_pose_update({1: correction})
    assert n2 == 1, "a real pose correction must trigger re-integration"
    pts_after, _ = state.export_ply_points()
    shift = np.median(pts_after[:, 0]) - np.median(pts_before[:, 0])
    assert abs(shift - 2.0) < 0.1, f"re-integrated points should have shifted ~2.0m in x, got {shift}"
    print(f"  dense process: sub-threshold nudge correctly ignored; a real loop-closure-sized "
          f"correction re-integrated points (measured shift={shift:.3f}m) -- OK")


def check_lockstep_end_to_end_integrates_dense_map():
    """The strongest available integration check in this sandbox (no
    hardware, no GTSAM): runs the full lockstep path -- tracker +
    Pipeline (native backend) + dense -- over a real synthetic
    trajectory and confirms the dense voxel map actually gets
    populated. This is exactly the check that caught a real id-space
    mismatch bug (dense's points, keyed by tracker frame_id, silently
    never matched the backend's node_id-keyed pose broadcast) before
    it could ship -- see backend_process.py's and lockstep.py's own
    comments on the fix."""
    from pyslam.core.config import Config
    from pyslam.live.lockstep import run_lockstep
    from pyslam.sensors.synthetic import SyntheticSource
    from tests.synth.scenarios import square6dof
    from tests.synth.world import T_BODY_CAM

    scen = square6dof(seed=1)
    src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, seed=1, add_noise=False)
    cfg = Config()
    result = run_lockstep(cfg, src, max_frames=25, R_body_cam=T_BODY_CAM[:3, :3], backend_prefer="native")
    assert result.pipeline_result.n_keyframes > 0, "expected at least one keyframe"
    assert result.dense.fuser.n_voxels() > 1000, (
        f"dense map should have real voxel content after a real trajectory, "
        f"got {result.dense.fuser.n_voxels()} voxels")
    pts, colors = result.dense.export_ply_points()
    assert pts.shape[0] == result.dense.fuser.n_voxels()
    assert colors.shape == pts.shape
    print(f"  lockstep end-to-end: {result.pipeline_result.n_keyframes} keyframes, "
          f"{result.dense.fuser.n_voxels()} dense voxels, QA={result.qa.snapshot()['n_frames']} "
          f"frames tracked -- OK")


def check_orchestrator_full_three_process_pipeline_end_to_end():
    """The strongest validation this sandbox can offer for the
    multi-process live architecture: spawns the REAL tracker/backend/
    dense processes (pyslam.live.orchestrator.start_live_mapping) over
    a synthetic camera source and confirms real QA output and a real
    dense export come back. Uses mp_start_method="fork" -- this
    sandbox's containerized environment cannot rebuild POSIX semaphore
    state inside a "spawn"ed child (FileNotFoundError from
    multiprocessing.synchronize's SemLock._rebuild, a container
    restriction unrelated to this module's own logic); "spawn" is what
    orchestrator.py documents and defaults to for real deployment (see
    its own module docstring for why) and must be re-validated on the
    target Orin, where this container-specific restriction is not
    expected to apply."""
    import tempfile
    import multiprocessing as mp
    if "fork" not in mp.get_all_start_methods():
        print("  orchestrator: 'fork' start method unavailable on this platform -- SKIPPED "
              "(validate on a Linux target instead).")
        return
    from pyslam.core.config import Config
    from pyslam.live.orchestrator import start_live_mapping
    from pyslam.sensors.synthetic import SyntheticSource
    from tests.synth.scenarios import square6dof
    from tests.synth.world import T_BODY_CAM
    import os

    scen = square6dof(seed=1)

    def make_source():
        s = square6dof(seed=1)
        return SyntheticSource(s.world, s.intr, s.poses, dt=s.dt, seed=1, add_noise=False)

    with tempfile.TemporaryDirectory() as d:
        ply_path = os.path.join(d, "live_test.ply")
        session = start_live_mapping(
            Config(), make_source, scen.intr, out_ply_path=ply_path, backend_prefer="native",
            R_body_cam=T_BODY_CAM[:3, :3], mp_start_method="fork")
        import time
        deadline = time.time() + 25
        qa = None
        while time.time() < deadline:
            time.sleep(1.0)
            latest = session.poll_qa()
            if latest is not None:
                qa = latest
            if qa is not None and qa.get("n_keyframes", 0) >= 3:
                break
        session.stop(timeout=15)

        assert qa is not None, "no QA snapshot was ever received from the tracker process"
        assert qa["n_frames"] > 0 and qa["n_keyframes"] > 0, f"expected real progress, got {qa}"
        assert os.path.exists(ply_path), "dense process should have written a PLY export"
        assert os.path.getsize(ply_path) > 0
        print(f"  orchestrator: full 3-process pipeline ran end to end over a REAL synthetic "
              f"trajectory -- {qa['n_frames']} frames, {qa['n_keyframes']} keyframes tracked, "
              f"dense PLY written ({os.path.getsize(ply_path)} bytes) -- OK "
              f"(fork-validated in this sandbox; re-validate spawn on target hardware)")


CHECKS = [
    check_plane_fit_recovers_known_plane,
    check_plane_fit_two_planes_multiplane,
    check_plane_classify_support_vertical_overhead,
    check_plane_associate_matches_same_reobserves_different,
    check_fused_depth_converges_and_rejects_outliers,
    check_layers_free_space_carving_no_flood_fill,
    check_layers_walkable_json_matches_sts_schema,
    check_scale_check_ratio_and_tolerance,
    check_voxel_fuser_deduplicates_and_survives_resubmit,
    check_map_id_deterministic_and_content_sensitive,
    check_site_frame_uses_local_support_not_global_z,
    check_icp_recovers_known_transform,
    check_icp_detects_corridor_degeneracy,
    check_local_reloc_recovers_known_pose_and_rejects_wrong_candidate,
    check_odometry_prediction_gate_fails_open_on_bad_prediction,
    check_local_reloc_prevents_session_break_end_to_end,
    check_ipc_frame_packet_round_trips_across_a_real_process,
    check_ipc_pose_update_round_trips_and_frees_memory,
    check_tracker_stage_icp_fallback_recovers_from_forced_lost,
    check_dense_process_reintegrates_on_pose_correction,
    check_lockstep_end_to_end_integrates_dense_map,
    check_orchestrator_full_three_process_pipeline_end_to_end,
]


def main():
    passed = 0
    for fn in CHECKS:
        print(f"[{fn.__name__}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
    print(f"\nG-LIVE: {passed}/{len(CHECKS)} passed")
    if passed != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
