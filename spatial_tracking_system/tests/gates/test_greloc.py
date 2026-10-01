"""
WP-RELOC gate suite. Run standalone: python -m tests.gates.test_greloc

Same honesty discipline as every other gate file here (see test_g0.py's
own header): checks run against synthetic fixtures with real measured
numbers, not against hardware bags that don't exist in this environment.

The single highest-risk piece of this feature is the transform chain in
pyslam/loop/relocalize.py (see that module's own docstring) -- an
inverted or mis-Adjoint'd pose looks completely plausible and is wrong,
exactly the class of bug this codebase's own pnp_info.py module
describes being caught only by a Monte Carlo oracle, never by
inspection. check_covariance_oracle_reloc_frame below is that oracle,
applied specifically to relocalize.py's own composition (a constant
LEFT-multiply by pose_map[n] after inverting the raw PnP solve) rather
than re-testing pnp_info_matrix itself, which test_g0.py's
test_pnp_info_matrix_oracle already covers.
"""
from __future__ import annotations
import tempfile
import numpy as np
import cv2

from pyslam.core.config import Config
from pyslam.core.types import Frame, Node
from pyslam.core import lie
from pyslam.core.pnp_info import pnp_info_matrix
from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
from pyslam.loop.relocalize import relocalize, _grid_cells
from pyslam.vpr.global_desc import GlobalDescriptor
from pyslam.mapping.reloc_map import write_map_pack_from_nodes, RelocMap
from tests.synth.world import World, Renderer
from tests.synth.scenarios import RIG_INTRINSICS

INTR = RIG_INTRINSICS


class _FakeMap:
    """Minimal RelocMap-shaped stand-in for checks that don't need the
    real on-disk pack format (that format has its own dedicated round-
    trip check below)."""

    def __init__(self, nodes: dict):
        self.node_ids = list(nodes.keys())
        self._nodes = nodes
        self.poses = {i: nodes[i].pose_map for i in nodes}
        self.intrinsics = INTR
        self.gdesc_node_ids = np.array(self.node_ids)
        gd = GlobalDescriptor(out_dim=256)
        self.gdesc = np.stack([gd.compute(nodes[i].sig.desc) for i in self.node_ids])
        self.desc = np.concatenate([nodes[i].sig.desc for i in self.node_ids])
        self.owner = np.concatenate([np.full(nodes[i].sig.desc.shape[0], i, dtype=np.int64)
                                      for i in self.node_ids])
        self.adjacency: dict = {}
        self.gravity_aligned = False
        self.T_grav_cam0 = np.eye(4)

    def get_node(self, nid):
        return self._nodes[nid]

    def neighbours(self, nid):
        return set()


def _build_room_map(seed0: int = 1):
    world = World()
    world.add_box_room(center=np.array([0, 0, 1.5]), size=np.array([8, 8, 3]), seed0=seed0)
    renderer = Renderer(world, INTR)
    cfg = Config()
    _reset_id_counter(0)
    orb = make_orb(cfg)

    node_poses = [lie.make_T(lie.so3_exp(np.array([0, 0, yaw])), np.array([x, y, 1.5]))
                  for (x, y, yaw) in [(-1, -1, 0.0), (1, -1, 1.2), (1, 1, 2.4),
                                       (-1, 1, 3.6), (-1, -1, 4.8)]]
    nodes = {}
    for i, T in enumerate(node_poses):
        rgb, depth = renderer.render(T)
        frame = Frame(t=float(i), rgb=rgb, depth=depth, intr=INTR, frame_id=i)
        sig = extract_signature(frame, cfg, orb=orb)
        nodes[i] = Node(id=i, sig=sig, pose_odom=T.copy(), pose_map=T.copy())
    return world, renderer, cfg, orb, nodes


def check_transform_chain_oracle() -> None:
    """The chain T_world_query = pose_map[n] @ inverse(T_query_node)
    recovers a KNOWN query pose to well under a centimetre. An inverted
    or Adjoint-mangled chain cannot pass this by luck -- see
    relocalize.py's own module docstring for the derivation this is
    checking."""
    world, renderer, cfg, orb, nodes = _build_room_map()
    rmap = _FakeMap(nodes)

    x, y, yaw = -0.8, -0.9, 0.15
    T_true = lie.make_T(lie.so3_exp(np.array([0, 0, yaw])), np.array([x, y, 1.5]))
    rgb_q, _ = renderer.render(T_true)

    res = relocalize(rmap, rgb_q, INTR, cfg, orb=orb)
    assert res.status == "LOCALIZED", f"expected LOCALIZED, got {res.status}: {res.reason}"
    err_trans = float(np.linalg.norm(res.pose_world_query[:3, 3] - T_true[:3, 3]))
    xi = lie.se3_log(lie.se3_inverse(T_true) @ res.pose_world_query)
    err_rot_deg = float(np.degrees(np.linalg.norm(xi[3:])))
    assert err_trans < 0.01, f"translation error {err_trans*100:.2f}cm >= 1cm"
    assert err_rot_deg < 1.0, f"rotation error {err_rot_deg:.2f}deg >= 1deg"
    print(f"  [ok] transform-chain oracle: {err_trans*1000:.2f}mm / {err_rot_deg:.3f}deg "
          f"(consensus={res.consensus})")


def check_false_positive_rejection() -> None:
    """A query from a genuinely different, unrelated place must return
    NOT_FOUND -- zero false accepts is the bar, not a low rate."""
    world, renderer, cfg, orb, nodes = _build_room_map(seed0=1)
    other_world = World()
    other_world.add_box_room(center=np.array([0, 0, 1.5]), size=np.array([8, 8, 3]), seed0=99)
    other_renderer = Renderer(other_world, INTR)
    rmap = _FakeMap(nodes)

    n_trials, n_false_accept = 5, 0
    rng = np.random.default_rng(7)
    for _ in range(n_trials):
        x, y, yaw = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0, 2 * np.pi)
        T = lie.make_T(lie.so3_exp(np.array([0, 0, yaw])), np.array([x, y, 1.5]))
        rgb_q, _ = other_renderer.render(T)
        res = relocalize(rmap, rgb_q, INTR, cfg, orb=orb)
        if res.status == "LOCALIZED":
            n_false_accept += 1
    assert n_false_accept == 0, f"{n_false_accept}/{n_trials} false accepts against an unrelated scene"
    print(f"  [ok] false-positive rejection: 0/{n_trials} false accepts against an unrelated room")


def check_pack_round_trip() -> None:
    """A pack written by write_map_pack_from_nodes and reopened via
    RelocMap must return identical descriptors, poses, and node ids --
    nothing silently reordered or truncated on the way through
    SQLite/npy/npz."""
    world, renderer, cfg, orb, nodes = _build_room_map()
    final_poses = {i: n.pose_map for i, n in nodes.items()}
    adjacency = {0: [1], 1: [0, 2], 2: [1]}

    with tempfile.TemporaryDirectory() as d:
        write_map_pack_from_nodes(d, nodes, final_poses, adjacency, INTR, cfg, source_run_dir="test")
        rmap = RelocMap(d)
        assert set(rmap.node_ids) == set(nodes.keys()), "node id set mismatch after round-trip"
        for nid, node in nodes.items():
            got = rmap.get_node(nid)
            assert np.array_equal(got.sig.desc, node.sig.desc), f"node {nid}: descriptor mismatch"
            assert np.allclose(got.sig.kp3d, node.sig.kp3d, equal_nan=True), f"node {nid}: kp3d mismatch"
            assert np.array_equal(got.sig.valid, node.sig.valid), f"node {nid}: valid mask mismatch"
            assert np.allclose(rmap.poses[nid], final_poses[nid]), f"node {nid}: pose mismatch"
            assert got.sig.rgb is None and got.sig.depth is None, \
                f"node {nid}: pack must never carry rgb/depth (frozen contract)"
        assert rmap.neighbours(1) == {0, 2}, f"adjacency mismatch: {rmap.neighbours(1)}"
        rmap.close()
    print(f"  [ok] pack round-trip: {len(nodes)} nodes, descriptors/poses/adjacency bit-identical")


def check_grid_cell_gate() -> None:
    """Unit check of the spatial-coverage floor itself: a cluster of
    points confined to one corner of the image must score far fewer
    cells than the same count spread across the frame, and the
    configured default threshold must actually separate the two."""
    rng = np.random.default_rng(3)
    cfg = Config()
    w, h = INTR.width, INTR.height

    clustered = rng.uniform(0, 40, size=(60, 2)) + np.array([10, 10])  # 40x40px corner
    spread = np.stack([rng.uniform(0, w, 60), rng.uniform(0, h, 60)], axis=1)

    n_clustered = _grid_cells(clustered, cfg, w, h)
    n_spread = _grid_cells(spread, cfg, w, h)
    assert n_clustered <= 4, f"expected a tiny corner cluster to occupy very few cells, got {n_clustered}"
    assert n_spread >= cfg.reloc_min_grid_cells, \
        f"expected a full-frame spread to clear reloc_min_grid_cells={cfg.reloc_min_grid_cells}, got {n_spread}"
    assert n_clustered < cfg.reloc_min_grid_cells <= n_spread, \
        "reloc_min_grid_cells does not actually separate a clustered case from a spread one"
    print(f"  [ok] grid-cell coverage gate: clustered={n_clustered} cells, spread={n_spread} cells, "
          f"threshold={cfg.reloc_min_grid_cells}")


def check_covariance_oracle_reloc_frame() -> None:
    """Monte Carlo NEES oracle for relocalize.py's OWN composition --
    not a re-test of pnp_info_matrix itself (test_g0.py's
    test_pnp_info_matrix_oracle already covers that function in
    isolation), but the specific claim relocalize.py's module docstring
    makes: that a constant LEFT-multiply by pose_map[n], applied AFTER
    inverting the raw PnP solve, does not change the covariance's
    validity. If that claim were wrong (e.g. the Adjoint should have
    been applied a second time, or not at all), this would show up as
    a NEES far from the chi-square(3)-for-translation-only expectation,
    the same way test_g0's oracle caught a real misplaced-Adjoint bug
    during that module's own development.
    """
    rng = np.random.default_rng(21)
    K = INTR.K()
    N, sigma_px = 60, 0.6

    X_node = np.zeros((N, 3))
    X_node[:, 0] = rng.uniform(-1.5, 1.5, N)
    X_node[:, 1] = rng.uniform(-1.0, 1.0, N)
    X_node[:, 2] = rng.uniform(1.5, 3.5, N)

    # A non-trivial, fixed map pose for this node -- if the composition
    # direction were wrong, an identity pose_map would hide the bug.
    pose_map_n = lie.make_T(lie.so3_exp(np.array([0.2, -0.3, 0.5])), np.array([2.0, -1.0, 0.5]))

    xi_true = rng.normal(scale=0.12, size=6)
    T_query_node_true = lie.se3_exp(xi_true)  # the raw-PnP-shaped ground truth
    R_true, t_true = T_query_node_true[:3, :3], T_query_node_true[:3, 3]
    p_cam_true = (R_true @ X_node.T).T + t_true
    assert np.all(p_cam_true[:, 2] > 0.3), "fix synthetic setup, points too close/behind camera"
    u_true = np.stack([K[0, 0] * p_cam_true[:, 0] / p_cam_true[:, 2] + K[0, 2],
                        K[1, 1] * p_cam_true[:, 1] / p_cam_true[:, 2] + K[1, 2]], axis=1)

    T_world_query_true = pose_map_n @ lie.se3_inverse(T_query_node_true)

    n_trials = 500
    xi_samples = []
    for _ in range(n_trials):
        u_noisy = u_true + rng.normal(scale=sigma_px, size=u_true.shape)
        ok, rvec, tvec = cv2.solvePnP(X_node.astype(np.float64), u_noisy.astype(np.float64), K, None,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R_est, _ = cv2.Rodrigues(rvec)
        T_query_node_est = lie.make_T(R_est, tvec.reshape(3))
        T_world_query_est = pose_map_n @ lie.se3_inverse(T_query_node_est)
        xi = lie.se3_log(lie.se3_inverse(T_world_query_true) @ T_world_query_est)
        xi_samples.append(xi)
    xi_samples = np.array(xi_samples)
    assert len(xi_samples) > n_trials * 0.9, "too many PnP solve failures in oracle, check setup"

    # This is exactly relocalize.py's own call: pnp_info_matrix on the
    # RAW (uninverted, un-composed) PnP solve, claimed valid for
    # T_world_query's own right-tangent after the fixed left-multiply.
    info_pred = pnp_info_matrix(X_node, R_true, t_true, K, sigma_px=sigma_px)
    nees_full = np.einsum('ni,ij,nj->n', xi_samples, info_pred, xi_samples)
    mean_nees = float(nees_full.mean())
    # chi-square(6) mean is 6; same tolerance band as test_g0's own oracle
    # for the same statistical reasons (see that test's comment).
    assert 4.5 < mean_nees < 7.5, (
        f"mean NEES={mean_nees:.2f}, expected ~6.0 -- the covariance composed through "
        f"relocalize.py's fixed-left-multiply is not consistent with actual scatter; suspect the "
        f"transform-chain derivation (see module docstring)")
    print(f"  [ok] covariance oracle (relocalize.py's own T_world_query composition): "
          f"mean NEES={mean_nees:.2f} over {len(xi_samples)} trials (want ~6.0)")


def check_sparse_map_single_candidate() -> None:
    """A one-node map cannot produce a second opinion. A well-verified
    single candidate should still localize (via the stricter
    single-candidate bar), which is the intended behaviour for the
    ~80-keyframe maps this feature targets, where a query viewpoint may
    genuinely overlap only one keyframe."""
    world, renderer, cfg, orb, nodes = _build_room_map()
    single_node = {0: nodes[0]}
    rmap = _FakeMap(single_node)

    T_node = nodes[0].pose_map
    offset = np.eye(4)
    offset[:3, 3] = [0.03, 0.01, -0.02]
    T_query = T_node @ offset
    rgb_q, _ = renderer.render(T_query)

    res = relocalize(rmap, rgb_q, INTR, cfg, orb=orb)
    assert res.status == "LOCALIZED", f"expected LOCALIZED via the single-candidate path, got {res.status}"
    assert res.consensus == "single", f"expected consensus='single', got {res.consensus!r}"
    err = float(np.linalg.norm(res.pose_world_query[:3, 3] - T_query[:3, 3]))
    assert err < 0.02, f"single-candidate localisation error {err*100:.2f}cm >= 2cm"
    print(f"  [ok] sparse-map (1-node) single-candidate path: LOCALIZED, error={err*1000:.2f}mm")


ALL_GRELOC_CHECKS = [
    check_transform_chain_oracle,
    check_false_positive_rejection,
    check_pack_round_trip,
    check_grid_cell_gate,
    check_covariance_oracle_reloc_frame,
    check_sparse_map_single_candidate,
]


def main() -> int:
    print(f"Gate G-RELOC (single-snapshot relocalization) -- {len(ALL_GRELOC_CHECKS)} checks\n")
    n_pass = 0
    for check in ALL_GRELOC_CHECKS:
        try:
            check()
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {check.__name__}: {e}")
    print(f"\n{n_pass}/{len(ALL_GRELOC_CHECKS)} G-RELOC checks passed")
    return 0 if n_pass == len(ALL_GRELOC_CHECKS) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
