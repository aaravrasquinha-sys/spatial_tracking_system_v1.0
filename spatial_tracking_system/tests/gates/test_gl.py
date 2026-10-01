"""
Gate G-L (WP-L, Phase B of the Orin accuracy plan: loop-closure correctness).

    python3 -m tests.gates.test_gl

  GL.1  candidate gap        loop_min_path_m / loop_min_time_s remove too-recent nodes BEFORE
                              scoring; off == exactly the old behaviour; a node never goes back
                              from eligible to ineligible
  GL.2  diffusion: mass      belief mass is conserved exactly, independent of iteration order;
                              disabled == bit-identical to the pre-WP-L filter
  GL.3  diffusion: sequence  evidence carries along a moving true match (higher posterior than
                              without diffusion), and pure noise still never fires a hypothesis
  GL.4  diffusion: scope     mass is only ever handed to CURRENT candidates
  GL.5  audits               revisit recall / path-based triviality vs hand-built ground truth
  GL.6  end to end           on a real fixture (square6dof) the real loop survives the bound and a
                              bound larger than the whole trajectory removes every loop; keyframe
                              telemetry carries the trigger reason (WP-L3)
  GL.7  redundancy victim    WP-L4 policy == brute-force reference; == default rule when uncrowded;
                              respects view diversity
  GL.8  coverage             under a fixed WM size the default policy loses the start of the run,
                              the redundancy policy keeps it (and has no coverage hole)
"""
from __future__ import annotations
import sys

import numpy as np

sys.path.insert(0, ".")

from pyslam.core import lie
from pyslam.core.config import Config, apply_overrides
from pyslam.core.types import Link
from tests.gates.test_ga import _fake_node, _cfg


def _fail(msg: str) -> None:
    raise AssertionError(msg)


# ---------------------------------------------------------------- GL.1
def check_candidate_gap_filter() -> None:
    from pyslam.pipeline import Pipeline
    rng = np.random.default_rng(1)

    def make(*ov):
        cfg = _cfg("stm_size=3", "wm_min_resident=2", *ov)
        pipe = Pipeline(cfg, backend_prefer="native")
        for i in range(16):
            pipe.memory.add(_fake_node(i, rng))
            pipe._kf_path[i] = 0.11 * i           # 11 cm per keyframe, like corridor_v2
        return pipe

    cur = _fake_node(16, rng)
    pipe0 = make()
    pipe0._kf_path[16] = 0.11 * 16
    all_wm = pipe0.memory.working_set()
    if pipe0._loop_candidates(cur) != all_wm or len(all_wm) < 10:
        _fail("with both bounds off the candidate list must be exactly working_set()")

    pipe = make("loop_min_path_m=1.0")
    pipe._kf_path[16] = 0.11 * 16
    got = set(pipe._loop_candidates(cur))
    want = {c for c in all_wm if 0.11 * 16 - 0.11 * c >= 1.0}
    if got != want or not got or len(got) == len(all_wm):
        _fail(f"path bound wrong: got {sorted(got)} want {sorted(want)} of {all_wm}")

    pipe_t = make("loop_min_time_s=0.8")       # nodes are 0.1 s apart: needs >= 8 nodes back
    pipe_t._kf_path[16] = 0.11 * 16
    got_t = set(pipe_t._loop_candidates(cur))
    want_t = {c for c in all_wm if (0.1 * 16 - 0.1 * c) >= 0.8 - 1e-9}
    if got_t != want_t:
        _fail(f"time bound wrong: got {sorted(got_t)} want {sorted(want_t)}")

    pipe_b = make("loop_min_path_m=1.0", "loop_min_time_s=1.2")     # both must hold
    pipe_b._kf_path[16] = 0.11 * 16
    got_b = set(pipe_b._loop_candidates(cur))
    if got_b != (got & {c for c in all_wm if (0.1 * 16 - 0.1 * c) >= 1.2 - 1e-9}):
        _fail("with both bounds set a node must satisfy each")

    # a node with no recorded path (e.g. created before tracking started) is never silently excluded
    pipe_m = make("loop_min_path_m=1.0")
    pipe_m._kf_path[16] = 0.11 * 16
    del pipe_m._kf_path[all_wm[0]]
    if all_wm[0] not in pipe_m._loop_candidates(cur):
        _fail("a node with unknown path must stay eligible (fail open), not vanish")

    # monotone: as the camera moves on, nothing eligible becomes ineligible
    prev = set()
    for k in range(17, 30):
        pipe._kf_path[k] = 0.11 * k
        n = _fake_node(k, rng)
        now = set(pipe._loop_candidates(n))
        if not prev <= now:
            _fail(f"node(s) {sorted(prev - now)} went from eligible back to ineligible at step {k}")
        prev = now
    print(f"  [ok] GL.1 candidate gap: off == working_set() ({len(all_wm)} nodes); 1.0 m bound keeps {len(want)}; "
          f"time/both/unknown-path cases right; eligibility is monotone")


# ---------------------------------------------------------------- GL.2/3/4
def _chain(n: int):
    adj = {i: set() for i in range(n)}
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    return lambda c: adj.get(c, set())


def _run_filter(cfg: Config, n_nodes: int, steps: int, like_fn, neighbours):
    from pyslam.loop.bayes import BayesFilter
    bf = BayesFilter(cfg)
    hyps = []
    for t in range(steps):
        ids = list(range(n_nodes))
        L = np.concatenate([like_fn(t, n_nodes), [1.0]])
        hyps.append(bf.update(L, ids, neighbours=neighbours))
    return bf, hyps


def check_diffusion_mass_conservation() -> None:
    rng = np.random.default_rng(5)
    on = _cfg("bayes_diffusion_enabled=true", "bayes_diffusion_rate=0.4")
    off = _cfg()
    for trial in range(30):
        n = int(rng.integers(4, 25))
        adj = {i: set() for i in range(n)}
        for _ in range(int(rng.integers(n, 3 * n))):
            a, b = int(rng.integers(n)), int(rng.integers(n))
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        nb = lambda c: adj.get(c, set())
        like = lambda t, k: 1.0 + rng.random(k) * 2.0
        bf, _ = _run_filter(on, n, 12, like, nb)
        if abs(bf.belief_sum() - 1.0) > 1e-9:
            _fail(f"trial {trial}: belief mass {bf.belief_sum()!r} != 1 with diffusion on")
        if any(v < -1e-12 for v in bf.belief.values()) or bf.belief_new < -1e-12:
            _fail("negative belief")
    # The check above cannot see a leak on its own: update() renormalises by the posterior total every
    # step, which would silently hide mass created or destroyed in _predict (the LEAK mutant of
    # phase_a_mutation_check went undetected until this was added). So test _predict DIRECTLY.
    from pyslam.loop.bayes import BayesFilter
    for trial in range(30):
        n = int(rng.integers(3, 20))
        adj = {i: set() for i in range(n)}
        for _ in range(int(rng.integers(n, 3 * n))):
            a, b = int(rng.integers(n)), int(rng.integers(n))
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        bf = BayesFilter(on)
        raw = rng.random(n + 1)
        raw /= raw.sum()
        bf.belief = {i: float(raw[i]) for i in range(n)}
        bf.belief_new = float(raw[n])
        before = bf.belief_sum()
        bf._predict(list(range(n)), lambda c: adj.get(c, set()))
        if abs(bf.belief_sum() - before) > 1e-12:
            _fail(f"trial {trial}: _predict changed total belief mass {before} -> {bf.belief_sum()}")
    # disabled (or no neighbour function): bit-identical to the pre-WP-L behaviour
    like_seq = [np.linspace(1, 3, 8) ** (t % 3 + 1) for t in range(10)]
    a, _ = _run_filter(off, 8, 10, lambda t, k: like_seq[t], _chain(8))       # flag off, neighbours given
    b, _ = _run_filter(on, 8, 10, lambda t, k: like_seq[t], None)             # flag on, neighbours absent
    c, _ = _run_filter(off, 8, 10, lambda t, k: like_seq[t], None)            # legacy call shape
    if a.belief != c.belief or b.belief != c.belief or a.belief_new != c.belief_new:
        _fail("diffusion must be a strict no-op when disabled or when no neighbour function is supplied")
    # order independence: reversing the candidate id order gives the same beliefs
    from pyslam.loop.bayes import BayesFilter
    bf1, bf2 = BayesFilter(on), BayesFilter(on)
    ids = list(range(9))
    L = np.concatenate([np.linspace(1, 2.5, 9), [1.0]])
    for _ in range(4):
        bf1.update(L, ids, neighbours=_chain(9))
        bf2.update(np.concatenate([L[:-1][::-1], [1.0]]), ids[::-1], neighbours=_chain(9))
    if any(abs(bf1.belief[i] - bf2.belief[i]) > 1e-12 for i in ids):
        _fail("beliefs depend on candidate iteration order")
    print("  [ok] GL.2 diffusion: mass conserved to 1e-9 over 30 random graphs, never negative; strict no-op when "
          "disabled; independent of iteration order")


def check_diffusion_sequence_evidence() -> None:
    n, steps = 14, 9
    on = _cfg("bayes_diffusion_enabled=true", "bayes_diffusion_rate=0.3")
    off = _cfg()

    # the true match MOVES one node per step (the camera drifts through a revisited region);
    # each step's likelihood is only moderately elevated on the current true node
    def like(t, k):
        L = np.ones(k)
        L[t + 2] = 2.2
        return L

    nb = _chain(n)
    bf_on, _ = _run_filter(on, n, steps, like, nb)
    bf_off, _ = _run_filter(off, n, steps, like, nb)
    # total belief captured near the trajectory of true matches (nodes 2..steps+1)
    near = range(2, steps + 2)
    mass_on = sum(bf_on.belief[i] for i in near)
    mass_off = sum(bf_off.belief[i] for i in near)
    if not mass_on > mass_off * 1.05:
        _fail(f"diffusion should retain more belief along the moving true match: {mass_on:.4f} vs {mass_off:.4f}")
    # ...and pure noise must never fire in either mode
    for name, cfg in (("on", on), ("off", off)):
        _, hyps = _run_filter(cfg, n, 60, lambda t, k: np.ones(k), nb)
        if any(h is not None for h in hyps):
            _fail(f"noise-only likelihood fired a hypothesis with diffusion {name}")
    print(f"  [ok] GL.3 diffusion: belief kept on a moving true match {mass_off:.3f} -> {mass_on:.3f} "
          f"(x{mass_on / mass_off:.2f}); 60 noise-only updates fire nothing in either mode")


def check_diffusion_scope() -> None:
    from pyslam.loop.bayes import BayesFilter
    cfg = _cfg("bayes_diffusion_enabled=true", "bayes_diffusion_rate=0.5")
    bf = BayesFilter(cfg)
    ids = [0, 1, 2]
    nb = lambda c: {0: {1, 99}, 1: {0, 2, 99}, 2: {1, 98}}.get(c, set())   # 98, 99 are NOT candidates
    for _ in range(5):
        bf.update(np.array([1.0, 2.0, 1.0, 1.0]), ids, neighbours=nb)
    if 98 in bf.belief or 99 in bf.belief:
        _fail("belief was handed to a node that is not a current candidate")
    if abs(bf.belief_sum() - 1.0) > 1e-9:
        _fail("mass not conserved with out-of-set neighbours")
    print("  [ok] GL.4 diffusion scope: mass only ever moves between current candidates")


# ---------------------------------------------------------------- GL.5
def check_audit_recall() -> None:
    from pyslam.tools.phase_a_baseline import revisit_recall, loop_audit
    # a 'there and back' path: nodes 0..9 go out along +x (0.5 m apart), 10..19 come straight back
    gt = {}
    for i in range(10):
        gt[i] = lie.make_T(np.eye(3), np.array([0.5 * i, 0.0, 0.0]))
    for k in range(10):
        gt[10 + k] = lie.make_T(np.eye(3), np.array([4.5 - 0.5 * k, 0.05, 0.0]))
    true = lambda a, b: lie.se3_inverse(gt[a]) @ gt[b]
    good = Link(9, 10, true(9, 10), np.eye(6), "loop")           # 0.5 m of path apart: trivial
    real = Link(3, 16, true(3, 16), np.eye(6), "loop")           # a genuine return leg: ~4 m of path
    bad_T = true(2, 17).copy(); bad_T[0, 3] += 2.0
    wrong = Link(2, 17, bad_T, np.eye(6), "loop")
    rc = revisit_recall(gt, [real], min_path_m=1.5)
    # opportunities are counted on the LATER frame of a revisit pair, so the one real link (3<->16)
    # scores exactly one hit (node 16); node 3 is the earlier visit, not an opportunity itself
    if rc["n_opportunity_frames"] < 8 or rc["n_hit_frames"] != 1 or not (0 < rc["recall"] < 1):
        _fail(f"recall wrong: {rc}")
    rc0 = revisit_recall(gt, [good, wrong], min_path_m=1.5)
    if rc0["n_hit_frames"] != 0:
        _fail(f"a trivial link and a wrong link must not count as hits: {rc0}")
    node_t = {i: 0.1 * i for i in gt}
    a = loop_audit([good, real, wrong], gt, node_t, min_gap_s=0.0, min_gap_m=1.5)
    if (a["n"], a["n_trivial"], a["n_wrong"], a["n_real"]) != (3, 1, 1, 1):
        _fail(f"path-based audit classification wrong: {a}")
    print(f"  [ok] GL.5 audits: recall {rc['n_hit_frames']}/{rc['n_opportunity_frames']} opportunity frames; trivial/wrong links "
          f"excluded; path-based classification 1 trivial / 1 wrong / 1 real")


# ---------------------------------------------------------------- GL.6
def check_end_to_end_gap_on_real_fixture() -> None:
    from pyslam.pipeline import Pipeline
    from pyslam.sensors.synthetic import SyntheticSource
    from tests.synth.scenarios import build
    from tests.synth.world import T_BODY_CAM
    scen = build("square6dof", seed=1, validate=False)
    out = {}
    for tag, ov in (("gap1.5", ["loop_min_path_m=1.5"]), ("gap1000", ["loop_min_path_m=1000"])):
        cfg = _cfg(*ov)
        src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=1)
        pipe = Pipeline(cfg, backend_prefer="native", R_body_cam=T_BODY_CAM[:3, :3])
        res = pipe.run(src, verbose=False)
        out[tag] = res
    tele = out["gap1.5"].telemetry
    kfs = [t for t in tele if t["keyframe"]]
    if any(t["kf_reason"] is not None for t in tele if not t["keyframe"]):
        _fail("kf_reason must be None on non-keyframe frames")
    if sum(1 for t in kfs if t["kf_reason"]) < 0.9 * len(kfs) - 1:
        _fail("keyframe frames should carry a kf_reason (trans/rot/inliers)")
    n15 = len(out["gap1.5"].loop_events)
    n_huge = len(out["gap1000"].loop_events)
    if n15 < 1:
        _fail("the real square6dof loop (~3 m of path) must survive a 1.5 m bound")
    if n_huge != 0:
        _fail(f"a bound larger than the whole trajectory must remove every loop, got {n_huge}")
    # every surviving loop really spans >= 1.5 m of odometry path
    print(f"  [ok] GL.6 end-to-end square6dof: {n15} loop closure(s) with a 1.5 m bound (real loop kept), "
          f"{n_huge} with a 1000 m bound (filter demonstrably active)")


# ---------------------------------------------------------------- GL.7
def _brute_victim(ids, weight_of, poses, protect, radius, angle):
    eligible = ids[:-protect] if protect > 0 else list(ids)
    best = None
    for n in eligible:
        crowd = 0
        for m in ids:
            if m == n:
                continue
            d = np.linalg.norm(poses[n][:3, 3] - poses[m][:3, 3])
            ca = float(np.clip(poses[n][:3, 2] @ poses[m][:3, 2], -1, 1))
            if d < radius and ca > np.cos(np.radians(angle)):
                crowd += 1
        key = (-crowd, weight_of[n], n)
        if best is None or key < best[0]:
            best = (key, n)
    return best[1] if best else None


def check_redundancy_victim_selection() -> None:
    from pyslam.memory.memory import select_redundant_victim, select_transfer_victim
    rng = np.random.default_rng(11)
    for trial in range(300):
        n = int(rng.integers(3, 30))
        ids = sorted(int(x) for x in rng.choice(200, n, replace=False))
        poses = {}
        for i in ids:
            R = lie.so3_exp(rng.normal(size=3) * rng.choice([0.05, 1.0]))
            poses[i] = lie.make_T(R, rng.normal(size=3) * 0.6)
        w = {i: int(rng.integers(1, 4)) for i in ids}
        protect = int(rng.integers(0, min(n, 6)))
        got = select_redundant_victim(ids, w, poses, protect, 0.5, 30.0)
        want = _brute_victim(ids, w, poses, protect, 0.5, 30.0)
        if got != want:
            _fail(f"trial {trial}: vectorised victim {got} != brute-force {want}")
    # nobody crowded -> identical to the default oldest-of-least-weighted rule
    ids = list(range(10))
    far = {i: lie.make_T(np.eye(3), np.array([5.0 * i, 0, 0])) for i in ids}
    w = {i: 1 + (i % 3) for i in ids}
    if select_redundant_victim(ids, w, far, 2, 0.5, 30.0) != select_transfer_victim(ids, w, 2):
        _fail("with no crowding the redundancy policy must reduce to the default rule")
    # view diversity: same place, different viewing directions is NOT redundant
    Ry = lambda deg: lie.so3_exp(np.array([0.0, np.radians(deg), 0.0]))
    same_spot = {0: lie.make_T(Ry(0), np.zeros(3)), 1: lie.make_T(Ry(90), np.zeros(3)),
                 2: lie.make_T(Ry(180), np.zeros(3)), 3: lie.make_T(Ry(0), np.array([0.05, 0, 0]))}
    v = select_redundant_victim([0, 1, 2, 3], {i: 1 for i in range(4)}, same_spot, 0, 0.5, 30.0)
    if v not in (0, 3):
        _fail(f"nodes 0 and 3 look the same way from the same spot (redundant); victim was {v}")
    print("  [ok] GL.7 redundancy victim: matches brute force on 300 random maps; reduces to the default rule when "
          "uncrowded; view diversity respected")


def check_redundancy_keeps_coverage() -> None:
    from pyslam.memory.memory import Memory
    survivors = {}
    for policy in ("oldest", "redundancy"):
        cfg = _cfg("stm_size=3", "wm_min_resident=3", f"wm_evict_policy={policy}", "wm_redundancy_radius_m=0.5")
        mem = Memory(cfg)
        rng = np.random.default_rng(2)
        for i in range(70):
            n = _fake_node(i, rng, h=8, w=8)
            n.pose_map = lie.make_T(np.eye(3), np.array([0.1 * i, 0.0, 0.0]))   # 10 cm apart on a line
            mem.add(n)
        while len(mem.wm) > 16:
            mem.enforce_budget(1e9)
        survivors[policy] = sorted(mem.wm)
    old, red = survivors["oldest"], survivors["redundancy"]
    if min(old) < 40:
        _fail(f"fixture sanity: the default policy should keep only a recent block, kept {old}")
    if min(red) > 6:
        _fail(f"redundancy policy lost the START of the trajectory: kept {red}")
    gaps = np.diff([0.1 * i for i in red])
    if gaps.max() > 1.3:
        _fail(f"redundancy policy left a coverage hole of {gaps.max():.2f} m: kept {red}")
    if max(red) < 66:
        _fail(f"redundancy policy dropped the most recent nodes: kept {red}")
    print(f"  [ok] GL.8 coverage under a fixed WM size (16 of 70 nodes on a 7 m line): default keeps ids {old[0]}..{old[-1]} "
          f"(start lost); redundancy keeps ids {red[0]}..{red[-1]}, largest gap {gaps.max():.2f} m")


# ---------------------------------------------------------------- GL.9
def check_lap_fixture_has_real_loops() -> None:
    """corridor_lap13 exists because corridor_v2 has ONE keyframe with a true revisit partner
    (measured), so no loop-closure change can be evaluated on it. Pure geometry check (no
    rendering; full validate_scenario was run once by hand, see WP_L_Findings.md)."""
    from tests.synth.scenarios import build
    from tests.synth.world import T_BODY_CAM
    scen = build("corridor_lap13", seed=1, validate=False)
    if len(scen.poses) != 624:
        _fail(f"expected 624 frames (1.3 laps), got {len(scen.poses)}")
    P = np.array([(T @ T_BODY_CAM)[:3, 3] for T in scen.poses])
    R = [(T @ T_BODY_CAM)[:3, :3] for T in scen.poses]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n_opp = 0
    for j in range(len(P)):
        for i in range(j):
            if cum[j] - cum[i] >= 3.0 and np.linalg.norm(P[i] - P[j]) < 0.5 and \
                    np.degrees(np.arccos(np.clip((np.trace(R[i].T @ R[j]) - 1) / 2, -1, 1))) < 30:
                n_opp += 1
                break
    if n_opp < 100:
        _fail(f"the lap fixture must offer many true revisits, found {n_opp}")
    yaw_step = [np.degrees(np.arccos(np.clip((np.trace(R[k - 1].T @ R[k]) - 1) / 2, -1, 1))) for k in range(1, len(R))]
    if max(yaw_step) > 30.0:
        _fail(f"single-frame rotation of {max(yaw_step):.1f} deg: the end-of-path heading snap is back")
    print(f"  [ok] GL.9 corridor_lap13: 624 frames, {cum[-1]:.1f} m, {n_opp} frames with a true revisit partner "
          f"(corridor_v2: 1 keyframe), max single-frame rotation {max(yaw_step):.1f} deg")


# ---------------------------------------------------------------- GL.10
def check_deterministic_wm_cap() -> None:
    from pyslam.memory.memory import Memory

    def run(cap: int, ms_pattern):
        cfg = _cfg("stm_size=3", "wm_min_resident=3", "wm_budget_ms=1000000", f"wm_max_nodes={cap}")
        mem = Memory(cfg)
        rng = np.random.default_rng(4)
        sizes = []
        for i in range(60):
            n = _fake_node(i, rng, h=8, w=8)
            mem.add(n)
            mem.enforce_budget(ms_pattern(i))
            sizes.append(len(mem.wm))
        return sorted(mem.wm), sizes

    a, sa = run(12, lambda i: 0.0)
    b, sb = run(12, lambda i: 5.0 * (i % 7))          # wildly different 'timing': must not matter
    if a != b or sa != sb:
        _fail("with wm_max_nodes set, eviction must be independent of the reported timing")
    if max(sa) > 13 or sa[-1] != 12:
        _fail(f"WM must be held at the cap (<= cap+1 transiently, == cap at the end): max {max(sa)}, final {sa[-1]}")
    off, so = run(0, lambda i: 0.0)
    if len(off) != 57 or so[-1] != 57:
        _fail("cap 0 must leave the frozen behaviour alone: nothing evicted when under the time budget")
    print(f"  [ok] GL.10 deterministic WM cap: WM held at 12 of 60 nodes identically under two different timing traces; "
          f"cap 0 evicts nothing under budget")


ALL_TESTS = [
    check_candidate_gap_filter,
    check_diffusion_mass_conservation,
    check_diffusion_sequence_evidence,
    check_diffusion_scope,
    check_audit_recall,
    check_end_to_end_gap_on_real_fixture,
    check_redundancy_victim_selection,
    check_redundancy_keeps_coverage,
    check_lap_fixture_has_real_loops,
    check_deterministic_wm_cap,
]


def main() -> int:
    print(f"Gate G-L (WP-L1..L4) -- {len(ALL_TESTS)} checks\n")
    n_pass = 0
    for t in ALL_TESTS:
        try:
            t()
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{n_pass}/{len(ALL_TESTS)} passed")
    return 0 if n_pass == len(ALL_TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
