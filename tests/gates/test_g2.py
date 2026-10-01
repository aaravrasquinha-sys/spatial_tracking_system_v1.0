"""
WP-P2 gate G2 (architecture doc section 5, P2): memory management.
Run standalone: python -m tests.gates.test_g2

Checks implemented here (lighter-weight than the doc's 10^4-cycle spec,
but exercising the same properties, each against the real code paths,
not a mock):
  1. select_transfer_victim matches a brute-force reference over many
     randomised WM populations.
  2. STM never exceeds cfg.stm_size across a real synthetic run.
  3. audit_consistency() is clean (zero dangling refs) after a run with
     eviction pressure forced on.
  4. A simulated crash mid-batch-write leaves the LTM sqlite file
     openable, with every row committed before the crash intact and
     nothing beyond it -- the "SIGKILL mid-write" property, achieved
     here by injecting an exception after N of M puts rather than an
     actual OS-level kill (not reproducible from inside the test
     process), then reopening a FRESH LtmStore against the same path.
  5. Graph-neighbour retrieval: forcing an id's neighbour into LTM and
     then calling on_loop() on it brings the neighbour back into WM.
"""
from __future__ import annotations
import os
import random
import tempfile
import numpy as np

from pyslam.core.config import Config
from pyslam.core.types import Node, Signature
from pyslam.memory.memory import Memory, select_transfer_victim
from pyslam.memory.ltm_store import LtmStore
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from tests.synth.scenarios import build


def _brute_force_victim(wm_ids_ordered, weight_of, protect_recent):
    eligible = wm_ids_ordered[:-protect_recent] if protect_recent > 0 else list(wm_ids_ordered)
    if not eligible:
        return None
    best = None
    for nid in eligible:
        key = (weight_of[nid], nid)
        if best is None or key < best[0]:
            best = (key, nid)
    return best[1]


def check_victim_selection_matches_brute_force(n_trials: int = 500) -> None:
    rng = random.Random(0)
    for _ in range(n_trials):
        n = rng.randint(0, 30)
        ids = sorted(rng.sample(range(1000), n))
        weights = {i: rng.randint(1, 20) for i in ids}
        protect = rng.randint(0, 10)
        got = select_transfer_victim(ids, weights, protect)
        expect = _brute_force_victim(ids, weights, protect)
        assert got == expect, (
            f"MISMATCH: ids={ids} weights={weights} protect={protect} "
            f"-> select_transfer_victim={got}, brute force={expect}"
        )
    print(f"  [ok] G2.1 victim selection matches brute-force reference over {n_trials} random trials")


def _make_light_node(nid: int, weight: int = 1) -> Node:
    sig = Signature(id=nid, t=float(nid), kp=np.zeros((0, 2), dtype=np.float32),
                     kp3d=np.zeros((0, 3), dtype=np.float32),
                     desc=np.zeros((0, 32), dtype=np.uint8),
                     valid=np.zeros((0,), dtype=bool))
    return Node(id=nid, sig=sig, pose_odom=np.eye(4), pose_map=np.eye(4), weight=weight)


def check_synthetic_run_stm_cap_and_audit() -> None:
    path = tempfile.mktemp(prefix="pyslam_g2_", suffix=".sqlite3")
    cfg = Config(wm_budget_ms=1.0, wm_min_resident=5, stm_size=10)  # deliberately
        # aggressive: forces real transfer activity within a 140-frame run
        # rather than relying on the machine happening to be slow enough.
    scen = build("square6dof", seed=1)
    src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=1)
    pipe = Pipeline(cfg, backend_prefer="native", ltm_path=path)
    max_stm = [0]
    orig_add = pipe.memory.add
    def _tracking_add(node):
        orig_add(node)
        max_stm[0] = max(max_stm[0], len(pipe.memory.stm))
    pipe.memory.add = _tracking_add
    result = pipe.run(src, verbose=False)

    assert max_stm[0] <= cfg.stm_size, f"STM exceeded cap: saw {max_stm[0]} > {cfg.stm_size}"
    assert len(pipe.memory.ltm_ids) > 0, (
        "sanity: this run's budget is aggressive enough that at least one "
        "transfer to LTM should have happened -- if not, the test fixture "
        "itself needs retuning, not the code"
    )
    problems = pipe.memory.audit_consistency()
    assert problems == [], f"MUTATION-CLASS FAILURE: dangling references after a real run: {problems}"
    print(f"  [ok] G2.2/G2.3 STM cap held (max {max_stm[0]}/{cfg.stm_size}), "
          f"{len(pipe.memory.ltm_ids)} node(s) transferred to LTM, audit_consistency() clean")
    pipe.memory.close()
    os.remove(path)


def check_budget_decoupled_from_frontend_time() -> None:
    """WP-J2 (Orin port, Tier 1): enforce_budget() must be driven by
    memory-management time (mem_duration_ms), not total per-frame time.
    A frontend that's simply slow (e.g. a different platform's ORB/PnP
    implementation) must NOT, by itself, trigger WM eviction -- eviction
    can't make the frontend faster, so coupling to it only costs
    loop-closure recall for no benefit. Simulated here by monkeypatching
    extract_signature to sleep well past wm_budget_ms on every frame,
    with wm_budget_ms set high enough that real memory-management work
    (which stays fast on this small fixture) never crosses it on its own."""
    import time as _time
    import pyslam.pipeline as pipeline_mod
    path = tempfile.mktemp(prefix="pyslam_g2_decouple_", suffix=".sqlite3")
    cfg = Config(wm_budget_ms=80.0, wm_min_resident=2, stm_size=10)
    scen = build("square6dof", seed=1)
    src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=1)
    pipe = Pipeline(cfg, backend_prefer="native", ltm_path=path)

    real_extract = pipeline_mod.extract_signature
    def _slow_extract(*a, **kw):
        _time.sleep(0.12)  # 120ms >> wm_budget_ms=80ms, entirely frontend-side
        return real_extract(*a, **kw)
    pipeline_mod.extract_signature = _slow_extract
    try:
        pipe.run(src, verbose=False, max_frames=40)
    finally:
        pipeline_mod.extract_signature = real_extract

    n_transferred = len(pipe.memory.ltm_ids)
    assert n_transferred == 0, (
        f"REGRESSION: {n_transferred} node(s) evicted to LTM purely because "
        f"the frontend was artificially slow (120ms/frame sleep) -- "
        f"enforce_budget() is reading total frame time again, not "
        f"mem_duration_ms; this will silently change loop-closure recall "
        f"on any platform whose frontend speed differs from the reference "
        f"machine's."
    )
    print(f"  [ok] G2.6 WP-J2: 120ms/frame artificial frontend slowdown over 40 frames "
          f"caused ZERO LTM transfers (budget correctly reads memory-management time only)")
    pipe.memory.close()
    os.remove(path)


def check_crash_mid_batch_leaves_openable_db() -> None:
    path = tempfile.mktemp(prefix="pyslam_g2_crash_", suffix=".sqlite3")
    store = LtmStore(path)
    committed_ids = []
    try:
        for i in range(20):
            if i == 12:
                raise RuntimeError("simulated SIGKILL mid-batch")
            store.put(_make_light_node(i))
            committed_ids.append(i)
    except RuntimeError:
        pass
    store.close()  # simulates the process dying right after -- the file on
        # disk is what matters, not this handle

    # Reopen fresh, as a new process would after a real crash+restart.
    reopened = LtmStore(path)
    assert sorted(reopened.all_ids()) == committed_ids, (
        f"DB not openable/consistent after crash: expected ids {committed_ids}, "
        f"got {sorted(reopened.all_ids())}"
    )
    for i in committed_ids:
        assert reopened.get(i) is not None
    reopened.close()
    os.remove(path)
    print(f"  [ok] G2.4 crash after {len(committed_ids)}/20 puts: DB reopens with exactly "
          f"those {len(committed_ids)} rows intact, none corrupted")


def check_graph_neighbour_retrieval() -> None:
    cfg = Config()
    mem = Memory(cfg)
    a, b = _make_light_node(1), _make_light_node(2)
    mem.wm[1] = a; mem._all[1] = a
    mem.wm[2] = b; mem._all[2] = b
    mem.record_adjacency(1, 2)

    mem._transfer_to_ltm(2)
    assert 2 in mem.ltm_ids and 2 not in mem.wm
    assert mem.audit_consistency() == []

    mem.on_loop(1)  # node 1 (still in WM) is the loop hit; its neighbour (2) should come back
    assert 2 in mem.wm and 2 not in mem.ltm_ids, (
        f"MUTATION MISSED: on_loop() did not rehydrate a graph-adjacent LTM node "
        f"(wm={list(mem.wm.keys())}, ltm_ids={mem.ltm_ids})"
    )
    assert mem.retrieval_events == [(1, [2])]
    assert mem.audit_consistency() == []
    print("  [ok] G2.5 on_loop() rehydrates a graph-adjacent LTM neighbour back into WM")


ALL_G2_CHECKS = [
    check_victim_selection_matches_brute_force,
    check_synthetic_run_stm_cap_and_audit,
    check_budget_decoupled_from_frontend_time,
    check_crash_mid_batch_leaves_openable_db,
    check_graph_neighbour_retrieval,
]


def main() -> int:
    print(f"Gate G2 (WP-P2 memory management) -- {len(ALL_G2_CHECKS)} checks\n")
    n_pass = 0
    for check in ALL_G2_CHECKS:
        try:
            check()
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {check.__name__}: {e}")
    print(f"\n{n_pass}/{len(ALL_G2_CHECKS)} G2 checks passed")
    return 0 if n_pass == len(ALL_G2_CHECKS) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
