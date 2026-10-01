"""
Gate G-A (WP-K, Phase A of the Orin accuracy plan): checks for the changes
in this work package. No hardware needed. Runs inside `pyslam.selftest`.

    python3 -m tests.gates.test_ga

What each check protects (every one exists because the thing it guards
was a real, observed defect or a decision that must not silently regress):

  GA.1  config overrides        --config-override was documented but never existed;
                                 a typo'd value used to be silently accepted
  GA.2  nvpmodel parsing        the old README said `nvpmodel -m 0` = MAXN; on the
                                 target unit mode 0 is 15W
  GA.3  imagery cache           evicted keyframes vanished from map.ply (~2/3 of them
                                 on corridor_v2)
  GA.4  streaming cloud         ...and restoring them must be bit-exact and must not
                                 need all imagery resident at once
  GA.5  export ORDER            map.ply used to be written BEFORE finalize()
  GA.6  export resilience       a failing finalize() must not cost the run its map
  GA.7  telemetry digest        wm_budget_ms retuning needs keyframe-only percentiles
  GA.8  baseline audits         loop/vertical/bridge measurements are pure functions
                                 checked against hand-computed truth
  GA.9  height-range honesty    height_range_m is not height when gravity alignment failed
  GA.10 END-TO-END              a real pipeline run that really evicts nodes: the map
                                 covers every keyframe, and the pre-fix behaviour
                                 (imagery_cache_enabled=False) demonstrably does not
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

from pyslam.core import lie
from pyslam.core.config import (Config, CONFIG_CHOICES, apply_overrides, parse_override)
from pyslam.core.types import Link, Node, Signature


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _cfg(*overrides: str) -> Config:
    return apply_overrides(Config(), list(overrides))


# ---------------------------------------------------------------- GA.1
def check_config_overrides() -> None:
    c = _cfg("proximity_enabled=true", "odometry_backend=f2m", "wm_budget_ms=80", "stm_size=12",
             "use_hessian_info=FALSE", "loop_robust_kernel=dcs")
    if not (c.proximity_enabled is True and c.odometry_backend == "f2m" and c.wm_budget_ms == 80.0
            and isinstance(c.wm_budget_ms, float) and c.stm_size == 12 and isinstance(c.stm_size, int)
            and c.use_hessian_info is False and c.loop_robust_kernel == "dcs"):
        _fail(f"override coercion wrong: {c}")
    base = Config()
    if apply_overrides(base, []) is not base or apply_overrides(base, None) is not base:
        _fail("empty overrides must return the same Config object")
    if base.proximity_enabled is not False:
        _fail("apply_overrides mutated the original (frozen) Config")

    for bad, needle in [("retrieval_backend=incremental", "bow_incremental"),   # the doc typo that started this
                        ("proxmity_enabled=1", "proximity_enabled"),             # close-match hint
                        ("stm_size=1.5", "integer"), ("use_hessian_info=maybe", "boolean"),
                        ("no_equals_sign", "key=value"), ("odometry_backend=orb", "f2f")]:
        try:
            parse_override(bad)
        except ValueError as e:
            if needle not in str(e):
                _fail(f"{bad!r}: error message should mention {needle!r}, got: {e}")
        else:
            _fail(f"{bad!r} was accepted but is invalid")
    # a whole-number float string is fine for an int field ('20.0'), a fractional one is not (above)
    if _cfg("stm_size=20.0").stm_size != 20:
        _fail("stm_size=20.0 should coerce to 20")

    # CONFIG_CHOICES must stay honest: real fields, and the shipped default must be a legal value
    names = {f for f in Config.__dataclass_fields__}
    d = Config()
    for k, allowed in CONFIG_CHOICES.items():
        if k not in names:
            _fail(f"CONFIG_CHOICES has {k!r}, which is not a Config field")
        if getattr(d, k) not in allowed:
            _fail(f"default {k}={getattr(d, k)!r} is not in its own allowed set {allowed}")
    print(f"  [ok] GA.1 --config-override: typed coercion, typo hints, {len(CONFIG_CHOICES)} enum fields validated")


# ---------------------------------------------------------------- GA.2
def check_nvpmodel_parsing() -> None:
    from pyslam.tools.env_probe import parse_nvpmodel_conf, parse_nvpmodel_query, power_mode_advice
    conf = ("< POWER_MODEL ID=0 NAME=15W >\nCPU_ONLINE CORE_0 1\n< POWER_MODEL ID=1 NAME=25W >\n"
            "< POWER_MODEL ID=2 NAME=MAXN_SUPER >\n< POWER_MODEL ID=3 NAME=7W >\n< PARAM TYPE=FILE NAME=X >\n")
    modes = parse_nvpmodel_conf(conf)
    if [(m["id"], m["name"]) for m in modes] != [(0, "15W"), (1, "25W"), (2, "MAXN_SUPER"), (3, "7W")]:
        _fail(f"nvpmodel.conf parse wrong: {modes}")
    q = parse_nvpmodel_query("NV Power Mode: 15W\n0\n")   # verbatim from the target unit
    if q != {"name": "15W", "id": 0}:
        _fail(f"nvpmodel -q parse wrong: {q}")
    adv = power_mode_advice(q["name"], modes)
    if adv["is_max_mode"] or adv["recommended"] != {"id": 2, "name": "MAXN_SUPER"}:
        _fail(f"the exact situation on the target unit (15W, mode 0) must recommend mode 2: {adv}")
    # SUPER preferred over plain MAXN regardless of ID order
    adv2 = power_mode_advice("15W", [{"id": 0, "name": "MAXN"}, {"id": 5, "name": "MAXN_SUPER"}])
    if adv2["recommended"]["name"] != "MAXN_SUPER":
        _fail(f"SUPER should be preferred: {adv2}")
    if not power_mode_advice("MAXN_SUPER", modes)["is_max_mode"]:
        _fail("MAXN_SUPER must count as a max mode")
    empty = power_mode_advice("15W", [])
    if empty["recommended"] is not None or "nvpmodel -p --verbose" not in empty["note"]:
        _fail(f"unparseable conf must fall back to telling the user to look: {empty}")
    print("  [ok] GA.2 nvpmodel: unit's '15W / mode 0' -> recommends MAXN_SUPER by NAME, not by assumed ID")


# ------------------------------------------------ fixtures for GA.3/4/5/6
def _fake_node(i: int, rng: np.random.Generator, h: int = 48, w: int = 64) -> Node:
    sig = Signature(id=i, t=0.1 * i, kp=np.zeros((0, 2), np.float32), kp3d=np.zeros((0, 3), np.float32),
                    desc=np.zeros((0, 32), np.uint8), valid=np.zeros(0, bool),
                    rgb=rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
                    depth=rng.integers(500, 3000, (h, w), dtype=np.uint16))
    T = lie.make_T(np.eye(3), np.array([0.2 * i, 0.0, 0.0]))
    return Node(id=i, sig=sig, pose_odom=T.copy(), pose_map=T.copy())


def _fill_and_evict(cfg: Config, n_nodes: int = 14, cache_dir=None):
    from pyslam.memory.memory import Memory
    rng = np.random.default_rng(7)
    mem = Memory(cfg, imagery_cache_dir=cache_dir)
    originals = {}
    for i in range(n_nodes):
        n = _fake_node(i, rng)
        originals[i] = (n.sig.rgb.copy(), n.sig.depth.copy())
        mem.add(n)
    for _ in range(50):           # over-budget every call -> one victim group per call until none eligible
        mem.enforce_budget(1e9)
    return mem, originals


# ---------------------------------------------------------------- GA.3
def check_imagery_cache_roundtrip() -> None:
    tmp = tempfile.mkdtemp(prefix="ga3_")
    try:
        cfg = _cfg("stm_size=3", "wm_min_resident=2")
        mem, orig = _fill_and_evict(cfg, cache_dir=os.path.join(tmp, "img"))
        evicted = sorted(mem.ltm_ids)
        if len(evicted) < 5:
            _fail(f"fixture should evict several nodes, got {evicted}")
        for nid in evicted:
            if mem.get(nid).sig.rgb is not None or mem.get(nid).sig.depth is not None:
                _fail(f"node {nid} still holds imagery in RAM after eviction (frozen contract broken)")
            got = mem.load_imagery(nid)
            if got is None:
                _fail(f"node {nid}: imagery not restorable from cache")
            if not (np.array_equal(got[0], orig[nid][0]) and np.array_equal(got[1], orig[nid][1])):
                _fail(f"node {nid}: imagery not BIT-EXACT after PNG round trip (rgb order / depth dtype?)")
        # resident nodes serve their own arrays without touching disk
        resident = sorted(mem.wm.keys())
        if not resident or mem.load_imagery(resident[0]) is None:
            _fail("resident node should still be loadable")
        st = mem.imagery_cache_stats()
        if st["n_cached_nodes"] != len(evicted) or st["bytes_on_disk"] <= 0:
            _fail(f"cache stats wrong: {st} vs {len(evicted)} evicted")
        # LTM's own contract is unchanged: its stored node blobs never carry imagery
        stored = mem.ltm_store.get(evicted[0])
        if stored.sig.rgb is not None or stored.sig.depth is not None:
            _fail("LTM database must still never persist rgb/depth")
        n_removed = mem.drop_imagery_cache()
        if n_removed != 2 * len(evicted) or os.path.isdir(os.path.join(tmp, "img")):
            _fail(f"drop_imagery_cache left files behind (removed {n_removed})")

        # disabled -> exact pre-WP-K3 behaviour: nothing written, evicted imagery gone
        mem2, _ = _fill_and_evict(_cfg("stm_size=3", "wm_min_resident=2", "imagery_cache_enabled=false"),
                                   cache_dir=os.path.join(tmp, "img2"))
        if mem2.load_imagery(sorted(mem2.ltm_ids)[0]) is not None or os.path.exists(os.path.join(tmp, "img2")):
            _fail("imagery_cache_enabled=False must write nothing and restore nothing")

        # a cache that cannot be written must degrade the MAP, never crash SLAM
        blocker = os.path.join(tmp, "not_a_dir")
        open(blocker, "w").close()
        mem3, _ = _fill_and_evict(_cfg("stm_size=3", "wm_min_resident=2"), cache_dir=blocker)
        if len(mem3.ltm_ids) < 5:
            _fail("eviction must still complete when the imagery cache is unwritable")
        if mem3.imagery_cache_stats()["disabled_reason"] is None:
            _fail("an unwritable cache should record why it disabled itself")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [ok] GA.3 imagery cache: {len(evicted)} evicted nodes restored bit-exact; LTM blobs still imagery-free; "
          f"disabled/unwritable cases degrade safely")


# ---------------------------------------------------------------- GA.4
def check_cloud_streams_evicted_nodes() -> None:
    from pyslam.mapping.cloud import assemble_cloud, assemble_cloud_with_stats
    K = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    tmp = tempfile.mkdtemp(prefix="ga4_")
    try:
        cfg = _cfg("stm_size=3", "wm_min_resident=2")
        # reference: same nodes, nothing evicted (evictions disabled by a huge budget never firing)
        from pyslam.memory.memory import Memory
        rng = np.random.default_rng(7)
        ref_mem = Memory(cfg)
        for i in range(14):
            ref_mem.add(_fake_node(i, rng))
        ref_nodes = [ref_mem.get(i) for i in ref_mem.all_node_ids()]
        ref_pts, ref_col = assemble_cloud(ref_nodes, K, 0.001, voxel=0)

        mem, _ = _fill_and_evict(cfg, cache_dir=os.path.join(tmp, "img"))
        nodes = [mem.get(i) for i in mem.all_node_ids()]
        n_evicted = len(mem.ltm_ids)
        # old behaviour (no loader): evicted nodes silently dropped
        _, _, st_old = assemble_cloud_with_stats(nodes, K, 0.001, voxel=0)
        if st_old["n_nodes_skipped_no_imagery"] < n_evicted or st_old["n_nodes_in_map"] >= len(nodes):
            _fail(f"without a loader evicted nodes must be skipped, as before: {st_old}")
        # new behaviour: every node contributes, and the cloud equals the never-evicted reference
        pts, col, st = assemble_cloud_with_stats(nodes, K, 0.001, voxel=0, imagery_loader=mem.load_imagery)
        if st["n_nodes_in_map"] != len(nodes) or st["n_nodes_from_imagery_cache"] != n_evicted:
            _fail(f"coverage wrong with loader: {st}")
        order = lambda p: np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
        if pts.shape != ref_pts.shape or not (np.allclose(pts[order(pts)], ref_pts[order(ref_pts)])
                                              and np.array_equal(col[order(pts)], ref_col[order(ref_pts)])):
            _fail("cloud rebuilt from the imagery cache differs from the never-evicted reference")
        # streaming: nothing was put back on the nodes
        if any(mem.get(i).sig.rgb is not None for i in mem.ltm_ids):
            _fail("load_imagery must not re-inflate evicted nodes in RAM")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [ok] GA.4 cloud: {st_old['n_nodes_in_map']}/{len(nodes)} keyframes mapped before -> "
          f"{st['n_nodes_in_map']}/{len(nodes)} now, identical to a never-evicted run, nothing re-inflated")


# ---------------------------------------------------------------- GA.5 / GA.6
class _FakeMemory:
    def __init__(self, nodes):
        self._n = {n.id: n for n in nodes}

    def get(self, i):
        return self._n[i]

    def all_node_ids(self):
        return list(self._n)

    def load_imagery(self, i):
        return None

    def imagery_cache_stats(self):
        return {"enabled": True, "n_cached_nodes": 0}

    def drop_imagery_cache(self):
        return 0


class _FakePipeline:
    """finalize() applies a known rigid shift to every pose_map, standing in
    for what a real loop correction does to the map."""
    SHIFT = np.array([5.0, 0.0, 0.0])

    def __init__(self, nodes, fail: bool = False):
        self.memory = _FakeMemory(nodes)
        self.fail = fail
        self.finalize_calls = 0

    def finalize(self, result):
        self.finalize_calls += 1
        if self.fail:
            raise RuntimeError("simulated GTSAM failure")
        for i in self.memory.all_node_ids():
            self.memory.get(i).pose_map[:3, 3] += self.SHIFT
        result.final_poses = {i: self.memory.get(i).pose_map.copy() for i in self.memory.all_node_ids()}
        return result.final_poses


class _FakeResult:
    final_poses = None


class _Intr:
    depth_scale = 0.001

    def K(self):
        return np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])


def _run_export(fail_finalize: bool):
    import pyslam.tools.run_outputs as ro
    rng = np.random.default_rng(3)
    nodes = [_fake_node(i, rng) for i in range(4)]
    pre = {n.id: n.pose_map.copy() for n in nodes}
    captured = {}
    real_write, real_exp, real_plots = ro.write_ply, ro.trajectory_export.export_all, ro.plots.export_plots
    ro.write_ply = lambda path, pts, cols: captured.update(pts=pts.copy(), path=path)
    ro.trajectory_export.export_all = lambda *a, **k: ({"gravity_aligned": True}, np.eye(4))
    ro.plots.export_plots = lambda *a, **k: None
    tmp = tempfile.mkdtemp(prefix="ga5_")
    try:
        pipe = _FakePipeline(nodes, fail=fail_finalize)
        out = ro.finalize_and_export(tmp, pipe, _FakeResult(), _Intr(), cloud_kwargs={"voxel": 0})
        return out, captured, pipe, nodes, pre
    finally:
        ro.write_ply, ro.trajectory_export.export_all, ro.plots.export_plots = real_write, real_exp, real_plots
        shutil.rmtree(tmp, ignore_errors=True)


def check_export_order_map_after_finalize() -> None:
    from pyslam.mapping.cloud import assemble_cloud
    out, cap, pipe, nodes, pre = _run_export(fail_finalize=False)
    if pipe.finalize_calls != 1 or not out["finalized"]:
        _fail(f"finalize must run exactly once: calls={pipe.finalize_calls}, out={out}")
    # what the OLD order (map first, finalize second) would have exported:
    old_nodes = []
    for n in nodes:
        m = Node(id=n.id, sig=n.sig, pose_odom=n.pose_odom.copy(), pose_map=pre[n.id].copy())
        old_nodes.append(m)
    old_pts, _ = assemble_cloud(old_nodes, _Intr().K(), 0.001, voxel=0)
    shift = cap["pts"].mean(axis=0) - old_pts.mean(axis=0)
    if not np.allclose(shift, _FakePipeline.SHIFT, atol=1e-3):
        _fail(f"exported map is not built from post-finalize poses: mean shift {shift}, expected {_FakePipeline.SHIFT} "
              f"(the pre-WP-K1 order would give ~0)")
    if abs(np.linalg.norm(old_pts.mean(axis=0) - cap["pts"].mean(axis=0))) < 1.0:
        _fail("vacuous check: pre- and post-finalize clouds are indistinguishable")
    if out["map"]["poses"] != "final (post-finalize)":
        _fail(f"map stats should say which poses were used: {out['map']}")
    print(f"  [ok] GA.5 export order: map is built AFTER finalize (mean shift {shift[0]:.2f} m == the correction; "
          f"old order would give 0)")


def check_export_survives_finalize_failure() -> None:
    out, cap, pipe, nodes, pre = _run_export(fail_finalize=True)
    if out["finalized"] or "simulated GTSAM failure" not in (out["finalize_error"] or ""):
        _fail(f"failure not recorded: {out}")
    if "pts" not in cap or cap["pts"].shape[0] == 0:
        _fail("a finalize() failure must not cost the run its map")
    if out["map"]["poses"] != "online (finalize failed)":
        _fail(f"map stats must admit the poses were online: {out['map']}")
    if out["trajectory_report"] is not None:
        _fail("no trajectory should be exported from a failed finalize")
    print("  [ok] GA.6 export resilience: finalize() failure is recorded, map still written from online poses")


# ---------------------------------------------------------------- GA.7
def check_telemetry_digest() -> None:
    from pyslam.tools.run_outputs import summarize_telemetry
    tele = []
    for i in range(100):
        kf = (i % 10 == 0)
        tele.append({"t": i * 0.1, "frame_id": i, "duration_ms": 100.0 + i, "keyframe": kf,
                     "mem_duration_ms": (20.0 + 10 * (i // 10)) if kf else 0.0,
                     "wm_size": i // 10, "odom_status": "LOST" if i in (50, 51) else "OK", "n_inliers": 100,
                     "enforce_budget_ms": (24.0 if i == 30 else 36.0 if i == 60 else 0.02)})
    d = summarize_telemetry(tele, wm_budget_ms=60.0)
    kf_mem = [20.0 + 10 * k for k in range(10)]           # 20..110
    if d["n_keyframe_frames"] != 10 or abs(d["keyframe_mem_duration_ms"]["p50"] - np.percentile(kf_mem, 50)) > 1e-9:
        _fail(f"mem percentiles must be over KEYFRAME frames only (zeros would drag them down): {d}")
    if d["keyframes_over_wm_budget"] != sum(1 for m in kf_mem if m > 60.0) or d["n_frames_odom_lost"] != 2:
        _fail(f"budget/lost counts wrong: {d}")
    if d["wm_size"] != {"max": 9, "final": 9} or summarize_telemetry([]) != {"n_frames": 0}:
        _fail(f"wm_size / empty handling wrong: {d}")
    ev = d["eviction_cost_ms"]   # only the 2 frames where an eviction really happened (24ms, 36ms)
    if ev is None or abs(ev["p50"] - 30.0) > 1e-9 or ev["max"] != 36.0:
        _fail(f"eviction cost must count only real evictions (>1ms), not the ~0ms no-op calls: {ev}")
    print("  [ok] GA.7 telemetry digest: keyframe-only mem_duration percentiles, budget overshoot fraction, LOST count")


# ---------------------------------------------------------------- GA.8
def _line_gt(n: int) -> dict:
    return {i: lie.make_T(np.eye(3), np.array([0.1 * i, 0.0, 0.0])) for i in range(n)}


def check_baseline_audits() -> None:
    from pyslam.tools.phase_a_baseline import loop_audit, vertical_horizontal_error, bridge_audit, WRONG_LINK_TRANS_M
    gt = _line_gt(60)                               # node i at x = 0.1*i, node time = 0.5*i s
    node_t = {i: 0.5 * i for i in gt}
    true = lambda a, b: lie.se3_inverse(gt[a]) @ gt[b]
    good_far = Link(0, 50, true(0, 50), np.eye(6), "loop")                                   # 25 s apart, correct
    trivial = Link(10, 13, true(10, 13), np.eye(6), "loop")                                  # 1.5 s apart, correct
    wrong_T = true(0, 40).copy(); wrong_T[0, 3] += 5.0
    wrong = Link(0, 40, wrong_T, np.eye(6), "loop")                                          # 20 s apart, 5 m off
    a = loop_audit([good_far, trivial, wrong], gt, node_t, min_gap_s=10.0)
    if (a["n"], a["n_trivial"], a["n_wrong"], a["n_real"]) != (3, 1, 1, 1):
        _fail(f"loop audit classification wrong: {a}")
    if abs(a["max_trans_err_m"] - 5.0) > 1e-6 or WRONG_LINK_TRANS_M >= 5.0:
        _fail(f"loop audit error magnitude wrong: {a}")
    # the corridor_v2 failure pattern: every 'loop' ~10 keyframes back -> zero REAL loops
    fake = [Link(i - 10, i, true(i - 10, i), np.eye(6), "loop") for i in range(12, 40, 3)]
    fa = loop_audit(fake, gt, node_t, min_gap_s=10.0)
    if fa["n"] != len(fake) or fa["n_real"] != 0 or fa["n_trivial"] != len(fake):
        _fail(f"'many loops, none real' pattern must report 0 real loops: {fa}")

    # vertical error: pure z drift of 0.1 m per step, zero horizontal error
    gtl = [gt[i] for i in range(20)]
    est = [T.copy() for T in gtl]
    for i, T in enumerate(est):
        T[2, 3] += 0.1 * i
    v = vertical_horizontal_error(est, gtl)
    want_rms = float(np.sqrt(np.mean((0.1 * np.arange(20)) ** 2)))
    if abs(v["vert_err_rms_m"] - want_rms) > 1e-9 or abs(v["vert_err_max_abs_m"] - 1.9) > 1e-9 or v["horiz_err_rms_m"] > 1e-9:
        _fail(f"vertical/horizontal split wrong: {v}")
    # ...and it is invariant to where the estimate's own frame starts (first-pose anchoring)
    off = lie.make_T(lie.so3_exp(np.array([0.0, 0.0, 0.7])), np.array([3.0, -2.0, 1.0]))
    v2 = vertical_horizontal_error([off @ T for T in est], gtl)
    if abs(v2["vert_err_rms_m"] - v["vert_err_rms_m"]) > 1e-9:
        _fail("vertical error must not depend on the estimate's arbitrary starting frame")
    # horizontal-only drift must NOT show up as vertical
    est_h = [T.copy() for T in gtl]
    for i, T in enumerate(est_h):
        T[1, 3] += 0.05 * i
    vh = vertical_horizontal_error(est_h, gtl)
    if vh["vert_err_rms_m"] > 1e-9 or vh["horiz_err_rms_m"] < 0.1:
        _fail(f"horizontal drift leaked into vertical: {vh}")

    # bridge audit: identity bridge across a real 0.4 m move -> the lie is 0.4 m
    b = bridge_audit([Link(5, 9, np.eye(4), np.eye(6) * 1e-3, "bridge"),
                      Link(1, 2, true(1, 2), np.eye(6), "odom")], gt)
    if b["n"] != 1 or abs(b["true_trans_m_sum"] - 0.4) > 1e-9:
        _fail(f"bridge audit wrong (odom links must be ignored): {b}")
    print("  [ok] GA.8 baseline audits: loop classification, 'many loops / zero real' pattern, vertical-vs-horizontal split "
          "(frame-invariant), bridge lie == true motion")


# ---------------------------------------------------------------- GA.9
def check_height_range_honesty() -> None:
    from pyslam.tools.trajectory_export import build_trajectory_report
    fp = {i: lie.make_T(np.eye(3), np.array([0.0, 0.0, 0.5 * i])) for i in range(5)}
    ids = list(fp)
    bad = build_trajectory_report(ids, fp, np.eye(4), False, "startup IMU window is not quasi-static",
                                  [], [], [], [])
    good = build_trajectory_report(ids, fp, np.eye(4), True, "aligned", [], [], [], [])
    if bad["height_range_valid"] is not False or "NOT HEIGHT" not in bad["height_range_note"]:
        _fail(f"unaligned run must flag height_range as invalid: {bad}")
    if good["height_range_valid"] is not True or "NOT HEIGHT" in good["height_range_note"]:
        _fail(f"aligned run must flag it valid: {good}")
    if bad["height_range_m"] != good["height_range_m"]:
        _fail("the numeric field must be unchanged (backward compatible), only annotated")
    print("  [ok] GA.9 trajectory_report: height_range_m is annotated as NOT HEIGHT when gravity alignment failed")


# ---------------------------------------------------------------- GA.10
def check_end_to_end_eviction_map() -> None:
    """A REAL pipeline, real synthetic frames, real evictions -- not fakes.
    (Small stm/wm floor and wm_budget_ms=0 force eviction on every over-
    budget keyframe so a 60-frame run exercises the whole path.)"""
    from pyslam.pipeline import Pipeline
    from pyslam.sensors.synthetic import SyntheticSource
    from pyslam.tools.run_outputs import finalize_and_export
    from tests.synth.scenarios import build
    from tests.synth.world import T_BODY_CAM

    scen = build("square6dof", seed=1, validate=False)
    results = {}
    for tag, extra in (("cache_on", []), ("cache_off", ["imagery_cache_enabled=false"])):
        tmp = tempfile.mkdtemp(prefix="ga10_")
        try:
            cfg = _cfg("stm_size=3", "wm_min_resident=3", "wm_budget_ms=0", *extra)
            src = SyntheticSource(scen.world, scen.intr, scen.poses, dt=scen.dt, add_noise=True, seed=1)
            pipe = Pipeline(cfg, backend_prefer="native", R_body_cam=T_BODY_CAM[:3, :3],
                            imagery_cache_dir=os.path.join(tmp, "imagery_cache"))
            res = pipe.run(src, max_frames=60, verbose=False)
            n_evicted = len(pipe.memory.ltm_ids_expanded())
            out = finalize_and_export(tmp, pipe, res, scen.intr, summary_path=None)
            files = set(os.listdir(tmp))
            results[tag] = (out, n_evicted, files, os.path.isdir(os.path.join(tmp, "imagery_cache")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    on, n_ev, files, cache_left = results["cache_on"]
    off, _, _, _ = results["cache_off"]
    if n_ev < 4:
        _fail(f"fixture did not evict enough nodes to test anything (evicted {n_ev})")
    if not on["finalized"] or "map.ply" not in files or "map_stats.json" not in files:
        _fail(f"expected finalize + map.ply + map_stats.json in the run dir, got {sorted(files)}")
    m_on, m_off = on["map"], off["map"]
    if m_on["n_nodes_in_map"] != m_on["n_nodes_total"] or m_on["n_nodes_from_imagery_cache"] < n_ev:
        _fail(f"with the cache every keyframe must be in the map: {m_on} (evicted {n_ev})")
    if m_off["n_nodes_in_map"] > m_off["n_nodes_total"] - n_ev or m_off["n_nodes_skipped_no_imagery"] < n_ev:
        _fail(f"without the cache the evicted keyframes must be missing (pre-fix behaviour): {m_off}")
    if m_on["n_points"] <= m_off["n_points"]:
        _fail("restoring evicted keyframes must add points")
    if cache_left:
        _fail("imagery cache directory should be deleted after export by default")
    print(f"  [ok] GA.10 end-to-end: {m_off['n_nodes_in_map']}/{m_off['n_nodes_total']} keyframes in the map before -> "
          f"{m_on['n_nodes_in_map']}/{m_on['n_nodes_total']} now ({m_off['n_points']} -> {m_on['n_points']} points), "
          f"finalize before map, cache cleaned up")


ALL_TESTS = [
    check_config_overrides,
    check_nvpmodel_parsing,
    check_imagery_cache_roundtrip,
    check_cloud_streams_evicted_nodes,
    check_export_order_map_after_finalize,
    check_export_survives_finalize_failure,
    check_telemetry_digest,
    check_baseline_audits,
    check_height_range_honesty,
    check_end_to_end_eviction_map,
]


def main() -> int:
    print(f"Gate G-A (WP-K1..K5) -- {len(ALL_TESTS)} checks\n")
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
