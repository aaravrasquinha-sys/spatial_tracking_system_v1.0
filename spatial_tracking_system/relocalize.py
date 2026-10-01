"""
Single-snapshot 6DOF relocalization against a map pack produced by an
earlier run_slam.py mapping run.

    python3 relocalize.py --map runs/run_.../reloc_map
    python3 relocalize.py --map runs/run_.../reloc_map --image query.png
    python3 relocalize.py --map runs/run_.../reloc_map --burst 7 --topk 20 \
        --overlay --save-query --depth-crosscheck --bench

Strictly read-only against the map pack: no graph optimisation, no
inserted nodes, nothing written into --map. See RELOCALIZATION.md for
the full design writeup and pyslam/loop/relocalize.py for the query
pipeline itself.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import cv2

from pyslam.core.config import Config, add_config_args
from pyslam.core.types import Intrinsics
from pyslam.core.log import get_logger
from pyslam.frontend.features import make_orb
from pyslam.mapping.reloc_map import RelocMap
from pyslam.loop.relocalize import relocalize, build_query_signature

log = get_logger("relocalize")


# --------------------------------------------------------------- capture

def _variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class SnapshotSource:
    """Owns the D435i for exactly as long as it takes to get one good
    RGB frame. Depth is captured too (RealSenseSource enables it
    unconditionally and builds an rs.align around it -- branching that
    apart to save a few ms on a burst that runs once was not worth the
    risk to live capture, see RELOCALIZATION.md's design note) but is
    only ever used for the optional --depth-crosscheck diagnostic, never
    for the reported pose: metric scale here comes from the MAP's
    depth, recorded at mapping time, not the query's.
    """

    def __init__(self, warmup_frames: int = 20):
        self.warmup_frames = warmup_frames
        self._source = None

    def capture(self, burst: int, blur_reject_var: float):
        from pyslam.sensors.realsense import RealSenseSource
        self._source = RealSenseSource(enable_imu=False)
        it = iter(self._source)
        try:
            intr = self._source.intrinsics()
            for _ in range(self.warmup_frames):
                next(it)
            candidates = []
            for _ in range(max(burst, 1)):
                frame = next(it)
                gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
                sharpness = _variance_of_laplacian(gray)
                candidates.append((sharpness, frame.rgb.copy(), frame.depth.copy(), frame.t))
            candidates.sort(key=lambda c: -c[0])
            best = candidates[0]
            if best[0] < blur_reject_var:
                raise RuntimeError(
                    f"every frame in the burst was below the blur-reject floor "
                    f"(best variance-of-Laplacian={best[0]:.1f} < {blur_reject_var}); "
                    f"hold the camera steadier, or lower --blur-reject-var if this is expected "
                    f"(e.g. a genuinely low-texture scene).")
            log.info(f"Burst of {len(candidates)}: kept sharpest (var={best[0]:.1f}), "
                     f"discarded {len(candidates)-1}.")
            return best[1], best[2], intr
        finally:
            self._source.close()


def load_image(path: str, intr_hint: Intrinsics):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != (intr_hint.width, intr_hint.height):
        log.warning(f"--image is {rgb.shape[1]}x{rgb.shape[0]}, map pack was built at "
                    f"{intr_hint.width}x{intr_hint.height} -- proceeding with the map's "
                    f"intrinsics anyway since --image has none of its own; this is only "
                    f"correct if the image was genuinely captured at that resolution.")
    return rgb, intr_hint


# ----------------------------------------------------------------- output

def _draw_overlay(rgb: np.ndarray, result, rmap: RelocMap) -> np.ndarray:
    """Best-candidate inlier correspondences, query-side only (no access
    to the map node's own image -- imagery is never stored in a pack,
    see reloc_map.py's docstring)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if not result.candidates:
        return img
    best = max(result.candidates, key=lambda c: c.n_inliers)
    color = (0, 200, 0) if result.status == "LOCALIZED" and best.node_id == result.node_id else (0, 0, 220)
    cv2.putText(img, f"{result.status} node={result.node_id} inliers={best.n_inliers}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def _print_summary(result) -> None:
    print(f"\n=== {result.status} ===")
    if result.reason:
        print(f"reason: {result.reason}")
    if result.status == "LOCALIZED":
        xyz = result.pose_world_query[:3, 3]
        print(f"node: {result.node_id}  consensus: {result.consensus}")
        print(f"position (W_cam0): x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f} m")
        print(f"sigma_trans: {result.sigma_trans_m:.3f} m")
        if result.pose_grav_query is not None:
            xyz_g = result.pose_grav_query[:3, 3]
            print(f"position (W_grav): x={xyz_g[0]:.3f} y={xyz_g[1]:.3f} z={xyz_g[2]:.3f} m")
    print(f"shortlisted={result.n_shortlisted} verified={result.n_verified} "
          f"retrieval_ambiguous={result.retrieval_ambiguous}")
    print(f"timing: {', '.join(f'{k}={v:.1f}ms' for k, v in result.timing_ms.items())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="path to a reloc_map/ pack directory")
    ap.add_argument("--image", type=str, default=None,
                     help="relocalize against a saved RGB image instead of the live D435i")
    ap.add_argument("--burst", type=int, default=5, help="frames to capture; sharpest is kept")
    ap.add_argument("--warmup-frames", type=int, default=20,
                     help="frames discarded before the burst, for auto-exposure to settle")
    ap.add_argument("--blur-reject-var", type=float, default=None,
                     help="override Config.reloc_blur_reject_var")
    ap.add_argument("--topk", type=int, default=None, help="override Config.reloc_shortlist_topk")
    ap.add_argument("--save-query", action="store_true", help="save the chosen query frame as a PNG")
    ap.add_argument("--overlay", action="store_true", help="write a debug overlay PNG")
    ap.add_argument("--depth-crosscheck", action="store_true",
                     help="compare live depth at inlier pixels against what PnP implies -- "
                          "diagnostic only, never affects the reported pose")
    ap.add_argument("--bench", action="store_true",
                     help="print per-stage timing only (still performs one real query)")
    ap.add_argument("--out", type=str, default="reloc_out", help="output directory")
    ap.add_argument("--json", type=str, default=None, help="also write the result to this exact path")
    add_config_args(ap)
    args = ap.parse_args()

    if not os.path.isdir(args.map):
        log.error(f"--map {args.map} is not a directory")
        sys.exit(1)
    rmap = RelocMap(args.map)
    log.info(f"Loaded map pack: {rmap.pack_dir} ({len(rmap)} nodes, "
             f"config_hash={rmap.meta.get('config_hash')})")

    # Base config is the MAP's own recorded config (feature/PnP settings
    # must match what the map was built with), not Config()'s bare
    # defaults -- config_from_args() always starts from the latter, so
    # overrides are applied by hand here instead.
    from dataclasses import replace
    from pyslam.core.config import apply_overrides
    cfg = Config(**rmap.meta["config"])
    try:
        cfg = apply_overrides(cfg, args.config_override)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    if args.topk is not None:
        cfg = replace(cfg, reloc_shortlist_topk=args.topk)
    if args.blur_reject_var is not None:
        cfg = replace(cfg, reloc_blur_reject_var=args.blur_reject_var)

    os.makedirs(args.out, exist_ok=True)
    orb = make_orb(cfg)

    if args.image:
        rgb, live_intr = load_image(args.image, rmap.intrinsics)
        depth = None
    else:
        snap = SnapshotSource(warmup_frames=args.warmup_frames)
        rgb, depth, live_intr = snap.capture(args.burst, cfg.reloc_blur_reject_var)

    mismatch = rmap.check_intrinsics(live_intr)
    if mismatch:
        log.warning(f"Live camera intrinsics do not match the map pack's ({mismatch}). "
                    f"Proceeding, but the reported pose will be wrong if this reflects a real "
                    f"resolution/optics difference rather than run-to-run auto-calibration wobble.")

    if args.save_query:
        p = os.path.join(args.out, "query.png")
        cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        log.info(f"Query frame saved: {p}")

    t0 = time.perf_counter()
    result = relocalize(rmap, rgb, live_intr, cfg, orb=orb)
    total_ms = (time.perf_counter() - t0) * 1000.0

    if args.depth_crosscheck and depth is not None and result.status == "LOCALIZED":
        _depth_crosscheck(rmap, result, rgb, depth, live_intr)

    _print_summary(result)
    if args.bench:
        print(f"end-to-end (incl. capture excluded): {total_ms:.1f} ms")

    out_json = args.json or os.path.join(args.out, "reloc_result.json")
    with open(out_json, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=float)
    log.info(f"Result written: {out_json}")

    log_path = os.path.join(args.out, "reloc_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **result.to_dict()}, default=float) + "\n")

    if args.overlay:
        p = os.path.join(args.out, "overlay.png")
        cv2.imwrite(p, _draw_overlay(rgb, result, rmap))
        log.info(f"Overlay saved: {p}")

    rmap.close()
    sys.exit(0 if result.status == "LOCALIZED" else 2)


def _depth_crosscheck(rmap: RelocMap, result, rgb: np.ndarray, depth: np.ndarray,
                       live_intr: Intrinsics) -> None:
    """Diagnostic only: for the winning candidate's inlier query pixels,
    compare the LIVE sensor's own depth (never used in the pose solve)
    against the depth implied by projecting the winning pose back into
    the query camera. Large, systematic disagreement is a useful early
    warning that something upstream is off (bad intrinsics, a stale
    map, a miscalibrated unit) even though it plays no role in the
    accept/reject decision itself."""
    winner = next((c for c in result.candidates if c.node_id == result.node_id), None)
    if winner is None or winner.inlier_img_pts is None or winner.T_world_query is None:
        return

    # Depth the winning pose IMPLIES at each inlier query pixel: project
    # that map node's own 3D inlier points into the query camera using
    # the winning T_world_query, and read off camera-frame Z. This is
    # independent of what the live sensor measured -- a genuine
    # cross-check, not a restatement of the PnP residual.
    from pyslam.core import lie
    from pyslam.loop.verify import match_descriptors

    node = rmap.get_node(result.node_id)
    T_query_world = lie.se3_inverse(winner.T_world_query)
    # obj_pts used in the original solve aren't retained on the trace
    # (kept small deliberately); re-derive them the same way _verify_one
    # did, matching descriptors fresh against this node.
    query_sig = build_query_signature(rgb, Config(**rmap.meta["config"]))
    matches = match_descriptors(node.sig.desc, query_sig.desc)
    if not matches:
        return
    idx_node = np.array([m[0] for m in matches])
    idx_query = np.array([m[1] for m in matches])
    valid = node.sig.valid[idx_node]
    obj_pts_node = node.sig.kp3d[idx_node[valid]]
    img_pts_all = query_sig.kp[idx_query[valid]]

    pts_world = lie.transform_points(rmap.poses[result.node_id], obj_pts_node)
    pts_query_frame = lie.transform_points(T_query_world, pts_world)
    implied_z = pts_query_frame[:, 2]

    depth_m = depth.astype(np.float64) * live_intr.depth_scale
    h, w = depth_m.shape
    ix = np.clip(np.round(img_pts_all[:, 0]).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(img_pts_all[:, 1]).astype(np.int64), 0, h - 1)
    measured_z = depth_m[iy, ix]
    have = measured_z > 0
    if have.sum() < 5:
        log.info("[depth-crosscheck] too few live depth samples at inlier pixels to compare.")
        return
    diff = measured_z[have] - implied_z[have]
    log.info(f"[depth-crosscheck] diagnostic only, not part of the accept/reject decision. "
             f"{have.sum()} points: median|diff|={np.median(np.abs(diff)):.3f}m "
             f"mean={np.mean(diff):+.3f}m std={np.std(diff):.3f}m")


if __name__ == "__main__":
    main()
