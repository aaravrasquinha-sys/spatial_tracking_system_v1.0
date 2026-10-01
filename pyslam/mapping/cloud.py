"""
Global cloud is always a VIEW: each node's per-node cloud (in its own
camera frame, computed once from its Signature) is re-transformed by the
node's current pose_map every time this is called. Nothing is
accumulated irreversibly, so a graph re-optimisation just means calling
this again.
"""
from __future__ import annotations
import numpy as np

from pyslam.core.types import Node
from pyslam.core import lie


def voxel_downsample(pts: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if pts.shape[0] == 0:
        return pts, colors
    keys = np.floor(pts / voxel).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return pts[unique_idx], colors[unique_idx]


def assemble_cloud(nodes: list[Node], K: np.ndarray, depth_scale: float,
                    stride: int = 4, max_depth_m: float = 4.0,
                    voxel: float = 0.03, imagery_loader=None) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the full world-frame cloud from scratch using each
    node's current pose_map. Deterministic given the node set + poses.

    imagery_loader (WP-K3): optional callable node_id -> (rgb, depth) | None
    (Memory.load_imagery). Used ONLY for nodes whose own rgb/depth were
    dropped on LTM eviction; each node's imagery is read, turned into
    points, and discarded before the next one, so peak memory does not
    grow with the number of evicted keyframes. Without it (the old
    signature) evicted nodes are skipped exactly as before."""
    pts, colors, _ = assemble_cloud_with_stats(nodes, K, depth_scale, stride, max_depth_m,
                                                voxel, imagery_loader)
    return pts, colors


def assemble_cloud_with_stats(nodes: list[Node], K: np.ndarray, depth_scale: float,
                               stride: int = 4, max_depth_m: float = 4.0,
                               voxel: float = 0.03, imagery_loader=None):
    """Same as assemble_cloud, plus a stats dict so a run can REPORT how
    much of its own keyframe set actually made it into the map (before
    WP-K3 this coverage was silently ~1/3 on corridor_v2 and nothing said so)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    all_pts = []
    all_colors = []
    n_resident = n_from_cache = n_skipped = 0
    for node in nodes:
        depth_full, rgb_full = node.sig.depth, node.sig.rgb
        if depth_full is None or rgb_full is None:
            loaded = imagery_loader(node.id) if imagery_loader is not None else None
            if loaded is None:
                n_skipped += 1
                continue
            rgb_full, depth_full = loaded
            from_cache = True
        else:
            from_cache = False
        depth = depth_full[::stride, ::stride].astype(np.float64) * depth_scale
        rgb = rgb_full[::stride, ::stride]
        hh, ww = depth.shape
        ys, xs = np.meshgrid(np.arange(hh), np.arange(ww), indexing="ij")
        px = xs * stride
        py = ys * stride
        z = depth
        valid = (z > 0) & (z < max_depth_m)
        if not np.any(valid):
            n_skipped += 1
            continue
        if from_cache:
            n_from_cache += 1
        else:
            n_resident += 1
        x = (px[valid] - cx) / fx * z[valid]
        y = (py[valid] - cy) / fy * z[valid]
        pts_cam = np.stack([x, y, z[valid]], axis=1).astype(np.float64)
        colors = rgb[valid]

        pts_world = lie.transform_points(node.pose_map, pts_cam)
        all_pts.append(pts_world)
        all_colors.append(colors)

    stats = {"n_nodes_total": len(nodes), "n_nodes_in_map": n_resident + n_from_cache,
             "n_nodes_resident": n_resident, "n_nodes_from_imagery_cache": n_from_cache,
             "n_nodes_skipped_no_imagery": n_skipped}
    if not all_pts:
        stats["n_points"] = 0
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), stats

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
    if voxel > 0:
        pts, colors = voxel_downsample(pts, colors, voxel)
    stats["n_points"] = int(pts.shape[0])
    return pts, colors, stats
