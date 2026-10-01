"""
WP-LIVE dense Level 2, CPU fallback: a voxel-hash point fuser.

This is deliberately NOT a TSDF/marching-cubes mesh -- it's an online
voxel-filtered, weight-accumulated coloured point cloud, keyed by
integer voxel coordinates in a dict (a "voxel hash"). Each voxel keeps
a running weighted centroid and colour, and a "last submap generation"
tag so a re-fused submap's voxels can be replaced in O(voxels in that
submap) rather than rebuilding the whole map.

Documented honestly, matching this project's own culture of not
overclaiming: this gives a live, loop-closure-correctable dense point
cloud suitable for a viewer and for surface-thickness/ICP-target use,
but it is NOT equivalent to nvblox's GPU TSDF+marching-cubes mesh --
no signed-distance field, no watertight surface, no free-space
integration for surface refinement. It exists so the dense stage has a
working, dependency-light (numpy only) default while
pyslam/tools/nvblox_eval.py's bounded evaluation decides whether nvblox
is viable on the target Orin Nano; nvblox_torch, if adopted, plugs in
as a second Level-2 backend behind the same interface (`DenseBackend`
Protocol below) without changing Level 1 (fused_depth.py) or anything
upstream of it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np


class DenseBackend(Protocol):
    """The interface both this module's VoxelHashFuser and a future
    nvblox_torch-backed implementation satisfy, so dense_process.py
    never has to know which one it's talking to."""

    def integrate_submap(self, submap_id: int, pts_world: np.ndarray,
                          colors: np.ndarray, generation: int) -> None: ...

    def drop_submap(self, submap_id: int) -> None: ...

    def export_points(self, max_points: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class VoxelHashFuser:
    voxel_size_m: float = 0.02

    def __post_init__(self):
        # key: (ix,iy,iz) -> [sum_x, sum_y, sum_z, sum_r, sum_g, sum_b, weight, submap_id, generation]
        self._voxels: dict[tuple, np.ndarray] = {}

    def _key(self, pts: np.ndarray) -> np.ndarray:
        return np.floor(pts / self.voxel_size_m).astype(np.int64)

    def integrate_submap(self, submap_id: int, pts_world: np.ndarray, colors: np.ndarray,
                          generation: int, weight: float = 1.0) -> None:
        """Fold a submap's world-frame points into the voxel hash. Points
        belonging to a STALE generation of this submap are dropped first
        (drop_submap) by the caller before re-integrating -- this module
        doesn't track per-voxel submap membership beyond overwrite-on-
        re-integration, kept simple deliberately (see module docstring's
        honesty note)."""
        if pts_world.shape[0] == 0:
            return
        keys = self._key(pts_world)
        for i in range(pts_world.shape[0]):
            k = tuple(keys[i].tolist())
            entry = self._voxels.get(k)
            p, c = pts_world[i], colors[i].astype(np.float64)
            if entry is None or entry[8] != generation:
                # new voxel, or a fresh generation for this submap replaces stale content
                self._voxels[k] = np.array([p[0] * weight, p[1] * weight, p[2] * weight,
                                             c[0] * weight, c[1] * weight, c[2] * weight,
                                             weight, submap_id, generation], dtype=np.float64)
            else:
                entry[0:3] += p * weight
                entry[3:6] += c * weight
                entry[6] += weight

    def drop_submap(self, submap_id: int) -> None:
        self._voxels = {k: v for k, v in self._voxels.items() if v[7] != submap_id}

    def export_points(self, max_points: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        if not self._voxels:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        arr = np.array(list(self._voxels.values()))
        w = np.maximum(arr[:, 6], 1e-9)
        pts = (arr[:, 0:3] / w[:, None]).astype(np.float32)
        colors = np.clip(arr[:, 3:6] / w[:, None], 0, 255).astype(np.uint8)
        if max_points is not None and pts.shape[0] > max_points:
            idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
            pts, colors = pts[idx], colors[idx]
        return pts, colors

    def n_voxels(self) -> int:
        return len(self._voxels)
