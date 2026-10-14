"""
Batch-C: export-time clean-up of the dense point map.

remove_isolated_voxels drops points that have too few neighbours: for every point it counts how many OTHER
points sit in the 26 surrounding cells of a grid with the dense map's own voxel size, and removes those with
fewer than `min_neighbors`. Real surfaces (walls, floors, furniture) are dense, so their points always have
many neighbours; stray points from depth noise, flying pixels and one-off bad keyframes have almost none.
Pure numpy, runs once per export (a few seconds for ~500k points), never in the live loop.
"""
from __future__ import annotations
import numpy as np


def remove_isolated_voxels(pts: np.ndarray, colors: np.ndarray, voxel_size_m: float, min_neighbors: int):
    """Returns (pts, colors) without the isolated points. min_neighbors <= 0 returns the input unchanged."""
    if min_neighbors <= 0 or pts.shape[0] == 0:
        return pts, colors
    cell = np.floor(pts / voxel_size_m).astype(np.int64)
    cell -= cell.min(axis=0)                           # non-negative
    span = cell.max(axis=0) + 3
    # pack (x, y, z) into one integer key; +1 shift keeps neighbours (-1) non-negative
    key = ((cell[:, 0] + 1) * span[1] + (cell[:, 1] + 1)) * span[2] + (cell[:, 2] + 1)
    order = np.argsort(key)
    skey = key[order]
    count = np.zeros(len(key), dtype=np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nkey = key + (dx * span[1] + dy) * span[2] + dz
                pos = np.searchsorted(skey, nkey)
                pos[pos >= len(skey)] = len(skey) - 1
                count += (skey[pos] == nkey)
    keep = count >= int(min_neighbors)
    return pts[keep], colors[keep]
