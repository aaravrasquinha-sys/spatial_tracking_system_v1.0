"""
Walkable grid in the ROOM frame, in the exact schema
poi_localization.tracking.gates.load_walkable_grid() reads
({"origin":[x,y], "resolution":r, "grid":[[bool]]}, row=y, col=x, True=walkable).

Built directly from the room-frame cloud (module 1 live mode has no free-space
carving product to reuse): a cell is walkable if

    it has FLOOR support (|z| < tol)  OR  it is a LOW occupied surface (a sofa/bed a person can sit on)
    AND it has no points between walk_sit_max and walk_band_hi metres (a wall / shelf / standing obstacle).

Cells that were never seen are NOT walkable (conservative; the mask is a plausibility GATE for M4 and a
mirror-reflection killer -- reflections land behind walls, where nothing is walkable), then dilated by a
small margin so a real footpoint at an obstacle's edge isn't rejected by 1-cell map noise.
Also reports how much of the floor the map actually saw.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation

from pyslam.anchor.config import AnchorConfig


def build_walkable(pts_room: np.ndarray, cfg: AnchorConfig) -> dict:
    r = cfg.walk_res_m
    x0, y0 = np.floor(pts_room[:, 0].min() / r) * r, np.floor(pts_room[:, 1].min() / r) * r
    cols = np.floor((pts_room[:, 0] - x0) / r).astype(int)
    rows = np.floor((pts_room[:, 1] - y0) / r).astype(int)
    ncol, nrow = cols.max() + 1, rows.max() + 1
    flat = rows * ncol + cols
    z = pts_room[:, 2]

    def count(mask):
        return np.bincount(flat[mask], minlength=nrow * ncol).reshape(nrow, ncol)

    n_floor = count(np.abs(z) < cfg.walk_floor_tol_m)
    n_low = count((z >= cfg.walk_band_lo_m) & (z <= cfg.walk_sit_max_m))
    n_high = count((z > cfg.walk_sit_max_m) & (z <= cfg.walk_band_hi_m))
    mp = cfg.walk_min_pts
    floor_ok = n_floor >= mp
    tall = n_high >= mp
    low_occ = (n_low >= mp) & ~tall
    walk = (floor_ok | low_occ) & ~tall
    if cfg.walk_dilate_cells > 0:
        walk = binary_dilation(walk, iterations=cfg.walk_dilate_cells)
    return {
        "origin": [float(x0), float(y0)], "resolution": float(r), "grid": walk,
        "stats": {"n_cells": int(walk.size), "n_walkable": int(walk.sum()),
                  "walkable_area_m2": float(walk.sum() * r * r),
                  "floor_supported_area_m2": float(floor_ok.sum() * r * r)},
    }


def walkable_to_json(w: dict) -> dict:
    return {"origin": w["origin"], "resolution": w["resolution"], "grid": w["grid"].astype(bool).tolist(),
            "frame": "room", "stats": w["stats"]}
