"""
WP-LIVE N6-generalised: layered occupancy, built by free-space ray
carving instead of "flood-fill the room interior" (which assumes a
single bounded room). Works identically for a corridor, an open-plan
floor, or several rooms connected by open doorways -- there's no
"interior" concept here at all, only "was this cell ever seen as
empty space along a camera ray" and "what's the tallest/lowest thing
observed there."

Grid is 2.5D over the SITE frame's x/y (pyslam/mapping/site_frame.py),
cell heights measured relative to the LOCAL support height field
(from PlaneLandmark instances of kind support), not one global z=0 --
see planning notes: a second-level landing or a ramp gets its own
correct "floor here" answer instead of being silently wrong relative
to the reference level.

Exports in the EXACT schema poi_localization/tracking/gates.py's
load_walkable_grid()/WalkableGrid already reads (grid/origin/
resolution, row=y, col=x, True=walkable) -- so the "grid" key in
walkable.json IS this module's person_plausible_mask(), and M4 needs
ZERO code changes to consume it. See gates.py's own WalkableGrid.is_walkable
for the exact row/col convention mirrored here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import numpy as np


# per-cell person-relevant height bands, metres above the LOCAL support height
TORSO_LOW_M = 0.15
TORSO_HIGH_M = 1.9
LOW_OCCUPIED_HIGH_M = 0.9   # up to this height above support = "sittable-ish" obstacle
UNOBSERVED, FREE, SUPPORT_OBSERVED, LOW_OCCUPIED, TALL_OCCUPIED = range(5)


@dataclass
class LayerGrid:
    origin_xy: tuple            # site-frame (x, y) of cell [0,0]'s min corner
    resolution_m: float
    support_height: np.ndarray  # (rows, cols) float, NaN = no support observed here
    min_height: np.ndarray      # (rows, cols) float, lowest point ever observed (site z)
    max_height: np.ndarray      # (rows, cols) float, highest point ever observed (site z)
    free_hits: np.ndarray       # (rows, cols) int32, ray-carve free-space hit count
    total_hits: np.ndarray      # (rows, cols) int32, any observation at all (free or surface)

    @classmethod
    def empty(cls, origin_xy, resolution_m, rows, cols):
        nanf = lambda: np.full((rows, cols), np.nan, dtype=np.float64)
        return cls(origin_xy, resolution_m, nanf(), nanf(), nanf(),
                   np.zeros((rows, cols), dtype=np.int32), np.zeros((rows, cols), dtype=np.int32))

    def _cell(self, x: float, y: float) -> Optional[tuple]:
        col = int((x - self.origin_xy[0]) / self.resolution_m)
        row = int((y - self.origin_xy[1]) / self.resolution_m)
        if row < 0 or col < 0 or row >= self.support_height.shape[0] or col >= self.support_height.shape[1]:
            return None
        return row, col

    def update_support(self, x: float, y: float, height_site: float) -> None:
        c = self._cell(x, y)
        if c is None:
            return
        r, cidx = c
        h = self.support_height[r, cidx]
        self.support_height[r, cidx] = height_site if np.isnan(h) else min(h, height_site)
        self.total_hits[r, cidx] += 1
        self._bump_extent(r, cidx, height_site)

    def update_surface_point(self, x: float, y: float, height_site: float) -> None:
        """Any observed point that ISN'T a support-plane inlier (walls,
        furniture, clutter) -- widens min/max height at that cell."""
        c = self._cell(x, y)
        if c is None:
            return
        r, cidx = c
        self.total_hits[r, cidx] += 1
        self._bump_extent(r, cidx, height_site)

    def _bump_extent(self, r, cidx, height_site):
        mn, mx = self.min_height[r, cidx], self.max_height[r, cidx]
        self.min_height[r, cidx] = height_site if np.isnan(mn) else min(mn, height_site)
        self.max_height[r, cidx] = height_site if np.isnan(mx) else max(mx, height_site)

    def update_free_ray(self, x: float, y: float) -> None:
        """A camera ray passed through empty space at this (x,y) --
        i.e. free-space carving. Called for cells along a ray from the
        camera centre to an observed surface point, EXCLUDING the
        surface cell itself. This is what lets an open doorway or an
        open-plan area register as walkable without ever seeing a
        bounding wall -- there is no flood-fill and no assumption that
        free space is enclosed."""
        c = self._cell(x, y)
        if c is None:
            return
        r, cidx = c
        self.free_hits[r, cidx] += 1

    def classify(self, support_seen_min_hits: int = 3) -> np.ndarray:
        """Returns an int grid of UNOBSERVED/FREE/SUPPORT_OBSERVED/
        LOW_OCCUPIED/TALL_OCCUPIED per cell, relative to that cell's own
        LOCAL support height (NaN support = height unknown, so the cell
        can never be classified as an occupied HEIGHT band, only free/
        unobserved) -- this is the local-height-field generalisation
        the flat single-floor version couldn't do."""
        out = np.full(self.support_height.shape, UNOBSERVED, dtype=np.int8)
        has_support = ~np.isnan(self.support_height)
        out[has_support] = SUPPORT_OBSERVED
        out[(self.free_hits >= 1) & ~has_support] = FREE
        out[(self.free_hits >= 1) & has_support] = SUPPORT_OBSERVED

        rel_max = np.where(has_support, self.max_height - self.support_height, np.nan)
        low_mask = has_support & (rel_max > TORSO_LOW_M) & (rel_max <= LOW_OCCUPIED_HIGH_M)
        tall_mask = has_support & (rel_max > LOW_OCCUPIED_HIGH_M)
        out[low_mask] = LOW_OCCUPIED
        out[tall_mask] = TALL_OCCUPIED
        return out

    def person_plausible_mask(self) -> np.ndarray:
        """Section 5's definition: support observed AND (free at torso
        height OR sittable). A mirror reflection or a spurious detection
        lands behind a surface the ray-carve never marked free/support
        (nothing was ever observed there from any real viewpoint), so it
        is rejected -- with NO notion of 'inside the room', which is
        exactly what generalises to open floor plans and doorways.
        Seated people (LOW_OCCUPIED cells, e.g. a sofa) are explicitly
        included, unlike a plain floor-inlier gate.
        """
        cls = self.classify()
        has_support = ~np.isnan(self.support_height)
        rel_max = np.where(has_support, self.max_height - self.support_height, np.nan)
        free_at_torso = has_support & (np.isnan(rel_max) | (rel_max <= TORSO_LOW_M) | (self.free_hits >= 1))
        sittable = cls == LOW_OCCUPIED
        return has_support & (free_at_torso | sittable)

    def to_walkable_json(self) -> dict:
        """STS poi_localization.tracking.gates.load_walkable_grid's exact
        schema: {"origin": [x,y], "resolution": r, "grid": [[bool,...]]}.
        `grid`'s [row][col] semantics match WalkableGrid.is_walkable
        exactly (row = (y-origin_y)/res, col = (x-origin_x)/res)."""
        mask = self.person_plausible_mask()
        return {
            "origin": [float(self.origin_xy[0]), float(self.origin_xy[1])],
            "resolution": float(self.resolution_m),
            "grid": mask.astype(bool).tolist(),
        }

    def write_walkable_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_walkable_json(), f)

    def to_layers_json(self) -> dict:
        """Fuller export (this repo's own extension, additive to the
        plan's walkable.json -- M4 doesn't need it, later M4 work
        described in the planning notes, e.g. per-cell floor-height
        ray-cast, would read this)."""
        cls = self.classify()
        return {
            "origin": [float(self.origin_xy[0]), float(self.origin_xy[1])],
            "resolution": float(self.resolution_m),
            "class_grid": cls.tolist(),
            "support_height": np.where(np.isnan(self.support_height), None,
                                        self.support_height).tolist(),
            "class_names": {"0": "unobserved", "1": "free", "2": "support_observed",
                             "3": "low_occupied", "4": "tall_occupied"},
        }

    def coverage_stats(self) -> dict:
        cls = self.classify()
        total = cls.size
        return {
            "frac_support_observed": float(np.mean(cls == SUPPORT_OBSERVED)),
            "frac_free": float(np.mean(cls == FREE)),
            "frac_unobserved": float(np.mean(cls == UNOBSERVED)),
            "n_cells": int(total),
        }


def carve_ray_free_cells(grid: LayerGrid, cam_xy: tuple, surface_xy: tuple, max_cells: int = 200) -> None:
    """Bresenham-style walk from the camera's (x,y) to a surface hit's
    (x,y) in the SAME layer grid, marking every intermediate cell free
    (excludes the surface cell itself, which the caller marks via
    update_support/update_surface_point). This is the free-space-
    carving step the module docstring describes -- it is what makes an
    open doorway register as walkable with no enclosing wall required.
    """
    x0, y0 = cam_xy
    x1, y1 = surface_xy
    dist = np.hypot(x1 - x0, y1 - y0)
    n_steps = min(max_cells, max(1, int(dist / max(grid.resolution_m, 1e-6))))
    for i in range(1, n_steps):  # exclude i=0 (camera cell) and the final surface cell
        t = i / n_steps
        grid.update_free_ray(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
