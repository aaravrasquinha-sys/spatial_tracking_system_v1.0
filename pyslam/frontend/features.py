"""
Phase-0 Sensory Memory: ORB features, grid-bucketed for spatial spread,
associated with depth (median-of-patch, range + gradient rejection),
back-projected into the camera frame.

Depth uncertainty model (used later by odometry/verify weighting):
  sigma_z ~= z^2 * sigma_d / (f * b)      (stereo depth camera)
"""
from __future__ import annotations
import numpy as np
import cv2

from pyslam.core.types import Frame, Signature, Intrinsics
from pyslam.core.config import Config

_next_id = [0]


def _reset_id_counter(start: int = 0) -> None:
    _next_id[0] = start


def _alloc_id() -> int:
    i = _next_id[0]
    _next_id[0] += 1
    return i


def make_orb(cfg: Config):
    return cv2.ORB_create(
        nfeatures=cfg.n_features * 3,   # over-detect, then grid-bucket down
        scaleFactor=cfg.orb_scale_factor,
        nlevels=cfg.orb_n_levels,
    )


def grid_bucket(kps, descs, cfg: Config, width: int, height: int):
    """Keep at most n_features total, spread across a grid_cols x grid_rows
    grid, ranked by response within each cell. Prevents all keypoints
    clustering on one high-texture patch.

    Orin port (Tier 1, WP-J1): vectorised. This used to be a pure-Python
    loop over every detected keypoint (the dominant grid_bucket cost --
    ~6.4ms/frame at ~900 pre-bucket keypoints, measured) building a dict
    of Python lists, then one `list.sort(key=lambda ...)` per grid cell
    (a lambda-per-comparison attribute lookup on `kps[i].response`).
    Below does the same grouping with one stable `np.argsort` over ALL
    keypoints instead of a per-keypoint dict-append loop, and the
    per-cell top-k selection is a second stable argsort restricted to
    each cell's slice -- so the only remaining Python-level loop is over
    grid cells (<= grid_cols*grid_rows, 48 by default), not keypoints.

    Bit-identical to the original by construction:
      - cell assignment: same floor-division + clamp arithmetic, in the
        same float64 precision numpy already uses for `kp.pt`.
      - within a cell, `list.sort(key=..., reverse=True)` is stable and
        keeps ties in their original (ascending-index) relative order;
        `np.argsort(-response, kind="stable")` on indices already in
        ascending-index order (guaranteed by the outer stable argsort
        used to group cells) produces the identical order.
      - cell VISIT order (which determines the final truncation when
        total kept exceeds n_features) is first-occurrence order over
        the original keypoint sequence, exactly like dict insertion
        order in the original.
    Gated bit-identical against the original implementation in
    tests/gates/test_g0.py::check_grid_bucket_vectorized_matches_loop.
    """
    if len(kps) == 0:
        return [], np.zeros((0, 32), dtype=np.uint8)
    cell_w = width / cfg.grid_cols
    cell_h = height / cfg.grid_rows
    per_cell = max(1, cfg.n_features // (cfg.grid_cols * cfg.grid_rows))
    n_cells = cfg.grid_cols * cfg.grid_rows

    pts = np.array([kp.pt for kp in kps], dtype=np.float64)
    resp = np.array([kp.response for kp in kps], dtype=np.float64)

    cx = np.minimum(cfg.grid_cols - 1, (pts[:, 0] // cell_w).astype(np.int64))
    cy = np.minimum(cfg.grid_rows - 1, (pts[:, 1] // cell_h).astype(np.int64))
    cell_id = cy * cfg.grid_cols + cx

    # Group original indices by cell, each group internally in ascending
    # original-index order (stable sort on already index-ordered input).
    order_by_cell = np.argsort(cell_id, kind="stable")
    sorted_cell_ids = cell_id[order_by_cell]
    counts = np.bincount(cell_id, minlength=n_cells)
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    # First-occurrence order of cells, matching dict insertion order.
    _, first_idx = np.unique(cell_id, return_index=True)
    cells_in_first_order = cell_id[np.sort(first_idx)]

    keep: list[int] = []
    for c in cells_in_first_order:
        c = int(c)
        members = order_by_cell[offsets[c]:offsets[c + 1]]  # ascending orig index
        local_order = np.argsort(-resp[members], kind="stable")
        keep.extend(members[local_order][:per_cell].tolist())

    keep = keep[: cfg.n_features] if len(keep) > cfg.n_features else keep
    out_kps = [kps[i] for i in keep]
    out_descs = descs[keep] if len(keep) > 0 else np.zeros((0, 32), dtype=np.uint8)
    return out_kps, out_descs


def associate_depth(kp_xy: np.ndarray, depth_raw: np.ndarray, intr: Intrinsics, cfg: Config):
    """For each 2D keypoint, sample a median depth patch, reject on range /
    local-gradient grounds, and back-project into the camera optical frame.
    Returns (kp3d (N,3) float32, valid (N,) bool).

    Vectorised across all keypoints at once (patch extraction via a single
    fancy-index gather + masked-array median/range) rather than a
    per-keypoint Python loop -- the latter measured as the dominant
    per-frame cost (~450ms/frame at ~900 keypoints), which is well outside
    even offline-iteration budget, let alone the >=10Hz live-camera target.
    """
    n = kp_xy.shape[0]
    kp3d = np.full((n, 3), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return kp3d, valid

    h, w = depth_raw.shape
    depth_m = depth_raw.astype(np.float64) * intr.depth_scale
    k = cfg.depth_patch

    padded = np.pad(depth_m, k, mode="constant", constant_values=0.0)
    ix = np.clip(np.round(kp_xy[:, 0]).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(kp_xy[:, 1]).astype(np.int64), 0, h - 1)
    cx, cy = ix + k, iy + k  # coords in the padded array

    offsets = np.arange(-k, k + 1)
    doff, joff = np.meshgrid(offsets, offsets, indexing="ij")
    doff, joff = doff.ravel(), joff.ravel()  # each (2k+1)^2,

    patch_y = cy[:, None] + doff[None, :]     # (N, P)
    patch_x = cx[:, None] + joff[None, :]     # (N, P)
    patches = padded[patch_y, patch_x]        # (N, P)

    valid_mask = patches > 0
    counts = valid_mask.sum(axis=1)
    enough = counts >= 5

    masked = np.ma.array(patches, mask=~valid_mask)
    z_med = np.ma.median(masked, axis=1).filled(0.0)
    z_max = np.ma.max(masked, axis=1).filled(0.0)
    z_min = np.ma.min(masked, axis=1).filled(0.0)
    spread = z_max - z_min

    ok = (enough & (z_med >= cfg.depth_min_m) & (z_med <= cfg.depth_max_m) &
          (spread <= cfg.depth_grad_max_m))

    px, py = kp_xy[:, 0], kp_xy[:, 1]
    x = (px - intr.cx) / intr.fx * z_med
    y = (py - intr.cy) / intr.fy * z_med
    kp3d[ok] = np.stack([x[ok], y[ok], z_med[ok]], axis=1).astype(np.float32)
    valid[ok] = True
    return kp3d, valid


def extract_signature(frame: Frame, cfg: Config, orb=None) -> Signature:
    if orb is None:
        orb = make_orb(cfg)
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    kps, descs = orb.detectAndCompute(gray, None)
    if descs is None:
        descs = np.zeros((0, 32), dtype=np.uint8)
    kps, descs = grid_bucket(kps, descs, cfg, frame.intr.width, frame.intr.height)

    kp_xy = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
    kp3d, valid = associate_depth(kp_xy, frame.depth, frame.intr, cfg)

    return Signature(
        id=_alloc_id(),
        t=frame.t,
        kp=kp_xy,
        kp3d=kp3d,
        desc=descs,
        valid=valid,
        rgb=frame.rgb,
        depth=frame.depth,
    )
