"""
WP-LIVE N-dense, Level 1: per-keyframe fused depth.

Every ordinary frame tracked against a keyframe (via odometry's own
T_ref_frame -- already computed, no new tracking math) is warped into
that keyframe's own camera frame and folded into a weighted running
average, per pixel. This is the "source of truth" the rest of the
live dense pipeline (Level 2 voxel submaps, ICP edges, plane
extraction, the M2 reference depth) reads from -- see the planning
notes' "a dense map that survives loop closures in real time" section
for why keyframe-relative fusion (rather than one global TSDF) is what
makes re-fusion after a loop closure a per-keyframe, not per-frame,
cost.

Deliberately pure numpy, no GPU/Open3D dependency: this runs in the
TRACKER stage at frame rate (30Hz budget), so it must be cheap. Level 2
(voxel submaps) is where GPU time is spent.

Noise model matches the rest of this codebase's frozen convention
(pyslam/frontend/features.py's docstring, WP-K's stereo_baseline fix):
    sigma_z ~= z^2 * sigma_d / (f * b)
using the REAL stereo baseline and depth-pair focal length, not the
colour stream's fx (the ~2.5x understatement the STS repo's own
accuracy-fix session found and fixed on the M4 side -- this module
gets it right from the start on the M1 side).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FusionParams:
    sigma_d_px: float = 0.1          # sub-pixel disparity noise, stereo pair
    depth_fx_px: float = 390.0       # IR-pair focal length, NOT colour fx (~607px) --
        # see module docstring; a caller passing colour fx here silently
        # understates sigma_z by ~2.5x, exactly the bug the STS M4 accuracy
        # session found and fixed on the measurement side.
    baseline_m: float = 0.0499
    outlier_sigma_mult: float = 4.0  # samples this many running-sigma away
        # from the current estimate are rejected outright (moving object,
        # a hand passing through frame, a bad match) rather than blended in
    min_weight_for_valid: float = 1.0  # a pixel needs at least this much
        # accumulated weight (roughly: this many confident observations)
        # before it's considered part of the map -- see is_valid()


def depth_sigma(z: np.ndarray, p: FusionParams) -> np.ndarray:
    """sigma_z(z), vectorised. z in metres, sigma in metres. Floors at a
    tiny positive value so 1/sigma^2 weighting never divides by zero at
    z=0 (which associate_depth() should already have excluded, but this
    function must not assume its caller got that right)."""
    z = np.maximum(z, 1e-6)
    sigma = (z ** 2) * p.sigma_d_px / max(p.depth_fx_px * p.baseline_m, 1e-9)
    return np.maximum(sigma, 1e-4)


class KeyframeFusedDepth:
    """Owns one keyframe's running (mean, variance-derived-weight) depth
    estimate, at the keyframe's own resolution. Call fuse_frame() once
    per tracked frame whose T_ref_frame targets this keyframe; call
    get_points() to read back a 3D point cloud (keyframe camera frame)
    for ICP, plane extraction, or promotion into the Level 2 voxel map.
    """

    def __init__(self, height: int, width: int, params: FusionParams = None):
        self.h, self.w = height, width
        self.p = params or FusionParams()
        self.mean = np.zeros((height, width), dtype=np.float64)   # metres, 0 = unobserved
        self.weight = np.zeros((height, width), dtype=np.float64)  # sum of 1/sigma^2
        self.n_obs = np.zeros((height, width), dtype=np.int32)

    def fuse_frame(self, depth_m: np.ndarray) -> None:
        """depth_m: (h,w) float metres, already warped into THIS
        keyframe's camera frame (T_ref_frame applied upstream -- this
        function does no reprojection, only per-pixel temporal fusion,
        so it's agnostic to how the warp was done). 0/NaN = no reading
        at that pixel this frame."""
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if not np.any(valid):
            return
        sigma = depth_sigma(depth_m, self.p)
        w_new = np.zeros_like(self.weight)
        w_new[valid] = 1.0 / (sigma[valid] ** 2)

        # Outlier rejection against the RUNNING estimate (not raw depth
        # noise) -- a pixel with enough prior weight to have a trustworthy
        # mean rejects a new sample that disagrees by more than
        # outlier_sigma_mult running-sigma. Pixels with no/weak prior
        # accept the new sample unconditionally (there's nothing to
        # reject against yet).
        has_prior = self.weight > self.p.min_weight_for_valid
        running_sigma = np.zeros_like(self.mean)
        running_sigma[has_prior] = 1.0 / np.sqrt(self.weight[has_prior])
        disagree = np.zeros_like(self.mean, dtype=bool)
        check = valid & has_prior
        disagree[check] = (np.abs(depth_m[check] - self.mean[check])
                            > self.p.outlier_sigma_mult * running_sigma[check])
        accept = valid & ~disagree

        # Weighted running mean update (Welford-style single pass,
        # numerically fine for the weight magnitudes here).
        new_w = self.weight[accept] + w_new[accept]
        self.mean[accept] = (self.mean[accept] * self.weight[accept]
                              + depth_m[accept] * w_new[accept]) / np.maximum(new_w, 1e-12)
        self.weight[accept] = new_w
        self.n_obs[accept] += 1

    def is_valid(self) -> np.ndarray:
        return self.weight >= self.p.min_weight_for_valid

    def get_points(self, K: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pts (N,3) float64 in this keyframe's camera frame,
        pixel_yx (N,2) int) for every valid pixel on the stride grid.
        No colour -- callers needing colour (dense export) re-sample
        the keyframe's own rgb at the same pixel_yx, kept separate so
        this module has zero dependency on which frame's rgb is still
        resident vs evicted to the imagery cache."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        valid = self.is_valid()
        ys, xs = np.where(valid[::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        if ys.size == 0:
            return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64)
        z = self.mean[ys, xs]
        x = (xs - cx) / fx * z
        y = (ys - cy) / fy * z
        pts = np.stack([x, y, z], axis=1)
        pixel_yx = np.stack([ys, xs], axis=1)
        return pts, pixel_yx

    def stats(self) -> dict:
        valid = self.is_valid()
        return {
            "n_valid_px": int(valid.sum()),
            "frac_valid": float(valid.mean()),
            "mean_n_obs_valid": float(self.n_obs[valid].mean()) if valid.any() else 0.0,
        }
