"""
WP-P3's "learned global descriptor channel", honestly scoped.

The architecture doc's phrase is "learned global descriptor" (the real
RTAB-Map/visual-place-recognition move is a trained CNN embedding --
NetVLAD, DELG, etc.). That needs a training pipeline, GPU time, and a
labelled dataset this project has none of, and no pretrained-weights
download path is available in this environment either (see section 8's
still-unanswered pip/dependency question). Shipping a fake "trained"
model would be exactly the kind of thing this project's own culture
(WP-B2's honest-NEES-gap, the vocabulary's "simple but correct beats
clever but wrong" note) argues against.

So this is a fixed random-projection pooling descriptor instead: each
ORB descriptor's 256 bits are projected through a fixed random
Gaussian matrix (Johnson-Lindenstrauss -- a real, well-understood
distance-preserving embedding, not a placeholder) and mean-pooled
across all of a frame's keypoints into one L2-normalised vector. It
plays the same interface role (a dense embedding an ANN index can do
cosine search over) and is fully reproducible (seeded, no training
step to drift), but it is NOT a learned descriptor in the sense the
doc's phrase evokes, and it should not be reported as one. If `torch`
plus real pretrained weights become available later, this is a drop-in
replacement behind the same three-method interface -- nothing else in
P3 needs to change.
"""
from __future__ import annotations
import numpy as np


class GlobalDescriptor:
    def __init__(self, out_dim: int = 256, desc_bits: int = 256, seed: int = 13):
        rng = np.random.default_rng(seed)
        self.out_dim = out_dim
        self.desc_bits = desc_bits
        # Fixed for the lifetime of the process -- seeded determinism,
        # per the architecture doc's rule 6 ("same input -> bit-identical
        # output, enforced by a gate").
        self.W = rng.normal(size=(desc_bits, out_dim)).astype(np.float32)

    def compute(self, desc_u8: np.ndarray) -> np.ndarray:
        """desc_u8: (N,32) uint8 ORB descriptors from one frame/node.
        Returns (out_dim,) float32, L2-normalised (zero vector if N==0).

        Pools with BOTH mean and max (concatenated, half the projected
        dimensionality each) rather than mean alone. Empirically
        checked, not assumed: mean-only pooling made this channel find
        ZERO of square6dof's 3 verified loop closures standalone (vs.
        raw_match's 1 and the incremental-BoW channel's 3) -- a single
        frame-average washes out exactly the localised, distinctive
        detail that makes one place different from another. Mean+max
        keeps some of that (a max response is dominated by whichever
        few keypoints projected most strongly onto each random
        direction, which survives averaging-out much less). This
        remains a coarse, honestly-scoped baseline (see module
        docstring) -- see WP_P3_Findings.md for the standalone-recall
        numbers after this change, including that it still does not
        match the BoW channel's recall alone, which is the actual,
        reported reason `retrieval_backend="both"` blends rather than
        relying on "learned" alone."""
        half = self.out_dim // 2
        if desc_u8.shape[0] == 0:
            return np.zeros(self.out_dim, dtype=np.float32)
        bits = np.unpackbits(desc_u8, axis=1).astype(np.float32) * 2.0 - 1.0  # (N, desc_bits)
        proj = bits @ self.W                                                   # (N, out_dim)
        mean_part = proj[:, :half].mean(axis=0)
        max_part = proj[:, half:].max(axis=0)
        pooled = np.concatenate([mean_part, max_part])
        norm = np.linalg.norm(pooled)
        if norm > 1e-9:
            pooled = pooled / norm
        return pooled.astype(np.float32)
