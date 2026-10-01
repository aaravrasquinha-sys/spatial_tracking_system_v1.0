"""
Fixed visual vocabulary (Phase 0 deliberate simplification -- see the
architecture brief, section on "the one non-obvious Phase 0 decision").
Built once offline via k-means over unpacked ORB bit-vectors, then
thresholded back to binary words. Assignment at runtime is a vectorised
Hamming-distance argmin, no incremental index, no deletion problem.

Incremental vocabulary is a Phase-3 upgrade behind the same
PlaceRecognizer interface.
"""
from __future__ import annotations
import numpy as np

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def build_vocabulary(descriptors: np.ndarray, vocab_size: int, seed: int = 0) -> np.ndarray:
    """descriptors: (M,32) uint8 ORB descriptors pooled from many frames.
    Returns (vocab_size,32) uint8 vocabulary words.

    Words are REAL descriptors, chosen by random sampling from the pool --
    not k-means centroids. This was an empirical correction: clustering
    unpacked bit-vectors with Euclidean k-means and thresholding the
    (float) centroids back to binary measurably destroyed the
    discriminative signal (verified on the corridor-loop synthetic fixture:
    the true loop-closure match dropped out of the top-1 candidate entirely
    under the k-means vocabulary, while raw descriptor Hamming matching
    -- and a random-sampled vocabulary -- both kept it clearly on top).
    Binary/Hamming descriptor spaces don't average the way Euclidean
    k-means assumes; a real descriptor is always a valid "prototype",
    whereas an averaged-then-rounded centroid can land in a region that
    matches many unrelated descriptors equally well. Simple, and correct
    beats clever-but-wrong here.
    """
    rng = np.random.default_rng(seed)
    if descriptors.shape[0] < vocab_size:
        raise ValueError(f"Need at least {vocab_size} descriptors to build a "
                          f"{vocab_size}-word vocabulary, got {descriptors.shape[0]}")
    idx = rng.choice(descriptors.shape[0], size=vocab_size, replace=False)
    return descriptors[idx].copy()


def hamming_distance_matrix(desc: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """desc: (N,32) uint8, vocab: (V,32) uint8 -> (N,V) int distances."""
    if desc.shape[0] == 0:
        return np.zeros((0, vocab.shape[0]), dtype=np.int32)
    xor = desc[:, None, :] ^ vocab[None, :, :]         # (N,V,32) uint8
    dist = _POPCOUNT_LUT[xor].sum(axis=2)               # (N,V)
    return dist.astype(np.int32)


def assign_words(desc: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Nearest-word assignment by Hamming distance. Returns (N,) int32."""
    if desc.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    dist = hamming_distance_matrix(desc, vocab)
    return np.argmin(dist, axis=1).astype(np.int32)


def save_vocabulary(path: str, vocab: np.ndarray) -> None:
    np.save(path, vocab)


def load_vocabulary(path: str) -> np.ndarray:
    return np.load(path)
