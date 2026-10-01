"""
WP-P3: cosine-similarity ANN index for the learned global-descriptor
channel. Mirrors graph/backend_gtsam.py's own pattern for exactly the
same unresolved question (architecture doc section 8, decision 4:
"can you pip install on the target machine?", never answered) -- try
the real dependency, fall back to a NumPy implementation that honours
the same tiny interface if it isn't importable. No silent quality
difference is hidden: `backend_name` reports which one is active so
gate G3's numbers are always attributable.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


class _HnswImpl:
    name = "hnswlib"

    def __init__(self, dim: int, initial_capacity: int):
        self.dim = dim
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max(initial_capacity, 16), ef_construction=200, M=16)
        self.index.set_ef(64)
        self._capacity = max(initial_capacity, 16)
        self._ids: set[int] = set()

    def add(self, node_id: int, vec: np.ndarray) -> None:
        if len(self._ids) >= self._capacity:
            self._capacity *= 2
            self.index.resize_index(self._capacity)
        if node_id in self._ids:
            self.index.mark_deleted(node_id)
        self.index.add_items(vec[None, :], np.array([node_id]))
        self._ids.add(node_id)

    def remove(self, node_id: int) -> None:
        if node_id in self._ids:
            self.index.mark_deleted(node_id)
            self._ids.discard(node_id)

    def query(self, vec: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        if not self._ids:
            return [], []
        k = min(k, len(self._ids))
        labels, distances = self.index.knn_query(vec[None, :], k=k)
        # hnswlib's "cosine" space returns distance = 1 - cosine_similarity
        sims = (1.0 - distances[0]).tolist()
        return labels[0].tolist(), sims

    def __len__(self) -> int:
        return len(self._ids)


class _NumpyBruteForceImpl:
    """Exact cosine top-k by direct computation. No dependency, and at
    the WM scale this project operates at (low thousands of resident
    vectors), simply the correct choice when hnswlib isn't installed --
    see gate G3's <5ms/query bound, checked against THIS implementation
    too, not just hnswlib's."""
    name = "numpy_bruteforce"

    def __init__(self, dim: int, initial_capacity: int = 0):
        self.dim = dim
        self._vecs: dict[int, np.ndarray] = {}

    def add(self, node_id: int, vec: np.ndarray) -> None:
        self._vecs[node_id] = vec

    def remove(self, node_id: int) -> None:
        self._vecs.pop(node_id, None)

    def query(self, vec: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        if not self._vecs:
            return [], []
        ids = list(self._vecs.keys())
        mat = np.stack([self._vecs[i] for i in ids])
        sims = mat @ vec  # inputs are L2-normalised by the caller -> dot == cosine
        order = np.argsort(-sims)[:k]
        return [ids[i] for i in order], sims[order].tolist()

    def __len__(self) -> int:
        return len(self._vecs)


class CosineAnnIndex:
    def __init__(self, dim: int, prefer: str = "auto", initial_capacity: int = 2000):
        if prefer in ("auto", "hnswlib") and HNSWLIB_AVAILABLE:
            self._impl = _HnswImpl(dim, initial_capacity)
        elif prefer == "hnswlib":
            raise RuntimeError("hnswlib requested but not importable")
        else:
            self._impl = _NumpyBruteForceImpl(dim, initial_capacity)

    @property
    def backend_name(self) -> str:
        return self._impl.name

    def add(self, node_id: int, vec: np.ndarray) -> None:
        self._impl.add(node_id, vec.astype(np.float32))

    def remove(self, node_id: int) -> None:
        self._impl.remove(node_id)

    def query(self, vec: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        return self._impl.query(vec.astype(np.float32), k)

    def __len__(self) -> int:
        return len(self._impl)
