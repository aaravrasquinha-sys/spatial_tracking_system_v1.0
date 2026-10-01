"""
TF-IDF scoring over a fixed vocabulary. Deliberately simple: no
incremental ANN index (Phase 0's fixed vocabulary makes exact matvec
scoring cheap enough that we don't need one).
"""
from __future__ import annotations
import numpy as np
from scipy import sparse

from pyslam.core.config import Config


class BowIndex:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._counts: dict[int, np.ndarray] = {}   # node_id -> (vocab_size,) float32 term counts

    def add(self, node_id: int, word_ids: np.ndarray) -> None:
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        if word_ids is not None and len(word_ids) > 0:
            valid = word_ids[word_ids >= 0]
            if len(valid) > 0:
                bc = np.bincount(valid, minlength=self.vocab_size)[: self.vocab_size]
                vec[: len(bc)] = bc
        self._counts[node_id] = vec

    def remove(self, node_id: int) -> None:
        self._counts.pop(node_id, None)

    def _idf(self, candidate_ids: list[int]) -> np.ndarray:
        n_docs = max(len(candidate_ids), 1)
        df = np.zeros(self.vocab_size, dtype=np.float64)
        for cid in candidate_ids:
            vec = self._counts.get(cid)
            if vec is not None:
                df += (vec > 0)
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return idf

    def score(self, query_word_ids: np.ndarray, candidate_ids: list[int]) -> np.ndarray:
        """Cosine TF-IDF similarity between the query and each candidate.
        Returns (len(candidate_ids),) float64 in [0,1] (approximately)."""
        if len(candidate_ids) == 0:
            return np.zeros(0, dtype=np.float64)
        idf = self._idf(candidate_ids)

        q_vec = np.zeros(self.vocab_size, dtype=np.float64)
        if query_word_ids is not None and len(query_word_ids) > 0:
            valid = query_word_ids[query_word_ids >= 0]
            if len(valid) > 0:
                bc = np.bincount(valid, minlength=self.vocab_size)[: self.vocab_size]
                q_vec[: len(bc)] = bc
        q_tfidf = q_vec * idf
        qn = np.linalg.norm(q_tfidf)

        mat = np.zeros((len(candidate_ids), self.vocab_size), dtype=np.float64)
        for i, cid in enumerate(candidate_ids):
            vec = self._counts.get(cid)
            if vec is not None:
                mat[i] = vec.astype(np.float64) * idf

        norms = np.linalg.norm(mat, axis=1)
        denom = norms * qn
        scores = np.zeros(len(candidate_ids), dtype=np.float64)
        nz = denom > 1e-12
        scores[nz] = (mat[nz] @ q_tfidf) / denom[nz]
        return scores
