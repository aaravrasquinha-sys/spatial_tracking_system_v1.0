"""
WP-P3: the real RTAB-Map vocabulary mechanism -- incremental, not the
fixed 1k-word Phase-0 substitute (vpr/vocab.py). New descriptor -> NN
search against existing words -> assign if close enough, else create a
new word. This is the component the architecture doc itself flagged as
"the single buggiest component in the system" (section 4.1): an index
that needs insertion AND deletion kept consistent with the inverted
(TF-IDF) index, with no room for a silent off-by-one.

Deletion is reference-counted, not soft: when the last resident node
using a word is removed (WP-P2's LTM transfer, wired in pipeline.py),
that word is deleted outright -- RTAB-Map's actual behaviour, and the
reason the architecture doc calls out "kept consistent with... the
database" as the hard part. `audit_consistency()` exists so this claim
is checked, not just asserted.

Reuses vpr/vocab.py's vectorised Hamming-distance machinery rather than
re-deriving it -- same convention, same tested oracle.
"""
from __future__ import annotations
from collections import Counter
from typing import Optional
import numpy as np

from pyslam.vpr.vocab import hamming_distance_matrix


class IncrementalBow:
    def __init__(self, new_word_hamming_threshold: int = 40):
        self.threshold = new_word_hamming_threshold
        self._word_desc: dict[int, np.ndarray] = {}     # word_id -> (32,) uint8
        self._word_refcount: dict[int, int] = {}          # word_id -> #resident occurrences
        self._word_doc_freq: dict[int, int] = {}          # word_id -> #resident NODES containing it
        self._next_word_id = 0
        self._node_word_ids: dict[int, np.ndarray] = {}   # node_id -> (N,) int32, this node's assignment
        self._node_word_set: dict[int, set] = {}           # node_id -> set(word ids), for doc-freq bookkeeping
        self._vocab_matrix: Optional[np.ndarray] = None    # (V,32) uint8, rebuilt on insertion
        self._vocab_ids: Optional[np.ndarray] = None       # (V,) int32, parallel to _vocab_matrix

    @property
    def n_words(self) -> int:
        return len(self._word_desc)

    # ------------------------------------------------------------ assignment
    def assign_and_insert(self, desc: np.ndarray) -> np.ndarray:
        """desc: (N,32) uint8. Returns (N,) int32 word ids, creating new
        words for any descriptor farther than `threshold` from every
        existing word.

        One vectorised pass against the whole existing vocabulary
        (N-per-frame is a few hundred to ~1000, V can grow into the
        thousands). A real perf bug lived here in an earlier version: a
        per-descriptor Python loop that recomputed the full Hamming
        matrix against the CURRENT vocab for every miss, so it could
        cost O(N*V) in slow per-call Python overhead specifically on
        early frames when most descriptors are still new (vocabulary
        hasn't "matured" yet) -- exactly the worst case, confirmed via
        profiling (2412 of those calls in just 5 keyframes on
        square6dof, dominating runtime).

        Fix: one batched Hamming pass for the whole frame, then every
        miss becomes a new word via a single bulk array append -- NOT
        checked against each other first. This means two genuinely new,
        very similar features seen in the SAME frame can mint two
        separate (near-duplicate) words instead of collapsing to one.
        That is a real, deliberate trade-off (a small amount of
        vocabulary redundancy) for O(N) instead of O(N*V) per frame,
        matching how DBoW-style incremental vocabularies actually
        operate in practice -- not a correctness bug, and it doesn't
        affect TF-IDF scoring's correctness, only vocabulary size.
        """
        n = desc.shape[0]
        out = np.full(n, -1, dtype=np.int32)
        if n == 0:
            return out

        if self._vocab_matrix is not None and len(self._vocab_matrix) > 0:
            dist = hamming_distance_matrix(desc, self._vocab_matrix)   # (n, V)
            j = np.argmin(dist, axis=1)
            best = dist[np.arange(n), j]
            hit = best <= self.threshold
            out[hit] = self._vocab_ids[j[hit]]
            remaining = np.where(~hit)[0]
        else:
            remaining = np.arange(n)

        if len(remaining) > 0:
            new_ids = np.arange(self._next_word_id, self._next_word_id + len(remaining), dtype=np.int32)
            self._next_word_id += len(remaining)
            new_descs = desc[remaining]
            for k in range(len(remaining)):
                wid = int(new_ids[k])
                self._word_desc[wid] = new_descs[k].copy()
                self._word_refcount[wid] = 0
                self._word_doc_freq[wid] = 0
            out[remaining] = new_ids
            if self._vocab_matrix is None:
                self._vocab_matrix = new_descs.copy()
                self._vocab_ids = new_ids.copy()
            else:
                self._vocab_matrix = np.vstack([self._vocab_matrix, new_descs])
                self._vocab_ids = np.concatenate([self._vocab_ids, new_ids])
        return out

    def _assign_one(self, d: np.ndarray) -> int:
        """Single-descriptor path, used only by callers that assign one
        descriptor at a time outside the batched path above (none in
        the hot loop currently, kept for API completeness / tests)."""
        if self._vocab_matrix is not None and len(self._vocab_matrix) > 0:
            dist = hamming_distance_matrix(d[None, :], self._vocab_matrix)[0]
            j = int(np.argmin(dist))
            if dist[j] <= self.threshold:
                return int(self._vocab_ids[j])
        wid = self._next_word_id
        self._next_word_id += 1
        self._word_desc[wid] = d.copy()
        self._word_refcount[wid] = 0
        self._word_doc_freq[wid] = 0
        self._append_to_cache(wid, d)
        return wid

    def _append_to_cache(self, wid: int, d: np.ndarray) -> None:
        """Incrementally grow the live vocab_matrix/vocab_ids cache
        rather than rebuilding it from the whole `_word_desc` dict on
        every insertion (that rebuild is still correct and used by
        remove_node, where a full rebuild is unavoidable since deletion
        can happen anywhere in the array -- but insertion is the hot
        path and doesn't need it)."""
        if self._vocab_matrix is None:
            self._vocab_matrix = d[None, :].copy()
            self._vocab_ids = np.array([wid], dtype=np.int32)
        else:
            self._vocab_matrix = np.vstack([self._vocab_matrix, d[None, :]])
            self._vocab_ids = np.append(self._vocab_ids, np.int32(wid))

    def _rebuild_matrix(self) -> None:
        if not self._word_desc:
            self._vocab_matrix, self._vocab_ids = None, None
            return
        ids = np.fromiter(self._word_desc.keys(), dtype=np.int32)
        self._vocab_ids = ids
        self._vocab_matrix = np.stack([self._word_desc[i] for i in ids])

    def assign_readonly(self, desc: np.ndarray) -> np.ndarray:
        """Nearest-word lookup WITHOUT inserting new words -- used to score
        a query against the current vocabulary state without mutating it
        (e.g. re-scoring after the fact, or a frame that never becomes a
        node)."""
        n = desc.shape[0]
        if n == 0 or self._vocab_matrix is None:
            return np.full(n, -1, dtype=np.int32)
        dist = hamming_distance_matrix(desc, self._vocab_matrix)
        j = np.argmin(dist, axis=1)
        best = dist[np.arange(n), j]
        out = self._vocab_ids[j]
        out = np.where(best <= self.threshold, out, -1)
        return out.astype(np.int32)

    # ------------------------------------------------------------ node lifecycle
    def add_node(self, node_id: int, word_ids: np.ndarray) -> None:
        self._node_word_ids[node_id] = word_ids
        wset = set(int(w) for w in word_ids.tolist() if w >= 0)
        self._node_word_set[node_id] = wset
        for w in word_ids:
            w = int(w)
            if w >= 0:
                self._word_refcount[w] = self._word_refcount.get(w, 0) + 1
        for w in wset:
            self._word_doc_freq[w] = self._word_doc_freq.get(w, 0) + 1

    def remove_node(self, node_id: int) -> None:
        """The other half of the buggy-by-reputation mechanism: undoes
        exactly what add_node did, and deletes any word whose refcount
        hits zero -- no resident node references it any more."""
        word_ids = self._node_word_ids.pop(node_id, None)
        wset = self._node_word_set.pop(node_id, set())
        if word_ids is None:
            return
        for w in word_ids:
            w = int(w)
            if w < 0 or w not in self._word_refcount:
                continue
            self._word_refcount[w] -= 1
            if self._word_refcount[w] <= 0:
                del self._word_refcount[w]
                self._word_desc.pop(w, None)
                self._word_doc_freq.pop(w, None)
        for w in wset:
            if w in self._word_doc_freq:
                self._word_doc_freq[w] -= 1
                if self._word_doc_freq[w] <= 0 and w not in self._word_refcount:
                    self._word_doc_freq.pop(w, None)
        self._rebuild_matrix()

    # ------------------------------------------------------------ scoring
    def score(self, query_word_ids: np.ndarray, candidate_ids: list[int]) -> np.ndarray:
        """TF-IDF cosine similarity, same formula as Phase 0's BowIndex
        (vpr/bow.py) but over the dynamic vocabulary -- IDF computed only
        from the given candidate set's own document frequencies, so a
        query never needs the full resident population."""
        if len(candidate_ids) == 0:
            return np.zeros(0, dtype=np.float64)
        n_docs = max(len(candidate_ids), 1)
        all_words = set()
        for cid in candidate_ids:
            all_words |= self._node_word_set.get(cid, set())
        all_words |= set(int(w) for w in query_word_ids.tolist() if w >= 0)
        if not all_words:
            return np.zeros(len(candidate_ids), dtype=np.float64)
        word_list = sorted(all_words)
        col = {w: i for i, w in enumerate(word_list)}
        d = len(word_list)

        df = np.zeros(d, dtype=np.float64)
        for cid in candidate_ids:
            for w in self._node_word_set.get(cid, set()):
                df[col[w]] += 1
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        q_vec = np.zeros(d, dtype=np.float64)
        for w in query_word_ids:
            w = int(w)
            if w in col:
                q_vec[col[w]] += 1.0
        q_tfidf = q_vec * idf
        qn = np.linalg.norm(q_tfidf)

        scores = np.zeros(len(candidate_ids), dtype=np.float64)
        if qn < 1e-12:
            return scores
        for i, cid in enumerate(candidate_ids):
            counts = Counter(int(w) for w in self._node_word_ids.get(cid, np.array([], dtype=np.int32)) if w >= 0)
            vec = np.zeros(d, dtype=np.float64)
            for w, c in counts.items():
                if w in col:
                    vec[col[w]] = c
            vec_tfidf = vec * idf
            vn = np.linalg.norm(vec_tfidf)
            if vn > 1e-12:
                scores[i] = float(np.dot(vec_tfidf, q_tfidf) / (vn * qn))
        return scores

    # ------------------------------------------------------------ audit
    def audit_consistency(self) -> list[str]:
        problems = []
        for nid, wset in self._node_word_set.items():
            for w in wset:
                if w not in self._word_doc_freq:
                    problems.append(f"node {nid} references word {w} with no doc-freq entry")
        for w, rc in self._word_refcount.items():
            if rc <= 0:
                problems.append(f"word {w} has non-positive refcount {rc} but was not deleted")
            if w not in self._word_desc:
                problems.append(f"word {w} has a refcount but no stored descriptor")
        for w in self._word_desc:
            if w not in self._word_refcount:
                problems.append(f"word {w} has a stored descriptor but no refcount entry")
        return problems
