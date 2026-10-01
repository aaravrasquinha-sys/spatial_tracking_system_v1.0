"""
WP-P3: the real `PlaceRecognizer` (architecture doc section 3's frozen
Protocol), replacing raw_match.py's direct-Hamming Phase-0 substitute.
Both channels behind ONE interface, run in parallel, disagreement
logged -- per the doc's own instruction for P3 ("Both behind the same
interface; run in parallel, log disagreement").

  - "bow_incremental": vpr/incremental_bow.py, the real (insert+delete)
    RTAB-Map vocabulary mechanism.
  - "learned": vpr/global_desc.py + vpr/ann_index.py, see global_desc's
    docstring for the honest scoping of "learned" here.
  - "both": runs both, returns a combined likelihood, and appends a
    disagreement record every call to `self.disagreement_log`.

`remove()` is real, not a stub -- called by pipeline.py whenever WP-P2
transfers a node out of WM, which is what keeps both channels' active
index scoped to "currently resident in STM/WM", matching RTAB-Map's
own behaviour (LTM nodes are not part of the fast retrieval index).
"""
from __future__ import annotations
import numpy as np

from pyslam.core.types import Node, Signature
from pyslam.core.config import Config
from pyslam.vpr.incremental_bow import IncrementalBow
from pyslam.vpr.global_desc import GlobalDescriptor
from pyslam.vpr.ann_index import CosineAnnIndex
from pyslam.vpr.likelihood import normalize_likelihood


class PlaceRecognizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mode = cfg.retrieval_backend
        assert self.mode in ("bow_incremental", "learned", "both"), (
            f"PlaceRecognizer only handles the P3 backends; 'raw' stays on "
            f"the Phase-0 path in pipeline.py. Got: {self.mode!r}"
        )
        self.use_bow = self.mode in ("bow_incremental", "both")
        self.use_learned = self.mode in ("learned", "both")

        self.bow = IncrementalBow(cfg.p3_new_word_hamming_threshold) if self.use_bow else None
        self.global_desc = GlobalDescriptor(out_dim=cfg.p3_global_desc_dim) if self.use_learned else None
        self.ann = (CosineAnnIndex(dim=cfg.p3_global_desc_dim, prefer=cfg.p3_ann_backend,
                                    initial_capacity=cfg.p3_ann_initial_capacity)
                    if self.use_learned else None)
        self._node_gdesc: dict[int, np.ndarray] = {}

        self.disagreement_log: list[dict] = []  # only populated in "both" mode

    # ------------------------------------------------------------ Protocol
    def add(self, node: Node) -> None:
        if self.use_bow:
            word_ids = self.bow.assign_and_insert(node.sig.desc)
            self.bow.add_node(node.id, word_ids)
            node.sig.word_ids = word_ids  # populate the frozen field for
                # anything downstream that reads it (audit, bookkeeping)
        if self.use_learned:
            gd = self.global_desc.compute(node.sig.desc)
            self.ann.add(node.id, gd)
            self._node_gdesc[node.id] = gd

    def remove(self, node_id: int) -> None:
        if self.use_bow:
            self.bow.remove_node(node_id)
        if self.use_learned:
            self.ann.remove(node_id)
            self._node_gdesc.pop(node_id, None)

    def likelihood(self, sig: Signature, candidates: list[int]) -> np.ndarray:
        """Returns len(candidates)+1, per the frozen Protocol."""
        bow_raw = self.bow.score(sig.word_ids, candidates) if self.use_bow else None
        learned_raw = self._learned_raw_scores(sig, candidates) if self.use_learned else None

        if self.mode == "bow_incremental":
            return normalize_likelihood(bow_raw)
        if self.mode == "learned":
            return normalize_likelihood(learned_raw)

        # "both": combine, and log disagreement between the two channels'
        # own independent top picks (each computed the same way a
        # single-channel mode would have).
        combined_raw = 0.5 * bow_raw + 0.5 * learned_raw if len(candidates) > 0 else bow_raw
        if len(candidates) > 0:
            bow_top = candidates[int(np.argmax(bow_raw))]
            learned_top = candidates[int(np.argmax(learned_raw))]
            self.disagreement_log.append({
                "n_candidates": len(candidates),
                "bow_top": bow_top, "bow_top_score": float(np.max(bow_raw)),
                "learned_top": learned_top, "learned_top_score": float(np.max(learned_raw)),
                "agree": bow_top == learned_top,
            })
        return normalize_likelihood(combined_raw)

    # ------------------------------------------------------------ helpers
    def _learned_raw_scores(self, sig: Signature, candidates: list[int]) -> np.ndarray:
        """Cosine similarity against exactly the given candidates (the
        Bayes filter already scoped these to WM) -- direct lookup of
        already-stored vectors, not an ANN query. `self.ann` is queried
        separately for the gate's latency benchmark
        (`ann_query_latency_ms`), which is the realistic usage pattern
        for an ANN index: generating candidates, not re-scoring a
        pre-given list."""
        if not candidates:
            return np.zeros(0, dtype=np.float64)
        q = self.global_desc.compute(sig.desc)
        out = np.zeros(len(candidates), dtype=np.float64)
        for i, cid in enumerate(candidates):
            v = self._node_gdesc.get(cid)
            if v is not None:
                out[i] = float(np.dot(q, v))  # both L2-normalised -> dot == cosine
        return out

    def ann_query_latency_ms(self, sig: Signature, k: int = 10) -> float:
        """Gate G3's "<5ms query at WM=1000" check, against whichever ANN
        backend is actually active (see `self.ann.backend_name`)."""
        import time
        q = self.global_desc.compute(sig.desc)
        t0 = time.perf_counter()
        self.ann.query(q, k)
        return (time.perf_counter() - t0) * 1000.0

    def audit_consistency(self) -> list[str]:
        problems = []
        if self.use_bow:
            problems.extend(f"[bow] {p}" for p in self.bow.audit_consistency())
        if self.use_learned:
            if len(self.ann) != len(self._node_gdesc):
                problems.append(f"[learned] ann index size {len(self.ann)} != "
                                 f"tracked gdesc count {len(self._node_gdesc)}")
        return problems
