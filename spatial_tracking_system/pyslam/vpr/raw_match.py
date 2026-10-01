"""
Phase-0 retrieval scoring.

The fixed 1000-word vocabulary (vpr/vocab.py, vpr/bow.py) is kept in the
codebase as the scaffolding for Phase 3's proper incremental-vocabulary
retrieval, but it is NOT used to drive loop-closure decisions in Phase 0.

Why: verified directly on the corridor-loop synthetic fixture that with
~800 valid descriptors per keyframe quantised into only 1000 words, word
overlap between ANY two keyframes sits in a narrow band (~170-190 words)
almost regardless of whether they are a true match or an unrelated
location -- the vocabulary doesn't have enough capacity at this WM scale
to preserve the discriminative signal that raw Hamming matching clearly
has (41 good matches for the true loop pair vs 1-4 for unrelated ones,
on the same fixture). A quantisation layer that erases the signal it was
meant to compress is worse than no quantisation.

So Phase 0 scores retrieval directly: one batched BFMatcher call between
the query's descriptors and the concatenation of every WM candidate's
descriptors (not one call per candidate -- WM can be a few hundred nodes
and per-candidate matching does not scale even for Phase 0's "WM stays
small" assumption). Each good ratio-test match increments its owning
candidate's count; counts are normalised by that candidate's own
descriptor count. This is O(1) matcher calls per keyframe rather than
O(|WM|), and is what the Bayes filter and verifier are actually tuned
against in this phase.
"""
from __future__ import annotations
import numpy as np
import cv2

from pyslam.core.types import Signature, Node


def score_candidates(query_sig: Signature, candidate_nodes: list[Node],
                      ratio: float = 0.75) -> np.ndarray:
    n = len(candidate_nodes)
    scores = np.zeros(n, dtype=np.float64)
    if n == 0 or query_sig.desc.shape[0] == 0:
        return scores

    all_desc = []
    owner = []
    for idx, node in enumerate(candidate_nodes):
        d = node.sig.desc
        if d.shape[0] == 0:
            continue
        all_desc.append(d)
        owner.append(np.full(d.shape[0], idx, dtype=np.int32))
    if not all_desc:
        return scores
    all_desc = np.concatenate(all_desc, axis=0)
    owner = np.concatenate(owner, axis=0)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(query_sig.desc, all_desc, k=2)
    counts = np.zeros(n, dtype=np.float64)
    for pair in knn:
        if len(pair) < 2:
            continue
        m, nn = pair
        if m.distance < ratio * nn.distance:
            counts[owner[m.trainIdx]] += 1.0

    denom = np.array([max(node.sig.desc.shape[0], 1) for node in candidate_nodes], dtype=np.float64)
    scores = counts / denom
    return scores
