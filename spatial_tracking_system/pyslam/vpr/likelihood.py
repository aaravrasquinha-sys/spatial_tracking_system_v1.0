"""
RTAB-Map-style likelihood normalisation.

Given raw similarity scores over the WM candidates, converts them to a
likelihood vector where an outlier-high score stands out (via a z-score
against the population), and appends an explicit "new place" hypothesis
so that a mediocre-everywhere frame does not spuriously win against
"I've never seen this before". This is the mechanism that makes the
downstream Bayes filter meaningful; get it wrong and either everything
looks like a loop closure or nothing ever does.
"""
from __future__ import annotations
import numpy as np


def normalize_likelihood(raw_scores: np.ndarray) -> np.ndarray:
    """raw_scores: (K,) similarity in [0,1] for K WM candidates.
    Returns (K+1,) unnormalised likelihoods; last entry is 'new place'.

    IMPORTANT: 'new place' uses a CONSTANT baseline (1.0), the same
    baseline every non-distinguished old location gets. It must not track
    max(raw_scores) -- an earlier version of this function set
    L_new = z_max + 1, which is *always* >= any candidate's own z-score
    and therefore mathematically guaranteed to beat the true match every
    time, silently disabling loop closure. The correct mechanism is: an
    old location only stands out (likelihood > 1) if its score is a
    genuine outlier (>= mean + 1 std) against the rest of the population;
    'new place' sits at the same baseline as an unremarkable old location,
    so a real match can beat it, and a population with no standout
    candidate leaves 'new place' competitive."""
    k = raw_scores.shape[0]
    if k == 0:
        return np.array([1.0])

    mu = float(np.mean(raw_scores))
    sigma = float(np.std(raw_scores))

    if sigma < 1e-9:
        return np.concatenate([np.ones(k), [1.0]])

    L = np.ones(k, dtype=np.float64)
    above = raw_scores >= (mu + sigma)
    L[above] = (raw_scores[above] - mu) / sigma

    L_new = 1.0
    return np.concatenate([L, [L_new]])
