"""
Discrete Bayes filter, the RTAB-Map core mechanism: maintains a belief
over WM candidates PLUS an explicit "new place" state. Each step:
  1. PREDICT: every candidate's belief decays a small fraction toward
     "new place" (this is what makes a single-frame likelihood spike
     insufficient to sustain a loop-closure hypothesis -- without decay,
     a one-off high posterior would persist indefinitely).
  2. UPDATE: multiply by the incoming likelihood, renormalise.
  3. HYPOTHESIS: the arg-max candidate fires only if it beats "new place",
     clears the threshold H, and has done so for `hypothesis_hysteresis`
     consecutive updates.
"""
from __future__ import annotations
import numpy as np

from pyslam.core.types import Hypothesis
from pyslam.core.config import Config


class BayesFilter:
    def __init__(self, cfg: Config, decay_per_step: float = 0.15):
        self.cfg = cfg
        self.decay = decay_per_step
        self.belief: dict[int, float] = {}
        self.belief_new: float = 1.0
        self._last_best_id: int | None = None
        self._streak: int = 0

    def _ensure(self, candidate_ids: list[int]) -> None:
        """Seed brand-new candidates' belief by splitting it OFF of
        'new place' mass, rather than inserting a negligible epsilon and
        renormalising the whole distribution against it.

        The earlier version did the latter: insert belief[c]=1e-3, then
        divide everything (including belief_new) by the new total. Because
        WM grows by ~1 new candidate almost every keyframe, that 1e-3 seed
        is minuscule against belief_new (~1.0), so belief_new absorbs
        essentially all mass within a handful of updates regardless of any
        real evidence -- verified by tracing belief on the corridor-loop
        fixture, where belief_new reached >99% within 10 updates. Splitting
        mass OUT of 'new place' for each new candidate keeps the total
        exactly 1.0 without a global renormalisation, and treats 'new
        place' as what it conceptually is: an unallocated pool that
        specific candidates get carved out of as they become known.
        """
        new_ids = [c for c in candidate_ids if c not in self.belief]
        if not new_ids:
            return
        alloc = 0.5 * self.belief_new
        per_candidate = alloc / len(new_ids)
        for c in new_ids:
            self.belief[c] = per_candidate
        self.belief_new -= alloc

    def _predict(self, candidate_ids: list[int], neighbours=None) -> None:
        """PREDICT step. Always: every candidate decays `self.decay` of its
        belief into 'new place'. WP-L2, only when cfg.bayes_diffusion_enabled
        and `neighbours` (node_id -> iterable of neighbour ids) is given:
        additionally each candidate hands cfg.bayes_diffusion_rate of its
        belief to its neighbours that are themselves current candidates,
        split evenly (a candidate with no such neighbour keeps it). Every
        move is computed from the belief as it stood BEFORE this step, so the
        result does not depend on iteration order, and total mass is
        conserved exactly (asserted by gate GB.2)."""
        rate = self.cfg.bayes_diffusion_rate if (self.cfg.bayes_diffusion_enabled and neighbours is not None) else 0.0
        old = {c: self.belief.get(c, 0.0) for c in candidate_ids}
        cand = set(candidate_ids)
        new_belief = {}
        moved_to_new = 0.0
        diffused_in = {c: 0.0 for c in candidate_ids}
        for c in candidate_ids:
            b = old[c]
            to_new = b * self.decay
            moved_to_new += to_new
            stay = b - to_new
            if rate > 0.0 and stay > 0.0:
                nbrs = [n for n in neighbours(c) if n in cand and n != c]
                if nbrs:
                    out = min(b * rate, stay)   # never move more than is left after decay
                    for n in nbrs:
                        diffused_in[n] += out / len(nbrs)
                    stay -= out
            new_belief[c] = stay
        for c in candidate_ids:
            self.belief[c] = new_belief[c] + diffused_in[c]
        self.belief_new += moved_to_new

    def update(self, likelihood: np.ndarray, candidate_ids: list[int], neighbours=None) -> Hypothesis | None:
        if len(candidate_ids) == 0:
            return None
        assert likelihood.shape[0] == len(candidate_ids) + 1, \
            "likelihood must be len(candidates)+1 (last entry = new place)"

        self._ensure(candidate_ids)
        self._predict(candidate_ids, neighbours)

        post = {}
        total = 0.0
        for i, cid in enumerate(candidate_ids):
            v = self.belief.get(cid, 0.0) * float(likelihood[i])
            post[cid] = v
            total += v
        v_new = self.belief_new * float(likelihood[-1])
        total += v_new
        total = max(total, 1e-300)

        for cid in candidate_ids:
            self.belief[cid] = post[cid] / total
        self.belief_new = v_new / total

        best_id = max(candidate_ids, key=lambda c: self.belief[c])
        best_p = self.belief[best_id]

        qualifies = (best_p > self.belief_new) and (best_p >= self.cfg.hypothesis_threshold)
        if qualifies and best_id == self._last_best_id:
            self._streak += 1
        elif qualifies:
            self._last_best_id = best_id
            self._streak = 1
        else:
            self._last_best_id = None
            self._streak = 0

        if qualifies and self._streak >= self.cfg.hypothesis_hysteresis:
            return Hypothesis(node_id=best_id, posterior=best_p, n_consecutive=self._streak)
        return None

    def belief_sum(self) -> float:
        return sum(self.belief.values()) + self.belief_new
