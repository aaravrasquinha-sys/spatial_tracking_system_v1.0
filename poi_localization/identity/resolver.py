"""Cue likelihoods, open-set posterior and commit rule (R1 spec sections 9 and 10). Pure maths, no I/O.

Evidence for "this track is missing person k" is a log-likelihood ratio against "a stranger":
  * structure: per feature, Gaussian same-person vs different-person on the difference of running means
  * appearance: per height band, Bhattacharyya similarity of the matching-view histograms, with a clothes-change
    mixture (an upper-body change is not evidence against, a lower-body change much less so)
Cues are correlated (limb lengths share the keypoint model, bands share lighting), so each cue's summed LLR is
scaled by CUE_SCALE and clipped to +-llr_clip. The constants below are PLACEHOLDERS until Phase 0 fits them
(configs/identity_priors.default.json); the per-session calibration from free positive/negative pairs is a later batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from poi_localization.identity.config import IdentityConfig
from poi_localization.identity.extract_appearance import LOWER_BANDS, UPPER_BANDS, bhattacharyya
from poi_localization.identity.extract_structure import SOFT_FEATURES
from poi_localization.identity.profile import AppearanceAccumulator, StructureModel

# population spread of each feature across people / within-person wobble of its running estimate (metres)
SIGMA_POP = {"stature": 0.07, "upper_arm": 0.02, "forearm": 0.02, "thigh": 0.03, "shank": 0.03,
             "torso": 0.03, "hip_width": 0.03, "shoulder_width": 0.03}
SIGMA_WITHIN = {"stature": 0.015, "upper_arm": 0.012, "forearm": 0.012, "thigh": 0.014, "shank": 0.014,
                "torso": 0.018, "hip_width": 0.02, "shoulder_width": 0.02}
CUE_SCALE = 0.5
APP_SLOPE, APP_S0, APP_BAND_CLIP = 12.0, 0.75, 3.0
MIN_FEATURE_N = 3
MIN_BAND_MASS = 40


def structure_llr(case: StructureModel, cand: StructureModel, cfg: IdentityConfig) -> Tuple[float, int]:
    total, n = 0.0, 0
    for name, sc in case.stats.items():
        sp = cand.stats.get(name)
        if sp is None or sc.n < MIN_FEATURE_N or sp.n < MIN_FEATURE_N or name not in SIGMA_POP:
            continue
        sw = SIGMA_WITHIN[name]
        extra = cfg.soft_feature_extra_sd_m ** 2 if name in SOFT_FEATURES else 0.0
        var_same = sc.sigma_mean(sw) ** 2 + sp.sigma_mean(sw) ** 2 + extra
        var_diff = var_same + 2.0 * SIGMA_POP[name] ** 2
        d = sc.mean - sp.mean
        total += -0.5 * d * d / var_same - 0.5 * np.log(var_same) + 0.5 * d * d / var_diff + 0.5 * np.log(var_diff)
        n += 1
    return float(np.clip(total * CUE_SCALE, -cfg.llr_clip, cfg.llr_clip)), n


def _pick_view(case: AppearanceAccumulator, cand: AppearanceAccumulator, band: int) -> Optional[str]:
    for view in ("front", "back", "side"):
        if (view, band) in case.hist and (view, band) in cand.hist:
            return view
    return "any" if ("any", band) in case.hist and ("any", band) in cand.hist else None


def appearance_llr(case: AppearanceAccumulator, cand: AppearanceAccumulator, cfg: IdentityConfig) -> Tuple[float, int]:
    total, n = 0.0, 0
    for band in range(len(cfg.bands) - 1):
        view = _pick_view(case, cand, band)
        if view is None:
            continue
        if min(case.mass[(view, band)], cand.mass[(view, band)]) < MIN_BAND_MASS:
            continue
        s = bhattacharyya(case.hist[(view, band)], cand.hist[(view, band)])
        l = float(np.clip(APP_SLOPE * (s - APP_S0), -APP_BAND_CLIP, APP_BAND_CLIP))
        rho = cfg.rho_upper_change if band in UPPER_BANDS else cfg.rho_lower_change
        total += float(np.log((1 - rho) * np.exp(l) + rho))
        n += 1
    return float(np.clip(total * CUE_SCALE, -cfg.llr_clip, cfg.llr_clip)), n


@dataclass
class Decision:
    kind: str                      # "matched" | "new" | "ambiguous"
    person_id: Optional[int]       # matched person / best candidate (ambiguous)
    conf: float
    reason: str


def posterior(llrs: Dict[int, float], pi_new: float) -> Dict[object, float]:
    """Open-set posterior over {candidate ids} U {"new"}. Candidates share (1 - pi_new) evenly."""
    k = len(llrs)
    if k == 0:
        return {"new": 1.0}
    logp = {pid: np.log((1 - pi_new) / k) + l for pid, l in llrs.items()}
    logp["new"] = np.log(pi_new)
    m = max(logp.values())
    z = sum(np.exp(v - m) for v in logp.values())
    return {key: float(np.exp(v - m) / z) for key, v in logp.items()}


def decide(llrs: Dict[int, float], n_obs: int, elapsed_s: float, cfg: IdentityConfig) -> Optional[Decision]:
    """None = keep resolving."""
    post = posterior(llrs, cfg.pi_new)
    best = max(llrs, key=lambda p: (llrs[p], -p)) if llrs else None
    enough = n_obs >= cfg.n_min_obs and elapsed_s >= cfg.t_min_s
    if enough and best is not None:
        second = max([l for p, l in llrs.items() if p != best], default=None)
        margin_ok = second is None or llrs[best] - second >= cfg.min_margin
        if post[best] >= cfg.p_commit and margin_ok:
            return Decision("matched", best, post[best], f"posterior {post[best]:.3f}")
    if enough and post["new"] >= cfg.p_commit_new:
        return Decision("new", None, post["new"], f"posterior(new) {post['new']:.3f}")
    if elapsed_s >= cfg.t_max_resolve_s:
        if n_obs < cfg.n_min_obs or best is None or post["new"] >= post[best]:
            return Decision("new", None, post["new"], "timeout: no candidate more likely than new")
        return Decision("ambiguous", best, post[best], "timeout: not decisive")
    return None
