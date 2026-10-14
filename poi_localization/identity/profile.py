"""Profile accumulators (R1 spec sections 7.2, 7.4, 9): robust running statistics and appearance EMA.

RobustStat: inverse-variance weighted running mean/variance with a Huber clip at 3 sigma against the current
mean once >= 5 samples exist, and an effective-sample cap (serially correlated frames are throttled to at most
`max_rate_hz` accepted samples per second). Deterministic: time comes from the caller.
AppearanceAccumulator: per view bin x band, an exponentially weighted average of normalised histograms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RobustStat:
    n: int = 0
    sum_w: float = 0.0
    sum_wx: float = 0.0
    sum_wxx: float = 0.0
    last_t: Optional[float] = None

    @property
    def mean(self) -> Optional[float]:
        return self.sum_wx / self.sum_w if self.n and self.sum_w > 0 else None

    @property
    def n_eff(self) -> int:
        return self.n

    @property
    def sigma_sensor_mean(self) -> Optional[float]:
        """Uncertainty of the mean from the sensor sigmas alone (1/sqrt(sum w))."""
        return float(1.0 / np.sqrt(self.sum_w)) if self.sum_w > 0 else None

    @property
    def spread(self) -> Optional[float]:
        """Weighted standard deviation of the accepted samples (the within-person wobble), None until n >= 2."""
        if self.n < 2 or self.sum_w <= 0:
            return None
        m = self.sum_wx / self.sum_w
        var = max(self.sum_wxx / self.sum_w - m * m, 0.0)
        return float(np.sqrt(var * self.n / (self.n - 1)))

    def sigma_mean(self, sigma_within: float) -> Optional[float]:
        """Uncertainty of the person's mean using the LEARNED within-person spread (clothing/keypoint bias does
        not average away): sigma_within / sqrt(n_eff), never below the pure sensor term."""
        if self.n == 0:
            return None
        return float(max(sigma_within / np.sqrt(self.n), self.sigma_sensor_mean or 0.0))

    def add(self, x: float, sigma: float, t: float, max_rate_hz: float = 3.0) -> bool:
        """Add a sample; returns False if throttled (too soon after the last accepted one)."""
        if self.last_t is not None and (t - self.last_t) < 1.0 / max_rate_hz:
            return False
        sigma = max(float(sigma), 1e-6)
        if self.n >= 5:
            s = max(sigma, self.spread or 0.0)
            m = self.mean
            x = float(np.clip(x, m - 3 * s, m + 3 * s))          # Huber-style clip against the current mean
        w = 1.0 / (sigma * sigma)
        self.n += 1
        self.sum_w += w
        self.sum_wx += w * x
        self.sum_wxx += w * x * x
        self.last_t = t
        return True

    def to_dict(self) -> dict:
        return {"n": self.n, "sum_w": self.sum_w, "sum_wx": self.sum_wx, "sum_wxx": self.sum_wxx, "last_t": self.last_t}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustStat":
        return cls(**d)


class StructureModel:
    """A person's structure features: one RobustStat per feature name."""

    def __init__(self, max_rate_hz: float = 3.0):
        self.stats: Dict[str, RobustStat] = {}
        self.max_rate_hz = max_rate_hz

    def add(self, features: Dict[str, Tuple[float, float]], t: float) -> int:
        """Accept a {name: (value, sigma)} dict; returns how many features were accepted (not throttled)."""
        return sum(self.stats.setdefault(k, RobustStat()).add(v, s, t, self.max_rate_hz) for k, (v, s) in features.items())

    def mean(self, name: str) -> Optional[float]:
        st = self.stats.get(name)
        return None if st is None else st.mean

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.stats.items()}

    @classmethod
    def from_dict(cls, d: dict, max_rate_hz: float = 3.0) -> "StructureModel":
        m = cls(max_rate_hz)
        m.stats = {k: RobustStat.from_dict(v) for k, v in d.items()}
        return m


class AppearanceAccumulator:
    """EMA of banded histograms, keyed by (view, band); `view` None/"any" accumulates into the "any" aggregate."""

    def __init__(self, halflife_s: float = 10.0, n_bins: int = 52):
        self.halflife_s = halflife_s
        self.n_bins = n_bins
        self.hist: Dict[Tuple[str, int], np.ndarray] = {}
        self.mass: Dict[Tuple[str, int], float] = {}
        self._last_t: Dict[Tuple[str, int], float] = {}

    def _update(self, key, h: np.ndarray, mass: float, t: float) -> None:
        if key not in self.hist:
            self.hist[key] = h.astype(np.float32).copy()
            self.mass[key] = float(mass)
        else:
            dt = max(t - self._last_t[key], 0.0)
            a = 1.0 - 0.5 ** (dt / self.halflife_s) if self.halflife_s > 0 else 1.0
            a = max(a, 0.02)                                     # never freeze completely on back-to-back samples
            self.hist[key] = ((1 - a) * self.hist[key] + a * h).astype(np.float32)
            self.mass[key] += float(mass)
        self._last_t[key] = t

    def add(self, hists: Dict[int, np.ndarray], mass: Dict[int, int], view: Optional[str], t: float) -> None:
        for b, h in hists.items():
            self._update(("any", b), h, mass[b], t)
            if view in ("front", "back", "side"):
                self._update((view, b), h, mass[b], t)

    def get(self, view: str, band: int) -> Optional[np.ndarray]:
        return self.hist.get((view, band))

    def to_dict(self) -> dict:
        return {f"{k[0]}:{k[1]}": {"hist": v.tolist(), "mass": self.mass[k], "t": self._last_t[k]}
                for k, v in self.hist.items()}

    @classmethod
    def from_dict(cls, d: dict, halflife_s: float = 10.0, n_bins: int = 52) -> "AppearanceAccumulator":
        a = cls(halflife_s, n_bins)
        for key, v in d.items():
            view, band = key.split(":")
            k = (view, int(band))
            a.hist[k] = np.array(v["hist"], dtype=np.float32)
            a.mass[k] = v["mass"]
            a._last_t[k] = v["t"]
        return a
