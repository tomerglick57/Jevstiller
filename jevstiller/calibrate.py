"""Threshold selection under a disagreement budget, with a finite-sample bound."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k) for X ~ Binomial(n, p)."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 0.0 if k >= n else -math.inf
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided upper confidence bound on a binomial proportion: smallest p with P(X<=k|n,p) <= delta."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    log_delta = math.log(delta)
    for _ in range(60):
        mid = (lo + hi) / 2
        if _log_binom_cdf(k, n, mid) > log_delta:
            lo = mid
        else:
            hi = mid
    return hi


def clopper_pearson_lower(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided lower confidence bound on a binomial proportion."""
    if n <= 0 or k <= 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(n - k, n, delta)


@dataclass
class RoutingPolicy:
    conf_threshold: float | None      # None -> student never answers
    ood_threshold: float
    expected_coverage: float
    disagreement_ub: float            # upper bound on P(disagree | student answered)
    expected_system_disagreement: float   # coverage * ub, must be <= budget
    budget: float
    delta: float
    n_calib: int

    @property
    def usable(self) -> bool:
        return self.conf_threshold is not None

    def accepts(self, conf: np.ndarray, ood: np.ndarray) -> np.ndarray:
        if self.conf_threshold is None:
            return np.zeros(len(conf), dtype=bool)
        return (conf >= self.conf_threshold) & (ood <= self.ood_threshold)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RoutingPolicy":
        return cls(**json.loads(path.read_text()))


def fit_policy(conf: np.ndarray, agree: np.ndarray, ood: np.ndarray, budget: float,
               delta: float = 0.05, ood_quantile: float = 0.99, grid: int = 200) -> RoutingPolicy:
    """Maximise coverage subject to coverage * UB(selective disagreement) <= budget.

    Inputs are from an IID calibration set: student max-prob, agreement with the teacher
    (bool), and OOD score per example.
    """
    conf = np.asarray(conf, dtype=float)
    agree = np.asarray(agree, dtype=bool)
    ood = np.asarray(ood, dtype=float)
    N = len(conf)
    ood_thr = float(np.quantile(ood, ood_quantile)) if N else 0.0
    in_dist = ood <= ood_thr
    best = None
    if N:
        cands = np.unique(np.quantile(conf[in_dist], np.linspace(0, 1, grid))) if in_dist.any() else []
        for t in sorted(cands, reverse=True):
            sel = in_dist & (conf >= t)
            n = int(sel.sum())
            if n == 0:
                continue
            k = int((sel & ~agree).sum())
            ub = clopper_pearson_upper(k, n, delta)
            cov = n / N
            if cov * ub <= budget and (best is None or cov > best[1]):
                best = (float(t), cov, ub)
    if best is None:
        return RoutingPolicy(None, ood_thr, 0.0, 1.0, 0.0, budget, delta, N)
    t, cov, ub = best
    return RoutingPolicy(t, ood_thr, cov, ub, cov * ub, budget, delta, N)
