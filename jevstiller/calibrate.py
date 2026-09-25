"""Threshold selection under a disagreement budget, with a finite-sample bound."""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
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
    expected_coverage: float          # share of calibration rows the student answers
    disagreement_ub: float            # upper bound (at 1 - delta) on P(student answers and disagrees), per request
    expected_system_disagreement: float   # that rate on the calibration rows (the bound is what meets the budget)
    budget: float
    delta: float
    n_calib: int
    deferred_labels: list[str] = field(default_factory=list)   # the student never answers these (rare classes)

    @property
    def usable(self) -> bool:
        return self.conf_threshold is not None

    def accepts(self, conf: np.ndarray, ood: np.ndarray, pred: np.ndarray | None = None) -> np.ndarray:
        """Which rows the student answers. `pred` (the student's predicted labels) is required when the
        policy defers labels."""
        if self.conf_threshold is None:
            return np.zeros(len(conf), dtype=bool)
        ok = (conf >= self.conf_threshold) & (ood <= self.ood_threshold)
        if self.deferred_labels:
            if pred is None:
                raise ValueError("this policy defers labels; pass the predicted labels")
            ok &= ~np.isin(np.asarray(pred, dtype=object), self.deferred_labels)
        return ok

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> RoutingPolicy:
        return cls(**json.loads(path.read_text()))


def threshold_grid(n_labels: int, size: int = 400) -> np.ndarray:
    """Candidate confidence thresholds, strictest first. Fixed before any calibration row is seen, so testing
    them in sequence costs no confidence. Dense near 1, where a student's usable answers are."""
    floor = 1.0 / max(n_labels, 2)
    return 1.0 - np.geomspace(1e-4, 1.0 - floor, size)


def fit_policy(conf: np.ndarray, agree: np.ndarray, ood: np.ndarray, budget: float, delta: float = 0.05,
               ood_threshold: float = math.inf, candidates: np.ndarray | None = None,
               eligible: np.ndarray | None = None, deferred_labels: Sequence[str] = ()) -> RoutingPolicy:
    """The loosest confidence threshold whose rate of *answered and disagreeing* requests is at most `budget`,
    with probability at least 1 - `delta`.

    Rows come from an IID calibration set: the student's max-prob, whether it agrees with the teacher, and its
    OOD score. The loss per row is 1[the student answers and disagrees], so its rate over all rows is exactly
    what the budget limits (disagreement over all requests).

    Candidates are tested strictest first with an exact Clopper-Pearson bound at `delta`, and the scan stops at
    the first failure (fixed-sequence testing, as in "Learn Then Test"). The loss can only grow as the
    threshold loosens, so the chance that the chosen threshold breaks the budget is at most `delta`, with no
    multiple-testing penalty. That needs everything else fixed without these rows: `candidates` (default: a
    fixed grid) and `ood_threshold` (the caller picks it on training data). `eligible` marks rows the student
    may answer at all (False where it predicts one of `deferred_labels`); the rate still counts every row.
    """
    conf = np.asarray(conf, dtype=float)
    agree = np.asarray(agree, dtype=bool)
    ood = np.asarray(ood, dtype=float)
    N = len(conf)
    in_dist = ood <= ood_threshold
    if eligible is not None:
        in_dist &= np.asarray(eligible, dtype=bool)
    grid = threshold_grid(2) if candidates is None else np.asarray(candidates, dtype=float)
    deferred = list(deferred_labels)
    best = None
    if N:
        for t in sorted(grid, reverse=True):
            sel = in_dist & (conf >= t)
            k = int((sel & ~agree).sum())
            ub = clopper_pearson_upper(k, N, delta)
            if ub > budget:
                break                                    # fixed sequence: every looser threshold is untested
            if sel.any():                                # a threshold that answers nothing is no policy
                best = (float(t), int(sel.sum()) / N, ub, k / N)
    ood_thr = float(ood_threshold) if math.isfinite(ood_threshold) else float(np.max(ood, initial=0.0))
    if best is None:
        return RoutingPolicy(None, ood_thr, 0.0, 1.0, 0.0, budget, delta, N, deferred)
    t, cov, ub, rate = best
    return RoutingPolicy(t, ood_thr, cov, ub, rate, budget, delta, N, deferred)
