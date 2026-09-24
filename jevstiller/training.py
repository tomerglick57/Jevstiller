"""Fit one candidate: student head, OOD reference, routing policy. Pure and picklable, so it can run in
a worker process (pass a ProcessPoolExecutor as `train_executor`) as easily as in the calling thread."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrate import RoutingPolicy, fit_policy
from .ood import KnnOOD
from .student import LinearStudent


@dataclass
class Candidate:
    student: LinearStudent
    ood: KnnOOD
    policy: RoutingPolicy
    fit: dict
    calib_agreement: float
    calib_accepted: int
    calib_disagree: int


def fit_candidate(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None, Xc: np.ndarray, yc: np.ndarray,
                  *, budget: float, delta: float, ood_quantile: float, ood_k: int, epochs: int, l2: float,
                  patience: int, seed: int) -> Candidate:
    """X, Y, w: training embeddings, teacher distributions, importance weights (None = unweighted).
    Xc, yc: IID calibration embeddings and teacher argmax indices. `budget` is the disagreement budget the
    policy is fitted at (already reduced by any headroom)."""
    student = LinearStudent(X.shape[1], Y.shape[1])
    fit = student.fit(X, Y, epochs=epochs, l2=l2, seed=seed, sample_weight=w, patience=patience)
    ood = KnnOOD(ood_k)
    ood.fit(X, seed=seed)
    Pc = student.predict_proba(Xc)
    conf = Pc.max(axis=1)
    agree = Pc.argmax(axis=1) == yc
    oodc = ood.score(Xc)
    policy = fit_policy(conf, agree, oodc, budget, delta, ood_quantile)
    acc = policy.accepts(conf, oodc)
    return Candidate(student, ood, policy, fit, float(agree.mean()), int(acc.sum()), int((acc & ~agree).sum()))
