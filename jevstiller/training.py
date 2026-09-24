"""Fit one candidate: student head, OOD reference, routing policy.

`run_fit_job` does the whole job from paths: it reads the samples, fits, scores the production version on
the same calibration rows, and writes the bundle files. Only the small job and result objects are pickled, so
with a ProcessPoolExecutor as `train_executor` the serving process does no heavy work for a fit (no row
parsing, no large pickles). It runs the same way in the calling thread when there is no executor.

The fit is BLAS-bound, and BLAS uses every core by default: one fit would take the CPU away from serving
whether it runs in a thread or another process. `threads` caps it.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .calibrate import RoutingPolicy, fit_policy
from .ood import KnnOOD
from .student import LinearStudent

FIT_PARAMS = ("budget", "delta", "ood_quantile", "ood_k", "epochs", "l2", "patience", "seed", "threads")


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
                  patience: int, seed: int, threads: int = 0) -> Candidate:
    """X, Y, w: training embeddings, teacher distributions, importance weights (None = unweighted).
    Xc, yc: IID calibration embeddings and teacher argmax indices. `budget` is the disagreement budget the
    policy is fitted at (already reduced by any headroom). `threads`: BLAS threads for the fit (0 = no cap).
    The cap is process-wide while the fit runs, so in thread mode serving shares it; serving's matrices are
    small enough not to notice."""
    if threads > 0:
        with threadpool_limits(limits=threads, user_api="blas"):
            return _fit(X, Y, w, Xc, yc, budget, delta, ood_quantile, ood_k, epochs, l2, patience, seed)
    return _fit(X, Y, w, Xc, yc, budget, delta, ood_quantile, ood_k, epochs, l2, patience, seed)


def _fit(X, Y, w, Xc, yc, budget, delta, ood_quantile, ood_k, epochs, l2, patience, seed) -> Candidate:
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


@dataclass
class FitJob:
    store_path: str
    task_version: str
    encoder_id: str
    labels: list[str]
    dim: int
    out_dir: str                   # an empty staging directory; bundle files are written here
    fit: dict                      # keyword arguments for fit_candidate (FIT_PARAMS)
    hard_labels: bool = False
    importance_weighting: bool = True
    prod_dir: str | None = None    # production version, scored on the same calibration rows


@dataclass
class FitResult:
    n_train: int
    n_calib: int
    policy: RoutingPolicy | None = None
    fit: dict | None = None
    calib_agreement: float = 0.0
    calib_accepted: int = 0
    calib_disagree: int = 0
    prod_calib_coverage: float | None = None


def run_fit_job(job: FitJob) -> FitResult:
    from .registry import write_bundle
    from .store import SampleStore

    store = SampleStore(Path(job.store_path), read_only=True)
    try:
        X, Y, _, w = store.training_set(job.task_version, job.encoder_id, job.labels, job.dim)
        Xc, _, yc, _ = store.calib_set(job.task_version, job.encoder_id, job.labels, job.dim)
    finally:
        store.close()
    if len(X) == 0 or len(Xc) == 0:
        return FitResult(len(X), len(Xc))
    if job.hard_labels:
        Y = np.eye(len(job.labels), dtype=np.float32)[Y.argmax(axis=1)]
    cand = fit_candidate(X, Y, w if job.importance_weighting else None, Xc, yc, **job.fit)
    prod_cov = None
    if job.prod_dir:
        d = Path(job.prod_dir)
        prod_student, prod_ood = LinearStudent.load(d / "head.npz"), KnnOOD.load(d / "ood.npz")
        prod_policy = RoutingPolicy.load(d / "policy.json")
        with threadpool_limits(limits=job.fit.get("threads") or None, user_api="blas"):
            pc = prod_student.predict_proba(Xc).max(axis=1)
            po = prod_ood.score(Xc)
        prod_cov = float(prod_policy.accepts(pc, po).mean())
    write_bundle(Path(job.out_dir), cand.student, cand.ood, cand.policy)
    return FitResult(len(X), len(Xc), cand.policy, cand.fit, cand.calib_agreement, cand.calib_accepted,
                     cand.calib_disagree, prod_cov)


def _lower_priority(niceness: int) -> None:
    try:
        os.nice(niceness)
    except (AttributeError, OSError):  # pragma: no cover - Windows, or not permitted
        pass


def train_pool(workers: int = 2, niceness: int = 10) -> ProcessPoolExecutor:
    """A process pool for `train_executor`, shared by every task in the process.

    `workers` caps how many trainings run at once (the rest queue). Workers run at lower OS priority, so
    the scheduler always prefers serving threads: training soaks up idle CPU instead of competing for it.
    Budget roughly `workers * Config.train_threads` cores for training.

    Like any process pool, it re-imports the main module in each worker: create it under
    `if __name__ == "__main__":` in scripts."""
    return ProcessPoolExecutor(workers, initializer=_lower_priority, initargs=(niceness,))
