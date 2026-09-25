"""Fit one candidate: student head, OOD reference, routing policy.

`run_fit_job` does the whole job from paths: it reads the samples, fits, scores the production version on
the same calibration rows, and writes the bundle files. Only the small job and result objects are pickled, so
with a ProcessPoolExecutor as `train_executor` the serving process does no heavy work for a fit (no row
parsing, no large pickles). It runs the same way in the calling thread when there is no executor.

The fit is BLAS-bound, and BLAS uses every core by default: one fit would take the CPU away from serving
whether it runs in a thread or another process. `threads` caps it.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .calibrate import RoutingPolicy, fit_policy, threshold_grid
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
                  patience: int, seed: int, threads: int = 0, labels: list[str] | None = None,
                  deferred_labels: list[str] = (), ood_max_ref: int = 5_000) -> Candidate:
    """X, Y, w: training embeddings, teacher distributions, importance weights (None = unweighted).
    Xc, yc: IID calibration embeddings and teacher argmax indices. `budget` is the disagreement budget the
    policy is fitted at (already reduced by any headroom). `threads`: BLAS threads for the fit (0 = no cap).
    The cap is process-wide while the fit runs, so in thread mode serving shares it; serving's matrices are
    small enough not to notice. `deferred_labels` (names from `labels`): the policy never lets the student
    answer when it predicts one of them."""
    args = (X, Y, w, Xc, yc, budget, delta, ood_quantile, ood_k, epochs, l2, patience, seed, labels,
            list(deferred_labels), ood_max_ref)
    if threads > 0:
        with threadpool_limits(limits=threads, user_api="blas"):
            return _fit(*args)
    return _fit(*args)


def _fit(X, Y, w, Xc, yc, budget, delta, ood_quantile, ood_k, epochs, l2, patience, seed, labels,
         deferred_labels, ood_max_ref) -> Candidate:
    student = LinearStudent(X.shape[1], Y.shape[1])
    fit = student.fit(X, Y, epochs=epochs, l2=l2, seed=seed, sample_weight=w, patience=patience)
    ood = KnnOOD(ood_k)
    ood.fit(X, max_ref=ood_max_ref, seed=seed, y=Y.argmax(axis=1))
    Pc = student.predict_proba(Xc)
    conf = Pc.max(axis=1)
    agree = Pc.argmax(axis=1) == yc
    oodc = ood.score(Xc)
    pred = None
    eligible = None
    if deferred_labels:
        if labels is None:
            raise ValueError("deferred_labels needs labels")
        pred = np.array(labels, dtype=object)[Pc.argmax(axis=1)]
        eligible = ~np.isin(pred, deferred_labels)
    # the OOD cutoff comes from the training data, the candidates from a fixed grid: the calibration rows
    # only test, which is what makes the bound hold (calibrate.fit_policy)
    policy = fit_policy(conf, agree, oodc, budget, delta, ood_threshold=ood.threshold(ood_quantile, seed=seed),
                        candidates=threshold_grid(Y.shape[1]), eligible=eligible, deferred_labels=deferred_labels)
    acc = policy.accepts(conf, oodc, pred)
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
    teacher_model: str | None = None   # train on this teacher lineage only
    since_id: int = 0                  # ... and on rows after this one (a confirmed drift restarts the data)
    max_train: int = 0                 # the most recent rows only (0 = all)
    max_calib: int = 0
    min_samples_per_class: int = 0
    defer_rare: bool = False       # classes below min_samples_per_class are deferred to the teacher


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
    deferred_labels: list[str] | None = None


def run_fit_job(job: FitJob) -> FitResult:
    from .registry import write_bundle
    from .store import SampleStore

    store = SampleStore(Path(job.store_path), read_only=True)
    try:
        X, Y, y, w = store.training_set(job.task_version, job.encoder_id, job.labels, job.dim, job.teacher_model,
                                        job.since_id, job.max_train)
        Xc, _, yc, _ = store.calib_set(job.task_version, job.encoder_id, job.labels, job.dim, job.teacher_model,
                                       job.since_id, job.max_calib)
    finally:
        store.close()
    if len(X) == 0 or len(Xc) == 0:
        return FitResult(len(X), len(Xc))
    if job.hard_labels:
        Y = np.eye(len(job.labels), dtype=np.float32)[Y.argmax(axis=1)]
    deferred = []
    if job.defer_rare:
        per_class = np.bincount(y[y >= 0], minlength=len(job.labels))
        deferred = [c for c, n in zip(job.labels, per_class, strict=True) if n < job.min_samples_per_class]
    cand = fit_candidate(X, Y, w if job.importance_weighting else None, Xc, yc, labels=job.labels,
                         deferred_labels=deferred, **job.fit)
    prod_cov = None
    if job.prod_dir:
        d = Path(job.prod_dir)
        prod_student, prod_ood = LinearStudent.load(d / "head.npz"), KnnOOD.load(d / "ood.npz")
        prod_student.reorder(json.loads((d / "meta.json").read_text()).get("labels") or job.labels, job.labels)
        prod_policy = RoutingPolicy.load(d / "policy.json")
        with threadpool_limits(limits=job.fit.get("threads") or None, user_api="blas"):
            P = prod_student.predict_proba(Xc)
            po = prod_ood.score(Xc)
        prod_pred = np.array(job.labels, dtype=object)[P.argmax(axis=1)]
        prod_cov = float(prod_policy.accepts(P.max(axis=1), po, prod_pred).mean())
    write_bundle(Path(job.out_dir), cand.student, cand.ood, cand.policy)
    return FitResult(len(X), len(Xc), cand.policy, cand.fit, cand.calib_agreement, cand.calib_accepted,
                     cand.calib_disagree, prod_cov, deferred)


def _lower_priority(niceness: int) -> None:
    try:
        os.nice(niceness)
    except (AttributeError, OSError):  # pragma: no cover - Windows, or not permitted
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:                    # pragma: no cover - exists, another user's
        return True
    try:                               # a dead parent nobody has reaped yet is a zombie: gone too
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _exit_with(server: int, every_s: float = 1.0) -> None:
    while _alive(server):
        time.sleep(every_s)
    os._exit(1)


def _init_worker(niceness: int, server: int) -> None:
    """Lower the worker's priority, and end it when the server process dies: after a kill -9 (or the OOM
    killer) a pool worker never sees its job queue close, and would otherwise live on, holding memory."""
    _lower_priority(niceness)
    threading.Thread(target=_exit_with, args=(server,), daemon=True, name="jevstiller-exit-with-server").start()


def train_pool(workers: int = 2, niceness: int = 10) -> ProcessPoolExecutor:
    """A process pool for `train_executor`, shared by every task in the process.

    `workers` caps how many trainings run at once (the rest queue). Workers run at lower OS priority, so
    the scheduler always prefers serving threads: training soaks up idle CPU instead of competing for it.
    Budget roughly `workers * Config.train_threads` cores for training.

    Workers start from a fresh interpreter (forkserver, or spawn where there is none), never by forking the
    server: before Python 3.14, Linux's default fork copied the live, multi-threaded server into each worker,
    its sockets included (security audit run 2). Like any such pool, it re-imports the main module in each
    worker: create it under `if __name__ == "__main__":` in scripts."""
    method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context(method), initializer=_init_worker,
                               initargs=(niceness, os.getpid()))
