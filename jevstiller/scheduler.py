"""Training jobs from many tasks share a few workers: fair across tenants, by priority within, with retries."""
from __future__ import annotations

import itertools
import logging
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("jevstiller")


@dataclass
class _Job:
    seq: int
    tenant: str
    key: str
    priority: Callable[[], float]
    fn: Callable
    args: tuple
    kwargs: dict
    outer: Future
    attempts: int = 0
    started: bool = field(default=False)


class TrainScheduler(Executor):
    """An Executor for training jobs, shared by every task of a process.

    - Runs at most `workers` jobs at once, on a pool from `pool_factory` (default: `training.train_pool`, low
      OS priority). A pool that breaks (a worker died, e.g. out of memory) is replaced.
    - Fair across tenants: the next job comes from the next tenant in round-robin order that has one queued,
      so one tenant's hundred tasks cannot starve another's one.
    - Within a tenant, the job whose task has the highest `priority()` (evaluated when a worker frees up)
      goes first; ties by submission order. Engines use their recent rate of teacher calls: training the task
      that costs the most teacher calls saves the most.
    - A failed job is retried up to `max_retries` times, after `retry_backoff_s * 2**(attempt-1)` seconds.
    - A job's Future can be cancelled while it is still queued.

    `for_task(key, tenant)` returns a per-task view to pass as a Jevstiller's `train_executor`.
    """

    def __init__(self, workers: int = 2, niceness: int = 10, max_retries: int = 2, retry_backoff_s: float = 30.0,
                 pool_factory: Callable[[], Executor] | None = None):
        from .training import train_pool
        self.workers, self.max_retries, self.retry_backoff_s = workers, max_retries, retry_backoff_s
        self.pool_factory = pool_factory or (lambda: train_pool(workers, niceness))
        self.pool_restarts = 0
        self.completed = self.failed = self.retried = 0
        self._pool: Executor | None = None
        self._lock = threading.Lock()
        self._queues: dict[str, list[_Job]] = {}
        self._rotation: deque[str] = deque()
        self._running = 0
        self._seq = itertools.count()
        self._shutdown = False

    # ---- submitting -------------------------------------------------------------
    def for_task(self, key: str, tenant: str = "", priority: Callable[[], float] | None = None) -> TaskExecutor:
        return TaskExecutor(self, key, tenant, priority)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        return self._submit("", "", lambda: 0.0, fn, args, kwargs)

    def _submit(self, tenant: str, key: str, priority: Callable[[], float], fn, args, kwargs) -> Future:
        outer: Future = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("scheduler is shut down")
            self._enqueue(_Job(next(self._seq), tenant, key, priority, fn, args, kwargs, outer))
        self._dispatch()
        return outer

    def _enqueue(self, job: _Job) -> None:
        if job.tenant not in self._queues:
            self._queues[job.tenant] = []
            self._rotation.append(job.tenant)
        self._queues[job.tenant].append(job)

    # ---- dispatching ------------------------------------------------------------
    def _next(self) -> _Job | None:
        """Round-robin over tenants with queued jobs; within one, highest priority first. Holds `_lock`."""
        for _ in range(len(self._rotation)):
            tenant = self._rotation[0]
            self._rotation.rotate(-1)
            jobs = self._queues.get(tenant)
            if not jobs:
                continue
            live = [j for j in jobs if not j.outer.cancelled()]
            if not live:
                jobs.clear()
                continue

            def prio(j: _Job) -> tuple[float, int]:
                try:
                    return (-float(j.priority()), j.seq)
                except Exception:
                    return (0.0, j.seq)
            job = min(live, key=prio)
            jobs.remove(job)
            return job
        return None

    def _dispatch(self) -> None:
        while True:
            with self._lock:
                if self._shutdown or self._running >= self.workers:
                    return
                job = self._next()
                if job is None:
                    return
                if not job.started:
                    if not job.outer.set_running_or_notify_cancel():
                        continue                         # cancelled while queued
                    job.started = True
                self._running += 1
                if self._pool is None:
                    self._pool = self.pool_factory()
                pool = self._pool
            job.attempts += 1
            try:
                inner = pool.submit(job.fn, *job.args, **job.kwargs)
            except Exception as e:                       # e.g. the pool broke between jobs
                inner = Future()
                inner.set_exception(e)
            inner.add_done_callback(lambda f, job=job, pool=pool: self._done(job, pool, f))

    def _done(self, job: _Job, pool: Executor, f: Future) -> None:
        exc = f.exception()
        with self._lock:
            self._running -= 1
            if isinstance(exc, BrokenProcessPool) and self._pool is pool:
                self._pool = None                        # the next dispatch builds a fresh pool
                self.pool_restarts += 1
                broken = pool
            else:
                broken = None
        if broken is not None:
            log.error("training pool broke (%r); replacing it", exc)
            # this callback runs on the broken pool's own manager thread, inside its shutdown lock: calling
            # its shutdown() here would deadlock. It is already terminating itself; tidy up from elsewhere.
            threading.Thread(target=broken.shutdown, kwargs={"wait": False, "cancel_futures": True},
                             daemon=True).start()
        if exc is None:
            job.outer.set_result(f.result())
            with self._lock:
                self.completed += 1
        elif job.attempts <= self.max_retries and not self._shutdown:
            delay = self.retry_backoff_s * 2 ** (job.attempts - 1)
            log.warning("training job for task %s failed (attempt %d): %r; retrying in %.0fs",
                        job.key or "-", job.attempts, exc, delay)
            with self._lock:
                self.retried += 1
            t = threading.Timer(delay, self._requeue, args=(job,))
            t.daemon = True
            t.start()
        else:
            job.outer.set_exception(exc)
            with self._lock:
                self.failed += 1
        self._dispatch()

    def _requeue(self, job: _Job) -> None:
        with self._lock:
            if self._shutdown:
                job.outer.set_exception(RuntimeError("scheduler shut down before retry"))
                return
            self._enqueue(job)
        self._dispatch()

    # ---- introspection / shutdown -----------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            queued = {t: sum(not j.outer.cancelled() for j in js) for t, js in self._queues.items()}
            return {"running": self._running, "queued": sum(queued.values()),
                    "queued_by_tenant": {t: n for t, n in queued.items() if n}, "completed": self.completed,
                    "failed": self.failed, "retried": self.retried, "pool_restarts": self.pool_restarts}

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            self._shutdown = True
            pool = self._pool
            if cancel_futures:
                for js in self._queues.values():
                    for j in js:
                        j.outer.cancel()
                self._queues.clear()
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class TaskExecutor(Executor):
    """One task's view of a TrainScheduler: its jobs carry the task's key, tenant and priority."""

    def __init__(self, scheduler: TrainScheduler, key: str, tenant: str,
                 priority: Callable[[], float] | None = None):
        self.scheduler, self.key, self.tenant = scheduler, key, tenant
        self.priority: Callable[[], float] = priority or (lambda: 0.0)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        return self.scheduler._submit(self.tenant, self.key, lambda: self.priority(), fn, args, kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        pass                                             # the scheduler is shared; its owner shuts it down
