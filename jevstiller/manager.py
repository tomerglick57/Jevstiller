"""Many tasks in one process: find the task a request belongs to, load it on demand, keep memory bounded.

A task is identified by what the teacher is asked, never by a caller-chosen name:

    key = hash(tenant, question type, Task.version)      # Task.version = hash(instructions, criteria)

so two services asking the same question (same tenant) share one trained student, and any change to the
instructions or a class description is a different task. Matching is exact on purpose: the teacher reads its
criteria literally, so a "similar" task is not the same task.

Layout: `<data_dir>/tasks/<key>/` holds `task.json` (the spec, tenant, timestamps) plus the task's sample store
and versions. One SQLite file per task: tasks never contend for a writer, and deleting a task is deleting a
directory. `load` cost on an LRU miss is measured by `benchmarks/manager.py`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from .core import Jevstiller, Result, TeacherError
from .encoders import Encoder
from .registry import atomic_write_text
from .scheduler import TaskExecutor, TrainScheduler
from .task import Config, State, Task, canonical_json
from .teachers import Teacher

log = logging.getLogger("jevstiller")


def _glibc():
    try:
        import ctypes
        import ctypes.util
        name = ctypes.util.find_library("c")
        libc = ctypes.CDLL(name) if name else None
        return libc if libc is not None and hasattr(libc, "malloc_trim") else None
    except OSError:  # pragma: no cover - not glibc
        return None


_LIBC = _glibc()
_M_MMAP_THRESHOLD = -3


def _allocator_setup() -> None:
    """Loading and unloading tasks frees many multi-megabyte arrays (OOD references) from many threads. glibc
    raises its mmap threshold after such frees, so later arrays land in per-thread heaps that are rarely
    returned to the OS: in benchmarks/manager.py resident memory climbed to ~1.7 GB with 50 tasks loaded,
    ~0.2 GB of it live. A fixed threshold keeps large arrays in their own mappings, freed straight back."""
    if _LIBC is not None and hasattr(_LIBC, "mallopt"):
        _LIBC.mallopt(_M_MMAP_THRESHOLD, 1 << 20)


def _release_free_memory() -> None:
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def task_key(tenant: str, task: Task, question_type: str = "choice") -> str:
    """Stable id of (tenant, question). Hex, so it is also a safe directory name."""
    blob = canonical_json({"tenant": tenant, "type": question_type, "version": task.version}).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


@dataclass
class TaskInfo:
    key: str
    tenant: str
    instructions: Any
    classes: dict
    target_agreement: float
    created: float
    last_seen: float

    def task(self) -> Task:
        return Task(self.key, self.instructions, self.classes, self.target_agreement)


class Admission:
    """Collect training data for a new task only once it has been seen `min_requests` times within
    `window_s`. Callers that build questions dynamically (per-request criteria) would otherwise create an
    unbounded number of one-off tasks. Counts are in memory, capped at `max_tracked` candidate keys
    (least recently seen dropped first); tasks already on disk are always admitted."""

    def __init__(self, min_requests: int = 50, window_s: float = 86_400, max_tracked: int = 100_000):
        self.min_requests, self.window_s, self.max_tracked = min_requests, window_s, max_tracked
        self._seen: OrderedDict[str, tuple[float, int]] = OrderedDict()   # key -> (window start, count)
        self._lock = threading.Lock()

    def observe(self, key: str, n: int = 1, now: float | None = None) -> bool:
        """Count `n` requests for `key`; True once the key qualifies."""
        if self.min_requests <= 1:
            return True
        now = time.time() if now is None else now
        with self._lock:
            start, count = self._seen.pop(key, (now, 0))
            if now - start > self.window_s:
                start, count = now, 0
            count += n
            if count >= self.min_requests:
                return True
            self._seen[key] = (start, count)
            while len(self._seen) > self.max_tracked:
                self._seen.popitem(last=False)
            return False

    def forget(self, key: str) -> None:
        with self._lock:
            self._seen.pop(key, None)


class TaskManager:
    """Routes requests to per-task engines (`Jevstiller`), creating, loading and unloading them.

    - `max_loaded` / `max_memory_mb`: loaded engines are kept in an LRU; an engine that is idle (no request in
      flight, no maintenance running) is unloaded when either limit is exceeded. Unloading closes it (store
      flushed, threads stopped); the next request reloads it from disk.
    - `admission`: see `Admission`. Before a task is admitted, its requests go to the teacher unrecorded
      (`routing_reason="not_admitted"`).
    - `max_tasks_per_tenant`: once reached, further new tasks of that tenant stay pass-through
      (`routing_reason="tenant_task_limit"`).
    - `idle_ttl_s`: tasks unused that long are deleted from disk by the janitor (None = never).
    - `blas_threads`: caps BLAS threads for the whole process (serving); None leaves it alone.

    One `encoder` (wrap it in `BatchingEncoder` to merge concurrent calls) and one `train_executor` are shared
    by every task; with a `TrainScheduler`, training is fair across tenants and prioritised by each task's
    rate of teacher calls. `config` is copied per task.
    """

    def __init__(self, data_dir: str | Path, teacher: Teacher, encoder: Encoder, config: Config | None = None,
                 *, target_agreement: float = 0.98, max_loaded: int = 64, max_memory_mb: float | None = None,
                 admission: Admission | None = None, max_tasks_per_tenant: int | None = None,
                 idle_ttl_s: float | None = None, train_executor: Executor | None = None,
                 janitor_interval_s: float = 5.0, blas_threads: int | None = 1):
        if blas_threads:
            # Serving runs many small matrix products from many threads; BLAS's default of one thread per
            # core for each of them oversubscribes the CPU (1 thread: 2.4x throughput, 4x lower p99 in
            # benchmarks/manager.py). Process-wide: the manager is meant to own its process. Training has
            # its own cap (Config.train_threads).
            threadpool_limits(limits=blas_threads, user_api="blas")
        _allocator_setup()
        self.root = Path(data_dir) / "tasks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.teacher, self.encoder = teacher, encoder
        self.cfg = config or Config()
        self.target_agreement = target_agreement
        self.max_loaded, self.max_memory_mb = max_loaded, max_memory_mb
        self.admission = admission if admission is not None else Admission()
        self.max_tasks_per_tenant, self.idle_ttl_s = max_tasks_per_tenant, idle_ttl_s
        self.train_executor = train_executor
        self.loads = self.unloads = 0
        self._lock = threading.Lock()
        self._index: dict[str, TaskInfo] = {}
        self._per_tenant: dict[str, int] = {}
        self._engines: OrderedDict[str, Jevstiller] = OrderedDict()
        self._inflight: dict[str, int] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._dirty: set[str] = set()                   # last_seen changed since task.json was written
        self._written: dict[str, float] = {}
        self._limit_logged: set[str] = set()
        for f in self.root.glob("*/task.json"):
            try:
                info = TaskInfo(**json.loads(f.read_text()))
            except (OSError, ValueError, TypeError):
                log.exception("skipping unreadable task file %s", f)
                continue
            self._index[info.key] = info
            self._per_tenant[info.tenant] = self._per_tenant.get(info.tenant, 0) + 1
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._janitor = threading.Thread(target=self._janitor_loop, args=(janitor_interval_s,), daemon=True,
                                         name="jevstiller-janitor")
        self._janitor.start()

    # ---- lookup -----------------------------------------------------------------
    def resolve(self, tenant: str, instructions: Any, classes: Mapping[str, Any] | Sequence[str]) -> tuple[str, Task]:
        """The key and Task for a question, without creating anything."""
        spec = Task("_", instructions, classes, self.target_agreement)
        key = task_key(tenant, spec)
        info = self._index.get(key)
        if info is not None:
            return key, info.task()
        return key, Task(key, spec.instructions, spec.classes, self.target_agreement)

    def tasks(self, tenant: str | None = None) -> list[TaskInfo]:
        with self._lock:
            return [i for i in self._index.values() if tenant is None or i.tenant == tenant]

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._engines)

    # ---- serving ----------------------------------------------------------------
    def classify(self, tenant: str, instructions: Any, classes: Mapping[str, Any] | Sequence[str],
                 states: Sequence[State], *, teacher: Teacher | None = None, errors: str = "raise") -> list[Result]:
        """Classify `states` for the question (`instructions`, `classes`) asked by `tenant`."""
        key, task = self.resolve(tenant, instructions, classes)
        teacher = teacher or self.teacher
        if key not in self._index:
            reason = self._admit(key, tenant, task, len(states))
            if reason:
                return self._pass_through(task, states, teacher, reason, errors)
        engine = self._acquire(key)
        try:
            return engine.classify_batch(states, errors=errors, teacher=teacher)
        finally:
            self._release(key)

    def engine(self, key: str) -> Jevstiller:
        """The loaded engine for `key` (loading it if needed). For admin use: the manager may unload it later."""
        e = self._acquire(key)
        self._release(key)
        return e

    def _admit(self, key: str, tenant: str, task: Task, n: int) -> str | None:
        """None when `key` is (now) a registered task, else why its requests are passed through."""
        if not self.admission.observe(key, n):
            return "not_admitted"
        with self._lock:
            if key in self._index:
                return None
            if self.max_tasks_per_tenant is not None and self._per_tenant.get(tenant, 0) >= self.max_tasks_per_tenant:
                if tenant not in self._limit_logged:
                    self._limit_logged.add(tenant)
                    log.warning("tenant %r reached max_tasks_per_tenant=%d; new tasks are passed through",
                                tenant, self.max_tasks_per_tenant)
                return "tenant_task_limit"
            now = time.time()
            info = TaskInfo(key, tenant, task.instructions, dict(task.classes), task.target_agreement, now, now)
            (self.root / key).mkdir(parents=True, exist_ok=True)
            self._write_info(info)
            self._index[key] = info
            self._per_tenant[tenant] = self._per_tenant.get(tenant, 0) + 1
        self.admission.forget(key)
        log.info("task %s admitted for tenant %r (%d classes)", key, tenant, len(task.classes))
        return None

    def _pass_through(self, task: Task, states: Sequence[State], teacher: Teacher, reason: str,
                      errors: str) -> list[Result]:
        t0 = time.perf_counter()
        try:
            outs = list(teacher.classify(list(states), task))
        except Exception as e:
            outs = [e] * len(states)
        dt = (time.perf_counter() - t0) * 1000 / max(len(states), 1)
        results: list[Result | None] = []
        failed: dict[int, Exception] = {}
        for i, o in enumerate(outs):
            if isinstance(o, Exception):
                failed[i] = o
                results.append(None if errors == "raise" else Result(None, {}, 0.0, "error", reason, dt, o))
            else:
                results.append(Result(o.label, o.probs, o.confidence, "teacher", reason, dt))
        if failed and errors == "raise":
            raise TeacherError(failed, results) from next(iter(failed.values()))
        return results  # type: ignore[return-value]

    # ---- loading ----------------------------------------------------------------
    def _acquire(self, key: str) -> Jevstiller:
        with self._lock:
            e = self._engines.get(key)
            if e is not None:
                self._engines.move_to_end(key)
                self._inflight[key] = self._inflight.get(key, 0) + 1
                self._touch(key)
                return e
            info = self._index.get(key)
            if info is None:
                raise KeyError(f"unknown task {key}")
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:                                     # one loader per key; unloading holds it too
            with self._lock:
                if key not in self._index:              # deleted while we waited
                    raise KeyError(f"unknown task {key}")
                e = self._engines.get(key)
                if e is not None:
                    self._engines.move_to_end(key)
                    self._inflight[key] = self._inflight.get(key, 0) + 1
                    self._touch(key)
                    return e
            ex = self.train_executor
            if isinstance(ex, TrainScheduler):              # jobs carry the task's tenant and priority
                ex = ex.for_task(key, info.tenant)
            e = Jevstiller(info.task(), self.teacher, self.root, encoder=self.encoder,
                           config=dataclasses.replace(self.cfg), train_executor=ex)
            if isinstance(ex, TaskExecutor):
                ex.priority = e.training_priority
            with self._lock:
                self._engines[key] = e
                self._inflight[key] = self._inflight.get(key, 0) + 1
                self._touch(key)
                self.loads += 1
                over = len(self._engines) > self.max_loaded
        if over or self.max_memory_mb is not None:
            self._wake.set()
        return e

    def _release(self, key: str) -> None:
        with self._lock:
            self._inflight[key] -= 1

    def _touch(self, key: str) -> None:
        self._index[key].last_seen = time.time()
        self._dirty.add(key)

    def _write_info(self, info: TaskInfo) -> None:
        atomic_write_text(self.root / info.key / "task.json", json.dumps(dataclasses.asdict(info), ensure_ascii=False))
        self._written[info.key] = time.time()

    def memory_mb(self) -> float:
        with self._lock:
            engines = list(self._engines.values())
        return sum(e.footprint_bytes() for e in engines) / 2**20

    # ---- janitor ----------------------------------------------------------------
    def _janitor_loop(self, interval: float) -> None:
        while not self._stop.is_set():
            self._wake.wait(interval)
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                self.sweep()
            except Exception:
                log.exception("task manager sweep failed")

    def sweep(self) -> None:
        """Unload engines over the limits, persist last-seen times, delete tasks idle beyond `idle_ttl_s`."""
        unloaded = self.unloads
        while True:
            with self._lock:
                over_n = len(self._engines) > self.max_loaded
            over_mem = self.max_memory_mb is not None and self.memory_mb() > self.max_memory_mb
            if not (over_n or over_mem) or not self._unload_one():
                break
        if self.unloads != unloaded:
            _release_free_memory()                      # hand freed heap pages back to the OS
        now = time.time()
        with self._lock:                                # persist last_seen at most once a minute per task
            due = [k for k in self._dirty if now - self._written.get(k, 0) >= 60 and k in self._index]
            self._dirty.difference_update(due)
            infos = [dataclasses.replace(self._index[k]) for k in due]
        for info in infos:
            self._write_info(info)
        if self.idle_ttl_s is not None:
            cutoff = time.time() - self.idle_ttl_s
            for info in self.tasks():
                if info.last_seen < cutoff:
                    self.delete(info.key, reason="idle")

    def _unload_one(self) -> bool:
        """Unload the least recently used idle engine. False if none can be unloaded right now."""
        with self._lock:
            for key, e in self._engines.items():         # oldest first
                if self._inflight.get(key, 0) == 0 and not e.busy():
                    klock = self._key_locks.setdefault(key, threading.Lock())
                    break
            else:
                return False
        with klock:
            with self._lock:
                e = self._engines.get(key)
                if e is None or self._inflight.get(key, 0) or e.busy():
                    return True                          # changed under us; let the caller re-check
                del self._engines[key]
                self._inflight.pop(key, None)
                self.unloads += 1
            e.close()
        return True

    def delete(self, key: str, reason: str = "requested") -> bool:
        """Delete a task and all of its data (samples, versions). Refused while a request is in flight."""
        with self._lock:
            info = self._index.get(key)
            if info is None:
                return False
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:
            with self._lock:
                if self._inflight.get(key, 0):
                    return False
                e = self._engines.pop(key, None)
                self._index.pop(key, None)
                self._per_tenant[info.tenant] -= 1
                self._dirty.discard(key)
            if e is not None:
                e.close()
            shutil.rmtree(self.root / key, ignore_errors=True)
        log.info("task %s of tenant %r deleted (%s)", key, info.tenant, reason)
        return True

    def close(self) -> None:
        """Stop the janitor, persist last-seen times, and close every loaded engine."""
        self._stop.set()
        self._wake.set()
        self._janitor.join()
        with self._lock:
            engines, self._engines = list(self._engines.items()), OrderedDict()
            dirty = [self._index[k] for k in self._dirty if k in self._index]
            self._dirty = set()
        for info in dirty:
            self._write_info(info)
        for _, e in engines:
            e.close()
