"""Many tasks in one process: find the task a request belongs to, load it on demand, keep memory bounded.

A task is identified by what the teacher is asked, never by a caller-chosen name:

    key = sha256(tenant, question type, Task.fingerprint [, requested model])[:20]
                                                          # fingerprint = sha256(instructions, criteria)

so two services asking the same question of the same model (same tenant) share one trained student, and any
change to the instructions or a class description is a different task. Matching is exact on purpose: the
teacher reads its criteria literally, so a "similar" task is not the same task.

Layout: `<data_dir>/tasks/<key>/` holds `task.json` (the spec, tenant, timestamps) plus the task's sample store
and versions. One SQLite file per task: tasks never contend for a writer, and deleting a task is deleting a
directory. `load` cost on an LRU miss is measured by `benchmarks/manager.py`.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from ._core import Jevstiller, Result, Routed, TeacherError
from ._registry import atomic_write_text
from ._scheduler import TaskExecutor, TrainScheduler
from ._store import SampleStore
from ._task import Config, State, Task, canonical_json, check_cutoffs, check_floor
from .encoders import Encoder
from .teachers import Teacher, TeacherOutput

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
_M_ARENA_MAX = -8
ARENAS = 2


def _allocator_setup() -> None:
    """Loading and unloading tasks frees many multi-megabyte arrays (OOD references) from many threads. glibc
    raises its mmap threshold after such frees, so later arrays land in per-thread heaps that are rarely
    returned to the OS: in benchmarks/manager.py resident memory climbed to ~1.7 GB with 50 tasks loaded,
    ~0.2 GB of it live. A fixed threshold keeps large arrays in their own mappings, freed straight back.

    Every task load starts threads (maintenance, the store's writer) and SQLite connections, and glibc gives
    threads their own heaps ("arenas", up to 8 per core). Freed memory scattered over many arenas is rarely
    returned: with ~50 loads a second, resident memory grew ~47 KB per load/unload (292 -> 641 MB with Python's
    own allocations flat). Two arenas: ~11 KB and slowing. Python holds the GIL anyway, so malloc contention
    is not the bottleneck. (Set `MALLOC_ARENA_MAX` in the environment to choose another value.)"""
    if _LIBC is not None and hasattr(_LIBC, "mallopt"):
        _LIBC.mallopt(_M_MMAP_THRESHOLD, 1 << 20)
        if "MALLOC_ARENA_MAX" not in os.environ:
            _LIBC.mallopt(_M_ARENA_MAX, ARENAS)


def _release_free_memory() -> None:
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def _weak(method):
    """A callable for a bound method that does not keep its object alive (0.0 once it is gone)."""
    ref = weakref.WeakMethod(method)

    def call() -> float:
        m = ref()
        return m() if m is not None else 0.0
    return call


def task_key(tenant: str, task: Task, question_type: str = "choice", model: str | None = None) -> str:
    """Stable id of (tenant, question, requested teacher model). Hex, so it is also a safe directory name.
    `model` is the model the caller asked for (e.g. "jev-latest"): callers asking different models are asking
    different teachers and must not share a student. The question is identified by its full 256-bit
    fingerprint, never the 48-bit `Task.version` (security audit run 1)."""
    ident = {"tenant": tenant, "type": question_type, "question": task.fingerprint}
    if model:
        ident["model"] = model
    return hashlib.sha256(canonical_json(ident).encode()).hexdigest()[:20]


@dataclass
class Routing:
    """One question of one request, between `TaskManager.route` and `TaskManager.complete`."""

    key: str
    task: Task
    engine: Jevstiller | None       # None: not admitted (or over the tenant cap); pass through
    routed: Routed | None
    reason: str | None              # why it passes through
    tenant: str | None = None
    model: str | None = None
    uncounted: bool = False         # a new question not counted towards admission yet (see TaskManager.observe)
    caller: str | None = None       # who asked (the proxy: the caller's key hash), for the per-caller task quota

    @property
    def local(self) -> bool:
        return self.routed is not None and self.routed.local


@dataclass
class TaskInfo:
    key: str
    tenant: str
    instructions: Any
    classes: dict
    target_agreement: float
    created: float
    last_seen: float
    model: str | None = None            # the teacher model requested by callers, part of the key
    mode: str | None = None             # operator override of Config.mode (admin API / config file)
    confidence_floor: float | None = None   # Task.confidence_floor
    kind: str = "choice"                # "noul": a yes/no question; part of the key
    cutoffs: list | None = None         # Task.cutoffs of a yes/no task

    def task(self) -> Task:
        return Task(self.key, self.instructions, self.classes, self.target_agreement, self.confidence_floor,
                    self.kind, tuple(self.cutoffs) if self.cutoffs else None)


class Admission:
    """Collect training data for a new task only once it has been seen `min_requests` times within
    `window_s`. Callers that build questions dynamically (per-request criteria) would otherwise create an
    unbounded number of one-off tasks. Counts are in memory, capped at `max_tracked` candidate keys
    (least recently seen dropped first); tasks already on disk are always admitted.

    One caller (the last to ask a question) holds at most `max_per_caller` of those keys, its least recently seen
    dropped first: with one shared table, a caller sending new questions could flush every tenant's counts, so
    nobody's new questions became tasks (security audit run 4). Default: a tenth of `max_tracked`."""

    def __init__(self, min_requests: int = 50, window_s: float = 86_400, max_tracked: int = 100_000,
                 max_per_caller: int | None = None):
        self.min_requests, self.window_s, self.max_tracked = min_requests, window_s, max_tracked
        self.max_per_caller = max(1, max_tracked // 10) if max_per_caller is None else max_per_caller
        self._seen: OrderedDict[str, tuple[float, int]] = OrderedDict()   # key -> (window start, count)
        self._owner: dict[str, str] = {}                                   # key -> the caller that last asked it
        self._by_caller: dict[str, OrderedDict[str, None]] = {}            # caller -> its keys, oldest first
        self._lock = threading.Lock()

    def observe(self, key: str, n: int = 1, now: float | None = None, caller: str | None = None) -> bool:
        """Count `n` requests for `key`; True once the key qualifies."""
        if self.min_requests <= 1:
            return True
        now = time.time() if now is None else now
        with self._lock:
            start, count = self._seen.pop(key, (now, 0))
            self._disown(key)
            if now - start > self.window_s:
                start, count = now, 0
            count += n
            if count >= self.min_requests:
                return True
            self._seen[key] = (start, count)
            if caller is not None:
                own = self._by_caller.setdefault(caller, OrderedDict())
                own[key] = None
                self._owner[key] = caller
                while len(own) > self.max_per_caller:
                    old, _ = own.popitem(last=False)
                    self._seen.pop(old, None)
                    self._owner.pop(old, None)
            while len(self._seen) > self.max_tracked:
                old, _ = self._seen.popitem(last=False)
                self._disown(old)
            return False

    def _disown(self, key: str) -> None:
        caller = self._owner.pop(key, None)
        if caller is not None:
            own = self._by_caller[caller]
            own.pop(key, None)
            if not own:
                del self._by_caller[caller]

    def forget(self, key: str) -> None:
        with self._lock:
            self._seen.pop(key, None)
            self._disown(key)


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
    - `text_retention_s`: blank the raw text of samples older than this, in every task, about hourly (hash,
      embedding and teacher answers are kept, so they still train). None keeps text (see Config.store_text).
    - `max_tasks`: new questions beyond this many tasks stay pass-through (`task_limit`).
    - `hash_key`: key for the per-row text hash (HMAC), e.g. the deployment salt, so `store_text=False` keeps
      no plain hash of the text.
    - `confidence_floor`: Task.confidence_floor for new tasks (None: none). Like `target_agreement`, each task
      keeps its own; change it with `set_confidence_floor` or `task_overrides`.
    - `task_overrides`: {task key: {"target_agreement": x, "mode": m, "confidence_floor": f}} written into those
      tasks at startup.
    Directories are created with mode 0700.

    One `encoder` (wrap it in `BatchingEncoder` to merge concurrent calls) and one `train_executor` are shared
    by every task; with a `TrainScheduler`, training is fair across tenants and prioritised by each task's
    rate of teacher calls. `config` is copied per task.
    """

    def __init__(self, data_dir: str | Path, teacher: Teacher, encoder: Encoder, config: Config | None = None,
                 *, target_agreement: float = 0.98, max_loaded: int = 64, max_memory_mb: float | None = None,
                 admission: Admission | None = None, max_tasks_per_tenant: int | None = None,
                 idle_ttl_s: float | None = None, train_executor: Executor | None = None,
                 janitor_interval_s: float = 5.0, blas_threads: int | None = 1,
                 text_retention_s: float | None = None, max_tasks: int | None = 10_000,
                 hash_key: bytes | None = None, task_overrides: Mapping[str, Mapping[str, Any]] | None = None,
                 max_new_tasks_per_caller: int | None = 100, confidence_floor: float | None = None,
                 noul_cutoffs: Sequence[float] | float | None = 0.5):
        if blas_threads:
            # Serving runs many small matrix products from many threads; BLAS's default of one thread per
            # core for each of them oversubscribes the CPU (1 thread: 2.4x throughput, 4x lower p99 in
            # benchmarks/manager.py). Process-wide: the manager is meant to own its process. Training has
            # its own cap (Config.train_threads).
            threadpool_limits(limits=blas_threads, user_api="blas")
        _allocator_setup()
        self.root = Path(data_dir) / "tasks"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_tasks, self.hash_key = max_tasks, hash_key
        self.teacher, self.encoder = teacher, encoder
        self.cfg = config or Config()
        self.target_agreement = target_agreement
        self.confidence_floor = check_floor(confidence_floor)
        self.noul_cutoffs = None if noul_cutoffs is None else check_cutoffs(noul_cutoffs)
        self.max_loaded, self.max_memory_mb = max_loaded, max_memory_mb
        self.admission = admission if admission is not None else Admission()
        self.max_tasks_per_tenant, self.idle_ttl_s = max_tasks_per_tenant, idle_ttl_s
        self.max_new_tasks_per_caller = max_new_tasks_per_caller
        self._caller_admits: OrderedDict[str, deque[float]] = OrderedDict()   # caller -> when it admitted tasks
        self.text_retention_s = text_retention_s
        self._retention_at = 0.0
        self.train_executor = train_executor
        self.loads = self.unloads = 0
        self._lock = threading.Lock()
        self._index: dict[str, TaskInfo] = {}
        self._per_tenant: dict[str, int] = {}
        self._engines: OrderedDict[str, Jevstiller] = OrderedDict()
        self._inflight: dict[str, int] = {}
        self._carry: dict[str, dict] = {}            # progress of unloaded engines (Jevstiller._carry_state)
        self._key_locks: dict[str, threading.Lock] = {}
        self._dirty: set[str] = set()                   # last_seen changed since task.json was written
        self._written: dict[str, float] = {}
        self._limit_logged: set[str] = set()
        # tasks whose task.json can't be loaded: not served, but text retention and tenant deletion still cover
        # their stores (they used to drop out of both: security audit run 4)
        self._unreadable: list[Path] = []
        for f in self.root.glob("*/task.json"):
            try:
                info = TaskInfo(**json.loads(f.read_text()))
                info = self._migrate_key(info, f.parent)
            except (OSError, ValueError, TypeError):
                log.exception("skipping unreadable task file %s (its data is kept, retention still applies)", f)
                self._unreadable.append(f.parent)
                continue
            self._index[info.key] = info
            self._per_tenant[info.tenant] = self._per_tenant.get(info.tenant, 0) + 1
        for key, o in (task_overrides or {}).items():
            if key in self._index:
                try:
                    if "target_agreement" in o:
                        self.set_target(key, float(o["target_agreement"]))
                    if "mode" in o:
                        self.set_mode(key, o["mode"])
                    if "confidence_floor" in o:
                        self.set_confidence_floor(key, o["confidence_floor"])
                    if "cutoffs" in o:
                        self.set_cutoffs(key, o["cutoffs"])
                except ValueError as e:                 # e.g. a floor for a yes/no task: the settings can't tell
                    raise ValueError(f"[tasks.{key!r}]: {e}") from None
            else:
                log.warning("config overrides for unknown task %s ignored", key)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._janitor = threading.Thread(target=self._janitor_loop, args=(janitor_interval_s,), daemon=True,
                                         name="jevstiller-janitor")
        self._janitor.start()

    def _migrate_key(self, info: TaskInfo, directory: Path) -> TaskInfo:
        """Tasks stored under an older key scheme are re-keyed (directory renamed) to the current one."""
        expected = task_key(info.tenant, info.task(), info.kind, model=info.model)
        if info.key == expected and directory.name == expected:
            return info
        target = self.root / expected
        if target.exists():
            raise ValueError(f"cannot re-key task {directory.name} to {expected}: target exists")
        os.replace(directory, target)
        info = dataclasses.replace(info, key=expected)
        self._write_info(info)
        log.info("re-keyed task %s -> %s (question fingerprint)", directory.name, expected)
        return info

    # ---- operator controls -------------------------------------------------------
    def _update_info(self, key: str, change: Callable[[TaskInfo], None],
                     apply: Callable[[Jevstiller], None] | None = None) -> None:
        """Change a task's TaskInfo: `change` edits it, the result must still make a valid Task, then it is
        persisted and `apply` passes it to the loaded engine, if any. Runs under the task's key lock, as loads and
        deletes do: a change racing a load was persisted but never reached the engine, one racing a delete left
        task.json behind (the task came back after a restart), and an invalid one was persisted before it was
        rejected, so the task could no longer load (security audit run 4). Raises KeyError for an unknown task and
        ValueError, changing nothing, for an invalid change."""
        with self._lock:
            if key not in self._index:
                raise KeyError(f"unknown task {key}")
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:
            with self._lock:
                info = self._index.get(key)
                if info is None:                        # deleted while we waited
                    raise KeyError(f"unknown task {key}")
                new = dataclasses.replace(info, classes=dict(info.classes),
                                          cutoffs=list(info.cutoffs) if info.cutoffs else info.cutoffs)
            change(new)
            new.task()                                  # e.g. a floor on a yes/no task: rejected before it is saved
            self._write_info(new)
            with self._lock:
                change(info)
                e = self._engines.get(key)
            if e is not None and apply is not None:
                apply(e)

    def set_target(self, key: str, target_agreement: float) -> None:
        """Change a task's target agreement (persisted; applied to the loaded engine at once)."""
        if not 0.5 <= target_agreement < 1.0:
            raise ValueError("target_agreement must be in [0.5, 1)")

        def change(info: TaskInfo) -> None:
            info.target_agreement = target_agreement

        def apply(e: Jevstiller) -> None:
            e.task = dataclasses.replace(e.task, target_agreement=target_agreement)
        self._update_info(key, change, apply)

    def set_mode(self, key: str, mode: str | None) -> None:
        """Pin a task's mode ("auto", "teacher_only", "cascade"; None = config default). Persisted."""
        from ._task import MODES
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES} or null")

        def change(info: TaskInfo) -> None:
            info.mode = mode
        self._update_info(key, change, lambda e: e.set_mode(mode or self.cfg.mode))

    def set_confidence_floor(self, key: str, floor: float | None) -> None:
        """Change a task's confidence floor (Task.confidence_floor; None or 0: none). Persisted; the loaded engine
        starts training a student calibrated for it at once (Jevstiller.set_confidence_floor). A yes/no task has
        no floor (ValueError)."""
        floor = check_floor(floor)

        def change(info: TaskInfo) -> None:
            info.confidence_floor = floor
        self._update_info(key, change, lambda e: e.set_confidence_floor(floor))

    def set_cutoffs(self, key: str, cutoffs: Sequence[float] | float) -> None:
        """Change a yes/no task's cut-offs (Task.cutoffs): one number, or two for an unsure band. Persisted; the
        loaded engine answers nothing locally until a student calibrated for them passes shadow
        (Jevstiller.set_cutoffs)."""
        cutoffs = check_cutoffs(cutoffs)

        def change(info: TaskInfo) -> None:
            if info.kind != "noul":
                raise ValueError("cutoffs are for noul tasks")
            info.cutoffs = list(cutoffs)
        self._update_info(key, change, lambda e: e.set_cutoffs(cutoffs))

    # ---- lookup -----------------------------------------------------------------
    def resolve(self, tenant: str, instructions: Any, classes: Mapping[str, Any] | Sequence[str] | None,
                model: str | None = None, kind: str = "choice") -> tuple[str, Task]:
        """The key and Task for a question, without creating anything. Raises ValueError for a question
        that cannot be a task (fewer than 2 or more than 255 classes, non-JSON values; a yes/no question when
        `noul_cutoffs` is None). For `kind="noul"`, `classes` are the question's criteria or None."""
        if kind == "noul":
            if self.noul_cutoffs is None:
                raise ValueError("yes/no questions are not answered locally (noul_cutoffs is None)")
            spec = Task("_", instructions, classes, self.target_agreement, None, "noul", self.noul_cutoffs)
        else:
            spec = Task("_", instructions, classes, self.target_agreement, self.confidence_floor, kind)
        key = task_key(tenant, spec, kind, model=model)
        info = self._index.get(key)
        if info is not None:
            return key, info.task()
        return key, dataclasses.replace(spec, name=key)

    def tasks(self, tenant: str | None = None) -> list[TaskInfo]:
        with self._lock:
            return [i for i in self._index.values() if tenant is None or i.tenant == tenant]

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._engines)

    # ---- serving ----------------------------------------------------------------
    def classify(self, tenant: str, instructions: Any, classes: Mapping[str, Any] | Sequence[str] | None,
                 states: Sequence[State], *, teacher: Teacher | None = None, errors: str = "raise",
                 model: str | None = None, kind: str = "choice") -> list[Result]:
        """Classify `states` for the question (`instructions`, `classes`) asked by `tenant`. `kind="noul"`: a
        yes/no question, with its criteria (or None) as `classes`."""
        key, task = self.resolve(tenant, instructions, classes, model, kind)
        teacher = teacher or self.teacher
        if key not in self._index:
            reason = self._admit(key, tenant, task, len(states), model)
            if reason:
                return self._pass_through(task, states, teacher, reason, errors)
        engine = self._acquire(key)
        try:
            return engine.classify_batch(states, errors=errors, teacher=teacher)
        finally:
            self._release(key)

    def route(self, tenant: str, instructions: Any, classes: Mapping[str, Any] | Sequence[str] | None,
              states: Sequence[State], model: str | None = None, stexts: Sequence[str] | None = None,
              X: Any = None, admit: bool = True, caller: str | None = None, kind: str = "choice") -> Routing:
        """First half of a request whose teacher call the caller makes itself (the proxy): find or admit the
        task and let its engine decide. Always follow with `complete` (it releases the engine). A task that
        is not admitted yet comes back with `engine=None` and `reason` set: send everything to the teacher.
        `X` may be a callable returning the embeddings: it is only called if an engine needs them.

        `admit=False`: the teacher hasn't answered yet, so a new question is not counted towards admission
        here (`uncounted` is set); call `observe` once the teacher answered it usably. Otherwise a caller could
        create tasks with questions the teacher rejects, at no cost (security audit run 2).

        `caller` identifies who asked (the proxy passes the caller's key hash): each caller may create at most
        `max_new_tasks_per_caller` tasks per admission window, so one caller can't fill a shared tenant's task
        cap (security audit run 3)."""
        key, task = self.resolve(tenant, instructions, classes, model, kind)
        if key not in self._index:
            reason = self._admit(key, tenant, task, len(states), model, count=admit, caller=caller)
            if reason:
                return Routing(key, task, None, None, reason, tenant, model,
                               uncounted=not admit and reason == "not_admitted", caller=caller)
        engine = self._acquire(key)
        try:
            routed = engine.route(states, stexts, X() if callable(X) else X)
        except BaseException:
            self._release(key)
            raise
        return Routing(key, task, engine, routed, None)

    def complete(self, routing: Routing, outs: Sequence[TeacherOutput | Exception],
                 teacher_name: str | None = None) -> list[Result] | None:
        """Second half: hand the teacher's answers for `routing.routed.to_teacher` to the engine (records and
        returns Results), and release it. None for a pass-through routing."""
        if routing.engine is None:
            return None
        try:
            return routing.engine.complete(routing.routed, outs, teacher_name)
        finally:
            self._release(routing.key)

    def observe(self, routing: Routing) -> bool:
        """Count one usable teacher answer towards admitting an `uncounted` routing's task. True if the task
        is (now) registered: later requests for it are routed to its engine."""
        if not routing.uncounted:
            return routing.key in self._index
        return self._admit(routing.key, routing.tenant or "", routing.task, 1, routing.model,
                           caller=routing.caller) is None

    def engine(self, key: str) -> Jevstiller:
        """The loaded engine for `key` (loading it if needed). The manager may unload and close it at any time
        afterwards: to use it, hold it (`hold`)."""
        e = self._acquire(key)
        self._release(key)
        return e

    @contextlib.contextmanager
    def hold(self, key: str) -> Iterator[Jevstiller]:
        """The engine for `key`, loaded if needed and kept open (not unloaded, closed or deleted) until the block
        exits. For admin calls: unlike a request, it doesn't count as use of the task (idle TTL, LRU order)."""
        e = self._acquire(key, touch=False)
        try:
            yield e
        finally:
            self._release(key)

    @contextlib.contextmanager
    def _hold_loaded(self, key: str) -> Iterator[Jevstiller | None]:
        """The engine for `key` if it is loaded (else None), kept open until the block exits. Loads nothing and
        doesn't count as use (for maintenance, e.g. text retention)."""
        with self._lock:
            e = self._engines.get(key)
            if e is not None:
                self._inflight[key] = self._inflight.get(key, 0) + 1
        try:
            yield e
        finally:
            if e is not None:
                self._release(key)

    def _caller_quota_left(self, caller: str | None, now: float) -> bool:
        """Whether `caller` may create another task in this admission window. Call with `_lock` held."""
        if caller is None or self.max_new_tasks_per_caller is None:
            return True
        seen = self._caller_admits.get(caller)
        if seen is None:
            return self.max_new_tasks_per_caller > 0
        while seen and now - seen[0] > self.admission.window_s:
            seen.popleft()
        return len(seen) < self.max_new_tasks_per_caller

    def _note_caller(self, caller: str | None, now: float) -> None:
        if caller is None or self.max_new_tasks_per_caller is None:
            return
        self._caller_admits.setdefault(caller, deque()).append(now)
        self._caller_admits.move_to_end(caller)
        while len(self._caller_admits) > self.admission.max_tracked:
            self._caller_admits.popitem(last=False)

    def _cap_reason(self, tenant: str, caller: str | None = None) -> str | None:
        """Why a new task of `tenant` asked by `caller` can't be registered now (a task cap or the caller's
        quota), or None. Call with `_lock` held."""
        if not self._caller_quota_left(caller, time.time()):
            return "caller_task_limit"
        if self.max_tasks is not None and len(self._index) >= self.max_tasks:
            return "task_limit"
        if self.max_tasks_per_tenant is not None and self._per_tenant.get(tenant, 0) >= self.max_tasks_per_tenant:
            if tenant not in self._limit_logged:
                self._limit_logged.add(tenant)
                log.warning("tenant %r reached max_tasks_per_tenant=%d; new tasks are passed through",
                            tenant, self.max_tasks_per_tenant)
            return "tenant_task_limit"
        return None

    def _admit(self, key: str, tenant: str, task: Task, n: int, model: str | None = None,
               count: bool = True, caller: str | None = None) -> str | None:
        """None when `key` is (now) a registered task, else why its requests are passed through. With
        `count=False`, the requests are not counted towards admission (so a new task is never registered)."""
        if not count:
            with self._lock:
                if key in self._index:
                    return None
                return self._cap_reason(tenant, caller) or "not_admitted"
        if not self.admission.observe(key, n, caller=caller):
            return "not_admitted"
        with self._lock:
            if key in self._index:
                return None
            if (reason := self._cap_reason(tenant, caller)) is not None:
                return reason
            now = time.time()
            self._note_caller(caller, now)
            info = TaskInfo(key, tenant, task.instructions, dict(task.classes), task.target_agreement, now, now,
                            model, confidence_floor=task.confidence_floor, kind=task.kind,
                            cutoffs=list(task.cutoffs) if task.cutoffs else None)
            (self.root / key).mkdir(parents=True, exist_ok=True, mode=0o700)
            self._write_info(info)
            self._index[key] = info
            self._per_tenant[tenant] = self._per_tenant.get(tenant, 0) + 1
        self.admission.forget(key)
        log.info("task %s admitted for tenant %r (%s)", key, tenant,
                 "yes/no" if task.kind == "noul" else f"{len(task.classes)} classes")
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
    def _acquire(self, key: str, touch: bool = True) -> Jevstiller:
        with self._lock:
            e = self._engines.get(key)
            if e is not None:
                if touch:
                    self._engines.move_to_end(key)
                    self._touch(key)
                self._inflight[key] = self._inflight.get(key, 0) + 1
                return e
            info = self._index.get(key)
            if info is None:
                raise KeyError(f"unknown task {key}")
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:                                     # one loader per key; unloading, deleting and changing
            with self._lock:                            # a task (_update_info) hold it too
                info = self._index.get(key)
                if info is None:                        # deleted while we waited
                    raise KeyError(f"unknown task {key}")
                e = self._engines.get(key)
                if e is not None:
                    if touch:
                        self._engines.move_to_end(key)
                        self._touch(key)
                    self._inflight[key] = self._inflight.get(key, 0) + 1
                    return e
            ex = self.train_executor
            if isinstance(ex, TrainScheduler):              # jobs carry the task's tenant and priority
                ex = ex.for_task(key, info.tenant)
            cfg = dataclasses.replace(self.cfg, mode=info.mode or self.cfg.mode)
            e = Jevstiller(info.task(), self.teacher, self.root, encoder=self.encoder, config=cfg,
                           train_executor=ex, hash_key=self.hash_key)
            if isinstance(ex, TaskExecutor):               # weakly: a strong one makes a cycle that keeps
                ex.priority = _weak(e._training_priority)    # an unloaded engine (and its arrays) until a GC
            with self._lock:
                carry = self._carry.pop(key, None)
            if carry:
                e._restore_state(carry)
            with self._lock:
                self._engines[key] = e
                self._inflight[key] = self._inflight.get(key, 0) + 1
                if touch:
                    self._touch(key)
                self.loads += 1
                over = len(self._engines) > self.max_loaded
        if over:
            # Hold the cap here, not only in the janitor: with a steady stream of loads, one janitor thread
            # unloading one engine at a time fell behind, and 505 tasks were loaded against max_loaded = 50
            # (benchmarks/manager.py). Outside this key's lock: taking another key's lock inside it could deadlock.
            for _ in range(2):
                with self._lock:
                    if len(self._engines) <= self.max_loaded:
                        break
                if not self._unload_one():
                    break
        if over or self.max_memory_mb is not None:
            self._wake.set()
        return e

    def _release(self, key: str) -> None:
        with self._lock:
            self._inflight[key] -= 1

    def _touch(self, key: str) -> None:
        self._index[key].last_seen = time.time()
        self._dirty.add(key)

    def _persist(self, key: str) -> None:
        """Write a task's task.json from the index, under its key lock (a write racing a delete left the file
        behind: security audit run 4). A key busy loading or unloading is retried on the next sweep."""
        with self._lock:
            klock = self._key_locks.setdefault(key, threading.Lock())
        if not klock.acquire(blocking=False):
            with self._lock:
                self._dirty.add(key)
            return
        try:
            with self._lock:
                info = self._index.get(key)
                info = dataclasses.replace(info) if info is not None else None
            if info is not None:
                self._write_info(info)
        finally:
            klock.release()

    def _write_info(self, info: TaskInfo) -> None:
        atomic_write_text(self.root / info.key / "task.json", json.dumps(dataclasses.asdict(info), ensure_ascii=False))
        self._written[info.key] = time.time()

    def memory_mb(self) -> float:
        with self._lock:
            engines = list(self._engines.values())
        return sum(e._footprint_bytes() for e in engines) / 2**20

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
        for key in due:
            self._persist(key)
        if self.idle_ttl_s is not None:
            cutoff = time.time() - self.idle_ttl_s
            for info in self.tasks():
                if info.last_seen < cutoff:
                    self.delete(info.key, reason="idle")
        if self.text_retention_s is not None and time.time() - self._retention_at > 3600:
            self._retention_at = time.time()
            self.apply_retention()

    def apply_retention(self) -> int:
        """Blank raw text older than `text_retention_s` in every task's store (loaded or not)."""
        if self.text_retention_s is None:
            return 0
        cutoff = time.time() - self.text_retention_s
        total = 0
        for info in self.tasks():
            try:
                with self._hold_loaded(info.key) as engine:  # an unload closing its store under us: SIGSEGV
                    if engine is not None:
                        total += engine.store.redact_text(cutoff)
                    elif (self.root / info.key / "samples.sqlite").exists():
                        store = SampleStore(self.root / info.key / "samples.sqlite")
                        try:
                            total += store.redact_text(cutoff)
                        finally:
                            store.close()
            except Exception:
                log.exception("text retention failed for task %s", info.key)
        for directory in list(self._unreadable):
            if (directory / "samples.sqlite").exists():
                try:
                    store = SampleStore(directory / "samples.sqlite")
                    try:
                        total += store.redact_text(cutoff)
                    finally:
                        store.close()
                except Exception:
                    log.exception("text retention failed for unreadable task %s", directory.name)
        if total:
            log.info("text retention: blanked %d samples older than %.0f s", total, self.text_retention_s)
        return total

    def delete_tenant(self, tenant: str) -> list[str]:
        """Delete every task of `tenant` and all their data. Returns the deleted keys; tasks with a request in
        flight are skipped (call again)."""
        deleted = [i.key for i in self.tasks(tenant) if self.delete(i.key, reason=f"tenant {tenant!r} deleted")]
        for directory in list(self._unreadable):        # tasks that didn't load still belong to their tenant
            try:
                owner = json.loads((directory / "task.json").read_text()).get("tenant")
            except (OSError, ValueError, AttributeError):
                owner = None
            if owner == tenant:
                shutil.rmtree(directory, ignore_errors=True)
                self._unreadable.remove(directory)
                deleted.append(directory.name)
                log.info("unreadable task %s of tenant %r deleted", directory.name, tenant)
        return deleted

    def _unload_one(self) -> bool:
        """Unload the least recently used idle engine. False if none can be unloaded right now."""
        with self._lock:
            for key, e in self._engines.items():         # oldest first
                if self._inflight.get(key, 0) == 0 and not e._busy():
                    klock = self._key_locks.setdefault(key, threading.Lock())
                    break
            else:
                return False
        with klock:
            with self._lock:
                e = self._engines.get(key)
                if e is None or self._inflight.get(key, 0) or e._busy():
                    return True                          # changed under us; let the caller re-check
                del self._engines[key]
                self._inflight.pop(key, None)
                self._carry[key] = e._carry_state()
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
                self._carry.pop(key, None)
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
