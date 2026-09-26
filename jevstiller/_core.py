"""The loop: route, record, train, calibrate, shadow, promote, monitor, fall back."""
from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import shutil
import sqlite3
import threading
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import Executor, Future
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._calibrate import RoutingPolicy, clopper_pearson_lower, clopper_pearson_upper
from ._registry import Bundle, Registry
from ._store import Record, SampleStore, text_hash
from ._task import MAX_TEXT_CHARS, MODES, Config, State, Task, state_text
from ._training import FitJob, run_fit_job
from .encoders import Encoder, HashEncoder
from .teachers import Teacher, TeacherOutput

log = logging.getLogger("jevstiller")


@dataclass
class Result:
    label: str | None           # None only when the teacher call failed (see `error`)
    probs: dict[str, float]
    confidence: float
    source: str                 # "teacher" | "student:vN" | "error"
    routing_reason: str
    latency_ms: float
    error: Exception | None = None


class DecayingRate:
    """An exponentially decaying event count, reported per minute. Thread-safe enough for a priority hint."""

    def __init__(self, half_life_s: float = 600):
        self.tau = half_life_s / math.log(2)
        self._v, self._t = 0.0, time.monotonic()

    def add(self, n: float) -> None:
        now = time.monotonic()
        self._v = self._v * math.exp(-(now - self._t) / self.tau) + n
        self._t = now

    def value(self) -> float:
        return self._v * math.exp(-(time.monotonic() - self._t) / self.tau) * 60 / self.tau


@dataclass
class Routed:
    """The outcome of `Jevstiller.route` for a batch: prepared records, the student's answers, and which
    items (indices) still need the teacher."""

    recs: list[Record]
    results: list[Result | None]
    to_teacher: list[int]
    t0: float

    @property
    def local(self) -> bool:
        """The student answers every item."""
        return not self.to_teacher


class TeacherError(RuntimeError):
    """Some items of a batch needed the teacher and its call failed.

    `errors` maps batch index -> exception; `results` holds every item that did get an answer (None for the
    failed ones). Answered items are recorded as usual; failed items are not recorded.
    """

    def __init__(self, errors: dict[int, Exception], results: list[Result | None]):
        self.errors, self.results = errors, results
        first = next(iter(errors.values()))
        super().__init__(f"{len(errors)} of {len(results)} teacher calls failed; first: {first!r}")


@dataclass
class TrainReport:
    version: str | None
    n_train: int
    n_calib: int
    policy: RoutingPolicy | None
    train_loss: float
    calib_agreement: float
    accepted: bool
    reason: str


_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _inert(x) -> str:
    """`x` as text with control characters escaped: class names come from callers, and the report is printed
    to operators' terminals (`jevstiller admin status`), where escape sequences would run."""
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", str(x))


@dataclass
class Status:
    task: str
    task_version: str
    mode: str
    audit_rate: float
    production: str | None
    shadow: str | None
    requests: int
    served_by_student: int
    served_by_teacher: int
    channels: dict
    student_share: float
    teacher_share: float
    audit_n: int
    audit_agreement: float | None        # point estimate of system agreement on audit channel
    audit_agreement_lb: float | None     # lower confidence bound
    audit_agreement_ub: float | None     # upper confidence bound
    target_agreement: float
    teacher_calls: int
    teacher_calls_avoided: int
    teacher_cost_usd: float
    teacher_cost_avoided_usd: float
    labelled_train: int
    labelled_calib: int
    policy: dict | None
    events: list = field(default_factory=list)
    teacher_errors: int = 0                 # failed teacher calls since this process started
    teacher_model: str | None = None        # the teacher lineage the loop trains and audits against
    readiness: dict | None = None           # before the first student: what training is waiting for

    def report(self, events: bool = True) -> str:
        """The status as text. `events=False` leaves out the last five loop events."""
        L = []
        L.append(f"Task: {self.task}   version {self.task_version}   mode: {self.mode}   "
                 f"audit rate {self.audit_rate:.0%}")
        L.append(f"Production: {self.production or '-'}   Shadow: {self.shadow or '-'}")
        L.append(f"Requests: {self.requests:,}   student {self.student_share:.1%}   teacher {self.teacher_share:.1%}")
        ch = "   ".join(f"{k} {v:,}" for k, v in sorted(self.channels.items()))
        L.append(f"Channels: {ch}")
        L.append(f"Teacher calls: {self.teacher_calls:,}   avoided: {self.teacher_calls_avoided:,}   "
                 f"spent ${self.teacher_cost_usd:.4f}   avoided ${self.teacher_cost_avoided_usd:.4f}"
                 + (f"   errors: {self.teacher_errors:,}" if self.teacher_errors else ""))
        if self.audit_agreement is not None:
            if self.audit_agreement_lb >= self.target_agreement:
                ok = "OK"
            elif self.audit_agreement_ub >= self.target_agreement:
                ok = "inconclusive (need more audit samples)"
            else:
                ok = "BROKEN"
            L.append(f"Agreement with teacher (audit, n={self.audit_n:,}): {self.audit_agreement:.2%} "
                     f"[{self.audit_agreement_lb:.2%}, {self.audit_agreement_ub:.2%}]   "
                     f"target {self.target_agreement:.0%}   {ok}")
            L.append("  note: agreement with the teacher is not accuracy.")
        L.append(f"Labelled: train {self.labelled_train:,}   calib {self.labelled_calib:,}"
                 + (f"   teacher {self.teacher_model}" if self.teacher_model else ""))
        if self.readiness:
            r = self.readiness
            line = (f"Waiting for a first student: train {r['train'][0]:,}/{r['train'][1]:,}   "
                    f"calib {r['calib'][0]:,}/{r['calib'][1]:,}")
            if r["rare"]:
                rare = ", ".join(f"{_inert(c)} {n}" for c, n in sorted(r["rare"].items(), key=lambda kv: kv[1]))
                line += f"\n  classes below {r['min_per_class']}: {rare}"
                line += ("   (blocking: set Config.rare_classes='defer' to train without them)" if r["blocked_by_rare"]
                         else "   (will be deferred to the teacher)")
            L.append(line)
        if self.policy:
            p = self.policy
            L.append(f"Policy: conf>={p['conf_threshold']:.3f} ood<={p['ood_threshold']:.3f}  "
                     f"expected coverage {p['expected_coverage']:.1%}  disagreement bound {p['disagreement_ub']:.2%}"
                     " of requests")
            if p.get("deferred_labels"):
                L.append(f"  rare classes, always sent to the teacher: {', '.join(map(_inert, p['deferred_labels']))}")
        for e in self.events[-5:] if events else []:
            L.append(f"  event: {_inert(e['kind'])} "
                     + " ".join(f"{_inert(k)}={_inert(round(v, 4) if isinstance(v, float) else v)}"
                                for k, v in e.items() if k not in ('kind', 'ts')))
        return "\n".join(L)


class Jevstiller:
    """One task's loop. Thread-safe: any number of threads may call classify/classify_batch at once.

    Locking: `_state` guards the routing state (production/shadow bundles, flags, rng) and is held only for
    reads and swaps, never across encoding, inference, the teacher call, the store or training. `_maint`
    serialises maintenance (judge shadow, train, drift check), which runs according to `config.training`:
    in a background worker thread (default), inline after each batch, or only when `maintain()` is called.

    `train_executor`: optional Executor for the training job (reading samples, fitting, writing the bundle).
    Recommended when several tasks share a process: one `training.train_pool()` (2 low-priority workers) for
    all of them. With 5 tasks training at once it cost serving ~1.12x at p99 and finished the fits ~2.5x
    faster than threads (benchmarks/concurrency.py). What matters is capping training's CPU: pool size
    x `config.train_threads`. Default (None): run the job in the maintenance thread (~1.11x, slower fits).
    """

    def __init__(self, task: Task, teacher: Teacher, data_dir: str | Path, *,
                 encoder: Encoder | None = None, config: Config | None = None,
                 train_executor: Executor | None = None, hash_key: bytes | None = None):
        self.task = task
        self.teacher = teacher
        self.cfg = config or Config()
        self.encoder = encoder or HashEncoder()
        self.train_executor = train_executor
        self.hash_key = hash_key                # keys the per-row text hash (HMAC) when given
        self.dir = Path(data_dir) / task.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = SampleStore(self.dir / "samples.sqlite", self.cfg.calib_fraction,
                                 split_seed=f"{self.cfg.seed}:{task.version}")
        self.registry = Registry(self.dir / "versions", keep=self.cfg.keep_versions)
        self.rng = np.random.default_rng(self.cfg.seed)
        self._state = threading.Lock()
        self._maint = threading.Lock()
        self.forced_fallback = False
        self._retrain_requested = False
        self._suspicious = False
        self._last_train_id = 0
        self._shadow_started_id = 0
        self._teacher_errors = 0
        self._pending: tuple[Future, Path, str | None, int] | None = None   # queued/running training job
        self._teacher_rate = DecayingRate(half_life_s=600)
        self._audit_new = 0                 # audit answers since the last drift check
        self._counted_at = -10**9           # store max id at the last readiness count
        self._pruned = False                # versions past keep_versions deleted on the first maintenance pass
        self._rows_unpruned: int | None = None   # rows recorded since the last row prune (None: not checked yet)
        self._prune_more = False            # the last row prune stopped at its limit: continue on the next pass
        self.labels = task.labels
        self._lidx = {c: i for i, c in enumerate(self.labels)}
        self._prod: Bundle | None = self._load(self.registry.production) if self.registry.production else None
        shadows = self.registry.in_state("shadow")
        self._shadow: Bundle | None = self._load(shadows[-1]) if shadows else None
        if self._shadow is not None:
            self._shadow_started_id = self._shadow.meta.get("shadow_from_id", self.store.max_id())
        versions = self.registry.versions()             # the last training, whatever became of its result
        if versions:
            with contextlib.suppress(OSError, ValueError):
                self._last_train_id = self.registry.meta(versions[-1]["name"]).get("trained_to_id", 0)
        # teacher lineage: the model the last teacher answer came from. A different model must answer
        # `teacher_change_confirm` times in a row before the loop switches to it.
        self._teacher_model: str | None = self.store.latest_teacher_model(task.version)
        self._pending_model: str | None = None
        self._pending_count = 0
        trained_on = self._prod.meta.get("teacher_model") if self._prod else None
        if trained_on and self._teacher_model and trained_on != self._teacher_model:
            self._teacher_changed(trained_on, self._teacher_model)
        # A confirmed drift restarts the training data at the row where it was detected (see _check_drift);
        # a fallback that no promotion or operator has ended since survives a restart.
        drift = self.store.last_event(("fallback",))
        self._since_id: int = drift.get("since_id", 0) if drift else 0
        last = self.store.last_event(("fallback", "promoted", "fallback_cleared"))
        if last and last["kind"] == "fallback" and self._prod is not None:
            self.forced_fallback = self._retrain_requested = True
        # background worker: `_kicks` counts batches that asked for maintenance, `_done` is the kick count
        # the last finished pass had seen; drain() waits for _done to catch up
        self._cv = threading.Condition()
        self._kicks = self._done = self._draining = 0
        self._stop = False
        self._worker: threading.Thread | None = None

    def _load(self, name: str) -> Bundle:
        """Load a version with its outputs in this task's label order (the task version ignores order)."""
        b = self.registry.load(name)
        b.student.reorder(b.meta.get("labels") or self.labels, self.labels)
        return b

    # ---- teacher lineage ---------------------------------------------------------
    def _observe_teacher(self, models: Sequence[str]) -> None:
        changed = None
        with self._state:
            for m in models:
                if self._teacher_model is None:
                    self._teacher_model = m
                elif m == self._teacher_model:
                    self._pending_model, self._pending_count = None, 0
                else:
                    if m == self._pending_model:
                        self._pending_count += 1
                    else:
                        self._pending_model, self._pending_count = m, 1
                    if self._pending_count >= self.cfg.teacher_change_confirm:
                        changed = (self._teacher_model, m)
                        self._teacher_model, self._pending_model, self._pending_count = m, None, 0
        if changed:
            self._teacher_changed(*changed)

    def _teacher_changed(self, old: str, new: str) -> None:
        """The teacher now answers from a different model: the old student's agreement with it is unknown.
        Start a new lineage: training, calibration, shadow and audit use only the new model's answers."""
        with self._state:
            self._teacher_model = new
            self._retrain_requested = True
            if self.cfg.teacher_change == "fallback" and self._prod is not None:
                self.forced_fallback = True
            else:
                self._suspicious = True
            dropped, self._shadow = self._shadow, None
        if dropped is not None:
            self.registry.set_state(dropped.name, "rejected")
        self.store.event("teacher_changed", previous=old, current=new, action=self.cfg.teacher_change,
                         dropped_shadow=dropped.name if dropped else None)
        log.warning("task %s: teacher model changed %s -> %s (%s)", self.task.name, old, new, self.cfg.teacher_change)

    # ---- modes ----------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        with self._state:
            self.cfg.mode = mode
            cleared = mode != "teacher_only" and self.forced_fallback
            if mode != "teacher_only":
                self.forced_fallback = False
        if cleared:
            self.store.event("fallback_cleared", mode=mode)

    @property
    def audit_rate(self) -> float:
        r = self.cfg.audit_rate
        if self._shadow is not None:
            r = max(r, self.cfg.audit_rate_shadow)
        if self._suspicious:
            r = max(r, self.cfg.audit_rate_elevated)
        return r

    @property
    def mode(self) -> str:
        if self.cfg.mode == "teacher_only" or self.forced_fallback:
            return "teacher_only"
        return "cascade" if self._prod else "teacher_only"

    # ---- inference --------------------------------------------------------------
    def _run(self, b: Bundle, X: np.ndarray):
        P = b.student.predict_proba(X)
        conf = P.max(axis=1)
        ood = b.ood.score(X)
        return P, conf, ood

    def classify(self, text: State, teacher: Teacher | None = None) -> Result:
        """Classify one state: text, or a JSON object/array. Raises TeacherError if it needed the teacher and
        the call failed."""
        return self.classify_batch([text], teacher=teacher)[0]

    def classify_batch(self, texts: Sequence[State], errors: str = "raise",
                       teacher: Teacher | None = None) -> list[Result]:
        """Classify many states (text, or JSON objects/arrays). Objects are encoded and stored as canonical
        JSON; the teacher receives them unchanged. If some teacher calls fail, `errors="raise"` (default)
        raises TeacherError after recording the rest; `errors="return"` returns Results with `error` set and
        `label=None`. `teacher` overrides the task's teacher for this call (e.g. the caller's own API key);
        its answers join the same lineage as long as they report the same model.

        Equivalent to `route` -> the teacher on `routed.to_teacher` -> `complete`."""
        if errors not in ("raise", "return"):
            raise ValueError("errors must be 'raise' or 'return'")
        teacher = teacher or self.teacher
        routed = self.route(texts)
        outs: list[TeacherOutput | Exception] = []
        if routed.to_teacher:                               # no lock held: callers' teacher calls overlap
            try:
                outs = list(teacher.classify([texts[i] for i in routed.to_teacher], self.task))
                if len(outs) != len(routed.to_teacher):
                    raise RuntimeError(f"teacher returned {len(outs)} answers for {len(routed.to_teacher)} texts")
            except Exception as e:
                outs = [e] * len(routed.to_teacher)
        results = self.complete(routed, outs, teacher_name=teacher.name)
        failed = {i: r.error for i, r in enumerate(results) if r.error is not None}
        if failed and errors == "raise":
            partial = [None if r.error is not None else r for r in results]
            raise TeacherError(failed, partial) from next(iter(failed.values()))
        return results

    def route(self, texts: Sequence[State], stexts: Sequence[str] | None = None,
              X: np.ndarray | None = None) -> Routed:
        """Decide, for each state, whether the student answers it or the teacher must. Encodes and runs the
        models; no network, no store writes. Finish with `complete` (always, even if the teacher call fails),
        after asking the teacher about `routed.to_teacher`. `stexts` / `X` (the states' canonical text and
        embeddings from this task's encoder) may be passed in when several tasks route the same states."""
        t0 = time.perf_counter()
        n = len(texts)
        stexts = [state_text(t) for t in texts] if stexts is None else list(stexts)
        X = self.encoder.encode(stexts) if X is None else X
        with self._state:                                   # one consistent snapshot of the routing state
            mode, prod, shadow, audit_rate = self.mode, self._prod, self._shadow, self.audit_rate
            draws = self.rng.random(n)
        P = conf = ood = None
        if prod is not None:
            P, conf, ood = self._run(prod, X)
        sP = sconf = sood = None
        if shadow is not None:
            sP, sconf, sood = self._run(shadow, X)

        recs: list[Record] = []
        results: list[Result | None] = [None] * n
        to_teacher: list[int] = []
        for i in range(n):
            # the stored text is cut like the encoder's input: a 4 MB state was stored whole, once per task in the
            # request, locally answered ones included (security audit run 3). The hash covers the whole text.
            r = Record(text=stexts[i][:MAX_TEXT_CHARS] if self.cfg.store_text else "",
                       text_hash=text_hash(stexts[i], self.hash_key),
                       task_version=self.task.version, encoder_id=self.encoder.id,
                       embedding=X[i], served_by="teacher", routing_reason="", channel="",
                       state_type="text" if isinstance(texts[i], str) else "json")
            if prod is not None:
                r.student_version = prod.name
                r.student_label = self.labels[int(P[i].argmax())]
                r.student_probs = {c: float(P[i, j]) for j, c in enumerate(self.labels)}
                r.student_confidence = float(conf[i])
                r.ood_score = float(ood[i])
            if shadow is not None:
                r.shadow_version = shadow.name
                r.shadow_label = self.labels[int(sP[i].argmax())]
                r.shadow_confidence = float(sconf[i])
                r.shadow_ood = float(sood[i])

            if mode == "teacher_only":
                r.channel = "bootstrap" if prod is None else "fallback"
                r.routing_reason = r.channel
                to_teacher.append(i)
            elif draws[i] < audit_rate:
                r.channel = r.routing_reason = "audit"
                if self.cfg.importance_weighting:
                    r.weight = min((1.0 / audit_rate) ** self.cfg.weight_power, self.cfg.max_weight)
                to_teacher.append(i)
            else:
                pol = prod.policy
                rare = r.student_label in pol.deferred_labels
                accept = pol.usable and conf[i] >= pol.conf_threshold and ood[i] <= pol.ood_threshold and not rare
                if accept:
                    r.served_by, r.channel, r.routing_reason = "student", "student", "confident"
                    results[i] = Result(r.student_label, r.student_probs, r.student_confidence,
                                        prod.name, "confident", 0.0)
                else:
                    r.channel = "deferred"
                    r.routing_reason = ("ood" if (pol.usable and ood[i] > pol.ood_threshold)
                                        else "rare_class" if rare else "low_confidence")
                    to_teacher.append(i)
            recs.append(r)
        return Routed(recs, results, to_teacher, t0)

    def defer(self, routed: Routed, i: int, reason: str) -> None:
        """Send item `i`, which the student would have answered, to the teacher anyway (e.g. the whole request
        goes to the teacher for another reason). Recorded as a deferred row: training data, never calibration."""
        if i in routed.to_teacher:
            return
        r = routed.recs[i]
        r.served_by, r.channel, r.routing_reason = "teacher", "deferred", reason
        routed.results[i] = None
        routed.to_teacher.append(i)

    def complete(self, routed: Routed, outs: Sequence[TeacherOutput | Exception],
                 teacher_name: str | None = None) -> list[Result]:
        """Finish a `route`: `outs` are the teacher's answers for `routed.to_teacher`, in that order (an
        Exception for an item the teacher did not answer). Records every answered item and returns one Result
        per state; failed items get `label=None` and `error` set and are not recorded."""
        recs, results, to_teacher = routed.recs, routed.results, routed.to_teacher
        if len(outs) != len(to_teacher):
            raise ValueError(f"{len(outs)} teacher answers for {len(to_teacher)} items")
        teacher_name = teacher_name or self.teacher.name
        failed: dict[int, Exception] = {}
        for i, o in zip(to_teacher, outs, strict=True):
            if isinstance(o, Exception):
                failed[i] = o
                continue
            r = recs[i]
            r.teacher_label, r.teacher_probs, r.teacher_confidence = o.label, o.probs, o.confidence
            r.teacher_model, r.teacher_input_tokens = o.model or teacher_name, o.input_tokens
            r.teacher_cost_usd, r.teacher_request_id, r.teacher_latency_ms = o.cost_usd, o.request_id, o.latency_ms
            results[i] = Result(o.label, o.probs, o.confidence, "teacher", r.routing_reason, 0.0)

        n = len(recs)
        dt = (time.perf_counter() - routed.t0) * 1000 / max(n, 1)
        for i, r in enumerate(recs):
            r.latency_ms = dt
            if results[i] is not None:
                results[i].latency_ms = dt
        self.store.insert([r for i, r in enumerate(recs) if i not in failed])
        with self._state:
            if self._rows_unpruned is not None:
                self._rows_unpruned += n - len(failed)
        answered = [recs[i].teacher_model for i in to_teacher if i not in failed]
        if answered:
            self._observe_teacher(answered)
        if to_teacher:
            self._teacher_rate.add(len(to_teacher))
            n_audit = sum(recs[i].channel == "audit" for i in to_teacher if i not in failed)
            if n_audit:
                with self._state:
                    self._audit_new += n_audit
        self._after_batch()
        if failed:
            with self._state:
                self._teacher_errors += len(failed)
            log.warning("task %s: %d of %d teacher calls failed: %r", self.task.name, len(failed), n,
                        next(iter(failed.values())))
            for i, e in failed.items():
                results[i] = Result(None, {}, 0.0, "error", recs[i].routing_reason, dt, e)
        return results  # type: ignore[return-value]

    # ---- maintenance ---------------------------------------------------------------
    def _after_batch(self) -> None:
        if self.cfg.training == "inline":
            self.maintain()
        else:
            self._kick()

    def _kick(self) -> None:
        """Ask the background worker for a maintenance pass (no-op unless training == "background")."""
        if self.cfg.training == "background":
            with self._cv:
                if self._stop:
                    return
                self._kicks += 1
                if self._worker is None:
                    self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                                    name=f"jevstiller-maint-{self.task.name}")
                    self._worker.start()
                self._cv.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while self._kicks == self._done and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                seen = self._kicks                       # batches after this point get another pass
            started = time.monotonic()
            try:
                self.maintain()
            except Exception:
                log.exception("task %s: maintenance failed", self.task.name)
            with self._cv:
                self._done = seen
                self._cv.notify_all()
                # a pass scans the store; under heavy traffic, back-to-back passes would starve serving
                rest = started + self.cfg.maintenance_interval_s - time.monotonic()
                if rest > 0:
                    self._cv.wait_for(lambda: self._stop or self._draining, rest)

    def drain(self, timeout: float | None = None) -> bool:
        """Block until background maintenance has caught up with every batch so far. False on timeout."""
        with self._cv:
            target = self._kicks
            self._draining += 1                          # skip the pacing pause while someone waits
            self._cv.notify_all()
            try:
                return self._cv.wait_for(lambda: self._done >= target or self._stop, timeout)
            finally:
                self._draining -= 1

    def maintain(self) -> None:
        """One maintenance pass: judge the shadow candidate, train if due, check drift. Safe from any thread;
        passes are serialised. Called automatically unless `config.training == "manual"`."""
        with self._maint:
            if not self._pruned:                         # versions kept by earlier releases, or a lower keep_versions
                self._pruned = True
                try:
                    if n := self.registry.prune():
                        log.info("task %s: deleted %d old student versions (keep_versions=%d)", self.task.name, n,
                                 self.cfg.keep_versions)
                except OSError:
                    log.exception("task %s: deleting old student versions failed", self.task.name)
            if self._pending is not None and self._pending[0].done():
                self._finish_train()
            if self._shadow is not None:
                self._judge_shadow()
            if self._shadow is None and self._pending is None and self._should_train():
                self._start_train(wait=self.cfg.training == "inline")
            if self._prod is not None and self.mode == "cascade":
                with self._state:
                    due = self._audit_new >= self.cfg.drift_check_every
                    if due:
                        self._audit_new = 0
                if due:
                    self._check_drift()
            self._prune_rows()

    PRUNE_EVERY_ROWS = 5_000        # delete old rows every this many new rows (every pass while a backlog remains)

    def _prune_rows(self) -> None:
        """Delete the rows nothing reads any more (`store.prune`). Training and calibration read at most the newest
        `max_train_samples` / `max_calib_samples` rows of a lineage, so older ones go. So do rows the student
        answered alone past `keep_local_rows`: they teach nothing. A shadow's rows stay until it is judged."""
        with self._state:                                # most passes stop here, without touching the store
            unpruned = self._rows_unpruned
            if not self._prune_more and unpruned is not None and unpruned < self.PRUNE_EVERY_ROWS:
                return
            self._rows_unpruned = 0
            keep_after = self._shadow_started_id if self._shadow is not None else None
        keeps = (self.cfg.max_train_samples, self.cfg.max_calib_samples, self.cfg.keep_local_rows)
        if not any(keeps) or self.store.max_id() <= min(k for k in keeps if k):   # never held more than a limit
            return
        try:
            n, self._prune_more = self.store.prune(self.cfg.max_train_samples, self.cfg.max_calib_samples,
                                                   self.cfg.keep_local_rows, keep_after_id=keep_after)
        except sqlite3.Error:
            self._prune_more = False
            log.exception("task %s: deleting old rows failed", self.task.name)
            return
        if n:
            log.debug("task %s: deleted %d old rows%s", self.task.name, n, " (more to go)" if self._prune_more else "")

    def _readiness(self, c: dict) -> dict:
        """What the first student is waiting for, from `store.counts` of the current lineage."""
        per, need = c["per_class_train"], self.cfg.min_samples_per_class
        rare = {l: per.get(l, 0) for l in self.labels if per.get(l, 0) < need}
        enough = len(self.labels) - len(rare)
        blocked = bool(rare) and (self.cfg.rare_classes == "wait" or enough < 2)
        ready = (c["labelled_train"] >= self.cfg.min_train_samples and c["labelled_calib"] >= self.cfg.min_calib_samples
                 and not blocked)
        return {"ready": ready, "train": (c["labelled_train"], self.cfg.min_train_samples),
                "calib": (c["labelled_calib"], self.cfg.min_calib_samples), "min_per_class": need,
                "rare": rare, "blocked_by_rare": blocked, "rare_classes": self.cfg.rare_classes}

    READINESS_CHECK_ROWS = 100      # re-count the store for a first student at most every this many new rows

    def _current(self, bundle: Bundle | None) -> bool:
        """The version was trained on the data the loop trains on now: this teacher lineage, since the last
        confirmed drift."""
        return (bundle is not None and bundle.meta.get("teacher_model") in (None, self._teacher_model)
                and bundle.meta.get("since_id", 0) >= self._since_id)

    def _answers(self, after_id: int, upto_id: int | None = None) -> int:
        return self.store.answered(self.task.version, self._teacher_model, after_id, upto_id)

    def _should_train(self) -> bool:
        """Cheap checks first: a pass runs about once a second per loaded task, and counting scans the store.
        Without a production student for the current data, train once the data is ready (readiness);
        with one, retrain on request or every `min_new_samples` new teacher answers. Only answers count: a
        row the student answered teaches a retrain nothing, and at a high local share counting rows retrained
        (and stored a new version) on a few dozen new answers."""
        if self._teacher_model is None:
            return False
        latest = self.store.max_id()
        fresh = not self._current(self._prod)
        if fresh:
            if latest - self._counted_at < self.READINESS_CHECK_ROWS:
                return False
            tried = self._last_train_id > self._since_id  # a candidate on this data trained, and didn't make it:
            if tried and not self._retrain_requested:     # wait for 25% more answers (at most min_new_samples)
                self._counted_at = latest
                had = self._answers(self._since_id, self._last_train_id)
                if self._answers(self._last_train_id) < min(self.cfg.min_new_samples,
                                                            max(self.READINESS_CHECK_ROWS, had // 4)):
                    return False
        elif not self._retrain_requested:
            if (latest - self._last_train_id < self.cfg.min_new_samples       # as many answers need as many rows
                    or latest - self._counted_at < self.READINESS_CHECK_ROWS):
                return False
            self._counted_at = latest
            if self._answers(max(self._last_train_id, self._since_id)) < self.cfg.min_new_samples:
                return False
        self._counted_at = latest
        c = self.store.counts(self.task.version, self._teacher_model, self._since_id)
        if fresh:
            return self._readiness(c)["ready"]
        return c["labelled_calib"] >= self.cfg.min_calib_samples

    def train_now(self) -> TrainReport:
        """Train a candidate now and wait for it (the job itself runs on `train_executor` if set). If a
        training job is already queued or running, wait for that one instead."""
        with self._maint:
            if self._pending is not None:
                futures_wait((self._pending[0],))
                return self._finish_train(raise_errors=True)
            return self._start_train(wait=True, raise_errors=True)

    def _carry_state(self) -> dict:
        """In-memory progress worth keeping across an unload and reload of this task (TaskManager keeps it):
        audit answers towards the next drift check, the recent rate of teacher calls, and how far the pruning of old
        versions and rows has got (so a reload doesn't redo it)."""
        with self._state:
            return {"audit_new": self._audit_new, "teacher_rate": self._teacher_rate, "pruned": self._pruned,
                    "rows_unpruned": self._rows_unpruned, "prune_more": self._prune_more}

    def _restore_state(self, carry: dict) -> None:
        with self._state:
            self._audit_new += carry.get("audit_new", 0)
            self._teacher_rate = carry.get("teacher_rate", self._teacher_rate)
            self._pruned = self._pruned or carry.get("pruned", False)
            if carry.get("rows_unpruned") is not None:
                self._rows_unpruned = carry["rows_unpruned"] + (self._rows_unpruned or 0)
            self._prune_more = carry.get("prune_more", self._prune_more)

    def _training_priority(self) -> float:
        """How much training this task could save: its recent rate of teacher calls (decayed, per minute)."""
        return self._teacher_rate.value()

    def _start_train(self, wait: bool, raise_errors: bool = False) -> TrainReport | None:
        """Prepare a training job and run it: in this thread without `train_executor`; otherwise submit it,
        and either wait (`wait=True`) or return None and let a later maintenance pass adopt the result, so
        shadow judging and drift checks keep running while the job waits for a worker."""
        tv, eid = self.task.version, self.encoder.id
        self.store.flush()                               # the job reads the store from its own connection
        with self._state:
            self._last_train_id = self.store.max_id()
            self._retrain_requested = False
            lineage, since = self._teacher_model, self._since_id
        prod = self._prod if self._current(self._prod) else None   # else: other data, not a fair comparison
        staged = self.registry.staging_dir()
        job = FitJob(store_path=str(self.store.path), task_version=tv, encoder_id=eid, labels=self.labels,
                     dim=self.encoder.dim, out_dir=str(staged),
                     fit=dict(budget=self.task.budget * (1 - self.cfg.fit_headroom), delta=1 - self.cfg.confidence,
                              ood_quantile=self.cfg.ood_quantile, ood_k=self.cfg.ood_k,
                              epochs=self.cfg.student_epochs, l2=self.cfg.student_l2,
                              patience=self.cfg.student_patience, seed=self.cfg.seed,
                              threads=self.cfg.train_threads, ood_max_ref=self.cfg.ood_max_ref),
                     hard_labels=self.cfg.label_target == "hard",
                     importance_weighting=self.cfg.importance_weighting,
                     prod_dir=str(self.registry.root / prod.name.replace(":", "-")) if prod else None,
                     teacher_model=lineage, since_id=since, min_samples_per_class=self.cfg.min_samples_per_class,
                     defer_rare=self.cfg.rare_classes == "defer", max_train=self.cfg.max_train_samples,
                     max_calib=self.cfg.max_calib_samples)
        if self.train_executor is None:
            fut: Future = Future()
            try:
                fut.set_result(run_fit_job(job))
            except BaseException as e:
                fut.set_exception(e)
        else:
            fut = self.train_executor.submit(run_fit_job, job)
        self._pending = (fut, staged, lineage, since)
        if wait or self.train_executor is None:
            futures_wait((fut,))
            return self._finish_train(raise_errors)
        me = weakref.ref(self)                           # no cycle: an unloaded engine is freed at once
        fut.add_done_callback(lambda _f: (e := me()) is not None and e._kick())
        self.store.event("training_queued", teacher_model=lineage)
        return None

    def _finish_train(self, raise_errors: bool = False) -> TrainReport:
        """Adopt the finished job's result as a shadow candidate (or record why not). Holds `_maint`.
        A failed job is recorded (`train_failed` event) and retried on a later pass; `raise_errors` re-raises
        it instead (explicit `train_now`)."""
        fut, staged, lineage, since = self._pending
        self._pending = None
        tv, eid = self.task.version, self.encoder.id
        try:
            res = fut.result()
        except BaseException as e:
            shutil.rmtree(staged, ignore_errors=True)
            with self._state:
                self._retrain_requested = True           # try again on a later pass
            self.store.event("train_failed", error=repr(e)[:500])
            log.error("task %s: training failed: %r", self.task.name, e)
            if raise_errors:
                raise
            return TrainReport(None, 0, 0, None, float("nan"), 0.0, False, f"failed: {e!r}")
        if res.policy is None:
            shutil.rmtree(staged, ignore_errors=True)
            return TrainReport(None, res.n_train, res.n_calib, None, float("nan"), 0.0, False, "no data")
        if self._teacher_model != lineage or self._since_id != since:     # changed while this candidate trained
            shutil.rmtree(staged, ignore_errors=True)
            return TrainReport(None, res.n_train, res.n_calib, res.policy, float("nan"), res.calib_agreement,
                               False, "teacher changed during training" if self._teacher_model != lineage
                               else "drift during training")
        policy, fit = res.policy, res.fit
        meta = {"n_train": res.n_train, "n_calib": res.n_calib, "train_loss": fit["loss"], "epochs": fit["epochs"],
                "prod_calib_coverage": res.prod_calib_coverage,
                "calib_agreement": res.calib_agreement, "calib_accepted": res.calib_accepted,
                "calib_disagree": res.calib_disagree, "encoder": eid, "task_version": tv,
                "teacher_model": lineage, "since_id": since, "labels": self.labels,
                "deferred_labels": res.deferred_labels, "trained_to_id": self._last_train_id,
                "config": self.cfg.to_dict()}
        if not policy.usable:
            name = self.registry.adopt(staged, meta, state="rejected")
            self.store.event("rejected", version=name, reason="no threshold satisfies the budget",
                             n_calib=res.n_calib, calib_agreement=round(res.calib_agreement, 4))
            return TrainReport(name, res.n_train, res.n_calib, policy, fit["loss"], res.calib_agreement, False,
                               "no threshold satisfies the budget")
        start = meta["shadow_from_id"] = self.store.max_id()   # kept, so a reload does not restart the count
        name = self.registry.adopt(staged, meta, state="shadow")
        bundle = self._load(name)
        with self._state:
            self._shadow = bundle
            self._shadow_started_id = start
        self.store.event("shadow", version=name, n_train=res.n_train, n_calib=res.n_calib, epochs=fit["epochs"],
                         expected_coverage=round(policy.expected_coverage, 4),
                         disagreement_ub=round(policy.disagreement_ub, 4),
                         deferred_labels=res.deferred_labels or None)
        return TrainReport(name, res.n_train, res.n_calib, policy, fit["loss"], res.calib_agreement, True, "shadow")

    def _judge_shadow(self) -> None:
        sh = self._shadow
        if not self._current(sh):
            self.registry.set_state(sh.name, "rejected")
            with self._state:
                self._shadow = None
            other = sh.meta.get("teacher_model") not in (None, self._teacher_model)
            self.store.event("rejected", version=sh.name,
                             reason="trained on another teacher model" if other else "trained before a drift")
            return
        if self.store.max_id() - self._shadow_started_id < self.cfg.shadow_min_samples:
            return
        rows = self.store.shadow_records(sh.name, self.task.version, self._teacher_model, self._since_id)
        N = len(rows)
        if N == 0:
            return
        s_lab = np.array([r[0] for r in rows], dtype=object)
        s_conf = np.array([r[1] for r in rows], float)
        s_ood = np.array([r[2] for r in rows], float)
        t_lab = np.array([r[3] for r in rows])
        acc = sh.policy.accepts(s_conf, s_ood, s_lab)
        # Pool the calibration rows (used to fit the policy) with the fresh shadow rows: both are IID
        # teacher-labelled traffic, and pooling keeps the test powered while adding out-of-time evidence.
        # Same loss as the calibration: answered and disagreeing, over every row, bounded at the full budget.
        n = int(acc.sum()) + sh.meta.get("calib_accepted", 0)
        k = int((acc & (s_lab != t_lab)).sum()) + sh.meta.get("calib_disagree", 0)
        N_pool = N + sh.meta.get("n_calib", 0)
        ub = clopper_pearson_upper(k, N_pool, 1 - self.cfg.confidence)
        cov = n / N_pool
        ok = ub <= self.task.budget
        # not worse than production, compared on the same calibration rows (fresh shadow rows are too few)
        prod_cov = sh.meta.get("prod_calib_coverage")
        if ok and prod_cov is not None:
            ok = sh.policy.expected_coverage >= 0.95 * prod_cov
        detail = dict(version=sh.name, n_shadow=N, n_pooled=N_pool, coverage=round(cov, 4),
                      disagreement_ub=round(ub, 4),
                      budget=self.task.budget, production_coverage=prod_cov)
        if ok:
            prev = self.registry.promote(sh.name)
            with self._state:
                self._prod, self._shadow = sh, None
                self.forced_fallback = False
                self._suspicious = False
            self.store.event("promoted", previous=prev, **detail)
        else:
            self.registry.set_state(sh.name, "rejected")
            with self._state:
                self._shadow = None
            self.store.event("rejected", reason="shadow evaluation", **detail)

    def _audit_stats(self, prod: Bundle):
        """System agreement on the recent audit window: the audit rows served while `prod` was in production,
        scored as they were served. Re-scoring them with `prod` itself measured it on rows it had since been
        trained on (audit rows are training data), which overstated agreement (security audit run 3)."""
        s_lab, conf, ood, t_lab = self.store.audit_served(self.task.version, prod.name, self.cfg.drift_window,
                                                          self._teacher_model, self._since_id)
        N = len(t_lab)
        if N == 0:
            return 0, None, None, None
        acc = prod.policy.accepts(conf, ood, s_lab)
        k = int((acc & (s_lab != t_lab)).sum())
        d = 1 - self.cfg.confidence
        return N, 1 - k / N, 1 - clopper_pearson_upper(k, N, d), 1 - clopper_pearson_lower(k, N, d)

    def _check_drift(self) -> None:
        """Two-step ladder on the audit channel's system agreement A.

        suspicious: A < target                -> retrain (a candidate goes to shadow), audit rate raised
        broken:     ub(A) < target - margin   -> teacher_only until a candidate passes shadow

        When broken, the teacher's answers or the inputs have changed enough that the older answers no longer
        describe the task (a teacher can change its answers without changing its model name). Training,
        calibration, shadow and audit restart from rows after this point; older rows stay in the store.
        """
        N, a, lb, ub = self._audit_stats(self._prod)
        if N < self.cfg.drift_min_samples:
            return
        tgt = self.task.target_agreement
        if ub < tgt - self.cfg.drift_margin:
            since = self.store.max_id()
            with self._state:
                self.forced_fallback = True
                self._retrain_requested = True
                self._since_id = since
                dropped, self._shadow = self._shadow, None
            if dropped is not None:
                self.registry.set_state(dropped.name, "rejected")
            self.store.event("fallback", reason="audit agreement confidently below target", n=N,
                             agreement=round(a, 4), upper_bound=round(ub, 4), target=tgt, since_id=since,
                             dropped_shadow=dropped.name if dropped else None)
        elif a < tgt and self._shadow is None and not self._retrain_requested:
            with self._state:
                self._retrain_requested = True
                self._suspicious = True
            self.store.event("suspicious", reason="audit agreement below target", n=N,
                             agreement=round(a, 4), lower_bound=round(lb, 4), target=tgt)

    # ---- control ---------------------------------------------------------------
    def promote(self, version: str) -> None:
        with self._maint:
            if self.registry.state(version) is None:
                raise ValueError(f"unknown version {version!r}")
            bundle = self._load(version)
            self.registry.promote(version)
            with self._state:
                self._prod = bundle
                if self._shadow is not None and self._shadow.name == version:
                    self._shadow = None
                self.forced_fallback = False
            self.store.event("promoted", version=version, manual=True)

    def rollback(self) -> str | None:
        with self._maint:
            target = self.registry.rollback()
            bundle = self._load(target) if target else None
            with self._state:
                self._prod = bundle
            self.store.event("rollback", to=target)
            return target

    def status(self) -> Status:
        with self._state:
            lineage, since = self._teacher_model, self._since_id
        c = self.store.counts(self.task.version, lineage, since)
        sb = c["served_by"]
        tot = c["total"] or 1
        avoided = sb.get("student", 0)
        avg_cost = (c["teacher_cost_usd"] / c["teacher_calls"]) if c["teacher_calls"] else 0.0
        N = a = lb = ub = None
        with self._state:
            prod, shadow, mode, audit_rate = self._prod, self._shadow, self.mode, self.audit_rate
            teacher_errors = self._teacher_errors
        if prod is not None:
            N, a, lb, ub = self._audit_stats(prod)
        return Status(
            task=self.task.name, task_version=self.task.version, mode=mode, audit_rate=audit_rate,
            production=prod.name if prod else None, shadow=shadow.name if shadow else None,
            requests=c["total"], served_by_student=sb.get("student", 0), served_by_teacher=sb.get("teacher", 0),
            channels=c["channel"], student_share=sb.get("student", 0) / tot, teacher_share=sb.get("teacher", 0) / tot,
            audit_n=N or 0, audit_agreement=a, audit_agreement_lb=lb, audit_agreement_ub=ub,
            target_agreement=self.task.target_agreement,
            teacher_calls=c["teacher_calls"], teacher_calls_avoided=avoided, teacher_cost_usd=c["teacher_cost_usd"],
            teacher_cost_avoided_usd=avoided * avg_cost, labelled_train=c["labelled_train"],
            labelled_calib=c["labelled_calib"], policy=prod.policy.__dict__ if prod else None,
            events=self.store.events(), teacher_errors=teacher_errors, teacher_model=lineage,
            readiness=None if self._current(prod) else self._readiness(c))

    def versions(self) -> list[dict]:
        """Every student version with its state (candidate, shadow, production, superseded, rejected, rolled_back)."""
        return self.registry.versions()

    def export(self, path: str | Path, version: str | None = None) -> Path:
        """Copy one student version (head, OOD reference, routing policy, metadata) plus the task and encoder
        identity into a standalone directory. Enough to serve that version without the sample store."""
        name = version or self.registry.production
        if not name:
            raise ValueError("no production version to export")
        src = Path(self.registry.load(name).meta.get("dir") or (self.registry.root / name.replace(":", "-")))
        dst = Path(path)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        (dst / "task.json").write_text(json.dumps({
            "name": self.task.name, "instructions": self.task.instructions, "classes": self.task.classes,
            "target_agreement": self.task.target_agreement, "task_version": self.task.version,
            "encoder_id": self.encoder.id, "version": name, "labels": self.labels,
            "teacher_model": self.registry.load(name).meta.get("teacher_model"), "config": self.cfg.to_dict()},
            indent=2, ensure_ascii=False))
        return dst

    def evaluate(self, texts: Sequence[State], teacher_labels: Sequence[str], version: str | None = None,
                 X: np.ndarray | None = None) -> dict:
        """Offline check of a version's routing policy against teacher labels on held-out texts.

        Returns coverage, selective disagreement, system agreement (= 1 - coverage * selective
        disagreement), and per-row decisions. Does not touch the store.
        """
        b = self._prod if version is None else self._load(version)
        if b is None:
            return {"version": None}
        if X is None:
            X = self.encoder.encode([state_text(t) for t in texts])
        P, conf, ood = self._run(b, X)
        pred = np.array([self.labels[i] for i in P.argmax(axis=1)], dtype=object)
        acc = b.policy.accepts(conf, ood, pred)
        t = np.asarray(teacher_labels)
        n = int(acc.sum())
        k = int((acc & (pred != t)).sum())
        cov = n / len(t) if len(t) else 0.0
        sel = k / n if n else 0.0
        return {"version": b.name, "n": len(t), "coverage": cov, "selective_disagreement": sel,
                "system_disagreement": cov * sel, "system_agreement": 1 - cov * sel,
                "student_agreement_all": float((pred == t).mean()) if len(t) else 0.0,
                "accepted": acc, "student_label": pred, "student_conf": conf, "ood": ood}

    def _busy(self) -> bool:
        """A maintenance pass (training, shadow judging, drift checks) is running, or a training job is queued or
        running. A pass that is only requested doesn't count: under steady traffic nearly every task has one
        requested, and counting it kept the manager from unloading anything (598 tasks loaded against
        `max_loaded = 50`, 2.7 GB, in benchmarks/manager.py). Closing drops it; the next load requests another."""
        return self._maint.locked() or self._pending is not None

    def _footprint_bytes(self) -> int:
        """Memory held by the loaded versions (student heads and OOD references)."""
        n = 0
        for b in (self._prod, self._shadow):
            if b is not None:
                n += b.student.W.nbytes + b.student.b.nbytes + (b.ood.X.nbytes if b.ood.X is not None else 0)
        return n

    def close(self, timeout: float | None = None) -> None:
        """Stop the background worker (after its current pass) and close the store."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
        pending = self._pending
        if pending is not None:                          # never adopted now: drop the job and its directory
            fut, staged = pending[0], pending[1]
            fut.cancel()                                 # still queued: cancelled; running: finishes first
            fut.add_done_callback(lambda _f: shutil.rmtree(staged, ignore_errors=True))
            self._pending = None
        self.store.close()
