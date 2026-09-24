"""The loop: route, record, train, calibrate, shadow, promote, monitor, fall back."""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibrate import RoutingPolicy, clopper_pearson_lower, clopper_pearson_upper
from .encoders import Encoder, HashEncoder
from .registry import Bundle, Registry
from .store import Record, SampleStore, text_hash
from .task import MODES, Config, Task
from .teachers import Teacher
from .training import FitJob, run_fit_job

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

    def report(self) -> str:
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
        L.append(f"Labelled: train {self.labelled_train:,}   calib {self.labelled_calib:,}")
        if self.policy:
            p = self.policy
            L.append(f"Policy: conf>={p['conf_threshold']:.3f} ood<={p['ood_threshold']:.3f}  "
                     f"expected coverage {p['expected_coverage']:.1%}  disagreement ub {p['disagreement_ub']:.2%}")
        for e in self.events[-5:]:
            L.append(f"  event: {e['kind']} " + " ".join(f"{k}={v}" for k, v in e.items() if k not in ('kind', 'ts')))
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

    def __init__(self, task: Task, teacher: Teacher, data_dir: str | Path,
                 encoder: Encoder | None = None, config: Config | None = None,
                 train_executor: Executor | None = None):
        self.task = task
        self.teacher = teacher
        self.cfg = config or Config()
        self.encoder = encoder or HashEncoder()
        self.train_executor = train_executor
        self.dir = Path(data_dir) / task.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = SampleStore(self.dir / "samples.sqlite", self.cfg.calib_fraction)
        self.registry = Registry(self.dir / "versions")
        self.rng = np.random.default_rng(self.cfg.seed)
        self._state = threading.Lock()
        self._maint = threading.Lock()
        self.forced_fallback = False
        self._retrain_requested = False
        self._suspicious = False
        self._last_train_id = 0
        self._shadow_started_id = 0
        self._teacher_errors = 0
        self._prod: Bundle | None = self.registry.load(self.registry.production) if self.registry.production else None
        shadows = self.registry.in_state("shadow")
        self._shadow: Bundle | None = self.registry.load(shadows[-1]) if shadows else None
        if self._shadow is not None:
            self._shadow_started_id = self.store.max_id()
        self.labels = task.labels
        self._lidx = {c: i for i, c in enumerate(self.labels)}
        # background worker: `_kicks` counts batches that asked for maintenance, `_done` is the kick count
        # the last finished pass had seen; drain() waits for _done to catch up
        self._cv = threading.Condition()
        self._kicks = self._done = self._draining = 0
        self._stop = False
        self._worker: threading.Thread | None = None

    # ---- modes ----------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        with self._state:
            self.cfg.mode = mode
            if mode != "teacher_only":
                self.forced_fallback = False

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

    def classify(self, text: str) -> Result:
        """Classify one text. Raises TeacherError if it needed the teacher and the call failed."""
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: Sequence[str], errors: str = "raise") -> list[Result]:
        """Classify many texts. If some teacher calls fail, `errors="raise"` (default) raises TeacherError
        after recording the rest; `errors="return"` returns Results with `error` set and `label=None`."""
        if errors not in ("raise", "return"):
            raise ValueError("errors must be 'raise' or 'return'")
        t0 = time.perf_counter()
        n = len(texts)
        X = self.encoder.encode(texts)
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
            r = Record(text=texts[i] if self.cfg.store_text else "", text_hash=text_hash(texts[i]),
                       task_version=self.task.version, encoder_id=self.encoder.id,
                       embedding=X[i], served_by="teacher", routing_reason="", channel="")
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
                accept = pol.usable and conf[i] >= pol.conf_threshold and ood[i] <= pol.ood_threshold
                if accept:
                    r.served_by, r.channel, r.routing_reason = "student", "student", "confident"
                    results[i] = Result(r.student_label, r.student_probs, r.student_confidence,
                                        prod.name, "confident", 0.0)
                else:
                    r.channel = "deferred"
                    r.routing_reason = "ood" if (pol.usable and ood[i] > pol.ood_threshold) else "low_confidence"
                    to_teacher.append(i)
            recs.append(r)

        failed: dict[int, Exception] = {}
        if to_teacher:                                      # no lock held: callers' teacher calls overlap
            try:
                outs = list(self.teacher.classify([texts[i] for i in to_teacher], self.task))
                if len(outs) != len(to_teacher):
                    raise RuntimeError(f"teacher returned {len(outs)} answers for {len(to_teacher)} texts")
            except Exception as e:
                outs = [e] * len(to_teacher)
            for i, o in zip(to_teacher, outs, strict=True):
                if isinstance(o, Exception):
                    failed[i] = o
                    continue
                r = recs[i]
                r.teacher_label, r.teacher_probs, r.teacher_confidence = o.label, o.probs, o.confidence
                r.teacher_model, r.teacher_input_tokens = self.teacher.name, o.input_tokens
                r.teacher_cost_usd, r.teacher_request_id, r.teacher_latency_ms = o.cost_usd, o.request_id, o.latency_ms
                results[i] = Result(o.label, o.probs, o.confidence, "teacher", r.routing_reason, 0.0)

        dt = (time.perf_counter() - t0) * 1000 / max(n, 1)
        for i, r in enumerate(recs):
            r.latency_ms = dt
            if results[i] is not None:
                results[i].latency_ms = dt
        self.store.insert([r for i, r in enumerate(recs) if i not in failed])
        self._after_batch()
        if failed:
            with self._state:
                self._teacher_errors += len(failed)
            log.warning("task %s: %d of %d teacher calls failed: %r", self.task.name, len(failed), n,
                        next(iter(failed.values())))
            if errors == "raise":
                raise TeacherError(failed, results) from next(iter(failed.values()))
            for i, e in failed.items():
                results[i] = Result(None, {}, 0.0, "error", recs[i].routing_reason, dt, e)
        return results  # type: ignore[return-value]

    # ---- maintenance ---------------------------------------------------------------
    def _after_batch(self) -> None:
        if self.cfg.training == "inline":
            self.maintain()
        elif self.cfg.training == "background":
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
            if self._shadow is not None:
                self._judge_shadow()
            if self._shadow is None and self._should_train():
                self._train()
            if self._prod is not None and self.mode == "cascade":
                self._check_drift()

    def _should_train(self) -> bool:
        c = self.store.counts(self.task.version)
        if c["labelled_calib"] < self.cfg.min_calib_samples:
            return False
        if self._retrain_requested:
            return True
        if self._prod is None:
            per = c["per_class_train"]
            return (c["labelled_train"] >= self.cfg.min_train_samples
                    and all(per.get(l, 0) >= self.cfg.min_samples_per_class for l in self.labels))
        return (self.store.max_id() - self._last_train_id) >= self.cfg.min_new_samples

    def train_now(self) -> TrainReport:
        """Train a candidate now, in the calling thread (the fit itself goes to `train_executor` if set)."""
        with self._maint:
            return self._train()

    def _train(self) -> TrainReport:
        tv, eid = self.task.version, self.encoder.id
        self.store.flush()                               # the job reads the store from its own connection
        with self._state:
            self._last_train_id = self.store.max_id()
            self._retrain_requested = False
        prod = self._prod
        staged = self.registry.staging_dir()
        job = FitJob(store_path=str(self.store.path), task_version=tv, encoder_id=eid, labels=self.labels,
                     dim=self.encoder.dim, out_dir=str(staged),
                     fit=dict(budget=self.task.budget * (1 - self.cfg.fit_headroom), delta=1 - self.cfg.confidence,
                              ood_quantile=self.cfg.ood_quantile, ood_k=self.cfg.ood_k,
                              epochs=self.cfg.student_epochs, l2=self.cfg.student_l2,
                              patience=self.cfg.student_patience, seed=self.cfg.seed,
                              threads=self.cfg.train_threads),
                     hard_labels=self.cfg.label_target == "hard",
                     importance_weighting=self.cfg.importance_weighting,
                     prod_dir=str(self.registry.root / prod.name.replace(":", "-")) if prod else None)
        try:
            if self.train_executor is None:
                res = run_fit_job(job)
            else:
                res = self.train_executor.submit(run_fit_job, job).result()
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        if res.policy is None:
            shutil.rmtree(staged, ignore_errors=True)
            return TrainReport(None, res.n_train, res.n_calib, None, float("nan"), 0.0, False, "no data")
        policy, fit = res.policy, res.fit
        meta = {"n_train": res.n_train, "n_calib": res.n_calib, "train_loss": fit["loss"], "epochs": fit["epochs"],
                "prod_calib_coverage": res.prod_calib_coverage,
                "calib_agreement": res.calib_agreement, "calib_accepted": res.calib_accepted,
                "calib_disagree": res.calib_disagree, "encoder": eid, "task_version": tv,
                "config": self.cfg.to_dict()}
        if not policy.usable:
            name = self.registry.adopt(staged, meta, state="rejected")
            self.store.event("rejected", version=name, reason="no threshold satisfies the budget",
                             n_calib=res.n_calib, calib_agreement=round(res.calib_agreement, 4))
            return TrainReport(name, res.n_train, res.n_calib, policy, fit["loss"], res.calib_agreement, False,
                               "no threshold satisfies the budget")
        name = self.registry.adopt(staged, meta, state="shadow")
        bundle = self.registry.load(name)
        with self._state:
            self._shadow = bundle
            self._shadow_started_id = self.store.max_id()
        self.store.event("shadow", version=name, n_train=res.n_train, n_calib=res.n_calib, epochs=fit["epochs"],
                         expected_coverage=round(policy.expected_coverage, 4),
                         disagreement_ub=round(policy.disagreement_ub, 4))
        return TrainReport(name, res.n_train, res.n_calib, policy, fit["loss"], res.calib_agreement, True, "shadow")

    def _judge_shadow(self) -> None:
        sh = self._shadow
        if self.store.max_id() - self._shadow_started_id < self.cfg.shadow_min_samples:
            return
        rows = self.store.shadow_records(sh.name, self.task.version)
        N = len(rows)
        if N == 0:
            return
        s_lab = np.array([r[0] for r in rows])
        s_conf = np.array([r[1] for r in rows], float)
        s_ood = np.array([r[2] for r in rows], float)
        t_lab = np.array([r[3] for r in rows])
        acc = sh.policy.accepts(s_conf, s_ood)
        # Pool the calibration rows (used to fit the policy) with the fresh shadow rows: both are IID
        # teacher-labelled traffic, and pooling keeps the test powered while adding out-of-time evidence.
        n = int(acc.sum()) + sh.meta.get("calib_accepted", 0)
        k = int((acc & (s_lab != t_lab)).sum()) + sh.meta.get("calib_disagree", 0)
        N_pool = N + sh.meta.get("n_calib", 0)
        ub = clopper_pearson_upper(k, n, 1 - self.cfg.confidence)
        cov = n / N_pool
        ok = cov * ub <= self.task.budget
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
        """System agreement on the recent audit window, re-scored with the given production model."""
        X, t_lab = self.store.audit_window(self.task.version, self.encoder.id, self.encoder.dim, self.cfg.drift_window)
        N = len(t_lab)
        if N == 0:
            return 0, None, None, None
        P, conf, ood = self._run(prod, X)
        s_lab = np.array([self.labels[i] for i in P.argmax(axis=1)], dtype=object)
        acc = prod.policy.accepts(conf, ood)
        k = int((acc & (s_lab != t_lab)).sum())
        d = 1 - self.cfg.confidence
        return N, 1 - k / N, 1 - clopper_pearson_upper(k, N, d), 1 - clopper_pearson_lower(k, N, d)

    def _check_drift(self) -> None:
        """Two-step ladder on the audit channel's system agreement A.

        suspicious: A < target                -> retrain (a candidate goes to shadow), audit rate raised
        broken:     ub(A) < target - margin   -> teacher_only until a candidate passes shadow
        """
        N, a, lb, ub = self._audit_stats(self._prod)
        if N < self.cfg.drift_min_samples:
            return
        tgt = self.task.target_agreement
        if ub < tgt - self.cfg.drift_margin:
            with self._state:
                self.forced_fallback = True
                self._retrain_requested = True
            self.store.event("fallback", reason="audit agreement confidently below target", n=N,
                             agreement=round(a, 4), upper_bound=round(ub, 4), target=tgt)
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
            bundle = self.registry.load(version)
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
            bundle = self.registry.load(target) if target else None
            with self._state:
                self._prod = bundle
            self.store.event("rollback", to=target)
            return target

    def status(self) -> Status:
        c = self.store.counts(self.task.version)
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
            events=self.store.events(), teacher_errors=teacher_errors)

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
            "encoder_id": self.encoder.id, "version": name, "config": self.cfg.to_dict()}, indent=2))
        return dst

    def evaluate(self, texts: Sequence[str], teacher_labels: Sequence[str], version: str | None = None,
                 X: np.ndarray | None = None) -> dict:
        """Offline check of a version's routing policy against teacher labels on held-out texts.

        Returns coverage, selective disagreement, system agreement (= 1 - coverage * selective
        disagreement), and per-row decisions. Does not touch the store.
        """
        b = self._prod if version is None else self.registry.load(version)
        if b is None:
            return {"version": None}
        if X is None:
            X = self.encoder.encode(texts)
        P, conf, ood = self._run(b, X)
        acc = b.policy.accepts(conf, ood)
        pred = np.array([self.labels[i] for i in P.argmax(axis=1)])
        t = np.asarray(teacher_labels)
        n = int(acc.sum())
        k = int((acc & (pred != t)).sum())
        cov = n / len(t) if len(t) else 0.0
        sel = k / n if n else 0.0
        return {"version": b.name, "n": len(t), "coverage": cov, "selective_disagreement": sel,
                "system_disagreement": cov * sel, "system_agreement": 1 - cov * sel,
                "student_agreement_all": float((pred == t).mean()) if len(t) else 0.0,
                "accepted": acc, "student_label": pred, "student_conf": conf, "ood": ood}

    def close(self, timeout: float | None = None) -> None:
        """Stop the background worker (after its current pass) and close the store."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
        self.store.close()
