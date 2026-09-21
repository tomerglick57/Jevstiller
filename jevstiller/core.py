"""The loop: route, record, train, calibrate, shadow, promote, monitor, fall back."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .calibrate import RoutingPolicy, clopper_pearson_lower, clopper_pearson_upper, fit_policy
from .encoders import Encoder, HashEncoder
from .ood import KnnOOD
from .registry import Bundle, Registry
from .store import Record, SampleStore
from .student import LinearStudent
from .task import Config, Task
from .teachers import Teacher, TeacherOutput


@dataclass
class Result:
    label: str
    probs: dict[str, float]
    confidence: float
    source: str                 # "teacher" | "student:vN"
    routing_reason: str
    latency_ms: float


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

    def report(self) -> str:
        L = []
        L.append(f"Task: {self.task}   version {self.task_version}   mode: {self.mode}   audit rate {self.audit_rate:.0%}")
        L.append(f"Production: {self.production or '-'}   Shadow: {self.shadow or '-'}")
        L.append(f"Requests: {self.requests:,}   student {self.student_share:.1%}   teacher {self.teacher_share:.1%}")
        ch = "   ".join(f"{k} {v:,}" for k, v in sorted(self.channels.items()))
        L.append(f"Channels: {ch}")
        L.append(f"Teacher calls: {self.teacher_calls:,}   avoided: {self.teacher_calls_avoided:,}   "
                 f"spent ${self.teacher_cost_usd:.4f}   avoided ${self.teacher_cost_avoided_usd:.4f}")
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
    def __init__(self, task: Task, teacher: Teacher, data_dir: str | Path,
                 encoder: Encoder | None = None, config: Config | None = None):
        self.task = task
        self.teacher = teacher
        self.cfg = config or Config()
        self.encoder = encoder or HashEncoder()
        self.dir = Path(data_dir) / task.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = SampleStore(self.dir / "samples.sqlite", self.cfg.calib_fraction)
        self.registry = Registry(self.dir / "versions")
        self.rng = np.random.default_rng(self.cfg.seed)
        self.forced_fallback = False
        self._retrain_requested = False
        self._suspicious = False
        self._last_train_id = 0
        self._shadow_started_id = 0
        self._prod: Bundle | None = self.registry.load(self.registry.production) if self.registry.production else None
        shadows = self.registry.in_state("shadow")
        self._shadow: Bundle | None = self.registry.load(shadows[-1]) if shadows else None
        if self._shadow is not None:
            self._shadow_started_id = self.store.max_id()
        self.labels = task.labels
        self._lidx = {c: i for i, c in enumerate(self.labels)}

    # ---- modes ----------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        assert mode in ("auto", "teacher_only", "cascade")
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
        if self.cfg.mode == "cascade":
            return "cascade" if self._prod else "teacher_only"
        return "cascade" if self._prod else "teacher_only"

    # ---- inference --------------------------------------------------------------
    def _run(self, b: Bundle, X: np.ndarray):
        P = b.student.predict_proba(X)
        conf = P.max(axis=1)
        ood = b.ood.score(X)
        return P, conf, ood

    def classify(self, text: str) -> Result:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: Sequence[str]) -> list[Result]:
        t0 = time.perf_counter()
        n = len(texts)
        X = self.encoder.encode(texts)
        mode = self.mode
        prod, shadow = self._prod, self._shadow
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
            r = Record(text=texts[i], task_version=self.task.version, encoder_id=self.encoder.id,
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
            elif self.rng.random() < self.audit_rate:
                r.channel = r.routing_reason = "audit"
                if self.cfg.importance_weighting:
                    r.weight = min((1.0 / self.audit_rate) ** self.cfg.weight_power, self.cfg.max_weight)
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

        if to_teacher:
            outs = self.teacher.classify([texts[i] for i in to_teacher], self.task)
            for i, o in zip(to_teacher, outs):
                r = recs[i]
                r.teacher_label, r.teacher_probs, r.teacher_confidence = o.label, o.probs, o.confidence
                r.teacher_model, r.teacher_input_tokens = self.teacher.name, o.input_tokens
                r.teacher_cost_usd, r.teacher_request_id, r.teacher_latency_ms = o.cost_usd, o.request_id, o.latency_ms
                results[i] = Result(o.label, o.probs, o.confidence, "teacher", r.routing_reason, 0.0)

        dt = (time.perf_counter() - t0) * 1000 / max(n, 1)
        for r, res in zip(recs, results):
            r.latency_ms = dt
            res.latency_ms = dt
        self.store.insert(recs)
        self._after_batch()
        return results  # type: ignore[return-value]

    # ---- the loop ---------------------------------------------------------------
    def _after_batch(self) -> None:
        if self._shadow is not None:
            self._judge_shadow()
        if self._shadow is None and self._should_train():
            self.train_now()
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
        tv, eid, dim = self.task.version, self.encoder.id, self.encoder.dim
        X, Y, _, w = self.store.training_set(tv, eid, self.labels, dim)
        Xc, _, yc, _ = self.store.calib_set(tv, eid, self.labels, dim)
        self._last_train_id = self.store.max_id()
        self._retrain_requested = False
        if len(X) == 0 or len(Xc) == 0:
            return TrainReport(None, len(X), len(Xc), None, float("nan"), 0.0, False, "no data")
        if self.cfg.label_target == "hard":
            Y = np.eye(len(self.labels), dtype=np.float32)[Y.argmax(axis=1)]
        student = LinearStudent(dim, len(self.labels))
        fit = student.fit(X, Y, epochs=self.cfg.student_epochs, l2=self.cfg.student_l2, seed=self.cfg.seed,
                          sample_weight=w if self.cfg.importance_weighting else None, patience=self.cfg.student_patience)
        ood = KnnOOD(self.cfg.ood_k)
        ood.fit(X, seed=self.cfg.seed)
        Pc = student.predict_proba(Xc)
        conf = Pc.max(axis=1)
        agree = Pc.argmax(axis=1) == yc
        oodc = ood.score(Xc)
        policy = fit_policy(conf, agree, oodc, self.task.budget * (1 - self.cfg.fit_headroom),
                            1 - self.cfg.confidence, self.cfg.ood_quantile)
        acc = policy.accepts(conf, oodc)
        prod_cov = None
        if self._prod is not None:                       # production's coverage on the same calibration rows
            _, pc, po = self._run(self._prod, Xc)
            prod_cov = float(self._prod.policy.accepts(pc, po).mean())
        meta = {"n_train": len(X), "n_calib": len(Xc), "train_loss": fit["loss"], "epochs": fit["epochs"],
                "prod_calib_coverage": prod_cov,
                "calib_agreement": float(agree.mean()), "calib_accepted": int(acc.sum()),
                "calib_disagree": int((acc & ~agree).sum()), "encoder": eid, "task_version": tv,
                "config": self.cfg.to_dict()}
        if not policy.usable:
            name = self.registry.save(student, ood, policy, meta, state="rejected")
            self.store.event("rejected", version=name, reason="no threshold satisfies the budget",
                             n_calib=len(Xc), calib_agreement=round(float(agree.mean()), 4))
            return TrainReport(name, len(X), len(Xc), policy, fit["loss"], float(agree.mean()), False,
                               "no threshold satisfies the budget")
        name = self.registry.save(student, ood, policy, meta, state="shadow")
        self._shadow = self.registry.load(name)
        self._shadow_started_id = self.store.max_id()
        self.store.event("shadow", version=name, n_train=len(X), n_calib=len(Xc), epochs=fit["epochs"],
                         expected_coverage=round(policy.expected_coverage, 4),
                         disagreement_ub=round(policy.disagreement_ub, 4))
        return TrainReport(name, len(X), len(Xc), policy, fit["loss"], float(agree.mean()), True, "shadow")

    def _judge_shadow(self) -> None:
        sh = self._shadow
        if self.store.max_id() - self._shadow_started_id < self.cfg.shadow_min_samples:
            return
        rows = self.store.shadow_records(sh.name, self.task.version)
        N = len(rows)
        if N == 0:
            return
        s_lab = np.array([r[0] for r in rows]); s_conf = np.array([r[1] for r in rows], float)
        s_ood = np.array([r[2] for r in rows], float); t_lab = np.array([r[3] for r in rows])
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
        detail = dict(version=sh.name, n_shadow=N, n_pooled=N_pool, coverage=round(cov, 4), disagreement_ub=round(ub, 4),
                      budget=self.task.budget, production_coverage=prod_cov)
        if ok:
            prev = self.registry.promote(sh.name)
            self._prod, self._shadow = self.registry.load(sh.name), None
            self.forced_fallback = False
            self._suspicious = False
            self.store.event("promoted", previous=prev, **detail)
        else:
            self.registry.set_state(sh.name, "rejected")
            self._shadow = None
            self.store.event("rejected", reason="shadow evaluation", **detail)

    def _audit_stats(self):
        """System agreement on the recent audit window, re-scored with the current production model."""
        X, t_lab = self.store.audit_window(self.task.version, self.encoder.id, self.encoder.dim, self.cfg.drift_window)
        N = len(t_lab)
        if N == 0:
            return 0, None, None, None
        P, conf, ood = self._run(self._prod, X)
        s_lab = np.array([self.labels[i] for i in P.argmax(axis=1)], dtype=object)
        acc = self._prod.policy.accepts(conf, ood)
        k = int((acc & (s_lab != t_lab)).sum())
        d = 1 - self.cfg.confidence
        return N, 1 - k / N, 1 - clopper_pearson_upper(k, N, d), 1 - clopper_pearson_lower(k, N, d)

    def _check_drift(self) -> None:
        """Two-step ladder on the audit channel's system agreement A.

        suspicious: A < target                -> retrain (a candidate goes to shadow), audit rate raised
        broken:     ub(A) < target - margin   -> teacher_only until a candidate passes shadow
        """
        N, a, lb, ub = self._audit_stats()
        if N < self.cfg.drift_min_samples:
            return
        tgt = self.task.target_agreement
        if ub < tgt - self.cfg.drift_margin:
            self.forced_fallback = True
            self.store.event("fallback", reason="audit agreement confidently below target", n=N,
                             agreement=round(a, 4), upper_bound=round(ub, 4), target=tgt)
            self._retrain_requested = True
        elif a < tgt and self._shadow is None and not self._retrain_requested:
            self.store.event("suspicious", reason="audit agreement below target", n=N,
                             agreement=round(a, 4), lower_bound=round(lb, 4), target=tgt)
            self._retrain_requested = True
            self._suspicious = True

    # ---- control ---------------------------------------------------------------
    def promote(self, version: str) -> None:
        self.registry.promote(version)
        self._prod = self.registry.load(version)
        self.forced_fallback = False
        self.store.event("promoted", version=version, manual=True)

    def rollback(self) -> str | None:
        target = self.registry.rollback()
        self._prod = self.registry.load(target) if target else None
        self.store.event("rollback", to=target)
        return target

    def status(self) -> Status:
        c = self.store.counts(self.task.version)
        sb = c["served_by"]; tot = c["total"] or 1
        avoided = sb.get("student", 0)
        avg_cost = (c["teacher_cost_usd"] / c["teacher_calls"]) if c["teacher_calls"] else 0.0
        N = a = lb = ub = None
        if self._prod is not None:
            N, a, lb, ub = self._audit_stats()
        return Status(
            task=self.task.name, task_version=self.task.version, mode=self.mode, audit_rate=self.audit_rate,
            production=self._prod.name if self._prod else None, shadow=self._shadow.name if self._shadow else None,
            requests=c["total"], served_by_student=sb.get("student", 0), served_by_teacher=sb.get("teacher", 0),
            channels=c["channel"], student_share=sb.get("student", 0) / tot, teacher_share=sb.get("teacher", 0) / tot,
            audit_n=N or 0, audit_agreement=a, audit_agreement_lb=lb, audit_agreement_ub=ub, target_agreement=self.task.target_agreement,
            teacher_calls=c["teacher_calls"], teacher_calls_avoided=avoided, teacher_cost_usd=c["teacher_cost_usd"],
            teacher_cost_avoided_usd=avoided * avg_cost, labelled_train=c["labelled_train"],
            labelled_calib=c["labelled_calib"], policy=self._prod.policy.__dict__ if self._prod else None,
            events=self.store.events())

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

    def close(self) -> None:
        self.store.close()
