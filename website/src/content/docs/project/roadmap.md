---
title: Roadmap
description: What has shipped, what is next, and the longer-term backlog.
---

## Shipped

**0.2.0:** the self-hosted, drop-in Jev proxy. Point `TYPESAFE_BASE_URL` at it, and every service keeps its code
and its own Jev key.

```text
 service A ─┐                         ┌──────── Jevstiller (one container) ────────┐
 service B ─┼─ TYPESAFE_BASE_URL ───► │ /v1/systemone  →  task per question        │ ──► Jev
 service C ─┘   + own Jev key         │ trained student answers, or forward to Jev │
                                      └────────────────────────────────────────────┘
```

- **Validated against live Jev:** Banking77, 70.7% held-out coverage at 99.45% agreement (target 98%).
- **Many tasks, many tenants, one process:** training runs in the background, fair across tenants.
- **Security:** keys are verified by Jev before any local answer and never stored.
- **Operations:** a TOML config, an admin API and CLI, Prometheus metrics, health checks, JSON logs, backup and
  restore.
- **Packaging:** a hardened container image and a Kubernetes manifest.

**0.3.0:**
- **A status page** at `/jevstiller/status`.
- **A defined public Python API,** pinned by a test.
- **A third security audit:** the audit channel now scores answers as they were served, calibration is split per
  request, and there is a per-key quota on new tasks.
- **Hash-pinned release builds.**

**0.3.4:**
- **Bounded disk use.** Old model versions and stored requests that nothing reads any more are deleted. A busy task
  levels off at ~300 MB. The 24-hour soak's data went from 53 GB to 4.3 GB.
- **A 24-hour soak:** memory bounded, and a silent change in Jev's answers at hour 12 recovered without intervention.
- **A benchmark on five public tasks,** with Jev's recorded answers, so anyone can rerun it.
- `rare_classes` defaults to `defer`: a label Jev never uses no longer blocks a task's first student.

**0.4.0:**
- **The guarantee can cover your confidence check.** Set a task's `confidence_floor` to the Jev confidence below
  which your code treats answers as unsure. The target then counts those requests too, and local answers report at
  least the floor. Without it, a check at 0.6 lost 8–37% of the flags Jev would have raised, on the benchmark.
- **A local answer's `Result.confidence`** in the Python API uses Jev's definition, as the proxy already did.
- **Apache 2.0** instead of MIT.

**0.5.0:**
- **Yes/no (`noul`) questions are answered locally,** on by default. You state the cut-off your code compares
  Jev's probability with (0.5 unless set), or two for an unsure band, and the target covers landing on the same
  outcome as Jev. 98% answered locally where Jev is decisive, 22–36% on noisy questions.

**0.5.1:**
- **Fixes from the fourth security audit** (docs/security.md, run 4): stored text is cut in bytes and dropped when the
  disk is nearly full, admin changes no longer race loads and deletes, one caller can't flush others' admission
  counts, and the website's deploy token no longer shares a job with the build.

## Next

- **A tolerance for yes/no answers.** Since 0.5 a `noul` answer is covered for its side of your cut-off. For
  pipelines that use the number itself: the local probability within ±ε of Jev's on at least the target share of
  requests. Measured as realistic where Jev is decisive (±0.1 on 96% of requests for two of four benchmark
  questions) and weak where it isn't (12–18%).
- **A second teacher: OpenAI's Decisions API.** Announced at DevDay on 2026-09-29, it has the same shape as Jev's
  `Choice`: a question, a finite set of answers, a confidence. The loop doesn't care which teacher labels a task, only
  the adapter does, so a Decisions API adapter is the natural next one, as soon as the API is generally available and
  its responses carry per-answer probabilities (the soft labels are worth 2–3 points of coverage).
- **An OpenAI-compatible endpoint.** The proxy speaks Jev's API only. A `/v1/chat/completions` front that maps a
  constrained-choice prompt onto the same task loop would let the same local model, guarantee and audit sit in front
  of other providers' classification calls.
- **Admin controls on tasks that are loading:** a mode or target change during a load reaches the live engine.
- **An in-process option** for teams that can't run a service: a `TypeSafeClient`-compatible wrapper over the same
  engine.
- **Hooks for a hosted version:** a pluggable store, an external key validator, metering.

## Later

- **Tighter, longer-lived guarantees:**
  - calibrate on post-deployment traffic with inverse-probability weights;
  - anytime-valid monitoring (confidence sequences);
  - a confidence budget spread across retrains;
  - one learned deferral score instead of two thresholds.
- **Forward only the questions** the student couldn't answer, instead of the whole request.
- **Distil Jev's `score` questions** (ordinal).
- **Postgres store and multiple replicas.**
- **Per-class thresholds and class-weighted budgets.**

The full, checkbox-level plan is in
[DEPLOYMENT_PLAN.md](https://github.com/tomerglick57/Jevstiller/blob/main/DEPLOYMENT_PLAN.md).
