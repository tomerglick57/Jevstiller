---
title: Roadmap
description: What 0.2.0 shipped, what is next, and the longer-term backlog.
---

## Shipped in 0.2.0

The self-hosted, drop-in Jev proxy: point `TYPESAFE_BASE_URL` at it, and every service keeps its code and its own
Jev key.

```text
 service A ─┐                         ┌──────── Jevstiller (one container) ────────┐
 service B ─┼─ TYPESAFE_BASE_URL ───► │ /v1/systemone  →  task per question        │ ──► Jev
 service C ─┘   + own Jev key         │ trained student answers, or forward to Jev │
                                      └────────────────────────────────────────────┘
```

- **Validated against live Jev:** Banking77, 70.7% held-out coverage at 99.45% agreement (target 98%).
- **Many tasks, many tenants, one process:** training runs in the background, fair across tenants.
- **Security:** keys are verified by Jev before any local answer and never stored. Two security audits, all
  findings fixed.
- **Operations:** a TOML config, an admin API and CLI, Prometheus metrics, health checks, JSON logs, backup and
  restore.
- **Packaging:** a hardened container image and a Kubernetes manifest.
- **A sound guarantee:** a joint-loss bound with fixed-sequence testing, a permanent audit, and automatic
  fallback. Teacher-model changes and silent drift are handled by themselves.

## Next

- **A read-only status page** served by the proxy: tasks, share answered locally, agreement against target, events.
- **An in-process option** for teams that can't run a service: a `TypeSafeClient`-compatible wrapper over the same
  engine.
- **A public API stability pass** ahead of 1.0.

## Later

- **Tighter, longer-lived guarantees:**
  - calibrate on post-deployment traffic with inverse-probability weights;
  - anytime-valid monitoring (confidence sequences);
  - a confidence budget spread across retrains;
  - one learned deferral score instead of two thresholds.
- **Forward only the questions** the student couldn't answer, instead of the whole request.
- **Distil Jev's `noul` and `score` question types.**
- **Postgres store and multiple replicas.**
- **Per-class thresholds and class-weighted budgets.**

The full, checkbox-level plan is in
[DEPLOYMENT_PLAN.md](https://github.com/tomerglick57/Jevstiller/blob/main/DEPLOYMENT_PLAN.md).
