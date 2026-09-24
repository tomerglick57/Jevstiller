---
title: Roadmap
description: From a Python library to a self-hosted, drop-in Jev proxy.
---

Jevstiller 0.1.0 is a Python library. The goal is a **self-hosted proxy that speaks Jev's API**: point
`TYPESAFE_BASE_URL` at it and every service keeps its code and its own Jev key.

```text
 service A ─┐                         ┌──────── Jevstiller (one container) ────────┐
 service B ─┼─ TYPESAFE_BASE_URL ───► │ /v1/systemone  →  task per question        │ ──► Jev
 service C ─┘   + own Jev key         │ trained student answers, or forward to Jev │
                                      └────────────────────────────────────────────┘
```

## Next up

1. **Validate against live Jev.** Replace the oracle numbers with real ones.
2. **Concurrency.** Serve many callers per task and move training off the request path.
3. **Many tasks per process.** Tasks keyed by their exact question, with a shared encoder.
4. **The proxy.** `POST /v1/systemone` with Jev's exact response shape, and pass-through for everything else.
5. **Security.** A key must be accepted by Jev before the student answers for it. Keys are never stored.
6. **Operations.** Config file, health checks, Prometheus metrics, Docker images.

## Later

- Forward only the questions the student couldn't answer.
- Distil Jev's `noul` and `score` question types.
- Postgres store and multiple replicas.
- Class-list changes, per-class thresholds, class-weighted budgets.

The full, checkbox-level plan is in
[DEPLOYMENT_PLAN.md](https://github.com/tomerglick57/Jevstiller/blob/main/DEPLOYMENT_PLAN.md).
