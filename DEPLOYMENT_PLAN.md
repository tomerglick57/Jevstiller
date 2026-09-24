# Jevstiller — Deployment Plan

From library (0.1.0) to a self-hosted, drop-in Jev proxy that runs many tasks for many services at once.

Status: draft, 2026-09-24. Tasks are checkboxes; tick them as they land.

---

## Decisions

| # | Decision | Consequence |
|---|---|---|
| D1 | **Drop-in = HTTP proxy speaking Jev's wire API.** Users set `TYPESAFE_BASE_URL` and change no code. | We must match `POST /v1/systemone` and `GET /v1/models` byte-for-byte in shape, headers, and errors. |
| D2 | **Self-hosted first.** A company runs Jevstiller inside its own network. A hosted/paid offering, if any, lives in a **separate repo** that imports this one. | Keep the engine a library with pluggable interfaces (storage, key validation, metering hooks) so the hosted repo can extend it without forking. No billing, signup, or multi-company UI here. |
| D3 | **Callers use their own Jev API keys.** The proxy forwards the caller's `Authorization` header upstream. | Jevstiller never needs a Jev key of its own. Keys are never stored — only a salted hash, used to identify and verify the caller. |

## Verified: the SDK supports the proxy (2026-09-24, `typesafe-sdk==0.7.1`)

- `TypeSafeClient` / `AsyncTypeSafeClient` read `TYPESAFE_BASE_URL` (or `base_url=`), default `https://api.typesafe.ai`, and call `base_url + "/v1/systemone"` (`typesafe_sdk/_core/config.py`, `_core/constants.py`).
- A fake local server, reached only through the env var, served the unmodified sync and async clients. The SDK parsed `choices`, `probabilities`, `usage`, and `request_id`, and retried a `429` with `retry-after` on its own. The server received the caller's `Authorization: Bearer <key>` untouched.

Wire facts the proxy must honour (from the SDK source):

- **Request**: `{"state": str | object | array, "model": str, "questions": {name: {"type": "choice"|"noul"|"score", "instructions"?, "criteria"}}}`. `instructions` and criteria descriptions may be strings, objects, or arrays, not only strings.
- **Response**: `{"model", "answers": {name: {"type": "choice", "choice", "confidence", "probabilities"}}, "usage": {"input_tokens", "output_tokens"}}`. `answers` and `usage` are required.
- **Header `x-typesafe-request-id` is effectively required**: `response.request_id` raises `TypeSafeError` if it is missing.
- **Retries**: the SDK retries `408`, `429`, and every `5xx`, honouring `retry-after` / `retry-after-ms`. The error body is parsed into `TypeSafeAPIError`.
- **Default client timeout is 10 s.** Nothing slow (training, model loading) may ever run on the request path.
- `response.model` "may differ from the alias supplied in the request". `jev-latest` resolves to a concrete version, and that is the real teacher identity.

---

## Architecture target

```text
 service A ─┐                         ┌──────────────── Jevstiller (one container) ────────────────┐
 service B ─┼─ TYPESAFE_BASE_URL ───► │ HTTP (/v1/systemone, /v1/models)                           │
 service C ─┘   + own Jev key         │   ├─ key check (hash → verified?)                           │
                                      │   ├─ parse questions → task key per Choice question         │
                                      │   ├─ TaskManager: task key → Engine (LRU-loaded)            │
                                      │   │     └─ Engine.route(): student | teacher | audit         │
                                      │   ├─ upstream client → api.typesafe.ai (caller's key,       │
                                      │   │     per-key rate limiter, pass-through for the rest)    │
                                      │   └─ response in Jev's exact shape (+ x-jevstiller-* hdrs)  │
                                      │ Shared: one encoder (micro-batched), sample store, registry │
                                      │ Background: training worker pool, shadow judge, GC          │
                                      │ Ops: /healthz /readyz /metrics, admin API, CLI              │
                                      └─────────────────────────────────────────────────────────────┘
```

**Task identity (how "find a model that's already trained" works).** For every Choice question in a request:

```text
task_key = hash(tenant, question.type, canonical_json(instructions), canonical_json(criteria), encoder_id)
lineage  = task_key + resolved Jev model (from response.model)
```

The match is exact, never "similar". Jev reads its criteria literally, so one changed word is a different task. A task key hit loads that task's production student (in memory, LRU). A miss starts the normal flow: forward to Jev, record, train when there is enough data.

**Tenant.** In a self-hosted install the default is one tenant for the whole deployment, so all of a company's services share trained models. This is the main win. An option `isolation: per_key` keeps each API key's data and models separate. Either way, a request must come with a key that Jev has accepted before the student may answer it (see P4.1).

---

## Phase 0 — Validate against live Jev (blocking, needs credits)

- [ ] **P0.1** Run `JevTeacher` against live Jev on a handful of requests. Check the label/probs/confidence mapping, `usage.input_tokens`, `request_id`, and the cost math. *Done when:* `tests/test_jev_adapter.py` has a live-marked twin that passes.
- [ ] **P0.2** Live benchmark: `experiments/run.py --dataset banking77 --teacher jev --encoder base` (and one smaller task). Record coverage, agreement with CI, Jev calls avoided, and $ spent. *Done when:* README numbers are replaced or confirmed, with the date and Jev model version.
- [ ] **P0.3** Record real wire samples (a 200, 401, 422, 429, and 529 if one can be provoked) as JSON fixtures for the proxy contract tests (P6.1). Redact keys.
- [ ] **P0.4** Measure real Jev latency and 429 behaviour at our concurrency. Set the default per-key rate limit from that.

## Phase 1 — Engine: correct under concurrency

The current `Jevstiller` class holds one lock across encoding, the Jev network call, the DB write, and training. That serialises everything: about 3 req/s per task.

- [ ] **P1.1** Split `_classify_batch` into three steps: *decide* (under lock: read prod/shadow, draw audit, compute routing) → *call teacher* (no lock) → *record* (short lock or lock-free queue). *Done when:* 32 concurrent callers to one task reach at least 20× the current throughput with a slow fake teacher, and the audit rate stays correct.
- [ ] **P1.2** Move training off the request path. `_after_batch` only *enqueues* "train/judge/drift-check task X". A background worker does the work and atomically swaps in the new `Bundle`. *Done when:* no `classify` call ever runs `_train_now`, and p99 latency during training is unchanged.
- [ ] **P1.3** Train in a **process** pool, not threads. numpy training and kNN fitting hold the GIL long enough to hurt serving. Set a max number of concurrent trainings (config). *Done when:* training 5 tasks at once doesn't move serving p99 by more than 10%.
- [ ] **P1.4** Atomic registry writes (write temp file + `os.replace`) and fsync on promote. *Done when:* a crash-injection test (kill during save/promote) never leaves an unreadable `registry.json`.
- [ ] **P1.5** Teacher failure isolation. One failed Jev call must not fail the batch. Per-item result or error. Timeouts on every upstream call.
- [ ] **P1.6** Generalise `Task`: accept JSON values (str/object/array/None) for `instructions` and criteria descriptions, with canonical JSON in `version`. Allow up to 255 classes (Jev's max).
- [ ] **P1.7** State to text: canonical serialisation of object/array `state` for the encoder (stable key order). String state is unchanged.
- [ ] **P1.8** Teacher identity in the lineage. Store the resolved `response.model`. A change in the resolved Jev model on the audit channel starts a new lineage (like a task-version bump) and triggers retraining.
- [ ] **P1.9** Readiness diagnostics for rare classes. Today a task with a rare class never trains (`min_samples_per_class=50`). Surface "waiting on class X (12/50)" in status. Add an option to train when rare classes are always deferred to Jev.
- [ ] **P1.10** Small fixes: `__version__` = `0.1.0` (and derive it from package metadata); `getattr(res, "request_id", None)` doesn't catch the SDK's `TypeSafeError`; replace `assert` in `set_mode` with `ValueError`; make DESIGN.md §13 modes match reality (`shadow`/`hedge` are not implemented); CI adds Python 3.13/3.14.

## Phase 2 — TaskManager: many tasks in one process

- [ ] **P2.1** `TaskManager` keyed by `task_key`: get-or-create an `Engine`, with an LRU of loaded models (config: max loaded tasks / max memory). Unloading keeps everything on disk. *Done when:* 1,000 registered tasks with 50 active run inside a fixed memory cap.
- [ ] **P2.2** A shared encoder across all tasks, with **micro-batching**. Collect requests over ~2–5 ms or N items and run one encoder batch (big win on GPU). One encoder instance per process.
- [ ] **P2.3** Shared storage layout. One SQLite DB for the deployment, with `task_key` on every row and index (replacing `max_id()` across tasks), or one DB per task with a shared connection pool. Pick one and benchmark it at 1k tasks. Keep the store behind an interface (`SampleStore` protocol) so Postgres can be added later (and by the hosted repo).
- [ ] **P2.4** Shrink the OOD reference. `KnnOOD` keeps up to 50k × dim float32 per task (~150 MB at dim 768). Cap it (e.g. 5k, stratified by class) or use float16. Check that coverage and agreement don't regress on the Banking77 replay.
- [ ] **P2.5** Task admission and GC. Many callers build questions dynamically (per-request criteria), which would create unbounded one-off tasks. Only start collecting training rows once a `task_key` has been seen N times within a window (cheap counter first). Set a cap on tasks per tenant. Remove tasks idle beyond a TTL (configurable, off by default).
- [ ] **P2.6** Global scheduler for training jobs: priority by traffic × expected savings, fairness across tenants, retries.

## Phase 3 — The proxy server

- [ ] **P3.1** ASGI app (Starlette or FastAPI + uvicorn), async upstream client (httpx), `jevstiller serve` entry point. One process by default (in-memory models + SQLite). Document why.
- [ ] **P3.2** `POST /v1/systemone`:
  - Parse and validate the body. On invalid input, forward it to Jev unchanged and let Jev return its own 422, so validation matches exactly.
  - For each `choice` question, route through its task's Engine.
  - If **every** question can be answered locally, respond locally. Otherwise **forward the full original request** upstream, return Jev's response as-is, and record each Choice answer as a training row (v1: simple and exact. Forwarding only the unanswered questions is a later optimisation, see Backlog).
  - `noul` / `score` questions: forwarded (and recorded for later) in v1.
- [ ] **P3.3** Local response shape: exact Jev schema. `model` = the resolved Jev model name of the lineage (callers may compare it). `usage` = `{"input_tokens": 0, "output_tokens": 0}` (no Jev tokens were used). A generated `x-typesafe-request-id` (`jvs_…`). Extra headers: `x-jevstiller-source: student:v7|teacher`, `x-jevstiller-task: <key>`, `x-jevstiller-reason`.
- [ ] **P3.4** Upstream pass-through fidelity. Forward the caller's headers (minus hop-by-hop), body, and query. Relay the status, body, and `retry-after*` / `x-typesafe-request-id` headers. The SDK's own retries then behave exactly as against Jev.
- [ ] **P3.5** `GET /v1/models` and any unknown path: transparent pass-through.
- [ ] **P3.6** Per-key upstream rate limiter (shared across all tasks using that key). Learn from `429` + `retry-after`. When a key is throttled and a request can't be answered locally, return Jev-style `429` with `retry-after` rather than queueing past the SDK's 10 s timeout.
- [ ] **P3.7** Timeouts and backpressure. Upstream timeout < the client's (default 10 s). Bounded in-flight queue. Return `503` + `retry-after` when overloaded.
- [ ] **P3.8** A Python in-process option, for teams that can't run a service: `jevstiller.TypeSafeClient` wrapper over the same Engine. Optional and small, once the proxy is done.

## Phase 4 — Security, tenancy, privacy

- [ ] **P4.1** **Key verification before local answers.** Without it, anyone reaching the proxy with a made-up key would get student answers without ever touching Jev. Keep a cache `hash(key) → verified_at`. An unknown key's first request is always forwarded, and a 2xx marks it verified. A 401/403 from upstream (including on audit traffic) evicts it immediately. Re-verify after a TTL.
- [ ] **P4.2** Never persist or log raw keys. Salted hash only (salt per deployment). Redact `Authorization` in all logs and errors (the SDK's `SECRET_HEADERS` list is a good reference).
- [ ] **P4.3** Tenancy modes: `shared` (default: one tenant per deployment) and `per_key`. Plus an optional mapping file `key-hash → tenant` for grouping. The tenant is part of `task_key`, and no query crosses tenants.
- [ ] **P4.4** Optional proxy-level auth (e.g. an extra header or mTLS) for deployments that want to restrict who may use the proxy at all.
- [ ] **P4.5** Admin API auth: a separate admin token. The admin API is off unless configured.
- [ ] **P4.6** Data retention. `store_text` per deployment/task (off keeps only hash + embedding). Retention TTL for raw text. `DELETE` of a task or tenant (store + versions). Document what is stored and where.
- [ ] **P4.7** TLS: document running behind a reverse proxy, and optionally built-in TLS (cert/key paths). Callers need a trusted cert if they use `https://`.
- [ ] **P4.8** Run the `security-audit` pass before the first release (request smuggling via pass-through, header injection, path traversal in task names/export, pickle-free model loading — `np.load` must stay `allow_pickle=False`).

## Phase 5 — Operations

- [ ] **P5.1** Config: one YAML file + env overrides (`JEVSTILLER_*`). Covers per-task `target_agreement` defaults and overrides by `task_key` or by question name, encoder tier, limits, tenancy, retention. Validated on start.
- [ ] **P5.2** `target_agreement` per task without code changes. Default from config, override via admin API. Callers can't pass it through the Jev API. Optionally honour a request header `x-jevstiller-target-agreement` (ignored by Jev).
- [ ] **P5.3** Health: `/healthz` (process up), `/readyz` (encoder loaded, store writable, upstream reachable).
- [ ] **P5.4** Prometheus `/metrics`: requests by source/reason/task, latency histograms (local vs upstream), Jev calls avoided and $ avoided, audit agreement and its bound per task, fallback events, training queue depth and durations, loaded-task count, memory.
- [ ] **P5.5** Structured JSON logs, one line per request (task, source, reason, latencies, request IDs). No text or keys unless debug is explicitly enabled.
- [ ] **P5.6** Admin API + CLI (`jevstiller tasks|status|versions|promote|rollback|mode|train|delete|export`), covering DESIGN.md §13's HTTP list under a separate prefix (e.g. `/jevstiller/v1/...`) so it never collides with Jev paths.
- [ ] **P5.7** Minimal read-only status page (HTML) served by the proxy: tasks, share served locally, agreement vs target, events. Optional, after metrics.
- [ ] **P5.8** Docker images: `jevstiller:cpu` (ONNX Runtime) and `jevstiller:gpu` (CUDA). Encoder weights baked in or downloaded on first start into a volume. Run as non-root. `docker-compose.yml` example with a data volume.
- [ ] **P5.9** Graceful shutdown (drain in-flight, flush the store, finish or abandon training cleanly) and startup recovery (reload registry, resume shadows).
- [ ] **P5.10** Backup/restore doc and command: what directory to snapshot, and a consistent SQLite backup.
- [ ] **P5.11** Kubernetes: Helm chart or plain manifests, single replica + PVC. Mark multi-replica as unsupported until P2.3's Postgres backend exists.

## Phase 6 — Testing

- [ ] **P6.1** Proxy contract tests with the **real `typesafe-sdk`** (sync + async) against the proxy, as in the 2026-09-24 check. Cover local answers, forwarded answers, 401/422/429/5xx pass-through, `request_id` present, multi-question requests, object `state`, non-Choice questions. Run in CI against the pinned SDK and the latest SDK.
- [ ] **P6.2** Golden fixtures from P0.3 replayed through the proxy. The response bodies must parse identically to direct Jev.
- [ ] **P6.3** Concurrency tests: many tasks × many threads/async callers. The audit rate stays at its target ±CI. No lost or double-written rows.
- [ ] **P6.4** Load test (locust or k6): throughput and p50/p99 for local-only, forward-only, and mixed traffic. Targets go in the README.
- [ ] **P6.5** Chaos: Jev down, Jev slow, 429 storms, 401 on a previously verified key, disk full, kill -9 during training/promote. Each has an expected behaviour written down and tested.
- [ ] **P6.6** Soak test: 24 h replay with drift injected midway. Check fallback → retrain → re-promote without intervention, and flat memory.
- [ ] **P6.7** End-to-end with live Jev through the proxy (after P0): one real task from cold start to promoted student. Record the report.

## Phase 7 — Docs and release

- [ ] **P7.1** README rewrite around the proxy: "set one env var". Quickstart with `docker run` + `export TYPESAFE_BASE_URL=...`. The library API moves to a secondary section.
- [ ] **P7.2** `docs/deploy.md`: install, config reference, TLS, sizing (CPU vs GPU, memory per task), backup, upgrade.
- [ ] **P7.3** `docs/operations.md` runbook: reading status, what fallback means, rollback, deleting data, common alerts.
- [ ] **P7.4** `docs/security.md` + update `SECURITY.md`: key handling, what is stored, tenancy guarantees, threat model.
- [ ] **P7.5** Compatibility statement: supported `typesafe-sdk` versions and Jev API surface; what is forwarded vs served locally; the "agreement ≠ accuracy" note.
- [ ] **P7.6** Release automation: PyPI trusted publishing, GHCR image publishing on tag, CHANGELOG, versioning policy. Keep DESIGN.md in sync (it still says "HTTP server out of scope").
- [ ] **P7.7** Public API stability pass on what the hosted repo will import (`Engine`, `TaskManager`, store/validator/metering protocols). Mark everything else private.

---

## Suggested order

1. **P0** (as soon as credits land) runs alongside **P1.1–P1.5** and **P1.10**. The engine fixes don't need Jev.
2. **P1.6–P1.9** → **P2.1–P2.3** → **P3.1–P3.5** + **P4.1–P4.2** + **P6.1**. This is the first usable proxy (single tenant, one container).
3. **P3.6–P3.7**, **P5.1–P5.6**, **P5.8–P5.9**, **P6.2–P6.5**. Release candidate.
4. **P4.3–P4.8**, **P2.4–P2.6**, **P6.6–P6.7**, **P7**. 1.0.
5. Everything else, then the backlog.

## Backlog (after 1.0)

- Forward only the questions the student couldn't answer (cheaper). Needs evidence that Jev answers questions independently, since DESIGN.md Appendix A notes "no structural invariants".
- Distil `noul` (binary) and `score` (ordinal) questions.
- Postgres store + multiple replicas.
- Warm-start a new task's student from a closely related task's student (training speed-up only; never serve with it).
- Degraded mode: when Jev is down or rate-limited, optionally serve the student below threshold, clearly flagged in headers. Off by default because it breaks the contract.
- Class-list changes, per-class thresholds, class-weighted budgets (DESIGN.md §16).
- Hooks the hosted repo needs: metering events, per-tenant quotas, external key validator.

## Open questions

1. Should the local response's `model` be the resolved Jev model (max compatibility) or something like `jevstiller/student:v7` (more honest)? Plan: Jev's name in `model`, source in headers. Revisit if users object.
2. Admission threshold for new tasks (P2.5): how many sightings before we start collecting? Start with 50 in 24 h. Tune with real traffic.
3. SQLite single file vs per-task files at 1k+ tasks (P2.3). Decide by benchmark.
