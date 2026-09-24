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

- [x] **P0.1** Run `JevTeacher` against live Jev on a handful of requests. Check the label/probs/confidence mapping, `usage.input_tokens`, `request_id`, and the cost math. *Done when:* `tests/test_jev_adapter.py` has a live-marked twin that passes.
  *Result:* `tests/test_live.py` (`JEVSTILLER_LIVE=1 pytest -m live`): `JevTeacher` maps labels, probabilities, the resolved model (`jev:jev-1.13.0`), tokens, cost and request ids correctly, and the proxy in front of live Jev works with the SDK. Both pass.
- [x] **P0.2** Live benchmark: `experiments/run.py --dataset banking77 --teacher jev --encoder base` (and one smaller task). Record coverage, agreement with CI, Jev calls avoided, and $ spent. *Done when:* README numbers are replaced or confirmed, with the date and Jev model version.
  *Result (2026-09-24, `jev-1.13.0`, bge-small CPU):* 70.6% held-out coverage at 99.40% agreement (target 98%). Audit channel 99.27% [98.13%, 99.80%]. 5,270 Jev calls for 11,083 messages, $0.38. The system's accuracy on the true labels was 78.7% vs Jev's 78.55%. See docs/benchmarks.md and DESIGN.md §15.5.
- [x] **P0.3** Record real wire samples (a 200, 401, 422, 429, and 529 if one can be provoked) as JSON fixtures for the proxy contract tests (P6.1). Redact keys.
  *Result:* `experiments/jev_profile.py` wrote `tests/fixtures/jev/` (choice 200, mixed choice+noul+score 200, 401, 422, models). Responses carry no keys, which was checked. `tests/test_live.py` replays them through the proxy: the SDK parses each identically direct and via the proxy. No 429 or 529 was provoked.
- [x] **P0.4** Measure real Jev latency and 429 behaviour at our concurrency. Set the default per-key rate limit from that.
  *Result:* p50 ~290 ms, p90 ~330 ms, p99 ~760 ms, flat from 1 to 16 concurrent; 50 req/s at 16 concurrent with no 429 (a short burst, under the documented 1,200/min). A 5-class question costs ~400 input tokens, and the 77-class Banking77 question ~1,700. The default per-key limit is left to Jev's own 429s (P3.6); a pre-emptive limit needs a longer run at the documented ceiling.

## Phase 1 — Engine: correct under concurrency

The current `Jevstiller` class holds one lock across encoding, the Jev network call, the DB write, and training. That serialises everything: about 3 req/s per task.

- [x] **P1.1** Split `_classify_batch` into three steps: *decide* (under lock: read prod/shadow, draw audit, compute routing) → *call teacher* (no lock) → *record* (short lock or lock-free queue). *Done when:* 32 concurrent callers to one task reach at least 20× the current throughput with a slow fake teacher, and the audit rate stays correct.
  *Result:* 19 → ~500 req/s (25×) with 32 callers and a 50 ms teacher (`benchmarks/concurrency.py`). Profiling showed the per-request SQLite commit was the next bottleneck, so the store now writes behind: one writer thread commits in batches, and reads through the store flush first.
- [x] **P1.2** Move training off the request path. `_after_batch` only *enqueues* "train/judge/drift-check task X". A background worker does the work and atomically swaps in the new `Bundle`. *Done when:* no `classify` call ever runs `_train_now`, and p99 latency during training is unchanged.
  *Result:* `Config.training = background | inline | manual`. The background worker is paced by `maintenance_interval_s` (back-to-back passes halved throughput because each pass scans the store). Tests check that the fit never runs on a caller's thread.
- [x] **P1.3** Keep training from slowing serving. *Done when:* training 5 tasks at once doesn't move serving p99 by more than 10%.
  *Result:* the contention is CPU, not the GIL. BLAS spreads every fit over all cores, so where the fit runs matters less than how much CPU training gets. Now:
  - `Config.train_threads` (default 2) caps BLAS per fit (`threadpoolctl`).
  - The training job reads its own samples and writes its own bundle, so only small objects cross into the serving process.
  - `training.train_pool(workers=2)` is a shared, low-priority process pool: its size caps concurrent trainings, and the rest queue.

  Median over 3 interleaved runs, 5 tasks training while one serves 8 callers (`benchmarks/concurrency.py --during-training --rows 15000 --repeats 3`, 16 vCPU WSL):

  | Setup | p99 slowdown | 5 fits took |
  |---|---|---|
  | `train_pool(2)`, 2 threads each | ×1.12 (1.01 / 1.34 / 1.12) | ~28 s |
  | threads, 2 BLAS threads each | ×1.11 | ~71 s |
  | 5 low-priority workers, 2 threads each | ×2.79 | ~26 s |
  | 5 workers, BLAS uncapped | ×12.1 | ~44 s |

  The median is at the target within this machine's noise. P2's TaskManager should own one `train_pool` and size it from the core count.
- [x] **P1.4** Atomic registry writes (write temp file + `os.replace`) and fsync on promote. *Done when:* a crash-injection test (kill during save/promote) never leaves an unreadable `registry.json`.
- [x] **P1.5** Teacher failure isolation. One failed Jev call must not fail the batch. Per-item result or error. Timeouts on every upstream call.
- [x] **P1.6** Generalise `Task`: accept JSON values (str/object/array/None) for `instructions` and criteria descriptions, with canonical JSON in `version`. Allow up to 255 classes (Jev's max).
  *Result:* text task versions are unchanged, so existing data keeps its lineage. Found and fixed along the way: the version ignores class order, but a saved student's outputs were positional. Students now store their label order and are reordered on load. For the proxy this means two callers listing the same classes in a different order share one model safely.
- [x] **P1.7** State to text: canonical serialisation of object/array `state` for the encoder (stable key order). String state is unchanged.
  *Result:* canonical JSON (sorted keys, compact, UTF-8). The teacher receives the original object. The new `state_type` column is added to old stores on open.
- [x] **P1.8** Teacher identity in the lineage. Store the resolved `response.model`. A change in the resolved Jev model on the audit channel starts a new lineage (like a task-version bump) and triggers retraining.
  *Result:* the switch waits for `teacher_change_confirm` (20) answers in a row from the new model, so a gradual rollout behind an alias doesn't flap. `teacher_change`: `fallback` (default: all traffic to Jev until a new-lineage student passes shadow) or `audit` (keep serving, raised audit rate, drift monitor decides). The lineage survives restarts. A candidate that finishes training after a switch, or a shadow from the old lineage, is rejected.
- [x] **P1.9** Readiness diagnostics for rare classes. Today a task with a rare class never trains (`min_samples_per_class=50`). Surface "waiting on class X (12/50)" in status. Add an option to train when rare classes are always deferred to Jev.
  *Result:* `Status.readiness`, plus a report line such as `classes below 20: other 3 (blocking: set Config.rare_classes='defer' ...)`. With `rare_classes="defer"`, the job defers classes below the minimum, calibration only counts rows the student may answer, and routing sends any rare-class prediction to Jev. A test checks the agreement contract still holds.
- [x] **P1.10** Small fixes: `__version__` = `0.1.0` (and derive it from package metadata); `getattr(res, "request_id", None)` doesn't catch the SDK's `TypeSafeError`; replace `assert` in `set_mode` with `ValueError`; make DESIGN.md §13 modes match reality (`shadow`/`hedge` are not implemented); CI adds Python 3.13/3.14.

## Phase 2 — TaskManager: many tasks in one process

- [x] **P2.1** `TaskManager` keyed by `task_key`: get-or-create an `Engine`, with an LRU of loaded models (config: max loaded tasks / max memory). Unloading keeps everything on disk. *Done when:* 1,000 registered tasks with 50 active run inside a fixed memory cap.
  *Result (`benchmarks/manager.py --requests 60000`):* 1,000 tasks, 50 hot (90% of traffic), `max_loaded=50`, 8 threads, 5,600 loads/unloads. Throughput about 550 req/s. Hits: p50 2.8 ms, p99 7.5 ms. Reloading an unloaded task: p50 100 ms under load (4 ms alone). A first open, which creates the store: p50 210 ms. Student memory is capped at 118 MB.
  - Fixed on the way: BLAS oversubscription (`TaskManager(blas_threads=1)`: 2.4× throughput, p99 80 → 13 ms at 50 tasks). Also, maintenance now only works when something changed.
  - Also fixed: glibc kept freed arrays from unloaded tasks, and RSS climbed to 1.7 GB. It wasn't a leak: `malloc_trim` returns it. The manager now sets a fixed mmap threshold and trims after unloading.
  - **Still open:** RSS ends at ~440 MB but peaks around 1 GB mid-run. The next suspects are SQLite page caches (2 MB × 2–3 connections per loaded task: try a smaller `cache_size`) and whether it's bounded over hours (P6.6 soak test). Also: reload p50 is 100 ms under load against 4 ms alone. That's lock and disk contention with the janitor's unloads, which is worth profiling before P3 ships.
- [x] **P2.2** Shared encoder with micro-batching.
  *Result:* `BatchingEncoder` adds no waiting: one worker thread, and calls that queue up while it runs are merged into the next batch (optionally `max_wait_ms`). The manager takes one encoder for all tasks. Not benchmarked with a real GPU encoder yet.
- [x] **P2.3** Shared storage layout. *Decision: one SQLite file per task* (`<data_dir>/tasks/<key>/samples.sqlite`). Tasks never share a writer, deleting a task is deleting a directory (P4.6), and a failure in one store doesn't touch the others. Reopening costs ~4 ms, after the store stopped re-running its schema on every open (`PRAGMA user_version`). A `Store` protocol documents the interface for a future Postgres backend. That backend would also need a way for the training job to read it.
- [x] **P2.4** Shrink the OOD reference.
  *Result:* `Config.ood_max_ref` defaults to 5,000, sampled stratified by class (was 50,000 at random). Replays with a perfect teacher and bge-small: Banking77 held-out coverage is 65.7% at 50k and 5k, and 65.5% at 1k, with agreement 98.95% in all three. CLINC150 is 82.8% / 99.35% at 50k against 83.7% / 99.30% at 5k. Neither dataset exceeds ~6.4k training rows, so re-check on a large live task.
- [x] **P2.5** Task admission and GC.
  *Result:* `Admission(min_requests=50, window_s=86400)` counts in memory, capped at 100k candidate keys. Requests before admission go to the teacher unrecorded (`not_admitted`). `max_tasks_per_tenant` gives `tenant_task_limit`. `idle_ttl_s` (off by default) and `delete()` remove a task's directory.
- [x] **P2.6** Global scheduler for training jobs.
  *Result:* `TrainScheduler` is round-robin across tenants and orders by highest recent rate of teacher calls within a tenant. It retries with exponential backoff, replaces a broken process pool (found and fixed a deadlock: the pool's callback runs inside its own shutdown lock), and lets queued jobs be cancelled. Training is asynchronous with any executor, so a task waiting for a worker keeps judging its shadow and checking drift.

## Phase 3 — The proxy server

- [x] **P3.1** ASGI app (Starlette or FastAPI + uvicorn), async upstream client (httpx), `jevstiller serve` entry point. One process by default (in-memory models + SQLite). Document why.
- [x] **P3.2** `POST /v1/systemone`:
  - Parse and validate the body. On invalid input, forward it to Jev unchanged and let Jev return its own 422, so validation matches exactly.
  - For each `choice` question, route through its task's Engine.
  - If **every** question can be answered locally, respond locally. Otherwise **forward the full original request** upstream, return Jev's response as-is, and record each Choice answer as a training row (v1: simple and exact. Forwarding only the unanswered questions is a later optimisation, see Backlog).
  - `noul` / `score` questions: forwarded (and recorded for later) in v1.
- [x] **P3.3** Local response shape: exact Jev schema. `model` = the resolved Jev model name of the lineage (callers may compare it). `usage` = `{"input_tokens": 0, "output_tokens": 0}` (no Jev tokens were used). A generated `x-typesafe-request-id` (`jvs_…`). Extra headers: `x-jevstiller-source: student:v7|teacher`, `x-jevstiller-task: <key>`, `x-jevstiller-reason`.
- [x] **P3.4** Upstream pass-through fidelity. Forward the caller's headers (minus hop-by-hop), body, and query. Relay the status, body, and `retry-after*` / `x-typesafe-request-id` headers. The SDK's own retries then behave exactly as against Jev.
- [x] **P3.5** `GET /v1/models` and any unknown path: transparent pass-through.
- [x] **P3.6** Per-key upstream rate limiter (shared across all tasks using that key). Learn from `429` + `retry-after`. When a key is throttled and a request can't be answered locally, return Jev-style `429` with `retry-after` rather than queueing past the SDK's 10 s timeout.
- [x] **P3.7** Timeouts and backpressure. Upstream timeout < the client's (default 10 s). Bounded in-flight queue. Return `503` + `retry-after` when overloaded.
  *Phase 3 result:* `jevstiller/server.py` (Starlette + httpx) and `jevstiller serve`. Nine contract tests drive the real `typesafe-sdk` (sync and async) over HTTP against a fake Jev. They cover forwarding, training behind the proxy, local answers the SDK parses, 401 relaying and revocation, mixed choice + noul requests, 429 back-off, pass-through paths and invalid bodies, upstream down, and per-key tenancy. The proxy adds ~3 ms to a forwarded request. In the CLI end-to-end run (real process pool), 1,631 of 6,000 requests were answered locally within 21 s. Found on the way: the training pool wasn't shut down on SIGTERM (orphaned workers), now fixed. The requested `model` is part of the task key: callers asking `jev-preview` and `jev-1.13.0` are asking different teachers. P3.6 is back-off from 429 only: there's no pre-emptive per-key rate limit yet (Jev enforces its own, and a pre-limit needs P0.4's numbers).
- [ ] **P3.8** A Python in-process option, for teams that can't run a service: `jevstiller.TypeSafeClient` wrapper over the same Engine. Optional and small, once the proxy is done.

## Phase 4 — Security, tenancy, privacy

- [x] **P4.1** **Key verification before local answers.** Without it, anyone reaching the proxy with a made-up key would get student answers without ever touching Jev. Keep a cache `hash(key) → verified_at`. An unknown key's first request is always forwarded, and a 2xx marks it verified. A 401/403 from upstream (including on audit traffic) evicts it immediately. Re-verify after a TTL.
- [x] **P4.2** Never persist or log raw keys. Salted hash only (salt per deployment). Redact `Authorization` in all logs and errors (the SDK's `SECRET_HEADERS` list is a good reference).
- [x] **P4.3** Tenancy modes: `shared` (default: one tenant per deployment) and `per_key`. Plus an optional mapping file `key-hash → tenant` for grouping. The tenant is part of `task_key`, and no query crosses tenants.
  *Result:* `shared` / `per_key`, plus a tenants map (`tenants` / `tenants_file`, key hashes from `jevstiller key-hash`, validated at startup).
- [x] **P4.4** Optional proxy-level auth (e.g. an extra header or mTLS) for deployments that want to restrict who may use the proxy at all.
- [x] **P4.5** Admin API auth: a separate admin token. The admin API is off unless configured.
- [x] **P4.6** Data retention. `store_text` per deployment/task (off keeps only hash + embedding). Retention TTL for raw text. `DELETE` of a task or tenant (store + versions). Document what is stored and where.
- [x] **P4.7** TLS: document running behind a reverse proxy, and optionally built-in TLS (cert/key paths). Callers need a trusted cert if they use `https://`.
- [x] **P4.8** Run the `security-audit` pass before the first release (request smuggling via pass-through, header injection, path traversal in task names/export, pickle-free model loading — `np.load` must stay `allow_pickle=False`).
  *Result:* security audit run 1 (2026-09-24) against `65fe64b`. It ran reconnaissance, 4 hunters, adversarial validation, and per-finding independent verification. It found 12 confirmed findings (4 Medium, 6 Low, 2 Informational), all fixed in `1f29f42` with regression tests (`tests/test_audit_fixes.py`). Summary and open hardening items: docs/security.md. The admin API, metrics and settings code came after the audit; run it again to cover them.

## Phase 5 — Operations

- [x] **P5.1** Config: one YAML file + env overrides (`JEVSTILLER_*`). Covers per-task `target_agreement` defaults and overrides by `task_key` or by question name, encoder tier, limits, tenancy, retention. Validated on start.
- [x] **P5.2** `target_agreement` per task without code changes. Default from config, override via admin API. Callers can't pass it through the Jev API. Optionally honour a request header `x-jevstiller-target-agreement` (ignored by Jev).
- [x] **P5.3** Health: `/healthz` (process up), `/readyz` (encoder loaded, store writable, upstream reachable).
- [x] **P5.4** Prometheus `/metrics`: requests by source/reason/task, latency histograms (local vs upstream), Jev calls avoided and $ avoided, audit agreement and its bound per task, fallback events, training queue depth and durations, loaded-task count, memory.
- [x] **P5.5** Structured JSON logs, one line per request (task, source, reason, latencies, request IDs). No text or keys unless debug is explicitly enabled.
- [x] **P5.6** Admin API + CLI (`jevstiller tasks|status|versions|promote|rollback|mode|train|delete|export`), covering DESIGN.md §13's HTTP list under a separate prefix (e.g. `/jevstiller/v1/...`) so it never collides with Jev paths.
- [ ] **P5.7** Minimal read-only status page (HTML) served by the proxy: tasks, share served locally, agreement vs target, events. Optional, after metrics.
- [x] **P5.8** Docker images: `jevstiller:cpu` (ONNX Runtime) and `jevstiller:gpu` (CUDA). Encoder weights baked in or downloaded on first start into a volume. Run as non-root. `docker-compose.yml` example with a data volume.
- [x] **P5.9** Graceful shutdown (drain in-flight, flush the store, finish or abandon training cleanly) and startup recovery (reload registry, resume shadows).
- [x] **P5.10** Backup/restore doc and command: what directory to snapshot, and a consistent SQLite backup.
- [x] **P5.11** Kubernetes: Helm chart or plain manifests, single replica + PVC. Mark multi-replica as unsupported until P2.3's Postgres backend exists.
  *Phase 5 result:*
  - Settings are a TOML file, `JEVSTILLER_*` env vars and flags (`jevstiller/settings.py`, strict validation, secrets from files, `jevstiller config`).
  - Per-task target and mode: `[tasks]` in the file, or the admin API, persisted in `task.json`.
  - `/healthz` and `/readyz` (manager, data dir, encoder).
  - Prometheus `/metrics`, gated by the admin token.
  - JSON access logs.
  - Admin API `/jevstiller/v1/*` with `jevstiller admin`.
  - Docker image (non-root, bge-small baked in, runs offline, read-only root, capabilities dropped: tested).
  - `docker-compose.yml`, and a Kubernetes manifest (1 replica, Recreate, probes, security context).
  - Graceful shutdown closes the upstream client, the manager and the training pool, and a restart resumes shadows and pending state.
  - `jevstiller backup` / `restore` (online-consistent, tested while writing).
  - Deferred: P5.7 (status page).

## Phase 6 — Testing

- [x] **P6.1** Proxy contract tests with the **real `typesafe-sdk`** (sync + async) against the proxy, as in the 2026-09-24 check. Cover local answers, forwarded answers, 401/422/429/5xx pass-through, `request_id` present, multi-question requests, object `state`, non-Choice questions. Run in CI against the pinned SDK and the latest SDK.
- [x] **P6.2** Golden fixtures from P0.3 replayed through the proxy. The response bodies must parse identically to direct Jev.
- [ ] **P6.3** Concurrency tests: many tasks × many threads/async callers. The audit rate stays at its target ±CI. No lost or double-written rows.
- [ ] **P6.4** Load test (locust or k6): throughput and p50/p99 for local-only, forward-only, and mixed traffic. Targets go in the README.
- [ ] **P6.5** Chaos: Jev down, Jev slow, 429 storms, 401 on a previously verified key, disk full, kill -9 during training/promote. Each has an expected behaviour written down and tested.
- [ ] **P6.6** Soak test: 24 h replay with drift injected midway. Check fallback → retrain → re-promote without intervention, and flat memory.
- [ ] **P6.7** End-to-end with live Jev through the proxy (after P0): one real task from cold start to promoted student. Record the report.

## Phase 7 — Docs and release

- [x] **P7.1** README rewrite around the proxy: "set one env var". Quickstart with `docker run` + `export TYPESAFE_BASE_URL=...`. The library API moves to a secondary section.
  *Result:* README leads with the proxy and a Docker quickstart.
- [x] **P7.2** `docs/deploy.md`: install, config reference, TLS, sizing (CPU vs GPU, memory per task), backup, upgrade.
  *Result:* docs/deploy.md: Docker, compose, Kubernetes, pip, configuration decisions, TLS and reverse proxies, sizing, upgrade.
- [x] **P7.3** `docs/operations.md` runbook: reading status, what fallback means, rollback, deleting data, common alerts.
  *Result:* docs/operations.md: health, admin CLI, status reports, metrics and alerts, logs, data, common situations.
- [x] **P7.4** `docs/security.md` + update `SECURITY.md`: key handling, what is stored, tenancy guarantees, threat model.
  *Result:* docs/security.md: threat model, what is stored, keys, access controls, audit results; SECURITY.md points to it.
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
