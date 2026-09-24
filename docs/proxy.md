# The drop-in proxy

`jevstiller serve` runs an HTTP server that speaks Jev's API. Services point the TypeSafe SDK at it and change nothing else:

```bash
export TYPESAFE_BASE_URL=http://jevstiller:8080
```

This works because the SDK takes its server address from `TYPESAFE_BASE_URL` (or `base_url=`) and calls `base_url + "/v1/systemone"`. Verified against `typesafe-sdk` 0.7.1, sync and async clients.

## What happens to a request

```text
POST /v1/systemone  {state, model, questions}  +  Authorization: Bearer <caller's key>
        │
        ├─ access checks (network, token, one Authorization header, body limit) ── fail ─► 4xx, never forwarded
        ├─ not understood (bad JSON, unknown top-level fields, no key, no choice question,
        │  more than max_questions distinct questions) ────────────────────────────► forward as is, record nothing
        │
        ├─ key NOT accepted by Jev yet ─► forward the whole request first; if Jev answers it (2xx, a valid
        │                                 systemone answer), the key becomes accepted and the choice answers
        │                                 are recorded as training rows
        │
        └─ key accepted (within key_ttl_s):
              each distinct choice question ─► its task: key = hash(tenant, "choice",
                                                  sha256(instructions, criteria), requested model)
                                               └─ the task's engine decides: student or teacher
              ├─ every question answerable locally ─► local response, in Jev's exact shape
              └─ otherwise ─► the WHOLE original request goes to Jev with the caller's key; Jev's response
                              is returned unchanged; each choice answer becomes a training row
```

- **Matching is exact.** Jev reads criteria literally, so a changed word is a different task. The order of classes and of JSON keys doesn't matter. The question is identified by its full SHA-256.
- **The requested `model` is part of the key.** Callers asking `jev-preview` and `jev-1.13.0` are asking different teachers.
- **Sharing:** services asking the same question (same tenant) share one task and one student.
- **Admission:** a new question becomes a task only after `--admit-after` requests (default 50 within 24 h). Before that its requests are forwarded and not recorded. This stops per-request dynamic questions from creating endless one-off tasks.
- **Partial answers don't happen.** A request is either answered entirely locally or entirely by Jev. When one question needs Jev, the other questions' local answers are discarded, recorded as `co_deferred` training rows.
- **Duplicate questions** (the same instructions and criteria under several names) are routed once and answered identically. At most `max_questions` (32) distinct choice questions are routed; a larger request is forwarded without routing.
- **The state is encoded once per request**, whatever the number of questions.

## Local responses

Same JSON as Jev:

```json
{"model": "jev-1.13.0",
 "answers": {"label": {"type": "choice", "choice": "billing", "confidence": 0.93,
                       "probabilities": {"billing": 0.95, "technical": 0.01, "...": 0.0}}},
 "usage": {"input_tokens": 0, "output_tokens": 0}}
```

- `model` is the concrete Jev version the task's student was trained against (what `jev-latest` resolved to).
- `confidence` uses Jev's definition, the peakedness of the distribution `(K·max − 1)/(K − 1)`, computed on the student's probabilities.
- `usage` is zero: no Jev tokens were spent.
- `x-typesafe-request-id` is `jvs_<uuid>`. The SDK requires the header, and the prefix tells you it was local.

Extra headers on every `/v1/systemone` response:

| Header | Values |
|---|---|
| `x-jevstiller-source` | `local` or `upstream` |
| `x-jevstiller-detail` | JSON, per question: the student version (`student:v7`) when local; the reason when forwarded (`co_deferred`, `key_unverified`, `not_admitted`, `tenant_task_limit`, `task_limit`, `unsupported`). Replaced by `{"questions": n}` above 64 questions. |

## API keys

- The caller's `Authorization` header is forwarded unchanged. Jevstiller has no Jev key of its own.
- Keys are never stored or logged. The proxy keeps a salted SHA-256 of each key in memory. The salt is a per-deployment secret created on first start at `<data_dir>/key-salt` (mode 0600).
- **A key is accepted only after Jev answered a `/v1/systemone` request made with it** (a 2xx whose body is a valid systemone answer), and stays accepted for `key_ttl_s` (1 h, refreshed by every such answer).
  - A 2xx from any other path (`/v1/models`, a public schema) does not count.
  - A 401/403 from Jev on any path revokes the key at once.
  - Until a key is accepted, nothing is answered locally for it and no task is created from its requests. They are forwarded first and recorded only after Jev's successful answer.
- **Tenancy.**
  - `shared` (default): every key's tasks are shared, so all of a company's services benefit.
  - `per_key`: each key's tasks, data and students are separate.
  - A tenants map (`tenants` or `tenants_file`: key hash → tenant name, hashes from `jevstiller key-hash`) groups chosen keys.

## Errors and limits

| Situation | Proxy response |
|---|---|
| Jev returns any status | returned unchanged (body, `x-typesafe-request-id`, `retry-after*`) |
| Jev returned 429 with `retry-after` | the proxy answers that key's forwarded requests with 429 + `retry-after` until then, without calling Jev (local answers continue) |
| Jev doesn't answer within `upstream_timeout_s` (9 s) | 504 |
| Jev unreachable | 502 |
| More than `max_upstream_inflight` (256) forwarded requests in flight | 503, `retry-after: 1` |
| Requests arrive faster than the encoder can embed them: a local answer would wait more than `max_encoder_wait_ms` (200 ms) | forwarded to Jev as it is (not routed or recorded; `questions_total{outcome="overloaded"}`), so the proxy is never much slower than Jev. The encoder is the local path's ceiling: ~150–340 texts/s for bge-small on 16 CPU cores, far more on a GPU. |
| Any error inside the proxy's own logic, or a body it can't parse | the request is forwarded instead |
| Network not in `allow_networks` | 403 |
| Missing or wrong `x-jevstiller-token` (when `access_token` is set) | 401 |
| More than one `Authorization` header, or `.`/`..` path segments | 400 |
| Body over `max_body_mb` (declared or streamed) | 413 |

Errors the proxy generates use Jev's shape (`{"detail": "..."}` with a `jvs_` request id), so the SDK raises the usual `TypeSafeAPIError` subclasses and retries 429/5xx as it would against Jev.

## Other endpoints

Served by the proxy itself, never forwarded:

| Path | What |
|---|---|
| `GET /healthz` | liveness: `{"ok": true}`, no authentication |
| `GET /readyz` | readiness: 200 `{"ready": true, "checks": {...}}` or 503 (manager open, data dir writable, encoder loaded); no authentication |
| `GET /metrics` | Prometheus metrics; needs the admin token unless `metrics_public` |
| `/jevstiller/v1/*` | the admin API (see [operations.md](operations.md)); off (404) without an admin token |

Other methods on `/healthz` and `/readyz` get 405. Every other path and method (`GET /v1/models`, anything new) is forwarded unchanged.

## Performance (measured, see [benchmarks.md](benchmarks.md))

- A forwarded request costs about 3 ms extra (p50 4.0 ms against 0.9 ms direct, on a local stub).
- Local answers with 1,000 registered tasks and 50 hot: p50 2.8 ms, p99 7.5 ms, about 550 req/s on 8 threads with the hash encoder. A real encoder adds its per-text cost: bge-small ONNX on CPU about 1.6 ms/text batched.

## Not yet

- Pre-emptive per-key rate limiting (today: back-off after a 429).
- Answering part of a request locally and forwarding only the rest.
- Distilling `noul` / `score` questions (they're always forwarded).
- Queueing audit/deferred calls while Jev is unreachable.
