# Security

## Threat model

Jevstiller is a self-hosted proxy on a company network. Services call it with their own Jev API keys. It forwards requests to Jev, trains local models from Jev's answers, and answers some requests itself.

| Actor | Trust | What the proxy guarantees |
|---|---|---|
| Caller service (anything that can reach the port) | untrusted input | It never gets a local answer unless its key was accepted by Jev; it cannot create tasks, write training data, or exhaust the proxy's disk without a key Jev accepts; it cannot reach any host but the configured upstream |
| Another caller / tenant | untrusted | No access to other tenants' tasks or answers in `per_key` tenancy; in `shared` tenancy (default) all callers share tasks **by design** |
| Jev (the upstream) | trusted | Its answers are training labels; its 401/403/429 decide key state |
| Operator | trusted | Config, data directory, admin token, tenants map |
| Someone with the data directory or a backup | trusted as much as your traffic | Holds request text (unless disabled/retained), embeddings and models, but never an API key |

## What is stored

Per task, under `<data_dir>/tasks/<key>/` (directories 0700, files 0600 when run with `jevstiller serve`):

- `samples.sqlite`: one row per request. It holds the state's text or canonical JSON (unless `store_text = false`), a 64-bit HMAC of it (keyed with the deployment salt), the embedding, Jev's answer and distribution, the student's answer, routing metadata, and the upstream request id. There's no API key and no caller identity beyond the tenant.
- `versions/`: trained models (numpy arrays, JSON policies). They contain up to 5,000 training embeddings per version (the OOD reference).
- `task.json`: the question (instructions and criteria), tenant, requested model, and timestamps.

Plus `<data_dir>/key-salt` (32 random bytes, 0600). It is used to hash API keys and text. Losing it means tenant-map entries no longer match; leaking it makes key hashes guessable offline for known keys.

Controls:
- `store_text = false`: keep no request text at all.
- `text_retention_days`: blank text older than N days. The old text is securely deleted: `secure_delete` plus a WAL truncate.
- `jevstiller admin delete <key>` / `delete-tenant <tenant>`: remove a task or a tenant's tasks completely.
- Back up the data directory as you would a database of your traffic (`jevstiller backup`).

## API keys

- The caller's `Authorization` header is forwarded to Jev unchanged; Jevstiller has no Jev key of its own.
- Keys are held only as `sha256(salt + key)` in memory, never written to disk or logs.
- **A key is "accepted" only after Jev answered a `/v1/systemone` request made with it** (a 2xx whose body parses as a Jev answer), and only for `key_ttl_s` (1 h).
  - Until then its requests are forwarded, and are recorded as training data only after Jev's successful answer.
  - A 401/403 from Jev on any path revokes it at once.
  - A 429 makes the proxy back off for that key.
- A request with more than one `Authorization` header is rejected (400).

## Access to the proxy

By default anyone who can reach the port can use it, with their own Jev key. Restrict it with any of these:

| Setting | Effect |
|---|---|
| `allow_networks = ["10.0.0.0/8"]` | Only these client networks (the TCP peer; `X-Forwarded-For` is honoured only from `trust_forwarded_for` proxies) |
| `access_token` / `access_token_file` | Callers must send `x-jevstiller-token: <token>`. The TypeSafe SDK can add it with `TypeSafeClient(headers={...})`, which is a one-line code change. The header is never forwarded to Jev. |
| TLS | `ssl_certfile` / `ssl_keyfile`, or a reverse proxy in front |
| `admin_token` / `admin_token_file` | Enables the admin API (`/jevstiller/v1/*`) and gates `/metrics`. Without it both are off (404). |

Always enforced, for every path:
- a body limit (`max_body_mb`, 4 MiB), checked on the declared length and while streaming;
- no `.`/`..` path segments;
- at most `max_questions` (32) distinct questions routed per request;
- task caps (`max_tasks` 10,000, `max_tasks_per_tenant` 1,000).

`GET`/`HEAD` on `/healthz` and `/readyz` need no token (probes). The proxy serves `/healthz`, `/readyz`, `/metrics` and `/jevstiller/*` itself and never forwards them.

## Security audit, 2026-09-24

A multi-agent audit ran against commit `65fe64b`: reconnaissance, four hunters, adversarial validation, and independent verification of every finding. All attacks were reproduced locally against mock upstreams. Twelve findings were confirmed and fixed in `1f29f42`, each with a regression test in `tests/test_audit_fixes.py`.

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | Medium | Any 2xx from any forwarded path marked a key as accepted, so a made-up key could get local answers via an upstream path that doesn't check keys | Only a parsed `/v1/systemone` answer accepts a key |
| 2 | Medium | `/healthz` / `/readyz` were exempt from access checks for every method and forwarded upstream; unbounded bodies there | Served locally for every method; only GET/HEAD skip the checks |
| 3 | Medium | Callers with no valid key created persistent tasks; admission counted per question name | Unaccepted keys are forwarded first and recorded after Jev's answer; admission per distinct question; task caps |
| 4 | Medium | Unbounded questions per request multiplied encoder work and memory | ≤ 32 distinct questions routed; one routing per question; state encoded once per request |
| 5 | Low | Chunked bodies bypassed the body limit | Limit enforced while streaming |
| 6 | Low | Duplicate `Authorization` headers: first hashed, last forwarded | Rejected |
| 7 | Low | `..` in paths escaped a path prefix in `upstream` | Rejected |
| 8 | Low | `X-Forwarded-For` from loopback bypassed `allow_networks` | Honoured only from `trust_forwarded_for` |
| 9 | Low | Retained text survived in the WAL and free pages; world-readable data files; unsalted text hash | secure_delete + WAL truncate; umask 0077 / 0700; HMAC |
| 10 | Low | Task identity used 48 bits of the question hash (collision into another task) | Full SHA-256 fingerprint; old tasks re-keyed automatically |
| 11 | Info | Upstream cookies were shared across callers | No cookie jar |
| 12 | Info | Config mistakes failed open (e.g. `STORE_TEXT=false` ignored, `key-hash` creating a new salt) | Strict settings validation; `key-hash` never creates a salt |

The audit also verified these properties:
- No injection or unsafe deserialization: SQL values are bound, `np.load` refuses pickles, and there is no `eval`/`exec`/`subprocess`.
- The upstream host can't be changed, and there are no redirects.
- Responses never cross between concurrent callers.
- Header CR/LF injection is impossible.
- 500s carry no stack traces.

**Open hardening items:**
- No Host-header allowlist (DNS rebinding reaches an unauthenticated proxy from a browser; the access token stops it).
- Raise dependency floors (`h11 >= 0.16`, `starlette >= 0.40`) and pin CI actions by SHA.
- A key Jev revokes keeps local answers until one of its requests is forwarded again (at least the audit share) or `key_ttl_s` expires.

One audit run finds only part of what several runs find. The admin API, metrics and settings code were added after it and should be covered by the next run.
