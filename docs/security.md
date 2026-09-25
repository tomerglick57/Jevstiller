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
- Back up the data directory as you would a database of your traffic (`jevstiller backup`). Backups and restored data directories get the same modes (files 0600, directories 0700). Text retention doesn't reach backups: rotate them within the retention period.

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

An empty token, token file or access-control variable stops startup rather than switching the control off, and the startup log names the controls in effect.

Always enforced, for every path:
- a body limit (`max_body_mb`, 4 MiB), checked on the declared length and while streaming;
- no `.`/`..` path segments, and no control characters or encoded `?`/`#` in the path; forwarded paths must stay under the `upstream` URL's path;
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

Dependency floors were raised (`h11 >= 0.16`, `starlette >= 1.3.1`) and CI actions pinned by SHA in `40bced8`.

## Security audit, 2026-09-25 (run 2)

A second run against commit `a8b75d6` weighted toward what run 1 did not cover: the admin API, `/metrics`, settings, backup/restore, the CLI, the deployment files, and the proxy changes since run 1 (encoder backpressure, upstream pools and retry, keep-alive, task unloading, drift restarts, training caps). Eight reviewers, adversarial validation, and independent verification of every finding, all against local mocks. Fifteen findings were confirmed, and all are fixed with regression tests in `tests/test_audit_run2.py`.

No finding lets a caller read another tenant's data, obtain keys, or reach the admin API. Keys, tenant isolation, admin authentication and request framing held under every probe.

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | Medium | One large request (or one stalled encode) latched encoder backpressure on: every request was forwarded unrouted, for every tenant, until restart | The encoder sees at most 32k characters of a text; an idle encoder is never "overloaded", so the estimate recovers |
| 2 | Medium | A non-ASCII header byte made the proxy answer 500 and never release the tasks it routed: the loaded-engine cap stopped holding and the tasks could not be deleted until restart | Raw header bytes are forwarded; any forwarding failure is a 502; routed engines are released whatever happens (shielded from cancellation); `retry-after` must be finite and is capped at 1 h |
| 3 | Medium | Closing a task's store while text retention (or an admin call) was using it crashed the process (SIGSEGV) | Retention and admin calls hold the engine like requests do; a store closes only after writes in progress |
| 4 | Medium | Privacy settings were silently ignored: `store_text = false` under `[engine]`, quoted booleans (`"false"` is true), `text_retention_days = 0` | Every value is type-checked; either `store_text = false` wins; retention must be > 0; the startup log says whether text is stored |
| 5 | Low | Run 1's dot-segment check was bypassable with a tab, CR or encoded `?`/`#` (e.g. `/.%09./x`), escaping a path prefix in `upstream` | Control characters and encoded `?`/`#` are rejected; the forwarded path must stay under the upstream path |
| 6 | Low | Concurrent `/readyz` calls (no token needed) raced on one probe file and made readiness flap | A unique probe file per check, at most once a second |
| 7 | Low | Before Python 3.14 (and in the Docker image), training workers were forked from the live server and held its sockets | Workers start with forkserver (spawn where there is none) |
| 8 | Low | An empty token file or environment variable switched caller access control off, or wiped the file's allowlist | Empty values stop startup; the effective controls are logged |
| 9 | Low | `key_ttl_s = inf` made every made-up key count as accepted | An unknown key is never accepted; `key_ttl_s` must be finite |
| 10 | Low | `trust_forwarded_for` entries with host bits passed validation but were ignored by uvicorn, so `allow_networks` checked the load balancer | Validated strictly; uvicorn floor raised to 0.31 (CIDR support) |
| 11 | Low | The tenants-file error printed the start of a raw key pasted in place of its hash; upstream credentials were logged and printed | Errors name the tenant; credentials in `upstream` are redacted |
| 12 | Low | `backup` / `restore` ignored the 0600/0700 policy | Owner-only modes for backups and restored data dirs |
| 13 | Low | Caller-chosen class names put terminal escape sequences into `jevstiller admin status` | Class names and event values are shown with control characters escaped |
| 14 | Low | A caller could create tasks with questions Jev rejects, filling the tenant's task cap at no cost | Only questions Jev answered count towards admission |
| 15 | Info | `jevstiller admin delete-tenant 'a#b'` deleted tenant `a` | Names are URL-escaped; the argument is required |

Also fixed: duplicate `server`/`date` response headers; `ci.yml` runs with read-only permissions; `docker build --build-arg PRELOAD_ENCODER=` no longer breaks startup.

**Open hardening items:**
- No Host-header allowlist (DNS rebinding reaches an unauthenticated proxy from a browser; the access token stops it).
- A key Jev revokes keeps local answers until one of its requests is forwarded again (at least the audit share) or `key_ttl_s` expires.
- The one retry of a forwarded request can re-send a POST Jev already read (billed to the caller's own key).
- No per-key fairness for upstream connections or engine loads; no rate limit on admin-token failures.
- The Kubernetes manifest has no NetworkPolicy, and docker-compose publishes on all interfaces: restrict both to your network.
- If Jev ever accepted a second credential (a query `api_key`, an `x-api-key` header), a 2xx earned with it would accept the bearer too; today Jev and its SDK use only the bearer.

One audit run finds only part of what several runs find. Run 2's availability and key-handling reviews were cut short, so a third run should weight toward availability and resource limits.
