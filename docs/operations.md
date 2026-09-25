# Operating Jevstiller

## Is it healthy?

| Check | Healthy |
|---|---|
| `GET /healthz` | `{"ok": true}` (the process answers) |
| `GET /readyz` | 200 `{"ready": true, "checks": {"manager", "data_dir_writable", "encoder"}}`; 503 names the failing check |
| `jevstiller admin stats` | tasks, loaded, training queue, request counters |
| `/jevstiller/status` | the status page, in a browser (below) |
| `/metrics` | see below |

## The status page

Open `http://<proxy>:8080/jevstiller/status` in a browser. Log in with any user name and the admin token as the password. The page shows:
- how much is answered locally since start;
- each task (the 200 most recently used): its tenant and question, what it's doing (learning, answering locally, a candidate in shadow, fallback to Jev), its local share, and its agreement with Jev on the audit channel with the interval and `OK` / `inconclusive` / `BROKEN`;
- recent events (promotions, drift, teacher changes).

Details:
- **Read-only.** Change things with `jevstiller admin`.
- **Cheap:** it refreshes every 30 s, is drawn at most every 5 s, and reads only tasks that are already loaded. Viewing it never loads a task or counts as using it.
- **Off (404) without an admin token.**
- **With `access_token` set,** every request needs `x-jevstiller-token`, and a browser can't add it. Open the page through a reverse proxy that adds the header, or from an allowed network without the access token.
- **Safe to show caller data:** the questions, tenant names and class names on it come from callers. They are HTML-escaped, and the page runs no scripts, with a Content-Security-Policy that forbids them.

## The admin API and CLI

Enable it with `admin_token_file` (or `JEVSTILLER_ADMIN_TOKEN`). The CLI reads the token from `--token-file`, `--token` or `JEVSTILLER_ADMIN_TOKEN`, and the URL from `--url` or `JEVSTILLER_ADMIN_URL`.

```bash
jevstiller admin tasks                       # every task: key, tenant, model, classes, target, mode, last seen
jevstiller admin tasks <tenant>              # one tenant's tasks
jevstiller admin status <key>                # the status report (below); --json for the raw data
jevstiller admin versions <key>              # student versions and their states
jevstiller admin target <key> 0.99           # change the target agreement (persisted)
jevstiller admin mode <key> teacher_only     # pin a mode: auto | teacher_only | cascade (persisted); "mode <key>" with no value unpins
jevstiller admin train <key>                 # train a candidate now (waits)
jevstiller admin promote <key> student:v3    # force a version into production
jevstiller admin rollback <key>              # back to the previous production version
jevstiller admin delete <key>                # delete the task and all its data
jevstiller admin delete-tenant <tenant>      # delete every task of a tenant
jevstiller admin stats
```

The same endpoints over HTTP are listed in `jevstiller/_admin.py` (`/jevstiller/v1/...`, `Authorization: Bearer <admin token>`).

### Reading a status report

An example:

```text
Task: 285bb040f956ee1d12ff   version 5f6f15d2613e   mode: cascade   audit rate 2%
Production: student:v2   Shadow: -
Requests: 4,000   student 49.3%   teacher 50.7%
Channels: audit 60   bootstrap 1,600   deferred 360   student 1,980
Teacher calls: 2,020   avoided: 1,980   spent $0.0085   avoided $0.0083
Agreement with teacher (audit, n=60): 100.00% [95.13%, 100.00%]   target 95%   inconclusive (need more audit samples)
Labelled: train 1,600   calib 400   teacher jev:jev-1.13.0
Policy: conf>=0.52 ...
```

- **mode:** `teacher_only` before the first student, or after a fallback; `cascade` when a student answers.
- **Agreement:** measured on the audit channel, with a 95% interval. `OK` means the lower bound is above the target. `inconclusive` means there aren't enough audit samples yet. `BROKEN` means the upper bound is below the target, and the task falls back to Jev by itself.
- **teacher:** the Jev version the task learns from (its lineage). A change is logged as `teacher_changed`.
- **Waiting for a first student:** the samples still missing, and any rare classes that block training.

Agreement is with Jev, not accuracy.

## Metrics (`GET /metrics`, Prometheus)

| Metric | What |
|---|---|
| `jevstiller_requests_total{route, source}` | `route` = systemone / passthrough; `source` = local / upstream |
| `jevstiller_request_duration_seconds{source}` | Latency of systemone requests, local vs forwarded |
| `jevstiller_questions_total{outcome}` | Questions answered locally, or why they went to Jev (`co_deferred`, `key_unverified`, `not_admitted`, `unsupported`, …) |
| `jevstiller_upstream_responses_total{status}` | `2xx` / `4xx` / `5xx` / `timeout` / `unreachable` / `backoff` / `overload` |
| `jevstiller_upstream_duration_seconds` | Jev latency as seen by the proxy |
| `jevstiller_rejected_total{reason}` | Refused by the proxy: `network`, `token`, `duplicate_auth`, `path`, `body` |
| `jevstiller_tasks`, `jevstiller_tasks_loaded`, `jevstiller_student_memory_bytes` | Task counts and memory |
| `jevstiller_training_jobs{state}` | `running` / `queued` |
| `jevstiller_verified_keys` | API keys currently accepted |
| `jevstiller_task_teacher_calls_per_minute{task, production, mode}` | Per loaded task: how much training it would save |
| `jevstiller_encoder_wait_seconds` | How long a request would now wait for the shared encoder. Above `max_encoder_wait_ms`, requests are forwarded (`questions_total{outcome="overloaded"}`). |
| `jevstiller_store_dropped_records_total` | Records the sample stores could not write (disk full, I/O errors). Serving goes on; those requests just don't train anything. |

Useful alerts:
- **Local share:** `rate(requests_total{source="local"}) / rate(requests_total)` dropping suddenly usually means a fallback. Check the tasks' status for `fallback` or `teacher_changed` events.
- **Upstream errors:** `upstream_responses_total{status=~"5xx|timeout|unreachable"}` rising means Jev is struggling. Callers see Jev's errors for forwarded requests; local answers continue.
- **Queued training:** `training_jobs{state="queued"}` growing for hours means training can't keep up. Raise `train_workers` or cores.
- **Refusals:** `rejected_total` rising means misconfigured callers (token, network) or abuse.
- **Readiness:** `/readyz` failing `data_dir_writable` usually means the disk is full.
- **Dropped records:** any increase in `store_dropped_records_total` means the disk is full or failing. Requests are still answered, but nothing new is learned, and a task's audit statistics go stale.
- **Loads:** `jevstiller admin stats` shows `loads`. If it climbs by more than a few per minute, there are more active tasks than `max_loaded`: raise it.

## Logs

With `log_format = "json"` there is one JSON object per line. Each `/v1/systemone` request produces an access-log line (`logger: jevstiller.access`) with these fields: `source`, `status`, `request_id`, `latency_ms`, `questions`, `tenant`, `tasks` (keys), `reasons` (counts). The line never contains an API key or request text.

Events worth watching (logger `jevstiller`):

| Event | Meaning |
|---|---|
| `task … admitted` | A question became a task |
| `teacher model changed` | Jev's resolved version moved; the task falls back or raises auditing |
| `training failed` | Retried later |
| `training pool broke` | A worker died (e.g. out of memory); the pool is replaced |
| `sample store … dropped N records` | A write failed (disk full); logged at most once a minute per task |
| `re-keyed task` | Migration on start |

## Data

- **Backup:** `jevstiller backup --data-dir /data --out /backups/<date>`. It is consistent while serving: SQLite's online backup, and the version files copied after the registry index. It includes `key-salt`.
- **Restore:** stop the server, then `jevstiller restore --from /backups/<date> --data-dir /data`. The data dir must be empty, or use `--force`.
- **Delete a task or tenant:** `jevstiller admin delete <key>`, `jevstiller admin delete-tenant <tenant>`. This removes its directory: samples, models and `task.json`.
- **Stop storing text:** `store_text = false` for new rows. Blank existing text with `text_retention_days` (applied hourly, securely deleted).
- **Lost `key-salt`:** a new one is created on start. Every key hash changes, so update the tenants map, and callers are re-verified by Jev on their next request. Tasks and models are unaffected.

## Common situations

| Symptom | Likely cause | What to do |
|---|---|---|
| Nothing is answered locally | Tasks not admitted yet (`admit_after`), the first student still collecting (`status` shows what it waits for), or the caller's key not yet accepted | `jevstiller admin tasks` / `status <key>`; wait, or lower `admit_after` / `min_train_samples` in `[engine]` |
| A task stopped answering locally | Fallback: audit agreement confidently below target, or Jev's model changed | `status <key>` events; it retrains by itself on answers from the fallback on (`since_id`), usually within a few thousand requests. `mode <key> teacher_only` to hold it while you investigate. `mode <key> auto` ends a fallback by hand (serving the old student again). |
| "classes below N … blocking" | A rare class | Wait, or set `rare_classes = "defer"` in `[engine]` |
| Callers get 401 from the proxy | `access_token` set and the caller doesn't send `x-jevstiller-token` | Add the header in the SDK: `TypeSafeClient(headers={"x-jevstiller-token": ...})` |
| Callers get 403 | Client network not in `allow_networks` | Add the CIDR, or `trust_forwarded_for` for a reverse proxy |
| Callers get 413 | Body larger than `max_body_mb` | Raise it (Jev's own limit is ~64k tokens) |
| Callers get 429 from the proxy | Jev rate-limited that key; the proxy waits out `retry-after` | Nothing; it clears by itself |
| Memory grows | Many loaded tasks | Lower `max_loaded` or set `max_memory_mb` |
| Local share drops at peak times; `questions_total{outcome="overloaded"}` rising | Traffic beyond the encoder's capacity; the excess is forwarded to Jev | A GPU, more cores, or a smaller encoder. Forwarding is the safe overflow: answers stay correct, only Jev usage rises. |
| `loads` climbing fast, local share low | More active tasks than `max_loaded`: tasks keep reloading | Raise `max_loaded` (a few MB per task) |
| Disk full | Samples can't be written | Serving continues (local and forwarded). Free space; `store_dropped_records_total` stops rising and learning resumes by itself. |
| The server was killed (`kill -9`, OOM) | — | Restart it. SQLite recovers its journal, half-built versions are discarded, and training workers of the dead process exit by themselves within seconds. |
