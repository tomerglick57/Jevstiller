# Configuration reference

Every knob, where it lives, and its default. There are four layers:

| Layer | Object | Scope |
|---|---|---|
| Engine | `jevstiller.Config` | one task's loop: routing, training, calibration, drift |
| Task manager | `jevstiller.TaskManager(...)` | many tasks in one process: loading, admission, memory |
| Proxy | `jevstiller.server.ProxySettings` | the HTTP proxy: upstream, keys, tenancy |
| Server | `jevstiller serve`: TOML file, `JEVSTILLER_*` env vars, flags | builds the three above |

A task's target is `Task.target_agreement` (default 0.98). The disagreement budget is `β = 1 − target_agreement`.

---

## Engine: `Config`

`Config(**overrides)`. Invalid enum values raise `ValueError`. With a `TaskManager`, the config is copied per task.

### Routing and the audit channel

| Field | Default | Meaning |
|---|---|---|
| `mode` | `"auto"` | `auto`: teacher only until a student is promoted, then cascade. `teacher_only`: everything to the teacher. `cascade`: student when confident (still needs a production version). |
| `audit_rate` | `0.02` | Share of all traffic always sent to the teacher: the unbiased view of production, and the price of the guarantee. |
| `audit_rate_shadow` | `0.10` | Audit rate while a candidate is in shadow, so it's judged on fresh traffic quickly. |
| `audit_rate_elevated` | `0.10` | Audit rate while the drift monitor is suspicious, or after a teacher change in `audit` mode. |
| `importance_weighting` | `True` | Audit rows weigh more in training, so the training set tracks traffic rather than the hard slice. |
| `weight_power` | `0.5` | Weight = `(1/audit_rate) ** power`. 1.0 is exact Horvitz–Thompson; 0.5 trades a little bias for much less variance. |
| `max_weight` | `20.0` | Cap on a row's weight. |

### Calibration and the guarantee

| Field | Default | Meaning |
|---|---|---|
| `confidence` | `0.95` | 1 − δ for every Clopper–Pearson bound. |
| `calib_fraction` | `0.20` | Share of IID rows (bootstrap, audit, fallback) hashed into the calibration split. The split is fixed per text forever. |
| `fit_headroom` | `0.15` | Policies are fitted at `(1 − headroom) · β`, and the shadow judges them at the full `β`. |
| `min_calib_samples` | `500` | No policy is fitted on fewer calibration rows. |

### Training triggers and the lifecycle

| Field | Default | Meaning |
|---|---|---|
| `min_train_samples` | `1000` | Teacher-labelled training rows before the first student. |
| `min_samples_per_class` | `50` | Per class, before the first student (see `rare_classes`). |
| `rare_classes` | `"wait"` | `wait`: no first student until every class has `min_samples_per_class`. `defer`: train anyway; the policy never lets the student answer a rare class (`routing_reason="rare_class"`). |
| `min_new_samples` | `2000` | New rows since the last training before a retrain. |
| `shadow_min_samples` | `1000` | Requests a candidate runs alongside production before it's judged. |
| `training` | `"background"` | `background`: a worker thread per task runs maintenance off the request path. `inline`: maintenance runs inside `classify_batch` (deterministic; for replays and tests). `manual`: only when you call `maintain()` / `train_now()`. |
| `maintenance_interval_s` | `1.0` | Background mode: at most one maintenance pass per interval per task. |
| `train_threads` | `2` | BLAS threads per fit (0 = library default, i.e. every core). An uncapped fit takes the CPU away from serving. |

### Student and OOD gate

| Field | Default | Meaning |
|---|---|---|
| `label_target` | `"probs"` | Train on the teacher's distribution, or `hard` (argmax). |
| `student_epochs` | `2000` | Upper bound; early stopping on a 10% validation slice decides. |
| `student_patience` | `4` | Early-stopping patience (evaluations every 25 epochs). |
| `student_l2` | `1e-6` | L2 on the head (kept tiny; early stopping regularises). |
| `ood_k` | `10` | kNN size for the out-of-distribution score. |
| `ood_quantile` | `0.99` | OOD threshold = this quantile of the training reference's leave-one-out scores (never the calibration rows, which only test the policy). |
| `ood_max_ref` | `5000` | Reference embeddings kept per version, sampled stratified by class. Memory and per-request cost scale with it. |
| `max_train_samples` | `50000` | A fit reads at most this many of the most recent training rows (`0`: all). Bounds training time and the training worker's memory as a task's history grows. |
| `max_calib_samples` | `20000` | The same for calibration rows. At 20,000 rows the Clopper–Pearson bound is already within ~0.2 points of the observed rate. |

### Drift and teacher changes

| Field | Default | Meaning |
|---|---|---|
| `drift_window` | `500` | Most recent audit rows in the agreement check. |
| `drift_min_samples` | `200` | The monitor is silent below this. |
| `drift_margin` | `0.0` | Hard fallback when the upper bound on agreement < target − margin. |
| `drift_check_every` | `20` | Re-check drift after this many new audit answers. |
| `teacher_change` | `"fallback"` | When the teacher's resolved model changes (e.g. `jev-latest` moves). `fallback`: all traffic to the teacher until a student of the new model passes shadow. `audit`: keep serving, raise the audit rate, retrain, and let the drift monitor decide. |
| `teacher_change_confirm` | `20` | Consecutive answers from a new model before the lineage switches (so a gradual rollout doesn't flap). |

### Storage

| Field | Default | Meaning |
|---|---|---|
| `store_text` | `True` | `False` keeps only a hash and the embedding of each request, not its text. |
| `seed` | `0` | Seeds the audit draw, training and OOD sampling. |

---

## Task manager: `TaskManager(data_dir, teacher, encoder, config=None, *, ...)`

| Parameter | Default | Meaning |
|---|---|---|
| `data_dir` | — | Tasks live in `<data_dir>/tasks/<key>/`. |
| `teacher` | — | The default teacher; a per-call teacher can be passed to `classify`. The proxy makes its own teacher calls. |
| `encoder` | — | One encoder for all tasks. Wrap it in `BatchingEncoder` to merge concurrent calls. |
| `config` | `Config()` | Copied per task. |
| `target_agreement` | `0.98` | For new tasks. |
| `max_loaded` | `64` | Loaded engines. Idle ones are unloaded least-recently-used. |
| `max_memory_mb` | `None` | Also unload while the loaded students exceed this (heads + OOD references). |
| `admission` | `Admission()` | `Admission(min_requests=50, window_s=86400, max_tracked=100_000)`: a question becomes a task only after this many requests in the window. Earlier requests go to the teacher unrecorded (`not_admitted`). |
| `max_tasks` | `10000` | Beyond this many tasks, new questions stay pass-through (`task_limit`). |
| `max_tasks_per_tenant` | `None` | Beyond this, new tasks of that tenant stay pass-through (`tenant_task_limit`). (`jevstiller serve` defaults to 1,000.) |
| `idle_ttl_s` | `None` | Delete tasks unused this long (off by default). |
| `text_retention_s` | `None` | Securely blank stored request text older than this, in every task, about hourly. |
| `hash_key` | `None` | Key for the per-row text hash (HMAC); `jevstiller serve` passes the deployment salt. |
| `task_overrides` | `None` | `{task key: {"target_agreement": x, "mode": m}}` applied at startup; `set_target()` / `set_mode()` change them at runtime (persisted in `task.json`). |
| `train_executor` | `None` | Shared executor for training jobs. `TrainScheduler(workers=2)` is fair across tenants and prioritises by teacher-call rate. `None` trains in each task's maintenance thread. |
| `janitor_interval_s` | `5.0` | How often the janitor unloads, persists last-seen times, and cleans up. |
| `blas_threads` | `1` | Caps BLAS threads process-wide for serving (2.4× throughput at 50 tasks). `None` leaves it alone. |

`TrainScheduler(workers=2, niceness=10, max_retries=2, retry_backoff_s=30.0, pool_factory=None)`: `workers` caps concurrent trainings, and workers run at lower OS priority. Budget about `workers × Config.train_threads` cores for training.

`BatchingEncoder(inner, max_batch=256, max_wait_ms=0.0)`: merges concurrent `encode` calls into one batch without waiting when idle.

---

## `jevstiller serve` settings

One source, three layers. Precedence, lowest to highest: built-in defaults < a TOML file (`--config`, or `JEVSTILLER_CONFIG`) < `JEVSTILLER_<NAME>` environment variables < command-line flags. Unknown keys and invalid values stop startup with an error. That includes a value of the wrong type (`store_text = "false"` is a string, not a boolean), an empty secret, secret file or access-control variable (an unset `JEVSTILLER_ACCESS_TOKEN=` in a compose file), and a retention period of 0 or less: a security or privacy setting either takes effect or stops startup. `jevstiller serve` logs the effective access controls and whether request text is stored. `jevstiller config` prints the effective settings (secrets and credentials in `upstream` redacted). A complete example is `deploy/jevstiller.toml`.

```toml
[server]
port = 8080
data_dir = "/data"
log_format = "json"
admin_token_file = "/run/secrets/admin-token"

[proxy]
tenancy = "per_key"
allow_networks = ["10.0.0.0/8"]

[manager]
target_agreement = 0.98
text_retention_days = 30

[encoder]
spec = "small"

[engine]                 # any `Config` field from the table above
audit_rate = 0.03

[tasks."<task key>"]     # per-task overrides (also settable at runtime via the admin API)
target_agreement = 0.99
mode = "teacher_only"
```

Environment variables use the setting's name in upper case: `JEVSTILLER_PORT`, `JEVSTILLER_ADMIN_TOKEN_FILE`, `JEVSTILLER_STORE_TEXT=false`, …
- Booleans accept `1/0/true/false/yes/no/on/off`.
- Lists are comma-separated.
- `none` unsets an optional value, or clears a list set in the file (`JEVSTILLER_ALLOW_NETWORKS=none`). An empty value is an error for secrets, secret files, `allow_networks`, `trust_forwarded_for` and `tenants_file`.
- `[engine]`, `[tasks]` and `tenants` are file-only.

### [server]

| Key | Default | Flag | Meaning |
|---|---|---|---|
| `host` | `0.0.0.0` | `--host` | Listen address. |
| `port` | `8080` | `--port` | |
| `data_dir` | `./jevstiller-data` | `--data-dir` | Tasks, samples, models and `key-salt`. Created 0700; the server runs with umask 0077. |
| `log_level` | `info` | `--log-level` | |
| `log_format` | `text` | `--log-format` | `json`: one JSON object per line, including a per-request access log (`logger: jevstiller.access`) with source, reasons, status, latency, request id and task keys, and never keys or text. |
| `ssl_certfile`, `ssl_keyfile` | none | `--ssl-certfile`, `--ssl-keyfile` | Built-in TLS (both or neither). |
| `admin_token` / `admin_token_file` | none | `--admin-token-file` | Enables the admin API (`/jevstiller/v1/*`) and protects `/metrics`. Prefer the file or the env var: flags show in the process list. |
| `metrics_public` | `false` | | Serve `/metrics` without the admin token. |

### [proxy]

| Key | Default | Flag | Meaning |
|---|---|---|---|
| `upstream` | `https://api.typesafe.ai` | `--upstream` | Where forwarded requests go. |
| `upstream_timeout_s` | `9.0` | `--upstream-timeout` | Under the SDK's 10 s client timeout. Timeout → 504. |
| `max_upstream_inflight` | `256` | | Concurrent forwarded requests; beyond → 503 `retry-after: 1`. |
| `tenancy` | `shared` | `--tenancy` | `shared`: one tenant, every key shares tasks. `per_key`: tasks separate per API key. |
| `tenants` / `tenants_file` | `{}` | `--tenants-file` | Key hash → tenant name (hashes from `jevstiller key-hash`; must be 32 lowercase hex). Unlisted keys follow `tenancy`. |
| `key_ttl_s` | `3600` | | How long a key stays accepted after Jev last answered a systemone request with it. Finite, ≥ 0. |
| `access_token` / `access_token_file` | none | `--access-token`, `--access-token-file` | Callers must send `x-jevstiller-token`. Never forwarded. |
| `allow_networks` | `[]` (all) | `--allow-network` (repeat) | Client CIDRs allowed to use the proxy. |
| `trust_forwarded_for` | `[]` | `--trust-forwarded-for` (repeat) | Proxies whose `X-Forwarded-For` is trusted for `allow_networks`: addresses or networks without host bits (`10.0.0.5`, `10.0.0.0/24`; `10.0.0.5/24` is an error, since uvicorn would silently ignore it). Empty: the TCP peer is used. |
| `max_body_mb` | `4.0` | `--max-body-mb` | Larger bodies get 413 (declared or streamed). |
| `max_questions` | `32` | `--max-questions` | Distinct choice questions routed per request; more → forwarded unrouted. |
| `max_encoder_wait_ms` | `200` | `--max-encoder-wait-ms` | If a local answer would wait longer than this for the shared encoder (traffic beyond its capacity), forward the request to Jev instead. `0` never forwards for this reason (requests queue). |
| `price_per_mtok` | `0.042` | | For cost accounting. |

### [manager]

| Key | Default | Flag | Meaning |
|---|---|---|---|
| `target_agreement` | `0.98` | `--target-agreement` | For new tasks. |
| `max_loaded` | `64` | `--max-loaded` | Tasks kept in memory (LRU). |
| `max_memory_mb` | none | | Also unload while loaded students exceed this. |
| `admit_after` / `admit_window_s` | `50` / `86400` | `--admit-after` | Requests (one per distinct question per request) within the window before a question becomes a task. |
| `max_tasks` | `10000` | `--max-tasks` | Global task cap (`task_limit`). |
| `max_tasks_per_tenant` | `1000` | `--max-tasks-per-tenant` | (`tenant_task_limit`) |
| `idle_ttl_days` | none | | Delete tasks unused this long (> 0). |
| `text_retention_days` | none | `--text-retention-days` | Securely blank stored request text older than this (hourly; > 0). |
| `store_text` | `true` | `--no-store-text` | `false`: keep only an HMAC and the embedding of each request. `store_text = false` under `[engine]` works too: either one saying `false` wins. |
| `train_workers` | `2` | `--train-workers` | Training processes (low priority). Budget ≈ workers × `Config.train_threads` cores. |
| `blas_threads` | `1` | | BLAS threads for serving (process-wide). |

### [encoder]

| Key | Default | Flag | Meaning |
|---|---|---|---|
| `spec` | `small` | `--encoder` | `small` / `base` / `large` (bge, ONNX or torch), `hash`, `onnx:<repo>`, `torch:<model>`. The Docker image bakes in `small`. |
| `backend` | `auto` | `--backend` | `onnx` / `torch` |
| `device` | `auto` | `--device` | `cpu` / `cuda` |

### [engine] and [tasks]

`[engine]` takes any `Config` field (first section of this page) and applies to every task. `[tasks."<key>"]` sets `target_agreement` and/or `mode` for one task at startup. Changes made through the admin API are saved in the task's `task.json`, and survive restarts.

## Library: `ProxySettings` and `create_app`

For embedding the proxy in your own ASGI stack: `jevstiller.server.create_app(manager, ProxySettings(...), KeyRegistry(salt, ttl), admin_token=..., metrics_public=..., ready=..., closers=[...])`, with `manager` a `jevstiller.TaskManager`. `jevstiller.server` exports these four names (`create_app`, `ProxySettings`, `KeyRegistry`, `load_salt`); the rest of the proxy is internal. `ProxySettings` has the `[proxy]` fields above, with `tenant_map` in place of `tenants` and `max_body_bytes` in place of `max_body_mb`.
