# Configuration reference

Every knob, where it lives, and its default. There are four layers:

| Layer | Object | Scope |
|---|---|---|
| Engine | `jevstiller.Config` | one task's loop: routing, training, calibration, drift |
| Task manager | `jevstiller.TaskManager(...)` | many tasks in one process: loading, admission, memory |
| Proxy | `jevstiller.server.ProxySettings` | the HTTP proxy: upstream, keys, tenancy |
| CLI | `jevstiller serve` flags / `JEVSTILLER_*` env vars | builds the three above |

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
| `ood_quantile` | `0.99` | OOD threshold = this quantile of calibration scores. |
| `ood_max_ref` | `5000` | Reference embeddings kept per version, sampled stratified by class. Memory and per-request cost scale with it. |

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
| `max_tasks_per_tenant` | `None` | Beyond this, new tasks of that tenant stay pass-through (`tenant_task_limit`). |
| `idle_ttl_s` | `None` | Delete tasks unused this long (off by default). |
| `train_executor` | `None` | Shared executor for training jobs. `TrainScheduler(workers=2)` is fair across tenants and prioritises by teacher-call rate. `None` trains in each task's maintenance thread. |
| `janitor_interval_s` | `5.0` | How often the janitor unloads, persists last-seen times, and cleans up. |
| `blas_threads` | `1` | Caps BLAS threads process-wide for serving (2.4× throughput at 50 tasks). `None` leaves it alone. |

`TrainScheduler(workers=2, niceness=10, max_retries=2, retry_backoff_s=30.0, pool_factory=None)`: `workers` caps concurrent trainings, and workers run at lower OS priority. Budget about `workers × Config.train_threads` cores for training.

`BatchingEncoder(inner, max_batch=256, max_wait_ms=0.0)`: merges concurrent `encode` calls into one batch without waiting when idle.

---

## Proxy: `ProxySettings`

| Field | Default | Meaning |
|---|---|---|
| `upstream` | `https://api.typesafe.ai` | Where forwarded requests go. |
| `upstream_timeout_s` | `9.0` | Kept under the SDK's default 10 s client timeout. A timeout returns 504. |
| `max_upstream_inflight` | `256` | Concurrent forwarded requests. Beyond this: 503 with `retry-after: 1` (the SDK retries). |
| `tenancy` | `"shared"` | `shared`: one tenant, so every key shares trained tasks. `per_key`: tasks are separate per API key. |
| `key_ttl_s` | `3600` | A key must have been accepted by Jev within this long before the proxy answers locally for it. |
| `price_per_mtok` | `0.042` | For cost accounting of forwarded calls. |

---

## CLI: `jevstiller serve`

| Flag | Env var | Default |
|---|---|---|
| `--host` | `JEVSTILLER_HOST` | `0.0.0.0` |
| `--port` | `JEVSTILLER_PORT` | `8080` |
| `--data-dir` | `JEVSTILLER_DATA_DIR` | `./jevstiller-data` |
| `--upstream` | `JEVSTILLER_UPSTREAM` | `https://api.typesafe.ai` |
| `--upstream-timeout` | | `9.0` |
| `--encoder` | `JEVSTILLER_ENCODER` | `small` (`small`/`base`/`large`, `hash`, `onnx:<repo>`, `torch:<model>`) |
| `--backend` / `--device` | | `auto` |
| `--target-agreement` | `JEVSTILLER_TARGET_AGREEMENT` | `0.98` |
| `--tenancy` | `JEVSTILLER_TENANCY` | `shared` |
| `--admit-after` | | `50` |
| `--max-loaded` | | `64` |
| `--max-tasks-per-tenant` | | none |
| `--train-workers` | | `2` |
| `--log-level` | `JEVSTILLER_LOG_LEVEL` | `info` |

The CLI uses `Config()` defaults for the engine; a config file is planned (DEPLOYMENT_PLAN P5.1).
