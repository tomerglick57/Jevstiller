# Changelog

## Unreleased

- **Disk use no longer grows with every retrain.** The 24-hour soak's data directory reached 53 GB, 23 GB of it old model versions: every superseded version was kept (~8–10 MB each, 645 in the busiest task). Now only the newest `keep_versions` (default 3) versions of each finished state (superseded, rejected, rolled back) keep their files. Production, the shadow and candidates always stay. Older ones stay listed as `deleted`, and rollback goes back at most that far. Versions an earlier release kept are deleted on each task's first maintenance pass after the upgrade.
- **A retrain waits for 2,000 new Jev answers, not 2,000 new requests** (`min_new_samples`). Requests the student answered counted too, so at 97% answered locally a busy task retrained every ~2 minutes on ~60 new answers, each time storing a new version. Drift and teacher changes still retrain at once. The count is an index seek, so it costs the number of new answers.
- **Stored requests are bounded too.** Each maintenance pass deletes the rows nothing reads any more:
  - **With a Jev answer:** all but the newest `max_train_samples` (50,000) training and `max_calib_samples` (20,000) calibration rows per teacher lineage. Training and calibration never read more than those.
  - **Answered by the student alone:** all but the newest `keep_local_rows` (new setting, 10,000).
  - **Kept anyway:** a shadow's rows, until it's judged.

  Status totals (requests, who answered, teacher calls and cost) still count deleted rows. A new store hands the space back as it goes. A store from an earlier release is rebuilt once its backlog is gone. A busy task's store now levels off at ~300 MB. On the 24-hour soak's data, everything together goes from 53 GB to 4.3 GB. **After upgrading,** a busy task's backlog is deleted over its first maintenance passes, 20,000 rows at a time. Expect disk I/O for a while: the soak's 6.8 million rows took 18 minutes.
- Docs: `calib_fraction` is a draw per request (since 0.3.0), not per text; the disk sizing in docs/deploy.md was low (~2 KB per request, no version growth).
- **`rare_classes` defaults to `"defer"`.** A task whose teacher never uses one of its labels used to wait for that label forever before training its first student (the benchmark's CLINC150 run: Jev never answered `reminder_update` in 12,000 messages). Now the loop trains once the other thresholds are met and keeps forwarding any request the student would label with a class that is still below `min_samples_per_class`. Set `rare_classes = "wait"` for the old behaviour.
- **Reproduce the headline result without an API key.** `bash experiments/reproduce.sh` replays Banking77 against Jev's recorded answers, which now ship with the repository (`experiments/cache/banking77.jsonl.gz`), and prints the numbers next to the published ones. `experiments/run.py` has a `--teacher cached` mode for it, and `experiments/record_answers.py` records a complete cache for a dataset.
- **Replays are reproducible.** The per-request train/calibration split is now seeded from `Config.seed` and the task version instead of the store's file path, so the same replay gives the same result on any machine. Production behaviour is unchanged: the split is still a uniform draw per request.
- `Status.report(events=False)` leaves out the loop events; floats in event lines are rounded to four decimals.
- **The 24-hour soak is written up** (docs/benchmarks.md §Soak): 3.6 million requests, 0.012% errors, memory bounded (peak 654 MB, 308 MB at the end), silent drift at hour 12 recovered without intervention. The README's memory claim now says "bounded" rather than "flat".
- **A benchmark, runnable by anyone**, with results (docs/benchmarks.md §Benchmark): on five public tasks the loop reaches 22–80% coverage at 98.7–99.7% agreement with Jev and matches Jev's accuracy on every task; over 100 random splits the calibrated threshold broke the 2% budget once while the point-estimate rule broke it on 6–12 of 20 splits per task. `bash experiments/bench.sh` records Jev's answers for five public tasks (Banking77, CLINC150, a 20k AG News sample, TweetEval sentiment and offensive), replays each through the loop, runs `experiments/baselines.py` (four threshold rules on identical splits: soft or hard labels × Clopper-Pearson bound or point estimate) and rebuilds the tables at the top of docs/benchmarks.md with `experiments/bench.py`. `experiments/target_curve.py` adds the coverage-vs-target table (what `admin target` buys on each task). `run.py --rare-classes defer` trains when Jev never uses one of a task's labels (CLINC150 needed it). AG News is now read from CSV (the `datasets` import shadowed the local module).
- `examples/race.py`: 200 decisions one at a time, straight to Jev and through the proxy, recorded separately and replayed on one clock: 60.2 s vs 24.7 s (docs/benchmarks.md, `docs/media/race.gif`). The website now links images in docs as the files themselves.
- **The pitch no longer leans on Jev's rate limit.** On 2026-09-26 one key sustained 190 requests/s to Jev for a minute with no 429 (published limit: 1,200/min); TypeSafe says limits adjust dynamically. The README's "Why" now argues from latency (~300 ms per answer at every load, ~15 ms locally), sequential decisions, egress and vendor dependence; the measurements are in docs/benchmarks.md.
- **The README opens with the proxy in front of live Jev.** `examples/proxy_demo.py` is a service that classifies 5,000 real customer messages with the unmodified SDK; recorded through `docker run ... jevstiller` it shows answers moving from Jev (~350 ms) to the local model (~40 ms at 8 concurrent callers) after about 4,000 requests, with no errors (`docs/media/proxy.gif`).
- `examples/quickstart_synthetic.py` starts from a fresh temporary directory on every run.
- Repository: GitHub Discussions for questions and results, a pull request template, links from the issue chooser.

## 0.3.3 — 2026-09-26

- **`jevstiller admin` works next to the server without flags.** When no token or URL is given, it uses the server's own settings: the config file (`JEVSTILLER_CONFIG`), `JEVSTILLER_ADMIN_TOKEN_FILE` and the port. So `docker exec jevstiller jevstiller admin tasks` (or `docker compose exec`, or `kubectl exec`) just works.
- The Kubernetes manifest passes its config file as `JEVSTILLER_CONFIG` instead of `--config`, so `kubectl exec ... jevstiller admin` finds the same settings.
- Website: the getting-started page has install tabs (Docker, Compose, Kubernetes, pip), a "check it works" step with the expected output, and a screenshot of the status page.

## 0.3.2 — 2026-09-26

- **One install command.** `pip install jevstiller` now includes everything most people need: the proxy, the default encoder (ONNX Runtime, CPU) and the Jev adapter. On a GPU machine, `pip install "jevstiller[gpu]"` adds PyTorch, which is used automatically when CUDA is available. ONNX Runtime is required only on platforms it ships for (x86-64 and ARM64); elsewhere `jevstiller serve` says what to do.
  - The `[server]`, `[onnx]`, `[jev]`, `[torch]` and `[all]` extras still work (the first three are now empty, the others mean `[gpu]`).
  - `[gpu]` used to mean ONNX Runtime for CUDA, which can't be installed next to the CPU ONNX Runtime; it now means PyTorch.
  - The Docker image's `EXTRAS` build argument defaults to nothing; a GPU image uses `EXTRAS=gpu`.

## 0.3.1 — 2026-09-25

- **`pip install jevstiller` now includes the proxy.** Its dependencies (Starlette, uvicorn, httpx, anyio, h11) are part of the plain install. Before, `jevstiller serve` failed with `No module named 'uvicorn'` unless you had installed `jevstiller[server]`. The `server` extra still exists (empty), so existing install commands keep working.
- **For the default encoder, install `jevstiller[onnx]`.** Without it, `jevstiller serve` and `load_encoder` now say so (`pip install "jevstiller[onnx]"`, or use `--encoder hash`) instead of failing with a traceback.

## 0.3.0 — 2026-09-25

A status page, a defined public Python API, and the fixes from a third security audit, including two in the checks behind the agreement guarantee.

**Upgrading from 0.2.0:**
- **Settings:** a blank `JEVSTILLER_*` value, an empty `JEVSTILLER_CONFIG` or `--config ""`, and a list of only commas now stop startup. Use `none` to clear a value.
- **New-task quota:** each API key may create at most `max_new_tasks_per_key` (100) new tasks per `admit_window_s` (24 h). Raise it, or set `none`, if one service legitimately starts more.
- **Readiness:** `/readyz` no longer includes `data_dir_writable`; watch `jevstiller_data_dir_writable` instead.
- **Python API:**
  - code that imported internal modules must import from `jevstiller` (or use the `_`-prefixed module);
  - `Jevstiller`'s options after `data_dir` are keyword-only;
  - `train_pool` is exported from `jevstiller`.
- **Data directories** from 0.2.0 work as they are. Existing rows keep their calibration split; new rows are split per request.

- **Security audit run 3** (2026-09-25; [docs/security.md](docs/security.md)): fixes with regression tests in `tests/test_audit_run3.py`.
  - **The audit now scores what was served.** The drift monitor, the status report and the status page measured agreement by re-scoring the audit window with the current production student, which had since been trained on most of those rows. That overstated agreement (about 1.6x less disagreement with default settings) and could miss a broken budget. The audit now uses each audit row's recorded answer from the version in production.
  - **Calibration is split per request, not per text.** A text that is a large share of a task's traffic could dominate the calibration set and break the bound on the real mix. Existing rows keep their split; new rows are split by an independent draw.
  - **Stored request text is cut to 32,768 characters,** like the encoder's input. It used to be stored whole (up to the body limit), once per task in the request, including locally answered ones. The text hash still covers the whole state.
  - **A per-key quota on new tasks:** `max_new_tasks_per_key` (default 100 per `admit_window_s`), so one caller can't fill a shared tenant's task cap. Beyond it, new questions pass through (`caller_task_limit`). `TaskManager` has `max_new_tasks_per_caller` and `route(..., caller=...)`.
  - **Blank settings stop startup:** an empty `JEVSTILLER_CONFIG` or `--config ""` (it used to skip the file, access controls and all), any empty `JEVSTILLER_*` value, and a list of only commas. `none` still clears a value.
  - **Malformed requests get a 4xx, not a 500 and a traceback:** a non-ASCII Basic credential on the status page is a 401; a URL httpx can't build is a 400. Paths with a backslash, a leading `//`, or a dot, slash or backslash still percent-encoded after decoding (`%2e`, `%2f`, `%5c`) are rejected (400).
  - **A full data dir no longer fails `/readyz`.** Serving goes on when records can't be written, but failing readiness took the only replica out of the Service. The new gauge `jevstiller_data_dir_writable` reports it.
  - **Release builds are hash-pinned end to end.** The build tools come from `build-requirements.txt` (with hashes) and the project is built without build isolation. In the image, every package is installed with `--require-hashes`. `release.yml` builds and uploads in one job with nothing else in it, tests in another, and publishes only if the files' hashes match what the build produced. The 0.2.0 notes said "hash-checked"; that covered the dependencies but not the build tools.
  - **Settings errors never print a secret:** a tenants map written the wrong way round printed the raw key; a token given where its file's path belongs printed the token.
  - `docker build --build-arg PRELOAD_ENCODER=` works: `/models` is writable by the runtime user when nothing is preloaded.

- **A status page** (P5.7) at `GET /jevstiller/status`. Open it in a browser and log in with the admin token as the password (any user name). It shows the share answered locally, each task's state, local share and agreement with Jev against its target, and recent events. It is read-only and cached for 5 s, and it never loads a task. It is HTML-escaped with no scripts (the CSP forbids them), and it is off without an admin token. The JSON admin API still takes only the bearer token.
- **A defined public API** (P7.7). It is what `jevstiller`, `jevstiller.server`, `jevstiller.encoders`, `jevstiller.teachers` and `jevstiller.teachers.jev` export in `__all__`, recorded in `tests/public_api.txt` and checked by a test. Everything else is internal (docs/compatibility.md). For code that imported internals:
  - Internal modules have a leading underscore: `jevstiller.core` → `jevstiller._core`, and likewise `manager`, `store`, `task`, `training`, `scheduler`, `registry`, `calibrate`, `ood`, `student`, `settings`, `admin`, `backup`, `metrics`, `cli`, the encoder and teacher implementations, and the proxy's internals (`jevstiller.server` keeps `create_app`, `ProxySettings`, `KeyRegistry` and `load_salt`).
  - Import from `jevstiller` instead: `train_pool` (was `jevstiller.training`), `State`, `Routed`, `Routing` and `TaskInfo`.
  - `Jevstiller(task, teacher, data_dir, *, encoder=..., config=..., train_executor=..., hash_key=...)`: the options are keyword-only.
  - The engine's methods for the task manager are private: `carry`, `restore`, `training_priority`, `busy`, `footprint_bytes`.
  - The `jevstiller` command and the proxy's HTTP behaviour are unchanged.

## 0.2.0 — 2026-09-25

The drop-in Jev proxy. Point `TYPESAFE_BASE_URL` at `jevstiller serve`, keep your services' own Jev keys, and the proxy learns each repeated `choice` question from Jev's answers, then answers it locally within the agreement budget you set, with a permanent audit to Jev and automatic fallback. First release on PyPI (`pip install "jevstiller[server,onnx]"`) and as a container image (`ghcr.io/tomerglick57/jevstiller`).

**Upgrading from 0.1.0:** existing task directories and sample stores are migrated on open, and stored tasks are re-keyed automatically. `RoutingPolicy.disagreement_ub` now means the per-request bound (versions trained before keep their old numbers). Teacher failures are per item (`TeacherError`, or `errors="return"`).

- **Packaging:** published to PyPI by the release workflow (trusted publishing; the built wheel must pass the test suite first), with the README's links pointing to GitHub. The container image is published to `ghcr.io/tomerglick57/jevstiller` for linux/amd64 and linux/arm64, with build provenance and an SBOM. The image installs dependencies at the versions in `uv.lock`, hash-checked. `deploy/smoke_test.py` checks any image end to end (non-root, read-only, no capabilities, offline start, forwarding, admin API, clean stop), and CI runs it on every image build. Dependabot keeps the pinned actions current.
- **Security audit run 2** (2026-09-25; [docs/security.md](docs/security.md)): 15 findings, all fixed, with regression tests in `tests/test_audit_run2.py`. None exposed keys, other tenants' data or the admin API. Behaviour changes:
  - **Settings are stricter.** Values of the wrong type stop startup (`store_text = "false"` used to mean *true*). So do an empty secret, secret file or access-control variable (an empty `JEVSTILLER_ACCESS_TOKEN` used to switch the token off), `text_retention_days` / `idle_ttl_days` ≤ 0, a non-finite `key_ttl_s`, and `trust_forwarded_for` entries with host bits (`10.0.0.5/24`; uvicorn ignored them). `store_text = false` under `[engine]` now works (either one saying `false` wins). `JEVSTILLER_<LIST>=none` clears a list from the config file. The startup log names the access controls in effect and whether request text is stored. Credentials in `upstream` are redacted in logs and `jevstiller config`.
  - **Encoder backpressure can't latch.** One very large request (or one stalled encode) used to switch local answers off for every tenant until restart. The encoder now sees at most 32,768 characters of a text (model encoders read only 256 tokens anyway), and an idle encoder is never "overloaded". Embeddings of texts longer than that change once.
  - **Admission counts only questions Jev answered.** A caller could create tasks with questions Jev rejects. The request whose answer admits a task is recorded as its first row.
  - **Forwarding never fails with a 500 or leaks a routed task:** header bytes are forwarded raw (a UTF-8 `User-Agent` used to give a 500 and pin the task in memory until restart); any forwarding failure is a 502; `retry-after` is capped at 1 h.
  - **No more crash when a store closes under text retention** (SIGSEGV): retention and admin calls hold the task's engine; admin calls don't count as use of the task.
  - Paths with control characters or an encoded `?`/`#` are rejected (400). They could escape a path prefix in `upstream`.
  - `/readyz` no longer flaps under concurrent probes.
  - Training workers start with forkserver on every Python version (they were forked from the server before 3.14).
  - `jevstiller backup` / `restore` write owner-only files; `jevstiller admin` URL-escapes names and requires the key argument; class names are escaped in `admin status`.
  - Also: no duplicate `server`/`date` headers; `ci.yml` has read-only permissions; `--build-arg PRELOAD_ENCODER=` works; `uvicorn>=0.31`.
- **The agreement guarantee is now sound** (found by a prior-art review, 2026-09-25). The threshold rule used to bound `coverage × UB(disagreement | answered)`, treating the estimated coverage as exact. It kept the widest of ~200 thresholds that each passed at the full `δ`, which has no `δ`-level guarantee. And it chose the OOD cutoff on the same calibration rows. Now:
  - the rate of *answered and disagreeing* requests over all calibration rows is bounded directly (exact Clopper–Pearson);
  - thresholds from a fixed grid are tested strictest-first, stopping at the first failure (fixed-sequence testing, as in Learn Then Test);
  - the OOD cutoff comes from leave-one-out scores of the training reference;
  - shadow judging uses the same loss.
  - `tests/test_calibrate.py` simulates 300 calibration sets and checks that the chosen threshold breaks the budget in at most ~`δ` of them.
  - `RoutingPolicy.disagreement_ub` now means that per-request bound. Versions trained before this keep their old numbers.
- **Connection reuse:** `jevstiller serve` keeps idle connections for 75 s (it was uvicorn's 5 s, the same as httpx's reuse window, and requests in flight were dropped: 37 in 1.15M in the soak). A forwarded request whose pooled connection Jev had just closed is retried once (7 in 1.15M were 502s).
- README: **How it compares**, covering stuntd, Distil Labs, routers, caches, open Jev-compatible models and the research lineage.
- **Phase 6 testing found and fixed:**
  - **Silent drift never recovered.** When Jev changed its answers without changing its model name, a task fell back and then retrained forever on mixed old and new answers; every candidate failed shadow. A confirmed drift (`fallback`) now restarts the training data: training, calibration, shadow and audit read only rows after the event's `since_id`. A replayed silent drift is back to 81% local answers within 2,000 requests and to its old ~96% within ~20,000. The fallback survives a restart; `mode auto` ends it by hand (`fallback_cleared` event). After a teacher change or a drift, the next student waits for the full readiness thresholds on the new data.
  - **Memory grew without bound when tasks load and unload often** (more active tasks than `max_loaded`): 555 MB after 4 minutes and accelerating. A reference cycle (engine → training executor → the engine's priority method) kept every unloaded engine and its arrays until a full garbage collection. The scheduler and training callbacks now hold engines weakly; an unloaded engine is freed at once (tested with the collector off). In the final 40-minute soak (~18 reloads a second and a drift) memory ended at 182 MB.
  - **`max_loaded` didn't hold under a stream of loads.** Only the janitor enforced it, one unload at a time, and it skipped every engine with a maintenance pass merely *requested*, which under steady traffic is nearly all of them. `benchmarks/manager.py` ended up with 505–666 tasks loaded against `max_loaded = 50` (2.7 GB). Now a load that pushes past the cap unloads the least recently used idle engine itself, and only a *running* pass or a pending training job keeps an engine loaded.
  - **glibc heap fragmentation under task churn:** every load starts threads and SQLite connections, spread over up to 8 arenas per core. Resident memory grew ~47 KB per load/unload with Python's own allocations flat. The manager now limits glibc to 2 arenas (unless `MALLOC_ARENA_MAX` is set): 7,365 load/unload cycles held 234 → 300 MB, with serving latency unchanged.
  - **Progress was lost on every reload:** a shadow's sample count restarted (so an often-unloaded task never got its shadow judged), the "rows since last training" counter restarted (a retrain on every reload once a task had `min_new_samples` rows), and the drift-check counter restarted. Versions now record `trained_to_id` and `shadow_from_id`, and the manager carries the in-memory counters across unloads.
  - **Training workers outlived a `kill -9` of the server**, with the forkserver and resource tracker: they never see their job queue close. Each worker now exits within a second of the server's death.
  - A task without a student for its current data (new, or after a teacher change or drift) whose candidate failed retrained every 100 rows. It now waits for 25% more data (at most `min_new_samples`).
  - **Beyond the encoder's capacity, latency grew without bound** (the load test: 15 s p99 at 256 callers with bge-small on CPU, and throughput halved). Requests that would wait longer than `max_encoder_wait_ms` (200) for the shared encoder are now forwarded to Jev unrouted (`overloaded`); `jevstiller_encoder_wait_seconds` shows the backlog. Also, requests that are only forwarded (questions not admitted as tasks) are no longer encoded at all.
  - **Forwarding collapsed beyond ~100 concurrent requests** (~75 req/s at 256 callers). The cause was the single httpx client to Jev: its pool's cost grows with connections × waiting requests. Upstream connections are now split over clients of 16 (`UPSTREAM_POOL`), with each request going to the least busy one: 614 req/s at 256 callers. The same limit in the load generator is why `benchmarks/load.py` spreads its connections over processes.
  - A failure while routing a request from a not-yet-verified key could raise after Jev had already answered; it now forwards Jev's answer unrecorded.
  - At the default `info` level, httpx logged a line per forwarded request next to the access log; it now logs warnings only.
  - **A fit read a task's whole history.** Training time and the training worker's memory grew without bound with a busy task's age (a task with 100k teacher answers a day would hold ~4.6 GB after a month). Fits now use the most recent `max_train_samples` (50,000) and `max_calib_samples` (20,000) rows.
  - A graceful shutdown during a training job left an empty staging directory (removed at the next start anyway); it is now removed.
  - A full disk logged a stack trace per failed batch. Now one line per store per minute, and the new counter `jevstiller_store_dropped_records_total`. Serving continues either way.
- **Tests:** `tests/test_chaos.py` (Jev down, Jev slow, a 429 storm, a full disk, and `kill -9` three times during training with restarts), `tests/test_concurrency_many.py` (16 threads over 12 tasks that unload and reload under load; 300 concurrent async clients through the proxy; every row recorded exactly once), and the drift and reload regressions. **Benchmarks:** `benchmarks/soak.py` (steady traffic through a real server with a silent drift at the midpoint, RSS and local share over time), `benchmarks/load.py` (forward-only, local and mixed traffic at fixed concurrency), `experiments/live_proxy.py` (cold start to local answers through the proxy against live Jev).
- Concurrency: the engine no longer holds a lock across encoding, inference, the teacher call, or the store. 32 callers with a 50 ms teacher went from 19 to ~500 req/s (`benchmarks/concurrency.py`).
- Training moved off the request path: `Config.training` = `background` (default, a worker thread per task, paced by `maintenance_interval_s`), `inline` (old behaviour; used by tests and experiment replays), or `manual` (call `maintain()` / `train_now()`). New `drain()`; `close()` stops the worker.
- `train_executor=`: run training in any `concurrent.futures.Executor`, e.g. a shared `ProcessPoolExecutor`. The job (`training.run_fit_job`) reads the samples from a read-only store connection, fits, scores production, and writes the bundle itself; the registry `adopt()`s the finished directory. Only small objects cross the process boundary.
- `training.train_pool(workers=2, niceness=10)`: a shared, low-priority process pool for `train_executor`. With 5 tasks training at once, serving p99 rose ~12% (vs ~12x with an uncapped 5-worker pool), and the fits finished ~2.5x faster than in threads.
- `Config.train_threads` (default 2) caps BLAS threads per fit via `threadpoolctl` (new dependency). Uncapped, one fit takes every core and slows serving whichever process it runs in.
- Sample store: write-behind batching on one writer thread, per-thread SQLite connections; reads see every earlier insert. `flush()`.
- Registry: atomic, fsynced writes; crash leftovers are cleaned up; unknown versions/states raise `ValueError`.
- Teacher failures are per item: adapters may return an `Exception` per text; `classify_batch` raises `TeacherError` (with the partial results) or, with `errors="return"`, returns `Result(label=None, error=...)`. Failed items are not recorded. `Status.teacher_errors`.
- `JevTeacher`: `timeout=` (default 10 s), per-item errors, token usage from `response.usage`, tolerant of a missing request id.
- Tasks: `instructions` and class descriptions may be any JSON value (text, object, array, or `None` for a name-only class), as in Jev's `criteria`; 2–255 classes; labels validated. Task versions of text tasks are unchanged from 0.1.0.
- A trained student stores its label order and is reordered on load. The task version ignores class order, so a task declared with its classes in another order used to silently misread a saved model.
- States: `classify` / `classify_batch` / `evaluate` take text or a JSON object/array. Objects are encoded and stored as canonical JSON (`state_type` column, added to existing stores on open); the teacher receives the object unchanged.
- Teacher lineage: `TeacherOutput.model` names the model that answered (`JevTeacher`: `jev:<resolved model>`, e.g. `jev-latest` -> `jev:jev-1.13.0`). After `teacher_change_confirm` (20) answers in a row from a new model, the loop starts a new lineage: training, calibration, shadow and audit use only that model's answers; `teacher_change="fallback"` (default) sends everything to the teacher until a new student passes shadow, `"audit"` keeps serving with a raised audit rate. Bundles record `teacher_model`; `Status.teacher_model`; `teacher_changed` event.
- Rare classes: `status()` / the report say what the first student is waiting for (samples, and which classes are below `min_samples_per_class`). `Config.rare_classes="defer"` trains without waiting; the policy's `deferred_labels` are never answered by the student (`routing_reason="rare_class"`), and calibration accounts for that.
- **Security audit** (2026-09-24, run 1 against `65fe64b`; details in docs/security.md). Twelve findings fixed, each with a regression test in `tests/test_audit_fixes.py`:
  - A key is accepted only after a parsed `/v1/systemone` answer, and requests with an unaccepted key are forwarded first and recorded afterwards.
  - `/healthz`, `/readyz`, `/metrics` and `/jevstiller/*` are local and never forwarded.
  - The body limit is enforced while streaming; one `Authorization` header only; no dot segments; `X-Forwarded-For` only from `trust_forwarded_for`.
  - One routing per distinct question, at most `max_questions` (32) per request, and the state encoded once per request. Task caps (`max_tasks`, per-tenant default 1,000).
  - `secure_delete` plus a WAL truncate for retention, 0700/0600 files, and an HMAC text hash.
  - Task keys use the full SHA-256 of the question (`Task.fingerprint`); stored tasks are re-keyed automatically on start.
  - No cookie jar; strict configuration; `key-hash` never creates a salt.
- **Operations:**
  - `jevstiller serve --config` reads TOML + `JEVSTILLER_*` + flags (`jevstiller/settings.py`, validated).
  - Admin API `/jevstiller/v1/*` and `jevstiller admin` (tasks, status, mode, target, train, promote, rollback, delete, delete-tenant, stats; mode/target persisted per task).
  - Prometheus `/metrics`, `/readyz`, JSON access logs, `jevstiller backup` / `restore`, `jevstiller config`.
  - Dockerfile (non-root, encoder baked in, runs offline and read-only), docker-compose, Kubernetes manifest.
  - `docs/deploy.md`, `docs/operations.md`, `docs/security.md`.
- **Validated against live Jev** (2026-09-24, `jev-1.13.0`; re-run with the corrected calibration on 2026-09-25: 70.7% at 99.45%): Banking77 replay, 70.6% held-out coverage at 99.40% agreement (target 98%), accuracy preserved (78.7% vs Jev's 78.55%). `experiments/jev_profile.py` measures Jev latency and records wire fixtures (`tests/fixtures/jev/`); `tests/test_live.py` has opt-in live tests (`JEVSTILLER_LIVE=1 pytest -m live`) and replays the recorded responses through the proxy in every run.
- **Docs:** DESIGN.md v3 (what was built, what the live run changed, what is still roadmap), `docs/proxy.md`, `docs/configuration.md`, `docs/benchmarks.md`, `benchmarks/README.md`; CONTRIBUTING and SECURITY updated.
- **Drop-in Jev proxy** (`jevstiller serve`, `server.create_app`; extra `server`). Speaks `POST /v1/systemone` and passes everything else through. Each `choice` question is routed to its task (tenant, instructions, criteria, requested model). If every question can be answered locally and the caller's key has been accepted by Jev within `key_ttl_s`, the proxy answers in Jev's exact shape: resolved `model`, zero `usage`, a `jvs_` `x-typesafe-request-id`, and `x-jevstiller-source` / `x-jevstiller-detail` headers. Otherwise the whole request goes to Jev with the caller's key, the response is returned unchanged, and the choice answers are recorded as training rows. The proxy never stores keys: it keeps salted hashes (`key-salt`, created 0600), and a 401/403 revokes a key's local answers at once. It remembers per-key 429 `retry-after` and answers 429 itself meanwhile. Upstream timeout is 9 s, then 504; unreachable gives 502; over `max_upstream_inflight` gives 503. Tenancy is `shared` or `per_key`. Any internal error falls back to forwarding. Tested with the unmodified `typesafe-sdk` (sync and async) over real HTTP; adds ~3 ms to a forwarded request.
- `Jevstiller.route()` / `defer()` / `complete()`: `classify_batch` split so the caller can make the teacher call itself; `TaskManager.route()` / `complete()`. The requested teacher model is part of a task's key.
- `TaskManager`: many tasks in one process. A task is found by `hash(tenant, question type, Task.version)`, so services asking the same question share one student and any wording change is a new task. Engines load on demand (~4 ms from disk) and are unloaded least-recently-used when `max_loaded` / `max_memory_mb` is exceeded; engines with requests in flight or maintenance running are never unloaded. `Admission` (default: 50 requests within 24 h) keeps one-off questions from creating tasks; `max_tasks_per_tenant`; `idle_ttl_s` deletes unused tasks; `delete()`. One SQLite store per task under `<data_dir>/tasks/<key>/`.
- `TrainScheduler`: one executor for every task's training. Round-robin across tenants, highest rate of teacher calls first within a tenant, retries with backoff, broken process pools replaced, queued jobs cancellable. With it (or any executor), training is asynchronous: maintenance keeps judging shadows and checking drift while a job waits for a worker. `Jevstiller.training_priority()`, `busy()`, `footprint_bytes()`; `classify_batch(..., teacher=)` for a per-call teacher (the caller's own key).
- `BatchingEncoder`: one encoder shared by every task and thread; concurrent calls are merged into batches with no added latency when idle.
- OOD reference capped at `Config.ood_max_ref` (default 5,000, stratified by class; was 50,000 at random). Held-out coverage and agreement unchanged on Banking77 (even at 1,000) and CLINC150; memory and per-request cost drop up to 10x on large tasks.
- Sample stores record a schema version and skip setup when current (reopening a store: ~0.4 ms).
- `__version__` comes from package metadata; `Config` and `set_mode` validate with `ValueError`; CI on Python 3.10–3.14.

## 0.1.0 — 2026-09-22

First public release.

- Core loop: route → record → train → shadow → promote → monitor → fall back, with a teacher-parity contract (`target_agreement`) and a permanent audit channel.
- Teachers: Jev via `typesafe-sdk`, plus synthetic, replay, and cached adapters.
- Encoders: hashing (no download), PyTorch (HF checkpoints, CUDA), ONNX Runtime (CPU/CUDA); `small`/`base`/`large` tiers.
- Student: numpy soft-target logistic regression with early stopping; kNN out-of-distribution gate; Clopper–Pearson threshold selection with headroom.
- Experiment runner for Banking77, CLINC150, AG News, and synthetic data.
- `store_text=False` for hash-only storage; thread-safe `classify`; `versions()` and `export()`.

Known limits: training runs synchronously inside `classify_batch` when a trigger fires; single process per task directory; live-Jev benchmark pending.
