# Changelog

## Unreleased

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
- **Validated against live Jev** (2026-09-24, `jev-1.13.0`): Banking77 replay, 70.6% held-out coverage at 99.40% agreement (target 98%), accuracy preserved (78.7% vs Jev's 78.55%). `experiments/jev_profile.py` measures Jev latency and records wire fixtures (`tests/fixtures/jev/`); `tests/test_live.py` has opt-in live tests (`JEVSTILLER_LIVE=1 pytest -m live`) and replays the recorded responses through the proxy in every run.
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
