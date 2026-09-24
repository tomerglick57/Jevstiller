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
