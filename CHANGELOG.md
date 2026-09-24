# Changelog

## Unreleased

- Concurrency: the engine no longer holds a lock across encoding, inference, the teacher call, or the store. 32 callers with a 50 ms teacher went from 19 to ~500 req/s (`benchmarks/concurrency.py`).
- Training moved off the request path: `Config.training` = `background` (default, a worker thread per task, paced by `maintenance_interval_s`), `inline` (old behaviour; used by tests and experiment replays), or `manual` (call `maintain()` / `train_now()`). New `drain()`; `close()` stops the worker.
- `train_executor=`: run the fit in any `concurrent.futures.Executor`, e.g. a shared `ProcessPoolExecutor`.
- Sample store: write-behind batching on one writer thread, per-thread SQLite connections; reads see every earlier insert. `flush()`.
- Registry: atomic, fsynced writes; crash leftovers are cleaned up; unknown versions/states raise `ValueError`.
- Teacher failures are per item: adapters may return an `Exception` per text; `classify_batch` raises `TeacherError` (with the partial results) or, with `errors="return"`, returns `Result(label=None, error=...)`. Failed items are not recorded. `Status.teacher_errors`.
- `JevTeacher`: `timeout=` (default 10 s), per-item errors, token usage from `response.usage`, tolerant of a missing request id.
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
