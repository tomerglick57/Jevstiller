# Changelog

## 0.1.0 — 2026-09-22

First public release.

- Core loop: route → record → train → shadow → promote → monitor → fall back, with a teacher-parity contract (`target_agreement`) and a permanent audit channel.
- Teachers: Jev via `typesafe-sdk`, plus synthetic, replay, and cached adapters.
- Encoders: hashing (no download), PyTorch (HF checkpoints, CUDA), ONNX Runtime (CPU/CUDA); `small`/`base`/`large` tiers.
- Student: numpy soft-target logistic regression with early stopping; kNN out-of-distribution gate; Clopper–Pearson threshold selection with headroom.
- Experiment runner for Banking77, CLINC150, AG News, and synthetic data.
- `store_text=False` for hash-only storage; thread-safe `classify`; `versions()` and `export()`.

Known limits: training runs synchronously inside `classify_batch` when a trigger fires; single process per task directory; live-Jev benchmark pending.
