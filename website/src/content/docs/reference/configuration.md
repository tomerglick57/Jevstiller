---
title: Configuration
description: Every Config field and its default.
---

The contract (`target_agreement`) lives on the `Task`. Everything about *how* the loop runs lives on `Config`:

```python
from jevstiller import Config, Jevstiller

cfg = Config(audit_rate=0.05, min_train_samples=400)
js = Jevstiller(task, teacher, data_dir="./jevstiller-data", config=cfg)
```

## Routing and the guarantee

| field | default | meaning |
|---|---|---|
| `confidence` | `0.95` | confidence level (1 − δ) for every bound |
| `audit_rate` | `0.02` | share of all traffic always sent to the teacher |
| `audit_rate_shadow` | `0.10` | audit rate while a candidate is in shadow |
| `audit_rate_elevated` | `0.10` | audit rate while drift is suspected |
| `calib_fraction` | `0.20` | share of IID samples hashed into the calibration split |
| `fit_headroom` | `0.15` | fit policies at 85% of the budget; shadow judges at 100% |
| `mode` | `"auto"` | `auto`, `teacher_only`, or `cascade` |

## When to train

| field | default | meaning |
|---|---|---|
| `min_train_samples` | `1000` | first training needs at least this many rows |
| `min_samples_per_class` | `50` | … and this many per class |
| `min_calib_samples` | `500` | never fit a policy on fewer |
| `min_new_samples` | `2000` | retrain after this many new rows |
| `shadow_min_samples` | `1000` | requests a candidate runs in shadow before it is judged |

## Out-of-distribution gate

| field | default | meaning |
|---|---|---|
| `ood_k` | `10` | neighbours in the kNN distance |
| `ood_quantile` | `0.99` | threshold as a quantile of calibration distances |

## Drift monitor

| field | default | meaning |
|---|---|---|
| `drift_window` | `500` | audit records in the rolling agreement check |
| `drift_min_samples` | `200` | the monitor stays silent below this |
| `drift_margin` | `0.0` | fall back when the upper bound on agreement < target − margin |

## Student training

| field | default | meaning |
|---|---|---|
| `label_target` | `"probs"` | train on the teacher's distribution, or `"hard"` (its top label) |
| `importance_weighting` | `True` | reweight audit rows so the training set tracks real traffic |
| `weight_power` | `0.5` | weight = (1 / audit_rate) ^ power; `1.0` is exact Horvitz–Thompson |
| `max_weight` | `20.0` | cap on any row's weight |
| `student_epochs` | `2000` | upper bound; early stopping decides |
| `student_patience` | `4` | early-stopping patience |
| `student_l2` | `1e-6` | L2 penalty |
| `seed` | `0` | training and audit sampling are seeded |

## Privacy

| field | default | meaning |
|---|---|---|
| `store_text` | `True` | `False` keeps only a hash and the embedding; no raw text is stored |

The Jev API key is read from `TYPESAFE_API_KEY`. It is never written to the task directory or logged.
