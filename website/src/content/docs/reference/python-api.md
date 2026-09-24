---
title: Python API
description: Task, Jevstiller, results, status, teachers, and encoders.
---

Everything below is importable from `jevstiller` unless noted. This page documents 0.1.0.

## `Task`

```python
Task(name: str, instructions: str, classes: dict[str, str] | list[str], target_agreement: float = 0.98)
```

What is being classified. `classes` maps each label to a description, and the descriptions are sent to Jev
verbatim. A plain list of labels is allowed (empty descriptions). At least two classes; `target_agreement` in
`[0.5, 1)`.

- `task.version`: a hash of `instructions` and `classes`. Changing either starts a new lineage.
- `task.budget`: `1 − target_agreement`.

## `Jevstiller`

```python
Jevstiller(task, teacher, data_dir, encoder=None, config=None)
```

- `teacher`: anything implementing the [`Teacher`](#teachers) protocol, usually `JevTeacher()`.
- `data_dir`: everything for the task lives in `<data_dir>/<task.name>/` (SQLite sample store and versions).
  Copy the directory and the task moves with it.
- `encoder`: defaults to `HashEncoder`, which needs no download but is weak. Pass `load_encoder("base")` for
  real use.
- `config`: a [`Config`](/reference/configuration/).

### Classifying

| method | returns |
|---|---|
| `classify(text)` | `Result` |
| `classify_batch(texts)` | `list[Result]` |

`Result` fields:

| field | |
|---|---|
| `label` | the answer |
| `probs` | distribution over all classes |
| `confidence` | Jev's own confidence score, or the student's top probability |
| `source` | `"teacher"` or `"student:vN"` |
| `routing_reason` | see [routing reasons](/concepts/how-it-works/#routing-reasons) |
| `latency_ms` | |

### Status and lifecycle

| method | does |
|---|---|
| `status()` | a `Status` with counts, shares, audit agreement and its bounds, cost, policy, and recent events. `status().report()` gives the text report. |
| `versions()` | every student version and its state: `candidate`, `shadow`, `production`, `superseded`, `rejected`, `rolled_back` |
| `set_mode(mode)` | `"auto"` (default), `"teacher_only"`, or `"cascade"` |
| `train_now()` | train a candidate immediately; returns a `TrainReport` |
| `promote(version)` | make a version production by hand |
| `rollback()` | go back to the previous production version |
| `export(path, version=None)` | copy a version (head, OOD reference, policy, task and encoder identity) into a standalone directory |
| `evaluate(texts, teacher_labels, version=None)` | offline check of a version's policy: coverage, selective disagreement, system agreement |
| `close()` | close the store |

## Teachers

A teacher is any object with:

```python
def classify(self, texts: list[str], task: Task) -> list[TeacherOutput]: ...
```

`TeacherOutput` carries the label, the full probability distribution, confidence, token usage, cost, latency,
and request id.

| teacher | use |
|---|---|
| `jevstiller.teachers.jev.JevTeacher` | real Jev via `typesafe-sdk`. Reads `TYPESAFE_API_KEY`. Pins `model="jev-1.13.0"` and rate-limits itself to 1,100 requests a minute. |
| `SyntheticTeacher` | deterministic fake for tests and demos; pair it with `SyntheticWorld` |
| `ReplayTeacher` | serves recorded answers, for offline experiments |
| `CachedTeacher` | wraps a live teacher and saves every answer to a JSONL file, so re-runs are free |

## Encoders

```python
load_encoder(spec="base", backend="auto", device="auto")
```

| `spec` | encoder |
|---|---|
| `"small"`, `"base"`, `"large"` | `bge-*-en-v1.5`. `backend="auto"` picks PyTorch when CUDA is available, ONNX otherwise. |
| `"hash"` or `"hash:<dim>"` | hashing encoder, no download |
| `"torch:<hf model>"` | any Hugging Face checkpoint |
| `"onnx:<hf repo>"` | any ONNX export on the Hub |

A custom encoder needs an `id`, a `dim`, and `encode(texts) -> np.ndarray` returning L2-normalised float32 rows.
