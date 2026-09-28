---
title: Python API
description: Task, Jevstiller, TaskManager, results and status, teachers, encoders and training executors.
---

Everything below is importable from `jevstiller` unless noted. It documents 0.4.0. The public API is what
`jevstiller`, `jevstiller.server`, `jevstiller.encoders`, `jevstiller.teachers` and `jevstiller.teachers.jev`
export; modules starting with `_` are internal. Until 1.0, the Python API may
change in a minor release ([compatibility](/proxy/compatibility/#versions-and-upgrades)); the proxy's HTTP
behaviour and its settings are what stay stable.

## `Task`

```python
Task(name: str, instructions, classes, target_agreement: float = 0.98, confidence_floor: float | None = None)
```

What is being classified: exactly what Jev is asked.
- `classes` maps each label to a description (text, a JSON object or array, or `None` for "by its name alone"); a
  plain list of labels means empty descriptions. Descriptions are sent to Jev verbatim. 2–255 classes.
- `instructions` may be text or any JSON value, as in Jev's `Choice`.
- `task.version` is a hash of the instructions and classes (not their order); changing either starts a new lineage.
- `task.budget` is `1 − target_agreement`, and `target_agreement` is in `[0.5, 1)`.
- `confidence_floor` is the Jev confidence below which your code treats an answer as unsure. With it, a local answer
  also counts as disagreeing when Jev would have been less confident, and reports at least the floor
  ([the guarantee](/concepts/guarantee/#if-your-code-acts-on-confidence)). `None` or `0`: none. Neither it nor
  `target_agreement` changes `task.version`.

## `Jevstiller`: one task

```python
Jevstiller(task, teacher, data_dir, *, encoder=None, config=None, train_executor=None, hash_key=None)
```

- `teacher`: anything implementing the [`Teacher`](#teachers) protocol, usually `JevTeacher()`.
- `data_dir`: the task lives in `<data_dir>/<task.name>/` (a SQLite sample store and its versions). Copy the
  directory and the task moves with it.
- `encoder`: defaults to `HashEncoder`, which needs no download but is weak. Pass `load_encoder("small")` or
  `"base"` for real use.
- `config`: a [`Config`](/reference/configuration/#engine-config).
- `train_executor`: where training runs. Default: a background thread per task (`Config.training`); pass a
  process pool (`train_pool()`) or a `TrainScheduler` to share one across tasks.
- `hash_key`: key for the per-row text hash (an HMAC), so a store kept with `store_text=False` holds no plain hash.

### Classifying

| method | returns |
|---|---|
| `classify(state, teacher=None)` | `Result` |
| `classify_batch(states, errors="raise", teacher=None)` | `list[Result]` |

A state is text or a JSON object/array (encoded as canonical JSON; Jev receives it unchanged). `teacher=`
overrides the teacher for one call, e.g. with the caller's own key. Teacher failures are per item: with
`errors="raise"`, a `TeacherError` carries the partial results; with `errors="return"`, failed items come back
with `label=None` and `error` set. Failed items are never recorded.

`Result` fields:

| field | |
|---|---|
| `label` | the answer |
| `probs` | distribution over all classes |
| `confidence` | Jev's own for its answers. For the student's: Jev's definition (the peakedness of `probs`), at least the confidence floor its version was calibrated for |
| `source` | `"teacher"` or `"student:vN"` |
| `routing_reason` | see [routing reasons](/concepts/how-it-works/#routing-reasons) |
| `latency_ms` | |
| `error` | the exception, for failed items with `errors="return"` |

When your code makes the teacher call itself (as the proxy does), split the call: `route(states)` decides and
returns a `Routed` whose `to_teacher` lists the items that need the teacher; `complete(routed, outputs)` records
the teacher's answers and returns the results. Always call `complete`, even when the teacher failed.

### Status and lifecycle

| method | does |
|---|---|
| `status()` | a `Status` with counts, shares, audit agreement and its bounds, cost, policy, readiness and recent events. `status().report()` gives the text report. |
| `versions()` | every student version and its state: `candidate`, `shadow`, `production`, `superseded`, `rejected`, `rolled_back`. `deleted` marks versions whose files are gone (past `Config.keep_versions`) |
| `set_mode(mode)` | `"auto"` (default), `"teacher_only"`, or `"cascade"` |
| `set_confidence_floor(floor)` | change the task's confidence floor. A student calibrated for it trains at once and takes over after shadow; until then production serves as it was calibrated |
| `train_now()` | train a candidate now and wait for it; returns a `TrainReport` |
| `maintain()` / `drain()` | run one maintenance pass (with `Config(training="manual")`) / wait for background work |
| `promote(version)` / `rollback()` | make a version production by hand / go back to the previous one |
| `export(path, version=None)` | copy a version (head, OOD reference, policy, task and encoder identity) into a standalone directory |
| `evaluate(states, teacher_labels, version=None, teacher_confidences=None)` | offline check of a version's policy: coverage, selective disagreement, system agreement. With `teacher_confidences`, a version's confidence floor counts as in calibration |
| `close()` | stop background work, flush and close the store |

## `TaskManager`: many tasks

```python
TaskManager(data_dir, teacher, encoder, config=None, *, target_agreement=0.98, confidence_floor=None, max_loaded=64,
            admission=None, ...)
```

The engine behind the proxy. A task is found by `(tenant, question, model)`, so callers asking the same question
share one student. Engines load on demand and unload least-recently-used past `max_loaded` / `max_memory_mb`.
`Admission(min_requests=50, window_s=86400)` keeps one-off questions from becoming tasks.

| method | does |
|---|---|
| `classify(tenant, instructions, classes, states, model=None)` | like `Jevstiller.classify_batch`, for any question |
| `route(..., caller=None)` / `complete(routing, outputs)` | the split form, when the caller calls the teacher; `caller` (e.g. an API key's hash) is counted against `max_new_tasks_per_caller` |
| `tasks(tenant=None)` / `loaded()` | registered / loaded tasks |
| `set_mode(key, mode)`, `set_target(key, target)`, `set_confidence_floor(key, floor)` | persisted per task; `target_agreement` and `confidence_floor` above are the defaults for new tasks |
| `delete(key)` / `delete_tenant(tenant)` | remove tasks and all their data |
| `apply_retention()`, `close()` | blank old text now; shut down |

Share one `BatchingEncoder` (concurrent calls merged into batches) and one `TrainScheduler` (fair across tenants,
busiest tasks first) across all tasks.

## Teachers

A teacher is any object with:

```python
def classify(self, texts: list, task: Task) -> list[TeacherOutput | Exception]: ...
```

`TeacherOutput` carries the label, the full probability distribution, confidence, token usage, cost, latency,
request id, and the model that answered.

| teacher | use |
|---|---|
| `jevstiller.teachers.jev.JevTeacher` | real Jev via `typesafe-sdk`. Reads `TYPESAFE_API_KEY`; `model="jev-1.13.0"`, `timeout=10`, and it rate-limits itself to 1,100 requests a minute. |
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
