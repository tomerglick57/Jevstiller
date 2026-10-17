# Compatibility
<!-- title: Compatibility: Jev versions, the TypeSafe SDK, Python and platforms -->

What Jevstiller works with, what callers see, and what a version number promises.

## Jev and the TypeSafe SDK

| | Tested with |
|---|---|
| TypeSafe SDK | `typesafe-sdk` 0.7.1 (the version in `uv.lock`), with both sync and async clients. CI runs the unmodified SDK against the proxy over real HTTP. 0.7.1 is also the version `pip install jevstiller` requires at least (for the Jev adapter). 0.7.5 (2026-10-14, adds images) was checked by hand on 2026-10-16: text and image requests (1, 6 and 20 MiB) through the proxy to a stub Jev. |
| Jev | `jev-1.13.0`, which `jev-latest` resolved to on 2026-09-24 and 2026-09-25: a live replay ([benchmarks](benchmarks.md)) and recorded wire responses (`tests/fixtures/jev/`) that every test run replays through the proxy. `jev-1.14.0` (released 2026-10-14 as `jev-preview`), tested live on 2026-10-17: its answers have the same fields, types and headers as 1.13's (`tests/fixtures/jev/*_jev-1.14.0.json`, replayed through the proxy by every test run), and a Banking77 replay against it ([benchmarks](benchmarks.md#jev-114)) |
| TypeSafe JavaScript SDK | Not tested. It speaks the same HTTP API, so it should work; nothing in CI checks it. |
| Other clients | Anything that speaks Jev's HTTP API. The proxy doesn't depend on the SDK; the SDK only needs `TYPESAFE_BASE_URL` (or `base_url=`) pointed at it. |

**When Jev changes:**
- **A new model behind `jev-latest`:** the task starts a new teacher lineage. Answers come from Jev until a student trained on the new model's answers passes shadow (`teacher_change`, DESIGN.md §7.12).
  - TypeSafe announced on 2026-10-14 that `jev-latest` moves to `jev-1.14.0` about a week later. Expect the local share of `jev-latest` tasks to drop then, and to come back as each task's new student trains on 1.14's answers (`min_train_samples`, `min_calib_samples`) and passes shadow (`shadow_min_samples`).
  - Measured on Banking77 ([benchmarks](benchmarks.md#jev-114)): 1.14 gives the same answer as 1.13 on 95% of messages, so a student of 1.13 would disagree with 1.14 on about 5% of requests, more than the 2% budget. That is why the default sends everything to Jev until a 1.14 student passes shadow.
  - With the default settings that takes each task at least about 3,500 requests after the switch: about 2,500 for its first calibration rows (500 at the 20% calibration share), then 1,000 in shadow. A quiet task stays on Jev for longer.
  - Tasks asked for `jev-1.13.0` by name keep their students. The requested model is part of the task, so moving a caller from `jev-latest` to `jev-1.13.0` starts new tasks from nothing.
- **Jev 1.14's experimental features** (the `TypeSafe-Danger-Zone-*` headers, which the proxy forwards unchanged):
  - images are forwarded and never learned from (below);
  - `box` questions are forwarded, like `score`;
  - with the long-context header the student still reads only the first 256 tokens ([coverage](coverage.md)).
- **A response the proxy doesn't recognise:** it is returned to the caller unchanged and not recorded.
- **New request fields or question types:** the request is forwarded as is (below).

The proxy never fails a request that Jev would have answered.

## Which requests are answered locally

| Request | What Jevstiller does |
|---|---|
| `POST /v1/systemone` whose questions are all `choice` questions of trained tasks, from a key Jev has accepted | Answered locally when every question clears its task's routing policy. Otherwise the whole request goes to Jev, and its choice answers become training rows. |
| `POST /v1/systemone` whose questions are `choice` and yes/no (`noul`) questions of trained tasks | The same: answered locally when every question clears its task's policy. A `noul` answer is `{"type": "noul", "noul": p}` with the student's probability, on Jev's side of your cut-off (`noul_cutoffs`, default 0.5) on at least `target_agreement` of requests. |
| `POST /v1/systemone` with other question types (`score`, …), a `noul` question with fields beyond `instructions` and `criteria`, or any `noul` question when `noul_cutoffs` is empty | Forwarded. The request's choice and noul answers are still recorded. |
| `POST /v1/systemone` the proxy doesn't understand: invalid JSON, unknown top-level fields, no bearer key, more than `max_questions` distinct questions | Forwarded as is; nothing is recorded |
| `POST /v1/systemone` with images (Jev 1.14): a multipart body, or `assets` in the JSON | Forwarded as is; nothing is recorded. With the `TypeSafe-Danger-Zone-Images: true` header the body may be up to `max_image_body_mb` (32 MiB, Jev's own limit). Without the header the usual `max_body_mb` (4 MiB) applies; Jev refuses an image request without it anyway. |
| Any other path or method (`GET /v1/models`, …) | Forwarded unchanged |
| `/healthz`, `/readyz`, `/metrics`, `/jevstiller/*` | Served by the proxy, never forwarded |

The details (admission, tenancy, keys, limits) are in [the proxy docs](proxy.md).

## What a local answer looks like

The same JSON as Jev's, so the SDK parses it into the same objects. The differences:

| Field | Local answer |
|---|---|
| `answers.<name>.choice` | The student's label. It matches Jev's on at least `target_agreement` of requests, as a statistical bound (DESIGN.md §7.6), not on every request. With a confidence floor, the bound also counts the requests Jev would have been less confident about than the floor. |
| `answers.<name>.probabilities` | The student's distribution over the question's classes |
| `answers.<name>.confidence` | Jev's definition (the peakedness of the distribution), computed on the student's probabilities, and at least the task's confidence floor if it has one. It is the student's own number, not a prediction of Jev's: without a floor, nothing bounds it. |
| `model` | The concrete Jev version the student learned from (e.g. `jev-1.13.0`, not `jev-latest`) |
| `usage` | Zero tokens |
| `x-typesafe-request-id` | `jvs_<uuid>` (Jev's own ids start with `req_`) |
| Extra headers | `x-jevstiller-source: local`, and `x-jevstiller-detail` naming the student version per question |
| Latency | Typically 11–25 ms at the median on CPU ([benchmarks](benchmarks.md)), against Jev's ~290 ms |

Errors the proxy produces itself (a 429 while Jev's `retry-after` lasts, 502/503/504, 4xx from access checks) use Jev's error shape, so the SDK raises and retries as it does against Jev.

**If your code acts on `confidence`,** for example sending answers below 0.6 to review, set the task's confidence floor to that number (`confidence_floor`, [configuration](configuration.md)). Without it, only the label is covered. The student answers only when it is confident, so a local answer never reports a low confidence (none went below 0.70 in the benchmark), even where Jev would have. On Banking77 with a check at 0.6:
- **Messages sent to review:** 11.5% calling Jev directly, 9.7% through Jevstiller.
- **Flags lost:** 15.5% of the messages Jev would have flagged came back with a confident local answer ([benchmarks](benchmarks.md#a-callers-confidence-check-confidence_floor-2026-09-27)).
- **Higher cut-offs** also flag messages Jev was sure about, because the student's numbers run lower than Jev's.

With the floor set, a local answer stands for "Jev would have given this label with at least this confidence", and the target bounds how often that's wrong.

**Agreement is not accuracy.** Every number Jevstiller reports is agreement with Jev. Where Jev is wrong, the student is wrong the same way, and the audit reports full agreement. Measuring accuracy needs labels from outside Jev.

## Python, platforms, deployment

| | Supported |
|---|---|
| Python | 3.10 to 3.14. CI tests every version, plus the lowest allowed version of each dependency. |
| Linux | Tested (CI, and the container image) |
| macOS | Expected to work; not tested in CI |
| Windows | Not tested. Run the container, or use WSL. |
| Install | `pip install jevstiller` includes the proxy, the default encoder (ONNX Runtime, CPU) and the Jev adapter. ONNX Runtime ships for x86-64 and ARM64 on Linux, macOS and Windows; elsewhere (32-bit ARM, Alpine/musl) use the container image, `jevstiller[gpu]` (PyTorch), or `--encoder hash`. On a GPU: `pip install "jevstiller[gpu]"`. |
| Container image | `ghcr.io/tomerglick57/jevstiller`, linux/amd64 and linux/arm64, CPU (ONNX Runtime). For a GPU, build on a CUDA base image ([deploy](deploy.md)). |
| Replicas | One process per data directory, and one replica per deployment. Several processes on one data directory are not supported. |

## Versions and upgrades

Jevstiller follows semantic versioning, with the usual caveat before 1.0: a minor release (0.2 → 0.3) may change behaviour, and each such change is listed in the [changelog](../CHANGELOG.md) with upgrade notes. A patch release (0.2.0 → 0.2.1) only fixes bugs.

Within a minor version, these stay compatible:
- the proxy's HTTP behaviour towards callers;
- the settings: TOML keys, `JEVSTILLER_*` variables and flags;
- the data directory format.

**The Python API** (defined from 0.3.0; 0.2.0 had no declared boundary) is what these modules export in `__all__`:
- `jevstiller`: the engine (`Jevstiller`, `Task`, `Config`, `Result`, `Status`, …), `TaskManager`, `TrainScheduler`, `train_pool`, the teacher and encoder protocols and the built-in ones;
- `jevstiller.server`: `create_app`, `ProxySettings`, `KeyRegistry`, `load_salt`;
- `jevstiller.encoders`, `jevstiller.teachers`, `jevstiller.teachers.jev` (`JevTeacher`).

`tests/public_api.txt` records every public name, signature and dataclass field, and a test keeps it current. Modules and names starting with an underscore (`jevstiller._core`, …) are internal and can change in any release. Until 1.0 the Python API may still change in a minor release; each change is listed in the changelog.

**Upgrades:** a newer version upgrades a data directory in place on first start: sample stores are migrated, and tasks are re-keyed if the key scheme changed. Back up first (`jevstiller backup`). Downgrading to an older version with an upgraded data directory isn't supported: restore the backup instead.

Jevstiller is an independent open-source project (Apache 2.0). It is not affiliated with, endorsed by, or supported by TypeSafe.
