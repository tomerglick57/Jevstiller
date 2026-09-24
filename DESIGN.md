# Jevstiller — Design

> Status: v3 (2026-09-24). v2 was the pre-implementation design. v3 records what was built, what the live
> measurements changed, and what is still roadmap: §14 lists implementation status, and anything not built is
> marked *(roadmap)* where it is described. The operational plan is in [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md),
> the proxy in [docs/proxy.md](docs/proxy.md), every setting in [docs/configuration.md](docs/configuration.md),
> and every measurement in [docs/benchmarks.md](docs/benchmarks.md).
> Jevstiller = **Jev** + di**stiller**. Jev is the default teacher; the design is teacher-agnostic.

---

## 1. What it is

Jevstiller is a layer that sits between a *repeated* classification task and a general-purpose model ("Jev"). You call Jevstiller instead of Jev. At first every request is forwarded to Jev unchanged. In the background, Jevstiller uses Jev's own answers to train a small task-specific model, verifies that the small model reproduces Jev's answers, and then starts answering requests itself — sending to Jev only the requests it is not sure about, plus a small random audit sample.

The closest analogy is a cache:

| Cache | Jevstiller |
|---|---|
| transparent to the caller | same API in, same shape out |
| warms up from traffic | learns from traffic |
| serves hits locally, misses go to origin | serves confident inputs locally, uncertain ones go to Jev |
| invalidates on change | detects drift and falls back to Jev |
| exact | **approximate, with a bounded disagreement rate** |

The last row is the whole design problem. Everything below exists to make the bound real, measurable, and visible.

### Two ways to run it

- **Drop-in proxy** (`jevstiller serve`): an HTTP server that speaks Jev's API. Services set `TYPESAFE_BASE_URL` to it and change nothing else. They keep their own Jev keys, which the proxy forwards and never stores. Every `Choice` question in the traffic becomes a task automatically, and one process serves many tasks for many services (§7.14–§7.15, [docs/proxy.md](docs/proxy.md)). This is the primary deployment: self-hosted, one container.
- **Library** (`Jevstiller(task, teacher, ...)`): the same loop embedded in a Python process, for batch jobs, workers and notebooks.

### The teacher

Jev (TypeSafe, `jev-1.13.0`) is a "System One" model: it takes a *state* (text or JSON) and typed *questions* and returns typed answers — no text generation. For a `Choice` question it returns the chosen option, **a probability distribution over every option**, and a scalar confidence derived from that distribution. Jevstiller maps a task to exactly one `Choice` question per request. The distribution is stored on every call and is the student's training target. Full reference in Appendix A.

### The contract

> For a task with many repeated requests, Jevstiller returns the label **Jev would have returned** on at least `target_agreement` of them, while sending as few requests to Jev as possible.

The user sets one number (`target_agreement`, e.g. 0.98) and Jevstiller spends the remaining **disagreement budget** (1 − 0.98 = 2% of traffic) to maximise how much it can answer itself.

---

## 2. What it is *not*

**It is not an accuracy system.** There is no ground truth anywhere in the product. The only label source is Jev. If Jev is systematically wrong about something, the student will be wrong in the same way, and Jevstiller will report 100% agreement while both are wrong. This is stated plainly, everywhere the metric is shown: the metric is *agreement with Jev*, never "accuracy" or "precision".

This is the honest version of the product, and it is also the *buildable* version: certifying parity with a teacher needs only the teacher, whereas certifying absolute accuracy needs thousands of human labels per class, re-verified on every drift. That second product is a labelling company; this one is not.

**It is not a fine-tuning platform.** One student architecture, one training recipe, chosen for portability and retraining speed rather than peak capability.

**It is not an accelerator for one-off calls.** It only pays off on volume (see §3).

---

## 3. When it pays off

Be precise about what Jev costs, because it is not what the original concept assumed.

| | Jev (`jev-1.13.0`, Sept 2026) | local student |
|---|---|---|
| price | $0.042 / M input tokens, output free. Input tokens include the instructions and every criterion, not just the message: a short support message with a 5-class question measured **400 input tokens** (2026-09-24) → ≈ $0.000017; **1M messages ≈ $17**. Long criteria lists (77 classes) cost more. | ~0 |
| latency | measured 2026-09-24: **p50 ~290 ms, p90 ~330 ms, p99 ~760 ms**, flat from 1 to 16 concurrent requests | single-digit ms on CPU |
| throughput | **1,200 requests/min (20/s), 250k tokens/s**, one state per request; limits "adjust dynamically" | thousands/s on one CPU; more on GPU |
| availability | hosted only; early access; `529 overloaded` exists; no self-hosting | runs wherever the process runs |

So the honest ranking of what Jevstiller buys you:

1. **Throughput beyond the rate limit.** 20 req/s is a hard ceiling of ~1.7M classifications/day per key. A backlog of 100k messages takes 83 minutes through Jev; the student does it in under a minute. Any bursty or high-volume workload hits this wall long before it hits a cost wall.
2. **Latency.** ~350 ms → ~5 ms p50 for the requests the student handles.
3. **Availability and independence.** Keeps answering during Jev outages, 429/529 storms, and network egress restrictions (the audit/deferral stream queues and drains later). Hedges an early-access API whose limits and pricing are explicitly provisional.
4. **Cost** — last. At current prices it only matters at very high volume (it is ~6× what v2 assumed, because the question's criteria are billed on every call), and TypeSafe says the price "may be subsidized", so it is a hedge rather than a saving.

It is worth using when volume is high enough to hit (1) or (2), the class list is stable, and the input distribution changes slowly. The bootstrap costs nothing extra — requests were going to Jev anyway, and every Jev answer (with its probability distribution) is training data. The ongoing Jev cost is the audit channel plus deferrals.

Jevstiller should tell the user when it is *not* paying off (§10) rather than hide it.

---

## 4. User experience

Define a task once. Classes carry descriptions because Jev is a literal reader: the `criteria` text *is* the specification, and it is what the student learns to reproduce.

```python
from jevstiller import Task, Jevstiller
from jevstiller.teachers.jev import JevTeacher

task = Task(
    name="support_router",
    instructions="Which team should handle this customer message?",
    classes={
        "billing":      "Charges, invoices, refunds, payment methods, subscription price",
        "technical":    "Bugs, errors, integrations, things not working",
        "cancellation": "Wants to cancel, downgrade, or close the account",
        "sales":        "Pre-sales questions, plan comparison, quotes, upgrades",
        "other":        "Anything that does not fit the categories above",
    },
    target_agreement=0.98,       # the contract
)

js = Jevstiller(task, teacher=JevTeacher(), data_dir="./jevstiller-data")   # reads TYPESAFE_API_KEY
```

Use it exactly like Jev:

```python
r = js.classify("Please cancel my subscription")
r.label       # "cancellation"
r.probs       # {"cancellation": 0.97, "billing": 0.02, ...}
r.confidence  # 0.96
r.source      # "teacher" at first, later "student:v3"
```

Changing `instructions` or any class description changes the teacher's behaviour, so it bumps `task_version` and starts a new lineage (old samples are kept but marked; the student retrains from the new version's samples only). Instructions and class descriptions may be any JSON value, as in Jev's `criteria`, and the state may be text or a JSON object/array. The version ignores the order of classes and of JSON keys.

Through the proxy there is no `Task` to write: the question in each request *is* the task definition (§7.14).

What the user sees over time (from `js.status()` or the CLI report):

```text
day 1     100% -> Jev                 collecting  (1,200 req/min ceiling)
day 3      shadow: student-v1 agrees with Jev on 97.1% (n=2,400)  -> not enough
day 6      shadow: student-v2 agrees with Jev on 99.2% (n=6,100)  -> promoted
day 6      71% -> student-v2   29% -> Jev
day 14     91% -> student-v3    9% -> Jev
day 30     97% -> student-v4    3% -> Jev
day 45     drift detected (fallback 3% -> 11%, OOD rate up)  -> retraining
day 46     96% -> student-v5    4% -> Jev
```

At every point the user can see how many Jev calls were avoided (and what that is in rate-limit time and dollars), the measured agreement with its confidence interval, and why any request went to Jev.

---

## 5. The guarantee, precisely

Notation, per task, over a window of traffic:

| symbol | meaning |
|---|---|
| `c` | **coverage** — fraction of requests answered by the student |
| `e` | **selective disagreement** — P(student label ≠ Jev label \| student answered) |
| `A` | **system agreement** — fraction of requests where Jevstiller's answer equals what Jev would have said |

Requests the student defers go to Jev, so they agree with Jev by definition. Therefore:

```text
A = 1 − c · e
```

The user's target `A*` gives a disagreement budget `β = 1 − A*`. The router's job:

```text
maximise   c
subject to c · e  ≤  β
```

Two things make this an actual guarantee rather than a hope:

1. **`e` is estimated on an IID held-out calibration set** — a random sample of traffic, labelled by Jev, never used for training, and never drawn from the actively-collected (deferred) stream, which is boundary-skewed by construction.
2. **The threshold is chosen against an upper confidence bound on `e`, not its point estimate.** With `n` calibration examples above a candidate threshold and `k` disagreements among them, use the one-sided Clopper–Pearson bound at confidence `1 − δ` (default δ = 0.05). Pick the lowest threshold whose bound still satisfies the budget. This is the SGR construction from selective-classification literature and gives a finite-sample guarantee: with probability ≥ 1 − δ the true selective disagreement is within budget.

The bound is then **re-verified continuously in production** on the audit channel (§7.8), which is the only unbiased view of live traffic once routing begins.

The student's confidence never has to be a *true probability* for this to work. It only has to *rank* inputs so that low-confidence ones disagree more often. Calibration is done on the ranking, not on the probabilities.

---

## 6. Architecture

```text
                          classify(text)
                                |
                                v
                     +---------------------+
                     |       Router        |
                     |  mode: teacher_only |
                     |        cascade      |
                     |  (shadow: a stage,  |
                     |   hedge: roadmap)   |
                     +---------------------+
                       |        |        |
        audit sample   |        |        |  confident
        (always ~2%)   |        |        |
                       v        v        v
                    +------+  +-----+  +---------+
                    | Jev  |  | Jev |  | Student |
                    +------+  +-----+  +---------+
                       |        |  ^        |
                       |        |  | defer  |
                       |        |  +--------+
                       v        v           v
                  +------------------------------+
                  |         Sample store          |
                  |  text · embedding · teacher   |
                  |  label/probs · student output |
                  |  · channel · split · version   |
                  +------------------------------+
                        |              |
              training  |              |  calibration + audit
              split     v              v
                  +-----------+   +--------------+
                  |  Trainer  |   |  Calibrator  |
                  | frozen    |   |  threshold   |
                  | encoder + |   |  under       |
                  | soft-label|   |  budget β    |
                  | head      |   +--------------+
                  +-----------+          |
                        |                |
                        v                v
                  +------------------------------+
                  |      Candidate = model        |
                  |          + routing policy     |
                  +------------------------------+
                                |
                        shadow evaluation
                          (live traffic)
                                |
                          meets budget?
                          /           \
                        no             yes
                        |               |
                   keep collecting   Registry -> production
                                          ^
                                          |
                  +------------------------------+
                  |        Drift monitor          |
                  |  audit agreement · fallback   |
                  |  rate · OOD rate · class mix  |
                  +------------------------------+
                           |          |
                     retrain     fall back to Jev
```

That loop runs once per task. The deployment around it (the proxy, §7.15) runs many of them in one process:

```text
 service A ─┐                          ┌──────────────── jevstiller serve (one process) ──────────────┐
 service B ─┼─ TYPESAFE_BASE_URL ────► │ POST /v1/systemone        (everything else: passed through)  │
 service C ─┘   + their own Jev key    │   key check: salted hash, accepted by Jev within the TTL?    │
                                       │   each choice question → task key → TaskManager (§7.14)      │
                                       │      └─ that task's loop: route → student | teacher          │
                                       │   all local? → answer in Jev's shape                         │
                                       │   else      → whole request to Jev (caller's key) → record   │
                                       │ shared: BatchingEncoder · TrainScheduler (process pool)      │
                                       │ per task: <data_dir>/tasks/<key>/  SQLite store + versions   │
                                       └───────────────────────────────────────────────────────────────┘
```

---

## 7. Components

### 7.1 Teacher adapter

A protocol, not a class hierarchy:

```python
class Teacher(Protocol):
    name: str
    def classify(self, states: list[State], task: Task) -> list[TeacherOutput | Exception]: ...
    # one entry per state; an Exception for an item that failed (raising fails the whole call)

@dataclass
class TeacherOutput:
    label: str
    probs: dict[str, float]          # full distribution over task.classes
    confidence: float                # teacher's own scalar, if it has one
    input_tokens: int
    cost_usd: float
    latency_ms: float
    request_id: str | None
    raw: Any                         # provider response, kept for audit
    model: str | None                # the model that actually answered, e.g. "jev:jev-1.13.0" (§7.12)
```

**`JevTeacher`** (the only real adapter in the MVP) wraps `typesafe-sdk`:

- one request per state: `state=text` (or the user's JSON object, passed through), `questions={"label": Choice(instructions=task.instructions, criteria=task.classes)}`. The default `model` is pinned (`jev-1.13.0`); with an alias, the resolved version is still tracked (§7.12);
- `probs` ← `answers["label"].probabilities` (renormalised over the task's labels), `label` ← `.choice`, `confidence` ← `.confidence`, `model` ← `jev:` + the response's resolved `model`;
- `cost_usd` ← `usage.input_tokens × price_per_mtok / 1e6`;
- concurrency via a thread pool (`concurrency=8`) behind a local token bucket (`rpm=1100`, under Jev's 1,200/min); the SDK's own retry policy handles 429/5xx; `timeout=10` s per call;
- one failed request fails only its own item;
- *(roadmap)* truncating state beyond the 32k-token budget.

In the proxy there is no `JevTeacher`: the proxy forwards the caller's own request and parses Jev's response into the same `TeacherOutput` for each choice question.

**Soft labels are the main sample-efficiency lever and Jev gives them away on every call.** The student trains against `probs`, not `label`. Where Jev is uncertain the target is flat, so the student learns to be uncertain there too — which is exactly what the router needs. The adapter still records a `label_source` field (`probs` | `hard`) so the experiment (§15) can compare training on the distribution vs. the argmax.

A `SyntheticTeacher` (deterministic rules + injectable noise + fake distributions) ships with the package for tests and demos; the default test suite never calls a real API (live tests are opt-in: `JEVSTILLER_LIVE=1 pytest -m live`). A `ReplayTeacher` serves cached Jev answers from the sample store, and `CachedTeacher` persists live answers to JSONL, so experiments can be re-run offline and for free.

### 7.2 Sample store

An append-only log of **every** request, whether served by student or teacher. Backend: one SQLite file per task (portable, zero-ops; tasks never share a writer; deleting a task is deleting its directory). The interface is the `Store` protocol; *(roadmap)* a Postgres backend for multiple replicas.

Implementation notes:

- **Write-behind.** `insert` queues records and returns. One writer thread per store commits them in batches, so no request waits on the disk; this took throughput from 19 to ~500 req/s with a 50 ms teacher. Every read through the store waits for earlier inserts first, and a crash loses at most the few milliseconds still queued.
- **Connections.** One SQLite connection per thread (WAL), and writers queue on an in-process lock instead of SQLite's sleeping busy handler.
- **Schema version.** `PRAGMA user_version` records it, so setup is skipped when current (reopening costs ~0.4 ms), and older stores are migrated in place (e.g. the `state_type` column).

Per record:

```text
id, ts, task_version
text, text_hash, state_type            # state_type: text | json (text = canonical JSON of the object)
embedding (blob, encoder_id)           # stored so retraining never re-encodes
teacher_label, teacher_probs, teacher_confidence, teacher_model,
teacher_input_tokens, teacher_cost_usd, teacher_request_id          # null if not sent to teacher
                                                                    # teacher_model = the lineage key (§7.12)
student_version, student_probs, student_confidence, ood_score           # null if no student ran
served_by (teacher|student), routing_reason
channel  (bootstrap | fallback | audit | deferred | student)
split    (train | calib)               # assigned at insert by hash(text_hash); fixed forever
latency_ms, teacher_cost
```

Two rules that keep the guarantee valid:

- **Split is assigned once, by hash, at insertion.** A text is always in the same split across every retrain. No leakage, no reshuffling drift.
- **Only `channel ∈ {bootstrap, audit, fallback}` records are eligible for `calib`** (all three are sent to the teacher regardless of the student). Deferred records are informative for *training* (they are exactly the hard cases) but are not IID and must never influence the threshold. That includes rows the proxy sends to Jev only because another question in the same request needed it (`co_deferred`).
- **Training, calibration, shadow and audit statistics read one teacher lineage only** (§7.12).

### 7.3 Encoder (frozen)

A pretrained sentence encoder, run frozen, behind one interface:

```python
class Encoder(Protocol):
    id: str                      # name + content hash
    dim: int
    def encode(self, texts: list[str]) -> np.ndarray   # (n, dim), L2-normalised
```

Two backends, same interface:

- **ONNX Runtime** (default): one exported artifact runs on CPU, CUDA (`onnxruntime-gpu`), and ARM with no code change. Device auto-detected, overridable.
- **PyTorch** (optional `[torch]` extra): loads Hugging Face checkpoints directly, uses CUDA if present. Handy for trying encoders before exporting them, and for the bigger models in §15.

Candidate sizes, all English-first, all common and well-tested:

| tier | model | params | dim | ONNX, CPU (16 cores) | torch, RTX 3090 | coverage @ 98% agreement, oracle teacher, 11k Banking77 rows |
|---|---|---|---|---|---|---|
| small | `bge-small-en-v1.5` | 33M | 384 | 1.56 ms/text | 0.38 ms/text | 56% |
| **base** (default) | `bge-base-en-v1.5` | 110M | 768 | 5.39 ms/text | 0.48 ms/text | **71%** |
| large | `bge-large-en-v1.5` | 335M | 1024 | 17.92 ms/text | 0.94 ms/text | ~60% (noisier; not better) |

Measured 2026-09-21 (batch 64, short support messages; the three backends produce identical embeddings to 1e-6 cosine). The coverage column is the §15 dry run with a perfect teacher — an upper bound on distillation, not a Jev result — after the linear head's regularisation was fixed (§7.4). **Decision: `base` is the default.** It buys ~15 points of coverage over `small` for the same GPU cost; on CPU it costs 3.5× more per text, which is the trade a CPU-only deployment should make consciously (`encoder: small`). `large` did not beat `base` at this data size and costs 3× more again.

Why frozen rather than fine-tuned, for v1:

- Embeddings become a **stable coordinate system**. Drift and OOD scoring (§7.5) compare new inputs to old ones in the same space; a fine-tuned encoder would move the space every retrain.
- **Retraining is seconds**, on CPU: embeddings are stored once, only the head is refit.
- The sample store's embeddings are reusable across every student version.
- Failure modes are simpler to reason about.

Swapping the encoder is a config change; it starts a new lineage (rows are keyed by `encoder_id`, and old students are invalid for the new encoder). *(roadmap)* Re-encoding stored text in the background, so the new lineage doesn't start from zero. The store keeps text unless `store_text=False`, so this is always possible.

**One encoder per process.** The task manager shares one encoder across every task. `BatchingEncoder` wraps it so concurrent `encode` calls from many requests run as one batch: one worker thread, and calls that queue while it runs join the next batch. There is no added wait when idle, and batches grow under load, which a GPU needs. The CLI defaults to `small` (CPU-friendly); the library's `load_encoder()` defaults to `base`.

### 7.4 Student head

Multinomial logistic regression on the embedding, trained with cross-entropy against the teacher's **soft** distribution. Implemented in numpy with a hand-written optimiser (full-batch Adam) so there is no framework dependency on the inference path. *(roadmap)* An optional small MLP behind the same interface for tasks where linear is not enough.

A saved student records its label order and is reordered to the task's order when loaded. This is needed because the task version ignores class order: a task declared with its classes in another order would otherwise misread a saved model (found and fixed in P1.6).

Regularisation is by **early stopping on an internal validation slice** (10% of the training rows, seeded), not by a fixed L2. Measured on Banking77 embeddings: L2 = 1e-4 cost 15–40 points of coverage at the same budget versus L2 ≈ 0, and made larger encoders look *worse* than smaller ones. Fixed hyperparameters are fragile across encoder dimensions and dataset sizes; a validation-selected stopping point is not.

```python
class Student(Protocol):
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray   # (n, n_classes)
```

**Importance weighting.** Once routing starts, teacher-labelled rows are no longer a sample of traffic: every deferred (hard) request is captured, but only `audit_rate` of the easy ones. Training on raw counts drifts the student toward the hard slice and *lowers* its confidence on easy traffic (observed: candidates trained on 5× more data had lower coverage than their predecessor). So each row carries a weight reflecting the number of requests it stands for — `(1/audit_rate)^power` for audit rows (capped), 1 otherwise — and the head is trained with those weights. With `power = 1` this is exact Horvitz–Thompson reweighting; the default `power = 0.5` tempers it, because at a 2% audit rate exact weights of 50 let a few hundred audit rows dominate thousands of deferred ones (effective sample size ≈ 600 out of 5,000, observed as erratic candidate quality). Tempering trades a little bias for a lot of variance.

Class-imbalance handling and a validation split for early stopping / L2 selection are internal; nothing here touches the calibration split.

### 7.5 Out-of-distribution scorer

Softmax confidence is systematically *over*confident on inputs unlike the training data — precisely where deferral matters most. So OOD is a separate gate, not folded into the class probabilities:

- Score = 1 − mean cosine similarity to the k nearest reference embeddings (k = 10). *(roadmap)* Mahalanobis distance to class centroids as an alternative.
- The reference set is at most `ood_max_ref` (5,000) training embeddings, **stratified by class**: each class keeps up to an equal share, and the rest of the budget is filled at random. Memory and per-request cost scale with it (5,000 × 384 floats ≈ 7.7 MB per version). Measured against keeping every row: no change in held-out coverage or agreement on Banking77 (even at 1,000) or CLINC150.
- Threshold = a high quantile (default 99th) of the same score computed on the calibration split.
- Any input above the threshold is deferred to Jev regardless of class confidence, and logged with `routing_reason = ood`.

The *rate* of OOD deferrals is one of the primary drift signals.

### 7.6 Calibrator

Input: a trained student, the calibration split (teacher-labelled, IID), the budget `β`, confidence `1 − δ`.
Output: a **routing policy** — the confidence threshold, the OOD threshold, and the numbers that justify them.

```text
for each candidate threshold t (descending):
    S_t = calib examples with confidence ≥ t
    k   = disagreements with teacher in S_t
    ub  = clopper_pearson_upper(k, |S_t|, δ)      # bound on e at this t
    cov = |S_t| / |calib|                          # estimated c at this t
    if cov · ub ≤ β: record (t, cov, ub)
choose t with the largest cov
```

The policy is fitted at `(1 − fit_headroom) · β` (default 85% of the budget) and the shadow stage judges it at the full `β`. Without headroom, a threshold chosen to sit exactly on the budget fails the out-of-time re-test about half the time by construction (observed).

The policy is versioned together with the model it was fitted for. A model without a policy cannot be routed to.

**Rare classes.** With `rare_classes="defer"`, classes with fewer than `min_samples_per_class` training rows become the policy's `deferred_labels`. The student never answers a request it would label with one of them (`routing_reason="rare_class"`), and calibration only counts rows the student may answer, so the bound stays honest. The default, `wait`, holds back the first student until every class has enough samples, and `status()` names the classes it is waiting for.

Roadmap: per-class thresholds, and a *class-weighted* budget ("a wrong `cancellation` costs five times a wrong `other`").

### 7.7 Router

Modes, per task, switchable at runtime:

| mode | behaviour | when |
|---|---|---|
| `teacher_only` | everything to Jev, everything logged | bootstrap; emergency fallback |
| `cascade` | student first; defer to Jev if below threshold or OOD | normal production |
| `auto` (default) | `teacher_only` until a student is promoted, `cascade` after | |
| *shadow (a stage, not a mode)* | a candidate runs on every request next to production; its answers are recorded, never returned | evaluating a candidate on live traffic at no quality risk |
| *(roadmap)* `hedge` | student and Jev called concurrently; return student if confident, else wait for Jev | latency-sensitive tasks |

In every mode except `teacher_only`, an **audit sample** (§7.8) is drawn first and sent to Jev unconditionally.

Every decision writes a `routing_reason`: `bootstrap`, `fallback`, `audit`, `confident`, `low_confidence`, `ood`, `rare_class`. The proxy adds `co_deferred` (another question in the request needed Jev) and `key_unverified` (the caller's key hasn't been accepted by Jev yet); requests for tasks that don't exist yet carry `not_admitted` or `tenant_task_limit`.

The router holds a lock only to snapshot the routing state (production and shadow versions, flags, the audit draw). Encoding, inference, the teacher call and the store write all run outside it, so concurrent callers' teacher calls overlap (§7.13).

### 7.8 Audit channel

A fixed fraction of *all* requests (`audit_rate`, default 2%) is sent to Jev regardless of what the student thinks, and the student's answer is recorded alongside. This is the single most important addition over the original concept, because once routing starts:

- the deferred stream is *the hard slice*, not traffic — training on it alone skews class priors and starves the student of evidence for the easy cases;
- drift metrics computed on the deferred stream measure the router, not the world;
- the production agreement rate cannot be measured at all without an unbiased sample.

The audit channel is therefore, simultaneously: the unbiased production estimate of `A` (with a confidence interval), the source of fresh IID calibration data, the drift monitor's input, and a steady trickle of unbiased training data. Its cost is `audit_rate × teacher cost` — a permanent floor on Jev usage, and the price of the guarantee.

`audit_rate` is raised automatically (to `audit_rate_shadow`) while a candidate is in shadow — so the candidate is judged on fresh traffic within a few thousand requests instead of starving behind a 2% trickle — and (to `audit_rate_elevated`) while the drift monitor is suspicious. It drops back on promotion.

### 7.9 Drift monitor

Runs continuously over sliding windows. Signals, in rough order of importance:

| signal | source | what it catches |
|---|---|---|
| audit agreement (with CI) | audit channel | the contract itself |
| fallback rate (low-confidence + OOD) | router | input shift the student notices |
| OOD rate alone | OOD scorer | novel content |
| predicted class mix vs. training class mix | router | prior shift |
| embedding centroid drift | store | gradual semantic shift |
| teacher/student disagreement on deferred | store | student getting worse at the boundary |

Escalation on the audit channel's *system agreement* `A` (the student's accepted answers vs. the teacher's, over the last `drift_window` audit records, both confidence bounds at `1 − δ`):

```text
suspicious:  A < target                 -> request a retrain; audit rate raised; candidate goes to shadow
broken:      ub(A) < target − margin    -> mode = teacher_only immediately; alert; retrain on data from here on
recovered:   a candidate passes shadow  -> back to cascade
```

**A broken contract restarts the training data.** Jev can change its answers without changing its model name, so a confirmed break may mean the older answers no longer describe the task. From the `fallback` event on (it records `since_id`, the last row before it), training, calibration, shadow and audit read only newer rows. Older rows stay in the store (for rollback and analysis) but no longer train. Before this rule, a task whose teacher silently changed retrained forever on mixed old and new answers, and every candidate failed shadow: the soak test (P6.6) found it. The rule also holds for plain input drift, where it costs some extra teacher calls. While in fallback every request goes to the teacher, so the new data fills quickly: in the replay test a task was back to 81% local answers within 2,000 requests of the fallback, and at its old ~96% within ~20,000. The fallback survives a restart; `mode auto` from the operator ends it (`fallback_cleared`).

The "broken" test uses the *upper* bound on purpose: it fires only when the data are confident the contract is violated, not merely when a small audit sample is noisy. "Suspicious" uses the point estimate: at n = 500 and a 98% target, the lower bound sits below target even at 1% disagreement, so gating retrains on it would keep the audit rate elevated permanently. Until `drift_min_samples` audit records exist the monitor stays silent. The status report shows the interval and one of `OK` / `inconclusive` / `BROKEN`.

The check runs after every `drift_check_every` (20) new audit answers, not on every request. *(roadmap)* Embedding-centroid, class-mix, fallback-rate and OOD-rate signals feeding the same ladder; today only audit agreement does.

---

### 7.10 Registry and lifecycle

Explicit versions: `student:v1`, `student:v2`, … Each version is a directory containing the head weights, the routing policy, the encoder hash, training metadata (sample counts, split hashes, dates), and shadow-evaluation results. Immutable once written.

```text
   collecting
       |   training trigger (§8)
       v
    training ----> failed
       |
       v
   candidate  (model + policy fitted on calib split)
       |
       v
    shadow    (runs on live traffic alongside Jev; no routing)
       |
       +-- pooled (calibration + shadow) bound fails budget  -> rejected, keep collecting
       |
       +-- passes, and coverage is not worse than production on the shadow rows
                  -> production
                        |
                        +-- drift monitor step 3 -> teacher_only (rollback)
                        +-- newer version passes shadow -> superseded (kept for rollback)
```

Rollback is a pointer change: `production -> student:v3` (or `-> teacher_only`). Nothing is deleted.

### 7.11 Training trigger

Simple thresholds, all configurable (§9):

```text
always:           ≥ min_calib_samples in the calibration split
first training:   ≥ min_train_samples  AND  ≥ min_samples_per_class  in the train split
                  (rare_classes="defer": at least two classes with enough, the rest deferred, §7.6)
                  also after a teacher change or a confirmed drift (no student for the current data);
                  after a candidate that didn't make it: 25% more rows first (at most min_new_samples)
retrain:          ≥ min_new_samples since last training (kept in the version's metadata, so a reload
                  or restart does not count from zero)
             OR   drift monitor or a teacher change requested it
             OR   elapsed time ≥ retrain_interval   (roadmap)
never:            while a candidate is already in shadow, or a training job is queued or running
```

Training is idempotent and cheap (frozen encoder), so being trigger-happy is fine; the shadow gate is what prevents bad promotions. The one exception is a task whose candidates keep failing (a hard task, or one still settling after a drift): it used to retrain every 100 rows, keeping a training worker busy forever. The 25% backoff bounds that to a logarithmic number of fits. The cheap checks come first (new rows since the last training); the store is only re-counted every 100 new rows, because a maintenance pass runs about once a second per loaded task.

### 7.12 Teacher lineage

A task's student reproduces *one* teacher. `jev-latest` is an alias, so the model behind it can change without the task changing. Every teacher answer therefore records the model that actually produced it (`teacher_model`, e.g. `jev:jev-1.13.0`, from Jev's response), and the loop keeps a current **lineage**:

- training, calibration, shadow judgement and the audit statistics read only rows of the current lineage;
- a new model must answer `teacher_change_confirm` (20) times in a row before the lineage switches, so a gradual rollout behind the alias can't make it flap;
- on a switch (`teacher_changed` event), any shadow candidate of the old lineage is rejected and a retrain is requested. `teacher_change="fallback"` (default) sends all traffic to the teacher until a student of the new model passes shadow, which is what the contract requires. `"audit"` keeps serving with a raised audit rate and lets the drift monitor decide, which is cheaper when a whole fleet's alias moves at once;
- the lineage survives restarts (it is the model of the latest teacher-labelled row), and a production version records the lineage it was trained on.

After a teacher change or a confirmed drift (§7.9), the loop has no production student *for the current data*. It then trains like a first student: once the readiness thresholds are met on the current data, not merely on request.

### 7.13 Training execution

Training never runs on a caller's thread. `Config.training`:

- `background` (default): after each batch, the task's maintenance worker is signalled. At most one pass per `maintenance_interval_s` judges the shadow, starts training if due, and checks drift.
- `inline`: maintenance runs inside `classify_batch`. It is deterministic, which replays and tests need, because a replay pushes weeks of traffic through in seconds.
- `manual`: only `maintain()` / `train_now()`.

A training job (`training.run_fit_job`) is self-contained. It reads the task's samples from a read-only connection, fits the head, the OOD reference and the policy, scores production on the same calibration rows, and writes the version files into a staging directory. The registry then `adopt`s that directory atomically. Only small objects cross a process boundary, so with a `train_executor` (a process pool) the serving process does no heavy work for a fit.

With an executor, training is **asynchronous**: maintenance submits the job and keeps judging shadows and checking drift while the job waits for a worker, and a later pass adopts the result. A failed job is recorded (`train_failed`) and retried on a later pass. It never fails a request.

Measured (docs/benchmarks.md): the fit is BLAS-bound, and BLAS uses every core by default, so an uncapped fit slows serving whether it runs in a thread or another process (serving p99 ×5–12 with 5 tasks training). What works is capping training's CPU: `train_threads` (BLAS threads per fit, default 2) and a small, low-priority shared pool (`training.train_pool(workers=2)`). With that, 5 tasks training at once cost serving about ×1.12 at p99, and the fits finish faster than with 5 workers fighting each other.

**`TrainScheduler`** is the shared executor for many tasks. It is round-robin across tenants, so one tenant's hundred tasks can't starve another's one. Within a tenant, the task with the highest recent rate of teacher calls goes first, because training it saves the most. Failed jobs are retried with backoff, a crashed worker pool is replaced, and queued jobs can be cancelled.

Pool workers end themselves when the server process dies. After a `kill -9` or the OOM killer, a worker never sees its job queue close (it holds a write end of the queue itself) and would otherwise live on, holding memory, along with the forkserver and resource tracker (P6.5 found this). Each worker has a watchdog thread that exits within a second of the server's death. A training job interrupted by a crash leaves only a hidden staging directory, removed when the task next loads.

### 7.14 Many tasks: the task manager

`TaskManager` owns every task of a process. A task is identified by what the teacher is asked, never by a caller-chosen name:

```text
key = sha256(tenant, question type, Task.version [, requested model])[:20]     Task.version = hash(instructions, criteria)
```

Matching is exact on purpose: Jev reads its criteria literally, so a "similar" question is not the same question. *(roadmap)* Warm-starting a new task from a close one, for training speed only; never serving with it.

- **Admission.** A new key becomes a task only after `min_requests` (50) requests within `window_s` (24 h). Before that, its requests go to the teacher unrecorded. Services that build questions per request would otherwise create an unbounded number of one-off tasks. `max_tasks_per_tenant` caps the rest.
- **Loading.** Engines load on demand (~4 ms to reopen from disk) and sit in an LRU bounded by `max_loaded` and `max_memory_mb`. Only idle engines are unloaded: no request in flight, no maintenance pass running, no training queued. (A pass that is only *requested* doesn't count: the next load requests another.) A load that pushes past `max_loaded` unloads the least recently used idle engine itself, and the janitor catches up on the rest. With the janitor alone, a steady stream of loads outran it: 505 tasks stayed loaded against a cap of 50.
  - An unloaded engine must be freed at once, by reference counting. Its arrays (student, OOD reference: up to ~10 MB) are invisible to the cycle collector's allocation-count trigger, so an engine kept alive by a reference cycle lingers until a full collection. With more active tasks than `max_loaded`, tasks load and unload many times a second, and memory grew without bound (the P6.6 soak: 555 MB after 4 minutes, still accelerating). The scheduler therefore holds an engine's priority weakly, and training callbacks hold it weakly. `tests/test_manager.py` checks that an unloaded engine is freed with the collector off.
  - Progress survives an unload: the last training point and a shadow's start are kept in the version's metadata, and the manager carries the in-memory counters (audit answers towards the next drift check, the teacher-call rate) to the next load. Without that, a task that was often unloaded never had its shadow judged, retrained on every reload, and never reached a drift check.
  - Size `max_loaded` above the number of active tasks. A loaded task costs a few MB (`jevstiller_student_memory_bytes`), while a reload costs a few ms of disk reads each time.
- **Layout.** `<data_dir>/tasks/<key>/` holds `task.json` (spec, tenant, model, created, last seen), `samples.sqlite` and `versions/`. `idle_ttl_s` (off by default) deletes unused tasks; `delete(key)` removes one.
- **Process-level settings** the manager applies, each found by measurement:
  - BLAS is capped at 1 thread for serving: small matrix products from many threads otherwise oversubscribe the CPU (2.4× throughput, p99 80 → 13 ms at 50 tasks).
  - glibc is told to return freed memory: arrays freed by unloaded tasks otherwise stay resident, and RSS climbed to 1.7 GB with 0.2 GB live.
  - glibc is limited to 2 heap arenas: every load starts threads and SQLite connections, and memory freed across many per-thread arenas stayed resident (~47 KB per load/unload, with Python's own allocations flat). With 2 arenas it's ~9 KB, with serving latency unchanged, because the GIL serialises most allocation anyway.

### 7.15 The proxy

`jevstiller serve` (Starlette + uvicorn + httpx, one process) implements Jev's `POST /v1/systemone` and forwards every other path. Behaviour, headers, key handling and errors are specified in [docs/proxy.md](docs/proxy.md). The design decisions:

- **All or nothing per request.** A request is answered entirely locally or entirely by Jev, and Jev's response is returned unchanged. That keeps the proxy exactly as correct as Jev whenever it forwards, and it never has to guess whether Jev answers questions independently. *(roadmap)* Forwarding only the questions the students can't answer.
- **Keys must be proven.** A key counts as accepted only after Jev answered a `/v1/systemone` request made with it. Requests with an unaccepted key are forwarded first, and recorded only after Jev's successful answer, so a caller without a working key can't get local answers or create tasks. A 401/403 revokes at once. Keys are held as salted hashes in memory only. (Security audit, 2026-09-24: docs/security.md.)
- **Bounded work per request.** At most `max_questions` distinct questions are routed; duplicate questions are routed once; the state is encoded once per request, and only if one of its questions is a task (a request that is only forwarded is never encoded); the body limit is enforced while streaming.
- **Never much slower than Jev.** The shared encoder is the local path's ceiling (bge-small on 16 CPU cores: ~150–340 texts/s depending on text length). Beyond it, requests used to queue without bound: in the load test, 15 s p99 at 256 callers, and throughput halved as the machine thrashed. Now a request goes to Jev as it is whenever a local answer would wait longer than `max_encoder_wait_ms` (200). The estimate counts everything ahead of it: requests already routing (waiting for a worker thread) and texts in the encoder's queue, times the measured time per text. Jev is the overflow. Answers stay correct, and only Jev usage rises.
- **Many small upstream pools.** httpcore's connection pool does work per connection for every waiting request, so one pool of 256 connections forwarded only ~90 req/s at 256 callers. Upstream connections are split over clients of 16, and each request takes the least busy: ~610 req/s through the proxy, against ~880 for Jev's own ceiling at 290 ms per answer.
- **The requested model is part of the task key**, and the local response reports the concrete model version the student was trained against.
- **Fail open to Jev.** Any error in the proxy's own logic forwards the request instead.
- **Split engine API.** `Jevstiller.route()` decides, the proxy makes the upstream call itself (asynchronously), and `complete()` records. `classify_batch` is `route` + teacher + `complete`.

---

## 8. Lifecycle of a task, end to end

```text
1.  Task created. mode = teacher_only. Every request -> Jev, logged with embedding.
2.  Thresholds met -> train student:v1 on train split, fit policy on calib split.
3.  student:v1 -> shadow. Live traffic still 100% Jev; student's answers logged.
4.  After ≥ shadow_min_samples: compute shadow agreement bound.
      fail -> reject, back to 1 with more data.
      pass -> promote; mode = cascade.
5.  Steady state: audit 2% -> Jev; confident -> student; low-confidence/OOD -> Jev.
    All Jev answers enter the store (audit -> train|calib by hash; deferred -> train only).
6.  Retrain on trigger -> candidate -> shadow -> promote if better.
7.  Drift: audit_rate up -> retrain -> promote, or teacher_only if the contract cannot be met.
8.  Repeat.
```

---

## 9. Configuration

Everything is a parameter with a default; nothing is hard-coded. The complete, current list (engine `Config`, `TaskManager`, `ProxySettings`, CLI flags and environment variables) is in [docs/configuration.md](docs/configuration.md).

Parameters that v2 proposed but that do not exist yet, all *(roadmap)*: `retrain_interval` (time-based retrain), `drift_fallback_multiplier` (a fallback-rate alarm), `student_arch` (the MLP head), `hedge` mode. The teacher's `model`, `rpm` and `price_per_mtok` are `JevTeacher` constructor arguments, not `Config` fields; the encoder tier, backend and device are `load_encoder` arguments or CLI flags.

---

## 10. Observability

**Every request** produces one structured log record (the sample-store row of §7.2 doubles as the log). From it, per task, over any window:

```text
Task: support_router                 mode: cascade      production: student:v4

Requests (30d)          1,241,493
  served by student     1,203,241   (96.9%)
  served by Jev            38,252   ( 3.1%)
     audit                24,830
     low confidence        9,912
     out of distribution   3,510

Jev calls avoided       1,203,241   = 16.7 h of Jev rate-limit time @1,200/min
                                      ≈ $3.03 at recorded token usage
Peak throughput (1m)    412 req/s   (Jev ceiling: 20 req/s)
Latency p50 / p99       6 ms / 520 ms   (Jev alone: 350 ms / 900 ms)

Agreement with Jev (audit channel, n = 24,830)
  point estimate        98.9%
  lower bound (95%)     98.7%        target 98.0%   ✓

Fallback rate           3.1%   baseline 2.9%   ✓
OOD rate                0.28%  baseline 0.25%  ✓
Class mix shift (JSD)   0.004                  ✓

Training samples        48,203     calibration samples  9,640
Last promotion          student:v3 -> student:v4   (12 days ago)
```

*That report is the target.* Today `status().report()` shows the mode, the production and shadow versions, the student/teacher shares and channels, teacher calls avoided and their cost, the audit agreement with its interval and OK / inconclusive / BROKEN, labelled counts and teacher lineage, what a first student is waiting for, the routing policy, and the last events. The proxy's `GET /healthz` has request counters. *(Phase 5)* Prometheus metrics and structured request logs.

Always shown next to the agreement number, verbatim: *"Agreement with Jev is not accuracy. If Jev is wrong, the student is wrong the same way."*

The one chart that matters is Jev usage over time — it should go down, and every bump should be explainable by a drift event:

```text
Jev share of traffic
100% |*
     | *
 75% |  *
     |   *
 50% |    *
     |      *
 25% |         *
     |              *        *
  0% +--------------------*-----*--------
                          ^ drift, retrain
```

Also: agreement bound over time with the target as a horizontal line; per-version shadow results; per-class agreement on the audit channel (to see *where* the student disagrees).

**"Is it worth it?"** is a first-class status. If, after `N` days, projected savings do not exceed the audit cost plus student hosting, say so.

---

## 11. Latency

`cascade` improves p50 dramatically (student inference is single-digit ms on CPU) but a deferred request costs student + Jev — **worse than Jev alone at p99**. Measured: the proxy adds ~3 ms to a forwarded request (the encoder's per-text cost comes on top with a real encoder), against Jev's ~290 ms p50 / ~760 ms p99. State this in the docs and the dashboard. For latency-SLO tasks, *(roadmap)* `hedge`: fire both, answer from the student if it clears the threshold, otherwise wait for Jev. Hedge gives up nothing on quality and nothing on p99, at the cost of Jev calls on deferred requests only (which were going to Jev anyway) plus wasted Jev calls when the student turns out to be confident — configurable by cancelling the Jev request when the student answers.

---

## 12. Engineering principles

- **Python ≥ 3.10**, typed, one package: `jevstiller`.
- **CPU is the default target; GPU is a first-class option, not an afterthought.** The only heavy compute is the encoder. On CPU it runs through ONNX Runtime; on NVIDIA through `onnxruntime-gpu` or the torch backend — selected by `device: auto|cpu|cuda`. The head (train and infer) is numpy on CPU and is fast enough there; a torch head on GPU is only used for the optional MLP. Both paths are exercised in CI (CPU always; GPU when a runner has one).
- **Minimal required dependencies:** `numpy` and `threadpoolctl` (to cap BLAS threads), plus stdlib `sqlite3`. Extras: `[jev]` (`typesafe-sdk`), `[onnx]` / `[gpu]` (ONNX Runtime encoders), `[torch]` (torch + transformers encoders), `[server]` (Starlette, uvicorn, httpx: the proxy), `[dev]` (tests, lint, and `jev` + `server` for the contract tests).
- **Deterministic.** Seeded training, hash-based splits, pinned encoder by content hash, pinned teacher model version, immutable student versions. Same store + same config ⇒ same student.
- **Offline-capable.** Once a student is promoted, the task keeps answering confident requests with no network. Audit and deferred requests need Jev: today the library raises `TeacherError` for them and the proxy returns 502/504. *(roadmap)* Queueing audit calls while Jev is unreachable (the student answers; the audit record is drained later), and an optional degraded mode that serves the student below threshold, flagged, when Jev is down.
- **Single directory per task**: `<data_dir>/<task>/` in the library, `<data_dir>/tasks/<key>/` under the task manager. It holds the SQLite store, the versions and (manager) `task.json`. Copy the directory and the task moves with it; `js.export(version)` produces a standalone inference bundle (head + OOD reference + policy + task and encoder identity) with no store.
- **Library and service share one core.** The proxy is a thin layer over the task manager, which is a thin layer over the per-task loop, so the loop can still be embedded in a worker, a batch job or a notebook.
- **Crash-safe state.** Registry writes are atomic (temp file, fsync, rename); a version directory is staged and moved in whole; leftovers of an interrupted save are removed on the next open. A test kills a writer with SIGKILL mid-save and checks that every listed version loads.
- **Tests never call a real teacher by default.** `SyntheticTeacher` drives the full loop (bootstrap, train, calibrate, shadow, promote, drift, fall back) in seconds, on CPU. Proxy contract tests drive the unmodified `typesafe-sdk` over HTTP against a fake Jev and against responses recorded from live Jev. Live tests are opt-in (`JEVSTILLER_LIVE=1 pytest -m live`).
- **Secrets.** In the library the Jev key is read from `TYPESAFE_API_KEY` (or a gitignored `.env`). In the proxy it is the caller's, forwarded per request and held only as a salted hash. It is never stored in a task directory, never logged, never in a sample-store row.

---

## 13. API

### Python: one task

```python
js = Jevstiller(task, teacher, data_dir=..., encoder=None, config=Config(), train_executor=None)

js.classify(state, teacher=None) -> Result(label, probs, confidence, source, routing_reason, latency_ms, error)
js.classify_batch(states, errors="raise" | "return", teacher=None) -> list[Result]   # TeacherError on failures
routed = js.route(states); js.defer(routed, i, reason); js.complete(routed, teacher_outputs)
js.status() -> Status  ;  js.status().report()
js.set_mode("teacher_only" | "cascade" | "auto")
js.train_now() ; js.maintain() ; js.drain() ; js.promote("student:v3") ; js.rollback()
js.versions() ; js.export(path, version=None) ; js.evaluate(states, teacher_labels)
js.training_priority() ; js.busy() ; js.footprint_bytes() ; js.close()
```

### Python: many tasks

```python
m = TaskManager(data_dir, teacher, BatchingEncoder(load_encoder("small")), Config(),
                train_executor=TrainScheduler(workers=2))
m.classify(tenant, instructions, classes, states, model=None, teacher=None) -> list[Result]
routing = m.route(tenant, instructions, classes, states, model); m.complete(routing, teacher_outputs)
m.resolve(tenant, instructions, classes, model) -> (key, Task) ; m.tasks(tenant=None) ; m.engine(key)
m.delete(key) ; m.sweep() ; m.close()
```

### HTTP: the proxy

Jev's own API: `POST /v1/systemone` (answered locally or forwarded) and every other path (forwarded). Served locally, never forwarded:
- `GET /healthz` and `GET /readyz`;
- `GET /metrics` (Prometheus);
- the admin API under `/jevstiller/v1/`: tasks, status, versions, mode, target, train, promote, rollback, delete a task or tenant, stats.

See [docs/proxy.md](docs/proxy.md) and [docs/operations.md](docs/operations.md).

---

## 14. Implementation status (2026-09-24)

v2's MVP list was built in full, except the `shadow` *mode* (shadow exists as a lifecycle stage). Beyond it, and in the order of [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md):

| Area | Status |
|---|---|
| Concurrency: routing lock only around state snapshots, write-behind store, crash-safe registry, per-item teacher failures | done (P1) |
| Training off the request path; async with a shared, low-priority process pool; BLAS cap | done (P1.2–P1.3) |
| JSON tasks and states; label-order safety; teacher lineage; rare classes | done (P1.6–P1.9) |
| Task manager, admission, LRU/memory limits, scheduler, shared batching encoder, OOD cap | done (P2) |
| Drop-in proxy, key verification, tenancy (shared / per key), contract tests with the real SDK | done (P3, P4.1–P4.2, P6.1) |
| Live Jev validation | done: see §15.5 (P0) |
| Access controls, tenancy map, data retention, TLS; security audit (12 findings fixed) | done (P4) |
| Config file, metrics, JSON logs, admin API and CLI, Docker/compose/Kubernetes, readiness, backup/restore | done (P5; status page deferred) |
| `hedge` mode, MLP head, per-class thresholds, class-list changes, multilabel, `noul`/`score` distillation, multiple teachers, Postgres | roadmap (§16) |

---

## 15. Validation experiment

Do this before building the platform polish. It is the same loop as production, replayed over fixed datasets with their original labels hidden.

### 15.1 The story the experiment should tell

A customer-support inbox uses Jev for intent routing (this is TypeSafe's own showcase pattern). Traffic grows. Jev caps out at 1,200 requests/min, every request costs ~350 ms, and the support tool now depends on an early-access API. Jevstiller is dropped in front of Jev. The demo shows three curves as traffic is replayed:

1. **Jev share of traffic** falling from 100% toward the audit floor;
2. **wall-clock time and requests/s** — Jev-only vs. Jevstiller — with the rate-limit ceiling drawn as a line;
3. **measured agreement with Jev** (lower confidence bound) sitting above the target the whole time.

If those three hold, the tool's importance is self-evident: same answers as Jev, at 20–100× the throughput, with an independently verified bound.

### 15.2 Datasets

| dataset | rows | classes | why |
|---|---|---|---|
| **Banking77** (PolyAI) — headline | 13,083 | 77 banking intents | Realistic customer messages, exactly the intent-routing use case; 77 classes stresses both Jev (long `criteria`) and the student. CC-BY-4.0. Load from the PolyAI `task-specific-datasets` CSVs (the HF loader script is deprecated). |
| **CLINC150 (`plus`)** — OOD test | 22,500 + 1,200 OOS | 150 intents + `oos` | Has a labelled *out-of-scope* class: the OOD gate (§7.5) should route OOS to Jev at a much higher rate than in-scope inputs. Verify the current HF dataset id before use. |
| **AG News** — scale | 127,600 | 4 topics | Volume. Lets the coverage-vs-examples curve run to 100k+ and gives a burst test big enough to show the rate-limit wall (100k rows = 83 min through Jev). |

Jev cost for all three, end to end, at ~40–60 tokens/row: **under $1**. The binding constraint is the rate limit: ~2.3 hours of Jev time total, cached once via `ReplayTeacher` and then free forever.

### 15.3 Procedure

1. **Profile the teacher first.** On 500 random rows per dataset, record Jev's confidence distribution and the share of rows below 0.6 (Jev's own documented "route to a human" floor). That share is inherently ambiguous input under the task's own definitions; it bounds how much traffic *any* student can take confidently, and the tool should compute it during onboarding. (Jev is deterministic per request, so self-consistency sampling does not apply; the distribution shape is the ambiguity signal.)
2. **Replay** each dataset in a fixed shuffled order through Jevstiller with `mode=auto`, real `JevTeacher` on the first pass, `ReplayTeacher` thereafter.
3. **Checkpoints** at 500, 1k, 2k, 5k, 10k, 20k, 50k, 100k examples (as available): train, calibrate, and report on the calib split — coverage at the budget, selective disagreement with its bound, OOD rate, per-class agreement.
4. **Arms**, run over the same replay:
   - label target: `probs` vs `hard`;
   - encoder tier: `small` vs `base` vs `large` (§7.3), CPU and GPU timings recorded;
   - head: `linear` vs `mlp`;
   - budget: `target_agreement` ∈ {0.95, 0.98, 0.99}.
5. **Guarantee check.** With the final policy fixed, replay a held-out tail of traffic and compare realised system agreement against the bound that was promised. This is the one number that says whether §5 is real.
6. **OOD check** (CLINC150): deferral rate on `oos` rows vs in-scope rows, with and without the kNN gate.
7. **Burst test** (AG News): 100k requests through Jev-only (respecting the rate limit) vs. through the trained Jevstiller; report wall-clock, req/s, Jev calls, agreement.

### 15.4 Output

One report (markdown + PNG plots, committed under `experiments/`), per dataset:

```text
examples   coverage@98%   Jev share   selective disagreement (ub)   OOD deferrals
     500        ??%           ??%               ??%                    ??%
   1,000        ??%           ??%               ??%                    ??%
   5,000        ??%           ??%               ??%                    ??%
  10,000        ??%           ??%               ??%                    ??%
 100,000        ??%           ??%               ??%                    ??%
```

plus the three curves of §15.1, the encoder-tier comparison table (coverage and ms/text per tier), and the guarantee-check result.

**Research use of the hidden labels (not a product feature):** compare Jev-vs-truth and student-vs-truth on the same rows. Banking77 has published fine-tuned baselines (~93% accuracy); knowing where Jev lands zero-shot, and whether the distilled student lands in the same place, is interesting context for §2's "agreement is not accuracy" — and a story if the student ever lands *above* Jev.

**Questions the experiment answers:**

1. How many examples until the student can take half the traffic within budget? All of it minus the audit floor?
2. How much do `probs` targets reduce that number vs `hard`?
3. Does the Clopper–Pearson bound hold on replayed traffic (is the guarantee real)?
4. Does a bigger encoder buy coverage per example, and is it worth its per-request cost?
5. What does the teacher's ambiguity ceiling look like on a realistic task?
6. Does the kNN OOD gate catch out-of-scope inputs?

---

### 15.5 Result against live Jev (2026-09-24)

Banking77, 11,083 messages replayed through the real Jev (`jev-1.13.0`), bge-small on CPU, target 98%. Full table in [docs/benchmarks.md](docs/benchmarks.md#banking77-against-live-jev).

- **The guarantee held.** Held-out system agreement 99.40% against a 98% target; the live audit channel measured 99.27% with a 95% interval of [98.13%, 99.80%].
- **The student took over most traffic.** It answered 70.6% of held-out messages, and ~65% of live traffic from message 5,000 on. Jev was called 5,270 times for 11,083 messages.
- **Accuracy was preserved.** Against the dataset's true labels, Jev scores 78.55% and the combined system 78.7%. That answers §15.4's research question for this task: the distilled system lands where Jev does, and agreement with Jev remains the only thing the product claims.
- **Cost of the teacher.** Jev billed ~1,700 input tokens per call for this 77-class question (the class descriptions are sent every time), so the $3-per-million estimate of v2 is off by more than 20× for large taxonomies (§3).

Questions still open from §15.4: `probs` vs `hard` targets, encoder tiers and the OOD check (CLINC150), all against live Jev rather than the oracle.

## 16. Roadmap and open questions

- **Class-list changes.** Real taxonomies change within months. Adding a class invalidates the student, the calibration, and the baselines. Plan: new class starts at 100% Jev (per-class teacher-only override), existing head warm-starts with a new output row, calibration refits once the new class has `min_samples_per_class`. Removing/merging classes is a relabel-by-mapping in the store.
- **Class-weighted budget.** Some disagreements are more expensive than others.
- **Per-class thresholds.** Better coverage on tasks with one noisy class.
- **Multiple teachers**, a cheaper teacher as an intermediate tier, or Jev as a *second* opinion on the student's near-threshold cases only.
- **Beyond classification.** The loop generalises to any repeated task with a checkable output (extraction, scoring, short structured generation); only the student and the agreement metric change. That is the long-term direction — automatic specialisation of repeated general-model workloads — and it is deliberately not in scope now.
- **Name.** The product is teacher-agnostic; the name is teacher-specific. Fine for now; worth revisiting before anything public.
- **Partial forwarding.** Send Jev only the questions the students can't answer. It's cheaper, but needs evidence that Jev answers each question independently (Appendix A: "no structural invariants").
- **`noul` and `score` questions.** Yes/no is a two-class task; score is ordinal. Only the student and the agreement metric change.
- **Offline queueing and degraded mode** (§12).
- **Re-encoding on encoder change** (§7.3).
- **Warm-starting a new task from a close one**, for training speed only (§7.14).
- **Postgres store and several replicas** (§7.2).

---

## 17. Glossary

- **Teacher** — the general model (Jev) whose behaviour is being reproduced.
- **Student** — the small task-specific model.
- **Agreement** — student label equals teacher label. *Not accuracy.*
- **Coverage** `c` — fraction of traffic the student answers.
- **Selective disagreement** `e` — disagreement rate among the requests the student answers.
- **Budget** `β = 1 − target_agreement` — the disagreement the user permits, per request, system-wide.
- **Audit channel** — the fixed random slice always sent to the teacher; the only unbiased view of production.
- **Routing policy** — the thresholds (confidence, OOD) fitted for a specific student version.
- **Shadow** — a candidate running on live traffic without its answers being returned.
- **Lineage** — the teacher model a task's rows, calibration and students belong to (§7.12).
- **Task key** — the hash identifying a task from the outside: tenant, question type, `Task.version`, requested model (§7.14).
- **Admission** — the point where a question seen often enough becomes a task (§7.14).
- **Tenant** — who a task belongs to: the whole deployment (`shared`) or one API key (`per_key`).

---

## Appendix A. Jev reference (verified 2026-09-24)

Facts the design depends on. Checked against `typesafe-sdk` 0.7.1's source and the live API on 2026-09-24, except where marked. Recorded responses are in `tests/fixtures/jev/`.

**API.** `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`, JSON body `{state, model, questions}`. `state` is a string, object or array. `questions` is a map of typed questions: `choice` (`instructions`, `criteria`: name → description, max 255), `noul` (yes/no), `score` (ordinal `criteria` list). `instructions` and descriptions may be text, objects or arrays. Response: `{"model", "answers": {"<id>": {"type":"choice","choice","confidence","probabilities":{...}}}, "usage": {"input_tokens","output_tokens"}}`, with header `x-typesafe-request-id` (`req_…`). `model` is the resolved version (`jev-latest` → `jev-1.13.0`). Errors are `{"detail": ...}`: 401 (bad key), 422 (validation, `detail` is a list), 429 (rate limit, `retry-after`), 5xx/529 (overloaded). `GET /v1/models` lists `jev-latest` and `jev-preview`.

**SDK.** `pip install typesafe-sdk` (0.7.1, Python ≥ 3.10, built on `httpx2`). `TypeSafeClient` / `AsyncTypeSafeClient(api_key=None, model=None, retry=None, timeout=None, headers=None, transport=None, http_client=None, base_url=None)`. The base URL comes from `base_url=` or `TYPESAFE_BASE_URL` (default `https://api.typesafe.ai`); this is what makes the proxy drop-in. Default timeout 10 s. Retries 408, 429 and 5xx (default 2 retries), honouring `retry-after` / `retry-after-ms`. `result.request_id` *raises* if the header is missing. Errors map to `TypeSafeAuthenticationError`, `TypeSafeRateLimitError`, `TypeSafeUnprocessableEntityError`, … (`TypeSafeAPIError` with `.status`). Env: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`, `TYPESAFE_LOG_LEVEL`.

**Model.** `jev-1.13.0`, which `jev-latest` resolves to. 64k tokens/request, 32k for state (docs, not verified). Text only; English best. Confidence is the peakedness of the distribution, `(K·max − 1)/(K − 1)`. Distributions can be fully peaked: probabilities of exactly 1.0 and 0.0 were observed.

**Price and limits.** $0.042 / M input tokens; output free (docs). Measured: a 5-class question with a one-sentence message used **400 input tokens**, because instructions and criteria are billed on every call. Measured latency: p50 ~290 ms, p90 ~330 ms, p99 ~760 ms, flat from 1 to 16 concurrent requests; 16 concurrent reached 50 req/s with no 429 in a short burst. Documented limit: 1,200 requests/min and 250k tokens/s per key, "adjust dynamically" (not provoked).

**Documented failure modes** relevant here (docs): literal reading of instructions/criteria (the descriptions are the spec); accuracy falls with large irrelevant state; adversarial content is not treated as hostile; no structural invariants (equivalent questions may not give equivalent answers). Vendor-reported accuracy on its own customer-service eval: 76% — a reminder that agreement with Jev is not accuracy.
