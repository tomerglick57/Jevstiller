# Jevstiller — Design

> Status: draft v2 (2026-09-21). Supersedes the initial concept note.
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
| price | $0.042 / M input tokens, output free → a 60-token message ≈ $0.0000025; **1M messages ≈ $3** | ~0 |
| latency | 70–500 ms end-to-end; ~350 ms median measured independently | single-digit ms on CPU |
| throughput | **1,200 requests/min (20/s), 250k tokens/s**, one state per request; limits "adjust dynamically" | thousands/s on one CPU; more on GPU |
| availability | hosted only; early access; `529 overloaded` exists; no self-hosting | runs wherever the process runs |

So the honest ranking of what Jevstiller buys you:

1. **Throughput beyond the rate limit.** 20 req/s is a hard ceiling of ~1.7M classifications/day per key. A backlog of 100k messages takes 83 minutes through Jev; the student does it in under a minute. Any bursty or high-volume workload hits this wall long before it hits a cost wall.
2. **Latency.** ~350 ms → ~5 ms p50 for the requests the student handles.
3. **Availability and independence.** Keeps answering during Jev outages, 429/529 storms, and network egress restrictions (the audit/deferral stream queues and drains later). Hedges an early-access API whose limits and pricing are explicitly provisional.
4. **Cost** — last. At current prices it only matters at very high volume, and TypeSafe says the price "may be subsidized", so it is a hedge rather than a saving.

It is worth using when volume is high enough to hit (1) or (2), the class list is stable, and the input distribution changes slowly. The bootstrap costs nothing extra — requests were going to Jev anyway, and every Jev answer (with its probability distribution) is training data. The ongoing Jev cost is the audit channel plus deferrals.

Jevstiller should tell the user when it is *not* paying off (§10) rather than hide it.

---

## 4. User experience

Define a task once. Classes carry descriptions because Jev is a literal reader: the `criteria` text *is* the specification, and it is what the student learns to reproduce.

```python
from jevstiller import Task, Jevstiller
from jevstiller.teachers import JevTeacher

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

Changing `instructions` or any class description changes the teacher's behaviour, so it bumps `task_version` and starts a new lineage (old samples are kept but marked; the student retrains from the new version's samples only).

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
                     |        shadow       |
                     |        cascade      |
                     |        hedge        |
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

---

## 7. Components

### 7.1 Teacher adapter

A protocol, not a class hierarchy:

```python
class Teacher(Protocol):
    def classify(self, texts: list[str], task: Task) -> list[TeacherOutput]: ...

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
```

**`JevTeacher`** (the only real adapter in the MVP) wraps `typesafe-sdk`:

- one request per text: `state=text` (or the user's JSON object, passed through), `questions={"label": Choice(instructions=task.instructions, criteria=task.classes)}`, `model` pinned to an explicit version (`jev-1.13.0`, not `jev-latest` — a silent model change is a teacher change);
- `probs` ← `answers["label"].probabilities`, `label` ← `.choice`, `confidence` ← `.confidence`;
- `cost_usd` ← `usage.input_tokens × price_per_token` from the task config, so savings reporting is exact;
- concurrency via the SDK's async client with `max_concurrency` set just under the rate limit (1,200/min); exponential backoff on 429/529; a local token bucket so bootstrap replays do not thrash the limit;
- state longer than the 32k-token budget is truncated with a logged warning (documented limitation: accuracy falls with large irrelevant state anyway).

**Soft labels are the main sample-efficiency lever and Jev gives them away on every call.** The student trains against `probs`, not `label`. Where Jev is uncertain the target is flat, so the student learns to be uncertain there too — which is exactly what the router needs. The adapter still records a `label_source` field (`probs` | `hard`) so the experiment (§15) can compare training on the distribution vs. the argmax.

A `SyntheticTeacher` (deterministic rules + injectable noise + fake distributions) ships with the package for tests and demos; the end-to-end test suite never calls a real API. A `ReplayTeacher` serves cached Jev answers from the sample store, so experiments can be re-run offline and for free.

### 7.2 Sample store

An append-only log of **every** request, whether served by student or teacher. Default backend: one SQLite file per task (portable, zero-ops, fine into the millions of rows). Backend is an interface; Postgres/object storage later.

Per record:

```text
id, ts, task_version
text, text_hash
embedding (blob, encoder_id)           # stored so retraining never re-encodes
teacher_label, teacher_probs, teacher_confidence, teacher_model,
teacher_input_tokens, teacher_cost_usd, teacher_request_id          # null if not sent to teacher
student_version, student_probs, student_confidence, ood_score           # null if no student ran
served_by (teacher|student), routing_reason
channel  (bootstrap | deferred | audit | shadow)
split    (train | calib)               # assigned at insert by hash(text_hash); fixed forever
latency_ms, teacher_cost
```

Two rules that keep the guarantee valid:

- **Split is assigned once, by hash, at insertion.** A text is always in the same split across every retrain. No leakage, no reshuffling drift.
- **Only `channel ∈ {bootstrap, audit}` records are eligible for `calib`.** Deferred records are informative for *training* (they are exactly the hard cases) but are not IID and must never influence the threshold.

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

Swapping the encoder is a config change; it starts a new lineage (embeddings recomputed from stored text in the background, old students invalid for the new encoder). The store keeps text, so this is always possible.

### 7.4 Student head

Multinomial logistic regression on the embedding, trained with cross-entropy against the teacher's **soft** distribution. Implemented in numpy with a hand-written optimiser so there is no framework dependency on the inference path; an optional small MLP (one hidden layer) behind the same interface for tasks where linear is not enough.

Regularisation is by **early stopping on an internal validation slice** (10% of the training rows, seeded), not by a fixed L2. Measured on Banking77 embeddings: L2 = 1e-4 cost 15–40 points of coverage at the same budget versus L2 ≈ 0, and made larger encoders look *worse* than smaller ones. Fixed hyperparameters are fragile across encoder dimensions and dataset sizes; a validation-selected stopping point is not.

```python
class Student(Protocol):
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray   # (n, n_classes)
```

**Importance weighting.** Once routing starts, teacher-labelled rows are no longer a sample of traffic: every deferred (hard) request is captured, but only `audit_rate` of the easy ones. Training on raw counts drifts the student toward the hard slice and *lowers* its confidence on easy traffic (observed: candidates trained on 5× more data had lower coverage than their predecessor). So each row carries a weight reflecting the number of requests it stands for — `(1/audit_rate)^power` for audit rows (capped), 1 otherwise — and the head is trained with those weights. With `power = 1` this is exact Horvitz–Thompson reweighting; the default `power = 0.5` tempers it, because at a 2% audit rate exact weights of 50 let a few hundred audit rows dominate thousands of deferred ones (effective sample size ≈ 600 out of 5,000, observed as erratic candidate quality). Tempering trades a little bias for a lot of variance.

Class-imbalance handling and a validation split for early stopping / L2 selection are internal; nothing here touches the calibration split.

### 7.5 Out-of-distribution scorer

Softmax confidence is systematically *over*confident on inputs unlike the training data — precisely where deferral matters most. So OOD is a separate gate, not folded into the class probabilities:

- Score = distance to the k nearest training embeddings (k ≈ 10, cosine), or Mahalanobis distance to the nearest class centroid. kNN is the default: no distributional assumption, trivially incremental.
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

Roadmap: per-class thresholds, and a *class-weighted* budget ("a wrong `cancellation` costs five times a wrong `other`").

### 7.7 Router

Modes, per task, switchable at runtime:

| mode | behaviour | when |
|---|---|---|
| `teacher_only` | everything to Jev, everything logged | bootstrap; emergency fallback |
| `shadow` | everything to Jev; student also runs and its answer is logged, never returned | evaluating a candidate on live traffic at no quality risk |
| `cascade` | student first; defer to Jev if below threshold or OOD | normal production |
| `hedge` | student and Jev called concurrently; return student if confident, else wait for Jev | latency-sensitive tasks — no p99 penalty, but no Jev savings on deferred requests' latency |

In every mode except `teacher_only`, an **audit sample** (§7.8) is drawn first and sent to Jev unconditionally.

Every decision writes a `routing_reason`: `bootstrap`, `audit`, `confident`, `low_confidence`, `ood`, `shadow`, `fallback`, `hedge_timeout`.

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
broken:      ub(A) < target − margin    -> mode = teacher_only immediately; alert; retrain
recovered:   a candidate passes shadow  -> back to cascade
```

The "broken" test uses the *upper* bound on purpose: it fires only when the data are confident the contract is violated, not merely when a small audit sample is noisy. "Suspicious" uses the point estimate: at n = 500 and a 98% target, the lower bound sits below target even at 1% disagreement, so gating retrains on it would keep the audit rate elevated permanently. Until `drift_min_samples` audit records exist the monitor stays silent. The status report shows the interval and one of `OK` / `inconclusive` / `BROKEN`.

Not yet implemented (roadmap): the elevated audit rate while suspicious; embedding-centroid and class-mix drift signals feeding the same ladder.

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
retrain:          ≥ min_new_samples since last training
             OR   drift monitor requested it
             OR   elapsed time ≥ retrain_interval   (roadmap)
never:            while a candidate is already in shadow
```

Training is idempotent and cheap (frozen encoder), so being trigger-happy is fine; the shadow gate is what prevents bad promotions.

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

Everything below is a parameter with a default; nothing is hard-coded.

| parameter | default | meaning |
|---|---|---|
| `target_agreement` | 0.98 | the contract; budget β = 1 − this |
| `confidence` | 0.95 | 1 − δ for all bounds |
| `audit_rate` | 0.02 | fraction of all traffic sent to Jev unconditionally |
| `audit_rate_shadow` | 0.10 | audit rate while a candidate is in shadow |
| `audit_rate_elevated` | 0.10 | audit rate while drift is suspected |
| `calib_fraction` | 0.20 | share of eligible (bootstrap/audit) samples hashed into `calib` |
| `min_train_samples` | 1000 | first training gate |
| `min_samples_per_class` | 50 | first training gate |
| `min_calib_samples` | 500 | do not fit a policy on less |
| `min_new_samples` | 2000 | retrain trigger |
| `retrain_interval` | 7d | retrain trigger |
| `shadow_min_samples` | 1000 | requests a candidate runs in shadow before it is judged |
| `ood_quantile` | 0.99 | OOD threshold on calib distances |
| `ood_k` | 10 | kNN size |
| `drift_fallback_multiplier` | 2.0 | fallback-rate alarm relative to baseline |
| `drift_margin` | 0.0 | hard fallback when `ub(A) < A* − margin` |
| `drift_window` | 500 | audit records in the rolling check |
| `drift_min_samples` | 200 | monitor is silent below this |
| `encoder` | `base` (`bge-base-en-v1.5`) | pinned by hash; `small` for CPU-only deployments |
| `encoder_backend` | `onnx` | or `torch` |
| `student_arch` | `linear` | or `mlp` |
| `label_target` | `probs` | train on the teacher's distribution, or `hard` (argmax) |
| `importance_weighting` | true | audit rows weigh `1/audit_rate` in training |
| `fit_headroom` | 0.15 | fit policies at 85% of budget; judge at 100% |
| `teacher_model` | `jev-1.13.0` | pinned; never an alias |
| `teacher_rpm` | 1100 | local rate limiter, under Jev's 1,200/min |
| `teacher_price_per_mtok` | 0.042 | for savings reporting |
| `mode` | `auto` | or force `teacher_only` \| `shadow` \| `cascade` \| `hedge` |
| `store_text` | true | false keeps only the text hash + embedding in the store |
| `device` | `auto` | `cpu` \| `cuda` \| ... |

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

`cascade` improves p50 dramatically (student inference is single-digit ms on CPU) but a deferred request costs student + Jev — **worse than Jev alone at p99**. State this in the docs and the dashboard. For latency-SLO tasks use `hedge`: fire both, answer from the student if it clears the threshold, otherwise wait for Jev. Hedge gives up nothing on quality and nothing on p99, at the cost of Jev calls on deferred requests only (which were going to Jev anyway) plus wasted Jev calls when the student turns out to be confident — configurable by cancelling the Jev request when the student answers.

---

## 12. Engineering principles

- **Python ≥ 3.10**, typed, one package: `jevstiller`.
- **CPU is the default target; GPU is a first-class option, not an afterthought.** The only heavy compute is the encoder. On CPU it runs through ONNX Runtime; on NVIDIA through `onnxruntime-gpu` or the torch backend — selected by `device: auto|cpu|cuda`. The head (train and infer) is numpy on CPU and is fast enough there; a torch head on GPU is only used for the optional MLP. Both paths are exercised in CI (CPU always; GPU when a runner has one).
- **Minimal required dependencies:** `numpy`, `onnxruntime`, `tokenizers`, stdlib `sqlite3`. Extras: `[jev]` (`typesafe-sdk`), `[gpu]` (`onnxruntime-gpu`), `[torch]` (torch + transformers, for the torch encoder backend, MLP head, and ONNX export), `[server]` (FastAPI), `[dev]`.
- **Deterministic.** Seeded training, hash-based splits, pinned encoder by content hash, pinned teacher model version, immutable student versions. Same store + same config ⇒ same student.
- **Offline-capable.** Once a student is promoted, the task keeps working with no network except for audit/deferral — and those queue if Jev is unreachable (the student answers; the audit record is marked `pending` and drained later). This matches the intended deployment: a server whose only egress is the Jev API.
- **Single directory per task** (`<data_dir>/<task>/`) containing the SQLite store, encoder, versions, and config. Copy the directory and the task moves with it; `js.export(version)` produces a standalone inference bundle (encoder + head + policy) with no store.
- **Library first, service second.** The Python API is the product; the HTTP server is a thin wrapper so the core can be embedded in a worker, a batch job, or a notebook. The CLI report (§10) is the MVP's "dashboard"; a web UI comes later, if at all.
- **Tests never call a real teacher.** `SyntheticTeacher` drives the full loop — bootstrap, train, calibrate, shadow, promote, drift, fall back — in seconds, on CPU. `ReplayTeacher` re-runs recorded Jev answers for reproducible experiments.
- **Secrets.** The Jev key is read from `TYPESAFE_API_KEY` (or a gitignored `.env`); it is never stored in the task directory, never logged, never in a sample-store row.

---

## 13. API

### Python

```python
js = Jevstiller(task, teacher, data_dir=...)

js.classify(text) -> Result(label, confidence, source, routing_reason, latency_ms)
js.classify_batch(texts) -> list[Result]
js.status() -> Status(...)              # everything in §10
js.set_mode("teacher_only" | "shadow" | "cascade" | "hedge" | "auto")
js.train_now() ; js.promote("student:v3") ; js.rollback()
js.versions() -> list[VersionInfo]
js.export(version) -> path             # a self-contained inference bundle
```

### HTTP (optional)

```http
POST /tasks                           create
POST /tasks/{id}/classify             {"text": "..."} -> {"label","confidence","source","routing_reason"}
GET  /tasks/{id}                      status (§10)
GET  /tasks/{id}/versions
POST /tasks/{id}/mode                 {"mode": "teacher_only"}
POST /tasks/{id}/rollback
GET  /tasks/{id}/metrics?window=30d
```

The response shape is identical whether the answer came from Jev or the student; `source` is the only difference.

---

## 14. MVP scope

Build, in this order, and nothing else until the loop is proven:

1. `Task`, `Teacher` protocol, `SyntheticTeacher`, `ReplayTeacher`, `JevTeacher` over `typesafe-sdk`.
2. Sample store (SQLite) with hash splits and channels.
3. Encoder via ONNX Runtime, embedding stored on insert.
4. Linear soft-label student in numpy.
5. kNN OOD scorer.
6. Calibrator with Clopper–Pearson thresholding.
7. Router: `teacher_only`, `shadow`, `cascade`, audit channel.
8. Registry with shadow → production and rollback.
9. Drift monitor with the three-step ladder.
10. `status()` and a text/CLI report; per-request structured logs.
11. The experiment in §15, as a runnable script with a report.

Explicitly out of scope for the MVP: `hedge` mode, HTTP server, dashboard UI, MLP head, per-class thresholds, multilabel, non-text input, multiple teachers, class-list changes, distributed anything.

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

## 16. Roadmap and open questions

- **Class-list changes.** Real taxonomies change within months. Adding a class invalidates the student, the calibration, and the baselines. Plan: new class starts at 100% Jev (per-class teacher-only override), existing head warm-starts with a new output row, calibration refits once the new class has `min_samples_per_class`. Removing/merging classes is a relabel-by-mapping in the store.
- **Class-weighted budget.** Some disagreements are more expensive than others.
- **Per-class thresholds.** Better coverage on tasks with one noisy class.
- **Multiple teachers**, a cheaper teacher as an intermediate tier, or Jev as a *second* opinion on the student's near-threshold cases only.
- **Beyond classification.** The loop generalises to any repeated task with a checkable output (extraction, scoring, short structured generation); only the student and the agreement metric change. That is the long-term direction — automatic specialisation of repeated general-model workloads — and it is deliberately not in scope now.
- **Name.** The product is teacher-agnostic; the name is teacher-specific. Fine for now; worth revisiting before anything public.

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

---

## Appendix A. Jev reference (as of 2026-09-21)

Facts the design depends on, from `docs.typesafe.ai` and independent write-ups. Re-verify before relying on any number.

**API.** `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`, JSON body `{state, model, questions}`. `state` is a string, object, or array of text. `questions` is a map of typed questions; Jevstiller uses one `Choice`: `{"type":"choice","instructions":"...","criteria":{"<class>":"<description>", ...}}`, max 255 options. Response: `{"model", "answers": {"<id>": {"type":"choice","choice","probabilities":{...},"confidence"}}, "usage": {"input_tokens","output_tokens"}}`. Errors: 401, 422, 429 (rate limit), 529 (overloaded) — back off exponentially.

**SDK.** `pip install typesafe-sdk` (Python ≥ 3.10). `TypeSafeClient` / `AsyncTypeSafeClient(api_key=None, base_url=..., model="jev-latest", timeout=None, retry=RetryPolicy(), max_concurrency=None)`. `client.system_one(state, questions)`; `result.choices["label"].choice / .probabilities / .confidence`; `result.request_id`; `TypeSafeAPIError(status, request_id)`. Env: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`.

**Model.** `jev-1.13.0` (`jev-latest` and `jev-preview` alias it). 64k tokens/request, 32k for state. Text only; English best. "Not trained on customer requests or responses." Confidence is a statistic of the distribution's peakedness (for 3 options: `(3·max − 1)/2`).

**Price and limits.** $0.042 / M input tokens; output free. 250k tokens/s and 1,200 requests/min per key; limits "adjust dynamically" and may change without notice; pricing "may be subsidized". Hosted only, early access. Latency 70–500 ms claimed; ~0.35 s median measured independently.

**Documented failure modes** relevant here: literal reading of instructions/criteria (the descriptions are the spec); accuracy falls with large irrelevant state; adversarial content is not treated as hostile; no structural invariants (equivalent questions may not give equivalent answers). Vendor-reported accuracy on its own customer-service eval: 76% — a reminder that agreement with Jev is not accuracy.

**Overlap with Jevstiller.** TypeSafe's "confidence-gated routing" pattern uses *Jev's* confidence to decide whether to act on *Jev's* answer. Jevstiller uses the *student's* confidence to decide whether to ask Jev at all. They compose: a user can still gate on the returned confidence (student or Jev) downstream.
