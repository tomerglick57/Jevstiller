---
title: How it works
description: The distillation loop (route, record, train, shadow, promote, monitor, fall back) and how the proxy wraps it.
---

Jevstiller behaves like a cache in front of Jev:

| a cache | Jevstiller |
|---|---|
| transparent to the caller | same call in, same shape out |
| warms up from traffic | learns from traffic |
| serves hits locally, misses go to origin | serves confident inputs locally, uncertain ones go to Jev |
| invalidates on change | detects drift and falls back to Jev |
| exact | **approximate, with a bounded disagreement rate** |

The last row is the whole design problem.

## From a request to a task

The proxy speaks Jev's API. For `POST /v1/systemone`, each `choice` question is identified by its exact content
(instructions and criteria), the caller's tenant, and the requested model. That identity is a **task**: one
student, one sample store, one lifecycle. Services asking the same question share it.

A request is answered locally only if every question in it can be; otherwise the whole request goes to Jev with the
caller's own key, Jev's response is returned unchanged, and each choice answer becomes a training row. A key is
trusted for local answers only after Jev has answered a request made with it. [The drop-in proxy](/proxy/overview/)
has the details.

## The loop, per task

```text
request ─► router ─┬─ confident & in-distribution ─► local student ─► answer
                   ├─ uncertain / novel / 2% audit ─► Jev ─► answer (and a new training row)
                   └─ no student yet ───────────────► Jev
                                        │
                sample store ◄──────────┘
                     │  train in the background (frozen encoder + small head, soft targets)
                     ▼
                candidate ─► shadow on live traffic ─► passes the budget? ─► production
                                                     audit breaks it? ─► fall back to Jev, retrain
```

1. **Collect.** With no student yet, every request goes to Jev. Each answer is stored with Jev's full probability
   distribution and the text's embedding.
2. **Train.** Once there are enough samples per class, a background worker fits a student and a routing policy.
   Serving never waits for training.
3. **Shadow.** The candidate runs next to production on live traffic. Its answers are recorded, never returned.
4. **Promote.** If the candidate meets the budget on the shadow traffic, it goes to production.
5. **Serve.** Confident, familiar inputs are answered locally. Everything else goes to Jev.
6. **Monitor.** A random audit slice keeps going to Jev. If agreement dips, Jevstiller raises the audit rate and
   retrains; if it is clearly broken, the task falls back to Jev and retrains on data from after the break.

## The pieces

**Encoder.** A frozen sentence encoder (`bge-small` by default; `base` and `large` too), run through ONNX Runtime
or PyTorch and shared by every task. Embeddings are stored once, so retraining never re-encodes. Frozen also means
a stable coordinate system for drift and novelty checks.

**Student.** Logistic regression in numpy, trained on Jev's *distribution* rather than its top label. Where Jev
is unsure, the target is flat, so the student learns to be unsure there too, which is exactly what the router
needs. Early stopping on a validation slice picks the regularisation.

**Out-of-distribution gate.** kNN distance in embedding space. An input unlike anything seen goes to Jev,
whatever the student's confidence says. Softmax confidence is overconfident on novel input, which is why this is
a separate gate.

**Calibrator.** Picks the loosest confidence threshold that keeps a statistical bound on
"answered and disagreeing" inside your budget. See [the guarantee](/concepts/guarantee/).

**Audit channel.** A fixed random 2% of requests always goes to Jev. It is the only unbiased view of production
once routing starts, and the price of the guarantee. It rises to 10% while a candidate is in shadow or drift is
suspected.

**Teacher lineage.** Every answer records the Jev model that gave it. When `jev-latest` moves to a new version,
the task starts over on the new model's answers, and Jev answers until a student trained on them passes.

**Task manager.** One process serves many tasks: engines load on demand, unload when idle, and share the encoder
and a training pool that is fair across tenants.

**Registry.** Every student is an immutable `student:vN` directory holding the head, the policy, and the encoder
identity. Promote, roll back, or `export()` a standalone bundle.

## Routing reasons

Every request records why it went where it went:

| reason | meaning |
|---|---|
| `bootstrap` | no production student yet |
| `audit` | drawn for the audit slice |
| `confident` | the student answered |
| `low_confidence` | below the policy's threshold |
| `ood` | too far from anything seen in training |
| `rare_class` | the student's label is a class with too few samples, always sent to Jev |
| `fallback` | teacher-only with a production student present: the contract broke, or an operator set it |

The proxy adds its own reasons in the `x-jevstiller-detail` header (`not_admitted`, `key_unverified`,
`co_deferred`, …), listed in [the proxy docs](/proxy/overview/).

## Task versions

The class descriptions are sent to Jev verbatim. Jev reads them literally, so they *are* the specification.
Changing the instructions or any description is a new task, and it learns from scratch.

For the full reasoning, including what the live runs and the soak tests changed, read
[DESIGN.md](https://github.com/tomerglick57/Jevstiller/blob/main/DESIGN.md).
