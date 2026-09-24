---
title: How it works
description: The distillation loop — route, record, train, shadow, promote, monitor, fall back.
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

## The loop

```text
request ─► router ─┬─ confident & in-distribution ─► local student ─► answer
                   ├─ uncertain / novel / 2% audit ─► Jev ─► answer (and a new training row)
                   └─ no model yet ─────────────────► Jev
                                        │
                sample store ◄──────────┘
                     │  train (seconds; frozen encoder + small head, soft targets)
                     ▼
                candidate ─► shadow on live traffic ─► passes the budget? ─► production
                                                                  drift? ─► retrain / fall back
```

1. **Collect.** With no model yet, every request goes to Jev. Each answer is stored with Jev's full probability
   distribution and the text's embedding.
2. **Train.** Once there are enough samples per class, Jevstiller fits a student and a routing policy.
3. **Shadow.** The candidate runs next to Jev on live traffic. Its answers are recorded, never returned.
4. **Promote.** If the candidate meets the budget on the shadow traffic, it goes to production.
5. **Serve.** Confident, familiar inputs are answered locally. Everything else goes to Jev.
6. **Monitor.** A random audit slice keeps going to Jev. If agreement drops, Jevstiller retrains; if it is
   clearly broken, it falls back to Jev entirely.

## The pieces

**Encoder.** A frozen sentence encoder (`bge-small`, `bge-base`, or `bge-large`), run through ONNX Runtime or
PyTorch. Embeddings are stored once, so retraining never re-encodes. Frozen also means a stable coordinate
system for drift and novelty checks.

**Student.** Logistic regression in numpy, trained on Jev's *distribution* rather than its top label. Where Jev
is unsure, the target is flat, so the student learns to be unsure there too, which is exactly what the router
needs. Early stopping on a validation slice picks the regularisation.

**Out-of-distribution gate.** kNN distance in embedding space. An input unlike anything seen goes to Jev,
whatever the student's confidence says. Softmax confidence is overconfident on novel input, which is why this is
a separate gate.

**Calibrator.** Picks the confidence threshold that maximises the share of traffic the student takes while
keeping a statistical bound on disagreement inside your budget. See [the guarantee](/concepts/guarantee/).

**Audit channel.** A fixed random 2% of requests always goes to Jev. It is the only unbiased view of production
once routing starts, and the price of the guarantee. It rises to 10% while a candidate is in shadow or drift is
suspected.

**Registry.** Every student is an immutable `student:vN` directory holding the head, the policy, and the
encoder identity. Promote, roll back, or `export()` a standalone bundle.

## Routing reasons

Every request records why it went where it went:

| reason | meaning |
|---|---|
| `bootstrap` | no production model yet |
| `audit` | drawn for the audit slice |
| `confident` | the student answered |
| `low_confidence` | below the policy's threshold |
| `ood` | too far from anything seen in training |
| `fallback` | teacher-only mode with a production model present: the contract broke, or you set it by hand |

## Task versions

The class descriptions are sent to Jev verbatim. Jev reads them literally, so they *are* the specification.
Changing `instructions` or any description changes the task's version, and the student retrains from the new
version's samples only.

For the full reasoning, including the loop bugs the first replays found, read
[DESIGN.md](https://github.com/tomerglick57/Jevstiller/blob/main/DESIGN.md).
