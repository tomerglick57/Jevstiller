# Jevstiller

**Jev, distilled on the fly.** Put Jevstiller in front of a repeated [Jev](https://docs.typesafe.ai) classification call. At first every request still goes to Jev. From Jev's own answers — with their full probability distributions — it trains a small local model on your traffic, checks that the model agrees with Jev within a budget you set, and then answers most requests itself. Uncertain or novel input, and a permanent random audit slice, keep going to Jev.

Same call. Same answers. Your hardware.

```text
your app ──► Jev                     your app ──► Jevstiller ──► Jev
                                                    │  local model    (only the 3% it isn't sure about,
                                                    ▼                  plus a 2% audit)
                                                 answer
```

## Why

Jev is fast, cheap and typed. It is also a ceiling: **1,200 requests per minute** per key, ~350 ms per answer, hosted only. Jevstiller is for the workload that outgrows that — bursts, backlogs, latency budgets in milliseconds, boxes with no egress, or simply not wanting every classification to depend on one external API.

In a replay of Banking77 (77 intents), the local model took **71% of traffic at 99.5% agreement** with the teacher and ran at ~2,000 rows/s on one GPU — a hundred times Jev's rate limit.¹

## The contract

You set one number. Jevstiller returns the label Jev would have returned on at least that share of requests:

```text
target_agreement = 0.98      →  disagreement budget = 2% of all requests
```

The routing threshold is chosen against a finite-sample upper bound (Clopper–Pearson) on a held-out, IID calibration set — not tuned by eye — and re-verified forever on the audit channel. If the audit shows the contract is broken, everything falls back to Jev automatically.

**Agreement with Jev is not accuracy.** If Jev is wrong, the student is wrong the same way. The status report says so next to every number. See [DESIGN.md §2](DESIGN.md).

## Install

```bash
pip install jevstiller                 # core: numpy only
pip install "jevstiller[jev,onnx]"     # Jev adapter + ONNX Runtime encoders (CPU)
pip install "jevstiller[jev,torch]"    # PyTorch encoders (CUDA if available)
```

Python 3.10+. CPU works out of the box; a GPU only speeds up the encoder.

## Quickstart

No API key needed — a synthetic teacher stands in for Jev:

```bash
python examples/quickstart_synthetic.py
```

With Jev (`TYPESAFE_API_KEY` in your environment):

```python
from jevstiller import Task, Jevstiller, load_encoder
from jevstiller.teachers.jev import JevTeacher

task = Task(
    name="support_router",
    instructions="Which team should handle this customer message?",
    classes={                              # descriptions are sent to Jev verbatim — they are the spec
        "billing":      "Charges, invoices, refunds, payment methods",
        "technical":    "Bugs, errors, integrations, things not working",
        "cancellation": "Wants to cancel, downgrade, or close the account",
        "sales":        "Pre-sales questions, plan comparison, quotes",
        "other":        "Anything that does not fit the categories above",
    },
    target_agreement=0.98,
)

js = Jevstiller(task, teacher=JevTeacher(), data_dir="./jevstiller-data", encoder=load_encoder("base"))

r = js.classify("Please cancel my subscription")
r.label        # "cancellation"
r.confidence   # 0.97
r.source       # "teacher" at first — later "student:v9"

print(js.status().report())
```

```text
Task: support_router   version 89a7438c2f7d   mode: cascade   audit rate 2%
Production: student:v9   Shadow: -
Requests: 11,083   student 69.8%   teacher 30.2%
Agreement with teacher (audit, n=415): 99.40% [98.46%, 99.84%]   target 98%   OK
  note: agreement with the teacher is not accuracy.
```

## How it works

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

- **Encoder**: frozen sentence encoder (`bge-small/base/large`), ONNX or PyTorch. Embeddings are stored, so retraining never re-encodes.
- **Student**: numpy logistic regression on the teacher's *distribution*, early-stopped on a validation slice.
- **OOD gate**: kNN distance in embedding space — unlike anything seen → Jev, whatever the head says.
- **Audit channel**: a fixed random slice always goes to Jev. The only unbiased view of production, and the price of the guarantee.
- **Versions**: immutable `student:vN` directories; promote, roll back, or `export()` a standalone bundle.

The full reasoning — including the five loop bugs the first real replay found and how they were fixed — is in [DESIGN.md](DESIGN.md).

## Status

Alpha (0.1.0). The loop is tested end-to-end (`pytest`, CPU, no network) and validated on replays with a perfect "oracle" teacher. **The benchmark against live Jev is pending**; treat the numbers above as an upper bound until then. Training runs synchronously inside `classify_batch` when a trigger fires — a few seconds on one unlucky request; a background trainer is next.

Not for: tasks with changing class lists (retrain from scratch), non-text input, or volumes too low to ever collect a few thousand examples.

## Experiments

```bash
python experiments/run.py --dataset banking77 --teacher oracle --encoder base   # dry run, no key
python experiments/run.py --dataset banking77 --teacher jev --encoder base      # the real thing
```

See [experiments/README.md](experiments/README.md).

## Contributing

Issues and PRs welcome — especially "I ran it on task X and here's the status report". See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. Jevstiller is an independent project and is not affiliated with, endorsed by, or supported by TypeSafe. "Jev" is their model; this tool only talks to its public API.

---
¹ Oracle-teacher dry run on 11,083 replayed messages with 2,000 held out; `bge-base` on an RTX 3090. Details and caveats in `experiments/README.md`.
