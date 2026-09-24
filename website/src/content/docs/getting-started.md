---
title: Getting started
description: Install Jevstiller, run the loop without an API key, then put it in front of Jev.
---

## Install

Jevstiller needs Python 3.10 or newer. CPU works out of the box; a GPU only speeds up the encoder.

It is not on PyPI yet, so install it from GitHub:

```bash
pip install "jevstiller @ git+https://github.com/tomerglick57/Jevstiller"                 # core: numpy only
pip install "jevstiller[jev,onnx] @ git+https://github.com/tomerglick57/Jevstiller"       # Jev adapter + ONNX encoders (CPU)
pip install "jevstiller[jev,torch] @ git+https://github.com/tomerglick57/Jevstiller"      # PyTorch encoders (CUDA if available)
```

| extra | adds |
|---|---|
| `jev` | `JevTeacher`, via `typesafe-sdk` |
| `onnx` | ONNX Runtime encoders on CPU |
| `gpu` | ONNX Runtime encoders on CUDA |
| `torch` | PyTorch encoders from Hugging Face checkpoints |

## Try it without a key

A synthetic teacher stands in for Jev, so you can watch the whole loop on CPU in under a minute:

```bash
git clone https://github.com/tomerglick57/Jevstiller
cd Jevstiller
pip install -e .
python examples/quickstart_synthetic.py
```

The student starts at 0% of traffic, gets trained and shadowed, gets promoted, and then takes over most
requests while the audit channel keeps checking agreement.

## Put it in front of Jev

Set `TYPESAFE_API_KEY` in your environment, then:

```python
from jevstiller import Task, Jevstiller, load_encoder
from jevstiller.teachers.jev import JevTeacher

task = Task(
    name="support_router",
    instructions="Which team should handle this customer message?",
    classes={                              # descriptions are sent to Jev verbatim: they are the spec
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
r.source       # "teacher" at first, later "student:v9"
```

Early on every request goes to Jev and becomes a training row. Once there is enough data, Jevstiller trains a
student, runs it in shadow, and promotes it only if it meets the budget. You don't call anything to make that
happen.

## Check on it

```python
print(js.status().report())
```

```text
Task: support_router   version 89a7438c2f7d   mode: cascade   audit rate 2%
Production: student:v9   Shadow: -
Requests: 11,083   student 69.8%   teacher 30.2%
Agreement with teacher (audit, n=415): 99.40% [98.46%, 99.84%]   target 98%   OK
  note: agreement with the teacher is not accuracy.
```

The agreement line shows the point estimate, its confidence interval, and one of `OK`, `inconclusive`, or
`BROKEN`. See [the guarantee](/concepts/guarantee/) for what those mean.

## Next

- [How it works](/concepts/how-it-works/): the loop, piece by piece.
- [Configuration](/reference/configuration/): every knob and its default.
- [When to use it](/concepts/when-to-use/): and when not to.
