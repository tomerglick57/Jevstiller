---
title: Getting started
description: Run the proxy, point your services at it, and watch a question move from Jev to the local model.
---

Jevstiller runs as one container next to your services. They keep the TypeSafe SDK and their own Jev keys; only
the base URL changes.

## 1. Run the proxy

```bash
export JEVSTILLER_ADMIN_TOKEN=$(openssl rand -hex 32)      # enables the admin API and /metrics
docker run -d --name jevstiller -p 8080:8080 -v jevstiller-data:/data \
  -e JEVSTILLER_ADMIN_TOKEN ghcr.io/tomerglick57/jevstiller:0.2.0
curl -s localhost:8080/readyz
```

The image runs on CPU with the encoder built in, and needs no network except to Jev. Without Docker:

```bash
pip install "jevstiller[server,onnx]"                # Python 3.10+
jevstiller serve --data-dir ./jevstiller-data
```

Every setting has a working default. For a real deployment (a config file, restricting who may call it, TLS,
Kubernetes), see [Deploy](/proxy/deploy/) and [Configuration](/reference/configuration/).

## 2. Point your services at it

```bash
export TYPESAFE_BASE_URL=http://localhost:8080
```

That's the whole change. Code like this keeps working as it is:

```python
import typesafe_sdk as ts

client = ts.TypeSafeClient()                     # reads TYPESAFE_API_KEY and TYPESAFE_BASE_URL
r = client.system_one(
    state="Please cancel my subscription",
    questions={"team": ts.Choice(
        instructions="Which team should handle this customer message?",
        criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs, errors, things not working",
                  "cancellation": "Wants to cancel or downgrade", "other": "Anything else"})},
)
r.choices["team"].choice          # "cancellation"
```

Every response carries `x-jevstiller-source: upstream` or `local`, so you can see which answered.

## 3. Watch it learn

At first, every request goes to Jev with the caller's key, and Jev's answer comes back unchanged.
- **The question becomes a task** after it has been asked 50 times (`admit_after`); from then on, each Jev answer is a
  training row.
- **A student is trained** in the background once there are enough rows per class. It is tested against
  calibration data and in shadow on live traffic.
- **Local answers start** once the student meets the budget. Requests it isn't sure about still go to Jev, and so
  does a random 2%, to keep checking.

Ask the proxy how each task is doing:

```bash
docker exec jevstiller jevstiller admin tasks                  # every task, with its key
docker exec jevstiller jevstiller admin status <task key>
```

```text
Task: 3f1c0a9e2b7d44e1a9c2   version 89a7438c2f7d   mode: cascade   audit rate 2%
Production: student:v3   Shadow: -
Requests: 11,083   student 69.8%   teacher 30.2%
Agreement with teacher (audit, n=415): 99.40% [98.46%, 99.84%]   target 98%   OK
  note: agreement with the teacher is not accuracy.
```

The agreement line shows the point estimate, its confidence interval, and one of `OK`, `inconclusive`, or
`BROKEN`. See [the guarantee](/concepts/guarantee/) for what those mean. For dashboards, `GET /metrics` serves
Prometheus metrics ([Operations](/proxy/operations/)).

## As a Python library

The engine inside the proxy is also a library, for a single task in your own process:

```python
from jevstiller import Jevstiller, Task, load_encoder
from jevstiller.teachers.jev import JevTeacher      # pip install "jevstiller[jev,onnx]"

task = Task("support_router", "Which team should handle this customer message?",
            {"billing": "Charges, invoices, refunds", "technical": "Bugs, errors, things not working",
             "cancellation": "Wants to cancel or downgrade", "other": "Anything else"},
            target_agreement=0.98)
js = Jevstiller(task, teacher=JevTeacher(), data_dir="./jevstiller-data", encoder=load_encoder("small"))

r = js.classify("Please cancel my subscription")
r.label, r.source      # ("cancellation", "teacher") at first, later "student:v3"
print(js.status().report())
```

To try the loop with no API key, a synthetic teacher stands in for Jev:

```bash
git clone https://github.com/tomerglick57/Jevstiller && cd Jevstiller
pip install -e . && python examples/quickstart_synthetic.py
```

## Next

- [How it works](/concepts/how-it-works/): the loop, piece by piece.
- [The drop-in proxy](/proxy/overview/): what is answered locally, keys, tenancy, errors and limits.
- [When to use it](/concepts/when-to-use/), and when not to.
