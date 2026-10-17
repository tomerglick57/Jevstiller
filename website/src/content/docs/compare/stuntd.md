---
title: "Jevstiller vs stuntd: two local proxies that learn Jev decisions"
description: "stuntd is the project closest to Jevstiller: a self-hosted Jev-compatible proxy that trains a head per question and checks 2% of live traffic. Where the two differ (the bound, automation, keys, CPU), where stuntd goes further (OpenAI and Anthropic APIs, score questions, no-key mode), and which to pick."
lastUpdated: 2026-10-17
editUrl: "https://github.com/tomerglick57/Jevstiller/edit/main/docs/compare/stuntd.md"
---

<!-- Generated from docs/compare/stuntd.md by website/scripts/sync-docs.mjs. Edit that file instead. -->

[stuntd](https://github.com/bladedevoff/stuntd) is the project closest to Jevstiller. Both are self-hosted proxies that speak Jev's `/v1/systemone` API, record Jev's answers to the typed questions your service already asks, train a small head per question on a frozen encoder, answer locally once the head is good enough, and keep sending a 2% sample to the provider to check. If you are choosing between the two, this page is the comparison, as of 2026-10-17 (stuntd v0.2.1, [its README at commit 75f45c7](https://github.com/bladedevoff/stuntd/blob/75f45c70c3621a7600a48954ede1a8ee89f10b8e/README.md)). Corrections welcome in an issue.

## Side by side

| | Jevstiller 0.5 | stuntd 0.2.1 |
|---|---|---|
| APIs spoken | Jev `/v1/systemone` (an OpenAI-compatible front is on the [roadmap](https://jevstiller.pages.dev/project/roadmap/)) | Jev `/v1/systemone`, OpenAI Decisions API, OpenAI Chat Completions, Anthropic Messages |
| Works without a provider | No. Jev is the teacher; every answer the local model learns from is Jev's | Yes. "Local Jev" mode answers zero-shot from the Laya checkpoint with no key and no training |
| Questions learned | `choice` and `noul` (yes/no); `score` is forwarded | `choice`, `noul` and `score` (2 to 10 levels) |
| Encoder and head | Frozen bge-small (ONNX, CPU) and a logistic-regression head trained on Jev's full probability distribution | Frozen Laya (a GPU is expected; about 60 ms per answer on CPU) or a stock sentence encoder, and a classification head trained on the recorded label |
| How the threshold is chosen | A Clopper–Pearson bound at 95% on disagreement over *all* requests, computed on held-out calibration rows, strictest threshold first; recomputed on every retrain | The confidence at which the head reaches `target_agreement` (0.99 by default) on a 20% holdout; the holdout agreement is reported with a 95% Wilson interval |
| What the number promises | With 95% confidence, at most 2% of all requests (at the default 98% target) are answered locally and differently from Jev | The agreement measured on the holdout at the operating point; the threshold itself is the point estimate |
| Unfamiliar input | kNN out-of-distribution gate (10 neighbours, 99th percentile of the training reference) sends it to Jev | Novelty gate (mean cosine to the 5 nearest training rows, 99th percentile of the holdout) sends it to the provider, since 0.1.3 |
| Checking live traffic | 2% audit for ever, 10% while a candidate is in shadow or the drift monitor is suspicious; falls back to Jev and retrains by itself when agreement drops | 2% `check_share`; agreement over the last 100 decisions, demotion to shadow when it falls below target |
| Automation | Collect, train, shadow, promote, monitor, retrain, all automatic, including when `jev-latest` moves to a new model | `stuntd train` and `stuntd enable` by default; `training.auto_retrain` and `serving.auto_promote` (after 24 h in shadow) are opt-in and close the loop |
| What identifies a question | A hash of its type, instructions, criteria and requested model: the same question text is the same task, whatever it is called | The question's name: two questions with the same name but different criteria train one head on mixed data |
| API keys | Local answers only for keys Jev has accepted; keys are forwarded and never stored; optional per-key tenancy | `jev.require_key` checks that a Bearer token is present, not that it is valid |
| Hardware and deployment | One CPU container (`ghcr.io/tomerglick57/jevstiller`), many tenants and tasks per process; Compose and Kubernetes manifests | A Python daemon; Laya heads are measured on CUDA (p50 22 ms, 60 ms on CPU); CPU/ONNX serving and a container image are on its roadmap |
| Data kept | Per task: request text (optional), embeddings, Jev's answers and distributions in SQLite, with retention controls | Captures in SQLite with built-in email and phone redaction; 100,000 rows or 30 days by default |
| Licence and status | Apache-2.0; alpha; validated against live Jev, five public benchmarks, a 24-hour soak | Apache-2.0; "early release"; four demos measured against their teachers |

## Where stuntd goes further

- **More than Jev.** It sits in front of OpenAI and Anthropic too, and learns the OpenAI Decisions API. Jevstiller is Jev-only today.
- **No key needed.** Its local mode answers from the Laya checkpoint alone. Jevstiller has nothing to say until Jev has answered a few thousand times.
- **`score` questions** are learned; Jevstiller forwards them.
- **Pluggable teachers.** Any `/v1/systemone` server can be its upstream (Kev, Mica, Perplexity Decider, Strom are documented).
- **Redaction before storage**, and `stuntd report --gold` to score a head against rows a person has checked.

## Where Jevstiller goes further

- **The bound.** stuntd picks its threshold from a holdout point estimate and reports an interval next to it. Jevstiller picks the threshold *from* the bound, over all requests rather than among the ones answered, and recomputes it on every retrain. [A bound, not a threshold](https://jevstiller.pages.dev/posts/the-guarantee/) shows a point estimate breaking a 2% budget on many splits where the bound did not.
- **Soft labels.** The head is trained on Jev's whole distribution, not the top label.
- **Automation by default**, including the case where TypeSafe moves `jev-latest` to a new model: traffic falls back to Jev until a student of the new model passes shadow.
- **Keys and tenants.** Local answers only for keys Jev has accepted; many tenants on one server; `confidence_floor` so the guarantee also covers a caller's "was Jev unsure?" check.
- **CPU footprint.** 16 ms median per local answer on CPU, encoder included, in one hardened container.
- **Measured.** Five public tasks against Jev's recorded answers, reproducible without a key; load, chaos and a 24-hour soak in [benchmarks](/project/benchmarks/).

## Which to pick

Pick stuntd if you have no Jev key, if the decisions you want to learn are made through OpenAI or Anthropic rather than Jev, or if you need `score` questions answered locally. Pick Jevstiller if your service already calls Jev, you want a stated bound on how often a local answer differs from Jev's, you want the loop to run itself, and you want to run on CPU next to your services. Both are Apache-2.0 and a few weeks old; read both READMEs and try the one that fits on a copy of your traffic.

Before deploying either in front of the paid Jev API, read [TypeSafe's terms](/project/terms/): both projects train on Jev's output.

## Sources

- stuntd: [README](https://github.com/bladedevoff/stuntd/blob/75f45c70c3621a7600a48954ede1a8ee89f10b8e/README.md) at commit 75f45c7 (v0.2.1, 2026-10-11): "How stuntd distils decisions into a Laya head", "The novelty gate", "Configuration", "Limitations", "Roadmap".
- Jevstiller: [README](/concepts/comparison/), [configuration reference](/reference/configuration/), [the guarantee](/posts/the-guarantee/), [benchmarks](/project/benchmarks/).
