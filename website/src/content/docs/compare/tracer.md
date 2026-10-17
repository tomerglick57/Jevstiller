---
title: "Jevstiller vs TRACER: two Clopper–Pearson bounds that promise different things"
description: "TRACER is a provider-agnostic Python library that fits a classical-ML student on your LLM traces and certifies it once with a lower bound on agreement among accepted requests. Jevstiller is a Jev proxy that bounds disagreement over all requests and keeps auditing live traffic. What each bound covers, what runs after deployment, and which to pick."
lastUpdated: 2026-10-17
editUrl: "https://github.com/tomerglick57/Jevstiller/edit/main/docs/compare/tracer.md"
---

<!-- Generated from docs/compare/tracer.md by website/scripts/sync-docs.mjs. Edit that file instead. -->

[TRACER](https://github.com/adrida/tracer) (Adam Rida, MIT, [arXiv 2604.14531](https://arxiv.org/abs/2604.14531)) is the project closest to Jevstiller in *method*: both learn a small student from a teacher's labels on frozen embeddings, both defer uncertain and out-of-distribution requests to the teacher, and both use a Clopper–Pearson bound to decide what the student may answer. They differ in form (a library against a proxy), in what the bound covers, and in what happens after deployment. As of 2026-10-17, from [TRACER's README and docs at commit 28de14f](https://github.com/adrida/tracer/blob/28de14fdbccf159993e34fd0defebaa725c982dc/README.md), which describes changes not yet released to PyPI (0.3.0, June 2026). Corrections welcome in an issue.

## Side by side

| | Jevstiller 0.5 | TRACER (main, 2026-09-29) |
|---|---|---|
| Form | A drop-in HTTP proxy for Jev. Services change one environment variable | A Python library (`tracer-llm`) with an optional small HTTP server that takes embeddings. Your application encodes text, calls `predict`, and sends deferred requests to the teacher itself |
| Teacher | Jev | Any: an LLM, a decision model, another classifier |
| Training data | Jev's answers with their full probability distributions, collected by the proxy from live traffic | Teacher-labelled traces you supply as JSONL plus embeddings; `tracer.watch()` records calls, `fit()` is a call you make |
| Student | Multinomial logistic regression on frozen bge-small, trained on the distribution | A search over logistic regression, SGD classifiers, trees, forests, boosted trees and compact MLPs (optional XGBoost), plus a learned acceptance policy and optional residual stages |
| The bound | Clopper–Pearson at 95% on the share of *all* requests answered locally and differently from the teacher, at most 1 − target (2% by default); threshold chosen on held-out calibration rows, strictest first | One-sided Clopper–Pearson lower bound (alpha 0.10, 90%) on teacher agreement *among accepted requests*, at least `target_teacher_agreement`, checked once on a 20% reserved split; coverage must also clear `min_deploy_coverage` |
| What it means for a caller | Out of every 100 requests, with 95% confidence at most 2 get a label Jev would not have given | Of the requests the student answers, at least the target share agree with the teacher (90% confidence); how many it answers is a separate number |
| When the bound is computed | On every retrain (every 2,000 new teacher answers by default), then watched | Once, at fit time; `update()` refits and re-certifies when you call it |
| After deployment | 2% of live traffic always goes to the teacher; a drift monitor raises it to 10%, falls back to the teacher and retrains; a change of teacher model is detected and handled | No audit of live traffic in the library; the README advises separate session and time-based evaluations |
| Unfamiliar input | kNN out-of-distribution gate | Distance guard that overrides acceptance |
| Serving | ~16 ms median on CPU including encoding, in Jev's exact response shape | CPU prediction; `tracer.serve` exposes `/predict` for embeddings without authentication or TLS |
| Research | Design document and launch post; prior art cited | Paper (arXiv 2604.14531); the README notes the paper's results predate the current certification implementation |
| Licence and status | Apache-2.0, alpha, 11 releases | MIT, about 1,000 stars; main carries unreleased changes |

## Where TRACER goes further

- **Any teacher, any API.** TRACER does not care who labelled the traces. Jevstiller speaks Jev's API only.
- **A wider student search**, with staged policies that try a second student before deferring.
- **Diagnostics**: a qualitative report with per-label slices and boundary pairs, an HTML report, and `scan()` to inspect traffic before fitting.
- **A paper**, and a JavaScript recorder for traces.

## Where Jevstiller goes further

- **What the bound covers.** TRACER bounds agreement conditional on acceptance. Jevstiller bounds the unconditional share of requests that come back with a different label than the teacher would give, which is the number a caller feels, and does it at 95% rather than 90%.
- **The bound stays true after day one.** A permanent audit sample keeps measuring agreement on live traffic, and the task falls back to Jev and retrains when it drops. TRACER certifies at fit time and leaves monitoring to you.
- **No code changes.** Point the TypeSafe SDK at the proxy. With TRACER your application embeds, predicts, routes and falls back.
- **Soft labels** from Jev's distributions, and `confidence_floor` so the bound also covers a caller's confidence check.
- **Teacher changes.** When `jev-latest` moves, traffic falls back until a student of the new model passes shadow.
- **Measured end to end**: five public tasks against Jev's recorded answers, load, chaos, a 24-hour soak ([benchmarks](/project/benchmarks/)).

## Which to pick

Pick TRACER if your teacher is not Jev, if you want to fit offline from traces you already have, or if you want to choose and inspect the student yourself. Pick Jevstiller if your service calls Jev and you want the loop, the audit and the bound to run themselves behind a base URL. The two bounds are not interchangeable: when comparing numbers, ask whether a figure is agreement among accepted requests or disagreement over all requests.

## Sources

- TRACER: [README](https://github.com/adrida/tracer/blob/28de14fdbccf159993e34fd0defebaa725c982dc/README.md) and [docs/concepts.md](https://github.com/adrida/tracer/blob/28de14fdbccf159993e34fd0defebaa725c982dc/docs/concepts.md) at commit 28de14f (2026-09-29): "How the quality check works", "OOD and teacher fallback", "Local serving and updates", "Release status".
- Jevstiller: [the guarantee](/posts/the-guarantee/), [configuration reference](/reference/configuration/), [benchmarks](/project/benchmarks/).
