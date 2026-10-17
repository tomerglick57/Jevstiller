---
title: "Jevstiller vs Distil Labs and cloud distillation: a swapped model or a bounded proxy"
description: "Distil Labs, Amazon Bedrock model distillation, OpenAI and Azure distillation post-train a small model from your traffic as a separate job and switch traffic to it. Jevstiller answers each Jev request locally only within a stated bound and forwards the rest. What each is for, what falls back when, and which to pick."
lastUpdated: 2026-10-17
editUrl: "https://github.com/tomerglick57/Jevstiller/edit/main/docs/compare/distil-labs.md"
---

<!-- Generated from docs/compare/distil-labs.md by website/scripts/sync-docs.mjs. Edit that file instead. -->

"Distillation" names two different products. [Distil Labs](https://www.distillabs.ai/) and the cloud offerings (Amazon Bedrock model distillation, OpenAI model distillation, Azure) take traffic or logs from a large model, train a smaller generative model to imitate it, evaluate the two side by side, and let you switch traffic to the small one. Jevstiller trains a per-question classifier from Jev's answers and routes each request between it and Jev under a bound. As of 2026-10-17, from [Distil Labs' documentation](https://www.distillabs.ai/docs/); the cloud products are described in general terms, check their current docs. Corrections welcome in an issue.

## Side by side

| | Jevstiller 0.5 | Distil Labs (documented 2026-10-17) |
|---|---|---|
| What it produces | One small classifier per repeated Jev question, a few hundred kilobytes each | A post-trained small language model for a task (classification, question answering, chat, agentic chat) |
| Where the data comes from | Jev's answers to your live traffic, collected by the proxy | Traces from an inference endpoint you route a slice of traffic through, or logs and labelled rows you upload; synthetic rows generated to expand the train and test sets |
| Teacher | Jev | The model your production runs today; also used as a judge |
| Evaluation | Agreement with Jev under a Clopper–Pearson bound, re-certified on every retrain, then audited on live traffic | Teacher and student scored on the same test split; the docs say scores vary between runs and to "treat every score as a sample" |
| Serving | The proxy answers in Jev's format; uncertain or novel requests go to Jev | A hosted deployment (temporary sessions; permanent serving on request) or a local download |
| Fallback | Per request, on uncertainty, novelty, a drift alarm, or a teacher change, plus a 2% audit for ever | "Behind the endpoint": the endpoint calls the small model first and falls back to the production model when it fails or times out. Not on uncertainty |
| A bound on agreement | Yes, stated and checked | None stated |
| API | Jev `/v1/systemone`; no code changes | OpenAI-style client; you change the model string to the new endpoint |
| Hardware | One CPU container, self-hosted | Training and hosting on their platform; local deployment available |
| Pricing | Free, Apache-2.0; you pay Jev for the audit and forwarded requests | Credits per route; the first two training runs are free |

## Where they go further

- **Scope.** A post-trained small model can replace an LLM on generative tasks: question answering, chat, agent steps. Jevstiller learns closed decisions only.
- **Independence from the teacher.** Once switched, the small model answers everything, so the teacher's latency, limits and bill are gone. Jevstiller keeps Jev in the loop by design.
- **Managed training.** Data expansion, training, evaluation and hosting are done for you.

## Where Jevstiller goes further

- **A bound you can state.** Every local answer is covered by a disagreement budget with a confidence level, checked on held-out rows and then on live traffic.
- **Per-request fallback on uncertainty.** The hard cases, the novel inputs and a permanent audit sample keep going to Jev. A swapped model answers them all, and its evaluation score is a sample.
- **No migration.** Services keep the TypeSafe SDK and their keys; only the base URL changes.
- **Drift and teacher changes are handled**, and the retrain is automatic.

## Which to pick

Pick Distil Labs or a cloud distillation job when the task is generative or open-ended and you want to leave the large model behind. Pick Jevstiller when the task is a repeated Jev decision and you want to keep Jev's accuracy on the cases that need it, with a stated bound on the rest, at no change to the calling code. For a service that already calls Jev, the two are not really alternatives: distillation would mean leaving Jev.

## Sources

- Distil Labs: [overview](https://www.distillabs.ai/docs/getting-started/overview/), [serving behind the endpoint](https://www.distillabs.ai/docs/deployment/behind-the-endpoint/), [platform reference](https://www.distillabs.ai/docs/reference/platform/), read on 2026-10-17.
- Jevstiller: [README](/concepts/comparison/), [the guarantee](/posts/the-guarantee/), [coverage of Jev's API](/concepts/coverage/).
