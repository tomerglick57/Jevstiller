# Cut your Jev API bill
<!-- title: Cut your Jev API bill: what Jev costs and three ways to pay less -->
<!-- description: Jev costs $0.042 per million input tokens and nothing for output, so the bill is requests × tokens per request. Shorter states, several questions per request, and answering repeated questions locally with Jevstiller (typically 70–80% of an intent or topic question): what each saves, when it applies, and when none of them helps. -->

## What Jev costs

From TypeSafe's [models page](https://docs.typesafe.ai/models), checked on 2026-10-17:

| | Jev 1.14 (`jev-1.14.0`, `jev-preview`) | Jev 1.13 (`jev-1.13.0`, `jev-latest`) |
|---|---|---|
| Input tokens | $0.042 per million ($42 per billion) | $0.042 per million |
| Output tokens | free | free |
| Context | 64k tokens per request, 32k per question | same |
| Long context (`TypeSafe-Danger-Zone-Long-Context: true`) | 262k, same price per token | same |

Everything you pay for is input: the state (your text), the instructions and the options or criteria of every question in the request. There is no per-call fee and no charge for the answer. So for a fixed question:

> spend ≈ requests × input tokens per request × $0.042 per million

A short support message with a five-class question runs to roughly 400 input tokens once the instructions and class descriptions are counted, which is about $17 per million requests. Banking77, with 77 described intents per request, cost $0.62 for 11,083 requests in [the benchmark](../benchmarks.md), about $56 per million. Jev is cheap; the bill only matters at high volume, and at high volume it is one more external call per decision.

## Three ways to pay less

| Option | Saves | Works when |
|---|---|---|
| Send less per request: shorter states, tighter instructions | tokens on every request | always; it is the first thing to try, and needs no software |
| Ask several questions per request | one state's tokens per extra question (the state is billed once) | your service asks several questions about the same input |
| Answer repeated questions locally from Jev's own answers ([Jevstiller](https://github.com/tomerglick57/Jevstiller)) | the share answered locally, typically 70–80% on intent and topic questions | the same question is asked thousands of times about different inputs |

Caching identical requests helps only with retries and duplicate events: most real text is never repeated exactly.

## Answering locally

Jevstiller sits in front of Jev as a drop-in proxy (set `TYPESAFE_BASE_URL`). For each repeated question it trains a small classifier on Jev's answers to your traffic. It answers a request itself only when a statistical bound says that keeps agreement with Jev above your target (98% by default). Everything else, plus a permanent 2% audit, goes to Jev.

On five public tasks replayed against Jev's recorded answers ([benchmarks](../benchmarks.md)), at a 98% target:

| Task | Requests answered locally (no Jev call) | Agreement with Jev |
|---|---:|---:|
| Banking77, 77 intents | 71.9% | 99.50% |
| CLINC150, 150 intents | 69.0% | 99.65% |
| AG News, 4 topics | 80.2% | 99.40% |
| TweetEval sentiment | 22.2% | 98.70% |
| TweetEval offensive | 24.1% | 98.80% |

The share answered locally is the share of calls, and of input tokens, that never reach Jev. It follows how consistent Jev is on the task. Where Jev itself is noisy (the tweet tasks), most requests still go to Jev. Lowering the target to 95% roughly doubles the local share on the noisy tasks.

`jevstiller admin status <task>` reports the calls avoided and their estimated cost (`price_per_mtok`, default $0.042 per million input tokens).

## When it won't help

- **Questions that change on every request** (criteria built per request, re-ranking) never repeat enough to train on. They are forwarded.
- **Low volume.** A question needs a few thousand of Jev's answers before local answers start.
- **Long states.** The encoder reads the first 256 tokens. When the answer depends on later text, little is answered locally.
- **`score` questions, and requests mixing question types** that aren't all trained, are forwarded. See [coverage of Jev's API](../coverage.md).

## Before you deploy

Jevstiller trains on Jev's output, and TypeSafe's terms restrict that. Read [TypeSafe's terms](../../README.md#typesafes-terms) and decide for your own use.

## FAQ

### How much does Jev cost?

$0.042 per million input tokens, for both Jev 1.13 and Jev 1.14, as of October 2026. Output tokens are free. A request's cost is its input tokens: the state plus every question's instructions and options.

### Does Jev charge for output tokens?

No. TypeSafe's models page says output tokens are free. Only input is billed.

### Does putting several questions in one request save money?

Yes. The state is sent and billed once per request, so each extra question costs only its own instructions and options, not another copy of the state.

### How much of a Jev bill can Jevstiller remove?

The share of requests it answers locally. On intent and topic classification that was 69% to 80% in the benchmarks at a 98% agreement target; on noisy sentiment tasks, 22% to 24%. The remaining requests and a 2% audit sample still go to Jev.

### Does Jevstiller save money from the first day?

No. Every request goes to Jev until a question has about a thousand of Jev's answers, a local model has trained, and it has passed a shadow test on live traffic. Those requests were going to Jev anyway; each one becomes a training row.

### Is answering locally from Jev's answers allowed under TypeSafe's terms?

TypeSafe's Master Customer Agreement restricts training on Jev's output. The project's reading, and why it may not apply to a small per-question classifier used only in your own deployment, is in the README under "TypeSafe's terms". It is not legal advice; decide for your own use.
