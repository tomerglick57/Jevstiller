# Cut your Jev API bill

Jev bills per input token. When a service asks the same question about a stream of inputs (route this ticket, flag this message, pick an intent), the bill grows with traffic, and so does the number of calls going to one external API. This page covers the options, what each one saves, and when Jevstiller is not the answer.

## Where the spend comes from

A Jev call is one state (the text) plus one or more questions. For a fixed question, the cost of a request is its input tokens, so the spend is roughly requests × average state length × price. Shortening states and asking several questions in one request (one state, many questions) both help, and need no new software.

## Options

| Option | Saves | Works when |
|---|---|---|
| Cache identical requests | the repeats | the same text arrives again: retries, duplicate events. Most real text is never exactly repeated. |
| Ask several questions per request | one state's tokens per extra question | your service asks several questions about the same input |
| Answer locally from Jev's own answers ([Jevstiller](https://github.com/tomerglick57/Jevstiller)) | the share answered locally, typically 70–80% on intent and topic questions | the same question is asked thousands of times about different inputs |

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
