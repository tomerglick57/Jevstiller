---
title: When to use Jevstiller, and when not to
description: What Jevstiller buys you, in order, and the workloads it is not for.
---

## What it buys you

In rough order of importance:

1. **Latency.** Jev takes ~300 ms per answer at every load we tried (16 to 64 concurrent callers, up to 190
   requests/s); a local answer takes ~16 ms at the median on CPU. It matters most for decisions made one after
   another (an agent loop, a game tick, classify-then-act): 200 of them took 65.7 s straight to Jev and 16.9 s
   through the proxy ([benchmarks](/project/benchmarks/)).
2. **Availability and independence.** The student keeps answering through Jev outages, rate limiting, and
   restricted network egress, and not every classification depends on one external API.
3. **Headroom.** Jev's rate limits are set per organization, in requests and tokens per second, and shown in
   TypeSafe's developer console (1,200 requests a minute was the published limit at launch). Local answers don't
   count against them. See [Jev rate limits and 429s](/guides/jev-rate-limits/).
4. **Cost**, last. Jev is cheap: $0.042 per million input tokens, about $17 per million short support messages with
   a 5-class question at roughly 400 tokens each, so savings only matter at very high volume. See
   [Cut your Jev API bill](/guides/cut-jev-cost/).

The warm-up costs nothing extra: the requests were going to Jev anyway, and each answer becomes a training row.
The ongoing Jev cost is the audit slice plus whatever the student passes on.

## A good fit

- The same classification question, asked many times.
- A class list that doesn't change often.
- Inputs that drift slowly.
- Enough volume to collect a few thousand examples.

## Not a fit

- **Changing class lists.** Adding or removing a class means retraining from scratch today.
- **Images or audio.** Only text and JSON states are learned. Requests with images (Jev 1.14) are forwarded to Jev
  unchanged.
- **Low volume.** If you'll never collect a few thousand examples, the student never gets trained.
- **One-off calls.** It only pays off on repetition. Questions built fresh for every request (dynamic criteria) never
  become tasks: a question must be asked `admit_after` times (50 by default) first.

- **Long inputs.** The student reads the first 256 tokens of the state. Classifying a long conversation or document
  by something near its end stays with Jev.
- **`score` questions,** for now: they are forwarded. Yes/no (`noul`) questions are answered locally since 0.5,
  for the outcome your code makes of the probability. See [coverage of Jev's API](/concepts/coverage/).

## Latency caveat

A request the student passes on costs the student's time *plus* Jev's, so the slowest requests get slightly
slower than calling Jev directly. The median gets much faster.
