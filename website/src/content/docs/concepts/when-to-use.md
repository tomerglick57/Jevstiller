---
title: When to use it
description: What Jevstiller buys you, in order, and the workloads it is not for.
---

## What it buys you

In rough order of importance:

1. **Throughput beyond the rate limit.** Jev allows 1,200 requests a minute per key, a ceiling of about 1.7M
   classifications a day. A backlog of 100k messages takes 83 minutes through Jev; the student does it in under
   a minute.
2. **Latency.** ~350 ms per Jev answer becomes single-digit milliseconds for the requests the student handles.
3. **Availability.** The student keeps answering through Jev outages, rate-limit storms, and restricted network
   egress.
4. **Cost**, last. Jev is cheap (about $3 per million short messages), so savings only matter at very high
   volume.

The warm-up costs nothing extra: the requests were going to Jev anyway, and each answer becomes a training row.
The ongoing Jev cost is the audit slice plus whatever the student passes on.

## A good fit

- The same classification question, asked many times.
- A class list that doesn't change often.
- Inputs that drift slowly.
- Enough volume to collect a few thousand examples.

## Not a fit

- **Changing class lists.** Adding or removing a class means retraining from scratch today.
- **Non-text input.**
- **Low volume.** If you'll never collect a few thousand examples, the student never gets trained.
- **One-off calls.** It only pays off on repetition.

## Latency caveat

A request the student passes on costs the student's time *plus* Jev's, so the slowest requests get slightly
slower than calling Jev directly. The median gets much faster.
