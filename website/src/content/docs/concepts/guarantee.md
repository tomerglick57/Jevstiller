---
title: The guarantee
description: What target_agreement promises, how the threshold is chosen, and how it is checked in production.
---

## The contract

> For a task with many repeated requests, Jevstiller returns the label **Jev would have returned** on at least
> `target_agreement` of them, while sending as few requests to Jev as possible.

You set one number. Everything left over is the **disagreement budget**:

```text
target_agreement = 0.98      →  budget β = 2% of all requests
```

## The arithmetic

| symbol | meaning |
|---|---|
| `c` | coverage: share of requests the student answers |
| `e` | selective disagreement: how often the student differs from Jev, among the requests it answers |
| `A` | system agreement: share of all requests where the answer equals Jev's |

Requests the student passes to Jev agree with Jev by definition, so:

```text
A = 1 − c · e
```

The router maximises `c` subject to `c · e ≤ β`. A student that disagrees with Jev 5% of the time can still
take 40% of traffic under a 2% budget.

## Why it is a bound and not a hope

1. **`e` is measured on a held-out calibration set** that is a random sample of traffic, labelled by Jev, never
   used for training. Rows that were sent to Jev because the student was unsure are excluded: they are the hard
   cases by construction, not a sample of traffic.
2. **The threshold is chosen against an upper confidence bound on `e`**, not its point estimate: a one-sided
   Clopper–Pearson bound at 95% confidence. With that probability, the true disagreement is inside the budget.
3. **Policies are fitted at 85% of the budget** and judged at 100% in shadow. A threshold fitted exactly on the
   budget fails a fresh re-test about half the time.

The student's confidence never has to be a true probability. It only has to *rank* inputs so that low-confidence
ones disagree more often.

## Checked forever

Once a student is in production, the audit channel keeps sending a random 2% of requests to Jev and compares.
The status report shows the result with its interval:

| status | condition | action |
|---|---|---|
| `OK` | lower bound ≥ target | none |
| `inconclusive` | interval straddles the target | collect more audit samples |
| `BROKEN` | upper bound < target | everything falls back to Jev; retrain |

The "broken" test uses the upper bound on purpose. It only fires when the data are confident the contract is
violated, not when a small audit sample is noisy.

## Agreement is not accuracy

There is no ground truth anywhere in Jevstiller. The only label source is Jev. If Jev is systematically wrong
about something, the student will be wrong the same way, and Jevstiller will report full agreement while both
are wrong.

That is also why this is buildable: proving parity with a teacher needs only the teacher. Proving accuracy needs
thousands of human labels per class, re-checked on every drift.
