---
title: The guarantee
description: What target_agreement promises, how the routing threshold is chosen so the promise holds, and how it is checked in production.
---

## The contract

> For a task with many repeated requests, Jevstiller returns the label **Jev would have returned** on at least
> `target_agreement` of them, while sending as few requests to Jev as possible.

You set one number. Everything left over is the **disagreement budget**:

```text
target_agreement = 0.98      →  budget β = 2% of all requests
```

A request the student passes to Jev agrees with Jev by definition. So the only way to miss is a request the student
answers *and* gets different from Jev, and the contract is exactly:

```text
P(the student answers and disagrees) ≤ β          over all requests
```

A student that disagrees with Jev 5% of the time on what it answers can still take 40% of traffic under a 2%
budget: 0.40 × 5% = 2%.

## How the threshold is chosen

The student answers a request when its confidence clears a threshold and the input is close enough to what it
was trained on (the out-of-distribution gate). The threshold is picked on a **calibration set**: a random sample
of traffic, labelled by Jev, never used for training. Then:

1. **The loss is the contract itself.** Each calibration row scores 1 if the student would answer it and disagrees
   with Jev. The rate of that over *all* rows is the quantity the contract bounds, and a row-level yes/no count is
   a binomial, so an exact Clopper–Pearson upper bound applies (95% confidence by default).
2. **Thresholds are tested strictest first, and testing stops at the first failure.** Loosening the threshold can
   only add answered rows, so the rate only grows. Testing a fixed grid of thresholds in that order, each at the
   full confidence level, keeps the chance of choosing a bad one within 5% with no multiple-testing penalty
   (fixed-sequence testing, as in *Learn Then Test*).
3. **The calibration rows only test; they never choose.** The grid of thresholds is fixed in advance, and the
   out-of-distribution cutoff comes from the training data (leave-one-out distances), not from the rows the bound
   is computed on.

Together: with probability at least 95%, the chosen policy's rate of answered-and-disagreeing requests is at most
`β`. Rows that went to Jev because the student was unsure are never calibration rows: they are the hard cases by
construction, not a sample of traffic.

Two margins on top:
- **Headroom.** The policy is fitted at 85% of the budget and judged at 100% in shadow, on the calibration rows
  pooled with fresh live traffic. A threshold fitted exactly on the budget would fail such a re-test about half the
  time.
- **Rare classes** can be deferred: the student never answers them, and calibration counts only what it may answer.

The student's confidence never has to be a true probability. It only has to *rank* inputs so that low-confidence
ones disagree more often.

## Checked forever

The bound holds for each student version as it is promoted. Traffic can change after that, and every retrain spends
the 5% again, so a random **audit slice** of 2% of requests always goes to Jev, and the answers are compared:

| status | condition | action |
|---|---|---|
| `OK` | lower bound ≥ target | none |
| `inconclusive` | interval straddles the target | collect more audit samples; if the point estimate is below target, the audit rate rises to 10% and a retrain starts |
| `BROKEN` | upper bound < target | every request goes back to Jev; retrain on data from after the break |

The "broken" test uses the upper bound on purpose. It only fires when the data are confident the contract is
violated, not when a small audit sample is noisy. A change of Jev's model (`jev-latest` moving to a new version)
starts a new lineage the same way: answers come from Jev until a student trained on the new model's answers passes.

## Agreement is not accuracy

There is no ground truth anywhere in Jevstiller. The only label source is Jev. If Jev is systematically wrong about
something, the student will be wrong the same way, and Jevstiller will report full agreement while both are wrong.

That is also why this is buildable: proving parity with a teacher needs only the teacher. Proving accuracy needs
thousands of human labels per class, re-checked on every drift.

The full derivation, including the flaws the first version had and how they were found, is in
[DESIGN.md §7.6](https://github.com/tomerglick57/Jevstiller/blob/main/DESIGN.md#76-calibrator).
