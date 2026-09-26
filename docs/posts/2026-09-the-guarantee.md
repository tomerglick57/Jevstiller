# A local model that answers like Jev 98% of the time, and how we know

*Draft for the launch. Numbers are from [docs/benchmarks.md](../benchmarks.md), reproducible with `bash experiments/bench.sh --no-record`.*

If you classify text with [Jev](https://docs.typesafe.ai), every answer is a network call to one vendor and comes back in about 300 ms, at any load. For a batch job that is fine. For an agent loop that decides, acts, and decides again, or a game tick, or anything that classifies then acts, 300 ms per step is the whole budget.

Jevstiller sits in front of that call, learns a small local model from Jev's own answers, and lets it answer what it is sure about in about 15 ms on a CPU. The interesting part is not the small model. It is the contract:

> Set one number, say 98%. Jevstiller returns the label Jev would have returned on at least that share of requests.

This post is about what it takes to make that sentence true, why the obvious way of picking a confidence threshold does not make it true, and what it costs.

## The contract, in one line

Per task, over a window of traffic, call `c` the share of requests the local model answers (coverage), and `e` the share of those where its label differs from Jev's. Requests it does not answer go to Jev and agree with Jev by definition. So the system's agreement with Jev is

```
A = 1 − c · e
```

Your target `A*` gives a budget `β = 1 − A*`: the share of *all* requests that may end up with an answer Jev would not have given. At 98%, that is 2 in 100. The router's job is to answer as much as it can while keeping `c · e` under `β`.

Two things are deliberately absent from that sentence. It says nothing about the model's accuracy against the truth: if Jev is wrong, the local model is wrong the same way, and the status report says so next to every number. And it is a statement about agreement over all requests, not about the local model's accuracy on the requests it chose to answer. The first framing is what you can verify; the second is what most tools report.

## The obvious way, and why it breaks

The usual recipe for a cascade like this: hold out some data, sweep a confidence threshold, keep the loosest one whose measured disagreement is within budget, ship it. It is what you would write in an afternoon, and it is what the point-estimate rule in our benchmark does.

Then it breaks the budget about half the time. On five public tasks, twenty random train/calibration/test splits each, the point-estimate rule exceeded the 2% budget on 6 to 12 of 20 splits per task, by up to a full percentage point:

| Task | Rule | Coverage | Disagreement, mean | worst | Budget broken |
|---|---|---:|---:|---:|---:|
| Banking77 (77 intents) | point estimate | 79.8% | 1.90% | 2.70% | 9 / 20 |
| Banking77 | **bound** | 74.9% | 1.15% | 1.55% | **0 / 20** |
| CLINC150 (151 intents) | point estimate | 84.3% | 2.09% | 2.85% | 12 / 20 |
| CLINC150 | **bound** | 78.9% | 1.25% | 1.80% | **0 / 20** |
| AG News (4 classes) | point estimate | 90.3% | 2.06% | 2.65% | 11 / 20 |
| AG News | **bound** | 86.5% | 1.34% | 1.75% | **0 / 20** |
| TweetEval sentiment | point estimate | 28.2% | 1.99% | 2.60% | 8 / 20 |
| TweetEval sentiment | **bound** | 24.0% | 1.46% | 2.10% | **1 / 20** |
| TweetEval offensive | point estimate | 36.8% | 1.81% | 2.70% | 6 / 20 |
| TweetEval offensive | **bound** | 28.7% | 1.10% | 1.40% | **0 / 20** |

This is not a bug in the recipe. It is what selecting the loosest passing threshold on finite data does. The measured disagreement at any threshold is a noisy estimate of the true one. Picking the loosest threshold that *looks* under budget picks, preferentially, a threshold whose noise happened to point downward. On the next sample of traffic the noise points elsewhere, and the threshold that measured 1.9% delivers 2.7%.

Across 100 splits the bound broke the budget once, at 2.10%. That once is expected: the bound is a 95% statement, so about 5 in 100 may miss. The point estimate is not a statement at all.

## What Jevstiller does instead

Four decisions, each small, together make the contract hold with probability at least 95% for every version of the local model that goes into production.

**1. The loss is the contract itself.** For each row of an IID calibration set, labelled by Jev and never used for training, score 1 if the local model *would answer it and would disagree with Jev*, else 0. The average of that indicator over all rows is exactly "disagreement over all requests", the quantity the budget limits. It is a binomial, so it has an exact confidence bound, Clopper–Pearson, with no approximation and no appeal to large samples.

**2. Test the thresholds strictest first, on a fixed grid, and stop at the first failure.** The rate can only grow as the threshold loosens, so walking from the strictest candidate to the loosest, checking each one's upper bound against the budget, and stopping at the first that fails, controls the chance of a wrong choice at the same 5% with no correction for having tested many candidates. This is fixed-sequence testing, as in [Learn Then Test](https://arxiv.org/abs/2110.01052). The grid is fixed before any calibration row is seen; the calibration rows only test.

**3. Nothing else touches the calibration rows.** The out-of-distribution gate, which sends unfamiliar inputs to Jev regardless of confidence, gets its cutoff from the training data. The candidate thresholds are a fixed grid. If either were tuned on the calibration rows, the bound would be computed on data that had already been used to choose what it bounds, which is the point-estimate mistake in a different coat.

**4. Leave headroom.** The threshold is fitted at 85% of the budget. A candidate then has to pass a second check at the full budget on the calibration rows pooled with fresh traffic it shadowed. That check is not an independent guarantee, since the calibration rows helped choose the threshold, but it catches a threshold that would sit exactly on the line.

The first version of this got two of the four wrong: it bounded the selective rate and multiplied by an estimated coverage as if it were exact, and it kept the loosest of 200 thresholds that each passed on their own, which selects on noise just like the point estimate. A prior-art review caught both. The corrected rule cost nothing on the Banking77 replay: coverage moved from 70.6% to 70.7%.

## Keeping it true after the first day

A bound at promotion time is not enough, for two reasons. The local model retrains as traffic accumulates, and every retrain spends the 5% again, so over many versions some will miss. And traffic changes, or Jev's answers do.

So a fixed 2% of *all* requests goes to Jev regardless of what the local model thinks, with the local model's answer recorded alongside. That audit slice is the only unbiased view of production once routing begins: the requests the local model declines are the hard ones by construction, and measuring agreement on them would measure the router, not the world. The audit gives a confidence interval on live agreement, per version, scored with the answer each request was actually served. When it drops below target the audit rate goes up; when its own bound confirms a breach, every request goes back to Jev and training restarts from that point.

In the 24-hour soak, a stand-in Jev silently changed every answer at hour twelve. The local share fell from 90% to 9% within four minutes as the audits caught it, and was back at 90% within 49 minutes, trained on post-change answers only, with nobody touching anything.

## What it costs

Coverage. The bound is conservative on finite data, and that is the point: it gave up four to eight points of coverage against the point-estimate rule on every task above. Training on Jev's full probability distributions rather than its top label buys two to three points back on the many-class tasks.

The target is a dial. On the same five tasks, moving it from 98% to 95% roughly doubles what the tweet tasks answer locally and lifts the intent tasks from about 70% to the low 80s. Across the whole range, from 99% down to 90%, the system's accuracy against the datasets' own labels stayed within a point of Jev's, because where the local model differs from Jev it is about as often right as Jev was. That held on these five tasks; it is not a law.

Coverage also tracks Jev's consistency, not the task's difficulty. On the two tweet tasks Jev agrees with the human labels only 64% and 74% of the time, so its answers near the class boundaries are noisy, and a local model cannot reproduce noise within a 2% budget. It answers the confident quarter and forwards the rest.

## What it does not promise

The bound assumes the calibration rows are a random sample of the traffic the threshold will be used on. If the traffic mix shifts, the bound on the old mix says nothing about the new one, which is what the audit is for. Agreement is not accuracy. And the contract is per request, not per class: a rare class can carry most of the disagreements. Per-class budgets are on the roadmap.

## Prior art

Selective classification with a risk guarantee is [Geifman and El-Yaniv, 2017](https://arxiv.org/abs/1705.08500); the fixed-sequence testing is from [Learn Then Test](https://arxiv.org/abs/2110.01052). Cascades that learn from a large model's answers appear in [OCaTS](https://aclanthology.org/2023.findings-emnlp.1000/) (EMNLP Findings 2023) and [Cache & Distil](https://aclanthology.org/2024.findings-acl.704/) (ACL Findings 2024), without a bound; [BARGAIN](https://arxiv.org/abs/2509.02896) makes the same agreement contract with finite-sample guarantees, for batch processing; [vCache](https://arxiv.org/abs/2502.03771) (ICLR 2026) gives a similar guarantee for a semantic cache. Jevstiller's contribution is the combination in a running system: the guarantee kept under continual retraining, a permanent audit, drift detection, and a lineage that restarts training when the teacher changes.

## Try it

```bash
git clone https://github.com/tomerglick57/Jevstiller && cd Jevstiller && pip install jevstiller
bash experiments/reproduce.sh          # the Banking77 result, from Jev's recorded answers, no key, ~10 min
bash experiments/bench.sh --no-record  # all five tasks and the threshold-rule comparison, an hour or two
```

Or put it in front of your own Jev calls: `docker run -d -p 8080:8080 -v jevstiller-data:/data ghcr.io/tomerglick57/jevstiller`, point `TYPESAFE_BASE_URL` at it, and read the status report after a few thousand requests. It prints the bound it achieved, and the interval the audit has measured since.
