# Benchmarks and measurements

Every number quoted in the README, DESIGN.md and DEPLOYMENT_PLAN.md, with the command that produced it. Unless noted, the machine is a 16-vCPU WSL2 VM (Linux 6.6, Python 3.14, numpy with OpenBLAS). Several runs on 2026-09-24 overlapped with unrelated load on the machine (load averages up to 30); those are marked, and comparisons were made under the same conditions or as medians of interleaved repeats.

## Live Jev (2026-09-24)

`python experiments/jev_profile.py` → `experiments/results/jev-profile.json`, `tests/fixtures/jev/*.json`

| Concurrency | Requests | p50 | p90 | p99 | Throughput | Errors |
|---|---|---|---|---|---|---|
| 1 | 30 | 299 ms | 353 ms | 764 ms | 3.2 req/s | none |
| 8 | 120 | 288 ms | 330 ms | 793 ms | 24 req/s | none |
| 16 | 240 | 286 ms | 330 ms | 756 ms | 50 req/s | none |

A 5-class question with a one-sentence message used 400 input tokens (instructions and criteria included). The first request of a session took 830 ms.

### Banking77 against live Jev

`python experiments/run.py --dataset banking77 --teacher jev --encoder small --backend onnx --device cpu --tag live`

2026-09-24, `jev-latest` → `jev-1.13.0`, bge-small (ONNX, CPU), target agreement 98%, audit rate 2%. 11,083 replayed messages, with 2,000 held out and labelled by Jev for evaluation. All Jev answers are cached in `experiments/cache/banking77.jsonl`, so re-runs are free.

| Measure | Result |
|---|---|
| Held-out: share answered by the student (coverage) | **70.6%** |
| Held-out: system agreement with Jev | **99.40%** (target 98%); selective disagreement 0.85% |
| Live stream: audit-channel agreement | 99.27%, 95% interval [98.13%, 99.80%] → OK |
| Live stream: student share | 0% for the first 1,000 messages, 46% at 3,000, ~65% from 5,000 on (52.4% cumulative) |
| Jev calls | 5,270 for 11,083 messages (5,813 avoided) |
| Cost | $0.38 for the stream; Jev billed ~1,700 input tokens per call (77 class descriptions) |
| Accuracy against the dataset's true labels | Jev 78.55%; the Jevstiller system 78.7% |
| Jev's own confidence | median 0.98; 11.8% of messages below 0.6 |
| Jev latency | 291 ms mean |
| Student path throughput (held-out, CPU) | 129 rows/s, encoder-bound at 7 ms/text on this machine (Jev's ceiling: 20 rows/s) |
| Wall-clock for the whole replay | 11 min |

With a live teacher, coverage came out higher than with the "oracle" labels and the same encoder (65.7%). A plausible explanation, not tested: a model's labels are more consistent with the surface of the text than human labels, so they're easier to reproduce.

## Replays with a perfect ("oracle") teacher

Upper bounds on distillation (the hidden labels play the teacher). `python experiments/run.py --dataset <d> --teacher oracle --encoder <tier>`

| Dataset | Encoder | Held-out coverage | System agreement | Notes |
|---|---|---|---|---|
| Banking77 (77 classes) | base (GPU) | 70.9% | 99.55% | 2026-09-21, RTX 3090 |
| Banking77 | small | 56.2% | 99.50% | 2026-09-21 |
| Banking77 | large | 59.8% | 99.45% | 2026-09-21 |
| Banking77 | small, CPU ONNX, `ood_max_ref` 50,000 / 5,000 / 1,000 | 65.7% / 65.7% / 65.5% | 98.95% (all) | 2026-09-24; the 50k run keeps every row |
| CLINC150 (151 classes) | small, CPU ONNX, `ood_max_ref` 50,000 / 5,000 | 82.8% / 83.7% | 99.35% / 99.30% | 2026-09-24 |

The small-encoder Banking77 numbers differ between the two dates (56.2% vs 65.7%) because the 2026-09-24 runs use the later loop, with the OOD gate and policy changes of Phase 1–2. The `ood_max_ref` comparison holds everything else fixed.

## Engine concurrency (P1.1)

`python benchmarks/concurrency.py` — 32 callers, one task, a fake teacher that sleeps 50 ms per call.

| | Throughput | p50 | p99 |
|---|---|---|---|
| 0.1.0 (one lock around everything) | 19 req/s | 1,712 ms | 1,730 ms |
| after P1.1 + write-behind store | ~500 req/s | 53 ms | 89–113 ms |

With an instant teacher and 8 callers (Python-bound path): 216 → ~1,900 req/s after the write-behind store, and ~1,750 with background maintenance paced at 1 s (it was 431 unpaced).

## Training vs serving (P1.3)

`python benchmarks/concurrency.py --during-training --rows 15000 --repeats 3` — serving p99 on one task (8 callers) while 5 other tasks train. Median of 3 interleaved runs, 2026-09-24.

| Where the fit runs | p99 slowdown (runs) | 5 fits took |
|---|---|---|
| `train_pool(2)`: 2 low-priority workers, 2 BLAS threads each | ×1.12 (1.01 / 1.34 / 1.12) | ~28 s |
| threads in the serving process, 2 BLAS threads | ×1.11 | ~71 s |
| 5 low-priority workers, 2 BLAS threads | ×2.79 | ~26 s |
| 5 workers, 2 BLAS threads | ×2.93 | ~26 s |
| 5 workers, BLAS uncapped | ×12.1 | ~44 s |

Single runs on this machine varied by 2–4× (other load), hence the medians.

## Many tasks (P2.1)

`python benchmarks/manager.py [--requests 60000]` — 1,000 tasks (clones of one trained task, 384-dim hash encoder), 50 hot tasks get 90% of traffic, `max_loaded=50`, 8 threads, one text per request.

| Stage | Throughput | Hit p50 / p99 | Reload p50 | Peak / end RSS |
|---|---|---|---|---|
| first version | 125 req/s | 16.4 / 77.5 ms | 392 ms | 714 / 594 MB |
| + BLAS capped at 1 thread for serving, maintenance only when something changed | 464 req/s | 2.3 / 7.0 ms | 95 ms | 980 / 766 MB |
| + glibc told to return freed memory (60k requests, 5,600 loads/unloads) | ~550 req/s | 2.8 / 7.5 ms | 100 ms | 1,067 / 441 MB |

Hits only, 50 tasks loaded: 340 → 1,210 req/s and p99 80 → 13 ms from the BLAS cap and the maintenance gating. One task alone: 840 → 2,050 req/s. Reopening a task on its own takes ~4 ms (0.4 ms store, ~3 ms model files); first opens, which create the store, ~15 ms alone and ~210 ms under this load. Open item: peak RSS during heavy churn (see DEPLOYMENT_PLAN P2.1).

## Proxy (P3)

Local stub upstream, SDK over HTTP, sequential, 400 requests:

| Path | p50 | p99 |
|---|---|---|
| SDK → stub directly | 0.94 ms | 1.20 ms |
| SDK → proxy → stub (forwarded) | 4.02 ms | 5.09 ms |

`jevstiller serve` end to end (real CLI, process-pool training, hash encoder, fake Jev): 1,631 of 6,000 sequential requests answered locally within 21 s, no errors.

## Test harness note

Pre-binding a listening socket for uvicorn without `TCP_NODELAY` adds a ~40 ms Nagle/delayed-ACK stall to every request. uvicorn's own sockets (what `jevstiller serve` uses) are not affected. The test helper sets the option.
