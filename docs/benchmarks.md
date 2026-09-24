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

### Through the proxy, cold start (P6.7)

`python experiments/live_proxy.py --requests 4000 --concurrency 8` → `experiments/results/live-proxy.json`

2026-09-24. A real `jevstiller serve` process (bge-small, `admit_after = 50`, defaults otherwise) in front of the real Jev. The unmodified `typesafe-sdk` sends 4,000 Banking77 messages with one 6-class routing question (cards / transfers / cash / account / fees / other), from 8 threads.

| Measure | Result |
|---|---|
| Errors seen by the SDK | none |
| First local answer | request 3,697: admission (50), then 2,098 train + 503 calibration rows, then 1,010 shadow rows |
| Local share, last 500 requests | 50.4% (the run ended just after the promotion) |
| Promotion | `student:v1`: pooled coverage 83.4%, disagreement upper bound 2.01% against a 2% budget |
| Latency, local answers | p50 15.1 ms, p90 21.5 ms, p99 28.1 ms (n = 252; bge-small on CPU) |
| Latency, forwarded | p50 292 ms, p90 324 ms, p99 393 ms (Jev's own latency plus ~3 ms) |
| Jev cost | $0.067 for 3,698 calls; the task's lineage is `jev:jev-1.13.0` |
| Wall-clock | 140 s |

The engine recorded one teacher answer it could not use (`teacher_errors: 1`), which the SDK still received from Jev unchanged.

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
| + the cap held on the load path, 2 glibc arenas (2026-09-25, 30k requests, 7,365 loads/unloads) | 207 req/s | 2.4 / 9.0 ms | 113 ms | 313 / 300 MB |

The last row is not directly comparable: before it, `max_loaded` wasn't really held. With the janitor as the only enforcer, loads outran it, and up to 505–666 tasks were loaded at once (peak RSS 2.7 GB when measured with the P6 changes in). The earlier rows' lower reload counts partly reflect that: hot tasks simply stayed loaded. With the cap held, cold traffic evicts hot tasks, so 25% of requests are reloads and throughput is set by reload cost. Size `max_loaded` above the working set.

Hits only, 50 tasks loaded: 340 → 1,210 req/s and p99 80 → 13 ms from the BLAS cap and the maintenance gating. One task alone: 840 → 2,050 req/s. Reopening a task on its own takes ~4 ms (0.4 ms store, ~3 ms model files); first opens, which create the store, ~15 ms alone and ~210 ms under this load. The churn peak (the P2.1 open item) is explained and fixed: see the last row.

## Proxy (P3)

Local stub upstream, SDK over HTTP, sequential, 400 requests:

| Path | p50 | p99 |
|---|---|---|
| SDK → stub directly | 0.94 ms | 1.20 ms |
| SDK → proxy → stub (forwarded) | 4.02 ms | 5.09 ms |

`jevstiller serve` end to end (real CLI, process-pool training, hash encoder, fake Jev): 1,631 of 6,000 sequential requests answered locally within 21 s, no errors.

## Load (P6.4)

`python benchmarks/load.py --encoder small --concurrency 16 64 256 --seconds 30` → `experiments/results/load.json`

A real `jevstiller serve` process (bge-small, ONNX on CPU) in front of a fake Jev in its own process, which answers after ~290 ms (lognormal), like the real one. The load comes from closed-loop async clients: each connection sends its next request as soon as the last returns, with at most 64 connections per client process. Three scenarios, each on a fresh server that is warmed up (trained) first:

- **forward:** nothing admitted, so everything goes to Jev.
- **local:** one trained question.
- **mixed:** 20 questions, Zipf-distributed.

Latency is the server's own measurement (its access log). The clients share the 16-vCPU machine with the server and with ONNX Runtime, which takes every core when the encoder is saturated, so from 64 callers on their own measurements are inflated (at 64 callers, local answers: client p99 2.2 s against server p99 0.2 s).

| Scenario | Callers | Throughput | Local share | Local answers p50 / p99 | Forwarded p50 / p99 |
|---|---|---|---|---|---|
| forward | 16 | 54 req/s | — | — | 294 / 415 ms |
| forward | 64 | 215 req/s | — | — | 294 / 417 ms |
| forward | 256 | 585 req/s | — | — | 406 / 562 ms |
| local | 16 | 126 req/s | 64% | 16 / 50 ms | 311 / 435 ms |
| local | 64 | 130 req/s | 63% | 25 / 201 ms | 341 / 562 ms |
| local | 256 | 424 req/s | 23% | 348 / 436 ms | 519 / 994 ms |
| mixed | 16 | 73 req/s | 31% | 11 / 37 ms | 307 / 423 ms |
| mixed | 64 | 209 req/s | 29% | 84 / 172 ms | 381 / 525 ms |
| mixed | 256 | 367 req/s | 11% | 346 / 438 ms | 560 / 1,037 ms |

What it shows:

- **Forwarding adds ~1–4 ms** up to 64 callers. At 256 callers one process forwards ~585 req/s, which is 30 Jev keys' worth at Jev's 1,200 requests/minute limit.
- **The local path is bounded by the encoder:** bge-small on this CPU embeds ~150 of these synthetic texts a second (~340/s for short English sentences). Below that, local answers take 11–25 ms p50. Above it, backpressure (`max_encoder_wait_ms`) sends the excess to Jev, so throughput keeps rising with load (424 req/s at 256 callers) instead of collapsing. At 256 callers the machine is CPU-bound, and local answers slow to ~350 ms, still no slower than Jev.
- **The local shares here are not meaningful:** a language-model encoder can't make sense of the synthetic `w123` vocabulary, so fewer answers pass the confidence gate than with real text (compare the live Banking77 run).
- **Found and fixed on the way:**
  - Forwarding collapsed to ~70 req/s at 256 callers (a single httpx pool of 256 connections; now pools of 16).
  - Past the encoder's capacity, latency grew to 15 s p99 and throughput halved (now backpressure).
  - Requests that were only forwarded were still being encoded (now lazily, only for tasks).
  - The load generator itself hit the same httpx limit, so it now spreads its connections over processes.

## Soak (P6.6)

`python benchmarks/soak.py --minutes 40` → `experiments/results/soak.json`

A real `jevstiller serve` process (hash encoder) in front of a fake Jev (50 ms). Traffic runs at 100 requests/s over 20 questions with Zipf-distributed popularity, from 3 API keys. `max_loaded = 12` is deliberately below the 20 active questions, so tasks load and unload ~18 times a second. At the midpoint, the fake Jev silently rotates its labels (same model name, same difficulty, every answer different).

**Drift and recovery.** The final run (2026-09-25, all fixes in): 40 minutes, 240,246 requests, no errors, no 5xx, ~1,050 task reloads a minute.

| Time | Local share |
|---|---|
| 2 / 4 / 8 / 14 min | 41% / 55% / 70% / 84% |
| drift at 20:00 | students keep answering for ~50 s, now against the new teacher |
| +50 s → +81 s | tasks fall back as their audits confirm the break (19 `fallback` events, 18 `suspicious`); low 19% |
| +2.7 / +12 min after the drift | 50% / 80% again, from students trained only on post-drift answers |
| end | 85% (85% before the drift) |

The detection lag is the price of catching a *silent* change through a 2% audit: until the audit window confirms the break, the students serve answers the new teacher would not give. Here that lasted about a minute. An earlier run, before the load-path fixes, took ~3.5 minutes for the last task (the tail tasks see few audits). A change Jev *announces* (a new resolved model name) needs no audit: 20 answers from the new model switch the lineage (§7.12).

**Memory.**
- **The leak (found and fixed):** reference cycles kept every unloaded task's arrays alive, reaching 555 MB after 4 minutes and still accelerating.
- **After the fix:**
  - **The final 40-minute run:** 90 MB at the start, 182 MB at the end, with one momentary peak of 304 MB. The models themselves held 33–50 MB. RSS minus model memory grew +3 MB/h in the first half and +40 MB/h in the second.
  - **A 20-minute run without drift:** 159 MB at the end, with −10 MB/h in its second half.
  - **One earlier 40-minute run** (after the leak fix, but before the upstream-pool and load-path changes) reached 506 MB with ±100 MB swings. That did not reproduce.
- **A 30-minute confirmation run after the last fixes** (the `max_loaded` cap held on the load path, 2 glibc arenas): peak 222 MB, end 191 MB. The drift was recovered by itself (low 30% at +71 s, 50% again at +3 min). From minute 21 on, the harness and the server stalled together for seconds at a time while another workload was running on the machine (load average ~10). That stretch is inconclusive, and the earlier 40-minute runs had no such stalls.
- **Two forwarded requests in 180k got 502** (`ReadError`). This is a keep-alive race: the fake Jev closes idle connections after 5 s, which is also httpx's reuse limit. The SDK retries 502s. Whether real Jev's longer keep-alive avoids it is open.
- **Open:** the plan's 24-hour soak (`--minutes 1440`), to confirm the bound over a day.

## Chaos (P6.5)

`pytest tests/test_chaos.py`. The expected behaviour is written in docs/operations.md, "Common situations".

| Failure | Observed |
|---|---|
| Jev unreachable, after training | Local answers continue; forwarded requests get 502 (Jev-style body); nothing recorded as a teacher answer; recovers when Jev returns |
| Jev slow (3 s, 1 s timeout), 16 callers stuck on it | Forwarded requests get 504; local answers p95 < 250 ms meanwhile |
| 429 storm on one key (100 requests, 8 at a time) | Jev saw ≤ 8 of them; the proxy answered the rest with 429 + `retry-after`; another key unaffected; local answers for the limited key continue |
| Disk full (every write fails) | Every request answered; 200 records dropped and counted in `jevstiller_store_dropped_records_total`; one log line a minute; recording resumes with space |
| `kill -9` three times during training, restart | Comes up each time; `integrity_check` ok; no staging directories left; still learns to answer locally; no orphaned training workers (fixed: they used to outlive the server) |

## Concurrency (P6.3)

`pytest tests/test_concurrency_many.py`: 16 threads over 12 tasks with `max_loaded = 3`, so tasks reload under load while training. Every item is recorded exactly once per task, nothing stays checked out, and at most 3 stay loaded. Through the proxy: 300 concurrent async clients on 3 questions and 2 keys (neither verified at the start), all answered with well-formed 200s, and the rows per task equal the requests.

## Test harness note

Pre-binding a listening socket for uvicorn without `TCP_NODELAY` adds a ~40 ms Nagle/delayed-ACK stall to every request. uvicorn's own sockets (what `jevstiller serve` uses) are not affected. The test helper sets the option.
