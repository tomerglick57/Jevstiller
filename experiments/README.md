# Experiments

Replays a labelled dataset through Jevstiller with the labels hidden (DESIGN.md §15). Outputs go to
`experiments/results/<run>/` (`results.json`, `report.md`, `curves.png`); Jev answers are cached in
`experiments/cache/<dataset>.jsonl` so a re-run is free.

```bash
bash experiments/reproduce.sh                                                     # the README's Banking77 result from Jev's recorded answers (shipped, gzipped); no key
python3 experiments/run.py --dataset banking77 --teacher jev --encoder base        # the real thing (needs TYPESAFE_API_KEY in .env)
python3 experiments/run.py --dataset banking77 --teacher oracle --encoder base     # dry run: hidden labels as a perfect teacher
python3 experiments/run.py --dataset synthetic --teacher synthetic --encoder hash  # no downloads, no GPU, ~1 min
python3 experiments/jev_profile.py                                                 # live Jev latency + wire fixtures
python3 experiments/record_answers.py --dataset banking77                          # record Jev's answer for every message into the cache (~$0.35)
```

More options: `--backend onnx --device cpu` (CPU-only), `--ood-max-ref N` (size of the OOD reference, default 5,000),
`--limit`, `--target`, `--audit`, `--min-*`. Replays train `inline` so results don't depend on thread timing.
All results with their dates are collected in [docs/benchmarks.md](../docs/benchmarks.md).

Arms: `--encoder small|base|large|hash`, `--label-target probs|hard`, `--target 0.98`, `--backend torch|onnx`, `--device cpu|cuda`.

## Dry-run results (oracle teacher, Banking77, 77 classes, target agreement 98%)

Upper bound on distillation with a perfect teacher; 11,083 streamed rows, 2,000 held out.

| encoder | held-out coverage | system agreement | student path rows/s (GPU) |
|---|---:|---:|---:|
| small | 56.2% | 99.50% | 1,868 |
| **base** | **70.9%** | 99.55% | 1,950 |
| large | 59.8% | 99.45% | 1,038 |

Jev's published limit is 20 rows/s (1,200/min); on 2026-09-26 our key sustained 190/s without errors (docs/benchmarks.md), so treat the limit as a policy, not a ceiling.

Things these runs found and fixed in the loop (all now in DESIGN.md): shadow candidates starved by the 2% audit
trickle; policies fitted exactly at the budget failing the re-test by construction; training drifting toward the
deferred (hard) slice; a fixed L2 that hid the encoder differences; exact importance weights collapsing the
effective sample size.

The live run against Jev (2026-09-24) is in [docs/benchmarks.md](../docs/benchmarks.md#banking77-against-live-jev) and DESIGN.md §15.5.
