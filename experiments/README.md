# Experiments

Replays a labelled dataset through Jevstiller with the labels hidden (DESIGN.md §15). Outputs go to
`experiments/results/<run>/` (`results.json`, `report.md`, `curves.png`); Jev answers are cached in
`experiments/cache/<dataset>.jsonl` so a re-run is free.

```bash
python3 experiments/run.py --dataset banking77 --teacher jev --encoder base        # the real thing (needs TYPESAFE_API_KEY in .env)
python3 experiments/run.py --dataset banking77 --teacher oracle --encoder base     # dry run: hidden labels as a perfect teacher
python3 experiments/run.py --dataset synthetic --teacher synthetic --encoder hash  # no downloads, no GPU, ~1 min
```

Arms: `--encoder small|base|large|hash`, `--label-target probs|hard`, `--target 0.98`, `--backend torch|onnx`, `--device cpu|cuda`.

## Dry-run results (oracle teacher, Banking77, 77 classes, target agreement 98%)

Upper bound on distillation with a perfect teacher; 11,083 streamed rows, 2,000 held out.

| encoder | held-out coverage | system agreement | student path rows/s (GPU) |
|---|---:|---:|---:|
| small | 56.2% | 99.50% | 1,868 |
| **base** | **70.9%** | 99.55% | 1,950 |
| large | 59.8% | 99.45% | 1,038 |

Jev's ceiling is 20 rows/s (1,200/min).

Things these runs found and fixed in the loop (all now in DESIGN.md): shadow candidates starved by the 2% audit
trickle; policies fitted exactly at the budget failing the re-test by construction; training drifting toward the
deferred (hard) slice; a fixed L2 that hid the encoder differences; exact importance weights collapsing the
effective sample size.

The real run against Jev is pending API credits.
