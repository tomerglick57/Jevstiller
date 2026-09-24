---
title: Benchmarks
description: Offline replay results, how to reproduce them, and what they do not show yet.
---

:::caution[Upper bound, not a Jev result]
These runs use the dataset's hidden labels as a perfect "oracle" teacher. The benchmark against live Jev is
pending. Treat these numbers as an upper bound until it lands.
:::

## Banking77, oracle teacher

77 banking intents, target agreement 98%, 11,083 streamed rows with 2,000 held out.

| encoder | held-out coverage | system agreement | student rows/s (GPU) |
|---|---:|---:|---:|
| `small` | 56.2% | 99.50% | 1,868 |
| **`base`** | **70.9%** | 99.55% | 1,950 |
| `large` | 59.8% | 99.45% | 1,038 |

Jev's ceiling is 20 rows/s. GPU numbers are from one RTX 3090.

`base` is the default: it buys about 15 points of coverage over `small` for the same GPU cost. `large` did not
beat it at this data size.

## Encoder cost per text

| tier | params | ONNX, CPU (16 cores) | PyTorch, RTX 3090 |
|---|---|---:|---:|
| `small` | 33M | 1.56 ms | 0.38 ms |
| `base` | 110M | 5.39 ms | 0.48 ms |
| `large` | 335M | 17.92 ms | 0.94 ms |

On CPU-only machines, `small` is a reasonable trade.

## Reproduce

```bash
python experiments/run.py --dataset banking77 --teacher oracle --encoder base      # dry run, no key
python experiments/run.py --dataset synthetic --teacher synthetic --encoder hash   # no downloads, ~1 min
python experiments/run.py --dataset banking77 --teacher jev --encoder base         # live Jev (needs a key)
```

Results land in `experiments/results/<run>/`. Jev answers are cached, so a re-run is free. See
[experiments/README.md](https://github.com/tomerglick57/Jevstiller/blob/main/experiments/README.md).

## Share yours

The most useful contribution is a result on your own task: `status().report()` at 1k, 5k, and 20k requests,
with the encoder tier and `Config` you used.
[Open a result issue](https://github.com/tomerglick57/Jevstiller/issues/new?template=result.md).
