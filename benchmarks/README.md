# Benchmarks

Performance scripts. No network, no API key; they use the synthetic teacher (`load.py` and `soak.py` run it as a fake Jev over HTTP). Results and dates are in
[docs/benchmarks.md](../docs/benchmarks.md).

| Script | Measures | Typical run |
|---|---|---|
| `concurrency.py` | Throughput and latency of one task with 32 concurrent callers and a slow fake teacher (P1.1) | `python benchmarks/concurrency.py` (seconds) |
| `concurrency.py --during-training` | Serving p50/p99 while 5 other tasks train, across 6 training setups (threads vs process pool, BLAS capped or not, pool size) (P1.3) | `--rows 15000 --repeats 3` (~30 min) |
| `manager.py` | Many tasks: resident memory, loads/unloads, hit/reload/first-open latency with a hot/cold traffic mix (P2.1) | `python benchmarks/manager.py` (1,000 tasks, ~1 min); `--requests 60000` for a longer run |
| `load.py` | A real `jevstiller serve` behind closed-loop clients: throughput and latency (client- and server-side) for forward-only, local and mixed traffic (P6.4) | `python benchmarks/load.py --encoder small` (~20 min); `--scenarios forward --encoder hash` in a minute |
| `soak.py` | A real server under steady traffic for a long time, with the fake Jev silently changing its answers halfway: local share, RSS vs model memory, reloads (P6.6) | `python benchmarks/soak.py --minutes 40`; `--minutes 1440` for the full day; `--no-drift`, `--max-loaded` |

Timings on a shared or loaded machine vary by 2–4× between single runs: compare setups within one run (the
scripts interleave them) and prefer `--repeats`.
