"""The proxy's text encoder path under concurrent callers, and nothing else: the encoder wrapped in BatchingEncoder as
`jevstiller serve` wraps it, N threads each encoding one text at a time in a closed loop. Unlike load.py there is no
HTTP, no fake teacher and no client processes sharing the CPU, so it shows what the encoder and its batching can do.

    python benchmarks/encoders.py                        # bge-small (ONNX, CPU), 4 text sets, 16/64/256 callers
    python benchmarks/encoders.py --sets synthetic --callers 64 --seconds 10

Real text (Banking77, AG News, TweetEval offensive) comes from experiments/datasets.py, which downloads it once;
`synthetic` is the text load.py sends. Writes experiments/results/encoders.json.

To compare two versions of the code, run it from outside both checkouts with PYTHONPATH set to each in turn;
`--embeddings old.npz` on the first run and `--same-as old.npz` on the second check that both embed alike.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import jevstiller  # noqa: E402
from jevstiller import BatchingEncoder, SyntheticWorld, load_encoder  # noqa: E402
from jevstiller._task import state_text  # noqa: E402

SETS = ["tweet_offensive", "banking77", "ag_news", "synthetic"]


def texts_for(name: str, n: int = 4096) -> list[str]:
    if name == "synthetic":
        return [t for t, _ in SyntheticWorld([f"label_{i}" for i in range(20)], seed=1).sample(n)]
    from datasets import LOADERS
    return [state_text(t) for t, _ in LOADERS[name]()["rows"][:n]]


def drive(enc, texts: list[str], callers: int, seconds: float) -> dict:
    lat: list[float] = []
    stop = time.perf_counter() + seconds

    def caller(seed: int) -> None:
        rng = random.Random(seed)
        while time.perf_counter() < stop:
            t0 = time.perf_counter()
            enc.encode([texts[rng.randrange(len(texts))]])
            lat.append(time.perf_counter() - t0)

    threads = [threading.Thread(target=caller, args=(i,)) for i in range(callers)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ms = np.array(lat) * 1000
    return {"callers": callers, "texts_per_s": round(len(lat) / (time.perf_counter() - t0), 1),
            "p50_ms": round(float(np.percentile(ms, 50)), 1), "p99_ms": round(float(np.percentile(ms, 99)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="small")
    ap.add_argument("--backend", default="onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--callers", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--batch-sizes", nargs="+", type=int,
                    help="the encoder's own batch sizes to compare, interleaved in one run (default: its default)")
    ap.add_argument("--repeats", type=int, default=1, help="per level; the order of batch sizes alternates")
    ap.add_argument("--embeddings", type=Path, help="save each set's embeddings (its first 2,048 texts, one call) here")
    ap.add_argument("--same-as", type=Path, help="an --embeddings file from another run: how far this run's differ")
    a = ap.parse_args()
    out = {"jevstiller": jevstiller.__version__, "code": str(Path(jevstiller.__file__).parent), "runs": []}
    emb = {}
    for name in a.sets:
        texts = texts_for(name)
        encs = [BatchingEncoder(load_encoder(a.encoder, backend=a.backend, device=a.device,
                                             **({"batch_size": b} if b else {})))
                for b in a.batch_sizes or [None]]
        for enc in encs:
            enc.encode(texts[:64])
        out["encoder"] = encs[0].id
        if a.embeddings or a.same_as:
            emb[name] = encs[0].encode(texts[:2048])
        for c in a.callers:
            for rep in range(a.repeats):
                for enc in encs if rep % 2 == 0 else encs[::-1]:
                    r = {"set": name, "batch_size": enc.inner.batch_size, "repeat": rep,
                         **drive(enc, texts, c, a.seconds)}
                    out["runs"].append(r)
                    print(f"{name:16} {c:4} callers  batches of {r['batch_size']:3}  {r['texts_per_s']:7.1f} texts/s  "
                          f"p50 {r['p50_ms']:7.1f} ms  p99 {r['p99_ms']:7.1f} ms", flush=True)
        for enc in encs:
            enc.close()
    if a.embeddings:
        np.savez(a.embeddings, **emb)
    if a.same_as:
        other = np.load(a.same_as)
        for name, X in emb.items():
            d = {"set": name, "max_abs_diff": float(np.abs(X - other[name]).max()),
                 "min_cosine": float((X * other[name]).sum(1).min())}
            out.setdefault("same_as", []).append(d)
            print(f"{name:16} vs {a.same_as.name}: largest difference {d['max_abs_diff']:.2e}, "
                  f"lowest cosine {d['min_cosine']:.7f}", flush=True)
    dst = ROOT / "experiments" / "results" / "encoders.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
