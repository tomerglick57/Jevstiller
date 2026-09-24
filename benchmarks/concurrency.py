"""Throughput and latency under concurrent callers, with a slow fake teacher. No network, no key.

    python benchmarks/concurrency.py                   # throughput: 32 callers, one task, 50 ms teacher
    python benchmarks/concurrency.py --during-training # serving p50/p99 while 5 other tasks train

The teacher sleeps like a network call, so the number that matters is how many of those sleeps overlap.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from jevstiller import Config, Jevstiller, SyntheticTeacher, SyntheticWorld, Task
from jevstiller.training import train_pool

LABELS = ["billing", "technical", "cancellation", "sales", "other"]


class SlowTeacher:
    """SyntheticTeacher plus a per-call sleep, standing in for Jev's network latency."""

    def __init__(self, inner, delay_s: float):
        self.inner, self.delay_s, self.name = inner, delay_s, "slow-synthetic"

    def classify(self, texts, task):
        time.sleep(self.delay_s)
        return self.inner.classify(texts, task)


def _callers(js, texts_per_caller: list[list[str]]) -> tuple[float, np.ndarray]:
    lat: list[float] = []
    lock = threading.Lock()

    def run(texts):
        mine = []
        for t in texts:
            t0 = time.perf_counter()
            js.classify(t)
            mine.append(time.perf_counter() - t0)
        with lock:
            lat.extend(mine)

    ts = [threading.Thread(target=run, args=(tx,)) for tx in texts_per_caller]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.perf_counter() - t0, np.array(lat) * 1000


def throughput(callers: int, per_caller: int, delay_s: float) -> None:
    task = Task(name="bench", instructions="Which team?", classes=LABELS)
    world = SyntheticWorld(LABELS, seed=1)
    teacher = SlowTeacher(SyntheticTeacher(world), delay_s)
    cfg = Config(min_calib_samples=10**9)          # never train: measure the request path alone
    with tempfile.TemporaryDirectory() as d:
        js = Jevstiller(task, teacher, d, config=cfg)
        work = [[t for t, _ in world.sample(per_caller)] for _ in range(callers)]
        wall, lat = _callers(js, work)
        js.close()
    n = callers * per_caller
    print(f"{callers} callers x {per_caller} requests, teacher {delay_s * 1000:.0f} ms: "
          f"{n / wall:,.0f} req/s   wall {wall:.2f}s   "
          f"p50 {np.median(lat):.0f} ms   p99 {np.quantile(lat, .99):.0f} ms   "
          f"(fully serial would be {1 / delay_s:,.0f} req/s)")


def _cfg(**kw) -> Config:
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=200)
    base.update(kw)
    return Config(**base)


def _serve(js, world, stop: threading.Event, callers: int = 8) -> np.ndarray:
    """Callers classify single texts back to back until `stop`; returns every latency in ms."""
    lat: list[float] = []
    lock = threading.Lock()

    def run():
        mine = []
        texts = iter([t for t, _ in world.sample(100_000)])
        while not stop.is_set():
            t0 = time.perf_counter()
            js.classify(next(texts))
            mine.append(time.perf_counter() - t0)
        with lock:
            lat.extend(mine)
    ts = [threading.Thread(target=run) for _ in range(callers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return np.array(lat) * 1000


def during_training(n_trainers: int = 5, rows: int = 30_000, repeats: int = 1) -> None:
    """P1.3: serving p50/p99 on one task while `n_trainers` other tasks train at once.

    Four setups: the fit in threads or in a process pool, each with BLAS uncapped (every core) or capped
    at Config.train_threads. Setups are interleaved `repeats` times and the median ratio is reported: a
    single run is at the mercy of whatever else the machine is doing."""
    world = SyntheticWorld(LABELS, seed=1)
    teacher = SyntheticTeacher(world)
    root = Path(tempfile.mkdtemp(prefix="jvs-bench-"))
    serve_task = Task(name="served", instructions="Which team?", classes=LABELS, target_agreement=0.95)
    served = Jevstiller(serve_task, teacher, root, config=_cfg(training="inline"))
    while not (served.status().production and served.status().shadow is None):
        served.classify_batch([t for t, _ in world.sample(200)])
    served.cfg.training = "manual"                          # freeze routing: measure serving alone
    print(f"served task: {served.status().production}, {served.mode}", flush=True)

    src = Task(name="train-src", instructions="Which team?", classes=LABELS, target_agreement=0.95)
    js = Jevstiller(src, teacher, root, config=_cfg(training="manual"))
    for _ in range(rows // 1000):
        js.classify_batch([t for t, _ in world.sample(1000)])
    js.close()
    print(f"{n_trainers} trainer tasks x {rows:,} rows\n", flush=True)

    def measure(executor: str, threads: int) -> float:
        pool = None
        if executor == "process":
            pool = ProcessPoolExecutor(n_trainers)
        elif executor.startswith("nice"):                   # nice<workers>: train_pool, low priority
            pool = train_pool(int(executor[4:]))
        trainers = []
        for k in range(n_trainers):
            t = Task(name=f"trainer-{k}", instructions="Which team?", classes=LABELS, target_agreement=0.95)
            d = root / f"{executor}-{threads}"
            shutil.rmtree(d / t.name, ignore_errors=True)
            shutil.copytree(root / src.name, d / t.name)
            trainers.append(Jevstiller(t, teacher, d, config=_cfg(training="manual", train_threads=threads),
                                       train_executor=pool))
        if pool:                                            # start the workers before measuring
            list(pool.map(abs, range(pool._max_workers)))
        stop = threading.Event()
        threading.Timer(5.0, stop.set).start()
        idle = _serve(served, world, stop)
        stop = threading.Event()
        t0 = time.perf_counter()
        fits = [threading.Thread(target=j.train_now) for j in trainers]
        for f in fits:
            f.start()
        watcher = threading.Thread(target=lambda: ([f.join() for f in fits], stop.set()))
        watcher.start()
        busy = _serve(served, world, stop)
        watcher.join()
        wall = time.perf_counter() - t0
        for j in trainers:
            j.close()
        if pool:
            pool.shutdown()
        cap = f"{threads} BLAS threads" if threads else "BLAS uncapped"
        print(f"{executor:>7} / {cap:<15}  {n_trainers} fits took {wall:5.1f}s   "
              f"idle p50 {np.median(idle):.2f} p99 {np.quantile(idle, .99):.2f} ms   "
              f"training p50 {np.median(busy):.2f} p99 {np.quantile(busy, .99):.2f} ms   "
              f"p99 x{np.quantile(busy, .99) / np.quantile(idle, .99):.2f}", flush=True)
        return float(np.quantile(busy, .99) / np.quantile(idle, .99))

    setups = [(e, t) for e in ("thread", "process") for t in (0, 2)] + [(f"nice{n_trainers}", 2), ("nice2", 2)]
    ratios: dict = {s: [] for s in setups}
    for _ in range(repeats):
        for s in setups:
            ratios[s].append(measure(*s))
    if repeats > 1:
        print("\nmedian p99 slowdown over", repeats, "runs:")
        for (e, t), r in ratios.items():
            print(f"  {e:>7} / {(f'{t} BLAS threads' if t else 'BLAS uncapped'):<15} x{np.median(r):.2f}   "
                  f"(runs: {', '.join(f'{x:.2f}' for x in r)})")
    served.close()
    shutil.rmtree(root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--callers", type=int, default=32)
    ap.add_argument("--per-caller", type=int, default=20)
    ap.add_argument("--delay-ms", type=float, default=50)
    ap.add_argument("--during-training", action="store_true")
    ap.add_argument("--trainers", type=int, default=5)
    ap.add_argument("--rows", type=int, default=30_000)
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    if a.during_training:
        during_training(a.trainers, a.rows, a.repeats)
    else:
        throughput(a.callers, a.per_caller, a.delay_ms / 1000)
