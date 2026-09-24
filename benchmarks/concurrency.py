"""Throughput and latency under concurrent callers, with a slow fake teacher. No network, no key.

    python benchmarks/concurrency.py                   # throughput: 32 callers, one task, 50 ms teacher
    python benchmarks/concurrency.py --during-training # serving p50/p99 while a candidate trains

The teacher sleeps like a network call, so the number that matters is how many of those sleeps overlap.
"""
from __future__ import annotations

import argparse
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from jevstiller import Config, Jevstiller, SyntheticTeacher, SyntheticWorld, Task

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


def during_training(executor: str) -> None:
    task = Task(name="bench", instructions="Which team?", classes=LABELS, target_agreement=0.95)
    world = SyntheticWorld(LABELS, seed=1)
    teacher = SyntheticTeacher(world)
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=10**9, training="manual")
    pool = ProcessPoolExecutor(1) if executor == "process" else None
    with tempfile.TemporaryDirectory() as d:
        js = Jevstiller(task, teacher, d, config=cfg, train_executor=pool)
        for _ in range(60):                           # a large training set so the fit takes a while
            js.classify_batch([t for t, _ in world.sample(500)])
        work = [[t for t, _ in world.sample(200)] for _ in range(8)]
        _, base = _callers(js, work)
        tr = threading.Thread(target=js.train_now)
        t0 = time.perf_counter()
        tr.start()
        lat = []
        while tr.is_alive():
            _, l = _callers(js, [[t for t, _ in world.sample(50)] for _ in range(8)])
            lat.append(l)
        tr.join()
        train_s = time.perf_counter() - t0
        js.close()
    if pool:
        pool.shutdown()
    during = np.concatenate(lat) if lat else np.array([np.nan])
    print(f"executor={executor}: training took {train_s:.1f}s\n"
          f"  idle      p50 {np.median(base):.2f} ms   p99 {np.quantile(base, .99):.2f} ms\n"
          f"  training  p50 {np.median(during):.2f} ms   p99 {np.quantile(during, .99):.2f} ms   (n={len(during)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--callers", type=int, default=32)
    ap.add_argument("--per-caller", type=int, default=20)
    ap.add_argument("--delay-ms", type=float, default=50)
    ap.add_argument("--during-training", action="store_true")
    a = ap.parse_args()
    if a.during_training:
        during_training("thread")
        during_training("process")
    else:
        throughput(a.callers, a.per_caller, a.delay_ms / 1000)
