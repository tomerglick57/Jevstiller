"""P2.1 / P2.3: many registered tasks, a few hot ones, bounded memory.

    python benchmarks/manager.py                     # 1,000 tasks, 50 hot, max_loaded=50
    python benchmarks/manager.py --tasks 2000 --max-loaded 100

One task is trained for real; the others are clones of it under different tenants (same question, so the
same Task.version and a valid lineage), each with its own empty sample store. Version files are hard-linked,
not copied, so the setup stays small on disk. Traffic: 90% to the hot tasks, 10% spread over the rest, one
text per request, from 8 threads. Reports resident memory, loads/unloads, and hit vs miss latency (a miss
loads the task from disk: open its SQLite store, read the registry, load the production version).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from jevstiller import Admission, Config, HashEncoder, Jevstiller, SyntheticTeacher, SyntheticWorld, TaskManager
from jevstiller.manager import TaskInfo, task_key
from jevstiller.task import Task

LABELS = ["billing", "technical", "cancellation", "sales", "other"]
Q = "Which team should handle this message?"


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def build(root: Path, n_tasks: int, dim: int, rows: int) -> None:
    world = SyntheticWorld(LABELS, seed=1)
    teacher = SyntheticTeacher(world)
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, training="inline")
    src = root / "src"
    task = Task("src", Q, LABELS, 0.95)
    js = Jevstiller(task, teacher, src, encoder=HashEncoder(dim=dim), config=cfg)
    for _ in range(rows // 500):
        js.classify_batch([t for t, _ in world.sample(500)])
    while js.status().shadow:                                # let the last candidate be judged
        js.classify_batch([t for t, _ in world.sample(500)])
    prod = js.status().production
    js.close()
    vroot = src / "src" / "versions"
    tasks = root / "tasks"
    for i in range(n_tasks):
        tenant = f"tenant-{i}"
        key = task_key(tenant, task)
        d = tasks / key
        (d / "versions").mkdir(parents=True)
        shutil.copy(vroot / "registry.json", d / "versions" / "registry.json")
        for vdir in vroot.iterdir():
            if vdir.is_dir():
                (d / "versions" / vdir.name).mkdir()
                for f in vdir.iterdir():
                    os.link(f, d / "versions" / vdir.name / f.name)
        now = time.time()
        info = TaskInfo(key, tenant, Q, {c: "" for c in LABELS}, 0.95, now, now)
        (d / "task.json").write_text(json.dumps(dataclasses.asdict(info)))
    print(f"built {n_tasks:,} tasks (production {prod}, dim {dim}) in {root}", flush=True)


def run(root: Path, n_tasks: int, hot: int, max_loaded: int, requests: int, dim: int, threads: int,
        cold_share: float = 0.1) -> None:
    world = SyntheticWorld(LABELS, seed=2)
    teacher = SyntheticTeacher(world)
    cfg = Config(audit_rate=0.02, min_calib_samples=10**9, training="background")
    base = rss_mb()
    m = TaskManager(root, teacher, HashEncoder(dim=dim), cfg, target_agreement=0.95,
                    admission=Admission(min_requests=1), max_loaded=max_loaded, janitor_interval_s=0.2)
    tenants = [f"tenant-{i}" for i in range(n_tasks)]
    texts = [t for t, _ in world.sample(20_000)]
    hits: list[float] = []
    misses: list[float] = []                 # reloads of a task opened before in this run
    firsts: list[float] = []                 # first open: the task's sample store is created
    opened: set[str] = set()
    peak = [base]
    timeline: list[tuple[float, float]] = []
    lock = threading.Lock()

    def worker(seed: int, n: int) -> None:
        rng = random.Random(seed)
        h, ms, fs = [], [], []
        for _ in range(n):
            cold = rng.random() < cold_share and hot < n_tasks
            t = tenants[rng.randrange(hot, n_tasks)] if cold else tenants[rng.randrange(hot)]
            key = task_key(t, Task("_", Q, LABELS, 0.95))
            was_loaded = key in m._engines
            with lock:
                first = key not in opened
                opened.add(key)
            t0 = time.perf_counter()
            m.classify(t, Q, LABELS, [rng.choice(texts)])
            (h if was_loaded else fs if first else ms).append((time.perf_counter() - t0) * 1000)
        with lock:
            hits.extend(h)
            misses.extend(ms)
            firsts.extend(fs)

    def sampler(stop: threading.Event) -> None:
        t_start = time.perf_counter()
        while not stop.is_set():
            r = rss_mb()
            peak[0] = max(peak[0], r)
            timeline.append((time.perf_counter() - t_start, r))
            time.sleep(0.2)
    stop = threading.Event()
    threading.Thread(target=sampler, args=(stop,), daemon=True).start()
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(s, requests // threads)) for s in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    m.sweep()
    stop.set()
    per_task_disk = sum(f.stat().st_size for f in (root / "tasks" / m.loaded()[0]).glob("samples.sqlite*"))
    print(f"{n_tasks:,} tasks, {hot} hot ({1 - cold_share:.0%} of traffic), max_loaded={max_loaded}, {requests:,} requests, "
          f"{threads} threads, {wall:.1f}s\n"
          f"  loaded now {len(m.loaded())}   loads {m.loads:,}   unloads {m.unloads:,}   "
          f"student memory {m.memory_mb():.1f} MB\n"
          f"  RSS: before {base:.0f} MB   peak {peak[0]:.0f} MB   end {rss_mb():.0f} MB\n"
          f"  hit  latency p50 {np.median(hits):.2f} ms  p99 {np.quantile(hits, .99):.2f} ms  (n={len(hits):,})\n"
          f"  reload latency p50 {np.median(misses):.2f} ms  p99 {np.quantile(misses, .99):.2f} ms  "
          f"(n={len(misses):,})   <- task unloaded earlier: open store + registry + load version\n"
          f"  first-open latency p50 {np.median(firsts):.2f} ms  p99 {np.quantile(firsts, .99):.2f} ms  "
          f"(n={len(firsts):,})   <- also creates the task's SQLite store\n"
          f"  RSS over the run (s: MB): " + "  ".join(f"{t:.0f}: {r:.0f}" for t, r in timeline[::max(1, len(timeline) // 8)])
          + "\n"
          f"  sample store of a lightly used task on disk: {per_task_disk / 1024:.0f} KiB", flush=True)
    m.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=1000)
    ap.add_argument("--hot", type=int, default=50)
    ap.add_argument("--max-loaded", type=int, default=50)
    ap.add_argument("--requests", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=384, help="embedding size (bge-small: 384, bge-base: 768)")
    ap.add_argument("--rows", type=int, default=20_000, help="training rows of the cloned task")
    ap.add_argument("--cold-share", type=float, default=0.1, help="share of requests to the non-hot tasks")
    a = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="jvs-manager-"))
    try:
        build(root, a.tasks, a.dim, a.rows)
        run(root, a.tasks, a.hot, a.max_loaded, a.requests, a.dim, a.threads, a.cold_share)
    finally:
        shutil.rmtree(root, ignore_errors=True)
