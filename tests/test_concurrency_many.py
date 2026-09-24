"""P6.3: many tasks x many threads (tasks unloading and reloading under load), and many concurrent clients
through the proxy. Nothing is lost, nothing stays checked out, and every answer is well formed."""
import asyncio
import random
import threading
from collections import Counter

import httpx
import pytest

from jevstiller import Admission, Config, HashEncoder, TaskManager

ts = pytest.importorskip("typesafe_sdk")
from test_proxy import GOOD, LABELS, OTHER, stack  # noqa: E402,F401,F811  (stack is a fixture)


def test_many_tasks_many_threads_with_unloads(tmp_path, teacher, world):
    # inline: training runs in the request threads, so engines are idle between requests and can unload
    cfg = Config(audit_rate=0.05, min_train_samples=300, min_samples_per_class=15, min_calib_samples=100,
                 min_new_samples=800, shadow_min_samples=100, seed=0, training="inline")
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=256), cfg, admission=Admission(min_requests=1),
                    max_loaded=3, janitor_interval_s=0.2)
    tasks = [(f"tenant{t}", f"Which team handles this? ({q})") for t in range(3) for q in range(4)]
    texts = [t for t, _ in world.sample(4000)]
    sent: Counter = Counter()
    lock, errors = threading.Lock(), []

    def worker(seed):
        rng = random.Random(seed)
        try:
            for _ in range(150):
                tenant, q = rng.choice(tasks)
                batch = rng.sample(texts, rng.randint(1, 5))
                res = m.classify(tenant, q, LABELS, batch)
                assert len(res) == len(batch) and all(r.label in LABELS for r in res)
                with lock:
                    sent[(tenant, q)] += len(batch)
        except Exception as e:                           # pragma: no cover - reported below
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert not errors, errors[:3]
    assert m.loads > len(tasks) and m.unloads > 0        # tasks were unloaded and reloaded under load
    assert sum(m._inflight.values()) == 0
    for tenant, q in tasks:                              # every item was recorded exactly once
        key = m.resolve(tenant, q, LABELS)[0]
        e = m.engine(key)
        e.drain()
        assert e.store.counts(e.task.version)["total"] == sent[(tenant, q)], (tenant, q)
    m.sweep()
    assert len(m.loaded()) <= 3
    m.close()


def test_many_concurrent_clients_through_the_proxy(stack):  # noqa: F811
    proxy, jev, m, app = stack
    questions = [{"type": "choice", "instructions": f"Which team handles this? ({i})",
                  "criteria": {c: "" for c in LABELS}} for i in range(3)]

    async def go():
        rng = random.Random(0)
        async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=100)) as c:
            async def one(i):
                q = i % 3
                r = await c.post(f"{proxy}/v1/systemone", json={
                    "state": " ".join(f"w{rng.randint(0, 200)}" for _ in range(6)), "model": "jev-latest",
                    "questions": {"label": questions[q]}},
                    headers={"Authorization": f"Bearer {GOOD if i % 2 else OTHER}"})
                return q, r
            return await asyncio.gather(*(one(i) for i in range(300)))
    results = asyncio.run(go())
    assert all(r.status_code == 200 for _, r in results), {r.status_code for _, r in results}
    assert all(r.json()["answers"]["label"]["choice"] in LABELS for _, r in results)
    per_q = Counter(q for q, _ in results)
    sources = Counter(r.headers["x-jevstiller-source"] for _, r in results)
    assert sources["upstream"] == jev.calls and sum(sources.values()) == 300
    assert sum(m._inflight.values()) == 0
    for q, spec in enumerate(questions):
        key = m.resolve("default", spec["instructions"], spec["criteria"], model="jev-latest")[0]
        e = m.engine(key)
        e.store.flush()
        assert e.store.counts(e.task.version)["total"] == per_q[q]
