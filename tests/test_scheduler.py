"""P2.6: shared training scheduler (fairness, priority, retries, broken pools) and async training."""
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

import pytest

from jevstiller import Config, Jevstiller, TrainScheduler


def _threads(workers=1, **kw):
    return TrainScheduler(workers=workers, pool_factory=lambda: ThreadPoolExecutor(workers), **kw)


def _blocker(s: TrainScheduler, tenant: str = "x"):
    gate = threading.Event()
    fut = s.for_task("blocker", tenant).submit(gate.wait, 30)
    time.sleep(0.05)
    return gate, fut


def test_round_robin_across_tenants():
    s = _threads()
    gate, _ = _blocker(s)
    order = []
    futs = [s.for_task(f"a{i}", "A").submit(order.append, f"A{i}") for i in range(5)]
    futs.append(s.for_task("b0", "B").submit(order.append, "B0"))
    gate.set()
    [f.result(5) for f in futs]
    assert order[:2] == ["A0", "B0"], order                 # B is not stuck behind A's backlog
    s.shutdown()


def test_priority_within_a_tenant():
    s = _threads()
    gate, _ = _blocker(s, "A")
    order = []
    futs = [s.for_task(f"k{p}", "A", priority=lambda p=p: p).submit(order.append, p) for p in (1, 5, 3)]
    gate.set()
    [f.result(5) for f in futs]
    assert order == [5, 3, 1]
    s.shutdown()


def test_retry_then_success_and_exhaustion():
    s = _threads(max_retries=2, retry_backoff_s=0.01)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise OSError("transient")
        return "ok"
    assert s.submit(flaky).result(5) == "ok" and len(calls) == 2

    def broken():
        raise ValueError("always")
    with pytest.raises(ValueError):
        s.submit(broken).result(5)
    st = s.stats()
    assert st["retried"] == 3 and st["failed"] == 1 and st["completed"] == 1
    s.shutdown()


def test_cancel_while_queued():
    s = _threads()
    gate, _ = _blocker(s)
    ran = []
    f = s.submit(ran.append, 1)
    assert f.cancel()
    gate.set()
    s.submit(lambda: None).result(5)
    assert ran == []
    s.shutdown()


def _die_once(flag):
    if not os.path.exists(flag):
        open(flag, "w").close()
        os._exit(1)                                          # the worker process dies, the pool breaks
    return 42


def test_broken_process_pool_is_replaced(tmp_path):
    s = TrainScheduler(workers=1, retry_backoff_s=0.01, pool_factory=lambda: ProcessPoolExecutor(1))
    assert s.submit(_die_once, str(tmp_path / "flag")).result(60) == 42
    assert s.stats()["pool_restarts"] == 1
    s.shutdown()


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0,
                training="background", maintenance_interval_s=0.0)
    base.update(kw)
    return Config(**base)


def test_maintenance_keeps_running_while_training_waits(tmp_path, task, world, teacher):
    s = _threads()
    gate, _ = _blocker(s)                                   # the only worker is busy elsewhere
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(), train_executor=s.for_task("t", "acme"))
    for _ in range(10):
        js.classify_batch([t for t, _ in world.sample(200)])
        assert js.drain(timeout=10)                         # passes complete: nothing waits on the job
    assert js._pending is not None and js.busy() and js.status().production is None
    assert any(e["kind"] == "training_queued" for e in js.status().events)
    gate.set()
    js._pending[0].result(timeout=120)                      # the job runs once the worker is free ...
    for _ in range(40):                                     # ... and the next passes adopt and judge it
        js.classify_batch([t for t, _ in world.sample(200)])
        js.drain(timeout=30)
        if js.status().production:
            break
    assert js.status().production, js.status().report()
    js.close()
    s.shutdown()


def test_train_now_waits_for_a_queued_job(tmp_path, task, world, teacher):
    s = _threads()
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual"), train_executor=s.for_task("t"))
    for _ in range(10):
        js.classify_batch([t for t, _ in world.sample(200)])
    gate, _ = _blocker(s)
    js._start_train(wait=False)
    threading.Timer(0.2, gate.set).start()
    rep = js.train_now()
    assert rep.accepted and js._pending is None
    js.close()
    s.shutdown()


def test_failed_background_training_is_retried_later(tmp_path, task, world, teacher, monkeypatch):
    import jevstiller.core as core
    real, calls = core.run_fit_job, []

    def fails_once(job):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("worker ran out of memory")
        return real(job)
    monkeypatch.setattr(core, "run_fit_job", fails_once)
    s = _threads(max_retries=0)
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(), train_executor=s.for_task("t"))
    for _ in range(80):
        js.classify_batch([t for t, _ in world.sample(200)])
        js.drain(timeout=30)
        pending = js._pending
        if pending is not None:
            wait([pending[0]], timeout=120)                 # (the first one fails: do not re-raise it)
        if js.status().production:
            break
    kinds = [e["kind"] for e in js.store.events(limit=100)]
    assert "train_failed" in kinds and js.status().production
    js.close()
    s.shutdown()
