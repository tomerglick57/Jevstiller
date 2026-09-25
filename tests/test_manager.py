"""P2.1 / P2.5: TaskManager finds tasks by question, admits, caps, loads/unloads, and cleans up."""
import time

import pytest

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller._manager import task_key

LABELS = ["billing", "technical", "cancellation", "sales", "other"]
Q = "Which team handles this?"


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0, training="inline")
    base.update(kw)
    return Config(**base)


@pytest.fixture
def manager(tmp_path, teacher):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), target_agreement=0.95,
                    admission=Admission(min_requests=1), janitor_interval_s=3600)
    yield m
    m.close()


def _texts(world, n):
    return [t for t, _ in world.sample(n)]


def test_same_question_shares_a_task(manager, world):
    manager.classify("acme", Q, LABELS, _texts(world, 5))
    manager.classify("acme", Q, list(reversed(LABELS)), _texts(world, 5))    # another service, other order
    manager.classify("globex", Q, LABELS, _texts(world, 5))                  # another tenant
    manager.classify("acme", Q, {**{c: "" for c in LABELS}, "other": "anything else"}, _texts(world, 5))
    keys = {i.key: i for i in manager.tasks()}
    assert len(keys) == 3
    k = task_key("acme", manager.resolve("acme", Q, LABELS)[1])
    assert manager.engine(k).store.counts(manager.engine(k).task.version)["total"] == 10


def test_task_trains_and_serves_through_the_manager(manager, world):
    for _ in range(40):
        res = manager.classify("acme", Q, LABELS, _texts(world, 200))
        if any(r.source.startswith("student") for r in res):
            break
    assert any(r.source.startswith("student") for r in res)


def test_admission_passes_through_until_seen_enough(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=10),
                    janitor_interval_s=3600)
    res = m.classify("acme", Q, LABELS, _texts(world, 6))
    assert {r.routing_reason for r in res} == {"not_admitted"} and all(r.label for r in res)
    assert m.tasks() == [] and not any((tmp_path / "tasks").iterdir())
    res = m.classify("acme", Q, LABELS, _texts(world, 4))                     # 10 requests: admitted
    assert {r.routing_reason for r in res} == {"bootstrap"} and len(m.tasks()) == 1
    m.close()


def test_admission_window_resets():
    a = Admission(min_requests=3, window_s=10)
    assert not a.observe("k", 2, now=0) and not a.observe("k", 1, now=11) and a.observe("k", 2, now=12)


def test_tenant_task_limit(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=1),
                    max_tasks_per_tenant=1, janitor_interval_s=3600)
    m.classify("acme", Q, LABELS, _texts(world, 2))
    res = m.classify("acme", "A different question?", LABELS, _texts(world, 2))
    assert {r.routing_reason for r in res} == {"tenant_task_limit"} and len(m.tasks("acme")) == 1
    assert m.classify("globex", "A different question?", LABELS, _texts(world, 2))[0].routing_reason == "bootstrap"
    m.close()


def test_lru_unloads_idle_engines_and_reloads_them(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=1),
                    max_loaded=2, janitor_interval_s=3600)
    tenants = [f"t{i}" for i in range(5)]
    for t in tenants:
        m.classify(t, Q, LABELS, _texts(world, 3))
    m.sweep()
    assert len(m.loaded()) == 2 and m.unloads == 3
    assert set(m.loaded()) == {task_key(t, m.resolve(t, Q, LABELS)[1]) for t in tenants[-2:]}
    m.classify("t0", Q, LABELS, _texts(world, 3))                               # reload from disk
    k0 = task_key("t0", m.resolve("t0", Q, LABELS)[1])
    assert m.engine(k0).store.counts(m.engine(k0).task.version)["total"] == 6
    m.close()


def test_engines_in_use_are_not_unloaded(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=1),
                    max_loaded=0, janitor_interval_s=3600)
    m.classify("acme", Q, LABELS, _texts(world, 2))
    key = m.tasks()[0].key
    m._acquire(key)                                                             # a request in flight
    assert m._unload_one() is False and m.loaded() == [key]
    m._release(key)
    m.sweep()
    assert m.loaded() == []
    m.close()


def test_memory_limit(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=1),
                    max_memory_mb=0.001, janitor_interval_s=3600)
    for _ in range(40):
        m.classify("acme", Q, LABELS, _texts(world, 200))
        if m.memory_mb() > 0:
            break
    assert m.memory_mb() > 0.001
    m.sweep()
    assert m.memory_mb() == 0 and m.loaded() == []
    m.close()


def test_restart_finds_existing_tasks(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=1))
    m.classify("acme", Q, LABELS, _texts(world, 3))
    m.close()
    m2 = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=100))
    assert len(m2.tasks()) == 1
    assert m2.classify("acme", Q, LABELS, _texts(world, 1))[0].routing_reason == "bootstrap"   # no re-admission
    m2.close()


def test_delete_and_idle_cleanup(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), admission=Admission(min_requests=1),
                    idle_ttl_s=3600, janitor_interval_s=3600)
    m.classify("acme", Q, LABELS, _texts(world, 3))
    m.classify("globex", Q, LABELS, _texts(world, 3))
    k_acme = task_key("acme", m.resolve("acme", Q, LABELS)[1])
    assert m.delete(k_acme) and not (tmp_path / "tasks" / k_acme).exists()
    [info] = m.tasks()
    info.last_seen = time.time() - 7200
    m.sweep()
    assert m.tasks() == [] and not any((tmp_path / "tasks").iterdir())
    m.close()


def test_per_call_teacher(manager, world, teacher):
    class Spy:
        name = "spy"

        def __init__(self):
            self.n = 0

        def classify(self, texts, task):
            self.n += len(texts)
            return teacher.classify(texts, task)
    spy = Spy()
    manager.classify("acme", Q, LABELS, _texts(world, 4), teacher=spy)
    assert spy.n == 4


@pytest.mark.parametrize("training", ["inline", "background", "manual"])
def test_unloaded_engines_are_freed_without_the_cycle_collector(tmp_path, teacher, world, training):
    """An unloaded engine holds megabytes of arrays. If a reference cycle kept it, it would stay until a full
    garbage collection, and with tasks loading and unloading all the time memory would pile up (P6.6)."""
    import gc
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    from jevstiller._scheduler import TrainScheduler
    sched = TrainScheduler(pool_factory=lambda: ThreadPoolExecutor(1))
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=256), _cfg(training=training, min_new_samples=500),
                    admission=Admission(min_requests=1), max_loaded=0, janitor_interval_s=3600, train_executor=sched)
    for _ in range(12):
        m.classify("acme", Q, LABELS, _texts(world, 200))
    key = m.tasks()[0].key
    e = m.engine(key)
    e.drain()
    ref = weakref.ref(e)
    del e
    gc.collect()
    gc.disable()
    try:
        for _ in range(100):
            m.sweep()
            if key not in m.loaded():
                break
            time.sleep(0.05)
        assert key not in m.loaded() and ref() is None
    finally:
        gc.enable()
        m.close()
        sched.shutdown()


def test_progress_survives_unload_and_reload(tmp_path, teacher, world):
    """A task that is often unloaded (more active tasks than max_loaded) still gets its shadow judged, is not
    retrained on every reload, and still reaches its drift checks."""
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(training="manual"),
                    admission=Admission(min_requests=1), max_loaded=0, janitor_interval_s=3600)
    for _ in range(12):                                  # enough calibration rows to certify the budget
        m.classify("acme", Q, LABELS, _texts(world, 200))
    key = m.tasks()[0].key
    e = m.engine(key)
    e.train_now()                                        # -> shadow
    shadow, start, trained = e._shadow.name, e._shadow_started_id, e._last_train_id
    e._audit_new = 7
    m.sweep()
    assert key not in m.loaded()
    e = m.engine(key)                                    # reloaded
    assert e._shadow.name == shadow and e._shadow_started_id == start and start > 0
    assert e._last_train_id == trained > 0 and e._audit_new == 7
    m.close()


def test_the_loaded_cap_holds_under_concurrent_loads(tmp_path, teacher, world):
    """Loads hold the cap themselves: with only the janitor enforcing it, a steady stream of loads outran its
    one-at-a-time unloads and hundreds of tasks stayed loaded against max_loaded = 50 (benchmarks/manager.py)."""
    import threading
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=128), _cfg(training="background", maintenance_interval_s=0.5),
                    admission=Admission(min_requests=1), max_loaded=4, janitor_interval_s=3600)
    questions = [f"Question {i}?" for i in range(40)]
    peak, lock = [0], threading.Lock()

    def worker(i):
        for j in range(60):
            m.classify("acme", questions[(i * 7 + j) % 40], LABELS, _texts(world, 1))
            with lock:
                peak[0] = max(peak[0], len(m.loaded()))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # the cap, plus requests in flight and engines mid-maintenance (neither can be unloaded), never all 40
    assert m.unloads > 0 and peak[0] <= 4 + 8 + 4, peak[0]
    m.sweep()
    assert len(m.loaded()) <= 4
    m.close()


def test_a_requested_maintenance_pass_does_not_pin_an_engine(tmp_path, teacher, world):
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=128), _cfg(training="background", maintenance_interval_s=60),
                    admission=Admission(min_requests=1), max_loaded=0, janitor_interval_s=3600)
    m.classify("acme", Q, LABELS, _texts(world, 3))       # requests a first pass
    e = m.engine(m.tasks()[0].key)
    deadline = time.time() + 30
    while (e._done < 1 or e._maint.locked()) and time.time() < deadline:
        time.sleep(0.01)                                  # until it has run; the worker then rests 60 s
    m.classify("acme", Q, LABELS, _texts(world, 3))       # so this pass stays requested, not running
    assert not e._busy()
    m.sweep()
    assert m.loaded() == []
    m.close()
