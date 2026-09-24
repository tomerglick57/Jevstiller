"""P1: teacher calls overlap, training stays off the request path, audit rate holds, failures stay local."""
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

import jevstiller.core as core
from jevstiller import Config, Jevstiller, TeacherError


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0)
    base.update(kw)
    return Config(**base)


class SlowTeacher:
    def __init__(self, inner, delay_s):
        self.inner, self.delay_s, self.name = inner, delay_s, "slow"

    def classify(self, texts, task):
        time.sleep(self.delay_s)
        return self.inner.classify(texts, task)


class FlakyTeacher:
    """Fails the items containing `bad`; raises for the whole call if any text contains `down`."""

    name = "flaky"

    def __init__(self, inner):
        self.inner = inner

    def classify(self, texts, task):
        if any("down" in t for t in texts):
            raise ConnectionError("teacher unreachable")
        outs = self.inner.classify(texts, task)
        return [TimeoutError(t) if "bad" in t else o for t, o in zip(texts, outs, strict=True)]


def _threads(n, fn):
    ts = [threading.Thread(target=fn) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]


def test_teacher_calls_overlap(tmp_path, task, world, teacher):
    js = Jevstiller(task, SlowTeacher(teacher, 0.1), tmp_path, config=_cfg(training="manual"))
    t0 = time.perf_counter()
    _threads(16, lambda: [js.classify(t) for t, _ in world.sample(4)])
    wall = time.perf_counter() - t0
    js.close()
    assert wall < 6.4 / 4, f"64 calls x 100 ms took {wall:.2f}s; serial would be 6.4s"


def test_training_runs_off_the_request_path(tmp_path, task, world, teacher, monkeypatch):
    fit_threads = []
    real = core.fit_candidate

    def spy(*a, **kw):
        fit_threads.append(threading.get_ident())
        return real(*a, **kw)
    monkeypatch.setattr(core, "fit_candidate", spy)
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="background", maintenance_interval_s=0.0))
    for _ in range(40):
        js.classify_batch([t for t, _ in world.sample(200)])
        assert js.drain(timeout=60)
        if js.status().production:
            break
    assert js.status().production, js.status().report()
    assert fit_threads and threading.get_ident() not in fit_threads
    js.close()


def test_close_commits_queued_records(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="background"))
    _threads(8, lambda: [js.classify_batch([t for t, _ in world.sample(25)]) for _ in range(4)])
    js.close()
    js2 = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual"))
    assert js2.store.counts(task.version)["total"] == 8 * 4 * 25
    js2.close()


def test_audit_rate_holds_under_concurrency(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="inline"))
    for _ in range(40):
        js.classify_batch([t for t, _ in world.sample(200)])
        if js.status().production and js.status().shadow is None:
            break
    assert js.mode == "cascade", js.status().report()
    js.cfg.training = "manual"                           # freeze the routing state while we measure
    rate = js.audit_rate
    start = js.store.max_id()
    _threads(8, lambda: [js.classify_batch([t for t, _ in world.sample(50)]) for _ in range(20)])
    n, audits = js.store.db.execute(
        "SELECT COUNT(*), SUM(channel='audit') FROM samples WHERE id > ?", (start,)).fetchone()
    assert n == 8 * 20 * 50
    sd = (n * rate * (1 - rate)) ** 0.5
    assert abs(audits - n * rate) < 5 * sd, (audits, n * rate)
    js.close()


def test_one_failed_teacher_item_does_not_fail_the_batch(tmp_path, task, world, teacher):
    js = Jevstiller(task, FlakyTeacher(teacher), tmp_path, config=_cfg(training="manual"))
    texts = ["w1 w2 good", "w3 bad", "w4 w5 good"]
    with pytest.raises(TeacherError) as ei:
        js.classify_batch(texts)
    err = ei.value
    assert set(err.errors) == {1} and isinstance(err.errors[1], TimeoutError)
    assert err.results[0].label and err.results[1] is None and err.results[2].label
    assert js.store.counts(task.version)["total"] == 2          # the failed item is not recorded

    res = js.classify_batch(texts, errors="return")
    assert res[1].label is None and res[1].source == "error" and isinstance(res[1].error, TimeoutError)
    assert res[0].source == "teacher" and res[2].source == "teacher"
    assert js.status().teacher_errors == 2
    js.close()


def test_teacher_outage_fails_only_teacher_items(tmp_path, task, world, teacher):
    js = Jevstiller(task, FlakyTeacher(teacher), tmp_path, config=_cfg(training="manual"))
    with pytest.raises(TeacherError) as ei:
        js.classify("down")
    assert isinstance(ei.value.__cause__, ConnectionError)
    assert js.store.counts(task.version)["total"] == 0
    js.close()


def test_train_in_a_worker_process(tmp_path, task, world, teacher):
    with ProcessPoolExecutor(1) as pool:
        js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual"), train_executor=pool)
        for _ in range(10):
            js.classify_batch([t for t, _ in world.sample(200)])
        rep = js.train_now()
        js.close()
    assert rep.accepted and rep.version == "student:v1"


def test_config_and_mode_validation(tmp_path, task, teacher):
    with pytest.raises(ValueError):
        Config(training="sometimes")
    with pytest.raises(ValueError):
        Config(mode="shadow")
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual"))
    with pytest.raises(ValueError):
        js.set_mode("hedge")
    with pytest.raises(ValueError):
        js.promote("student:v9")
    js.close()
