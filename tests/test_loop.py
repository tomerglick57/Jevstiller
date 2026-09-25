"""End to end: the loop trains, shadows, promotes, routes, and the agreement bound holds."""

from jevstiller import Config, HashEncoder, Jevstiller


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0, training="inline")
    base.update(kw)
    return Config(**base)


def test_loop_promotes_and_meets_budget(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=512), config=_cfg())
    stream = world.sample(6000)
    texts = [t for t, _ in stream]
    # what the teacher would say on every text (it is deterministic), for checking the contract
    truth = {t: teacher.answer(t, task).label for t in texts}

    for i in range(0, len(texts), 100):
        js.classify_batch(texts[i:i + 100])

    st = js.status()
    assert st.production is not None, st.report()
    assert st.mode == "cascade"

    # measure the contract on a fresh tail
    tail = [t for t, _ in world.sample(2000)]
    res = js.classify_batch(tail)
    student_served = sum(r.source != "teacher" for r in res)
    disagree = sum(r.source != "teacher" and r.label != teacher.answer(t, task).label for t, r in zip(tail, res, strict=False))
    assert student_served / len(tail) > 0.3, js.status().report()
    assert disagree / len(tail) <= task.budget + 0.01, js.status().report()
    assert all(r.label == truth.get(t, r.label) or r.source != "teacher" for t, r in zip(tail, res, strict=False))

    st = js.status()
    assert st.teacher_calls_avoided > 0 and st.audit_n > 0
    # audit rows carry importance weights, deferred rows do not
    w = js.store.db.execute("SELECT channel, MIN(weight), MAX(weight) FROM samples GROUP BY channel").fetchall()
    w = {c: (lo, hi) for c, lo, hi in w}
    assert w["deferred"] == (1.0, 1.0) and w["audit"][0] > 1.0
    print(st.report())


def test_persistence_and_rollback(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg())
    for _i in range(0, 4000, 200):
        js.classify_batch([t for t, _ in world.sample(200)])
    prod = js.status().production
    assert prod
    js.close()
    js2 = Jevstiller(task, teacher, tmp_path, config=_cfg())
    assert js2.status().production == prod and js2.mode == "cascade"
    js2.set_mode("teacher_only")
    r = js2.classify("w1 w2 w3")
    assert r.source == "teacher" and r.routing_reason == "fallback"


def test_drift_triggers_fallback(tmp_path, task, world, teacher):
    from jevstiller.teachers import SyntheticTeacher, SyntheticWorld
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(audit_rate=0.3, drift_margin=0.0))
    for _i in range(0, 4000, 200):
        js.classify_batch([t for t, _ in world.sample(200)])
    assert js.mode == "cascade"
    # the teacher changes its mind: a different world's vocabulary mapping
    js.teacher = SyntheticTeacher(SyntheticWorld(task.labels, seed=99), temperature=0.7, noise=0.5)
    for _i in range(0, 3000, 200):
        js.classify_batch([t for t, _ in world.sample(200)])
        if js.mode == "teacher_only":
            break
    assert js.mode == "teacher_only"
    assert any(e["kind"] == "fallback" for e in js.status().events)


class _Swapped:
    """The same teacher (same name, same model) with its answers silently rotated among the labels."""

    name = "synthetic"

    def __init__(self, inner, labels):
        self.inner, self.perm = inner, dict(zip(labels, labels[1:] + labels[:1], strict=True))

    def classify(self, texts, task):
        outs = self.inner.classify(texts, task)
        for o in outs:
            o.label, o.probs = self.perm[o.label], {self.perm[c]: p for c, p in o.probs.items()}
        return outs


def test_recovers_from_a_silent_drift(tmp_path, task, world, teacher):
    """The teacher changes its answers without changing its name: fall back, restart the training data at the
    fallback, and serve locally again from a student trained only on the new answers."""
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(min_new_samples=3000, drift_min_samples=60))
    for _ in range(30):
        js.classify_batch([t for t, _ in world.sample(200)])
    old = js.status().production
    assert old and js.mode == "cascade"
    new = _Swapped(teacher, task.labels)
    js.teacher = new
    for _ in range(30):
        js.classify_batch([t for t, _ in world.sample(200)])
        if js.mode == "teacher_only":
            break
    [fb] = [e for e in js.status().events if e["kind"] == "fallback"]
    assert fb["since_id"] > 0 and js._since_id == fb["since_id"]
    js.close()

    js = Jevstiller(task, new, tmp_path, config=_cfg(min_new_samples=3000, drift_min_samples=60))
    assert js.mode == "teacher_only" and js._since_id == fb["since_id"]     # the fallback survives a restart
    for _ in range(40):
        js.classify_batch([t for t, _ in world.sample(200)])
        if js.mode == "cascade":
            break
    st = js.status()
    assert js.mode == "cascade" and st.production != old, st.report()
    meta = js.registry.load(st.production).meta
    n_new = js.store.db.execute("SELECT COUNT(*) FROM samples WHERE id>? AND split='train' "
                                "AND teacher_label IS NOT NULL", (fb["since_id"],)).fetchone()[0]
    assert meta["since_id"] == fb["since_id"] and meta["n_train"] <= n_new
    tail = [t for t, _ in world.sample(2000)]
    res = js.classify_batch(tail)
    served = [(t, r) for t, r in zip(tail, res, strict=True) if r.source != "teacher"]
    assert len(served) / len(tail) > 0.5, st.report()
    wrong = sum(r.label != new.classify([t], task)[0].label for t, r in served)
    assert wrong / len(tail) <= task.budget + 0.01, st.report()
    js.close()


def test_operator_can_end_a_drift_fallback(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg())
    for _ in range(20):
        js.classify_batch([t for t, _ in world.sample(200)])
    assert js.mode == "cascade"
    js.store.event("fallback", reason="test", since_id=js.store.max_id())    # as _check_drift records it
    js.close()
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual"))
    assert js.mode == "teacher_only"
    js.set_mode("auto")                                  # the operator overrides: serve the old student
    assert js.mode == "cascade" and js.store.last_event(("fallback", "fallback_cleared"))["kind"] == "fallback_cleared"
    js.close()
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual"))
    assert js.mode == "cascade"                          # and that survives a restart
    js.close()


def test_without_a_student_a_failed_candidate_backs_off(tmp_path, task, world, teacher):
    """With no production student for the current data, a candidate that didn't make it is retried after 25% more
    data (capped at min_new_samples), not every 100 rows: a hard task would otherwise train non-stop."""
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual", min_new_samples=10**6))
    for _ in range(5):                                   # ~200 calibration rows (a random 20% per request) >= 150
        js.classify_batch([t for t, _ in world.sample(200)])
    assert js._should_train()
    js.train_now()
    js._shadow = None                                    # as if it had failed shadow
    tried = js._last_train_id
    js.classify_batch([t for t, _ in world.sample(tried // 4 - 50)])
    assert not js._should_train()
    js.classify_batch([t for t, _ in world.sample(100)])
    assert js._should_train()
    js.close()


def test_fits_use_only_the_most_recent_rows(tmp_path, task, world, teacher):
    """A fit reads at most max_train_samples / max_calib_samples rows, the newest: otherwise a busy task's retrains
    read (and hold in memory) its whole history, growing without bound over months."""
    import numpy as np
    import pytest

    from jevstiller import Config
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual", max_train_samples=500,
                                                          max_calib_samples=200))
    for _ in range(10):
        js.classify_batch([t for t, _ in world.sample(200)])
    rep = js.train_now()
    assert rep.n_train == 500 and rep.n_calib == 200
    args = (task.version, js.encoder.id, task.labels, js.encoder.dim)
    X_all = js.store.training_set(*args)[0]
    assert len(X_all) > 500 and np.array_equal(js.store.training_set(*args, limit=500)[0], X_all[-500:])
    js.close()
    with pytest.raises(ValueError, match="max_train_samples"):
        Config(min_train_samples=1000, max_train_samples=100)
