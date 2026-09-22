"""End to end: the loop trains, shadows, promotes, routes, and the agreement bound holds."""

from jevstiller import Config, HashEncoder, Jevstiller


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0)
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
