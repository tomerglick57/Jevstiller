"""Task.confidence_floor: a local answer also counts as disagreeing when the teacher would have been less confident
than the caller's floor, local answers report at least the floor, and changing the floor retrains."""
import json
from dataclasses import replace

import httpx
import numpy as np
import pytest
from test_loop import _cfg
from test_proxy import FakeJev, ask, client, serve

from jevstiller import Admission, Config, HashEncoder, Jevstiller, Task, TaskManager
from jevstiller._calibrate import RoutingPolicy
from jevstiller._core import _disagrees, reported_confidence
from jevstiller._settings import load
from jevstiller._training import fit_candidate
from jevstiller.server import KeyRegistry, ProxySettings, create_app
from jevstiller.teachers import peakedness

FLOOR = 0.8          # the synthetic teacher is less confident than this on ~7% of requests: more than the budget
TOKEN = "admin-token-0123456789"


def _feed(js, world, n, batch=100):
    for _ in range(0, n, batch):
        js.classify_batch([t for t, _ in world.sample(batch)])


def test_the_floor_is_a_confidence_or_none():
    assert Task("t", "q", ["a", "b"], confidence_floor=0.8).confidence_floor == 0.8
    assert Task("t", "q", ["a", "b"], confidence_floor=1).confidence_floor == 1.0
    assert Task("t", "q", ["a", "b"], confidence_floor=0).confidence_floor is None          # 0: no floor
    for bad in (-0.1, 1.5, float("nan"), True, "0.8"):
        with pytest.raises(ValueError, match="confidence_floor"):
            Task("t", "q", ["a", "b"], confidence_floor=bad)
    # it doesn't change what the teacher is asked
    assert Task("t", "q", ["a", "b"], confidence_floor=0.8).version == Task("t", "q", ["a", "b"]).version


def test_disagreeing_and_the_reported_confidence():
    pol = RoutingPolicy(0.5, 1.0, 1.0, 0.0, 0.0, 0.02, 0.05, 100, confidence_floor=0.6)
    s = np.array(["a", "a", "b"], dtype=object)
    t = np.array(["a", "a", "a"], dtype=object)
    c = np.array([0.9, 0.5, 0.9])
    assert _disagrees(s, t, c, pol).tolist() == [False, True, True]        # a less confident teacher disagrees
    assert _disagrees(s, t, c, replace(pol, confidence_floor=None)).tolist() == [False, False, True]
    assert reported_confidence({"a": 0.7, "b": 0.3}, pol) == 0.6           # Jev's definition says 0.4
    assert reported_confidence({"a": 0.9, "b": 0.1}, pol) == pytest.approx(0.8)
    assert reported_confidence({"a": 0.7, "b": 0.3}, replace(pol, confidence_floor=None)) == pytest.approx(0.4)


def test_calibration_counts_a_less_confident_teacher_as_disagreeing():
    # one signal; the teacher is unsure near the boundary, as a teacher is on ambiguous input
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 4000)
    X = np.column_stack([x, np.ones_like(x), rng.normal(0, 0.01, (len(x), 6))]).astype(np.float32)
    p = 1 / (1 + np.exp(-8 * x))
    Y = np.column_stack([1 - p, p]).astype(np.float32)
    conf = np.abs(2 * p - 1)                                              # Jev's definition, two classes
    tr, ca = slice(0, 3000), slice(3000, None)
    kw = dict(budget=0.05, delta=0.05, ood_quantile=0.99, ood_k=10, epochs=2000, l2=1e-6, patience=4, seed=0)
    plain = fit_candidate(X[tr], Y[tr], None, X[ca], Y[ca].argmax(1), **kw)
    floored = fit_candidate(X[tr], Y[tr], None, X[ca], Y[ca].argmax(1), yc_conf=conf[ca], confidence_floor=0.6, **kw)
    assert plain.policy.confidence_floor is None and floored.policy.confidence_floor == 0.6
    assert floored.policy.usable and floored.policy.expected_coverage < plain.policy.expected_coverage
    # the calibration rows it answers where the teacher was unsure are counted, and bounded
    Pc = floored.student.predict_proba(X[ca])
    acc = floored.policy.accepts(Pc.max(1), floored.ood.score(X[ca]))
    wrong = acc & ((Pc.argmax(1) != Y[ca].argmax(1)) | (conf[ca] < 0.6))
    assert floored.calib_disagree == int(wrong.sum()) and wrong.mean() <= 0.05
    with pytest.raises(ValueError, match="yc_conf"):
        fit_candidate(X[tr], Y[tr], None, X[ca], Y[ca].argmax(1), confidence_floor=0.6, **kw)


def test_the_loop_keeps_the_teacher_s_doubt(tmp_path, task, world, teacher):
    # at a 2% budget, answering the ~7% of requests the teacher is unsure about matters: without the floor the
    # caller's outcome (label, or unsure) changed on 2.6% of this tail, with it on 1.1%
    task = replace(task, target_agreement=0.98)
    stream = [t for t, _ in world.sample(6000)]
    tail = [t for t, _ in world.sample(2000)]
    answers = {t: teacher.answer(t, task) for t in tail}
    local_unsure = {}
    for floor in (None, FLOOR):
        js = Jevstiller(replace(task, confidence_floor=floor), teacher, tmp_path / str(floor),
                        encoder=HashEncoder(dim=512), config=_cfg())
        for i in range(0, len(stream), 100):
            js.classify_batch(stream[i:i + 100])
        st = js.status()
        assert st.production is not None and st.policy["confidence_floor"] == floor, st.report()
        res = js.classify_batch(tail)
        local = [(t, r) for t, r in zip(tail, res, strict=True) if r.source != "teacher"]
        assert len(local) > 0.3 * len(tail), st.report()
        if floor is None:        # Jev's definition of confidence, not the student's top probability
            assert all(r.confidence == pytest.approx(peakedness(r.probs)) for _, r in local)
        else:
            assert all(r.confidence >= FLOOR for _, r in local)
            wrong = sum(r.label != answers[t].label or answers[t].confidence < FLOOR for t, r in local)
            assert wrong / len(tail) <= task.budget + 0.01, st.report()
            # the offline check counts the same way when given the teacher's confidences (it scores every request
            # the policy accepts, audited ones included, so it isn't the served count exactly)
            ev = js.evaluate(tail, [answers[t].label for t in tail],
                             teacher_confidences=[answers[t].confidence for t in tail])
            labels_only = js.evaluate(tail, [answers[t].label for t in tail])
            assert labels_only["system_disagreement"] <= ev["system_disagreement"] <= task.budget + 0.01
            flagged = sum(answers[t].confidence < FLOOR for t in np.array(tail, dtype=object)[ev["accepted"]])
            assert ev["system_disagreement"] * len(tail) >= flagged
            report = js.status().report()
            assert f"confidence floor {FLOOR:.2f}" in report
            assert js.status().audit_n == 0 or f"less confident than {FLOOR:.2f}" in report
        local_unsure[floor] = sum(answers[t].confidence < FLOOR for t, _ in local) / len(tail)
        js.close()
    assert local_unsure[FLOOR] < local_unsure[None]


def test_a_new_floor_retrains_and_takes_over(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=512), config=_cfg())
    _feed(js, world, 5000)
    first = js.status().production
    assert first is not None
    js.set_confidence_floor(FLOOR)
    assert js.store.last_event(("confidence_floor",))["current"] == FLOOR
    assert js.status().production == first                       # production keeps serving as calibrated
    assert f"calibrated for confidence floor none, not the task's {FLOOR:.2f}" in js.status().report()
    _feed(js, world, 4000)
    st = js.status()
    assert st.production != first and st.policy["confidence_floor"] == FLOOR, st.report()
    res = [r for r in js.classify_batch([t for t, _ in world.sample(500)]) if r.source != "teacher"]
    assert res and all(r.confidence >= FLOOR for r in res)
    js.close()
    # a floor that changed while the task was unloaded (config file, admin API) retrains on load
    js = Jevstiller(replace(task, confidence_floor=0.7), teacher, tmp_path, encoder=HashEncoder(dim=512),
                    config=_cfg(training="manual"))
    assert js._retrain_requested and not js._calibrated(js._prod)
    js.close()


def test_a_candidate_calibrated_for_an_old_floor_is_not_promoted(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=512), config=_cfg(training="manual"))
    _feed(js, world, 4000)
    report = js.train_now()
    assert report.accepted and js.status().shadow is not None
    first = js.status().shadow
    js.set_confidence_floor(FLOOR)                               # while the candidate is in shadow
    _feed(js, world, 300)
    js.maintain()                                                # rejects it, and trains one for the new floor
    rejected = [e for e in js.store.events() if e["kind"] == "rejected" and e.get("version") == first]
    assert rejected and rejected[-1]["reason"] == "calibrated for another confidence floor"
    assert js._shadow is not None and js._shadow.name != first and js._shadow.policy.confidence_floor == FLOOR
    js.close()


def test_the_manager_keeps_a_floor_per_task(tmp_path, world, teacher):
    q, labels = "Which team handles this?", ["billing", "technical", "cancellation", "sales", "other"]
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=64), Config(training="manual"), confidence_floor=0.6,
                    admission=Admission(min_requests=1), janitor_interval_s=3600)
    m.classify("acme", q, labels, [t for t, _ in world.sample(3)])
    [info] = m.tasks()
    assert info.confidence_floor == 0.6 and m.engine(info.key).task.confidence_floor == 0.6
    m.set_confidence_floor(info.key, FLOOR)
    saved = json.loads((tmp_path / "tasks" / info.key / "task.json").read_text())
    assert saved["confidence_floor"] == FLOOR and m.engine(info.key).task.confidence_floor == FLOOR
    with pytest.raises(ValueError):
        m.set_confidence_floor(info.key, 2)
    m.close()
    # the default is for new tasks; an override at startup changes one task (0: none)
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=64), Config(training="manual"), confidence_floor=0.5,
                    task_overrides={info.key: {"confidence_floor": 0}}, janitor_interval_s=3600)
    assert m.tasks()[0].confidence_floor is None
    m.close()


def test_settings(tmp_path):
    p = tmp_path / "j.toml"
    p.write_text("[manager]\nconfidence_floor = 0.6\n[tasks.'abc']\nconfidence_floor = 0\n")
    s = load(p, environ={})
    assert s.confidence_floor == 0.6 and s.tasks["abc"]["confidence_floor"] == 0
    assert load(p, environ={"JEVSTILLER_CONFIDENCE_FLOOR": "0.75"}).confidence_floor == 0.75
    assert load(p, environ={"JEVSTILLER_CONFIDENCE_FLOOR": "none"}).confidence_floor is None
    assert load(environ={}).confidence_floor is None                                   # default: none
    for bad in ("[manager]\nconfidence_floor = 1.5\n", "[tasks.'abc']\nconfidence_floor = 'high'\n",
                "[manager]\nconfidence_floor = true\n"):
        p.write_text(bad)
        with pytest.raises(ValueError, match="confidence_floor"):
            load(p, environ={})


@pytest.fixture
def stack(tmp_path, world):
    jev = FakeJev(world)
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=150, seed=0, training="inline")
    with serve(jev.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=512), cfg, target_agreement=0.95, confidence_floor=FLOOR,
                        admission=Admission(min_requests=1), janitor_interval_s=3600)
        settings = ProxySettings(upstream=upstream, tenancy="shared")
        app = create_app(m, settings, KeyRegistry(b"salt", settings.key_ttl_s), admin_token=TOKEN)
        with serve(app) as proxy:
            yield proxy, m
        m.close()


def test_behind_the_proxy(stack, world):
    proxy, m = stack
    c = client(proxy)
    local = []
    for t, _ in world.sample(4000):
        r = ask(c, t)
        if r.raw_http_response.headers["x-jevstiller-source"] == "local":
            local.append(r.choices["label"])
        if len(local) >= 50:
            break
    [info] = m.tasks()
    assert len(local) >= 50, m.engine(info.key).status().report()
    assert all(FLOOR <= a.confidence <= 1 for a in local)                # the caller's check keeps them

    def admin(path, **kw):
        return httpx.post(f"{proxy}/jevstiller/v1{path}", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60,
                          **kw)
    listed = httpx.get(f"{proxy}/jevstiller/v1/tasks", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert listed[0]["confidence_floor"] == FLOOR
    assert admin(f"/tasks/{info.key}/floor", json={"confidence_floor": 0.7}).json()["confidence_floor"] == 0.7
    assert m.engine(info.key).task.confidence_floor == 0.7
    assert admin(f"/tasks/{info.key}/floor", json={"confidence_floor": None}).status_code == 200
    assert m.tasks()[0].confidence_floor is None
    assert admin(f"/tasks/{info.key}/floor", json={}).status_code == 422               # null is a value; no value isn't
    assert admin(f"/tasks/{info.key}/floor", json={"confidence_floor": 2}).status_code == 422
    assert admin("/tasks/nope/floor", json={"confidence_floor": 0.5}).status_code == 404
