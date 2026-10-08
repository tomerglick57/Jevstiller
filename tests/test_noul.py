"""Yes/no (`noul`) questions: the task kind, the cut-off contract in the engine, the manager and the proxy."""
import json

import numpy as np
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from test_proxy import GOOD, client, serve

from jevstiller import Admission, Config, HashEncoder, Jevstiller, SyntheticTeacher, SyntheticWorld, Task, TaskManager
from jevstiller._manager import task_key
from jevstiller._task import check_cutoffs, noul_labels, noul_scores
from jevstiller.server import KeyRegistry, ProxySettings, create_app

ts = pytest.importorskip("typesafe_sdk")

LABELS = ["billing", "technical", "cancellation", "sales", "other"]
QUESTION = "Is this about billing or sales?"


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0, training="inline")
    base.update(kw)
    return Config(**base)


@pytest.fixture
def world():
    return SyntheticWorld(LABELS, seed=1)


@pytest.fixture
def teacher(world):
    return SyntheticTeacher(world, noul_positive=["billing", "sales"])


def _until_production(js, world, batches=40):
    for _ in range(batches):
        js.classify_batch([t for t, _ in world.sample(200)])
        st = js.status()
        if st.production and st.shadow is None and st.mode == "cascade":
            return st
    raise AssertionError(js.status().report())


# ---- the task and its cut-offs -----------------------------------------------------------------------------
def test_a_noul_task_has_two_classes_a_default_cutoff_and_its_own_identity():
    t = Task.noul("q", QUESTION)
    assert t.kind == "noul" and t.labels == ["false", "true"] and t.cutoffs == (0.5,)
    assert t.classes == {"false": None, "true": None} and t.confidence_floor is None
    described = Task.noul("q", QUESTION, {"true": "invoices, quotes", "false": "anything else"}, cutoffs=[0.4, 0.6])
    assert described.cutoffs == (0.4, 0.6) and described.version != t.version      # criteria are part of the question
    assert Task.noul("q", QUESTION, cutoffs=0.8).version == t.version              # the cut-off is not
    # a choice question with the same words is a different question, and a different task
    choice = Task("q", QUESTION, {"false": None, "true": None})
    assert choice.fingerprint != t.fingerprint
    assert task_key("acme", choice) != task_key("acme", t, "noul")


@pytest.mark.parametrize("bad", [[], [0.5, 0.5], [0.6, 0.4], [0.0], [1.0], [0.2, 0.5, 0.8], "0.5", [True], None])
def test_invalid_cutoffs_are_rejected(bad):
    with pytest.raises(ValueError):
        check_cutoffs(bad)


def test_noul_tasks_reject_what_belongs_to_choice_tasks():
    with pytest.raises(ValueError):
        Task("q", QUESTION, {"yes": "", "no": ""}, kind="noul")       # criteria are true / false
    with pytest.raises(ValueError):
        Task("q", QUESTION, None, confidence_floor=0.6, kind="noul")  # the unsure band is two cutoffs
    with pytest.raises(ValueError):
        Task("q", QUESTION, ["a", "b"], cutoffs=0.5)                   # cutoffs are for noul tasks
    with pytest.raises(ValueError):
        Task("q", QUESTION)                                            # a choice task needs classes


def test_outcomes_and_routing_scores_follow_the_cutoffs():
    p = [0.0, 0.49, 0.5, 0.51, 1.0]
    assert list(noul_labels(p, (0.5,))) == ["false", "false", "true", "true", "true"]      # at the cut-off: yes
    assert list(noul_labels([0.39, 0.4, 0.59, 0.6], (0.4, 0.6))) == ["false", "unsure", "unsure", "true"]
    s = noul_scores([0.5, 0.45, 0.0, 1.0], (0.5,))
    assert np.allclose(s, [0.5, 0.55, 1.0, 1.0])
    assert np.allclose(noul_scores([0.5, 0.4, 0.95], (0.4, 0.6)), [0.6, 0.5, 0.85])       # nearest cut-off


# ---- the engine ---------------------------------------------------------------------------------------------
def test_a_noul_task_trains_and_answers_on_the_teachers_side_of_the_cutoff(tmp_path, world, teacher):
    task = Task.noul("billing_or_sales", QUESTION, target_agreement=0.95)
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=512), config=_cfg())
    st = _until_production(js, world)
    assert st.kind == "noul" and st.cutoffs == (0.5,) and st.policy["cutoffs"] == [0.5]
    assert "yes/no, cut-off 0.50" in st.report()
    tail = [t for t, _ in world.sample(3000)]
    res = js.classify_batch(tail)
    served = [(t, r) for t, r in zip(tail, res, strict=True) if r.source != "teacher"]
    assert len(served) > 300, js.status().report()
    for _, r in served[:50]:
        assert r.label in ("false", "true") and set(r.probs) == {"false", "true"}
        assert r.label == ("true" if r.probs["true"] >= 0.5 else "false")
    wrong = sum(r.label != ("true" if teacher.answer(t, task).probs["true"] >= 0.5 else "false") for t, r in served)
    assert wrong / len(tail) <= task.budget + 0.01, js.status().report()
    # teacher answers are stored under the outcome their probability stands for
    labels = {row[0] for row in js.store.db.execute("SELECT DISTINCT teacher_label FROM samples "
                                                    "WHERE teacher_label IS NOT NULL")}
    assert labels == {"false", "true"}
    js.close()


def test_an_unsure_band_makes_three_outcomes(tmp_path, world, teacher):
    task = Task.noul("billing_or_sales", QUESTION, cutoffs=(0.3, 0.7), target_agreement=0.95)
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=512), config=_cfg())
    st = _until_production(js, world)
    assert "unsure from 0.30 to 0.70" in st.report()
    tail = [t for t, _ in world.sample(3000)]
    res = js.classify_batch(tail)
    served = [(t, r) for t, r in zip(tail, res, strict=True) if r.source != "teacher"]
    assert served, js.status().report()
    truth = {t: str(noul_labels([teacher.answer(t, task).probs["true"]], (0.3, 0.7))[0]) for t, _ in served}
    assert all(r.label == str(noul_labels([r.probs["true"]], (0.3, 0.7))[0]) for _, r in served)
    assert sum(r.label != truth[t] for t, r in served) / len(tail) <= task.budget + 0.01, js.status().report()
    stored = {row[0] for row in js.store.db.execute("SELECT DISTINCT teacher_label FROM samples "
                                                    "WHERE teacher_label IS NOT NULL")}
    assert "unsure" in stored
    js.close()


def test_changing_the_cutoff_stops_local_answers_until_a_student_is_calibrated_for_it(tmp_path, world, teacher):
    task = Task.noul("billing_or_sales", QUESTION, target_agreement=0.95)
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=512), config=_cfg(training="manual"))
    for _ in range(12):
        js.classify_batch([t for t, _ in world.sample(200)])
        js.maintain()
    assert js.status().production, js.status().report()
    assert any(r.source != "teacher" for r in js.classify_batch([t for t, _ in world.sample(300)]))
    js.set_cutoffs(0.8)
    assert js.task.cutoffs == (0.8,)
    after = js.classify_batch([t for t, _ in world.sample(300)])
    assert all(r.source == "teacher" for r in after)                 # checked against 0.5, so it answers nothing
    assert all(r.label == ("true" if r.probs["true"] >= 0.8 else "false") for r in after)
    for _ in range(12):
        js.maintain()
        js.classify_batch([t for t, _ in world.sample(200)])
    st = js.status()
    assert st.policy and st.policy["cutoffs"] == [0.8], st.report()
    assert any(r.source != "teacher" for r in js.classify_batch([t for t, _ in world.sample(300)]))
    with pytest.raises(ValueError):
        Jevstiller(Task("c", "x", ["a", "b"]), teacher, tmp_path / "c", encoder=HashEncoder(dim=64)).set_cutoffs(0.5)
    js.close()


def test_evaluate_takes_the_teachers_probabilities(tmp_path, world, teacher):
    task = Task.noul("billing_or_sales", QUESTION, target_agreement=0.95)
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=512), config=_cfg())
    _until_production(js, world)
    texts = [t for t, _ in world.sample(1000)]
    p = [teacher.answer(t, task).probs["true"] for t in texts]
    e = js.evaluate(texts, p)
    assert e["coverage"] > 0.1 and e["system_agreement"] >= task.target_agreement - 0.01
    named = js.evaluate(texts, ["true" if x >= 0.5 else "false" for x in p])        # or the outcomes by name
    assert named["system_agreement"] == e["system_agreement"] and named["coverage"] == e["coverage"]
    js.close()


# ---- the manager --------------------------------------------------------------------------------------------
def test_the_manager_keeps_noul_tasks_apart_and_persists_their_cutoffs(tmp_path, world, teacher):
    def manager(**kw):
        return TaskManager(tmp_path, teacher, HashEncoder(dim=128), _cfg(), admission=Admission(min_requests=1),
                           janitor_interval_s=3600, **kw)
    m = manager(noul_cutoffs=[0.4, 0.6])
    texts = [t for t, _ in world.sample(20)]
    m.classify("acme", QUESTION, None, texts, kind="noul")
    m.classify("acme", QUESTION, {"false": None, "true": None}, texts)          # the same words as a choice
    infos = {i.kind: i for i in m.tasks()}
    assert set(infos) == {"noul", "choice"} and infos["noul"].key != infos["choice"].key
    assert infos["noul"].cutoffs == [0.4, 0.6] and infos["choice"].cutoffs is None
    key = infos["noul"].key
    m.set_cutoffs(key, 0.8)
    with pytest.raises(ValueError):
        m.set_cutoffs(infos["choice"].key, 0.8)
    m.close()
    m = manager()                                   # a restart keeps the kind, the key and the task's own cut-off
    info = next(i for i in m.tasks() if i.kind == "noul")
    assert info.key == key and info.cutoffs == [0.8] and info.task().cutoffs == (0.8,)
    assert m.resolve("acme", QUESTION, None, kind="noul")[0] == key
    m.close()
    off = manager(noul_cutoffs=None)
    with pytest.raises(ValueError):
        off.resolve("acme", "Another question?", None, kind="noul")
    off.close()


# ---- the proxy ----------------------------------------------------------------------------------------------
class NoulJev:
    """A Jev that answers noul questions with a probability that depends on the text."""

    def __init__(self, world):
        self.teacher = SyntheticTeacher(world, noul_positive=["billing", "sales"])
        self.calls = 0

    async def system_one(self, request: Request):
        self.calls += 1
        body = await request.json()
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                o = self.teacher.classify([body["state"]], Task.noul("t", q.get("instructions"), q.get("criteria")))[0]
                answers[name] = {"type": "noul", "noul": round(o.probs["true"], 2)}
            else:
                o = self.teacher.classify([body["state"]], Task("t", q.get("instructions"), list(q["criteria"])))[0]
                answers[name] = {"type": "choice", "choice": o.label, "confidence": o.confidence,
                                 "probabilities": o.probs}
        return JSONResponse({"model": "jev-1.13.0", "answers": answers,
                             "usage": {"input_tokens": 50, "output_tokens": 1}},
                            headers={"x-typesafe-request-id": f"req_{self.calls}"})

    def app(self):
        return Starlette(routes=[Route("/v1/systemone", self.system_one, methods=["POST"])])


def _stack(tmp_path, world, **manager_kw):
    jev = NoulJev(world)
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=150, seed=0, training="inline")
    m = TaskManager(tmp_path, None, HashEncoder(dim=512), cfg, target_agreement=0.95,
                    admission=Admission(min_requests=1), janitor_interval_s=3600, **manager_kw)
    return jev, m


def test_the_proxy_answers_noul_questions_locally_in_jevs_shape(tmp_path, world):
    jev, m = _stack(tmp_path, world)
    with serve(jev.app()) as upstream:
        settings = ProxySettings(upstream=upstream, tenancy="shared")
        with serve(create_app(m, settings, KeyRegistry(b"salt", settings.key_ttl_s))) as proxy:
            c = client(proxy, GOOD)
            local = []
            for t, _ in world.sample(3000):
                r = c.system_one(state=t, questions={"yes": ts.Noul(instructions=QUESTION)})
                if r.raw_http_response.headers["x-jevstiller-source"] == "local":
                    local.append((t, r))
                if len(local) >= 60:
                    break
            assert len(local) >= 60, m.engine(m.tasks()[0].key).status().report()
            t, r = local[-1]
            raw = r.raw_http_response.json()
            assert raw["answers"]["yes"].keys() == {"type", "noul"} and raw["answers"]["yes"]["type"] == "noul"
            assert 0.0 <= r.nouls["yes"].noul <= 1.0 and r.usage.input_tokens == 0
            assert json.loads(r.raw_http_response.headers["x-jevstiller-detail"])["yes"].startswith("student:")
            [info] = m.tasks()
            assert info.kind == "noul" and info.cutoffs == [0.5]
            # each local answer is on Jev's side of the cut-off, within the budget
            task = Task.noul("t", QUESTION)
            wrong = sum((r.nouls["yes"].noul >= 0.5) != (round(jev.teacher.answer(t, task).probs["true"], 2) >= 0.5)
                        for t, r in local)
            assert wrong <= 6, wrong
    m.close()


def test_a_request_mixing_choice_and_noul_is_answered_locally_once_both_are_trained(tmp_path, world):
    jev, m = _stack(tmp_path, world)
    choice = ts.Choice(instructions="Which team handles this?", criteria={c: "" for c in LABELS})
    with serve(jev.app()) as upstream:
        settings = ProxySettings(upstream=upstream, tenancy="shared")
        with serve(create_app(m, settings, KeyRegistry(b"salt", settings.key_ttl_s))) as proxy:
            c = client(proxy, GOOD)
            both = None
            for t, _ in world.sample(4000):
                r = c.system_one(state=t, questions={"team": choice, "yes": ts.Noul(instructions=QUESTION)})
                if r.raw_http_response.headers["x-jevstiller-source"] == "local":
                    both = r
                    break
            assert both is not None, [m.engine(i.key).status().report() for i in m.tasks()]
            assert both.choices["team"].choice in LABELS and 0.0 <= both.nouls["yes"].noul <= 1.0
            assert {i.kind for i in m.tasks()} == {"choice", "noul"}
    m.close()


def test_with_noul_off_the_proxy_forwards_them_as_before(tmp_path, world):
    jev, m = _stack(tmp_path, world, noul_cutoffs=None)
    with serve(jev.app()) as upstream:
        settings = ProxySettings(upstream=upstream, tenancy="shared")
        with serve(create_app(m, settings, KeyRegistry(b"salt", settings.key_ttl_s))) as proxy:
            c = client(proxy, GOOD)
            for t, _ in world.sample(30):
                r = c.system_one(state=t, questions={"yes": ts.Noul(instructions=QUESTION)})
                assert r.raw_http_response.headers["x-jevstiller-source"] == "upstream"
            assert m.tasks() == [] and jev.calls == 30
    m.close()


def test_a_noul_question_with_unknown_fields_is_forwarded(tmp_path, world):
    import httpx
    jev, m = _stack(tmp_path, world)
    with serve(jev.app()) as upstream:
        settings = ProxySettings(upstream=upstream, tenancy="shared")
        with serve(create_app(m, settings, KeyRegistry(b"salt", settings.key_ttl_s))) as proxy:
            body = {"state": "w1 w2", "model": "jev-latest",
                    "questions": {"yes": {"type": "noul", "instructions": QUESTION, "threshold": 0.8}}}
            for _ in range(3):
                r = httpx.post(f"{proxy}/v1/systemone", json=body, headers={"Authorization": f"Bearer {GOOD}"})
                assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream"
            assert m.tasks() == []
    m.close()


# ---- settings and the admin API -----------------------------------------------------------------------------
def test_settings(tmp_path):
    from jevstiller._settings import load
    assert load(environ={}).noul_cutoffs == [0.5]                                      # on by default, at 0.5
    p = tmp_path / "j.toml"
    p.write_text("[manager]\nnoul_cutoffs = [0.4, 0.6]\n[tasks.'abc']\ncutoffs = [0.8]\n")
    s = load(p, environ={})
    assert s.noul_cutoffs == [0.4, 0.6] and s.tasks["abc"]["cutoffs"] == [0.8]
    assert load(p, environ={"JEVSTILLER_NOUL_CUTOFFS": "0.8"}).noul_cutoffs == [0.8]
    assert load(p, environ={"JEVSTILLER_NOUL_CUTOFFS": "none"}).noul_cutoffs is None   # forward them all
    p.write_text("[manager]\nnoul_cutoffs = []\n")
    assert load(p, environ={}).noul_cutoffs is None
    for bad in ("[manager]\nnoul_cutoffs = [0.6, 0.4]\n", "[manager]\nnoul_cutoffs = [1.5]\n",
                "[manager]\nnoul_cutoffs = ['half']\n", "[tasks.'abc']\ncutoffs = [0.1, 0.2, 0.3]\n"):
        p.write_text(bad)
        with pytest.raises(ValueError, match="cutoffs"):
            load(p, environ={})


def test_the_admin_api_lists_and_changes_cutoffs(tmp_path, world):
    import httpx
    jev, m = _stack(tmp_path, world)
    token = "admin-token-for-the-noul-test"
    with serve(jev.app()) as upstream:
        settings = ProxySettings(upstream=upstream, tenancy="shared")
        with serve(create_app(m, settings, KeyRegistry(b"salt", settings.key_ttl_s), admin_token=token)) as proxy:
            c = client(proxy, GOOD)
            for t, _ in world.sample(5):
                c.system_one(state=t, questions={"yes": ts.Noul(instructions=QUESTION),
                                                 "team": ts.Choice(instructions="Team?", criteria={"a": "", "b": ""})})
            h = {"Authorization": f"Bearer {token}"}
            listed = {i["kind"]: i for i in httpx.get(f"{proxy}/jevstiller/v1/tasks", headers=h).json()}
            assert listed["noul"]["cutoffs"] == [0.5] and listed["choice"]["cutoffs"] is None
            key = listed["noul"]["key"]

            def post(k, body):
                return httpx.post(f"{proxy}/jevstiller/v1/tasks/{k}/cutoffs", headers=h, json=body, timeout=60)
            assert post(key, {"cutoffs": [0.4, 0.6]}).json() == {"key": key, "cutoffs": [0.4, 0.6]}
            assert m.engine(key).task.cutoffs == (0.4, 0.6)
            report = httpx.get(f"{proxy}/jevstiller/v1/tasks/{key}", headers=h).json()["report"]
            assert "yes/no, unsure from 0.40 to 0.60" in report
            assert post(key, {"cutoffs": [0.9, 0.1]}).status_code == 422
            assert post(key, {}).status_code == 422
            assert post(listed["choice"]["key"], {"cutoffs": [0.5]}).status_code == 422     # not a yes/no task
            page = httpx.get(f"{proxy}/jevstiller/status", auth=("admin", token)).text
            assert "yes/no" in page and "unsure 0.40" in page
    m.close()
