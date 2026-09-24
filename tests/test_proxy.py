"""P3 + P4.1 + P6.1: the unmodified typesafe-sdk, pointed at the proxy over real HTTP, against a fake Jev."""
import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from jevstiller import Admission, Config, HashEncoder, SyntheticTeacher, SyntheticWorld, Task, TaskManager
from jevstiller.server import KeyRegistry, ProxySettings, create_app

ts = pytest.importorskip("typesafe_sdk")

LABELS = ["billing", "technical", "cancellation", "sales", "other"]
QUESTION = {"type": "choice", "instructions": "Which team handles this?", "criteria": {c: "" for c in LABELS}}
GOOD, OTHER = "tsk_good", "tsk_other"


class FakeJev:
    """Jev's wire format: bearer auth, choice + noul answers, request ids, 429 on demand, a model list."""

    def __init__(self, world):
        self.teacher = SyntheticTeacher(world)
        self.calls = 0
        self.rate_limit_next = 0
        self.valid = {GOOD, OTHER}

    async def system_one(self, request: Request):
        self.calls += 1
        self.last_headers = dict(request.headers)
        auth = request.headers.get("authorization", "")
        if auth.removeprefix("Bearer ") not in self.valid:
            return JSONResponse({"detail": "invalid API key"}, 401, headers={"x-typesafe-request-id": "req_401"})
        if self.rate_limit_next:
            self.rate_limit_next -= 1
            return JSONResponse({"detail": "rate limited"}, 429, headers={"retry-after": "30"})
        body = await request.json()
        if "questions" not in body or not body["questions"]:
            return JSONResponse({"detail": [{"loc": ["body", "questions"], "msg": "Field required",
                                             "type": "missing"}]}, 422)
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "choice":
                task = Task("t", q.get("instructions"), list(q["criteria"]))
                o = self.teacher.classify([body["state"]], task)[0]
                answers[name] = {"type": "choice", "choice": o.label, "confidence": o.confidence,
                                 "probabilities": o.probs}
            else:
                answers[name] = {"type": "noul", "noul": 0.9}
        return JSONResponse({"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 50,
                                                                                  "output_tokens": 1}},
                            headers={"x-typesafe-request-id": f"req_{self.calls}"})

    async def models(self, request: Request):
        return JSONResponse({"models": [{"name": "jev-latest", "description": "alias", "release_date": "2026-09-01"}]},
                            headers={"x-typesafe-request-id": "req_models"})

    def app(self):
        return Starlette(routes=[Route("/v1/systemone", self.system_one, methods=["POST"]),
                                 Route("/v1/models", self.models, methods=["GET"])])


@contextlib.contextmanager
def serve(app):
    sock = socket.socket()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # as uvicorn does for sockets it creates
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    t = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(5)


@pytest.fixture
def world():
    return SyntheticWorld(LABELS, seed=1)


@pytest.fixture
def stack(tmp_path, world):
    """(proxy url, fake jev, manager). Low thresholds and inline training so a task trains within a test."""
    jev = FakeJev(world)
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=150, seed=0, training="inline")
    with serve(jev.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=512), cfg, target_agreement=0.95,
                        admission=Admission(min_requests=1), janitor_interval_s=3600)
        settings = ProxySettings(upstream=upstream, tenancy="shared")
        app = create_app(m, settings, KeyRegistry(b"salt", settings.key_ttl_s))
        with serve(app) as proxy:
            yield proxy, jev, m, app
        m.close()


def client(url, key=GOOD):
    return ts.TypeSafeClient(base_url=url, api_key=key, retry=ts.RetryPolicy(max_retries=0))


def ask(c, text, **extra):
    return c.system_one(state=text, questions={"label": ts.Choice(instructions=QUESTION["instructions"],
                                                                   criteria=QUESTION["criteria"]), **extra})


def test_first_request_is_forwarded_and_parsed(stack, world):
    proxy, jev, m, _ = stack
    r = ask(client(proxy), "w1 w2 w3")
    assert r.raw_http_response.headers["x-jevstiller-source"] == "upstream"
    assert r.request_id == "req_1" and r.model == "jev-1.13.0" and r.usage.input_tokens == 50
    assert r.choices["label"].choice in LABELS and jev.calls == 1
    [info] = m.tasks()
    assert info.model == "jev-latest" and info.tenant == "default"


def test_task_trains_behind_the_proxy_and_answers_locally(stack, world):
    proxy, jev, m, app = stack
    c = client(proxy)
    local = []
    for t, _ in world.sample(3000):
        r = ask(c, t)
        if r.raw_http_response.headers["x-jevstiller-source"] == "local":
            local.append((t, r))
        if len(local) >= 50:
            break
    assert len(local) >= 50, m.engine(m.tasks()[0].key).status().report()
    t, r = local[-1]
    assert r.request_id.startswith("jvs_") and r.model == "jev-1.13.0"
    assert r.usage.input_tokens == 0 and r.usage.output_tokens == 0
    ans = r.choices["label"]
    assert set(ans.probabilities) == set(LABELS) and ans.choice == max(ans.probabilities, key=ans.probabilities.get)
    assert 0 <= ans.confidence <= 1
    detail = json.loads(r.raw_http_response.headers["x-jevstiller-detail"])
    assert detail["label"].startswith("student:")
    engine = m.engine(m.tasks()[0].key)
    assert engine.status().teacher_model == "jev:jev-1.13.0"
    assert app.state.proxy.stats["local"] == len(local)


def test_async_client(stack):
    proxy, *_ = stack

    async def go():
        async with ts.AsyncTypeSafeClient(base_url=proxy, api_key=GOOD, retry=ts.RetryPolicy(max_retries=0)) as c:
            q = {"label": ts.Choice(instructions=QUESTION["instructions"], criteria=QUESTION["criteria"])}
            return await c.system_one(state={"subject": "w1", "body": "w2 w3"}, questions=q)
    r = asyncio.run(go())
    assert r.choices["label"].choice in LABELS


def test_bad_key_is_relayed_and_never_answered_locally(stack, world):
    proxy, jev, m, app = stack
    good = client(proxy)
    for t, _ in world.sample(1500):                       # train while the good key is verified
        ask(good, t)
    bad = client(proxy, "tsk_revoked")
    with pytest.raises(ts.TypeSafeAuthenticationError):
        ask(bad, "w1 w2")
    calls = jev.calls
    for t, _ in world.sample(20):                         # never answered locally for an unverified key
        with pytest.raises(ts.TypeSafeAuthenticationError):
            ask(bad, t)
    assert jev.calls == calls + 20
    jev.valid.discard(GOOD)                               # the key is revoked at Jev ...
    with contextlib.suppress(ts.TypeSafeAuthenticationError):
        for t, _ in world.sample(200):                    # ... the first forwarded request sees the 401 ...
            ask(good, t)
    for t, _ in world.sample(20):                         # ... and nothing is answered locally after that
        with pytest.raises(ts.TypeSafeAuthenticationError):
            ask(good, t)


def test_mixed_questions_are_forwarded_and_the_choice_is_recorded(stack):
    proxy, jev, m, _ = stack
    r = ask(client(proxy), "w1 w2", spam=ts.Noul(instructions="Is this spam?"))
    assert r.nouls["spam"].noul == 0.9 and r.choices["label"].choice in LABELS
    engine = m.engine(m.tasks()[0].key)
    assert engine.store.counts(engine.task.version)["total"] == 1


def test_rate_limit_is_relayed_and_remembered(stack):
    proxy, jev, *_ = stack
    c = client(proxy)
    jev.rate_limit_next = 1
    with pytest.raises(ts.TypeSafeRateLimitError):
        ask(c, "w1")
    calls = jev.calls
    with pytest.raises(ts.TypeSafeRateLimitError) as ei:   # the proxy backs off without calling Jev
        ask(c, "w2")
    assert jev.calls == calls and int(ei.value.headers["retry-after"]) > 0
    ask(client(proxy, OTHER), "w3")                        # other keys are unaffected


def test_passthrough_paths_and_invalid_bodies(stack):
    proxy, jev, *_ = stack
    models = client(proxy).models.list()
    assert models.models[0].name == "jev-latest"
    r = httpx.post(f"{proxy}/v1/systemone", headers={"Authorization": f"Bearer {GOOD}"},
                   json={"state": "x", "model": "jev-latest", "questions": {}})
    assert r.status_code == 422                            # Jev judged it, not the proxy
    r = httpx.post(f"{proxy}/v1/systemone", headers={"Authorization": f"Bearer {GOOD}"}, content=b"{not json")
    assert jev.calls >= 2 and r.status_code in (400, 422, 500)


def test_upstream_down(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                    admission=Admission(min_requests=1), janitor_interval_s=3600)
    app = create_app(m, ProxySettings(upstream="http://127.0.0.1:9", upstream_timeout_s=2))
    with serve(app) as proxy:
        with pytest.raises(ts.TypeSafeAPIError) as ei:
            ask(client(proxy), "w1")
    assert ei.value.status == 502
    m.close()


def test_per_key_tenancy_keeps_tasks_apart(tmp_path, world):
    jev = FakeJev(world)
    with serve(jev.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                        admission=Admission(min_requests=1), janitor_interval_s=3600)
        app = create_app(m, ProxySettings(upstream=upstream, tenancy="per_key"), KeyRegistry(b"s", 3600))
        with serve(app) as proxy:
            ask(client(proxy, GOOD), "w1")
            ask(client(proxy, OTHER), "w1")
        tenants = {i.tenant for i in m.tasks()}
        assert len(tenants) == 2 and all(t.startswith("key:") and GOOD not in t for t in tenants)
        m.close()
