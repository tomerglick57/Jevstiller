"""Live Jev (P0.1), and real Jev responses replayed through the proxy (P6.2).

Live tests call the real API and cost a fraction of a cent:

    JEVSTILLER_LIVE=1 pytest -m live          # TYPESAFE_API_KEY from the environment or .env

The fixture tests replay responses recorded from live Jev (`experiments/jev_profile.py`) and always run.
"""
import json
import os
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from jevstiller import Admission, Config, HashEncoder, Task, TaskManager
from jevstiller.server import KeyRegistry, ProxySettings, create_app

ts = pytest.importorskip("typesafe_sdk")
from test_proxy import serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "jev"


def _key() -> str | None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
    return None


live = pytest.mark.skipif(os.environ.get("JEVSTILLER_LIVE") != "1" or not _key(),
                          reason="set JEVSTILLER_LIVE=1 and TYPESAFE_API_KEY to call the real Jev API")

CLASSES = {"billing": "Charges, invoices, refunds, payment methods",
           "technical": "Bugs, errors, integrations, things not working",
           "cancellation": "Wants to cancel, downgrade, or close the account",
           "sales": "Pre-sales questions, plan comparison, quotes",
           "other": "Anything that does not fit the categories above"}


@pytest.mark.live
@live
def test_jev_teacher_live():
    from jevstiller.teachers.jev import JevTeacher
    task = Task("live", "Which team should handle this customer message?", CLASSES)
    outs = JevTeacher(model="jev-latest", api_key=_key()).classify(
        ["I was charged twice", "The app crashes on upload", {"subject": "Bye", "body": "Close my account"}], task)
    assert all(not isinstance(o, Exception) for o in outs), outs
    assert [o.label for o in outs] == ["billing", "technical", "cancellation"]
    for o in outs:
        assert set(o.probs) == set(CLASSES) and abs(sum(o.probs.values()) - 1) < 1e-6
        assert o.model.startswith("jev:jev-") and o.model != "jev:jev-latest"     # the resolved version
        assert o.input_tokens > 0 and o.cost_usd > 0 and o.request_id.startswith("req_")


@pytest.mark.live
@live
def test_proxy_in_front_of_live_jev(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                    admission=Admission(min_requests=1), janitor_interval_s=3600)
    app = create_app(m, ProxySettings(upstream="https://api.typesafe.ai"), KeyRegistry(b"s", 3600))
    with serve(app) as proxy:
        c = ts.TypeSafeClient(base_url=proxy, api_key=_key(), retry=ts.RetryPolicy(max_retries=1))
        r = c.system_one(state="I was charged twice", questions={
            "label": ts.Choice(instructions="Which team should handle this customer message?", criteria=CLASSES)})
    assert r.raw_http_response.headers["x-jevstiller-source"] == "upstream"
    assert r.choices["label"].choice == "billing" and r.request_id.startswith("req_")
    engine = m.engine(m.tasks()[0].key)
    assert engine.status().teacher_model == f"jev:{r.model}"
    assert engine.store.counts(engine.task.version)["teacher_calls"] == 1
    m.close()


# ---- recorded live responses, replayed (always runs) -------------------------------------------------------
def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _replayer(names: dict[str, str]):
    """A stub upstream that answers each (method, path) with a recorded response, verbatim."""
    async def handle(request: Request):
        fx = _fixture(names[f"{request.method} {request.url.path}"])
        return JSONResponse(fx["body"], status_code=fx["status"], headers=fx["headers"])
    return Starlette(routes=[Route("/{p:path}", handle, methods=["GET", "POST"])])


@pytest.mark.parametrize("name", ["systemone_choice_200", "systemone_mixed_200"])
def test_recorded_responses_parse_the_same_through_the_proxy(tmp_path, name):
    fx = _fixture(name)
    req = fx["request"]["json"]
    stub = _replayer({"POST /v1/systemone": name})
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                    admission=Admission(min_requests=1), janitor_interval_s=3600)
    with serve(stub) as upstream, serve(create_app(m, ProxySettings(upstream=upstream),
                                                   KeyRegistry(b"s", 3600))) as proxy:
        results = []
        for base in (upstream, proxy):
            c = ts.TypeSafeClient(base_url=base, api_key="tsk_test", retry=ts.RetryPolicy(max_retries=0))
            results.append(c.system_one(state=req["state"], questions=req["questions"], model=req["model"]))
    direct, proxied = results
    assert proxied.model_dump() == direct.model_dump()
    assert proxied.request_id == direct.request_id == fx["headers"]["x-typesafe-request-id"]
    engine = m.engine(m.tasks()[0].key)                   # the choice answer was recorded for training
    assert engine.store.counts(engine.task.version)["teacher_calls"] == 1
    m.close()


@pytest.mark.parametrize("name,error", [("systemone_401", "TypeSafeAuthenticationError"),
                                        ("systemone_422", "TypeSafeUnprocessableEntityError")])
def test_recorded_errors_raise_the_same_through_the_proxy(tmp_path, name, error):
    fx = _fixture(name)
    req = fx["request"]["json"]
    stub = _replayer({"POST /v1/systemone": name})
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                    admission=Admission(min_requests=1), janitor_interval_s=3600)
    with serve(stub) as upstream, serve(create_app(m, ProxySettings(upstream=upstream),
                                                   KeyRegistry(b"s", 3600))) as proxy:
        for base in (upstream, proxy):
            c = ts.TypeSafeClient(base_url=base, api_key="tsk_test", retry=ts.RetryPolicy(max_retries=0))
            with pytest.raises(getattr(ts, error)) as ei:
                c.system_one(state=req["state"], questions=req["questions"] or {"x": ts.Noul()}, model=req["model"])
            assert ei.value.status == fx["status"]
    m.close()


def test_recorded_models_list_passes_through(tmp_path):
    stub = _replayer({"GET /v1/models": "models_200"})
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), janitor_interval_s=3600)
    with serve(stub) as upstream, serve(create_app(m, ProxySettings(upstream=upstream))) as proxy:
        names = [x.name for x in ts.TypeSafeClient(base_url=proxy, api_key="tsk_test").models.list().models]
    assert names == [x["name"] for x in _fixture("models_200")["body"]["models"]]
    m.close()
