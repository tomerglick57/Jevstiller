"""P5.3 / P5.4 / P5.6: readiness, Prometheus metrics, admin API (auth, persistence, never forwarded)."""
import json

import httpx
import pytest
from test_audit_fixes import Upstream, _body, _post
from test_proxy import serve

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller.server import ProxySettings, create_app

TOKEN = "admin-token-0123456789"


@pytest.fixture
def stack(tmp_path):
    up = Upstream()
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=150, training="inline")
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), cfg, admission=Admission(1), target_agreement=0.95,
                    janitor_interval_s=3600)
    with serve(up.app()) as u:
        app = create_app(m, ProxySettings(upstream=u), admin_token=TOKEN, ready=lambda: {"extra": True})
        with serve(app) as proxy:
            yield proxy, up, m, tmp_path
    m.close()


def _admin(proxy, method, path, token=TOKEN, **kw):
    return httpx.request(method, f"{proxy}/jevstiller/v1{path}", headers={"Authorization": f"Bearer {token}"},
                         timeout=60, **kw)


def test_admin_requires_its_token_and_is_never_forwarded(stack):
    proxy, up, m, _ = stack
    assert _admin(proxy, "GET", "/tasks", token="wrong").status_code == 401
    assert httpx.get(f"{proxy}/jevstiller/v1/tasks").status_code == 401
    assert _admin(proxy, "GET", "/nope").status_code == 404
    assert httpx.get(f"{proxy}/jevstiller/whatever").status_code == 404
    assert up.requests == []


def test_admin_api_off_without_a_token(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), janitor_interval_s=3600)
    with serve(create_app(m, ProxySettings(upstream="http://127.0.0.1:9"))) as proxy:
        assert _admin(proxy, "GET", "/tasks").status_code == 404
        assert httpx.get(f"{proxy}/metrics").status_code == 404
    m.close()


def test_task_lifecycle_through_the_admin_api(stack, world):
    proxy, up, m, data = stack
    for t, _ in world.sample(1500):
        _post(proxy, _body(state=t))
    [task] = _admin(proxy, "GET", "/tasks").json()
    key = task["key"]
    assert task["tenant"] == "default" and task["loaded"] and task["classes"] == 5
    st = _admin(proxy, "GET", f"/tasks/{key}").json()
    assert "Task:" in st["report"] and st["status"]["requests"] >= 1500 and st["status"]["production"]
    assert _admin(proxy, "GET", f"/tasks/{key}/versions").json()[0]["name"] == "student:v1"

    assert _admin(proxy, "POST", f"/tasks/{key}/target", json={"target_agreement": 0.99}).status_code == 200
    assert _admin(proxy, "POST", f"/tasks/{key}/target", json={"target_agreement": 2}).status_code == 422
    assert _admin(proxy, "POST", f"/tasks/{key}/mode", json={"mode": "teacher_only"}).status_code == 200
    assert _admin(proxy, "POST", f"/tasks/{key}/mode", json={"mode": "hedge"}).status_code == 422
    saved = json.loads((data / "tasks" / key / "task.json").read_text())
    assert saved["target_agreement"] == 0.99 and saved["mode"] == "teacher_only"          # persisted
    assert m.engine(key).task.target_agreement == 0.99 and m.engine(key).mode == "teacher_only"   # live
    r = _post(proxy, _body(state="w1 w2"))
    assert r.headers["x-jevstiller-source"] == "upstream"                                  # teacher_only now

    stats = _admin(proxy, "GET", "/stats").json()
    assert stats["tasks"] == 1 and stats["proxy"]["forwarded"] >= 1
    assert _admin(proxy, "DELETE", f"/tasks/{key}").json() == {"deleted": key}
    assert _admin(proxy, "GET", "/tasks").json() == [] and not (data / "tasks" / key).exists()
    assert _admin(proxy, "GET", f"/tasks/{key}").status_code == 404
    assert _admin(proxy, "GET", "/tasks/../../etc").status_code == 404


def test_metrics_and_readiness(stack):
    proxy, up, m, _ = stack
    _post(proxy, _body())
    httpx.post(f"{proxy}/v1/systemone", headers={"Authorization": "Bearer a", "authorization": "Bearer b"},
               json=_body())
    assert httpx.get(f"{proxy}/metrics").status_code == 401
    text = httpx.get(f"{proxy}/metrics", headers={"Authorization": f"Bearer {TOKEN}"}).text
    assert 'jevstiller_requests_total{route="systemone",source="upstream"}' in text
    assert "jevstiller_upstream_duration_seconds_count" in text and "jevstiller_tasks 1" in text
    assert "# TYPE jevstiller_request_duration_seconds histogram" in text
    r = httpx.get(f"{proxy}/readyz").json()
    assert r == {"ready": True, "checks": {"manager": True, "extra": True}}
    assert up.requests and all(p != "/metrics" for _, p, _ in up.requests)


def test_readiness_fails_when_a_check_fails(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), janitor_interval_s=3600)
    with serve(create_app(m, ProxySettings(upstream="http://127.0.0.1:9"), ready=lambda: {"disk": False})) as proxy:
        r = httpx.get(f"{proxy}/readyz")
        assert r.status_code == 503 and r.json()["checks"]["disk"] is False
    m.close()


@pytest.fixture
def world():
    from jevstiller import SyntheticWorld
    return SyntheticWorld(["billing", "technical", "cancellation", "sales", "other"], seed=1)
