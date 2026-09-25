"""Regression tests for security audit run 1 (2026-09-24): one test per finding, each reproducing the
original attack against the fixed code. See SECURITY.md."""
import json
import socket
import time

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from test_proxy import GOOD, LABELS, QUESTION, serve

from jevstiller import Admission, Config, HashEncoder, SyntheticTeacher, SyntheticWorld, Task, TaskManager
from jevstiller._manager import task_key
from jevstiller.server import KeyRegistry, ProxySettings, create_app

CRITERIA = QUESTION["criteria"]


class Upstream:
    """A mock Jev that records what reaches it. /v1/systemone checks the key; /openapi.json does not
    (like a public schema endpoint); it can set a cookie on every response."""

    def __init__(self, set_cookie: bool = False):
        self.teacher = SyntheticTeacher(SyntheticWorld(LABELS, seed=1))
        self.requests: list[tuple[str, str, dict]] = []
        self.set_cookie = set_cookie

    async def handle(self, request: Request):
        body = await request.body()
        self.requests.append((request.method, request.url.path, dict(request.headers)))
        headers = {"x-typesafe-request-id": f"req_{len(self.requests)}"}
        if self.set_cookie:
            headers["set-cookie"] = f"session=caller{len(self.requests)}; Path=/"
        if request.url.path == "/openapi.json":
            return JSONResponse({"openapi": "3.1.0"}, headers=headers)
        if request.headers.get("authorization") != f"Bearer {GOOD}":
            return JSONResponse({"detail": "invalid API key"}, 401, headers=headers)
        if request.url.path != "/v1/systemone":
            return JSONResponse({"ok": True}, headers=headers)
        try:
            req = json.loads(body)
        except (ValueError, RecursionError):
            return JSONResponse({"detail": "invalid JSON"}, 422, headers=headers)
        answers = {}
        for name, q in req["questions"].items():
            if q.get("type") == "choice":
                o = self.teacher.classify([req["state"]], Task("t", q.get("instructions"), list(q["criteria"])))[0]
                answers[name] = {"type": "choice", "choice": o.label, "confidence": o.confidence,
                                 "probabilities": o.probs}
            else:
                answers[name] = {"type": "noul", "noul": 0.5}
        return JSONResponse({"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 10,
                                                                                  "output_tokens": 1}},
                            headers=headers)

    def app(self):
        return Starlette(routes=[Route("/{p:path}", self.handle, methods=["GET", "POST", "PUT", "DELETE"])])


def _manager(tmp_path, **kw):
    cfg = kw.pop("config", Config(training="manual"))
    return TaskManager(tmp_path, None, HashEncoder(dim=64), cfg, admission=kw.pop("admission", Admission(1)),
                       janitor_interval_s=3600, **kw)


def _body(n_names=1, questions=None, state="w1 w2"):
    qs = questions or {f"q{i}": QUESTION for i in range(n_names)}
    return {"state": state, "model": "jev-latest", "questions": qs}


def _post(proxy, body, key=GOOD, headers=None, **kw):
    h = {"Authorization": f"Bearer {key}", **(headers or {})}
    return httpx.post(f"{proxy}/v1/systemone", json=body, headers=h, timeout=30, **kw)


@pytest.fixture
def up():
    return Upstream()


# 1. any 2xx verified a key -> only a parsed /v1/systemone answer does
def test_public_upstream_path_does_not_verify_a_key(tmp_path, up):
    m = _manager(tmp_path)
    keys = KeyRegistry(b"s", 3600)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u), keys)) as proxy:
        for _ in range(5):
            assert httpx.get(f"{proxy}/openapi.json", headers={"Authorization": "Bearer fake"}).status_code == 200
        assert not keys.verified(keys.hash("fake"))
        r = _post(proxy, _body(), key="fake")
        assert r.status_code == 401 and r.headers["x-jevstiller-source"] == "upstream"
        _post(proxy, _body())
        assert keys.verified(keys.hash(GOOD))                     # a real systemone answer still verifies
    m.close()


# 2. health paths bypassed the access checks and were forwarded
def test_health_paths_are_local_and_gated(tmp_path, up):
    m = _manager(tmp_path)
    s = ProxySettings(access_token="a-long-enough-token", allow_networks=["10.0.0.0/8"])
    with serve(up.app()) as u:
        s.upstream = u
        with serve(create_app(m, s)) as proxy:
            assert httpx.get(f"{proxy}/healthz").json() == {"ok": True}
            assert httpx.get(f"{proxy}/readyz").status_code == 200
            for method in ("POST", "PUT", "DELETE", "OPTIONS"):
                for path in ("/healthz", "/readyz", "/%68ealthz"):
                    assert httpx.request(method, f"{proxy}{path}", content=b"x" * 1000).status_code == 403
            assert up.requests == []                             # nothing ever reached the upstream
    m.close()


# 3. callers without a valid key created persistent tasks
def test_unverified_keys_create_no_tasks(tmp_path, up):
    m = _manager(tmp_path, admission=Admission(50))
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        questions = {f"q{d}_{i}": {**QUESTION, "instructions": f"Question {d}?"} for d in range(20) for i in range(50)}
        r = _post(proxy, _body(questions=questions), key="garbage")
        assert r.status_code in (401, 200)
        assert m.tasks() == [] and not any((tmp_path / "tasks").iterdir())
    m.close()


# 3b. one request with the same question under 50 names no longer admits it
def test_duplicate_names_count_once_for_admission(tmp_path, up):
    m = _manager(tmp_path, admission=Admission(3))
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        _post(proxy, _body(n_names=50))
        assert m.tasks() == []
        _post(proxy, _body(n_names=50))
        _post(proxy, _body(n_names=50))
        assert len(m.tasks()) == 1                                # three requests, not one
    m.close()


# 4. chunked bodies bypassed max_body_bytes
def test_chunked_body_limit_is_enforced_while_streaming(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u, max_body_bytes=64 * 1024))) as proxy:
        def chunks():
            for _ in range(200):
                yield b"x" * 65536                               # 12.5 MiB, no Content-Length
        r = httpx.post(f"{proxy}/v1/systemone", content=chunks(), headers={"Authorization": f"Bearer {GOOD}"})
        assert r.status_code == 413 and up.requests == []
        r = httpx.post(f"{proxy}/anything", content=chunks(), headers={"Authorization": f"Bearer {GOOD}"})
        assert r.status_code == 413 and up.requests == []
    m.close()


# 5. unbounded questions per request
def test_question_cap(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u, max_questions=4))) as proxy:
        _post(proxy, _body())                                     # verify the key
        questions = {f"q{d}": {**QUESTION, "instructions": f"Question {d}?"} for d in range(5)}
        r = _post(proxy, _body(questions=questions))
        assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream"
        assert len(m.tasks()) == 1                                # the 5-question request routed nothing
    m.close()


# 6. duplicate Authorization headers
def test_duplicate_authorization_is_rejected(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        host, port = proxy.removeprefix("http://").split(":")
        body = json.dumps(_body()).encode()
        raw = (f"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer alias\r\n"
               f"Authorization: Bearer {GOOD}\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
        with socket.create_connection((host, int(port))) as sock:
            sock.sendall(raw)
            resp = sock.recv(65536).decode()
        assert resp.startswith("HTTP/1.1 400") and up.requests == []
    m.close()


# 7. dot segments escaped an upstream path prefix
def test_dot_segments_are_rejected(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u + "/jev"))) as proxy:
        host, port = proxy.removeprefix("http://").split(":")
        for target in ("/../secret", "/%2e%2e/secret", "/a/./b"):
            with socket.create_connection((host, int(port))) as sock:
                sock.sendall(f"GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
                assert sock.recv(4096).decode().startswith("HTTP/1.1 400"), target
        assert up.requests == []
    m.close()


# 8. X-Forwarded-For trusted by default
def test_cli_trusts_forwarded_for_only_when_configured(tmp_path, monkeypatch):
    import uvicorn

    from jevstiller import _cli as cli
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    base = ["serve", "--data-dir", str(tmp_path), "--encoder", "hash", "--log-level", "warning"]
    cli.main(base)
    cli.main(base + ["--trust-forwarded-for", "10.0.0.5"])
    assert calls[0]["proxy_headers"] is False and calls[0]["forwarded_allow_ips"] is None
    assert calls[1]["proxy_headers"] is True and calls[1]["forwarded_allow_ips"] == "10.0.0.5"


# 9. retention left text in the WAL; files readable by others; unsalted hash
def test_retention_removes_text_from_the_files(tmp_path):
    t = SyntheticTeacher(SyntheticWorld(LABELS, seed=1))
    m = TaskManager(tmp_path, t, HashEncoder(dim=64), Config(training="manual"), admission=Admission(1),
                    janitor_interval_s=3600, text_retention_s=1, hash_key=b"k" * 32)
    secrets_ = [f"customer-secret-{i:04d} w1" for i in range(300)]
    m.classify("acme", "Q?", LABELS, secrets_)
    e = m.engine(m.tasks()[0].key)
    e.store.db.execute("UPDATE samples SET ts = ?", (time.time() - 10,))
    e.store.db.commit()
    assert m.apply_retention() == 300
    files = [p for p in (tmp_path / "tasks").rglob("samples.sqlite*")]
    blob = b"".join(p.read_bytes() for p in files)
    assert b"customer-secret" not in blob
    hashes = [r[0] for r in e.store.db.execute("SELECT text_hash FROM samples")]
    import hashlib
    assert hashlib.sha256(secrets_[0].encode()).hexdigest()[:16] not in hashes   # keyed (HMAC), not plain
    assert ((tmp_path / "tasks").stat().st_mode & 0o077) == 0
    m.close()


# 10. 48-bit task identity
def test_task_key_uses_the_full_question_fingerprint(monkeypatch):
    a, b = Task("t", "Question A?", LABELS), Task("t", "Question B?", LABELS)
    monkeypatch.setattr(Task, "version", property(lambda self: "000000000000"))   # a 48-bit collision
    assert a.version == b.version and task_key("t", a) != task_key("t", b)


def test_tasks_stored_under_old_keys_are_re_keyed(tmp_path):
    import hashlib

    from jevstiller._task import canonical_json
    t = SyntheticTeacher(SyntheticWorld(LABELS, seed=1))
    m = TaskManager(tmp_path, t, HashEncoder(dim=64), Config(training="manual"), admission=Admission(1),
                    janitor_interval_s=3600)
    m.classify("acme", "Q?", LABELS, ["w1 w2"])
    new_key = m.tasks()[0].key
    m.close()
    old_key = hashlib.sha256(canonical_json({"tenant": "acme", "type": "choice",
                                             "version": Task("_", "Q?", LABELS).version}).encode()).hexdigest()[:20]
    (tmp_path / "tasks" / new_key).rename(tmp_path / "tasks" / old_key)
    info = json.loads((tmp_path / "tasks" / old_key / "task.json").read_text())
    (tmp_path / "tasks" / old_key / "task.json").write_text(json.dumps({**info, "key": old_key}))
    m2 = TaskManager(tmp_path, t, HashEncoder(dim=64), Config(training="manual"), admission=Admission(100),
                     janitor_interval_s=3600)
    assert [i.key for i in m2.tasks()] == [new_key] and (tmp_path / "tasks" / new_key).exists()
    assert m2.engine(new_key).store.counts(m2.engine(new_key).task.version)["total"] == 1
    m2.close()


# 11. cookies shared across callers
def test_upstream_cookies_are_not_shared(tmp_path):
    up = Upstream(set_cookie=True)
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        for _ in range(3):
            _post(proxy, _body())
        assert all("cookie" not in h for _, _, h in up.requests)
    m.close()


# hardening: fail open on unparseable JSON; malformed upstream answers release engines
def test_deeply_nested_json_is_forwarded(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        body = b'{"state": ' + b"[" * 100_000 + b"]" * 100_000 + b', "model": "m", "questions": {}}'
        headers = {"Authorization": f"Bearer {GOOD}"}
        direct = httpx.post(f"{u}/v1/systemone", content=body, headers=headers)
        r = httpx.post(f"{proxy}/v1/systemone", content=body, headers=headers)
        # Jev's verdict, whatever it is (Python 3.14.7 parses this nesting; earlier versions give up), and
        # never a proxy 500
        assert r.status_code == direct.status_code != 500 and len(up.requests) == 2
    m.close()


def test_malformed_upstream_answer_releases_engines(tmp_path):
    async def bad(request: Request):
        return JSONResponse({"model": "jev-1.13.0", "usage": {"input_tokens": "lots"},
                             "answers": {"q0": {"type": "choice", "choice": "billing",
                                                "probabilities": {"billing": "high"}}}})
    m = _manager(tmp_path)
    with serve(Starlette(routes=[Route("/v1/systemone", bad, methods=["POST"])])) as u:
        keys = KeyRegistry(b"s", 3600)
        with serve(create_app(m, ProxySettings(upstream=u), keys)) as proxy:
            for _ in range(3):
                assert _post(proxy, _body()).status_code == 200
    assert m.tasks() and all(v == 0 for v in m._inflight.values())
    m.close()
