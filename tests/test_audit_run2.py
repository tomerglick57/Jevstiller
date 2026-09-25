"""Regression tests for security audit run 2 (2026-09-25): one or more tests per finding, each reproducing the
original problem against the fixed code. See docs/security.md."""
import json
import os
import socket
import stat
import threading
import time

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from test_audit_fixes import Upstream, _body, _manager, _post
from test_proxy import GOOD, LABELS, QUESTION, serve

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller._server import KeyRegistry, ProxySettings, _retry_after_s, create_app, load_salt
from jevstiller._settings import load
from jevstiller.encoders._batching import BatchingEncoder


@pytest.fixture
def up():
    return Upstream()


def _raw(proxy: str, request: bytes) -> str:
    host, port = proxy.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port))) as sock:
        sock.sendall(request)
        return sock.recv(65536).decode("latin-1")


def _rows(m: TaskManager) -> int:
    e = m.engine(m.tasks()[0].key)
    e.store.flush()
    return e.store.counts(e.task.version)["total"]


def _toml(tmp_path, text: str):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


# 1. one slow encode latched encoder backpressure on until restart
def test_encoder_input_is_capped():
    seen = []

    class Inner(HashEncoder):
        def encode(self, texts):
            seen.extend(len(t) for t in texts)
            return super().encode(texts)
    enc = BatchingEncoder(Inner(dim=16))
    enc.encode(["x " * 2_000_000])                        # a 4 MB state
    assert seen == [enc.max_chars] and enc.max_chars <= 65_536
    enc.close()


def test_a_stale_encoder_estimate_does_not_latch_backpressure(tmp_path, up):
    m = TaskManager(tmp_path, None, BatchingEncoder(HashEncoder(dim=64)), Config(training="manual"),
                    admission=Admission(1), janitor_interval_s=3600)
    with serve(up.app()) as u:
        app = create_app(m, ProxySettings(upstream=u))
        with serve(app) as proxy:
            _post(proxy, _body())                         # key accepted, task created
            rows = _rows(m)
            m.encoder._per_text_s = 5.0                   # as after one huge (or stalled) encode
            for i in range(3):
                assert _post(proxy, _body(state=f"w{i} w9")).status_code == 200
            assert _rows(m) == rows + 3                   # routed and recorded, not forwarded unrouted
            assert m.encoder.time_per_text_s() < 5.0 * 0.8 ** 2   # the estimate recovers
            assert app.state.proxy.metrics.questions.value("overloaded") == 0
    m.close()


# 2. a non-ASCII header byte made the proxy fail with 500 and never release the routed engines
def test_non_ascii_header_is_forwarded_and_engines_released(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        _post(proxy, _body())                             # key accepted, task created
        body = json.dumps(_body(questions={"q": QUESTION, "n": {"type": "noul"}})).encode()
        resp = _raw(proxy, (f"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {GOOD}\r\n"
                            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                            f"Connection: close\r\n").encode() + b"X-Note: Z\xfcrich \xff\r\n\r\n" + body)
        assert resp.startswith("HTTP/1.1 200"), resp[:200]
        assert up.requests[-1][2]["x-note"] == "Z\xfcrich \xff"     # the bytes reached Jev unchanged
    assert all(v == 0 for v in m._inflight.values())
    assert m.delete(m.tasks()[0].key)                     # not pinned "in flight" forever
    m.close()


def test_any_forwarding_failure_is_a_502_and_releases_engines(tmp_path, up, monkeypatch):
    m = _manager(tmp_path)
    with serve(up.app()) as u:
        app = create_app(m, ProxySettings(upstream=u))
        with serve(app) as proxy:
            _post(proxy, _body())
            for c in app.state.proxy.clients:
                monkeypatch.setattr(c, "send", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
            r = _post(proxy, _body(questions={"q": QUESTION, "n": {"type": "noul"}}))
            assert r.status_code == 502
    assert all(v == 0 for v in m._inflight.values())
    m.close()


def test_retry_after_is_finite_and_capped():
    assert _retry_after_s({"retry-after": "inf"}) == 0.0
    assert _retry_after_s({"retry-after": "nan"}) == 0.0
    assert _retry_after_s({"retry-after-ms": "-5"}) == 0.0
    assert _retry_after_s({"retry-after": "1e12"}) == 3600.0
    assert _retry_after_s({"retry-after": "2.5"}) == 2.5


# 3. closing a store while another thread used it crashed the process (SIGSEGV)
def test_retention_holds_the_engine_it_redacts(tmp_path):
    m = _manager(tmp_path, text_retention_s=0.0001)
    r = m.route("acme", "Q?", LABELS, ["w1 w2"])          # admitted and loaded
    m.complete(r, [RuntimeError("no answer")] * len(r.routed.to_teacher), "jev")
    key = m.tasks()[0].key
    e = m.engine(key)
    during = {}
    real = e.store.redact_text

    def redact(cutoff):
        m.max_loaded = 0
        during["unloaded"] = m._unload_one()             # a request thread over the cap, mid-retention
        during["deleted"] = m.delete(key)
        return real(cutoff)
    e.store.redact_text = redact
    m.apply_retention()
    assert during == {"unloaded": False, "deleted": False} and key in m.loaded()
    m.close()


def test_admin_calls_hold_the_engine(tmp_path):
    from jevstiller._admin import Admin
    m = _manager(tmp_path)
    r = m.route("acme", "Q?", LABELS, ["w1 w2"])
    m.complete(r, [RuntimeError("no answer")] * len(r.routed.to_teacher), "jev")
    key = m.tasks()[0].key
    during = {}
    e = m.engine(key)
    real = e.status
    seen_before = m.tasks()[0].last_seen

    def status():
        during["deleted"] = m.delete(key)                 # e.g. an idle-TTL sweep during the call
        return real()
    e.status = status
    admin = Admin(m, "a-long-enough-admin-token")
    with serve(Starlette(routes=admin.routes())) as url:
        r = httpx.get(f"{url}/jevstiller/v1/tasks/{key}", headers={"Authorization": "Bearer a-long-enough-admin-token"})
    assert r.status_code == 200 and during == {"deleted": False}
    assert m.tasks()[0].last_seen == seen_before          # looking at a task is not using it
    m.close()


def test_store_close_waits_for_a_write_in_progress(tmp_path):
    from jevstiller._store import SampleStore
    s = SampleStore(tmp_path / "s.sqlite")
    closed = threading.Event()
    with s._write_lock:                                   # e.g. redact_text's UPDATE
        t = threading.Thread(target=lambda: (s.close(), closed.set()))
        t.start()
        time.sleep(0.2)
        assert not closed.is_set()
    t.join(5)
    assert closed.is_set()


# 4. privacy settings silently ignored
def test_store_text_false_under_engine_is_honoured(tmp_path):
    s = load(_toml(tmp_path, "[engine]\nstore_text = false\n"), environ={})
    assert s.store_text is False and "store_text" not in s.engine
    assert load(_toml(tmp_path, "[engine]\nstore_text = true\n[manager]\nstore_text = false\n"),
                environ={}).store_text is False


@pytest.mark.parametrize("text", [
    '[manager]\nstore_text = "false"\n',
    '[server]\nmetrics_public = "false"\n',
    '[manager]\ntext_retention_days = "30"\n',
    "[manager]\ntext_retention_days = 0\n",
    "[manager]\nidle_ttl_days = -1\n",
    '[engine]\naudit_rate = "0.03"\n',
    "[proxy]\nkey_ttl_s = inf\n",
])
def test_mistyped_or_meaningless_values_stop_startup(tmp_path, text):
    with pytest.raises(ValueError):
        load(_toml(tmp_path, text), environ={})


# 5. the dot-segment check was bypassable with control characters or an encoded ? or #
def test_dot_segments_hidden_by_control_characters_are_rejected(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u + "/jev"))) as proxy:
        for target in ("/.%09./internal", "/.%0d./.%0d./x?y=1", "/..%3F", "/..%23", "/.%01./x", "/a%00b"):
            resp = _raw(proxy, f"GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
            assert resp.startswith("HTTP/1.1 400"), target
        assert up.requests == []
        assert httpx.get(f"{proxy}/v1/models", headers={"Authorization": f"Bearer {GOOD}"}).status_code == 200
        assert up.requests[-1][1] == "/jev/v1/models"
    m.close()


# 6. concurrent /readyz calls raced on one probe file and reported the data dir unwritable
def test_readyz_under_concurrent_probes(tmp_path, monkeypatch):
    import uvicorn

    from jevstiller import _cli as cli
    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: apps.append(app))
    cli.main(["serve", "--data-dir", str(tmp_path), "--encoder", "hash", "--log-level", "warning"])
    results = []
    with serve(apps[0]) as proxy:
        def probe():
            with httpx.Client() as c:
                results.extend(c.get(f"{proxy}/readyz").status_code for _ in range(25))
        threads = [threading.Thread(target=probe) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert results == [200] * 200
    assert not list(tmp_path.glob(".ready-probe*"))


# 7. before Python 3.14, the training pool forked the live server
def test_training_workers_are_not_forked_from_the_server():
    from jevstiller._training import train_pool
    pool = train_pool(1)
    try:
        assert pool._mp_context.get_start_method() in ("forkserver", "spawn")
    finally:
        pool.shutdown()


# 8. empty values silently switched access control off
@pytest.mark.parametrize("env", [
    {"JEVSTILLER_ACCESS_TOKEN": ""},
    {"JEVSTILLER_ACCESS_TOKEN": "  "},
    {"JEVSTILLER_ADMIN_TOKEN": ""},
    {"JEVSTILLER_ALLOW_NETWORKS": ""},
    {"JEVSTILLER_TRUST_FORWARDED_FOR": ""},
    {"JEVSTILLER_TENANTS_FILE": ""},
    {"JEVSTILLER_ACCESS_TOKEN_FILE": ""},
])
def test_empty_access_control_values_stop_startup(env):
    with pytest.raises(ValueError, match="empty"):
        load(environ=env)


def test_empty_secret_files_stop_startup(tmp_path):
    (tmp_path / "token").write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load(environ={"JEVSTILLER_ACCESS_TOKEN_FILE": str(tmp_path / "token")})
    with pytest.raises(ValueError, match="empty"):
        load(_toml(tmp_path, '[proxy]\naccess_token = ""\n'), environ={})
    # clearing a config file's list on purpose is still possible
    p = _toml(tmp_path, '[proxy]\nallow_networks = ["10.0.0.0/8"]\n')
    assert load(p, environ={"JEVSTILLER_ALLOW_NETWORKS": "none"}).allow_networks == []


# 9. key_ttl_s = inf verified every bearer string
def test_unknown_keys_are_never_verified():
    keys = KeyRegistry(b"s", float("inf"))
    assert not keys.verified(keys.hash("made-up"))
    keys.accept(keys.hash(GOOD))
    assert keys.verified(keys.hash(GOOD))
    keys.revoke(keys.hash(GOOD))
    assert not keys.verified(keys.hash(GOOD))


# 10. trust_forwarded_for entries with host bits were accepted, then ignored by uvicorn
def test_trusted_proxies_must_be_exact_networks():
    with pytest.raises(ValueError, match="host bits"):
        load(environ={"JEVSTILLER_TRUST_FORWARDED_FOR": "10.0.0.5/24"})
    assert load(environ={"JEVSTILLER_TRUST_FORWARDED_FOR": "10.0.0.0/24,10.0.0.5"}).trust_forwarded_for == [
        "10.0.0.0/24", "10.0.0.5"]


# 11. secrets in logs
def test_tenants_error_names_the_tenant_not_the_key(tmp_path):
    raw = "tsk_live_9f8e7d6c5b4a39281706f5e4d3c2b1a0"
    (tmp_path / "t.json").write_text(json.dumps({raw: "acme"}))
    with pytest.raises(ValueError) as e:
        load(environ={"JEVSTILLER_TENANTS_FILE": str(tmp_path / "t.json")})
    assert "acme" in str(e.value) and raw[:12] not in str(e.value)


def test_upstream_credentials_are_redacted(tmp_path, monkeypatch, capsys):
    import uvicorn

    from jevstiller import _cli as cli
    from jevstiller._settings import redact_url
    assert redact_url("https://u:pw@gw.example:8443/jev") == "https://***@gw.example:8443/jev"
    assert redact_url("https://api.typesafe.ai") == "https://api.typesafe.ai"
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    up = "https://gateway-user:s3cret-pw@127.0.0.1:9"
    cli.main(["config", "--upstream", up])
    cli.main(["serve", "--data-dir", str(tmp_path), "--encoder", "hash", "--upstream", up])
    out = capsys.readouterr()
    assert "s3cret-pw" not in out.out + out.err and "***@127.0.0.1:9" in out.out


# 12. backup and restore ignored the 0600/0700 policy
def test_backup_and_restore_are_owner_only(tmp_path):
    from jevstiller._backup import backup, restore
    src = tmp_path / "data"
    load_salt(src)
    m = _manager(src)
    r = m.route("acme", "Q?", LABELS, ["w1 w2"])
    m.complete(r, [RuntimeError("no answer")] * len(r.routed.to_teacher), "jev")
    m.close()
    old = os.umask(0o022)                                 # a typical shell
    try:
        backup(src, tmp_path / "bk")
        for p in (tmp_path / "bk").rglob("*"):            # a copy with the source's (or worse) modes
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        restore(tmp_path / "bk", tmp_path / "restored")
    finally:
        os.umask(old)
    root = tmp_path / "restored"
    for p in [root, *root.rglob("*")]:
        assert stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600), p
    old = os.umask(0o022)
    try:
        backup(src, tmp_path / "bk2")
    finally:
        os.umask(old)
    for p in [tmp_path / "bk2", *(tmp_path / "bk2").rglob("*")]:
        assert stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600), p


# 13. caller-chosen class names reached the operator's terminal raw
def test_class_names_are_inert_in_the_report(tmp_path):
    m = _manager(tmp_path)
    evil = "x\x1b]52;c;cm0gLXJmIH4=\x07\x1b[2K\x1b[1AAgreement: OK"
    r = m.route("acme", "Q?", {evil: "", "b": ""}, ["w1 w2"])
    m.complete(r, [RuntimeError("no answer")] * len(r.routed.to_teacher), "jev")
    report = m.engine(m.tasks()[0].key).status().report()
    assert "\x1b" not in report and "\x07" not in report and "\\x1b]52" in report
    m.close()


# 14. tasks were created for questions Jev rejects
def test_only_answered_questions_count_towards_admission(tmp_path, up):
    async def jev(request: Request):
        req = json.loads(await request.body())
        if req["model"] == "rejected":
            return JSONResponse({"detail": "unknown model"}, 400)
        return await up.handle(request)
    m = _manager(tmp_path, admission=Admission(3))
    with serve(Starlette(routes=[Route("/{p:path}", jev, methods=["POST"])])) as u, \
            serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        _post(proxy, {**_body(), "questions": {"n": {"type": "noul"}, "q": QUESTION}})   # accept the key
        new = {"q": {**QUESTION, "instructions": "A new question?"}}
        for _ in range(10):
            assert _post(proxy, {**_body(questions=new), "model": "rejected"}).status_code == 400
        assert m.tasks() == []                            # ten rejected requests, no task
        for _ in range(2):
            _post(proxy, _body(questions=new))
        assert m.tasks() == []
        _post(proxy, _body(questions=new))                # the third answered request admits it ...
        assert len(m.tasks()) == 1 and _rows(m) == 1      # ... and its answer is the task's first row
    m.close()


# 15. the admin CLI pasted names into URLs unescaped
def test_admin_cli_escapes_names(monkeypatch):
    from jevstiller import _cli as cli
    calls = []

    def request(method, url, **kw):
        calls.append((method, url))
        return httpx.Response(200, json={"deleted": [], "remaining": []})
    monkeypatch.setattr(httpx, "request", request)
    cli.main(["admin", "--token", "t", "delete-tenant", "acme#eu"])
    assert calls == [("DELETE", "http://127.0.0.1:8080/jevstiller/v1/tenants/acme%23eu")]
    with pytest.raises(SystemExit):
        cli.main(["admin", "--token", "t", "delete-tenant"])
    assert len(calls) == 1


# hardening: the proxy relayed Jev's server and date headers next to uvicorn's own
def test_no_duplicate_server_or_date_headers(tmp_path, up):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        r = httpx.get(f"{proxy}/openapi.json")
        assert len(r.headers.get_list("date")) == 1 and len(r.headers.get_list("server")) <= 1
    m.close()
