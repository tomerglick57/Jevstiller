"""Regression tests for security audit run 3 (2026-09-25). See docs/security.md."""
import json
import logging
import socket

import httpx
import numpy as np
import pytest
from test_admin import TOKEN, stack  # noqa: F401  (fixture)
from test_audit_fixes import Upstream, _body, _manager, _post
from test_proxy import LABELS, QUESTION, serve

from jevstiller import Admission, Config, HashEncoder, Jevstiller, Task, TaskManager
from jevstiller._settings import load
from jevstiller._store import Record, text_hash
from jevstiller.server import ProxySettings, create_app


def _trained(tmp_path, teacher, world):
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=150, seed=0, training="inline")
    task = Task("support", "Which team handles this?", LABELS, target_agreement=0.95)
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=128), config=cfg)
    for _ in range(40):
        js.classify_batch([t for t, _ in world.sample(100)])
        if js.status().production:
            return js
    raise AssertionError("no student was promoted")


# 1. the audit re-scored its window with a student that had since been trained on it
def test_the_audit_scores_what_was_served(tmp_path, teacher, world):
    js = _trained(tmp_path, teacher, world)
    prod = js._prod
    lineage = js._teacher_model
    before = js._audit_stats(prod)[0]

    def audit_row(version, student_label, teacher_label="billing"):
        return Record(text="t", task_version=js.task.version, encoder_id=js.encoder.id,
                      embedding=np.zeros(128, np.float32), served_by="teacher", routing_reason="audit",
                      channel="audit", teacher_label=teacher_label, teacher_probs={teacher_label: 1.0},
                      teacher_model=lineage, student_version=version, student_label=student_label,
                      student_confidence=1.0, ood_score=0.0)
    # served by production, confidently answered, and different from the teacher: whatever production would say
    # about these rows now (e.g. after training on them), they were disagreements when served
    js.store.insert([audit_row(prod.name, "technical") for _ in range(50)])
    # rows another version served don't measure this one
    js.store.insert([audit_row("student:v999", "technical") for _ in range(50)])
    js.store.flush()
    N, a, lb, ub = js._audit_stats(prod)
    assert N == min(before + 50, js.cfg.drift_window)
    assert a <= 1 - 50 / N + 1e-9                          # every planted row counted as a disagreement
    js.close()


# 3. the stored text was the whole state, however large, once per task
def test_stored_text_is_cut_like_the_encoder_input(tmp_path, teacher):
    from jevstiller._task import MAX_TEXT_CHARS
    task = Task("t", "Which team handles this?", LABELS)
    js = Jevstiller(task, teacher, tmp_path, encoder=HashEncoder(dim=32), config=Config(training="manual"))
    big = "word " * 50_000                                 # 250 KB
    js.classify(big)
    js.store.flush()
    text, h = js.store.db.execute("SELECT text, text_hash FROM samples").fetchone()
    assert len(text) == MAX_TEXT_CHARS and text == big[:MAX_TEXT_CHARS]
    assert h == text_hash(big)                             # the identity still covers the whole state
    js.close()


# 4. one caller could fill a shared tenant's task cap
def test_each_caller_creates_a_limited_number_of_tasks(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=32), Config(training="manual"), admission=Admission(1),
                    max_new_tasks_per_caller=2, janitor_interval_s=3600)

    def ask(caller, i):
        r = m.route("default", f"Question {i}?", LABELS, ["w1 w2"], caller=caller)
        if r.engine is not None:
            m.complete(r, [RuntimeError("no answer")] * len(r.routed.to_teacher), "jev")
        return r.reason
    assert [ask("a", i) for i in range(3)] == [None, None, "caller_task_limit"]
    assert ask("b", 3) is None                             # other callers are unaffected
    assert ask("a", 0) is None                             # existing tasks keep working for everyone
    assert len(m.tasks()) == 3
    m.close()


def test_the_proxy_applies_the_quota_per_key(tmp_path):
    up = Upstream()
    m = TaskManager(tmp_path, None, HashEncoder(dim=32), Config(training="manual"), admission=Admission(1),
                    max_new_tasks_per_caller=1, janitor_interval_s=3600)
    questions = {f"q{i}": {**QUESTION, "instructions": f"Question {i}?"} for i in range(3)}
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u))) as proxy:
        _post(proxy, _body(questions=questions))
        r = _post(proxy, _body(questions=questions))
    assert len(m.tasks()) == 1 and "caller_task_limit" in r.headers["x-jevstiller-detail"]
    m.close()


# 5. a blank config path or environment value silently reset settings
@pytest.mark.parametrize("env", [
    {"JEVSTILLER_CONFIG": ""},
    {"JEVSTILLER_ALLOW_NETWORKS": ","},
    {"JEVSTILLER_TRUST_FORWARDED_FOR": " , "},
    {"JEVSTILLER_TEXT_RETENTION_DAYS": ""},
    {"JEVSTILLER_MAX_TASKS": " "},
])
def test_blank_values_stop_startup(env):
    with pytest.raises(ValueError, match="empty|no values"):
        load(environ=env)


def test_none_still_clears_a_value(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[proxy]\nallow_networks = ["10.0.0.0/8"]\n[manager]\ntext_retention_days = 30\n')
    s = load(p, environ={"JEVSTILLER_ALLOW_NETWORKS": "none", "JEVSTILLER_TEXT_RETENTION_DAYS": "none"})
    assert s.allow_networks == [] and s.text_retention_days is None
    with pytest.raises(ValueError, match="empty"):
        load("", environ={})


# 6. malformed requests got a 500 and logged a traceback, with no credentials
def _raw(proxy, target, extra=b""):
    host, port = proxy.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port))) as sock:
        sock.sendall(f"GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n".encode() + extra + b"\r\n")
        return sock.recv(65536).decode("latin-1")


def test_malformed_basic_auth_is_a_401(stack):  # noqa: F811
    proxy, up, _, _ = stack
    for value in (b"\xff", b"YWRtaW46\xe9", "é".encode() * 100):
        resp = _raw(proxy, "/jevstiller/status", b"Authorization: Basic " + value + b"\r\n")
        assert resp.startswith("HTTP/1.1 401"), resp[:80]
    assert up.requests == []


def test_unusual_paths_are_400_without_a_traceback(tmp_path, up, caplog):
    m = _manager(tmp_path)
    with serve(up.app()) as u, serve(create_app(m, ProxySettings(upstream=u + "/jev"))) as proxy:
        with caplog.at_level(logging.INFO):
            for target in ("//x:abc/", "//x/metrics", "/%252e%252e/x", "/a%252fb", "/a%5Cb", "/a\\b"):
                resp = _raw(proxy, target)
                assert resp.startswith("HTTP/1.1 400"), target
        assert not [r for r in caplog.records if r.exc_info], "no traceback for a caller's malformed request"
        assert up.requests == []
        assert httpx.get(f"{proxy}/v1/models", headers={"Authorization": "Bearer x"}).status_code == 401
        assert up.requests[-1][1] == "/jev/v1/models"
    m.close()


@pytest.fixture
def up():
    return Upstream()


# 8. settings errors printed the value the operator had put in the wrong place
def test_settings_errors_never_print_secrets(tmp_path):
    raw = "tsk_live_9f8e7d6c5b4a39281706f5e4d3c2b1a0"
    for mapping in ({raw: "acme"}, {"acme": raw}):         # the right way round, and reversed
        (tmp_path / "t.json").write_text(json.dumps(mapping))
        with pytest.raises(ValueError) as e:
            load(environ={"JEVSTILLER_TENANTS_FILE": str(tmp_path / "t.json")})
        assert raw[:12] not in str(e.value) and "acme" not in str(e.value) and "entry #1" in str(e.value)
    with pytest.raises(ValueError) as e:                   # the token itself where its file's path belongs
        load(environ={"JEVSTILLER_ACCESS_TOKEN_FILE": "s3cret-token-value-0123456789"})
    assert "s3cret" not in str(e.value)


# hardening: a full data dir no longer fails readiness (it took the only replica out of service)
def test_readiness_does_not_depend_on_disk_space(tmp_path, monkeypatch):
    import uvicorn

    from jevstiller import _cli as cli
    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: apps.append(app))
    cli.main(["serve", "--data-dir", str(tmp_path), "--encoder", "hash", "--log-level", "warning"])
    tmp_path.chmod(0o500)                                 # the data dir stops accepting writes
    try:
        with serve(apps[0]) as proxy:
            r = httpx.get(f"{proxy}/readyz")
            assert r.status_code == 200 and "data_dir_writable" not in r.json()["checks"]
            metrics = apps[0].state.metrics.render()
            assert "jevstiller_data_dir_writable 0" in metrics
    finally:
        tmp_path.chmod(0o700)
