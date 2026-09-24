"""P4: tenants file, proxy access control, body limit, key handling, text retention, tenant deletion."""
import logging
import os
import stat
import time

import httpx
import pytest
from test_proxy import GOOD, LABELS, OTHER, QUESTION, FakeJev, ask, client, serve

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller.server import KeyRegistry, ProxySettings, create_app, load_salt

ts = pytest.importorskip("typesafe_sdk")


@pytest.fixture
def jev(world):
    return FakeJev(world)


@pytest.fixture
def world():
    from jevstiller import SyntheticWorld
    return SyntheticWorld(LABELS, seed=1)


def _manager(tmp_path, **kw):
    return TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                       admission=Admission(min_requests=1), janitor_interval_s=3600, **kw)


def _body():
    return {"state": "w1 w2", "model": "jev-latest", "questions": {"label": QUESTION}}


def test_tenants_file_maps_keys(tmp_path, jev):
    keys = KeyRegistry(b"salt", 3600)
    settings = ProxySettings(tenancy="per_key", tenant_map={keys.hash(GOOD): "acme", keys.hash(OTHER): "acme"})
    m = _manager(tmp_path)
    with serve(jev.app()) as up:
        settings.upstream = up
        with serve(create_app(m, settings, keys)) as proxy:
            ask(client(proxy, GOOD), "w1")
            ask(client(proxy, OTHER), "w1")
    assert [i.tenant for i in m.tasks()] == ["acme"]          # both keys share one task
    m.close()


def test_access_token(tmp_path, jev):
    m = _manager(tmp_path)
    with serve(jev.app()) as up, serve(create_app(m, ProxySettings(upstream=up, access_token="s3cret"))) as proxy:
        h = {"Authorization": f"Bearer {GOOD}"}
        assert httpx.post(f"{proxy}/v1/systemone", headers=h, json=_body()).status_code == 401
        bad = httpx.post(f"{proxy}/v1/systemone", headers={**h, "x-jevstiller-token": "nope"}, json=_body())
        assert bad.status_code == 401 and jev.calls == 0
        ok = httpx.post(f"{proxy}/v1/systemone", headers={**h, "x-jevstiller-token": "s3cret"}, json=_body())
        assert ok.status_code == 200
        assert "x-jevstiller-token" not in jev.last_headers          # never forwarded to Jev
        c = ts.TypeSafeClient(base_url=proxy, api_key=GOOD, headers={"x-jevstiller-token": "s3cret"})
        assert ask(c, "w3").choices["label"].choice in LABELS         # the SDK can send it
        assert httpx.get(f"{proxy}/healthz").json() == {"ok": True}  # probes need no token
    m.close()


def test_allowed_networks(tmp_path, jev):
    m = _manager(tmp_path)
    with serve(jev.app()) as up:
        with serve(create_app(m, ProxySettings(upstream=up, allow_networks=["10.0.0.0/8"]))) as proxy:
            r = httpx.post(f"{proxy}/v1/systemone", headers={"Authorization": f"Bearer {GOOD}"}, json=_body())
            assert r.status_code == 403 and jev.calls == 0
            assert httpx.get(f"{proxy}/healthz").status_code == 200
        with serve(create_app(m, ProxySettings(upstream=up, allow_networks=["127.0.0.0/8", "::1/128"]))) as proxy:
            assert ask(client(proxy), "w1").choices["label"].choice in LABELS
    m.close()


def test_body_limit(tmp_path, jev):
    m = _manager(tmp_path)
    with serve(jev.app()) as up, serve(create_app(m, ProxySettings(upstream=up, max_body_bytes=2000))) as proxy:
        big = {**_body(), "state": "w1 " * 2000}
        r = httpx.post(f"{proxy}/v1/systemone", headers={"Authorization": f"Bearer {GOOD}"}, json=big)
        assert r.status_code == 413 and jev.calls == 0 and r.json()["detail"]
        assert httpx.post(f"{proxy}/v1/systemone", headers={"Authorization": f"Bearer {GOOD}"},
                          json=_body()).status_code == 200
    m.close()


def test_keys_never_reach_disk_or_logs(tmp_path, jev, caplog):
    caplog.set_level(logging.DEBUG)
    salt = load_salt(tmp_path)
    assert stat.S_IMODE(os.stat(tmp_path / "key-salt").st_mode) == 0o600
    m = _manager(tmp_path)
    with serve(jev.app()) as up, serve(create_app(m, ProxySettings(upstream=up), KeyRegistry(salt, 3600))) as proxy:
        for i in range(5):
            ask(client(proxy), f"w{i} w9")
        with pytest.raises(ts.TypeSafeAuthenticationError):
            ask(client(proxy, "tsk_leaky_key_123"), "w1")
    m.close()
    for key in (GOOD, "tsk_leaky_key_123"):
        for f in tmp_path.rglob("*"):
            if f.is_file():
                assert key.encode() not in f.read_bytes(), f
        assert key not in caplog.text


def test_text_retention_and_tenant_deletion(tmp_path, world):
    from jevstiller import SyntheticTeacher
    m = TaskManager(tmp_path, SyntheticTeacher(world), HashEncoder(dim=64), Config(training="manual"),
                    admission=Admission(min_requests=1), janitor_interval_s=3600, text_retention_s=3600)
    m.classify("acme", "Q1?", LABELS, ["old w1", "old w2"])
    m.classify("acme", "Q2?", LABELS, ["old w3"])
    m.classify("globex", "Q1?", LABELS, ["new w4"])
    k1 = m.tasks("acme")[0].key
    for info in m.tasks():                              # age the acme rows; unload one task first
        e = m.engine(info.key)
        if info.tenant == "acme":
            e.store.db.execute("UPDATE samples SET ts = ?", (time.time() - 7200,))
            e.store.db.commit()
    m.max_loaded = 1
    m.sweep()                                           # unloads two tasks, and applies retention (due)
    assert m.apply_retention() == 0                     # nothing left: loaded and unloaded stores were both done
    texts = m.engine(k1).store.db.execute("SELECT text, text_hash, embedding IS NOT NULL FROM samples").fetchall()
    assert all(t == "" and h and e for t, h, e in texts)      # text gone; hash and embedding kept
    g = m.tasks("globex")[0].key
    assert m.engine(g).store.db.execute("SELECT text FROM samples").fetchone()[0] == "new w4"
    deleted = m.delete_tenant("acme")
    assert len(deleted) == 2 and [i.tenant for i in m.tasks()] == ["globex"]
    assert not any((tmp_path / "tasks" / k).exists() for k in deleted)
    m.close()
