"""Regression tests for security audit run 4 (2026-10-09). See docs/security.md."""
import json
import threading
import time

import httpx
import numpy as np
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from test_audit_fixes import Upstream, _body, _post
from test_proxy import GOOD, LABELS, QUESTION, serve

from jevstiller import Admission, Config, HashEncoder, Jevstiller, Task, TaskManager, _core
from jevstiller import _manager as manager_module
from jevstiller._settings import ServeSettings
from jevstiller.server import KeyRegistry, ProxySettings, create_app

CHOICE = (QUESTION["instructions"], QUESTION["criteria"])
YES_NO = "Is this about money?"


def _manager(tmp_path, **kw):
    return TaskManager(tmp_path, _Teacher(), HashEncoder(dim=64), Config(training="manual"), admission=Admission(1),
                       janitor_interval_s=3600, **kw)


def _with_tasks(tmp_path, **kw):
    """A manager holding one choice task and one yes/no task (both admitted, no student)."""
    m = _manager(tmp_path, **kw)
    m.classify("acme", *CHOICE, ["w1 w2"])
    m.classify("acme", YES_NO, None, ["w1 w2"], kind="noul")
    keys = {i.kind: i.key for i in m.tasks()}
    return m, keys["choice"], keys["noul"]


def _stored(m, key):
    with m.hold(key) as e:
        e.store.flush()
        return [t for (t,) in e.store.db.execute("SELECT text FROM samples ORDER BY id")]


class _Teacher:
    name = "t"

    def classify(self, texts, task):
        from jevstiller.teachers import TeacherOutput
        return [TeacherOutput(task.labels[0], {c: float(c == task.labels[0]) for c in task.labels}, 1.0)
                for _ in texts]


# 1. stored text was cut at 32,768 characters (~128 KiB in 4-byte characters), with no limit on the volume
def test_stored_text_is_cut_in_bytes(tmp_path):
    from jevstiller._task import MAX_STORED_TEXT_BYTES
    js = Jevstiller(Task("t", "Which team?", LABELS), _Teacher(), tmp_path, encoder=HashEncoder(dim=32),
                    config=Config(training="manual"))
    big = "ok " + "\U0001F600" * 32_768
    js.classify(big)
    js.store.flush()
    (text,) = js.store.db.execute("SELECT text FROM samples").fetchone()
    assert len(text.encode()) <= MAX_STORED_TEXT_BYTES and big.startswith(text) and len(text) > 2_000
    js.close()


def test_rows_keep_no_text_when_the_volume_is_nearly_full(tmp_path, monkeypatch):
    js = Jevstiller(Task("t", "Which team?", LABELS), _Teacher(), tmp_path, encoder=HashEncoder(dim=32),
                    config=Config(training="manual", min_free_disk_mb=1024))
    monkeypatch.setattr(_core, "free_bytes", lambda path: 100 * 2**20)        # 100 MB free
    js.classify("a short text")
    monkeypatch.setattr(_core, "free_bytes", lambda path: 50 * 2**30)
    js.classify("another text")
    js.store.flush()
    assert [t for (t,) in js.store.db.execute("SELECT text FROM samples ORDER BY id")] == ["", "another text"]
    js.close()
    with pytest.raises(ValueError, match="min_free_disk_mb"):
        Config(min_free_disk_mb=float("nan"))


# 2. a confidence floor on a yes/no task was saved before it was rejected: the task stopped answering, and after a
# restart it dropped out of the index, so text retention and tenant deletion skipped its stored text
def test_an_invalid_change_is_rejected_before_anything_is_saved(tmp_path):
    m, choice, noul = _with_tasks(tmp_path)
    before = (tmp_path / "tasks" / noul / "task.json").read_text()
    for loaded in (False, True):
        if loaded:
            m.engine(noul)
        with pytest.raises(ValueError, match="no confidence_floor"):
            m.set_confidence_floor(noul, 0.6)
        assert (tmp_path / "tasks" / noul / "task.json").read_text() == before
        assert next(i for i in m.tasks() if i.key == noul).confidence_floor is None
        assert m.resolve("acme", YES_NO, None, kind="noul")[0] == noul           # still a usable task
    m.set_confidence_floor(choice, 0.6)                                       # a choice task takes one
    m.close()
    m = _manager(tmp_path)
    assert {i.key for i in m.tasks()} == {choice, noul}
    m.close()


def test_a_floor_override_for_a_yes_no_task_stops_startup(tmp_path):
    m, _, noul = _with_tasks(tmp_path)
    m.close()
    with pytest.raises(ValueError, match="no confidence_floor"):
        _manager(tmp_path, task_overrides={noul: {"confidence_floor": 0.8}})
    m = _manager(tmp_path)
    assert next(i for i in m.tasks() if i.key == noul).confidence_floor is None
    m.close()


def test_retention_and_tenant_deletion_cover_tasks_that_cannot_load(tmp_path):
    m, choice, _ = _with_tasks(tmp_path, text_retention_s=1)
    assert _stored(m, choice) == ["w1 w2"]
    m.close()
    f = tmp_path / "tasks" / choice / "task.json"
    info = json.loads(f.read_text())
    info["classes"] = {"only one": None}                    # no longer a valid task
    f.write_text(json.dumps(info))
    time.sleep(1.1)
    m = _manager(tmp_path, text_retention_s=1)
    assert choice not in {i.key for i in m.tasks()}
    assert m.apply_retention() >= 1
    from jevstiller._store import SampleStore
    s = SampleStore(tmp_path / "tasks" / choice / "samples.sqlite")
    assert [t for (t,) in s.db.execute("SELECT text FROM samples")] == [""]
    s.close()
    assert choice in m.delete_tenant("acme")
    assert not (tmp_path / "tasks" / choice).exists()
    m.close()


# 4. one shared admission table: a caller sending new questions flushed every tenant's counts
def test_one_caller_cannot_flush_other_callers_admission_counts():
    a = Admission(min_requests=3, max_tracked=1_000, max_per_caller=100)
    assert not a.observe("victim", caller="b")
    for i in range(10_000):                                 # the flood: always-new questions
        a.observe(f"junk {i}", caller="a")
    assert not a.observe("victim", caller="b")
    assert a.observe("victim", caller="b")                  # counted 3 times despite the flood
    assert len(a._seen) <= 100 + 1 and set(a._by_caller) == {"a"}
    a.forget("junk 9999")
    assert "junk 9999" not in a._owner


# 5. a change racing an engine load was saved but never reached the engine
def test_a_change_during_a_load_reaches_the_engine(tmp_path, monkeypatch):
    m, _, noul = _with_tasks(tmp_path)
    m.close()
    m = _manager(tmp_path)
    started = threading.Event()
    real = manager_module.Jevstiller

    class SlowLoad(real):
        def __init__(self, *a, **kw):
            started.set()
            time.sleep(0.3)                                 # the change arrives while the engine is built
            super().__init__(*a, **kw)
    monkeypatch.setattr(manager_module, "Jevstiller", SlowLoad)
    loader = threading.Thread(target=m.engine, args=(noul,))
    loader.start()
    started.wait(5)
    m.set_cutoffs(noul, 0.8)
    loader.join()
    with m.hold(noul) as e:
        assert e.task.cutoffs == (0.8,)
    m.close()


# 6. a change racing a delete wrote task.json into the deleted directory: the task came back after a restart
def test_a_change_racing_a_delete_cannot_bring_the_task_back(tmp_path, monkeypatch):
    """The race as it happened: the change has read the task, the delete removes its files, the change writes
    task.json into the directory, and the delete's last rmdir fails (ignored), leaving the task to come back."""
    import os
    m, choice, _ = _with_tasks(tmp_path)
    writing, emptied, wrote = threading.Event(), threading.Event(), threading.Event()
    real_write, real_rmtree = m._write_info, manager_module.shutil.rmtree

    def slow_write(info):
        writing.set()
        emptied.wait(0.5)
        real_write(info)
        wrote.set()

    def rmtree(path, ignore_errors=False):                  # shutil.rmtree's order: files, then directories
        if not str(path).startswith(str(tmp_path)):
            return real_rmtree(path, ignore_errors=ignore_errors)
        dirs = []
        for root, _, files in os.walk(path, topdown=False):
            for f in files:
                os.unlink(os.path.join(root, f))
            dirs.append(root)
        emptied.set()
        wrote.wait(0.5)
        for d in dirs:
            try:
                os.rmdir(d)
            except OSError:
                pass
    monkeypatch.setattr(m, "_write_info", slow_write)
    monkeypatch.setattr(manager_module.shutil, "rmtree", rmtree)
    change = threading.Thread(target=m.set_confidence_floor, args=(choice, 0.6))
    change.start()
    writing.wait(5)
    assert m.delete(choice)
    change.join()
    monkeypatch.setattr(m, "_write_info", real_write)
    assert not (tmp_path / "tasks" / choice).exists()
    m._dirty.add(choice)
    m._written.clear()
    m.sweep()                                               # the janitor's last_seen write doesn't either
    assert not (tmp_path / "tasks" / choice).exists()
    m.close()
    m = _manager(tmp_path)
    assert choice not in {i.key for i in m.tasks()}
    m.close()


def test_admin_changes_to_a_deleted_task_are_404(tmp_path, monkeypatch):
    from test_admin import TOKEN
    m, choice, _ = _with_tasks(tmp_path)
    monkeypatch.setattr(m, "set_confidence_floor", lambda *a: (_ for _ in ()).throw(KeyError(choice)))
    with serve(Upstream().app()) as upstream:
        settings = ProxySettings(upstream=upstream)
        with serve(create_app(m, settings, KeyRegistry(b"s", 60), admin_token=TOKEN)) as proxy:
            r = httpx.post(f"{proxy}/jevstiller/v1/tasks/{choice}/floor", json={"confidence_floor": 0.6},
                           headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 404
    m.close()


# 7. `jevstiller admin --url X` sent the local server's admin token to X
def test_the_cli_sends_the_servers_token_only_to_the_server(tmp_path, monkeypatch):
    from jevstiller import _cli
    (tmp_path / "token").write_text("SERVER-ADMIN-TOKEN-0123456789abcdef")
    cfg = tmp_path / "jevstiller.toml"
    cfg.write_text(f'[server]\nadmin_token_file = "{tmp_path / "token"}"\n')
    monkeypatch.setenv("JEVSTILLER_CONFIG", str(cfg))
    for k in ("JEVSTILLER_ADMIN_TOKEN", "JEVSTILLER_ADMIN_URL"):
        monkeypatch.delenv(k, raising=False)
    sent = []
    monkeypatch.setattr(httpx, "request", lambda method, url, **kw: sent.append((url, kw["headers"])) or
                        httpx.Response(200, json=[]))
    with pytest.raises(SystemExit, match="token is required"):
        _cli.main(["admin", "--url", "http://elsewhere.example:8080", "tasks"])
    monkeypatch.setenv("JEVSTILLER_ADMIN_URL", "http://elsewhere.example:8080")
    with pytest.raises(SystemExit, match="token is required"):
        _cli.main(["admin", "tasks"])
    assert sent == []
    monkeypatch.delenv("JEVSTILLER_ADMIN_URL")
    _cli.main(["admin", "tasks"])                            # next to the server: its own token and port
    assert sent and sent[0][0].startswith("http://127.0.0.1:") and "SERVER-ADMIN-TOKEN" in sent[0][1]["Authorization"]


# also fixed: encodings, hop-by-hop headers and repeated headers on the forwarded path
def test_forwarded_requests_ask_for_encodings_the_proxy_can_decode(tmp_path):
    seen = {}

    async def handle(request: Request):
        seen.update(request.headers)
        r = Response(b'{"ok": true}', media_type="application/json")
        r.raw_headers += [(b"set-cookie", b"a=1"), (b"set-cookie", b"b=2"), (b"x-secret-hop", b"1"),
                          (b"connection", b"x-secret-hop")]
        return r
    m = _manager(tmp_path)
    with serve(Starlette(routes=[Route("/{p:path}", handle, methods=["GET"])])) as upstream:
        with serve(create_app(m, ProxySettings(upstream=upstream), KeyRegistry(b"s", 60))) as proxy:
            r = httpx.get(f"{proxy}/v1/other", headers={"Authorization": f"Bearer {GOOD}", "Accept-Encoding": "br",
                                                        "Connection": "x-drop-me", "x-drop-me": "1"})
    assert seen["accept-encoding"] == "gzip, deflate" and "x-drop-me" not in seen
    assert r.json() == {"ok": True}
    assert r.headers.get_list("set-cookie") == ["a=1", "b=2"] and "x-secret-hop" not in r.headers
    m.close()


def test_credentials_in_the_upstream_url_are_rejected():
    with pytest.raises(ValueError, match="credentials"):
        ServeSettings(upstream="https://gw:secret@api.example").validate()
    with pytest.raises(ValueError, match="http"):
        ServeSettings(upstream="api.example").validate()
    ServeSettings(upstream="http://127.0.0.1:9000/base").validate()


def test_spellings_of_the_same_yes_no_criteria_count_once(tmp_path):
    up = Upstream()
    m = _manager(tmp_path)
    aliases = {"a": {"type": "noul", "instructions": YES_NO}, "b": {"type": "noul", "instructions": YES_NO,
               "criteria": {}}, "c": {"type": "noul", "instructions": YES_NO, "criteria": {"true": None}},
               "d": {"type": "noul", "instructions": YES_NO, "criteria": {"false": None, "true": None}}}
    with serve(up.app()) as upstream:
        with serve(create_app(m, ProxySettings(upstream=upstream), KeyRegistry(b"s", 60))) as proxy:
            for _ in range(2):
                assert _post(proxy, _body(questions=aliases)).status_code == 200
    [info] = m.tasks()
    assert len(_stored(m, info.key)) == 2                    # one row per request, not four
    m.close()


def test_an_unknown_teacher_confidence_is_a_disagreement_under_a_floor():
    from jevstiller._calibrate import RoutingPolicy
    pol = RoutingPolicy.__new__(RoutingPolicy)
    object.__setattr__(pol, "confidence_floor", 0.6)
    wrong = _core._disagrees(np.array(["a", "a"]), np.array(["a", "a"]), np.array([np.nan, 0.9]), pol)
    assert wrong.tolist() == [True, False]


def test_config_and_manager_still_take_their_defaults(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=8), Config(training="manual"), janitor_interval_s=3600)
    assert m.admission.max_per_caller == 10_000
    m.close()
