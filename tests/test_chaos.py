"""P6.5: failures around the proxy. Jev down, slow or rate limiting; a full disk; kill -9 and restart."""
import asyncio
import contextlib
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller._server import KeyRegistry, ProxySettings, _no_cookies, create_app
from jevstiller._store import SampleStore

ts = pytest.importorskip("typesafe_sdk")
from test_proxy import GOOD, OTHER, FakeJev, ask, client, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ChaosJev(FakeJev):
    """FakeJev that can be slow, and rate-limit some keys (and only those)."""

    def __init__(self, world):
        super().__init__(world)
        self.delay_s = 0.0
        self.limited: set[str] = set()

    async def system_one(self, request: Request):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if request.headers.get("authorization", "").removeprefix("Bearer ") in self.limited:
            self.calls += 1
            return JSONResponse({"detail": "rate limited"}, 429, headers={"retry-after": "30"})
        return await super().system_one(request)


class Switch(httpx.AsyncBaseTransport):
    """The proxy's way to Jev, with an off switch: while `down`, connections are refused."""

    def __init__(self):
        self.inner = httpx.AsyncHTTPTransport()
        self.down = False

    async def handle_async_request(self, request):
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        return await self.inner.handle_async_request(request)

    async def aclose(self):
        await self.inner.aclose()


@contextlib.contextmanager
def trained_stack(tmp_path, world, **settings):
    """(proxy url, jev, manager, switch) with one question trained far enough to answer locally."""
    jev, sw = ChaosJev(world), Switch()
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=150, seed=0, training="inline")
    with serve(jev.app()) as upstream:
        s = ProxySettings(upstream=upstream, **settings)
        m = TaskManager(tmp_path, None, HashEncoder(dim=512), cfg, target_agreement=0.95,
                        admission=Admission(min_requests=1), janitor_interval_s=3600)
        up = httpx.AsyncClient(base_url=upstream, timeout=s.upstream_timeout_s, transport=sw, cookies=_no_cookies())
        app = create_app(m, s, KeyRegistry(b"salt", s.key_ttl_s), client=up, metrics_public=True)
        with serve(app) as proxy:
            c, local = client(proxy), 0
            for t, _ in world.sample(3000):
                local += ask(c, t).raw_http_response.headers["x-jevstiller-source"] == "local"
                if local >= 30:
                    break
            assert local >= 30, m.engine(m.tasks()[0].key).status().report()
            client(proxy, OTHER).models.list()           # OTHER: known to Jev, not yet verified here
            yield proxy, jev, m, sw
        m.close()


def _source(c, text):
    """'local', 'upstream', or the HTTP status of the error the SDK raised."""
    try:
        return ask(c, text).raw_http_response.headers["x-jevstiller-source"]
    except ts.TypeSafeAPIError as e:
        return e.status


def _labelled(m):
    e = m.engine(m.tasks()[0].key)
    return e.store.counts(e.task.version)["teacher_calls"]


def test_jev_outage_keeps_local_answers_then_recovers(tmp_path, world):
    with trained_stack(tmp_path, world) as (proxy, jev, m, sw):
        c = client(proxy)
        before = _labelled(m)
        sw.down = True
        seen = [_source(c, t) for t, _ in world.sample(300)]
        assert seen.count("local") > 150 and set(seen) == {"local", 502}, set(seen)
        assert _labelled(m) == before                    # nothing was recorded as a teacher answer
        assert sum(m._inflight.values()) == 0
        sw.down = False
        seen = [_source(c, t) for t, _ in world.sample(100)]
        assert set(seen) <= {"local", "upstream"} and "upstream" in seen
        assert _labelled(m) > before


def test_slow_jev_times_out_without_slowing_local_answers(tmp_path, world):
    with trained_stack(tmp_path, world, upstream_timeout_s=1.0) as (proxy, jev, m, sw):
        jev.delay_s = 3.0
        stop, statuses = threading.Event(), []

        def slow_caller():                               # an untrained question: always forwarded
            c = client(proxy)
            while not stop.is_set():
                try:
                    c.system_one(state="w1 w2", questions={"q": ts.Choice(instructions="Untrained?",
                                                                          criteria={"a": "", "b": ""})})
                except ts.TypeSafeAPIError as e:
                    statuses.append(e.status)
        threads = [threading.Thread(target=slow_caller) for _ in range(16)]
        for t in threads:
            t.start()
        try:
            time.sleep(0.3)
            c, lat = client(proxy), []
            for t, _ in world.sample(200):
                t0 = time.perf_counter()
                if _source(c, t) == "local":
                    lat.append(time.perf_counter() - t0)
        finally:
            stop.set()
            for t in threads:
                t.join(10)
        lat.sort()
        assert len(lat) > 100 and lat[int(0.95 * len(lat))] < 0.25, lat[-10:]
        assert statuses and set(statuses) == {504}


def test_rate_limit_storm_backs_off_per_key(tmp_path, world):
    with trained_stack(tmp_path, world) as (proxy, jev, m, sw):
        jev.limited = {GOOD}
        calls = jev.calls

        def untrained(key):
            try:
                client(proxy, key).system_one(state="w1", questions={
                    "q": ts.Choice(instructions="Untrained?", criteria={"a": "", "b": ""})})
                return 200
            except ts.TypeSafeAPIError as e:
                return e.status
        with ThreadPoolExecutor(8) as ex:
            storm = list(ex.map(untrained, [GOOD] * 100))
        assert set(storm) == {429}
        assert jev.calls - calls <= 8                    # after the first 429, the proxy answers the rest itself
        assert untrained(OTHER) == 200                   # another key is not affected
        seen = [_source(client(proxy), t) for t, _ in world.sample(100)]
        assert seen.count("local") > 50 and set(seen) <= {"local", 429}   # local answers go on for GOOD


def test_full_disk_drops_records_but_keeps_serving(tmp_path, world, monkeypatch):
    with trained_stack(tmp_path, world) as (proxy, jev, m, sw):
        e = m.engine(m.tasks()[0].key)
        e.store.flush()
        rows = e.store.counts(e.task.version)["total"]

        def full(self, recs):
            raise sqlite3.OperationalError("database or disk is full")
        with monkeypatch.context() as mp:
            mp.setattr(SampleStore, "_write", full)
            seen = [_source(client(proxy), t) for t, _ in world.sample(200)]
            e.store.flush()
        assert set(seen) <= {"local", "upstream"} and seen.count("local") > 100
        assert e.store.counts(e.task.version)["total"] == rows
        assert e.store.write_errors == 200
        metrics = httpx.get(f"{proxy}/metrics").text
        assert "jevstiller_store_dropped_records_total" in metrics
        dropped = [ln for ln in metrics.splitlines() if ln.startswith("jevstiller_store_dropped_records_total ")]
        assert float(dropped[0].split()[1]) >= 200
        for t, _ in world.sample(10):                    # space again: recording resumes
            ask(client(proxy), t)
        e.store.flush()
        assert e.store.counts(e.task.version)["total"] == rows + 10


# ---- kill -9 -----------------------------------------------------------------------------------------------
def _descendants(pid: int) -> set[int]:
    parents = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            with contextlib.suppress(OSError, IndexError):
                parents[int(d.name)] = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        kids = {c for c, pp in parents.items() if pp == p}
        out |= kids
        todo += kids
    return out


def _alive(pid: int) -> bool:
    try:
        return (Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]) != "Z"
    except OSError:
        return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_kill_9_during_training_and_restart(tmp_path, world):
    jev = ChaosJev(world)
    with serve(jev.app()) as upstream:
        cfgfile = tmp_path / "j.toml"
        cfgfile.write_text(f'''
[server]
data_dir = "{tmp_path}/data"
log_level = "warning"
[proxy]
upstream = "{upstream}"
[manager]
admit_after = 1
target_agreement = 0.95
[encoder]
spec = "hash"
[engine]
min_train_samples = 400
min_samples_per_class = 20
min_calib_samples = 150
shadow_min_samples = 150
min_new_samples = 600
''')
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        base = f"http://127.0.0.1:{port}"
        cmd = [str(Path(sys.executable).parent / "jevstiller"), "serve", "--config", str(cfgfile), "--port", str(port)]

        def start():
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(tmp_path / "server.log", "a"))
            for _ in range(300):
                with contextlib.suppress(httpx.HTTPError):
                    if httpx.get(f"{base}/readyz", timeout=1).status_code == 200:
                        return p
                time.sleep(0.1)
            raise AssertionError((tmp_path / "server.log").read_text()[-3000:])

        texts = [t for t, _ in world.sample(20000)]
        sent = [0]

        def traffic(stop):
            c = client(base)
            while not stop.is_set():
                with contextlib.suppress(ts.TypeSafeError, httpx.HTTPError):
                    ask(c, texts[sent[0] % len(texts)])
                sent[0] += 1

        orphans = set()
        for delay in (4.0, 2.5, 6.0):                    # kill at different points of the training cycle
            p = start()
            stop = threading.Event()
            th = threading.Thread(target=traffic, args=(stop,))
            th.start()
            time.sleep(delay)
            kids = _descendants(p.pid)
            os.kill(p.pid, signal.SIGKILL)
            p.wait()
            stop.set()
            th.join(15)
            deadline = time.time() + 10
            while time.time() < deadline and any(_alive(k) for k in kids):
                time.sleep(0.2)
            orphans |= {k for k in kids if _alive(k)}

        p = start()                                      # comes back after every crash ...
        try:
            c, local = client(base), 0
            for t in texts[:6000]:
                with contextlib.suppress(ts.TypeSafeError):
                    local += ask(c, t).raw_http_response.headers["x-jevstiller-source"] == "local"
                if local >= 20:
                    break
            assert local >= 20, (tmp_path / "server.log").read_text()[-3000:]   # ... and still learns
        finally:
            p.terminate()
            p.wait(30)
        for k in orphans:
            with contextlib.suppress(OSError):
                os.kill(k, signal.SIGKILL)
        assert not orphans, "training workers outlived a kill -9 of the server"
    tasks = list((tmp_path / "data" / "tasks").iterdir())
    assert tasks
    for d in tasks:
        db = sqlite3.connect(d / "samples.sqlite")
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        db.close()
        assert not list((d / "versions").glob(".staging-*"))      # half-built versions were cleaned up


class Flaky(httpx.AsyncBaseTransport):
    """The way to Jev, where the next `drops` requests find their pooled connection closed."""

    def __init__(self):
        self.inner, self.drops = httpx.AsyncHTTPTransport(), 0

    async def handle_async_request(self, request):
        if self.drops:
            self.drops -= 1
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        return await self.inner.handle_async_request(request)

    async def aclose(self):
        await self.inner.aclose()


def test_a_connection_closed_by_jev_is_retried_once(tmp_path, world):
    """A pooled connection the upstream just closed as idle (7 in 1.15M requests in the soak) costs no error."""
    jev, flaky = ChaosJev(world), Flaky()
    with serve(jev.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                        admission=Admission(min_requests=1), janitor_interval_s=3600)
        up = httpx.AsyncClient(base_url=upstream, transport=flaky, cookies=_no_cookies())
        app = create_app(m, ProxySettings(upstream=upstream), KeyRegistry(b"s", 3600), client=up)
        with serve(app) as proxy:
            flaky.drops = 1
            assert _source(client(proxy), "w1 w2") == "upstream"
            assert app.state.proxy.metrics.upstream.value("retried") == 1
            flaky.drops = 2                               # twice in a row: Jev really is unreachable
            assert _source(client(proxy), "w3 w4") == 502
        m.close()


def test_idle_connections_are_kept_longer_than_clients_reuse_them(tmp_path, monkeypatch):
    import uvicorn

    from jevstiller import _cli as cli
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    cli.main(["serve", "--data-dir", str(tmp_path), "--encoder", "hash", "--log-level", "warning"])
    assert calls[0]["timeout_keep_alive"] >= 60           # httpx reuses for 5 s, load balancers ~60 s


def test_a_caller_hanging_up_mid_request_is_not_an_error(tmp_path, world):
    import logging

    class Errors(logging.Handler):                        # uvicorn's loggers don't propagate to caplog's root
        def __init__(self):
            super().__init__(logging.ERROR)
            self.records = []

        def emit(self, record):
            self.records.append(record)
    jev = ChaosJev(world)
    with serve(jev.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), janitor_interval_s=3600)
        with serve(create_app(m, ProxySettings(upstream=upstream))) as proxy:
            errors = Errors()                             # after the servers start: uvicorn resets its loggers
            logging.getLogger("uvicorn.error").addHandler(errors)
            try:
                host, port = proxy.removeprefix("http://").split(":")
                s = socket.create_connection((host, int(port)))
                s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\n"
                          b"Content-Type: application/json\r\nContent-Length: 1000\r\n\r\n{\"state\": ")
                time.sleep(0.3)                           # the handler is now waiting for the rest
                s.close()                                 # gone before the body arrived
                time.sleep(0.5)
                assert not errors.records, [r.getMessage() for r in errors.records]
                assert _source(client(proxy), "w1") == "upstream"   # and the proxy carries on
            finally:
                logging.getLogger("uvicorn.error").removeHandler(errors)
        m.close()
