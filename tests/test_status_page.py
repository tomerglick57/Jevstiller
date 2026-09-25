"""P5.7: the read-only HTML status page at /jevstiller/status."""
import re

import httpx
from test_admin import TOKEN, stack  # noqa: F401  (fixture)
from test_audit_fixes import _body, _post
from test_proxy import QUESTION, serve

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller._status_page import render
from jevstiller.server import ProxySettings, create_app


def _page(proxy, auth=("anyone", TOKEN), **kw):
    return httpx.get(f"{proxy}/jevstiller/status", auth=auth, timeout=60, **kw)


def test_the_page_needs_the_admin_token_and_is_never_forwarded(stack):  # noqa: F811
    proxy, up, _, _ = stack
    r = httpx.get(f"{proxy}/jevstiller/status")
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic ")   # the browser asks
    assert _page(proxy, auth=("anyone", "wrong")).status_code == 401
    assert httpx.get(f"{proxy}/jevstiller/status", headers={"Authorization": "Basic !!!"}).status_code == 401
    assert _page(proxy).status_code == 200                                                # any user name
    assert httpx.get(f"{proxy}/jevstiller/status", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert _page(proxy, auth=None, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert httpx.post(f"{proxy}/jevstiller/status", auth=("a", TOKEN)).status_code == 405
    # Basic auth is for the page only: the JSON admin API still wants the bearer token
    assert httpx.get(f"{proxy}/jevstiller/v1/tasks", auth=("a", TOKEN)).status_code == 401
    assert up.requests == []


def test_off_without_an_admin_token(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), janitor_interval_s=3600)
    with serve(create_app(m, ProxySettings(upstream="http://127.0.0.1:9"))) as proxy:
        assert _page(proxy).status_code == 404
    m.close()


def test_the_page_shows_tasks_and_is_locked_down(stack, world):  # noqa: F811
    proxy, _, m, _ = stack
    for t, _ in world.sample(1500):                       # enough for a first student (inline training)
        _post(proxy, _body(state=t))
    r = _page(proxy)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    key = m.tasks()[0].key
    assert key[:12] in page and "Which team handles this?" in page and "default" in page
    assert "answering locally" in page and "Agreement with Jev" in page and re.search(r"\d+\.\d+%", page)
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "script-src" not in csp and "frame-ancestors 'none'" in csp
    nonce = re.search(r"style-src 'nonce-([^']+)'", csp).group(1)
    assert f'<style nonce="{nonce}">' in page and "<script" not in page
    assert r.headers["x-frame-options"] == "DENY" and r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_caller_strings_are_escaped(tmp_path, world):
    """Questions, tenants and class names come from callers: nothing of theirs may become markup."""
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), admission=Admission(1),
                    janitor_interval_s=3600)
    evil = '<script>alert(1)</script>"><img src=x onerror=alert(2)>'
    r = m.route(evil, evil, {evil: "", "b": ""}, ["w1 w2"])
    m.complete(r, [RuntimeError("no answer")] * len(r.routed.to_teacher), "jev")
    page, _ = render(m, {"local": 1, "forwarded": 1}, "test")
    own = {"html", "head", "meta", "title", "style", "body", "main", "h1", "h2", "div", "p", "table", "thead",
           "tbody", "tr", "th", "td", "span", "code", "br"}
    assert set(re.findall(r"<([a-zA-Z0-9]+)", page)) <= own        # no markup but the page's own
    assert "&lt;script&gt;alert(1)&lt;/script&gt;&quot;&gt;&lt;img" in page
    m.close()


def test_drawing_the_page_loads_nothing_and_counts_as_no_use(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), admission=Admission(1),
                    max_loaded=1, janitor_interval_s=3600)
    for i in range(3):                                    # three tasks; only the last stays loaded
        q = {**QUESTION, "instructions": f"Question {i}?"}
        r = m.route("acme", q["instructions"], q["criteria"], ["w1 w2"])
        m.complete(r, [RuntimeError("no answer")] * len(r.routed.to_teacher), "jev")
    seen = {i.key: i.last_seen for i in m.tasks()}
    loads = m.loads
    page, _ = render(m, None, "test", max_rows=2)
    assert m.loads == loads and {i.key: i.last_seen for i in m.tasks()} == seen
    assert page.count("not loaded") == 1 and "Showing the 2 most recently used of 3 tasks" in page
    m.close()


def test_the_page_is_cached_briefly(stack, monkeypatch):  # noqa: F811
    proxy, _, _, _ = stack
    import jevstiller._status_page as sp
    calls = []
    real = sp.render
    monkeypatch.setattr(sp, "render", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for _ in range(5):
        assert _page(proxy).status_code == 200
    assert len(calls) == 1
