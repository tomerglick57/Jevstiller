"""`GET /jevstiller/status`: a read-only HTML page for operators (P5.7). What each task is doing, how much is answered
locally, agreement with Jev against the target, and recent events.

It reads only what is already in memory: tasks that aren't loaded show their registration and last use, and are not
loaded to draw the page. Everything that comes from callers (questions, tenants, class names, event details) is
HTML-escaped; the page has no scripts, and its Content-Security-Policy allows none.
"""
from __future__ import annotations

import html
import secrets
import time
from typing import Any

from ._task import canonical_json

MAX_ROWS = 200           # tasks shown, most recently used first
MAX_EVENTS = 30
REFRESH_S = 30


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _short(x: Any, n: int = 120) -> str:
    s = x if isinstance(x, str) else canonical_json(x)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _ago(ts: float, now: float) -> str:
    d = max(0.0, now - ts)
    for unit, sec in (("d", 86400), ("h", 3600), ("min", 60)):
        if d >= sec:
            return f"{int(d // sec)} {unit} ago"
    return f"{int(d)} s ago"


def _badge(text: str, kind: str) -> str:
    return f'<span class="b {kind}">{_e(text)}</span>'


def _state(info, st) -> str:
    """What the task is doing, as a badge."""
    if st is None:
        return _badge("not loaded", "grey")
    if info.mode == "teacher_only":
        return _badge("Jev only (operator)", "grey")
    if st.production is None:
        return _badge("learning", "blue")
    if st.mode == "teacher_only":
        return _badge("fallback to Jev", "red")
    if st.shadow:
        return _badge("answering; candidate in shadow", "green")
    return _badge("answering locally", "green")


def _agreement(st) -> str:
    if st is None or st.audit_agreement is None:
        return '<span class="dim">–</span>'
    a, lb, ub, t = st.audit_agreement, st.audit_agreement_lb, st.audit_agreement_ub, st.target_agreement
    verdict = ("OK", "green") if lb >= t else ("inconclusive", "amber") if ub >= t else ("BROKEN", "red")
    return (f"{a:.2%} {_badge(verdict[0], verdict[1])}<br><span class=dim>[{lb:.2%}, {ub:.2%}] n={st.audit_n:,}"
            f" · target {t:.0%}</span>")


def _progress(st) -> str:
    if st is None:
        return ""
    r = st.readiness
    if st.production is None and r:
        return (f"<br><span class=dim>train {r['train'][0]:,}/{r['train'][1]:,} · "
                f"calib {r['calib'][0]:,}/{r['calib'][1]:,}</span>")
    return f"<br><span class=dim>{_e(st.production)}</span>"


def render(manager, proxy_stats: dict | None, version: str, now: float | None = None,
           max_rows: int = MAX_ROWS) -> tuple[str, str]:
    """(html, nonce): the page, and the nonce its one <style> element carries (for the CSP header)."""
    now = time.time() if now is None else now
    nonce = secrets.token_urlsafe(16)
    infos = sorted(manager.tasks(), key=lambda i: i.last_seen, reverse=True)
    rows, events = [], []
    for info in infos[:max_rows]:
        st = None
        try:
            with manager._hold_loaded(info.key) as e:      # never loads a task, never counts as use
                if e is not None:
                    st = e.status()
        except Exception:                                   # deleted or closing meanwhile: show it as not loaded
            st = None
        if st is not None:
            events += [(ev.get("ts", 0), info.key, ev) for ev in st.events[-5:]]
        rows.append(
            "<tr>"
            f'<td><code title="{_e(info.key)}">{_e(info.key[:12])}</code>'
            f"<br><span class=dim>{_e(info.tenant)}</span></td>"
            f"<td>{_e(_short(info.instructions))}<br><span class=dim>{len(info.classes)} classes"
            f"{' · ' + _e(info.model) if info.model else ''}</span></td>"
            f"<td>{_state(info, st)}{_progress(st)}</td>"
            f"<td class=n>{f'{st.student_share:.1%}' if st is not None and st.requests else '–'}</td>"
            f"<td>{_agreement(st)}</td>"
            f"<td class=n>{f'{st.requests:,}' if st is not None else '–'}</td>"
            f"<td class=n>{_e(_ago(info.last_seen, now))}</td>"
            "</tr>")

    p = proxy_stats or {}
    local, fwd = p.get("local", 0), p.get("forwarded", 0)
    share = f"{local / (local + fwd):.1%}" if local + fwd else "–"
    sched = getattr(manager.train_executor, "stats", None)
    train = sched() if callable(sched) else None
    cards = [
        ("Answered locally", share, f"{local:,} local · {fwd:,} forwarded, since start"),
        ("Tasks", f"{len(infos):,}", f"{len(manager.loaded()):,} loaded (max {manager.max_loaded})"),
        ("Training", f"{train['running']} running" if train else "–",
         f"{train['queued']} queued · {train['failed']} failed" if train else ""),
        ("Student memory", f"{manager.memory_mb():.0f} MB", ""),
    ]
    card_html = "".join(f"<div class=card><div class=dim>{_e(t)}</div><div class=big>{_e(v)}</div>"
                        f"<div class=dim>{_e(s)}</div></div>" for t, v, s in cards)
    more = (f"<p class=dim>Showing the {max_rows} most recently used of {len(infos):,} tasks.</p>"
            if len(infos) > max_rows else "")
    events.sort(key=lambda x: x[0], reverse=True)
    def detail(ev: dict) -> str:
        return _short(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in ev.items() if k not in ("ts", "kind")), 160)
    ev_rows = "".join(
        f"<tr><td class=n>{_e(_ago(ts, now))}</td><td><code>{_e(key[:12])}</code></td>"
        f"<td>{_e(ev.get('kind', ''))}</td><td class=dim>{_e(detail(ev))}</td></tr>"
        for ts, key, ev in events[:MAX_EVENTS]) or "<tr><td colspan=4 class=dim>No events from loaded tasks.</td></tr>"
    table = "".join(rows) or '<tr><td colspan=7 class=dim>No tasks yet.</td></tr>'
    page = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<meta http-equiv=refresh content="{REFRESH_S}"><meta name=robots content="noindex">
<title>Jevstiller status</title>
<style nonce="{nonce}">
:root{{--bg:#fff;--fg:#1b1f23;--dim:#6a737d;--line:#e1e4e8;--card:#f6f8fa;--green:#1a7f37;--amber:#9a6700;
--red:#cf222e;--blue:#0969da;--grey:#6a737d}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--line:#30363d;--card:#161b22;
--green:#3fb950;--amber:#d29922;--red:#f85149;--blue:#58a6ff;--grey:#8b949e}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1200px;margin:0 auto}} h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
.dim{{color:var(--dim)}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;
margin-top:16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}}
.big{{font-size:22px;font-weight:600}} .wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid var(--line)}}
th{{font-weight:600;color:var(--dim);white-space:nowrap}} td.n{{text-align:right;white-space:nowrap}}
code{{font:12px ui-monospace,monospace}} .b{{display:inline-block;border:1px solid;border-radius:10px;
padding:0 7px;font-size:12px;white-space:nowrap}} .green{{color:var(--green)}} .amber{{color:var(--amber)}}
.red{{color:var(--red)}} .blue{{color:var(--blue)}} .grey{{color:var(--grey)}}
</style></head><body><main>
<h1>Jevstiller status</h1>
<div class=dim>version {_e(version)} · {_e(time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(now)))} ·
refreshes every {REFRESH_S} s · agreement is with Jev, not accuracy</div>
<div class=cards>{card_html}</div>
<h2>Tasks</h2>{more}
<div class=wrap><table><thead><tr><th>Task / tenant</th><th>Question</th><th>State</th><th>Local</th>
<th>Agreement with Jev (audit)</th><th>Requests</th><th>Last used</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>Recent events</h2>
<div class=wrap><table><tbody>{ev_rows}</tbody></table></div>
<p class=dim>Read-only. Change things with <code>jevstiller admin</code> (docs: operations).</p>
</main></body></html>
"""
    return page, nonce
