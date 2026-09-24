"""P6.7: `jevstiller serve` in front of the real Jev, from cold start to local answers.

    python experiments/live_proxy.py [--requests 4000] [--concurrency 8] [--encoder small]

Real Banking77 messages go through the unmodified TypeSafe SDK -> the proxy (a real `jevstiller serve` process)
-> the real Jev, with one 6-class routing question. Reports when the first local answer came, the local share
over time, the task's audit agreement, local vs forwarded latency, and the Jev cost. Needs TYPESAFE_API_KEY
(.env is read). Costs cents.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

QUESTION = {
    "instructions": "Which team should handle this banking customer's message?",
    "criteria": {
        "cards": "Card delivery, activation, PIN, card payments, lost, stolen or declined cards",
        "transfers": "Sending or receiving transfers, top-ups, pending or failed transfers, beneficiaries",
        "cash": "ATM withdrawals, cash, exchange rates, currencies",
        "account": "Identity verification, account settings, personal details, closing the account, passcode",
        "fees": "Fees, charges, refunds, disputed or wrong amounts",
        "other": "Anything else",
    },
}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--encoder", default="small")
    ap.add_argument("--admit-after", type=int, default=50)
    a = ap.parse_args()

    from datasets import banking77
    from jev_profile import load_env
    load_env()
    key = os.environ["TYPESAFE_API_KEY"]
    import typesafe_sdk as ts

    rows = [t for t, _ in banking77()["rows"]]
    random.Random(0).shuffle(rows)
    texts = (rows * (a.requests // len(rows) + 1))[:a.requests]

    tmp = Path(tempfile.mkdtemp(prefix="jvs-live-"))
    port = free_port()
    env = {**os.environ, "JEVSTILLER_LOG_FORMAT": "json", "JEVSTILLER_LOG_LEVEL": "warning"}
    admin_token = "live-e2e-admin-token-0123456789"
    (tmp / "admin").write_text(admin_token)
    server = subprocess.Popen([str(ROOT / ".venv/bin/jevstiller"), "serve", "--port", str(port),
                               "--data-dir", str(tmp / "data"), "--encoder", a.encoder,
                               "--admit-after", str(a.admit_after), "--admin-token-file", str(tmp / "admin")],
                              env=env, stderr=open(tmp / "server.log", "w"))
    base = f"http://127.0.0.1:{port}"
    import httpx
    for _ in range(600):
        try:
            if httpx.get(f"{base}/readyz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    print(f"proxy ready on {base} (encoder {a.encoder}); sending {a.requests:,} messages at concurrency "
          f"{a.concurrency}", flush=True)

    local = threading.local()
    results: list[tuple[int, str, float, int]] = []          # (index, source, latency ms, status)
    lock = threading.Lock()
    first_local = [None]
    q = {"route": ts.Choice(instructions=QUESTION["instructions"], criteria=QUESTION["criteria"])}

    def one(i: int) -> None:
        c = getattr(local, "c", None)
        if c is None:
            c = local.c = ts.TypeSafeClient(base_url=base, api_key=key, retry=ts.RetryPolicy(max_retries=3))
        t0 = time.perf_counter()
        try:
            r = c.system_one(state=texts[i], questions=q)
            src, status = r.raw_http_response.headers.get("x-jevstiller-source", "?"), 200
        except ts.TypeSafeAPIError as e:
            src, status = "error", e.status
        ms = (time.perf_counter() - t0) * 1000
        with lock:
            results.append((i, src, ms, status))
            if src == "local" and first_local[0] is None:
                first_local[0] = i
    t_start = time.time()
    try:
        with ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(one, range(a.requests)))
        wall = time.time() - t_start
        h = {"Authorization": f"Bearer {admin_token}"}
        tasks = httpx.get(f"{base}/jevstiller/v1/tasks", headers=h).json()
        status = httpx.get(f"{base}/jevstiller/v1/tasks/{tasks[0]['key']}", headers=h, timeout=60).json() \
            if tasks else {}
    finally:
        server.terminate()
        server.wait(60)

    results.sort()
    windows = []
    for s in range(0, len(results), 500):
        w = results[s:s + 500]
        windows.append((s + len(w), sum(r[1] == "local" for r in w) / len(w)))
    lat = {k: sorted(r[2] for r in results if r[1] == k) for k in ("local", "upstream")}

    def q_(v, p):
        return round(v[min(len(v) - 1, int(p * len(v)))], 1) if v else None
    st = status.get("status", {})
    out = {
        "date": time.strftime("%Y-%m-%d"), "requests": len(results), "wall_s": round(wall, 1),
        "errors": {s: sum(1 for r in results if r[3] == s) for s in {r[3] for r in results} if s != 200},
        "first_local_answer_at": first_local[0], "local_share_by_500": windows,
        "local_share_total": sum(r[1] == "local" for r in results) / len(results),
        "latency_ms": {k: {"p50": q_(v, .5), "p90": q_(v, .9), "p99": q_(v, .99), "n": len(v)} for k, v in lat.items()},
        "task": {k: st.get(k) for k in ("production", "mode", "audit_n", "audit_agreement", "audit_agreement_lb",
                                        "audit_agreement_ub", "teacher_calls", "teacher_calls_avoided",
                                        "teacher_cost_usd", "teacher_model")},
        "report": status.get("report"),
    }
    dst = ROOT / "experiments" / "results" / "live-proxy.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "report"}, indent=2))
    print(out["report"])


if __name__ == "__main__":
    main()
