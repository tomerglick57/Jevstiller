"""A service that classifies customer messages with Jev, unchanged except for TYPESAFE_BASE_URL.

Run it with the proxy in front (`export TYPESAFE_BASE_URL=http://localhost:8080`) and watch the answers
move from Jev to the local model: the `x-jevstiller-source` header says who answered each request.

    python examples/proxy_demo.py [--requests 5000] [--threads 8] [--every 25]

Needs TYPESAFE_API_KEY (the proxy forwards it to Jev). Sends real Banking77 customer messages with one
6-class routing question; 5,000 requests cost about $0.10 of Jev calls.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import typesafe_sdk as ts

ROUTE = ts.Choice(
    instructions="Which team should handle this banking customer's message?",
    criteria={
        "cards": "Card delivery, activation, PIN, card payments, lost, stolen or declined cards",
        "transfers": "Sending or receiving transfers, top-ups, pending or failed transfers, beneficiaries",
        "cash": "ATM withdrawals, cash, exchange rates, currencies",
        "account": "Identity verification, account settings, personal details, closing the account, passcode",
        "fees": "Fees, charges, refunds, disputed or wrong amounts",
        "other": "Anything else",
    },
)


def messages(n: int) -> list[str]:
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
    rows = []
    for split in ("train", "test"):
        with urllib.request.urlopen(base + f"{split}.csv") as r:
            rows += [row["text"] for row in csv.DictReader(io.TextIOWrapper(r, encoding="utf-8"))]
    random.Random(0).shuffle(rows)
    return (rows * (n // len(rows) + 1))[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--every", type=int, default=25, help="print one request in this many")
    a = ap.parse_args()
    if "TYPESAFE_API_KEY" not in os.environ and os.path.exists(".env"):        # convenience for this example
        for line in open(".env"):
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip()
    base = os.environ.get("TYPESAFE_BASE_URL")
    if base:                                                                   # wait for a proxy that is still starting
        for _ in range(120):
            try:
                urllib.request.urlopen(base.rstrip("/") + "/readyz", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
    texts = messages(a.requests)
    print(f"{a.requests:,} customer messages  →  {os.environ.get('TYPESAFE_BASE_URL', 'https://api.typesafe.ai')}",
          flush=True)

    local = threading.local()
    lock = threading.Lock()
    done: list[tuple[str, float]] = []                       # (source, latency ms), in completion order
    errors = [0]

    def one(i: int) -> None:
        c = getattr(local, "c", None)
        if c is None:
            c = local.c = ts.TypeSafeClient(retry=ts.RetryPolicy(max_retries=5))
        t0 = time.perf_counter()
        try:
            r = c.system_one(state=texts[i], questions={"route": ROUTE})
            label, src = r.choices["route"].choice, r.raw_http_response.headers.get("x-jevstiller-source", "jev")
        except ts.TypeSafeAPIError as e:
            label, src = f"error {e.status}", "error"
            errors[0] += 1
        ms = (time.perf_counter() - t0) * 1000
        with lock:
            done.append((src, ms))
            k = len(done)
        if k % a.every == 0 or k <= 3:
            who = {"local": "local", "upstream": "jev", "jev": "jev"}.get(src, src)
            print(f"#{k:>5,}  {label:<10} {ms:5.0f} ms  {who:<5}  \"{texts[i][:52]}\"", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, range(a.requests)))
    wall = time.time() - t0
    tail = done[-500:]
    p50 = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
    print()
    local_ms = p50([m for s, m in tail if s == "local"])
    jev_ms = p50([m for s, m in done if s != "local"])
    print(f"{len(done):,} requests in {wall:.0f} s, {errors[0]} errors.  Last 500: "
          f"{sum(s == 'local' for s, _ in tail) / len(tail):.0%} answered locally"
          + (f", p50 {local_ms:.0f} ms vs {jev_ms:.0f} ms via Jev." if local_ms == local_ms else "."))


if __name__ == "__main__":
    main()
