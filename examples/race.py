"""The race: the same customer messages classified one at a time, straight to Jev and through Jevstiller.

    export TYPESAFE_BASE_URL=http://localhost:8080          # a running proxy with a trained student
    python examples/race.py run [--steps 200] [--warmup 5000]
    python examples/race.py replay                           # animate the two recorded timelines on one clock

A chain of dependent decisions (an agent loop, a game, classify-then-act) waits for every answer before the
next request, so per-answer latency is the whole story. `run` records two timelines to
experiments/results/race.json: lane one sends `--steps` messages to Jev, one at a time; lane two sends the same
messages to the proxy, one at a time. They run one after the other so they never compete for the machine or
the key. `--warmup N` first sends N messages through the proxy (the same pass as proxy_demo.py) so a student
exists; skip it if the proxy is already trained. `--threads` and `--seconds` instead run both lanes as a
throughput race with concurrent callers. Needs TYPESAFE_API_KEY. With JEVSTILLER_ADMIN_TOKEN set, the
proxy's audit agreement is recorded too. Costs a few cents (about 10 with the warm-up).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typesafe_sdk as ts  # noqa: E402
from proxy_demo import ROUTE, messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "race.json"


def lane(base_url: str, texts: list[str], seconds: float, threads: int,
         steps: int = 0) -> list[tuple[float, str, float]]:
    """Call `base_url` from `threads` callers for `seconds`, or `steps` requests one at a time (threads=1).
    Returns (t, source, latency ms) per success."""
    key = os.environ["TYPESAFE_API_KEY"]
    local = threading.local()
    lock = threading.Lock()
    done: list[tuple[float, str, float]] = []
    errors = [0]
    stop = time.perf_counter() + (seconds if not steps else 1e9)
    if steps:
        threads = 1
    t0 = time.perf_counter()
    n = [0]

    def worker() -> None:
        c = getattr(local, "c", None)
        if c is None:
            c = local.c = ts.TypeSafeClient(base_url=base_url, api_key=key, retry=ts.RetryPolicy(max_retries=8))
        while time.perf_counter() < stop:
            with lock:
                i = n[0]
                n[0] += 1
            if steps and i >= steps:
                break
            t1 = time.perf_counter()
            try:
                r = c.system_one(state=texts[i % len(texts)], questions={"route": ROUTE})
                src = r.raw_http_response.headers.get("x-jevstiller-source", "jev")
            except ts.TypeSafeAPIError:
                errors[0] += 1
                continue
            t2 = time.perf_counter()
            with lock:
                done.append((round(t2 - t0, 3), "local" if src == "local" else "jev", round((t2 - t1) * 1000, 1)))

    with ThreadPoolExecutor(threads) as ex:
        for _ in range(threads):
            ex.submit(worker)
    took = done[-1][0] if done else 0.0
    print(f"  {len(done):,} answered in {took if steps else seconds:.1f} s, {errors[0]} errors", flush=True)
    return done


def audit(base: str) -> dict:
    tok = os.environ.get("JEVSTILLER_ADMIN_TOKEN")
    if not tok:
        return {}
    try:
        h = {"Authorization": f"Bearer {tok}"}
        req = urllib.request.Request(f"{base}/jevstiller/v1/tasks", headers=h)
        tasks = json.load(urllib.request.urlopen(req, timeout=10))
        tasks = tasks.get("tasks", tasks) if isinstance(tasks, dict) else tasks
        if not tasks:
            return {}
        req = urllib.request.Request(f"{base}/jevstiller/v1/tasks/{tasks[0]['key']}", headers=h)
        st = json.load(urllib.request.urlopen(req, timeout=60)).get("status", {})
        return {k: st.get(k) for k in ("audit_n", "audit_agreement", "audit_agreement_lb", "audit_agreement_ub",
                                       "target_agreement", "production", "student_share")}
    except Exception as e:  # the race still stands without it
        print(f"  (no audit numbers: {e})", flush=True)
        return {}


def run(a: argparse.Namespace) -> None:
    if "TYPESAFE_API_KEY" not in os.environ and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip()
    proxy = os.environ.get("TYPESAFE_BASE_URL")
    if not proxy:
        sys.exit("set TYPESAFE_BASE_URL to the proxy")
    texts = messages(20_000)
    if a.warmup:
        print(f"warming the proxy with {a.warmup:,} messages ...", flush=True)
        os.environ["TYPESAFE_BASE_URL"] = proxy
        import proxy_demo
        sys.argv = ["proxy_demo.py", "--requests", str(a.warmup), "--every", "1000"]
        proxy_demo.main()
    how = f"{a.steps} steps, one at a time" if a.steps else f"{a.threads} callers, {a.seconds:.0f} s"
    print(f"lane 1: Jev direct, {how}", flush=True)
    direct = lane(ts.constants.DEFAULT_BASE_URL, texts, a.seconds, a.threads, a.steps)
    time.sleep(3)
    print(f"lane 2: via Jevstiller ({proxy}), {how}", flush=True)
    via = lane(proxy, texts, a.seconds, a.threads, a.steps)
    out = {"date": time.strftime("%Y-%m-%d"), "seconds": a.seconds, "threads": a.threads, "steps": a.steps,
           "direct": direct, "via": via, "audit": audit(proxy)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}")


BAR = 36


def replay(a: argparse.Namespace) -> None:
    r = json.loads(Path(a.file).read_text())
    lanes = {"direct": r["direct"], "via": r["via"]}
    steps = r.get("steps", 0)
    ends = {k: (v[-1][0] if v else 0.0) for k, v in lanes.items()}
    seconds = max(ends.values()) if steps else r["seconds"]
    total = steps or max(len(r["direct"]), len(r["via"])) or 1
    p50 = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0.0  # noqa: E731
    au = r.get("audit") or {}
    if steps:
        print(f"{steps} decisions, one at a time, each waiting for the last: same question, same messages, "
              "same API key\n")
    else:
        print(f"same question, same customer messages, same API key, {r['threads']} concurrent callers, "
              f"{seconds:.0f} s each\n")
    print("\n" * 4, end="")
    step = 0.1
    t_wall = time.perf_counter()
    frames = [round(i * step, 3) for i in range(int(seconds / step) + 1)] + [seconds]
    for t in frames:
        lines = []
        for name, label in (("direct", "Jev direct    "), ("via", "via Jevstiller")):
            ev = lanes[name]
            k = sum(1 for e in ev if e[0] <= t)
            recent = [e for e in ev if e[0] <= t] if steps else [e for e in ev if t - 2 <= e[0] <= t]
            rate = len(recent) / 2 if t >= 2 else (len(recent) / t if t > 0 else 0)
            fill = int(BAR * k / total)
            lat = p50([e[2] for e in recent]) if recent else 0
            loc = sum(e[1] == "local" for e in recent)
            share = f"  {loc / len(recent):3.0%} local" if name == "via" and recent else ""
            if steps:
                fin = f"  done in {ends[name]:.1f} s" if k >= steps else f"  p50 {lat:4.0f} ms{share}"
                lines.append(f"{label}  {'█' * fill}{'░' * (BAR - fill)}  {k:>3}/{steps}{fin}\033[K")
            else:
                lines.append(f"{label}  {'█' * fill}{'░' * (BAR - fill)}  {k:>6,}  {rate:4.0f}/s"
                             f"  p50 {lat:4.0f} ms{share}\033[K")
        tail = f"t = {t:4.1f} s"
        if au.get("audit_agreement") is not None and t >= seconds:
            lb, ub = au["audit_agreement_lb"], au["audit_agreement_ub"]
            tail += (f"      audit: agreement with Jev {au['audit_agreement']:.1%} [{lb:.1%}, {ub:.1%}]"
                     f"  target {au['target_agreement']:.0%}")
        sys.stdout.write("\033[4A" + "\n".join(lines) + "\n\n" + tail + "\033[K\n")
        sys.stdout.flush()
        if not a.fast:
            time.sleep(max(0.0, t_wall + t / a.speed - time.perf_counter()))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--steps", type=int, default=200, help="sequential decisions per lane (0: throughput race)")
    p.add_argument("--warmup", type=int, default=0, help="messages to send through the proxy first")
    p = sub.add_parser("replay")
    p.add_argument("--file", default=str(OUT))
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--fast", action="store_true", help="no pacing (for tests)")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else replay(a)


if __name__ == "__main__":
    main()
