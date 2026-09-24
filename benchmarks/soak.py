"""P6.6: long-running soak with drift injected halfway.

    python benchmarks/soak.py --minutes 40          # the plan's full run: --minutes 1440
    python benchmarks/soak.py --minutes 5 --rate 50  # smoke

A real `jevstiller serve` process (hash encoder) sits in front of a fake Jev. Traffic: `--rate` requests/s over
20 questions (Zipf-distributed) and 3 API keys. Halfway through, the fake Jev silently changes its answers (it
swaps the labels around, keeping its model name), so every trained student starts disagreeing with it. Without
intervention the proxy should fall back, retrain on the new answers, and serve locally again.

Sampled every 10 s: the server's RSS, the memory its models hold, reloads, queued training, the local share and
errors. Writes experiments/results/soak.json and prints a summary: proxy 5xx, recovery after the drift, and the
growth of memory beyond what the models hold (RSS minus models: should be flat).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jevstiller import SyntheticTeacher, SyntheticWorld, Task  # noqa: E402

LABELS = ["billing", "technical", "cancellation", "sales", "other"]
KEYS = ["tsk_soak_a", "tsk_soak_b", "tsk_soak_c"]
ADMIN = "soak-admin-token-0123456789"


class DriftingJev:
    def __init__(self, latency_s: float):
        self.world = SyntheticWorld(LABELS, seed=1)
        self.teacher = SyntheticTeacher(self.world)
        self.latency_s = latency_s
        self.calls = 0
        self.perm = {c: c for c in LABELS}

    def drift(self) -> None:
        self.perm = dict(zip(LABELS, LABELS[1:] + LABELS[:1], strict=True))   # same difficulty, other answers

    async def systemone(self, request: Request):
        self.calls += 1
        await asyncio.sleep(self.latency_s * random.lognormvariate(0, 0.3))
        if request.headers.get("authorization", "").removeprefix("Bearer ") not in KEYS:
            return JSONResponse({"detail": "invalid API key"}, 401)
        body = await request.json()
        answers = {}
        for name, q in body["questions"].items():
            o = self.teacher.classify([body["state"]], Task("t", q.get("instructions"), list(q["criteria"])))[0]
            answers[name] = {"type": "choice", "choice": self.perm[o.label], "confidence": o.confidence,
                             "probabilities": {self.perm[c]: p for c, p in o.probs.items()}}
        return JSONResponse({"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 50, "output_tokens": 1}},
                            headers={"x-typesafe-request-id": f"req_{self.calls}"})


def serve_in_thread(app) -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
    while not server.started:
        time.sleep(0.01)
    return f"http://127.0.0.1:{sock.getsockname()[1]}"


def rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


async def traffic(base: str, rate: float, stop: asyncio.Event, stats: dict, world: SyntheticWorld) -> None:
    rng = random.Random(0)
    weights = [1 / (i + 1) for i in range(20)]
    questions = [{"type": "choice", "instructions": f"Which team handles this? (queue {i})",
                  "criteria": {c: "" for c in LABELS}} for i in range(20)]
    texts = [t for t, _ in world.sample(50_000)]
    sem = asyncio.Semaphore(64)
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=64)) as c:
        async def one():
            async with sem:
                q = rng.choices(questions, weights)[0]
                body = {"state": rng.choice(texts), "model": "jev-latest", "questions": {"label": q}}
                try:
                    r = await c.post(f"{base}/v1/systemone", json=body,
                                     headers={"Authorization": f"Bearer {rng.choice(KEYS)}"})
                    key = r.headers.get("x-jevstiller-source", "?") if r.status_code == 200 else f"http_{r.status_code}"
                except httpx.HTTPError as e:
                    key = f"exc_{type(e).__name__}"
                stats[key] = stats.get(key, 0) + 1
        tasks: set[asyncio.Task] = set()
        interval = 1 / rate
        nxt = time.perf_counter()
        while not stop.is_set():
            t = asyncio.create_task(one())
            tasks.add(t)
            t.add_done_callback(tasks.discard)
            nxt += interval
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
        await asyncio.gather(*tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--rate", type=float, default=100)
    ap.add_argument("--latency-ms", type=float, default=50)
    ap.add_argument("--max-loaded", type=int, default=12, help="below 20 (the questions), tasks load and unload")
    ap.add_argument("--no-drift", action="store_true")
    a = ap.parse_args()
    duration = a.minutes * 60

    jev = DriftingJev(a.latency_ms / 1000)
    up = serve_in_thread(Starlette(routes=[Route("/v1/systemone", jev.systemone, methods=["POST"])]))
    tmp = Path(tempfile.mkdtemp(prefix="jvs-soak-"))
    (tmp / "admin").write_text(ADMIN)
    (tmp / "j.toml").write_text(f'''
[server]
data_dir = "{tmp}/data"
log_level = "warning"
log_format = "json"
admin_token_file = "{tmp}/admin"
[proxy]
upstream = "{up}"
[manager]
admit_after = 10
target_agreement = 0.95
max_loaded = {a.max_loaded}
[encoder]
spec = "hash"
[engine]
min_train_samples = 400
min_samples_per_class = 20
min_calib_samples = 150
shadow_min_samples = 300
min_new_samples = 3000
drift_min_samples = 60
''')
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    wrap = os.environ.get("SOAK_SERVER_WRAP", "").split()   # e.g. "memray run --native -o soak.bin"
    server = subprocess.Popen([*wrap, str(ROOT / ".venv/bin/jevstiller"), "serve", "--config", str(tmp / "j.toml"),
                               "--port", str(port)], stderr=open(tmp / "server.log", "w"))
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            if httpx.get(f"{base}/readyz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)

    stats: dict[str, int] = {}
    timeline = []
    stop = asyncio.Event()
    drifted_at = None
    t0 = time.time()

    async def monitor():
        nonlocal drifted_at
        prev = {}
        while not stop.is_set():
            await asyncio.sleep(10)
            now = time.time() - t0
            if drifted_at is None and now >= duration / 2 and not a.no_drift:
                jev.drift()
                drifted_at = now
            cur = dict(stats)
            window = {k: cur.get(k, 0) - prev.get(k, 0) for k in cur}
            prev = cur
            n = sum(window.values()) or 1
            try:                                         # the server's own view: model memory, reloads, training
                async with httpx.AsyncClient(timeout=10) as c:
                    st = (await c.get(f"{base}/jevstiller/v1/stats",
                                      headers={"Authorization": f"Bearer {ADMIN}"})).json()
            except (httpx.HTTPError, ValueError):
                st = {}
            timeline.append({"t": round(now), "rss_mb": round(rss_mb(server.pid), 1),
                             "models_mb": round(st.get("student_memory_mb", float("nan")), 1),
                             "loads": st.get("loads"), "training": st.get("training", {}).get("queued"),
                             "local_share": round(window.get("local", 0) / n, 3), "requests": n,
                             "errors": {k: v for k, v in window.items() if k not in ("local", "upstream")}})
            last = timeline[-1]
            print(f"t={last['t']:>5}s  rss {last['rss_mb']:>7.1f} MB  models {last['models_mb']:>6.1f} MB  "
                  f"loads {last['loads']}  queued {last['training']}  local {last['local_share']:.0%}  "
                  f"req {n}  {last['errors'] or ''}{'  <- drift' if drifted_at and now - drifted_at < 10 else ''}",
                  flush=True)
            if now >= duration:
                stop.set()

    async def run():
        await asyncio.gather(traffic(base, a.rate, stop, stats, jev.world), monitor())
    try:
        asyncio.run(run())
        h = {"Authorization": f"Bearer {ADMIN}"}
        tasks = httpx.get(f"{base}/jevstiller/v1/tasks", headers=h).json()
        events = []
        for t in tasks[:20]:
            st = httpx.get(f"{base}/jevstiller/v1/tasks/{t['key']}", headers=h, timeout=60).json()["status"]
            events += [e["kind"] for e in st["events"]]
    finally:
        server.terminate()
        server.wait(60)

    cut = drifted_at if drifted_at is not None else float("inf")
    pre = [p for p in timeline if p["t"] < cut]
    post = [p for p in timeline if p["t"] >= cut]
    pre_share = float(np.mean([p["local_share"] for p in pre[-6:]])) if pre else 0.0
    post_min = min((p["local_share"] for p in post), default=0.0)
    post_end = float(np.mean([p["local_share"] for p in post[-6:]])) if post else 0.0

    def slope(points):
        if len(points) < 4:
            return float("nan")
        x = np.array([p["t"] for p in points]) / 3600
        return float(np.polyfit(x, [p["rss_mb"] - (p["models_mb"] if p["models_mb"] == p["models_mb"] else 0)
                                    for p in points], 1)[0])
    half = len(timeline) // 2
    errors = {}
    for p in timeline:
        for k, v in p["errors"].items():
            errors[k] = errors.get(k, 0) + v
    summary = {
        "minutes": a.minutes, "rate": a.rate, "requests": sum(stats.values()), "errors": errors,
        "local_share_before_drift": round(pre_share, 3),
        "local_share_min_after_drift": round(post_min, 3) if drifted_at is not None else None,
        "local_share_end": round(post_end, 3) if drifted_at is not None else round(pre_share, 3),
        "recovered": (post_end >= 0.8 * pre_share and post_min < pre_share) if drifted_at is not None else None,
        "rss_mb": {"start": timeline[0]["rss_mb"], "max": max(p["rss_mb"] for p in timeline), "end": timeline[-1]["rss_mb"]},
        "models_mb": {"max": max(p["models_mb"] for p in timeline), "end": timeline[-1]["models_mb"]},
        "loads_per_minute": round(((timeline[-1]["loads"] or 0) - (timeline[0]["loads"] or 0))
                                  / max(1e-9, (timeline[-1]["t"] - timeline[0]["t"]) / 60), 1),
        "rss_minus_models_slope_mb_per_hour": {"first_half": round(slope(timeline[2:half]), 1),
                                  "second_half": round(slope(timeline[half:]), 1)},
        "events": {k: events.count(k) for k in sorted(set(events))},
        "proxy_5xx": sum(v for k, v in errors.items() if k.startswith("http_5")),
    }
    out = ROOT / "experiments" / "results" / "soak.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "timeline": timeline}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
