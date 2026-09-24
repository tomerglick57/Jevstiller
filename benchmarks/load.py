"""P6.4: load test. A real `jevstiller serve` process in front of a fake Jev (its own process, ~290 ms like the
real one), driven by async clients at a fixed concurrency (closed loop: each connection sends its next request when the
last one returns; at most 64 connections per client process).

    python benchmarks/load.py                          # all scenarios, hash encoder, 30 s each
    python benchmarks/load.py --encoder small --seconds 60 --concurrency 64 256

Scenarios:
  forward  nothing is admitted: every request goes to Jev. Measures the proxy's own overhead and how many
           requests it can keep in flight.
  local    one trained question: most requests are answered locally (the rest: audit, deferred).
  mixed    20 questions, Zipf-distributed, trained during a warm-up: a realistic spread.

Each scenario runs one server: a warm-up (training, up to --warmup-s), then each concurrency level in turn.

Writes experiments/results/load.json and prints a table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = ["billing", "technical", "cancellation", "sales", "other"]
KEY = "tsk_load"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def fake_jev(port: int, latency_ms: float) -> None:
    """Runs in its own process: Jev's wire format, answers from the synthetic teacher after ~latency_ms."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from jevstiller import SyntheticTeacher, SyntheticWorld, Task
    teacher = SyntheticTeacher(SyntheticWorld(LABELS, seed=1))
    rng = random.Random(0)
    n = [0]

    async def systemone(request: Request):
        n[0] += 1
        await asyncio.sleep(latency_ms / 1000 * rng.lognormvariate(0, 0.15))
        body = await request.json()
        answers = {}
        for name, q in body["questions"].items():
            o = teacher.classify([body["state"]], Task("t", q.get("instructions"), list(q["criteria"])))[0]
            answers[name] = {"type": "choice", "choice": o.label, "confidence": o.confidence, "probabilities": o.probs}
        return JSONResponse({"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 50, "output_tokens": 1}},
                            headers={"x-typesafe-request-id": f"req_{n[0]}"})
    uvicorn.run(Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"])]), port=port,
                log_level="warning", backlog=4096)


def wait_ready(url: str, proc: subprocess.Popen, log: Path) -> None:
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(log.read_text()[-3000:] if log.exists() else "exited")
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{url} not ready")


def questions(n: int) -> list[dict]:
    return [{"type": "choice", "instructions": f"Which team handles this? (queue {i})",
             "criteria": {c: "" for c in LABELS}} for i in range(n)]


async def drive(base: str, qs: list[dict], weights: list[float], texts: list[str], seconds: float,
                concurrency: int, offset: int = 0) -> list[tuple[str, float]]:
    """(source or error, latency s) per request, closed loop at `concurrency`."""
    out: list[tuple[str, float]] = []
    deadline = time.perf_counter() + seconds
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=60, limits=limits) as c:
        async def conn(seed: int):
            rng = random.Random(seed * 7919 + offset)
            while time.perf_counter() < deadline:
                q = rng.choices(qs, weights)[0]
                body = {"state": rng.choice(texts), "model": "jev-latest", "questions": {"label": q}}
                t0 = time.perf_counter()
                try:
                    r = await c.post(f"{base}/v1/systemone", json=body, headers={"Authorization": f"Bearer {KEY}"})
                    src = r.headers.get("x-jevstiller-source", "?") if r.status_code == 200 else f"http_{r.status_code}"
                except httpx.HTTPError as e:
                    src = f"exc_{type(e).__name__}"
                out.append((src, time.perf_counter() - t0))
        await asyncio.gather(*(conn(i) for i in range(concurrency)))
    return out


PER_PROCESS = 64          # one Python httpx client can't drive more connections than this: at 256 in one process,
                          # throughput against a bare fake Jev fell from ~850 to ~85 req/s


def drive_procs(base: str, nq: int, seconds: float, concurrency: int) -> list[tuple[str, float]]:
    """`drive` split over enough client processes (at most PER_PROCESS connections each)."""
    procs = -(-concurrency // PER_PROCESS)
    if procs == 1:
        return asyncio.run(drive(base, questions(nq), zipf(nq), texts_for_load(), seconds, concurrency))
    shares = [concurrency // procs + (i < concurrency % procs) for i in range(procs)]
    ps = [subprocess.Popen([sys.executable, __file__, "--drive", base, "--nq", str(nq), "--seconds", str(seconds),
                            "--concurrency", str(c), "--seed", str(i)], stdout=subprocess.PIPE, text=True)
          for i, c in enumerate(shares)]
    out: list[tuple[str, float]] = []
    for pr in ps:
        out += [tuple(x) for x in json.loads(pr.communicate()[0])]
    return out


def zipf(n: int) -> list[float]:
    return [1 / (i + 1) for i in range(n)]


def texts_for_load(seed: int = 1) -> list[str]:
    from jevstiller import SyntheticWorld
    return [t for t, _ in SyntheticWorld(LABELS, seed=seed).sample(50_000)]


def pct(v: list[float], p: float) -> float | None:
    return round(v[min(len(v) - 1, int(p * len(v)))] * 1000, 1) if v else None


def summarize(res: list[tuple[str, float]], seconds: float) -> dict:
    by: dict[str, list[float]] = {}
    for src, dt in res:
        by.setdefault(src, []).append(dt)
    for v in by.values():
        v.sort()
    every = sorted(dt for _, dt in res)
    return {"requests": len(res), "throughput_rps": round(len(res) / seconds, 1),
            "share": {k: round(len(v) / len(res), 3) for k, v in by.items()},
            "latency_ms": {k: {"p50": pct(v, .5), "p90": pct(v, .9), "p99": pct(v, .99)}
                           for k, v in [("all", every), *by.items()]},
            "errors": sum(len(v) for k, v in by.items() if k not in ("local", "upstream"))}


def run_scenario(name: str, a, jev_url: str) -> list[dict]:
    """One server per scenario: warm it up (train), then measure each concurrency level in turn."""
    tmp = Path(tempfile.mkdtemp(prefix=f"jvs-load-{name}-"))
    admit = 10**9 if name == "forward" else 1
    nq = 20 if name == "mixed" else 1
    (tmp / "j.toml").write_text(f'''
[server]
data_dir = "{tmp}/data"
log_level = "info"
log_format = "json"
[proxy]
upstream = "{jev_url}"
max_upstream_inflight = 1024
[manager]
admit_after = {admit}
target_agreement = 0.95
[encoder]
spec = "{a.encoder}"
[engine]
min_train_samples = 400
min_samples_per_class = 20
min_calib_samples = 150
shadow_min_samples = 300
min_new_samples = 100000
''')
    port = free_port()
    log = tmp / "server.log"
    server = subprocess.Popen([str(ROOT / ".venv/bin/jevstiller"), "serve", "--config", str(tmp / "j.toml"),
                               "--port", str(port)], stderr=open(log, "w"))
    base = f"http://127.0.0.1:{port}"
    out = []
    try:
        wait_ready(f"{base}/readyz", server, log)
        warm, t0 = None, time.time()
        if name != "forward":                            # train until most answers are local
            while time.time() - t0 < a.warmup_s:
                warm = summarize(drive_procs(base, nq, 10, 32), 10)
                if warm["share"].get("local", 0) >= (0.85 if name == "local" else 0.7):
                    break
        warm_s = round(time.time() - t0) if warm else 0
        for conc in a.concurrency:
            t_start = time.time()
            r = summarize(drive_procs(base, nq, a.seconds, conc), a.seconds)
            r["server_latency_ms"] = server_side(log, t_start, time.time())
            r.update(scenario=name, concurrency=conc, warmup_s=warm_s,
                     warmup_local_share=warm["share"].get("local") if warm else None)
            rss = [ln for ln in Path(f"/proc/{server.pid}/status").read_text().splitlines() if ln.startswith("VmRSS")]
            r["server_rss_mb"] = round(int(rss[0].split()[1]) / 1024, 1) if rss else None
            out.append(r)
            report(r)
        return out
    finally:
        server.terminate()
        server.wait(60)


def server_side(log: Path, t0: float, t1: float) -> dict:
    """Per-source latency as the server measured it (its access log), for requests finished in [t0, t1]. The
    client's own numbers include the load generator, which shares the machine (and, with a CPU encoder, fights
    ONNX Runtime for cores)."""
    by: dict[str, list[float]] = {}
    for line in log.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("logger") != "jevstiller.access" or not t0 <= d.get("ts", 0) <= t1:
            continue
        m = json.loads(d["msg"]) if isinstance(d.get("msg"), str) and d["msg"].startswith("{") else d
        by.setdefault(m.get("source", "?"), []).append(m["latency_ms"] / 1000)
    out = {}
    for k, v in by.items():
        v.sort()
        out[k] = {"p50": pct(v, .5), "p99": pct(v, .99), "n": len(v)}
    return out


def report(r: dict) -> None:
    lat = r["latency_ms"]
    print(f"{r['scenario']:8s} c={r['concurrency']:<4d} {r['throughput_rps']:>8.1f} req/s   "
          f"all p50 {lat['all']['p50']} p99 {lat['all']['p99']} ms   "
          + "   ".join(f"{k} {r['share'][k]:.0%} p50 {lat[k]['p50']} p99 {lat[k]['p99']}"
                     for k in ("local", "upstream") if k in r["share"])
          + (f"   errors {r['errors']}" if r["errors"] else "") + f"   rss {r['server_rss_mb']} MB"
          + "\n" + " " * 14 + "server-side: " + "   ".join(f"{k} p50 {v['p50']} p99 {v['p99']}"
                                                          for k, v in sorted(r.get("server_latency_ms", {}).items())),
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["forward", "local", "mixed"])
    ap.add_argument("--concurrency", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--warmup-s", type=float, default=240)
    ap.add_argument("--latency-ms", type=float, default=290)
    ap.add_argument("--encoder", default="hash")
    ap.add_argument("--fake-jev", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--drive", help=argparse.SUPPRESS)          # a client process of drive_procs
    ap.add_argument("--nq", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.fake_jev:
        fake_jev(a.fake_jev, a.latency_ms)
        return
    if a.drive:
        res = asyncio.run(drive(a.drive, questions(a.nq), zipf(a.nq), texts_for_load(), a.seconds,
                                a.concurrency[0], offset=a.seed))
        print(json.dumps(res))
        return
    jport = free_port()
    jlog = Path(tempfile.mkdtemp(prefix="jvs-load-jev-")) / "jev.log"
    jev = subprocess.Popen([sys.executable, __file__, "--fake-jev", str(jport), "--latency-ms", str(a.latency_ms)],
                           stderr=open(jlog, "w"))
    results = {"latency_ms_upstream": a.latency_ms, "encoder": a.encoder, "seconds": a.seconds, "runs": []}
    try:
        wait_ready(f"http://127.0.0.1:{jport}/", jev, jlog)
        for sc in a.scenarios:
            results["runs"] += run_scenario(sc, a, f"http://127.0.0.1:{jport}")
    finally:
        jev.terminate()
        jev.wait(30)
    out = ROOT / "experiments" / "results" / "load.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
