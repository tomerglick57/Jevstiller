"""Image questions through the real proxy: the product's image encoder, its default settings, and Jev's recorded
answers. No Jev call.

    python3 experiments/image_proxy_replay.py --dataset food101 --encoder siglip

The images are the ones step 2 sent to Jev (image_jev.py), in typesafe-sdk 0.7.5's multipart format with a new random
reference each time. A stand-in upstream answers each with Jev's recorded answer for that image (the step 2 cache),
so only images in the cache are streamed. Reports the share answered locally over the stream, the local answers'
agreement with Jev, and their latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tests"))
from image_data import ROOT as DATA  # noqa: E402
from image_data import load  # noqa: E402
from image_jev import SHORT_SIDE, cache_path  # noqa: E402

GOOD = "tsk_replay"


def main() -> None:
    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from test_proxy import serve

    from jevstiller import Admission, Config, Task, TaskManager, _images, load_encoder
    from jevstiller.encoders import BatchingEncoder, load_image_encoder
    from jevstiller.server import KeyRegistry, ProxySettings, create_app

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="food101")
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="jev-1.14.0")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-per-class", type=int, default=None, help="Config.min_samples_per_class (default: 50)")
    ap.add_argument("--jev-ms", type=float, default=290, help="the stand-in Jev's latency (Jev's image median, step 2)")
    a = ap.parse_args()

    ds = load(a.dataset)
    task = Task(name=f"img_{a.dataset}", instructions=ds["instructions"], classes=ds["classes"], target_agreement=0.98)
    answers = {}
    for line in open(cache_path(a.dataset, a.model)):
        d = json.loads(line)
        version, image_id = d["key"].split("\t", 1)
        if version == task.version:
            answers[image_id] = d
    sent = DATA / a.dataset / f"sent-s{SHORT_SIDE[a.dataset] or 'native'}"
    images = {i: (sent / f"{i.split('/')[1]}.jpg").read_bytes() for i in answers}
    by_digest = {hashlib.sha256(b).hexdigest(): answers[i] for i, b in images.items()}
    ids = sorted(answers)
    np.random.default_rng(a.seed).shuffle(ids)
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids):,} images with Jev's recorded answer ({a.model})", flush=True)

    calls = [0]

    async def system_one(request: Request):                 # the stand-in Jev: its recorded answers, at its pace
        import anyio
        calls[0] += 1
        await anyio.sleep(a.jev_ms / 1000)
        r = _images.read(request.headers["content-type"], await request.body(), True)
        d = by_digest[r.digest]
        conf = d["confidence"]
        return JSONResponse({"model": a.model, "usage": {"input_tokens": d["input_tokens"], "output_tokens": 1},
                             "answers": {"label": {"type": "choice", "choice": d["label"], "confidence": conf,
                                                   "probabilities": d["probs"]}}},
                            headers={"x-typesafe-request-id": d.get("request_id") or "req"})

    upstream_app = Starlette(routes=[Route("/v1/systemone", system_one, methods=["POST"])])
    question = {"type": "choice", "instructions": ds["instructions"], "criteria": dict(ds["classes"])}
    t_load = time.perf_counter()
    image_encoder = BatchingEncoder(load_image_encoder(a.encoder, device=a.device), max_batch=16)
    print(f"image encoder {image_encoder.id} loaded in {time.perf_counter() - t_load:.1f}s", flush=True)
    results = []
    lock = threading.Lock()
    with serve(upstream_app) as upstream, tempfile.TemporaryDirectory() as tmp:
        cfg = Config() if a.min_per_class is None else Config(min_samples_per_class=a.min_per_class)
        m = TaskManager(tmp, None, load_encoder("hash"), cfg, admission=Admission(),
                        janitor_interval_s=3600, image_encoder=image_encoder)          # the product's defaults
        app = create_app(m, ProxySettings(upstream=upstream), KeyRegistry(b"salt", 3600))
        with serve(app) as proxy, httpx.Client(timeout=60) as c:
            def one(n_i):
                n, i = n_i
                ref = f"<image:{uuid.uuid4().hex}>"
                req = {"state": ref, "model": "jev-latest", "questions": {"label": question},
                       "assets": {ref: {"type": "image", "part": "asset-0"}}}
                b = uuid.uuid4().hex
                body = (f'--{b}\r\nContent-Disposition: form-data; name="request"\r\nContent-Type: application/json'
                        f'\r\n\r\n').encode() + json.dumps(req).encode() + (
                    f'\r\n--{b}\r\nContent-Disposition: form-data; name="asset-0"; filename="asset-0"\r\n'
                    f'Content-Type: image/jpeg\r\n\r\n').encode() + images[i] + f"\r\n--{b}--\r\n".encode()
                t0 = time.perf_counter()
                r = c.post(f"{proxy}/v1/systemone", content=body,
                           headers={"authorization": f"Bearer {GOOD}", "TypeSafe-Danger-Zone-Images": "true",
                                    "content-type": f"multipart/form-data; boundary={b}"})
                dt = (time.perf_counter() - t0) * 1000
                ans = r.json()["answers"]["label"]["choice"]
                with lock:
                    results.append((n, r.headers.get("x-jevstiller-source"), ans == answers[i]["label"], dt))
            t0 = time.perf_counter()
            with ThreadPoolExecutor(a.concurrency) as ex:
                list(ex.map(one, enumerate(ids)))
            wall = time.perf_counter() - t0
            [info] = m.tasks()
            status = m.engine(info.key).status()
            print(status.report(), flush=True)
            print("events:", [(e.get("kind"), {k: v for k, v in e.items() if k in ("version", "reason", "n_train", "n_calib",
                   "coverage", "error")}) for e in (status.events or [])][-12:], flush=True)
            counters = [line for line in app.state.metrics.render().splitlines()
                        if line.startswith(("jevstiller_questions_total", "jevstiller_unrouted_total"))]
        m.close()
    results.sort()
    src = np.array([s for _, s, _, _ in results])
    same = np.array([ok for _, _, ok, _ in results])
    lat = np.array([d for _, _, _, d in results])
    local = src == "local"
    out = {"dataset": a.dataset, "encoder": image_encoder.id, "n": len(ids), "seconds": round(wall, 1),
           "jev_calls": calls[0], "local_share": float(local.mean()),
           "local_share_last_1000": float(local[-1000:].mean()),
           "agreement_of_local_answers": float(same[local].mean()) if local.any() else None,
           "agreement_overall": float(same.mean()),
           "first_local_at": int(np.argmax(local)) if local.any() else None,
           "latency_ms_p50_local": float(np.median(lat[local])) if local.any() else None,
           "latency_ms_p50_upstream": float(np.median(lat[~local])),
           "production": status.production, "audit_agreement": status.audit_agreement,
           "teacher_rows_recorded": status.teacher_calls, "counters": counters}
    (ROOT / "experiments" / "results").mkdir(exist_ok=True)
    tag = "" if a.min_per_class is None else f"-minclass{a.min_per_class}"
    (ROOT / "experiments" / "results" / f"image-proxy-{a.dataset}-{a.encoder}{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
