"""Jev as the teacher for an image dataset (step 2 of the image benchmark). Needs TYPESAFE_API_KEY.

    python3 experiments/image_jev.py --dataset crack --probe 3         # 3 images: tokens per image, projected cost
    python3 experiments/run.py --dataset img_crack --teacher jev --encoder clip-b32 --max-jev-calls 5000 --tag img

Every answer is cached (experiments/cache/img_<dataset>-<model>-s<size>.jsonl), so nothing is paid for twice and a
replay with the same settings runs again without the key. A live replay only asks Jev about the held-out rows and
the rows the loop routes to the teacher, not the whole dataset.

Each request is one image as the whole state (`TypeSafe-Danger-Zone-Images: true`, multipart, as typesafe-sdk
0.7.5 sends it) and one Choice question. To spend as few tokens as Jev allows, an image is resized so that its
shorter side is `short_side` px (Jev's minimum is 280), and sent as JPEG; document pages keep a readable size.
The embeddings (image_embed.py) are of the original images, which these resizes barely change at the encoders'
224 px.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_data import ROOT as DATA  # noqa: E402
from image_data import SPECS, iter_images  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHORT_SIDE = {"food101": 280, "crack": 280, "rvl_cdip": 0}     # 0: as is (RVL-CDIP pages are ~750 px wide scans)
JEV_MIN, JEV_MAX = 280, 1568


def as_sent(data: bytes, short_side: int) -> bytes:
    """The image as Jev gets it: shorter side `short_side` (0: unchanged unless outside Jev's limits), each side
    within 280-1568 px, JPEG."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = short_side or min(max(min(w, h), JEV_MIN), JEV_MAX)
        scale = s / min(w, h)
        w2, h2 = max(JEV_MIN, round(w * scale)), max(JEV_MIN, round(h * scale))
        if max(w2, h2) > JEV_MAX:                        # a very long image: crop the long side to Jev's maximum
            im = im.resize((w2, h2), Image.LANCZOS)
            left, top = (w2 - min(w2, JEV_MAX)) // 2, (h2 - min(h2, JEV_MAX)) // 2
            im = im.crop((left, top, left + min(w2, JEV_MAX), top + min(h2, JEV_MAX)))
        elif (w2, h2) != (w, h):
            im = im.resize((w2, h2), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=90)
        return out.getvalue()


def sent_dir(dataset: str) -> Path:
    """Every image of the dataset as Jev gets it (<n>.jpg), written in one pass the first time: a replay asks for
    ids in batches, and finding them in the parquet or zip files each time took one pass over the dataset per batch."""
    d = DATA / dataset / f"sent-s{SHORT_SIDE[dataset] or 'native'}"
    if not (d / ".complete").exists():
        d.mkdir(exist_ok=True)
        for i, data, _ in iter_images(dataset, decode=False):
            (d / f"{i.rsplit('/', 1)[1]}.jpg").write_bytes(as_sent(data, SHORT_SIDE[dataset]))
        (d / ".complete").touch()
    return d


def cache_path(dataset: str, model: str) -> Path:
    return ROOT / "experiments" / "cache" / f"img_{dataset}-{model}-s{SHORT_SIDE[dataset] or 'native'}.jsonl"


class BudgetExceeded(SystemExit):
    pass


class ImageJevTeacher:
    """Asks Jev about images by id. Stops the run (SystemExit) before a batch that would exceed `max_calls`: never
    inside one, since the cache is written when the batch returns and answers already paid for would be lost."""

    def __init__(self, dataset: str, model: str = "jev-1.14.0", max_calls: int = 0, rpm: int = 600,
                 concurrency: int = 4, price_per_mtok: float = 0.042, timeout: float = 60.0):
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("TYPESAFE_API_KEY is not set")
        self.dataset, self.model, self.name = dataset, model, f"jev:{model}"
        self.max_calls, self.calls, self.tokens = max_calls, 0, 0
        self.price, self.concurrency = price_per_mtok, concurrency
        self.interval, self.next_at, self.lock = 60.0 / rpm, 0.0, threading.Lock()
        base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
        self.client = httpx.Client(base_url=base, timeout=timeout,
                                   headers={"authorization": f"Bearer {key}", "TypeSafe-Danger-Zone-Images": "true"})
        self.sent = sent_dir(dataset)

    def _image(self, image_id: str) -> bytes:
        return (self.sent / f"{image_id.rsplit('/', 1)[1]}.jpg").read_bytes()

    def _wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_at)
            self.next_at = t + self.interval
        if t > now:
            time.sleep(t - now)

    def _one(self, image_id: str, task):
        from jevstiller.teachers import TeacherOutput
        ref = f"<image:{uuid.uuid4().hex}>"
        body = {"state": ref, "model": self.model,
                "questions": {"label": {"type": "choice", "instructions": task.instructions,
                                        "criteria": dict(task.classes)}},
                "assets": {ref: {"type": "image", "part": "asset-0"}}}
        files = {"request": (None, json.dumps(body), "application/json"),
                 "asset-0": ("asset-0", self._image(image_id), "image/jpeg")}
        for attempt in range(6):
            self._wait()
            t0 = time.perf_counter()
            r = self.client.post("/v1/systemone", files=files)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(60.0, float(r.headers.get("retry-after", 2 ** attempt))))
                continue
            r.raise_for_status()
            break
        d = r.json()
        a = d["answers"]["label"]
        probs = {c: float(a["probabilities"].get(c, 0.0)) for c in task.labels}
        z = sum(probs.values()) or 1.0
        toks = int(d.get("usage", {}).get("input_tokens", 0))
        with self.lock:
            self.tokens += toks
        return TeacherOutput(label=str(a["choice"]), probs={c: p / z for c, p in probs.items()},
                             confidence=float(a["confidence"]), input_tokens=toks, cost_usd=toks * self.price / 1e6,
                             latency_ms=dt, request_id=r.headers.get("x-typesafe-request-id"),
                             model=f"jev:{d.get('model', self.model)}")

    def _safe(self, image_id: str, task):
        try:
            return self._one(image_id, task)
        except Exception as e:                           # one failed request fails only its own row
            return e

    def classify(self, texts: Sequence[str], task):
        from concurrent.futures import ThreadPoolExecutor
        if self.max_calls and self.calls + len(texts) > self.max_calls:
            raise BudgetExceeded(f"stopped before {len(texts)} more Jev calls: {self.calls:,} made, --max-jev-calls "
                                 f"{self.max_calls:,}. Answers so far are cached; run again with a higher cap to go on.")
        self.calls += len(texts)
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(lambda t: self._safe(t, task), texts))


def main() -> None:
    from image_data import load

    from jevstiller import Task
    from jevstiller.teachers import CachedTeacher
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPECS))
    ap.add_argument("--model", default="jev-1.14.0")
    ap.add_argument("--probe", type=int, default=3, help="images to send (cached, so they count towards the run)")
    ap.add_argument("--plan", type=int, default=5000, help="Jev calls the planned run would make, for the projection")
    ap.add_argument("--target", type=float, default=0.98, help="must match run.py's --target (part of the task version)")
    a = ap.parse_args()
    ds = load(a.dataset)
    task = Task(name=f"img_{a.dataset}", instructions=ds["instructions"], classes=ds["classes"],
                target_agreement=a.target)
    jev = ImageJevTeacher(a.dataset, a.model, max_calls=a.probe)
    teacher = CachedTeacher(jev, cache_path(a.dataset, a.model))
    ids = [i for i, _ in ds["rows"][:a.probe]]
    out = teacher.classify(ids, task)
    ok = [o for o in out if not isinstance(o, Exception)]
    for i, o, (_, truth) in zip(ids, out, ds["rows"][:a.probe], strict=True):
        print(i, "error: " + str(o)[:120] if isinstance(o, Exception) else
              f"{o.label!r} (truth {truth!r}) conf {o.confidence:.2f}, {o.input_tokens} tokens, {o.latency_ms:.0f} ms, {o.model}")
    if ok:
        per = sum(o.input_tokens for o in ok) / len(ok)
        print(f"~{per:.0f} input tokens per request; {a.plan:,} calls ≈ {per * a.plan / 1e6:.2f}M tokens ≈ "
              f"${per * a.plan * jev.price / 1e6:.2f} at ${jev.price}/M")


if __name__ == "__main__":
    main()
