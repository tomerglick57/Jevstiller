"""Jev 1.14's image requests pass through the proxy unchanged, up to Jev's own body limit, and nothing is learned
from them. Bodies are built as typesafe-sdk 0.7.5 sends them: a multipart `request` part, then one file part per
image, which `assets` names. Base64 assets in JSON are the other format Jev accepts.

With the usual 4 MiB limit, one large PNG got a 413 from the proxy that Jev would have answered."""
import json
import logging
import threading
import time
import uuid

import anyio
import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from test_proxy import GOOD, LABELS, serve

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller.server import ProxySettings, create_app

USUAL, IMAGES = 2_000, 20_000          # max_body_bytes, max_image_body_bytes (small, so the bodies are)
HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
QUESTIONS = {"label": {"type": "choice", "instructions": "Which team handles this?",
                       "criteria": {c: "" for c in LABELS}}}


class ImageJev:
    """Answers anything, records what it got; `hold` keeps the first request waiting until set."""

    def __init__(self, hold: threading.Event | None = None):
        self.got: list[tuple[str, bytes]] = []
        self.hold = hold

    async def system_one(self, request: Request):
        body = await request.body()
        self.got.append((request.headers.get("content-type", ""), body))
        if self.hold is not None and len(self.got) == 1:
            while not self.hold.is_set():
                await anyio.sleep(0.01)
        probs = {c: 0.9 if c == "billing" else 0.1 / (len(LABELS) - 1) for c in LABELS}
        return JSONResponse({"model": "jev-1.14.0", "usage": {"input_tokens": 1500, "output_tokens": 1},
                             "answers": {"label": {"type": "choice", "choice": "billing", "confidence": 0.9,
                                                   "probabilities": probs}}})

    def app(self):
        return Starlette(routes=[Route("/v1/systemone", self.system_one, methods=["POST"])])


def multipart(image_bytes: int) -> tuple[str, bytes]:
    """An image request as typesafe-sdk 0.7.5 encodes it (its `with_assets`)."""
    ref = f"<image:{uuid.uuid4().hex}>"
    req = {"state": {"photo": ref}, "model": "jev-latest", "questions": QUESTIONS,
           "assets": {ref: {"type": "image", "part": "asset-0"}}}
    b = uuid.uuid4().hex
    out = b""
    for name, ct, content in (("request", "application/json", json.dumps(req).encode()),
                              ("asset-0", "image/jpeg", b"\xff\xd8\xff" + b"\0" * image_bytes)):
        out += (f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}"\r\n'
                f"Content-Type: {ct}\r\n\r\n").encode() + content + b"\r\n"
    return f"multipart/form-data; boundary={b}", out + f"--{b}--\r\n".encode()


def post(proxy, content, content_type, headers=HEADER, path="/v1/systemone"):
    return httpx.post(f"{proxy}{path}", content=content, timeout=10,
                      headers={"authorization": f"Bearer {GOOD}", "content-type": content_type, **headers})


@pytest.fixture
def manager(tmp_path):
    m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"),
                    admission=Admission(min_requests=1), janitor_interval_s=3600)
    yield m
    m.close()


def settings(up, **kw):
    return ProxySettings(upstream=up, max_body_bytes=USUAL, max_image_body_bytes=IMAGES, **kw)


def test_image_requests_up_to_jevs_limit_are_forwarded_unchanged(manager):
    jev = ImageJev()
    with serve(jev.app()) as up, serve(create_app(manager, settings(up))) as proxy:
        ct, body = multipart(10_000)                       # above the usual limit, under the image limit
        r = post(proxy, body, ct)
        assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream"
        assert jev.got == [(ct, body)]                     # byte for byte, boundary included
        ct, small = multipart(100)
        assert post(proxy, small, ct).status_code == 200 and jev.got[-1] == (ct, small)

        assert post(proxy, body, ct, headers={}).status_code == 413                  # without Jev's header
        assert post(proxy, body, ct, headers={"TypeSafe-Danger-Zone-Images": "false"}).status_code == 413
        assert post(proxy, *reversed(multipart(IMAGES))).status_code == 413          # over Jev's limit
        assert post(proxy, body, ct, path="/v1/other").status_code == 413             # only /v1/systemone
        chunked = httpx.post(f"{proxy}/v1/systemone", content=iter([body[:5000], body[5000:]]), timeout=10,
                             headers={"authorization": f"Bearer {GOOD}", "content-type": ct, **HEADER})
        assert chunked.status_code == 200 and jev.got[-1] == (ct, body)
        assert len(jev.got) == 3
    assert manager.tasks() == []                           # nothing parsed, nothing learned


def test_json_image_requests_are_forwarded_and_never_tasks(manager, caplog):
    caplog.set_level(logging.INFO, logger="jevstiller.access")
    jev = ImageJev()
    with serve(jev.app()) as up, serve(create_app(manager, settings(up))) as proxy:
        ref = f"<image:{uuid.uuid4().hex}>"
        req = {"state": ref, "model": "jev-latest", "questions": QUESTIONS,
               "assets": {ref: {"type": "image", "format": "jpg", "data_base64": "/9j/" + "A" * 400}}}
        for _ in range(3):
            assert post(proxy, json.dumps(req).encode(), "application/json").status_code == 200
        ct, body = multipart(100)
        post(proxy, body, ct)
    assert len(jev.got) == 4 and manager.tasks() == []
    reasons = [json.loads(r.getMessage()).get("reason") for r in caplog.records if r.name == "jevstiller.access"]
    assert reasons == ["images"] * 4


def test_large_bodies_held_at_once_are_capped(manager):
    hold = threading.Event()
    jev = ImageJev(hold)
    with serve(jev.app()) as up:
        app = create_app(manager, settings(up, max_large_bodies=1))
        with serve(app) as proxy:
            ct, body = multipart(10_000)
            first: list[httpx.Response] = []
            t = threading.Thread(target=lambda: first.append(post(proxy, body, ct)))
            t.start()
            deadline = time.monotonic() + 10
            while not jev.got and time.monotonic() < deadline:      # the first is at Jev, holding the slot
                time.sleep(0.01)
            busy = post(proxy, body, ct)
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
            chunked = httpx.post(f"{proxy}/v1/systemone", content=iter([body[:5000], body[5000:]]), timeout=10,
                                 headers={"authorization": f"Bearer {GOOD}", "content-type": ct, **HEADER})
            assert chunked.status_code == 503                        # caught while it streams in
            small_ct, small = multipart(100)
            assert post(proxy, small, small_ct).status_code == 200   # under the usual limit: no slot needed
            hold.set()
            t.join(10)
            assert first[0].status_code == 200
            assert post(proxy, body, ct).status_code == 200          # the slot was given back
            assert len(jev.got) == 3
            metrics = app.state.metrics.render()
    assert 'jevstiller_rejected_total{reason="large_bodies"} 2' in metrics
