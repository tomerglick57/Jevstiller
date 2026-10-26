"""Regression tests for security audit run 5 (2026-10-24). See docs/security.md."""
import base64
import contextlib
import json
import socket
import sqlite3
import struct
import threading
import time
import zlib

import httpx
import numpy as np
import pytest
from test_audit_fixes import Upstream, _body
from test_image_tasks import HEADER as IMAGES
from test_image_tasks import QUESTION as IMAGE_Q
from test_image_tasks import ColourEncoder, png, request_body
from test_images import ImageJev
from test_proxy import GOOD, serve

from jevstiller import Admission, Config, HashEncoder, Jevstiller, Task, TaskManager, _images
from jevstiller._images import UNREADABLE
from jevstiller.encoders import BatchingEncoder
from jevstiller.encoders._image import check_image
from jevstiller.server import KeyRegistry, ProxySettings, create_app

LABELS = ["billing", "technical", "cancellation", "sales", "other"]
QA, QB = "Which team handles this?", "Which team handles this, really?"


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=10**9, shadow_min_samples=150, training="inline", train_threads=1)
    return Config(**{**base, **kw})


def _rows(m, tmp_path, question, where="1"):
    key = m.resolve("acme", question, LABELS)[0]
    m.engine(key).store.flush()
    with sqlite3.connect(tmp_path / "tasks" / key / "samples.sqlite") as c:
        return c.execute(f"SELECT channel, split, COUNT(*) FROM samples WHERE {where} GROUP BY 1, 2").fetchall()


def test_a_task_reloaded_before_every_request_is_still_audited(tmp_path, teacher, world):
    """1: every engine load restarted the audit draw at the same first number (0.637 for seed 0), so a task whose
    engine was reloaded before each request (more active tasks than max_loaded, or a caller cycling its own) was
    never audited: no drift fallback, no shadow judged. The draw's position now survives an unload."""
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(audit_rate=0.1), max_loaded=1,
                    admission=Admission(1), target_agreement=0.9, janitor_interval_s=3600)
    for _ in range(30):                                  # A alone until its student answers
        res = m.classify("acme", QA, LABELS, [t for t, _ in world.sample(200)])
        if any(r.source.startswith("student") for r in res):
            break
    keyA = m.resolve("acme", QA, LABELS)[0]
    assert m.engine(keyA)._prod is not None
    m.engine(keyA).store.flush()
    last = m.engine(keyA).store.max_id()
    loads = m.loads
    for _ in range(300):                                 # A and B in turn: each request reloads its engine
        m.classify("acme", QA, LABELS, [world.sample(1)[0][0]])
        m.classify("acme", QB, LABELS, [world.sample(1)[0][0]])
    assert m.loads - loads >= 550
    audits = sum(n for ch, _, n in _rows(m, tmp_path, QA, f"id > {last}") if ch == "audit")
    assert audits >= 8                                   # ~30 expected at 10%; 0 before the fix (0.637 > 0.1)
    m.close()


def test_rows_of_a_task_reloaded_before_every_request_still_reach_calibration(tmp_path, teacher, world):
    """1: the split draw restarted too, so every row of such a task landed in the same split, and it never got the
    calibration rows a first student needs (or never any training rows)."""
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=512), _cfg(), max_loaded=1, admission=Admission(1),
                    janitor_interval_s=3600)
    for _ in range(200):
        m.classify("acme", QA, LABELS, [world.sample(1)[0][0]])
        m.classify("acme", QB, LABELS, [world.sample(1)[0][0]])
    splits = {s: n for ch, s, n in _rows(m, tmp_path, QB) if ch == "bootstrap"}
    assert splits.get("calib", 0) >= 15 and splits.get("train", 0) >= 100   # ~40 / ~160 expected
    m.close()


def test_a_servers_draws_are_keyed_and_a_replays_are_not(tmp_path, teacher):
    """1: the draws were seeded from public values (the config seed and the task version), so a caller who knew
    the question could compute which of its requests would be audited or held out. With a hash key (the server's
    salt) they are keyed with it; without one (a replay), unchanged."""
    task = Task(name="q", instructions=QA, classes=LABELS)
    plain = Jevstiller(task, teacher, tmp_path / "a", config=Config(seed=0))
    keyed = Jevstiller(task, teacher, tmp_path / "b", config=Config(seed=0), hash_key=b"salt")
    other = Jevstiller(task, teacher, tmp_path / "c", config=Config(seed=0), hash_key=b"other")
    first = [e.rng.random(4).tolist() for e in (plain, keyed, other)]
    assert first[0] == np.random.default_rng(0).random(4).tolist()
    assert first[1] != first[0] and first[2] != first[1]
    assert plain.store.split_state() != keyed.store.split_state()
    for e in (plain, keyed, other):
        e.close()


class _Tokenizer:
    """A slow encoder that refuses what the tokenizers library refuses (a lone surrogate: TypeError), so that
    concurrent calls queue up into one batch."""

    def __init__(self):
        self.inner = HashEncoder(dim=64)
        self.id, self.dim, self.sizes = self.inner.id, self.inner.dim, []

    def encode(self, texts):
        self.sizes.append(len(texts))
        time.sleep(0.05)
        if any("poison" in t or any(0xD800 <= ord(c) <= 0xDFFF for c in t) for t in texts):
            raise TypeError("TextEncodeInput must be Union[TextInputSequence, ...]")
        return self.inner.encode(texts)


def test_one_callers_bad_text_fails_only_its_own_call():
    """2: the shared encoder merged every queued call into one batch and, when it raised, failed every call in it,
    so one request the tokenizer refused cost every other caller in the batch its local answer and training row."""
    enc = BatchingEncoder(_Tokenizer())
    results: dict = {}

    def call(name, text):
        try:
            results[name] = enc.encode([text])
        except Exception as e:
            results[name] = e
    first = threading.Thread(target=call, args=("first", "keeps the worker busy"))
    first.start()
    time.sleep(0.01)                                     # the rest queue up behind it, as one batch
    ts = [threading.Thread(target=call, args=(f"good{i}", f"honest text {i}")) for i in range(5)]
    ts.append(threading.Thread(target=call, args=("bad", "poison")))
    [t.start() for t in ts]
    [t.join() for t in [first, *ts]]
    assert max(enc.inner.sizes) >= 6                     # they were one batch
    assert isinstance(results["bad"], TypeError)
    assert all(results[f"good{i}"].shape == (1, 64) for i in range(5))
    enc.close()


def test_a_lone_surrogate_is_encoded():
    """2: a JSON escape such as \\udc80 becomes a lone surrogate, which the model encoders' tokenizer refuses; it is
    replaced before encoding, so the request that carried it is served too."""
    enc = BatchingEncoder(_Tokenizer())
    X = enc.encode(["hello \udc80"])
    assert np.allclose(X, HashEncoder(dim=64).encode(["hello \ufffd"]))
    enc.close()


# ---- the image path (3, 4, 8) and the HTTP layer (6, 7) -------------------------------------------------------

@contextlib.contextmanager
def _image_proxy(tmp_path, upstream_app=None, verified=True, image_encoder=True, **settings):
    """A proxy with an image encoder in front of a mock Jev that answers anything (ImageJev)."""
    with serve(upstream_app or ImageJev().app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=64), _cfg(), admission=Admission(1),
                        janitor_interval_s=3600, image_encoder=ColourEncoder() if image_encoder else None)
        keys = KeyRegistry(b"salt", 3600)
        if verified:
            keys.accept(keys.hash(GOOD))
        app = create_app(m, ProxySettings(upstream=upstream, **settings), keys)
        with serve(app) as proxy:
            yield proxy, app, keys
        m.close()


def _send(proxy, content_type, body, **headers):
    return httpx.post(f"{proxy}/v1/systemone", content=body, timeout=30,
                      headers={"authorization": f"Bearer {GOOD}", "content-type": content_type, **IMAGES, **headers})


def test_a_boundary_hidden_in_another_parameter_is_not_read():
    """3: the boundary was the first `boundary=` anywhere in Content-Type, even inside another parameter's quoted
    value, so the proxy and Jev (reading the real parameter) could frame one body differently, and the proxy record
    Jev's answer about one image against another. Now: exactly one, parsed as standard parsers parse it."""
    ct, body = request_body(png("red"))
    b = ct.split("boundary=")[1]
    assert isinstance(_images.read(ct, body, True), _images.ImageRequest)
    assert _images.read(f'multipart/form-data; foo="x; boundary={b}"; boundary=REAL', body, True) == UNREADABLE
    assert _images.read(f"{ct}; boundary=OTHER", body, True) == UNREADABLE


def test_an_asset_with_two_sources_is_not_read():
    """3: with both `part` and `data_base64`, the proxy used the part, on the unverified assumption that Jev does."""
    asset = {"type": "image", "part": "asset-0", "format": "png",
             "data_base64": base64.b64encode(png("green")).decode()}
    ct, body = request_body(png("red"), asset=asset)
    assert _images.read(ct, body, True) == _images.INVALID


def test_bytes_after_the_image_are_not_read():
    """3: an image part with bytes after the image's end passed (Pillow ignores them), so the image the proxy
    embedded could carry what another reader frames as more of the request."""
    ct, body = request_body(png("red") + b"\r\n--REAL\r\nmore")
    assert _images.read(ct, body, True) == _images.SCOPE


def test_malformed_image_requests_are_forwarded_not_failed(tmp_path):
    """8: these raised out of the reader and the caller got a 500, not Jev's answer."""
    ref = "<image:1>"
    data = base64.b64encode(png("red")).decode()

    def js(assets, state=f'"{ref}"'):
        return (f'{{"state": {state}, "model": "jev-latest", "questions": {json.dumps({"label": IMAGE_Q})}, '
                f'"assets": {json.dumps(assets)}}}').encode()
    deep = "[" * 1200 + "]" * 1200
    bodies = [js({ref: {"type": "image", "data_base64": data, "format": {}}}),
              js({ref: {"type": "image", "data_base64": data, "format": []}}),
              js({ref: {"type": "image", "data_base64": data, "format": "png"}},
                 state=f'{{"photo": "{ref}", "deep": {deep}}}')]
    ct, mixed = request_body(png("red"), extra_parts=[("asset-1", "image/png", png("blue"))],
                             asset={"type": "image", "part": "asset-0"})
    mixed = mixed.replace(b'"assets": {', b'"assets": {"<image:2>": {"type": "image", "part": 5}, ', 1)
    with _image_proxy(tmp_path) as (proxy, _, _):
        for body in bodies:
            r = _send(proxy, "application/json", body)
            assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream", r.text
        r = _send(proxy, ct, mixed)
        assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream", r.text


def test_an_unaccepted_keys_image_requests_are_forwarded_unread(tmp_path, monkeypatch):
    """4: an image body was parsed, and its image's header checked, before anything had checked the caller's key:
    up to 800 MiB and seconds of CPU per request from anyone who could reach the port. Now a key Jev hasn't accepted
    is forwarded unread, Jev's answer accepts it, and its next image request is read."""
    reads = []
    real = _images.read
    monkeypatch.setattr(_images, "read", lambda *a: reads.append(1) or real(*a))
    ct, body = request_body(png("red"))
    with _image_proxy(tmp_path, verified=False) as (proxy, app, keys):
        r = _send(proxy, ct, body)
        assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream"
        assert reads == [] and keys.verified(keys.hash(GOOD))
        assert 'jevstiller_unrouted_total{reason="key_unverified"} 1' in app.state.metrics.render()
        _send(proxy, *request_body(png("blue")))
        assert reads == [1]


def test_image_reads_are_capped(tmp_path, monkeypatch):
    """4: nothing bounded how many image bodies were read at once beyond anyio's 40 threads; past the cap they are
    forwarded."""
    monkeypatch.setattr(_images, "read", lambda *a: pytest.fail("read past the cap"))
    with _image_proxy(tmp_path, max_image_reads=0) as (proxy, app, _):
        r = _send(proxy, *request_body(png("red")))
        assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream"
        assert 'jevstiller_unrouted_total{reason="overloaded"} 1' in app.state.metrics.render()


def test_a_json_body_with_too_many_items_is_not_parsed(monkeypatch):
    """4: a 32 MiB `[{},{},...]` body cost ~800 MiB to parse; 16 at once went past a 4 GiB container."""
    body = b"[" + b"{}," * 60_000 + b"{}]"
    monkeypatch.setattr(_images.json, "loads", lambda *a, **k: pytest.fail("parsed"))
    assert _images.read("application/json", body, True) == UNREADABLE


def _png_with(chunk: bytes, n: int) -> bytes:
    """png("red") with `n` copies of a chunk (type + data) before its pixel data."""
    data = png("red")
    one = struct.pack(">I", len(chunk) - 4) + chunk + struct.pack(">I", zlib.crc32(chunk))
    return data[:33] + one * n + data[33:]               # after the signature and IHDR


def test_pngs_with_many_chunks_before_their_pixels_are_refused():
    """4: Pillow reads every chunk before IDAT (and inflates each zTXt up to 1 MiB) in the header check:
    325,000 empty chunks took 0.7 s, 63 zTXt chunks 64 MiB."""
    assert check_image(_png_with(b"prVt", 4)).ok
    assert not check_image(_png_with(b"prVt", 300)).ok
    ztxt = b"zTXtk\x00\x00" + zlib.compress(b"x" * 1000)
    assert check_image(_png_with(ztxt, 2)).ok
    assert not check_image(_png_with(ztxt, 9)).ok


def test_connection_cannot_drop_the_bearer(tmp_path):
    """6: a caller could name `authorization` in Connection; the proxy then forwarded the request without the bearer
    it had taken the caller's identity from, and accepted that key if Jev authenticated another credential."""
    up = Upstream()
    with serve(up.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=64), _cfg(), admission=Admission(1), janitor_interval_s=3600)
        app = create_app(m, ProxySettings(upstream=upstream), KeyRegistry(b"salt", 3600))
        with serve(app) as proxy:
            r = httpx.post(f"{proxy}/v1/systemone", json=_body(), timeout=10,
                           headers={"Authorization": f"Bearer {GOOD}", "Connection": "authorization"})
        m.close()
    assert r.status_code == 200
    assert up.requests[-1][2].get("authorization") == f"Bearer {GOOD}"


def _headers_only(port: int, length: int, body: bytes = b"") -> socket.socket:
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall((f"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {GOOD}\r\n"
               f"TypeSafe-Danger-Zone-Images: true\r\nContent-Type: application/json\r\n"
               f"Content-Length: {length}\r\n\r\n").encode() + body)
    return s


def test_a_connection_that_sends_only_headers_holds_no_large_body_slot(tmp_path):
    """7: a large-body slot was taken on the declared Content-Length, before any byte arrived, and nothing bounded
    how long the body could take: 16 connections that sent only headers denied every caller's large request."""
    with _image_proxy(tmp_path, image_encoder=False, max_body_bytes=2_000, max_image_body_bytes=20_000,
                      max_large_bodies=1) as (proxy, _, _):
        idle = _headers_only(int(proxy.rsplit(":", 1)[1]), 15_000)
        time.sleep(0.3)
        r = _send(proxy, "application/json", b"x" * 10_000)
        idle.close()
    assert r.status_code == 200


def test_a_body_that_never_arrives_gets_408(tmp_path):
    """7: with no body deadline, a request whose body stopped arriving held its connection (and, past 4 MiB, its
    slot) for as long as the client kept it open."""
    with _image_proxy(tmp_path, image_encoder=False, body_timeout_s=0.5) as (proxy, _, _):
        s = _headers_only(int(proxy.rsplit(":", 1)[1]), 15_000, b"x" * 100)
        s.settimeout(5)
        reply = s.recv(4096)
        s.close()
    assert reply.startswith(b"HTTP/1.1 408")


def _trained(tmp_path, teacher, world, encoder):
    m = TaskManager(tmp_path, teacher, encoder, _cfg(), admission=Admission(1), target_agreement=0.9,
                    janitor_interval_s=3600)
    for _ in range(30):
        if any(r.source.startswith("student") for r in m.classify("acme", QA, LABELS, [t for t, _ in world.sample(200)])):
            break
    key = m.resolve("acme", QA, LABELS)[0]
    assert m.engine(key)._prod is not None
    m.close()
    return key


def test_a_student_from_another_encoder_is_not_served(tmp_path, teacher, world):
    """5: nothing compared a production student's encoder with the current one. After a change to another dimension
    (small -> base, or images = "auto" on a host that lost its GPU) every request failed in the student and was
    forwarded unrecorded, so the task never learned again; with the same dimension the old student answered from
    embeddings it wasn't trained on."""
    key = _trained(tmp_path, teacher, world, HashEncoder(dim=512))
    for other in (HashEncoder(dim=256), HashEncoder(dim=512, seed=1)):        # other dimension; same, other id
        m = TaskManager(tmp_path, teacher, other, _cfg(), admission=Admission(1), target_agreement=0.9,
                        janitor_interval_s=3600)
        e = m.engine(key)
        assert e._prod is None and e.mode == "teacher_only"
        before = e.store.max_id()
        res = m.classify("acme", QA, LABELS, [t for t, _ in world.sample(50)])
        assert all(r.source == "teacher" for r in res)
        e.store.flush()
        assert e.store.max_id() - before == 50                                 # recorded: it learns again
        assert e.store.last_event(("encoder_changed",))["encoder"] == other.id
        m.close()


def test_a_task_learns_again_after_an_encoder_change(tmp_path, teacher, world):
    """5: ... and the rows it records under the new encoder train a new student."""
    key = _trained(tmp_path, teacher, world, HashEncoder(dim=512))
    m = TaskManager(tmp_path, teacher, HashEncoder(dim=256), _cfg(), admission=Admission(1), target_agreement=0.9,
                    janitor_interval_s=3600)
    for _ in range(30):
        if any(r.source.startswith("student") for r in m.classify("acme", QA, LABELS, [t for t, _ in world.sample(200)])):
            break
    prod = m.engine(key)._prod
    assert prod is not None and prod.meta["encoder"] == HashEncoder(dim=256).id
    m.close()
