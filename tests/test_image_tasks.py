"""Image questions as tasks (Jev 1.14, experimental): what `_images.read` takes and refuses, and the whole path through
the proxy with an image encoder: learned from Jev's answers, answered locally, no image bytes stored."""
import base64
import io
import json
import sqlite3
import uuid

import httpx
import numpy as np
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from test_proxy import GOOD, serve

from jevstiller import Admission, Config, HashEncoder, TaskManager, _images
from jevstiller.server import KeyRegistry, ProxySettings, create_app

Image = pytest.importorskip("PIL.Image")

COLOURS = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 220)}
QUESTION = {"type": "choice", "instructions": "Which colour is this tile?", "criteria": {c: "" for c in COLOURS}}
HEADER = {"TypeSafe-Danger-Zone-Images": "true"}


def png(colour, size=(300, 300), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    px = np.clip(np.array(COLOURS[colour]) + rng.normal(0, 20, (*size[::-1], 3)), 0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(px).save(out, format="PNG")
    return out.getvalue()


def request_body(image: bytes, state=None, questions=None, ref=None, extra_parts=(), asset=None):
    """An image request as typesafe-sdk 0.7.5 encodes it: multipart, the JSON first, then the file part."""
    ref = ref or f"<image:{uuid.uuid4().hex}>"
    req = {"state": ref if state is None else state(ref), "model": "jev-latest",
           "questions": questions or {"label": QUESTION}, "assets": {ref: asset or {"type": "image", "part": "asset-0"}}}
    b = uuid.uuid4().hex
    parts = [("request", "application/json", json.dumps(req).encode()), ("asset-0", "image/png", image), *extra_parts]
    out = b""
    for name, ct, content in parts:
        fn = "" if name == "request" else f'; filename="{name}"'
        out += f'--{b}\r\nContent-Disposition: form-data; name="{name}"{fn}\r\nContent-Type: {ct}\r\n\r\n'.encode()
        out += content + b"\r\n"
    return f"multipart/form-data; boundary={b}", out + f"--{b}--\r\n".encode()


# ---- what `read` takes and refuses ----------------------------------------------------------------------------

def test_a_request_like_the_sdks_is_read_with_its_image_named_by_content():
    img = png("red")
    ct, body = request_body(img)
    r = _images.read(ct, body, True)
    assert isinstance(r, _images.ImageRequest) and r.image == img
    assert r.req["state"] == f"<image:sha256:{r.digest[:32]}>" == r.state_text and "assets" not in r.req
    ct2, body2 = request_body(img)                      # another random reference, the same image: the same row
    assert _images.read(ct2, body2, True).state_text == r.state_text


def test_one_field_of_an_object_state_can_be_the_image():
    ct, body = request_body(png("blue"), state=lambda ref: {"photo": ref, "note": "from the warehouse"})
    r = _images.read(ct, body, True)
    assert isinstance(r, _images.ImageRequest)
    assert r.req["state"] == {"photo": f"<image:sha256:{r.digest[:32]}>", "note": "from the warehouse"}


def test_base64_in_json_is_read_too():
    img = png("green")
    ref = "<image:abc>"
    body = json.dumps({"state": ref, "model": "jev-latest", "questions": {"label": QUESTION},
                       "assets": {ref: {"format": "png", "data_base64": base64.b64encode(img).decode()}}}).encode()
    r = _images.read("application/json", body, True)
    assert isinstance(r, _images.ImageRequest) and r.image == img


@pytest.mark.parametrize("case,expected", [
    ("no header", _images.INVALID),
    ("too small", _images.INVALID),                     # Jev's minimum is 280 px a side
    ("not an image", _images.INVALID),
    ("bad part type", _images.INVALID),
    ("unclaimed part", _images.INVALID),
    ("unreferenced", _images.INVALID),
    ("bad base64", _images.INVALID),
    ("two images", _images.SCOPE),
    ("also in the question", _images.SCOPE),
    ("also in another field", _images.SCOPE),
    ("inside a sentence", _images.SCOPE),
    ("not multipart", _images.UNREADABLE),
])
def test_what_is_forwarded_instead(case, expected):
    img = png("red")
    header = case != "no header"
    if case == "too small":
        ct, body = request_body(png("red", size=(100, 100)))
    elif case == "not an image":
        ct, body = request_body(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    elif case == "bad part type":
        ct, body = request_body(img)
        body = body.replace(b"Content-Type: image/png", b"Content-Type: text/html")
    elif case == "unclaimed part":
        ct, body = request_body(img, extra_parts=[("asset-1", "image/png", img)])
    elif case == "unreferenced":
        ct, body = request_body(img, state=lambda ref: "no image here")
    elif case == "bad base64":
        ref = "<image:x>"
        body = json.dumps({"state": ref, "model": "m", "questions": {"label": QUESTION},
                           "assets": {ref: {"format": "png", "data_base64": "not base64!"}}}).encode()
        ct = "application/json"
    elif case == "two images":
        b = uuid.uuid4().hex
        req = {"state": {"a": "<image:1>", "b": "<image:2>"}, "model": "m", "questions": {"label": QUESTION},
               "assets": {"<image:1>": {"part": "p1"}, "<image:2>": {"part": "p2"}}}
        body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{n}"\r\nContent-Type: {t}\r\n\r\n'.encode()
                        + c + b"\r\n" for n, t, c in (("request", "application/json", json.dumps(req).encode()),
                                                       ("p1", "image/png", img), ("p2", "image/png", img)))
        body += f"--{b}--\r\n".encode()
        ct = f"multipart/form-data; boundary={b}"
    elif case == "also in the question":
        ref = "<image:q>"
        q = {**QUESTION, "instructions": f"Which colour is {ref}?"}
        ct, body = request_body(img, ref=ref, questions={"label": q})
    elif case == "also in another field":
        ct, body = request_body(img, state=lambda ref: {"photo": ref, "copy": ref})
    elif case == "inside a sentence":
        ct, body = request_body(img, state=lambda ref: f"Look at {ref} closely")
    else:
        ct, body = "multipart/form-data; boundary=zzz", b"garbage"
    assert _images.read(ct, body, header) == expected


# ---- the whole path through the proxy ---------------------------------------------------------------------------

class ColourEncoder:
    """A stand-in image encoder: the image's mean colour, so the colour question is learnable."""
    id, dim = "test-colour:1", 3

    def encode(self, items):
        out = []
        for data in items:
            assert isinstance(data, bytes)
            px = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)
            out.append(px / np.linalg.norm(px))
        return np.array(out, dtype=np.float32)


class ColourJev:
    """Jev for image requests: answers the colour question from the image it was sent."""

    def __init__(self):
        self.calls = 0

    async def system_one(self, request: Request):
        self.calls += 1
        body, ct = await request.body(), request.headers.get("content-type", "")
        if ct.startswith("multipart/"):
            assert request.headers.get("typesafe-danger-zone-images") == "true"
            parts = _images._multipart(ct, body)              # any image request, in the scope of v1 or not
            image = next(p.data for p in parts if p.name != "request")
            mean = np.asarray(Image.open(io.BytesIO(image)).convert("RGB")).reshape(-1, 3).mean(0)
            label, questions = list(COLOURS)[int(np.argmax(mean))], json.loads(parts[0].data)["questions"]
        else:                                           # a text request: the text task in one test
            label, questions = "red", json.loads(body)["questions"]
        probs = {c: (0.96 if c == label else 0.02) for c in COLOURS}
        return JSONResponse({"model": "jev-1.14.0", "usage": {"input_tokens": 300, "output_tokens": 1},
                             "answers": {name: {"type": "choice", "choice": label, "confidence": 0.94,
                                                "probabilities": probs} for name in questions}},
                            headers={"x-typesafe-request-id": f"req_{self.calls}"})

    def app(self):
        return Starlette(routes=[Route("/v1/systemone", self.system_one, methods=["POST"])])


@pytest.fixture
def image_stack(tmp_path):
    jev = ColourJev()
    cfg = Config(audit_rate=0.05, min_train_samples=150, min_samples_per_class=10, min_calib_samples=60,
                 min_new_samples=10**9, shadow_min_samples=60, seed=0, training="inline")
    with serve(jev.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=64), cfg, target_agreement=0.95,
                        admission=Admission(min_requests=1), janitor_interval_s=3600, image_encoder=ColourEncoder())
        app = create_app(m, ProxySettings(upstream=upstream), KeyRegistry(b"salt", 3600))
        with serve(app) as proxy:
            yield proxy, jev, m, app
        m.close()


def ask(proxy, image, **kw):
    ct, body = request_body(image, **kw)
    return httpx.post(f"{proxy}/v1/systemone", content=body, timeout=30,
                      headers={"authorization": f"Bearer {GOOD}", "content-type": ct, **HEADER})


def test_an_image_question_is_learned_and_answered_locally(image_stack, tmp_path):
    proxy, jev, m, app = image_stack
    colours = list(COLOURS)
    sources = []
    for i in range(700):
        r = ask(proxy, png(colours[i % 3], seed=i))
        assert r.status_code == 200, r.text
        assert r.json()["answers"]["label"]["choice"] == colours[i % 3]
        sources.append(r.headers["x-jevstiller-source"])
    assert sources[0] == "upstream" and sources[-50:].count("local") > 30, sources[-50:]
    local = next(r for r in (ask(proxy, png("blue", seed=999)) for _ in range(20))
                 if r.headers["x-jevstiller-source"] == "local")
    body = local.json()                                 # Jev's own shape
    assert set(body) == {"model", "answers", "usage"} and body["answers"]["label"]["type"] == "choice"
    [info] = m.tasks()
    assert info.modality == "image"
    store = tmp_path / "tasks" / info.key / "samples.sqlite"
    rows = sqlite3.connect(store).execute("SELECT text FROM samples LIMIT 5").fetchall()
    assert all(t.startswith("<image:sha256:") for (t,) in rows), rows
    assert png("red", seed=0)[100:200] not in store.read_bytes()      # no image bytes stored
    metrics = app.state.metrics.render()
    assert "jevstiller_unrouted_total" not in metrics or 'reason="images"' not in metrics


def test_the_same_question_about_text_is_another_task(image_stack):
    proxy, jev, m, _ = image_stack
    ask(proxy, png("red"))
    httpx.post(f"{proxy}/v1/systemone", timeout=10, headers={"authorization": f"Bearer {GOOD}"},
               json={"state": "a red tile", "model": "jev-latest", "questions": {"label": QUESTION}})
    assert sorted(i.modality for i in m.tasks()) == ["image", "text"]


def test_requests_outside_v1_still_go_to_jev(image_stack):
    proxy, jev, m, app = image_stack
    r = ask(proxy, png("red"), state=lambda ref: f"Look at {ref}")
    assert r.status_code == 200 and r.headers["x-jevstiller-source"] == "upstream"
    assert 'jevstiller_unrouted_total{reason="image_scope"} 1' in app.state.metrics.render()
    assert m.tasks() == []


def test_a_task_admitted_after_several_requests_is_an_image_task(tmp_path):
    """With the default admission (here 5 requests), a known key's new question is admitted by `observe` once Jev
    has answered it. That path made it a text task: its engine used the text encoder, and every training failed
    (embeddings of the image encoder's dimension)."""
    jev = ColourJev()
    with serve(jev.app()) as upstream:
        m = TaskManager(tmp_path, None, HashEncoder(dim=64), Config(training="manual"), target_agreement=0.95,
                        admission=Admission(min_requests=5), janitor_interval_s=3600, image_encoder=ColourEncoder())
        with serve(create_app(m, ProxySettings(upstream=upstream), KeyRegistry(b"salt", 3600))) as proxy:
            for i in range(10):
                assert ask(proxy, png("red", seed=i)).status_code == 200
        [info] = m.tasks()
        assert info.modality == "image"
        engine = m.engine(info.key)
        assert engine.encoder.dim == ColourEncoder.dim
        assert engine.store.counts(engine.task.version)["teacher_calls"] > 0
        m.close()


def test_an_encoder_without_a_pooled_output_embeds_with_its_first_token():
    """DINOv2's ONNX export returns every token (n, tokens, dim): one vector per image is its first token."""
    from jevstiller.encoders._image import OnnxImageEncoder
    enc = object.__new__(OnnxImageEncoder)               # no model download: a stand-in session
    enc.batch_size, enc.dim, enc.input_name, enc.output_name = 4, 3, "pixel_values", "last_hidden_state"
    enc.preprocess = lambda data: np.zeros((3, 2, 2), np.float32)

    class Session:
        def run(self, names, feed):
            n = len(feed["pixel_values"])
            out = np.zeros((n, 5, 3), np.float32)
            out[:, 0] = [3.0, 4.0, 0.0]                  # the first token
            out[:, 1:] = 9.0
            return [out]
    enc.sess = Session()
    X = enc.encode_images([b"a", b"b"])
    assert X.shape == (2, 3) and np.allclose(X, [[0.6, 0.8, 0.0]] * 2)


def test_preprocessing_follows_the_models_config():
    """Both image encoders preprocess from preprocessor_config.json in Pillow, with no transformers processor."""
    from jevstiller.encoders._image import _Preprocess
    out = io.BytesIO()
    Image.new("RGB", (600, 300), (255, 0, 51)).save(out, format="PNG")
    flat = out.getvalue()
    siglip = _Preprocess({"size": {"height": 384, "width": 384}, "resample": 3, "rescale_factor": 1 / 255,
                          "image_mean": [0.5] * 3, "image_std": [0.5] * 3})
    x = siglip(flat)                                     # straight to 384 x 384, then (x/255 - 0.5) / 0.5
    assert x.shape == (3, 384, 384) and np.allclose(x[:, 100, 100], [1.0, -1.0, -0.6], atol=1e-5)
    clip = _Preprocess({"size": {"shortest_edge": 224}, "do_center_crop": True,
                        "crop_size": {"height": 224, "width": 224}, "image_mean": [0, 0, 0], "image_std": [1, 1, 1]})
    assert clip(flat).shape == (3, 224, 224)             # shorter side to 224 (448 x 224), then the centre
    assert siglip.hash != clip.hash                      # the preprocessing is part of an encoder's id


@pytest.mark.parametrize("spec,backend,cuda,torch,expected", [
    ("auto", "auto", True, True, ("siglip-so400m", "torch")),      # a GPU: the best overall model
    ("auto", "auto", False, True, ("siglip", "onnx")),             # CPU: SigLIP B/16 through ONNX
    ("auto", "auto", True, False, ("siglip", "onnx")),             # CUDA without PyTorch installed
    ("auto", "onnx", True, True, ("siglip", "onnx")),              # ONNX asked for: a model that has an export
    ("siglip", "auto", True, True, ("siglip", "torch")),
    ("siglip", "auto", False, False, ("siglip", "onnx")),
    ("siglip-so400m", "auto", False, True, ("siglip-so400m", "torch")),   # named: runs on CPU, slowly
    ("clip", "auto", True, True, ("clip", "onnx")),                # no PyTorch checkpoint: ONNX, even on a GPU
])
def test_which_image_encoder(spec, backend, cuda, torch, expected):
    from jevstiller.encoders import image_choice
    name, chosen, _ = image_choice(spec, backend, cuda, torch)
    assert (name, chosen) == expected


def test_impossible_image_encoders_are_refused():
    from jevstiller.encoders import image_choice
    with pytest.raises(ValueError, match="no ONNX export"):
        image_choice("siglip-so400m", "onnx")
    with pytest.raises(ValueError, match="no PyTorch"):
        image_choice("clip", "torch", True, True)
    with pytest.raises(ValueError, match="unknown image encoder"):
        image_choice("dinov3")

