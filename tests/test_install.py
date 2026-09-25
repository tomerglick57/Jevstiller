"""What a plain `pip install jevstiller` gives you (0.3.1)."""
import sys
from importlib.metadata import requires

import pytest


def test_the_plain_install_includes_the_proxy():
    core = {r.split(";")[0].split(">")[0].split("=")[0].strip() for r in requires("jevstiller") if "extra ==" not in r}
    assert {"starlette", "uvicorn", "httpx", "anyio", "h11"} <= core


def test_serve_without_the_encoder_extra_says_what_to_install(tmp_path, monkeypatch):
    import uvicorn

    from jevstiller import _cli as cli
    monkeypatch.setitem(sys.modules, "onnxruntime", None)          # as if jevstiller[onnx] were not installed
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--data-dir", str(tmp_path), "--encoder", "small", "--backend", "onnx"])
    assert 'pip install "jevstiller[onnx]"' in str(e.value) and "hash" in str(e.value)
