"""What a plain `pip install jevstiller` gives you (0.3.1)."""
import sys
from importlib.metadata import requires

import pytest


def test_the_plain_install_includes_the_proxy():
    core = {r.split(";")[0].split(">")[0].split("=")[0].strip() for r in requires("jevstiller") if "extra ==" not in r}
    assert {"starlette", "uvicorn", "httpx", "anyio", "h11", "onnxruntime", "tokenizers", "huggingface-hub",
            "typesafe-sdk"} <= core


def test_serve_without_the_encoder_extra_says_what_to_install(tmp_path, monkeypatch):
    import uvicorn

    from jevstiller import _cli as cli
    monkeypatch.setitem(sys.modules, "onnxruntime", None)          # a platform without ONNX Runtime
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--data-dir", str(tmp_path), "--encoder", "small", "--backend", "onnx"])
    assert 'pip install "jevstiller[gpu]"' in str(e.value) and "--encoder hash" in str(e.value)


def test_admin_cli_uses_the_servers_own_settings(tmp_path, monkeypatch):
    """Next to the server (`docker exec ... jevstiller admin tasks`), no flags are needed: the token and the port come
    from the same settings `jevstiller serve` reads."""
    import httpx

    from jevstiller import _cli as cli
    (tmp_path / "admin-token").write_text("a-long-enough-admin-token\n")
    (tmp_path / "c.toml").write_text(f'[server]\nport = 9999\nadmin_token_file = "{tmp_path / "admin-token"}"\n')
    monkeypatch.setenv("JEVSTILLER_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.delenv("JEVSTILLER_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("JEVSTILLER_ADMIN_URL", raising=False)
    calls = []

    def request(method, url, headers=None, **kw):
        calls.append((url, headers["Authorization"]))
        return httpx.Response(200, json=[])
    monkeypatch.setattr(httpx, "request", request)
    cli.main(["admin", "tasks"])
    assert calls == [("http://127.0.0.1:9999/jevstiller/v1/tasks", "Bearer a-long-enough-admin-token")]
