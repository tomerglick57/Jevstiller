"""P5.1: config file, environment and flags, with precedence and validation."""
import math

import pytest

from jevstiller._settings import SECTIONS, load

TOML = """
[server]
port = 9000
admin_token_file = "{secret}"
[proxy]
tenancy = "per_key"
allow_networks = ["10.0.0.0/8"]
tenants_file = "{tenants}"
[manager]
target_agreement = 0.99
text_retention_days = 30
[encoder]
spec = "base"
[engine]
audit_rate = 0.05
[tasks."abc"]
target_agreement = 0.995
mode = "teacher_only"
"""


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "secret").write_text("adm1n\n")
    (tmp_path / "tenants.json").write_text('{"%s": "acme"}' % ("a" * 32))
    p = tmp_path / "j.toml"
    p.write_text(TOML.format(secret=tmp_path / "secret", tenants=tmp_path / "tenants.json"))
    return p


def test_file_env_cli_precedence(cfg):
    s = load(cfg, environ={})
    assert (s.port, s.tenancy, s.target_agreement, s.encoder) == (9000, "per_key", 0.99, "base")
    assert s.admin_token == "adm1n" and s.tenants == {"a" * 32: "acme"} and s.allow_networks == ["10.0.0.0/8"]
    assert s.engine == {"audit_rate": 0.05} and s.tasks["abc"]["mode"] == "teacher_only"
    s = load(cfg, environ={"JEVSTILLER_PORT": "9100", "JEVSTILLER_STORE_TEXT": "false",
                           "JEVSTILLER_ALLOW_NETWORKS": "10.1.0.0/16, 192.168.0.0/16"})
    assert s.port == 9100 and s.store_text is False and s.allow_networks == ["10.1.0.0/16", "192.168.0.0/16"]
    s = load(cfg, cli={"port": 9200, "tenancy": None}, environ={"JEVSTILLER_PORT": "9100"})
    assert s.port == 9200 and s.tenancy == "per_key"               # None flags don't override


def test_config_path_from_env(cfg):
    assert load(environ={"JEVSTILLER_CONFIG": str(cfg)}).port == 9000


def test_secrets_are_redacted(cfg):
    d = load(cfg, environ={"JEVSTILLER_ACCESS_TOKEN": "tok"}).redacted()
    assert d["admin_token"] == "***" and d["access_token"] == "***"


@pytest.mark.parametrize("toml,err", [
    ("[server]\nprot = 1\n", "unknown key"),
    ("[proxxy]\nx = 1\n", "unknown section"),
    ("[proxy]\ntenancy = 'everyone'\n", "tenancy"),
    ("[engine]\naudit_rat = 0.1\n", "engine"),
    ("[engine]\nmode = 'hedge'\n", "mode"),
    ("[tasks.k]\ntarget = 0.9\n", "unknown keys"),
    ("[server]\nssl_certfile = 'a.pem'\n", "ssl_keyfile"),
    ("[proxy]\nmax_encoder_wait_ms = -1\n", "max_encoder_wait_ms"),
    ("[proxy]\nmax_image_body_mb = inf\n", "max_image_body_mb"),
    ("[proxy]\nmax_image_body_mb = 0\n", "max_image_body_mb"),
    ("[proxy]\nmax_large_bodies = 0\n", "max_large_bodies"),
])
def test_invalid_configs_are_rejected(tmp_path, toml, err):
    p = tmp_path / "bad.toml"
    p.write_text(toml)
    with pytest.raises(ValueError, match=err):
        load(p, environ={})


def test_bad_env_values(tmp_path):
    with pytest.raises(ValueError):
        load(environ={"JEVSTILLER_STORE_TEXT": "maybe"})
    with pytest.raises(ValueError):
        load(environ={"JEVSTILLER_PORT": "eighty"})


def test_bad_tenants_file(tmp_path):
    (tmp_path / "t.json").write_text('["not", "a", "map"]')
    with pytest.raises(ValueError, match="key hash"):
        load(cli={"tenants_file": str(tmp_path / "t.json")}, environ={})


def test_tenant_map_keys_must_be_key_hashes(tmp_path):
    (tmp_path / "t.json").write_text('{"tsk_raw_key_by_mistake": "acme"}')
    with pytest.raises(ValueError, match="key hash"):
        load(cli={"tenants_file": str(tmp_path / "t.json")}, environ={})
    with pytest.raises(ValueError, match="network"):
        load(cli={"allow_networks": ["10.0.0.0/33"]}, environ={})


def test_encoder_wait_setting(tmp_path):
    p = tmp_path / "j.toml"
    p.write_text("[proxy]\nmax_encoder_wait_ms = 50\n")
    assert load(p, environ={}).max_encoder_wait_ms == 50
    assert load(p, environ={"JEVSTILLER_MAX_ENCODER_WAIT_MS": "0"}).max_encoder_wait_ms == 0
    assert load(environ={}).max_encoder_wait_ms == 200


NON_FINITE_FLOATS = ("max_encoder_wait_ms", "admit_window_s", "max_body_mb", "upstream_timeout_s")


@pytest.mark.parametrize("name", NON_FINITE_FLOATS)
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_float_from_file_is_rejected(tmp_path, name, bad):
    section = next(s for s, keys in SECTIONS.items() if name in keys)
    p = tmp_path / "j.toml"
    p.write_text(f"[{section}]\n{name} = {bad}\n")
    with pytest.raises(ValueError, match=f"{name} must be a finite number"):
        load(p, environ={})


@pytest.mark.parametrize("name", NON_FINITE_FLOATS)
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_float_from_env_is_rejected(name, bad):
    with pytest.raises(ValueError, match=f"{name} must be a finite number"):
        load(environ={f"JEVSTILLER_{name.upper()}": bad})


@pytest.mark.parametrize("name", NON_FINITE_FLOATS)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_float_from_cli_is_rejected(name, bad):
    with pytest.raises(ValueError, match=f"{name} must be a finite number"):
        load(cli={name: bad}, environ={})


def test_finite_floats_still_load():
    s = load(cli={"max_body_mb": 8.0, "upstream_timeout_s": 2.5}, environ={"JEVSTILLER_ADMIT_WINDOW_S": "60"})
    assert (s.max_body_mb, s.upstream_timeout_s, s.admit_window_s) == (8.0, 2.5, 60.0)
