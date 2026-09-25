"""Settings for `jevstiller serve`: one TOML file, `JEVSTILLER_*` environment variables, and CLI flags.

Precedence, lowest to highest: built-in defaults < config file < environment < command-line flags.

```toml
[server]
port = 8080
data_dir = "/data"
admin_token_file = "/run/secrets/admin-token"

[proxy]
tenancy = "per_key"
allow_networks = ["10.0.0.0/8"]

[manager]
target_agreement = 0.98
text_retention_days = 30

[encoder]
spec = "small"

[engine]                 # any jevstiller.Config field
audit_rate = 0.03

[tasks."3f1c...e2"]      # per-task overrides, by task key
target_agreement = 0.99
mode = "teacher_only"
```

Unknown keys are errors (a typo must not silently do nothing), and so are values of the wrong type (a quoted
`"false"`), an empty secret, secret file or access-control environment variable, and a retention period <= 0: a
security or privacy setting either takes effect or stops startup. Secrets (`admin_token`, `access_token`) can be
given directly or as `*_file` paths; they, and credentials in the `upstream` URL, are redacted when settings are
printed or logged.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SECRETS = ("admin_token", "access_token")
# present-but-empty is an error for these: an unset deployment variable must not switch a control off silently
NO_EMPTY = ("admin_token", "admin_token_file", "access_token", "access_token_file", "allow_networks",
            "trust_forwarded_for", "tenants_file")


def redact_url(url: str) -> str:
    """`url` with any user:password@ replaced by ***@."""
    from urllib.parse import urlsplit, urlunsplit
    try:
        u = urlsplit(url)
    except ValueError:
        return "***"
    if u.username is None and u.password is None:
        return url
    return urlunsplit(u._replace(netloc="***@" + u.netloc.rpartition("@")[2]))


def _type_problem(name: str, value: Any, t: str) -> str | None:
    """Why `value` doesn't fit the annotation `t` of field `name`, or None."""
    opts = [o.strip() for o in str(t).split("|")]
    if value is None:
        return None if "None" in opts else f"{name} must be set"
    for o in opts:
        if (o == "bool" and isinstance(value, bool)
                or o == "int" and isinstance(value, int) and not isinstance(value, bool)
                or o == "float" and isinstance(value, (int, float)) and not isinstance(value, bool)
                or o == "str" and isinstance(value, str)
                or o.startswith("list") and isinstance(value, list) and all(isinstance(x, str) for x in value)
                or o.startswith("dict") and isinstance(value, dict)):
            return None
    want = " or ".join(o for o in opts if o != "None")
    shown = "" if name in (*SECRETS, "tenants") else f" {value!r}"   # may hold a key pasted by mistake
    return f"{name} must be {want}, got {type(value).__name__}{shown}"


@dataclass
class ServeSettings:
    # [server]
    host: str = "0.0.0.0"
    port: int = 8080
    data_dir: str = "./jevstiller-data"
    log_level: str = "info"
    log_format: str = "text"                    # text | json
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    admin_token: str | None = None              # enables the admin API (/jevstiller/v1/*) and protects /metrics
    admin_token_file: str | None = None
    metrics_public: bool = False                # serve /metrics without the admin token
    # [proxy]
    upstream: str = "https://api.typesafe.ai"
    upstream_timeout_s: float = 9.0
    max_upstream_inflight: int = 256
    tenancy: str = "shared"                     # shared | per_key
    key_ttl_s: float = 3600.0
    access_token: str | None = None
    access_token_file: str | None = None
    allow_networks: list[str] = field(default_factory=list)
    trust_forwarded_for: list[str] = field(default_factory=list)   # proxies allowed to set X-Forwarded-For
    max_body_mb: float = 4.0
    max_questions: int = 32                     # per request; more are forwarded without routing
    max_encoder_wait_ms: float = 200.0          # encoder backed up beyond this: forward instead (0 = never)
    tenants: dict[str, str] = field(default_factory=dict)          # key hash -> tenant
    tenants_file: str | None = None             # JSON {"<key hash>": "<tenant>"}, merged over `tenants`
    price_per_mtok: float = 0.042
    # [manager]
    target_agreement: float = 0.98
    max_loaded: int = 64
    max_memory_mb: float | None = None
    admit_after: int = 50
    admit_window_s: float = 86400.0
    max_tasks: int | None = 10000
    max_tasks_per_tenant: int | None = 1000
    max_new_tasks_per_key: int | None = 100     # tasks one API key may create per admit_window_s
    idle_ttl_days: float | None = None
    text_retention_days: float | None = None
    store_text: bool = True
    train_workers: int = 2
    blas_threads: int | None = 1
    # [encoder]
    encoder: str = "small"                      # small | base | large | hash | onnx:<repo> | torch:<model>
    backend: str = "auto"
    device: str = "auto"
    # [engine], [tasks]
    engine: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> ServeSettings:
        import math

        from ._task import Config
        for f in fields(self):
            if (problem := _type_problem(f.name, getattr(self, f.name), f.type)) is not None:
                raise ValueError(problem)
        cfg_types = {f.name: f.type for f in fields(Config)}
        for k, v in self.engine.items():
            if k in cfg_types and (problem := _type_problem(k, v, cfg_types[k])) is not None:
                raise ValueError(f"[engine]: {problem}")
        if "store_text" in self.engine:
            # [engine] takes any Config field, store_text included; [manager] store_text used to override it
            # with its default (True). Either one saying False keeps text out of the store.
            self.store_text = self.store_text and self.engine["store_text"]
            self.engine = {k: v for k, v in self.engine.items() if k != "store_text"}
        for name in NO_EMPTY:
            v = getattr(self, name)
            if isinstance(v, str) and not v.strip():
                raise ValueError(f"{name} is empty; remove it to leave it unset")
        for name in ("text_retention_days", "idle_ttl_days"):
            v = getattr(self, name)
            if v is not None and not (0 < v < math.inf):
                raise ValueError(f"{name} must be a number of days > 0 (or unset: keep forever)")
        if not 0 <= self.key_ttl_s < math.inf:
            raise ValueError("key_ttl_s must be a finite number of seconds >= 0")
        if self.tenancy not in ("shared", "per_key"):
            raise ValueError(f"tenancy must be 'shared' or 'per_key', got {self.tenancy!r}")
        if self.log_format not in ("text", "json"):
            raise ValueError(f"log_format must be 'text' or 'json', got {self.log_format!r}")
        if not 0.5 <= self.target_agreement < 1:
            raise ValueError("target_agreement must be in [0.5, 1)")
        if bool(self.ssl_certfile) != bool(self.ssl_keyfile):
            raise ValueError("ssl_certfile and ssl_keyfile go together")
        try:
            Config(**self.engine)
        except TypeError as e:
            raise ValueError(f"[engine]: {e}") from None
        for key, o in self.tasks.items():
            unknown = set(o) - {"target_agreement", "mode"}
            if unknown:
                raise ValueError(f"[tasks.{key!r}]: unknown keys {sorted(unknown)}")
        for name in ("max_tasks", "max_tasks_per_tenant", "max_new_tasks_per_key"):
            if (v := getattr(self, name)) is not None and v < 0:
                raise ValueError(f"{name} must be >= 0 (or none: no limit)")
        for name in ("max_loaded", "admit_after", "train_workers", "max_questions", "max_encoder_wait_ms", "port"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        import ipaddress
        import re
        for net in self.allow_networks:
            try:
                ipaddress.ip_network(net, strict=False)
            except ValueError:
                raise ValueError(f"not a network address: {net!r}") from None
        for net in self.trust_forwarded_for:
            # strictly: uvicorn silently ignores an entry with host bits set (10.0.0.5/24), and would then keep
            # the load balancer's address as every caller's
            try:
                ipaddress.ip_network(net, strict=True)
            except ValueError:
                raise ValueError(f"trust_forwarded_for: not an address or network (no host bits set, e.g. "
                                 f"10.0.0.0/24 or 10.0.0.5): {net!r}") from None
        if not all(isinstance(v, str) and v.strip() for v in self.tenants.values()):
            raise ValueError("tenants: every tenant name must be a non-empty string")
        bad = [i for i, k in enumerate(self.tenants, 1) if not re.fullmatch(r"[0-9a-f]{32}", k)]
        if bad:
            # name the entry by position only: the likely mistakes are a raw API key in place of its hash, on
            # either side of the map, so neither side may be printed (security audits runs 2 and 3)
            raise ValueError(f"tenants: {len(bad)} of {len(self.tenants)} entries don't map a key hash from "
                             f"`jevstiller key-hash` (32 lowercase hex) to a tenant name; the first is entry #{bad[0]}")
        return self

    def resolve_secrets(self) -> ServeSettings:
        """Read `*_file` secrets and the tenants file into the settings."""
        for name in SECRETS:
            path = getattr(self, f"{name}_file")
            if path and not getattr(self, name):
                try:
                    value = Path(path).read_text().strip()
                except OSError as e:
                    # not the path itself: a secret pasted where its file's path belongs would be printed
                    raise ValueError(f"{name}_file: can't read that file ({e.strerror or type(e).__name__}); it "
                                     f"must be the path of a file that holds the {name}") from None
                if not value:
                    raise ValueError(f"{name}_file: the file is empty")
                setattr(self, name, value)
        if self.tenants_file:
            import json
            data = json.loads(Path(self.tenants_file).read_text())
            if not (isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) and v
                                                   for k, v in data.items())):
                raise ValueError(f"{self.tenants_file}: expected a JSON object of key hash -> tenant name")
            self.tenants = {**self.tenants, **data}
        return self

    def redacted(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for name in SECRETS:
            if d.get(name):
                d[name] = "***"
        d["upstream"] = redact_url(d["upstream"])
        return d


SECTIONS = {
    "server": ("host", "port", "data_dir", "log_level", "log_format", "ssl_certfile", "ssl_keyfile", "admin_token",
               "admin_token_file", "metrics_public"),
    "proxy": ("upstream", "upstream_timeout_s", "max_upstream_inflight", "tenancy", "key_ttl_s", "access_token",
              "access_token_file", "allow_networks", "trust_forwarded_for", "max_body_mb", "max_questions",
              "max_encoder_wait_ms", "tenants", "tenants_file", "price_per_mtok"),
    "manager": ("target_agreement", "max_loaded", "max_memory_mb", "admit_after", "admit_window_s", "max_tasks",
                "max_tasks_per_tenant", "max_new_tasks_per_key", "idle_ttl_days", "text_retention_days", "store_text",
                "train_workers",
                "blas_threads"),
    "encoder": ("spec", "backend", "device"),
}
_FIELDS = {f.name: f for f in fields(ServeSettings)}


def _coerce(name: str, raw: str) -> Any:
    """Parse an environment string into the type of settings field `name`."""
    t = str(_FIELDS[name].type)
    if not raw.strip():
        # a set-but-blank variable (an unset `${VAR}` in a template) must not silently reset a setting
        raise ValueError(f"JEVSTILLER_{name.upper()} is set but empty; unset it (or set 'none' to clear the "
                         f"config file's value)")
    if raw.strip().lower() in ("none", "null") and "None" in t:
        return None
    if raw.strip().lower() in ("none", "null") and t.startswith("list"):
        return []
    if t.startswith("bool"):
        if raw.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if raw.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"JEVSTILLER_{name.upper()}: expected a boolean, got {raw!r}")
    if t.startswith("int"):
        return int(raw)
    if t.startswith("float"):
        return float(raw)
    if t.startswith("list"):
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if not items:                                   # "," or " , ": as blank as ""
            raise ValueError(f"JEVSTILLER_{name.upper()} has no values; unset it (or set 'none' to clear the "
                             f"config file's value)")
        return items
    return raw


def from_file(path: str | Path) -> dict[str, Any]:
    """Flat {field: value} from a TOML file; unknown sections or keys raise ValueError."""
    data = tomllib.loads(Path(path).read_text())
    out: dict[str, Any] = {}
    for section, values in data.items():
        if section in ("engine", "tasks"):
            if not isinstance(values, dict):
                raise ValueError(f"{path}: [{section}] must be a table")
            out[section] = values
            continue
        if section not in SECTIONS:
            raise ValueError(f"{path}: unknown section [{section}] (known: {sorted([*SECTIONS, 'engine', 'tasks'])})")
        for key, value in values.items():
            if key not in SECTIONS[section]:
                raise ValueError(f"{path}: unknown key {key!r} in [{section}] (known: {sorted(SECTIONS[section])})")
            out["encoder" if (section, key) == ("encoder", "spec") else key] = value
    return out


def from_env(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    out = {}
    for name in _FIELDS:
        if name in ("engine", "tasks", "tenants"):
            continue
        var = f"JEVSTILLER_{name.upper()}"
        if var in env:
            out[name] = _coerce(name, env[var])
    return out


def load(config_path: str | Path | None = None, cli: dict[str, Any] | None = None,
         environ: dict[str, str] | None = None) -> ServeSettings:
    """Defaults < file (`config_path`, or JEVSTILLER_CONFIG) < environment < `cli` (None values ignored)."""
    env = os.environ if environ is None else environ
    merged: dict[str, Any] = {}
    path = config_path if config_path is not None else env.get("JEVSTILLER_CONFIG")
    if path is not None and not str(path).strip():
        # an empty path used to mean "no file", silently dropping every setting in it, access controls included
        raise ValueError("the config file path (--config or JEVSTILLER_CONFIG) is empty; unset it to run without one")
    if path:
        merged.update(from_file(path))
    merged.update(from_env(env))
    merged.update({k: v for k, v in (cli or {}).items() if v is not None})
    unknown = set(merged) - set(_FIELDS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return ServeSettings(**merged).resolve_secrets().validate()
