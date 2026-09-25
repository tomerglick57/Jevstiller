"""`jevstiller` command line.

    jevstiller serve [--config jevstiller.toml] [flags]      run the drop-in Jev proxy
    jevstiller config [--config ...]                         print the effective settings (secrets redacted)
    jevstiller key-hash --data-dir DIR  < key                salted hash of an API key, for the tenants map
    jevstiller backup --data-dir DIR --out BACKUP            consistent copy, safe while serving
    jevstiller restore --from BACKUP --data-dir DIR          into an empty data dir (server stopped)
    jevstiller admin [--url URL] [--token T] <command>       the admin API: tasks, status, mode, target, ...

Settings: built-in defaults < config file (--config or JEVSTILLER_CONFIG) < JEVSTILLER_* environment < flags.
See docs/configuration.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Access-log records (already JSON) are merged in as fields."""

    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": round(record.created, 3), "level": record.levelname.lower(), "logger": record.name}
        msg = record.getMessage()
        if record.name == "jevstiller.access":
            try:
                out.update(json.loads(msg))
            except ValueError:
                out["msg"] = msg
        else:
            out["msg"] = msg
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def _logging(level: str, fmt: str) -> None:
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore"):                  # a line per upstream call; the access log has it all
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def _settings(a: argparse.Namespace):
    from ._settings import load
    cli = {k: v for k, v in vars(a).items() if k not in ("cmd", "config", "func") and v is not None}
    return load(a.config, cli)


def _serve(a: argparse.Namespace) -> None:
    import uvicorn

    from ._manager import Admission, TaskManager
    from ._scheduler import TrainScheduler
    from ._server import KeyRegistry, ProxySettings, create_app, load_salt
    from ._settings import redact_url
    from ._task import Config
    from .encoders import BatchingEncoder, load_encoder

    s = _settings(a)
    os.umask(0o077)                                     # task data is traffic: owner-only files and dirs
    _logging(s.log_level, s.log_format)
    log = logging.getLogger("jevstiller")
    data = Path(s.data_dir)
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    if data.stat().st_mode & 0o077:
        log.warning("data dir %s is readable by other users (mode %o); it holds request text",
                    data, data.stat().st_mode & 0o777)
    if s.access_token and len(s.access_token) < 16:
        log.warning("the access token is shorter than 16 characters")
    if s.tenants:
        log.info("tenants map: %d keys -> %d tenants; other keys use tenancy=%s",
                 len(s.tenants), len(set(s.tenants.values())), s.tenancy)
    # the effective controls, so a setting that didn't take is visible in the first lines of the log
    log.info("access: token %s, client networks %s, trusted proxies %s, /metrics %s",
             "required" if s.access_token else "not required", ", ".join(s.allow_networks) or "any",
             ", ".join(s.trust_forwarded_for) or "none",
             "public" if s.metrics_public else "admin token" if s.admin_token else "off")
    log.info("request text: %s", "not stored" if not s.store_text else
             f"stored, blanked after {s.text_retention_days:g} days" if s.text_retention_days else
             "stored until the task is deleted")
    salt = load_salt(data)
    encoder = BatchingEncoder(load_encoder(s.encoder, backend=s.backend, device=s.device))
    encoder.encode(["warm up"])
    scheduler = TrainScheduler(workers=s.train_workers)

    class _NoTeacher:                                   # the proxy makes every teacher call itself
        name = "jev"

        def classify(self, texts, task):
            raise RuntimeError("the proxy calls the teacher itself")

    manager = TaskManager(
        data, _NoTeacher(), encoder, Config(**{**s.engine, "store_text": s.store_text}),
        target_agreement=s.target_agreement, max_loaded=s.max_loaded, max_memory_mb=s.max_memory_mb,
        admission=Admission(min_requests=s.admit_after, window_s=s.admit_window_s),
        max_tasks_per_tenant=s.max_tasks_per_tenant, max_tasks=s.max_tasks,
        max_new_tasks_per_caller=s.max_new_tasks_per_key,
        idle_ttl_s=s.idle_ttl_days * 86400 if s.idle_ttl_days else None,
        text_retention_s=s.text_retention_days * 86400 if s.text_retention_days else None,
        train_executor=scheduler, blas_threads=s.blas_threads, hash_key=salt, task_overrides=s.tasks)
    settings = ProxySettings(upstream=s.upstream, upstream_timeout_s=s.upstream_timeout_s,
                             max_upstream_inflight=s.max_upstream_inflight, tenancy=s.tenancy,
                             key_ttl_s=s.key_ttl_s, price_per_mtok=s.price_per_mtok, tenant_map=s.tenants,
                             access_token=s.access_token, allow_networks=s.allow_networks,
                             max_body_bytes=int(s.max_body_mb * 2**20), max_questions=s.max_questions,
                             max_encoder_wait_ms=s.max_encoder_wait_ms)

    import tempfile
    checked = {"at": -1e9, "writable": True}

    def writable() -> bool:
        # at most once a second, to a file of this check's own (with one fixed name, concurrent probes deleted
        # each other's file and reported the disk unwritable)
        now = time.monotonic()
        if now - checked["at"] >= 1.0:
            try:
                with tempfile.NamedTemporaryFile(dir=data, prefix=".ready-probe-") as f:
                    f.write(b"ok")
                    f.flush()
                writable = True
            except OSError:
                writable = False
            checked.update(at=now, writable=writable)
        return checked["writable"]

    def ready() -> dict[str, bool]:
        # A full data dir is not a reason to stop serving: records are dropped (and counted), answers go on.
        # Failing readiness on it took the only replica out of the Service (security audit run 3), so it is
        # a metric (jevstiller_data_dir_writable) instead.
        return {"encoder": encoder.dim > 0}

    app = create_app(manager, settings, KeyRegistry(salt, s.key_ttl_s), admin_token=s.admin_token,
                     metrics_public=s.metrics_public, ready=ready,
                     closers=[lambda: scheduler.shutdown(wait=True, cancel_futures=True), encoder.close])
    app.state.metrics.gauge("jevstiller_data_dir_writable", "1 if the data directory accepts writes (0: disk full?)",
                            fn=lambda: [((), 1.0 if writable() else 0.0)])
    log.info("jevstiller serving on %s:%d, upstream %s, tenancy %s, admin API %s", s.host, s.port,
             redact_url(s.upstream), s.tenancy, "on" if s.admin_token else "off")
    # X-Forwarded-For is honoured only from proxies the operator lists (it decides allow_networks).
    # Idle connections are kept 75 s, not uvicorn's 5 s: clients (httpx: 5 s) and load balancers (60 s) reuse
    # them for about that long, and a server closing first drops the request in flight (37 in 1.15M in the soak).
    uvicorn.run(app, host=s.host, port=s.port, workers=1, log_level=s.log_level.lower(), access_log=False,
                ssl_certfile=s.ssl_certfile, ssl_keyfile=s.ssl_keyfile, log_config=None, timeout_keep_alive=75,
                proxy_headers=bool(s.trust_forwarded_for),
                forwarded_allow_ips=",".join(s.trust_forwarded_for) if s.trust_forwarded_for else None)


def _config(a: argparse.Namespace) -> None:
    print(json.dumps(_settings(a).redacted(), indent=2))


def _key_hash(a: argparse.Namespace) -> None:
    """Print the deployment's salted hash of an API key read from stdin (for the tenants map). Refuses to
    create a salt: a hash made with a different salt would never match the server's."""
    import getpass

    from ._server import KeyRegistry, load_salt
    try:
        salt = load_salt(a.data_dir, create=False)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from None
    key = sys.stdin.readline().strip() if not sys.stdin.isatty() else getpass.getpass("API key: ").strip()
    if not key:
        raise SystemExit("no key given")
    print(KeyRegistry(salt, 0).hash(key))


def _backup(a: argparse.Namespace) -> None:
    from ._backup import backup
    print(json.dumps(backup(a.data_dir, a.out), indent=2))


def _restore(a: argparse.Namespace) -> None:
    from ._backup import restore
    print(json.dumps(restore(getattr(a, "from"), a.data_dir, force=a.force), indent=2))


def _admin(a: argparse.Namespace) -> None:
    import httpx
    token = a.token or os.environ.get("JEVSTILLER_ADMIN_TOKEN")
    if not token and a.token_file:
        token = Path(a.token_file).read_text().strip()
    if not token:
        raise SystemExit("an admin token is required (--token, --token-file or JEVSTILLER_ADMIN_TOKEN)")
    base = a.url.rstrip("/") + "/jevstiller/v1"
    if a.action not in ("tasks", "stats") and not a.key:
        raise SystemExit(f"`admin {a.action}` needs a {'tenant' if a.action == 'delete-tenant' else 'task key'}")
    k = quote(a.key or "", safe="")                    # a name with # or ? must not address another tenant
    calls = {
        "tasks": ("GET", "/tasks", None),
        "status": ("GET", f"/tasks/{k}", None),
        "versions": ("GET", f"/tasks/{k}/versions", None),
        "mode": ("POST", f"/tasks/{k}/mode", {"mode": a.value}),
        "target": ("POST", f"/tasks/{k}/target", {"target_agreement": _num(a.value)}),
        "train": ("POST", f"/tasks/{k}/train", None),
        "promote": ("POST", f"/tasks/{k}/promote", {"version": a.value}),
        "rollback": ("POST", f"/tasks/{k}/rollback", None),
        "delete": ("DELETE", f"/tasks/{k}", None),
        "delete-tenant": ("DELETE", f"/tenants/{k}", None),
        "stats": ("GET", "/stats", None),
    }
    method, path, body = calls[a.action]
    params = {"tenant": a.key} if a.action == "tasks" and a.key else None
    r = httpx.request(method, base + path, json=body, params=params, timeout=600,
                      headers={"Authorization": f"Bearer {token}"})
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    if a.action == "status" and r.is_success and isinstance(data, dict) and not a.json:
        print(data.get("report", ""))
    else:
        print(json.dumps(data, indent=2) if not isinstance(data, str) else data)
    if not r.is_success:
        raise SystemExit(1)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevstiller", description="Jev, distilled on the fly.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def settings_args(p: argparse.ArgumentParser) -> None:
        """Flags override the config file and environment; unset flags (None) don't."""
        p.add_argument("--config", help="TOML settings file (default: $JEVSTILLER_CONFIG)")
        p.add_argument("--host")
        p.add_argument("--port", type=int)
        p.add_argument("--data-dir")
        p.add_argument("--upstream")
        p.add_argument("--upstream-timeout", dest="upstream_timeout_s", type=float)
        p.add_argument("--encoder", help="small | base | large | hash | onnx:<repo> | torch:<model>")
        p.add_argument("--backend")
        p.add_argument("--device")
        p.add_argument("--target-agreement", type=float)
        p.add_argument("--tenancy", choices=["shared", "per_key"])
        p.add_argument("--admit-after", type=int, help="requests before a new question becomes a task")
        p.add_argument("--max-loaded", type=int)
        p.add_argument("--max-tasks", type=int)
        p.add_argument("--max-tasks-per-tenant", type=int)
        p.add_argument("--max-new-tasks-per-key", type=int, help="tasks one API key may create per admission window")
        p.add_argument("--max-questions", type=int, help="distinct choice questions routed per request")
        p.add_argument("--max-encoder-wait-ms", type=float,
                       help="forward to Jev when a local answer would wait longer than this for the encoder (0: never)")
        p.add_argument("--train-workers", type=int)
        p.add_argument("--log-level")
        p.add_argument("--log-format", choices=["text", "json"])
        p.add_argument("--tenants-file", help='JSON {"<key hash>": "<tenant>"}; hashes from `jevstiller key-hash`')
        p.add_argument("--access-token", help="require callers to send this in x-jevstiller-token "
                                              "(prefer JEVSTILLER_ACCESS_TOKEN or access_token_file: flags are "
                                              "visible in the process list)")
        p.add_argument("--access-token-file")
        p.add_argument("--admin-token-file", help="enables the admin API and protects /metrics")
        p.add_argument("--allow-network", dest="allow_networks", action="append",
                       help="client CIDR allowed to use the proxy (repeatable)")
        p.add_argument("--trust-forwarded-for", action="append",
                       help="proxy address whose X-Forwarded-For is trusted (repeatable)")
        p.add_argument("--max-body-mb", type=float)
        p.add_argument("--no-store-text", dest="store_text", action="store_const", const=False,
                       help="keep only hashes and embeddings of requests, never their text")
        p.add_argument("--text-retention-days", type=float, help="blank stored request text older than this")
        p.add_argument("--ssl-certfile")
        p.add_argument("--ssl-keyfile")

    s = sub.add_parser("serve", help="run the drop-in Jev proxy")
    settings_args(s)
    s.set_defaults(func=_serve)
    c = sub.add_parser("config", help="print the effective settings (secrets redacted)")
    settings_args(c)
    c.set_defaults(func=_config)

    k = sub.add_parser("key-hash", help="print the salted hash of an API key (read from stdin)")
    k.add_argument("--data-dir", default=os.environ.get("JEVSTILLER_DATA_DIR", "./jevstiller-data"))
    k.set_defaults(func=_key_hash)

    b = sub.add_parser("backup", help="consistent copy of a data dir, safe while serving")
    b.add_argument("--data-dir", default=os.environ.get("JEVSTILLER_DATA_DIR", "./jevstiller-data"))
    b.add_argument("--out", required=True)
    b.set_defaults(func=_backup)
    r = sub.add_parser("restore", help="restore a backup into an empty data dir (server stopped)")
    r.add_argument("--from", required=True)
    r.add_argument("--data-dir", default=os.environ.get("JEVSTILLER_DATA_DIR", "./jevstiller-data"))
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=_restore)

    ad = sub.add_parser("admin", help="call the admin API")
    ad.add_argument("--url", default=os.environ.get("JEVSTILLER_ADMIN_URL", "http://127.0.0.1:8080"))
    ad.add_argument("--token")
    ad.add_argument("--token-file")
    ad.add_argument("--json", action="store_true", help="raw JSON for `status`")
    ad.add_argument("action", choices=["tasks", "status", "versions", "mode", "target", "train", "promote",
                                       "rollback", "delete", "delete-tenant", "stats"])
    ad.add_argument("key", nargs="?", help="task key (tenant for `tasks` / `delete-tenant`)")
    ad.add_argument("value", nargs="?", help="mode, target agreement or version")
    ad.set_defaults(func=_admin)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
