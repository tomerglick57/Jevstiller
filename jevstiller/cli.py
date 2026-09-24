"""`jevstiller serve`: run the drop-in Jev proxy.

    pip install "jevstiller[server,onnx]"
    jevstiller serve --data-dir ./jevstiller-data --port 8080
    export TYPESAFE_BASE_URL=http://localhost:8080          # in the calling services; nothing else changes
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .task import Config


def _serve(a: argparse.Namespace) -> None:
    import uvicorn

    from .encoders import BatchingEncoder, load_encoder
    from .manager import Admission, TaskManager
    from .scheduler import TrainScheduler
    from .server import KeyRegistry, ProxySettings, create_app, load_salt

    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    encoder = BatchingEncoder(load_encoder(a.encoder, backend=a.backend, device=a.device))
    scheduler = TrainScheduler(workers=a.train_workers)

    class _NoTeacher:                                       # the proxy makes every teacher call itself
        name = "jev"

        def classify(self, texts, task):
            raise RuntimeError("the proxy calls the teacher itself")

    retention = a.text_retention_days * 86400 if a.text_retention_days else None
    manager = TaskManager(a.data_dir, _NoTeacher(), encoder, Config(store_text=not a.no_store_text),
                          target_agreement=a.target_agreement, max_loaded=a.max_loaded,
                          admission=Admission(min_requests=a.admit_after), max_tasks_per_tenant=a.max_tasks_per_tenant,
                          train_executor=scheduler, text_retention_s=retention)
    tenant_map = json.loads(Path(a.tenants_file).read_text()) if a.tenants_file else {}
    settings = ProxySettings(upstream=a.upstream, upstream_timeout_s=a.upstream_timeout, tenancy=a.tenancy,
                             tenant_map=tenant_map, access_token=a.access_token or None,
                             allow_networks=[n for n in a.allow_network for n in n.split(",") if n],
                             max_body_bytes=int(a.max_body_mb * 2**20))
    app = create_app(manager, settings, KeyRegistry(load_salt(a.data_dir), settings.key_ttl_s),
                     closers=[lambda: scheduler.shutdown(wait=True, cancel_futures=True), encoder.close])
    uvicorn.run(app, host=a.host, port=a.port, workers=1, log_level=a.log_level.lower(), access_log=False,
                ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile)


def _key_hash(a: argparse.Namespace) -> None:
    """Print the deployment's salted hash of an API key read from stdin (for the tenants file)."""
    import getpass
    import sys

    from .server import KeyRegistry, load_salt
    key = sys.stdin.readline().strip() if not sys.stdin.isatty() else getpass.getpass("API key: ").strip()
    if not key:
        raise SystemExit("no key given")
    print(KeyRegistry(load_salt(a.data_dir), 0).hash(key))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevstiller")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the drop-in Jev proxy")
    env = os.environ.get
    s.add_argument("--host", default=env("JEVSTILLER_HOST", "0.0.0.0"))
    s.add_argument("--port", type=int, default=int(env("JEVSTILLER_PORT", "8080")))
    s.add_argument("--data-dir", default=env("JEVSTILLER_DATA_DIR", "./jevstiller-data"))
    s.add_argument("--upstream", default=env("JEVSTILLER_UPSTREAM", "https://api.typesafe.ai"))
    s.add_argument("--upstream-timeout", type=float, default=9.0)
    s.add_argument("--encoder", default=env("JEVSTILLER_ENCODER", "small"), help="small | base | large | hash | ...")
    s.add_argument("--backend", default="auto")
    s.add_argument("--device", default="auto")
    s.add_argument("--target-agreement", type=float, default=float(env("JEVSTILLER_TARGET_AGREEMENT", "0.98")))
    s.add_argument("--tenancy", choices=["shared", "per_key"], default=env("JEVSTILLER_TENANCY", "shared"))
    s.add_argument("--admit-after", type=int, default=50, help="requests before a new question becomes a task")
    s.add_argument("--max-loaded", type=int, default=64)
    s.add_argument("--max-tasks-per-tenant", type=int, default=None)
    s.add_argument("--train-workers", type=int, default=2)
    s.add_argument("--log-level", default=env("JEVSTILLER_LOG_LEVEL", "info"))
    # security and data
    s.add_argument("--tenants-file", default=env("JEVSTILLER_TENANTS_FILE"),
                   help='JSON {"<key hash>": "<tenant>"}; hashes from `jevstiller key-hash`')
    s.add_argument("--access-token", default=env("JEVSTILLER_ACCESS_TOKEN"),
                   help="require callers to send this in x-jevstiller-token")
    s.add_argument("--allow-network", action="append", default=[n for n in [env("JEVSTILLER_ALLOW_NETWORKS")] if n],
                   help="client CIDR allowed to use the proxy (repeat, or comma-separated)")
    s.add_argument("--max-body-mb", type=float, default=4.0)
    s.add_argument("--no-store-text", action="store_true", default=env("JEVSTILLER_STORE_TEXT", "1") == "0",
                   help="keep only hashes and embeddings of requests, never their text")
    s.add_argument("--text-retention-days", type=float, default=float(env("JEVSTILLER_TEXT_RETENTION_DAYS", "0")),
                   help="blank stored request text older than this (0 = keep)")
    s.add_argument("--ssl-certfile", default=env("JEVSTILLER_SSL_CERTFILE"))
    s.add_argument("--ssl-keyfile", default=env("JEVSTILLER_SSL_KEYFILE"))

    k = sub.add_parser("key-hash", help="print the salted hash of an API key (read from stdin), for --tenants-file")
    k.add_argument("--data-dir", default=env("JEVSTILLER_DATA_DIR", "./jevstiller-data"))
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        _serve(a)
    elif a.cmd == "key-hash":
        _key_hash(a)


if __name__ == "__main__":
    main()
