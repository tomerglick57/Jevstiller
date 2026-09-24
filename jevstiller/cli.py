"""`jevstiller serve`: run the drop-in Jev proxy.

    pip install "jevstiller[server,onnx]"
    jevstiller serve --data-dir ./jevstiller-data --port 8080
    export TYPESAFE_BASE_URL=http://localhost:8080          # in the calling services; nothing else changes
"""
from __future__ import annotations

import argparse
import logging
import os

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

    manager = TaskManager(a.data_dir, _NoTeacher(), encoder, Config(), target_agreement=a.target_agreement,
                          max_loaded=a.max_loaded, admission=Admission(min_requests=a.admit_after),
                          max_tasks_per_tenant=a.max_tasks_per_tenant, train_executor=scheduler)
    settings = ProxySettings(upstream=a.upstream, upstream_timeout_s=a.upstream_timeout, tenancy=a.tenancy)
    app = create_app(manager, settings, KeyRegistry(load_salt(a.data_dir), settings.key_ttl_s),
                     closers=[lambda: scheduler.shutdown(wait=True, cancel_futures=True), encoder.close])
    uvicorn.run(app, host=a.host, port=a.port, workers=1, log_level=a.log_level.lower(), access_log=False)


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
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        _serve(a)


if __name__ == "__main__":
    main()
