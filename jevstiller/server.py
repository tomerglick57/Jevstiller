"""The drop-in Jev proxy: speaks Jev's HTTP API, answers from local students when it can, forwards otherwise.

Callers change nothing but the base URL (`TYPESAFE_BASE_URL=http://jevstiller:8080`) and keep their own Jev
API keys. For `POST /v1/systemone`:

- every `choice` question maps to a task (tenant, instructions, criteria, requested model);
- if every question can be answered locally, and the caller's key has been accepted by Jev before, the proxy
  answers in Jev's exact response shape;
- otherwise the *whole original request* is forwarded to Jev with the caller's key, Jev's response is returned
  unchanged, and each choice answer becomes a training row for its task.

Anything else (other paths, `GET /v1/models`, requests the proxy does not understand) is forwarded as is.
A failure inside the proxy's own logic falls back to forwarding: Jevstiller never makes a request fail that Jev
would have answered.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .manager import Routing, TaskManager
from .teachers import TeacherOutput

log = logging.getLogger("jevstiller.server")

SYSTEM_ONE = "/v1/systemone"
REQUEST_ID = "x-typesafe-request-id"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}
RESPONSE_DROP = HOP_BY_HOP | {"content-encoding"}         # httpx hands us the decoded body
PROXY_TOKEN_HEADER = "x-jevstiller-token"
FORWARD_DROP = HOP_BY_HOP | {PROXY_TOKEN_HEADER}          # the proxy's own credential never reaches Jev
KNOWN_FIELDS = {"state", "model", "questions"}


@dataclass
class ProxySettings:
    upstream: str = "https://api.typesafe.ai"
    upstream_timeout_s: float = 9.0                        # under the SDK's default 10 s client timeout
    max_upstream_inflight: int = 256                       # beyond this: 503 + retry-after (the SDK retries)
    tenancy: str = "shared"                                # shared: one tenant; per_key: one per API key
    key_ttl_s: float = 3600.0                              # re-verify a key with Jev after this long
    price_per_mtok: float = 0.042
    tenant_map: dict[str, str] = field(default_factory=dict)   # key hash -> tenant (overrides `tenancy`)
    access_token: str | None = None                        # if set, callers must send x-jevstiller-token
    allow_networks: list[str] = field(default_factory=list)    # if set, only these client CIDRs (direct peer)
    max_body_bytes: int = 4 * 2**20                        # larger requests get 413 (Jev's own limit is ~64k tokens)

    def tenant_for(self, kh: str | None) -> str:
        if kh and kh in self.tenant_map:
            return self.tenant_map[kh]
        return "default" if self.tenancy == "shared" else f"key:{kh}"


class KeyRegistry:
    """Which API keys Jev has accepted, by salted hash only; raw keys are never stored or logged.

    A key must have been accepted by Jev (a 2xx on a forwarded request) within `ttl_s` before the proxy
    answers anything for it locally; a 401/403 from Jev revokes it at once. Also remembers per-key upstream
    back-off from 429 `retry-after`.
    """

    def __init__(self, salt: bytes, ttl_s: float):
        self.salt, self.ttl_s = salt, ttl_s
        self._verified: dict[str, float] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def hash(self, key: str | None) -> str | None:
        return hashlib.sha256(self.salt + key.encode()).hexdigest()[:32] if key else None

    def verified(self, kh: str | None) -> bool:
        with self._lock:
            return kh is not None and time.monotonic() - self._verified.get(kh, -1e18) < self.ttl_s

    def accept(self, kh: str | None) -> None:
        if kh:
            with self._lock:
                self._verified[kh] = time.monotonic()

    def revoke(self, kh: str | None) -> None:
        if kh:
            with self._lock:
                self._verified.pop(kh, None)

    def block(self, kh: str | None, seconds: float) -> None:
        if kh and seconds > 0:
            with self._lock:
                self._blocked_until[kh] = max(self._blocked_until.get(kh, 0), time.monotonic() + seconds)

    def blocked_for(self, kh: str | None) -> float:
        with self._lock:
            return max(0.0, self._blocked_until.get(kh, 0) - time.monotonic()) if kh else 0.0


def load_salt(data_dir: str | Path) -> bytes:
    """A per-deployment secret for hashing API keys, created on first start (mode 0600)."""
    p = Path(data_dir) / "key-salt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(32))
    return p.read_bytes()


def _retry_after_s(headers: Mapping[str, str]) -> float:
    try:
        if "retry-after-ms" in headers:
            return float(headers["retry-after-ms"]) / 1000
        return float(headers.get("retry-after", 0))
    except ValueError:
        return 0.0


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() or None if auth.lower().startswith("bearer ") else None


def _error(status: int, detail: str, retry_after: float | None = None) -> JSONResponse:
    headers = {REQUEST_ID: f"jvs_{uuid.uuid4().hex}"}
    if retry_after is not None:
        headers["retry-after"] = str(max(1, round(retry_after)))
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


def _peakedness(probs: Mapping[str, float]) -> float:
    k = len(probs)
    return 1.0 if k < 2 else max(0.0, (k * max(probs.values()) - 1.0) / (k - 1.0))


class Proxy:
    def __init__(self, manager: TaskManager, settings: ProxySettings, keys: KeyRegistry,
                 client: httpx.AsyncClient | None = None):
        self.manager, self.settings, self.keys = manager, settings, keys
        self.client = client or httpx.AsyncClient(base_url=settings.upstream.rstrip("/"),
                                                  timeout=settings.upstream_timeout_s)
        self._upstream_slots = settings.max_upstream_inflight   # event-loop only: no lock needed
        self.stats = {"local": 0, "forwarded": 0, "passthrough": 0, "errors": 0}

    # ---- upstream -----------------------------------------------------------------
    async def forward(self, request: Request, body: bytes, kh: str | None) -> httpx.Response | Response:
        """Send the caller's request to Jev unchanged. Returns Jev's response, or a Jev-style error response
        of our own (back-off, overload, timeout, unreachable)."""
        wait = self.keys.blocked_for(kh)
        if wait > 0:
            return _error(429, "rate limited by the upstream API; retry later (jevstiller)", wait)
        if self._upstream_slots <= 0:
            return _error(503, "jevstiller is overloaded; retry shortly", 1)
        self._upstream_slots -= 1
        try:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in FORWARD_DROP}
            req = self.client.build_request(request.method, request.url.path, params=request.query_params,
                                            headers=headers, content=body)
            resp = await self.client.send(req)
        except httpx.TimeoutException:
            return _error(504, "upstream API timed out (jevstiller)")
        except httpx.HTTPError as e:
            log.warning("upstream unreachable: %s", type(e).__name__)
            return _error(502, "upstream API unreachable (jevstiller)")
        finally:
            self._upstream_slots += 1
        if resp.status_code == 429:
            self.keys.block(kh, _retry_after_s(resp.headers))
        elif resp.status_code in (401, 403):
            self.keys.revoke(kh)
        elif resp.is_success:
            self.keys.accept(kh)
        return resp

    @staticmethod
    def relay(resp: httpx.Response | Response, extra: Mapping[str, str] | None = None) -> Response:
        if isinstance(resp, Response):
            if extra:
                resp.headers.update(extra)
            return resp
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESPONSE_DROP}
        headers.update(extra or {})
        return Response(resp.content, status_code=resp.status_code, headers=headers)

    # ---- handlers -----------------------------------------------------------------
    async def _body(self, request: Request) -> bytes:
        body = await request.body()
        if len(body) > self.settings.max_body_bytes:
            raise _TooLarge()
        return body

    async def passthrough(self, request: Request) -> Response:
        body = await self._body(request)
        kh = self.keys.hash(_bearer(request))
        self.stats["passthrough"] += 1
        return self.relay(await self.forward(request, body, kh))

    async def system_one(self, request: Request) -> Response:
        body = await self._body(request)
        key = _bearer(request)
        kh = self.keys.hash(key)
        try:
            req = json.loads(body)
        except ValueError:
            req = None
        if not (isinstance(req, dict) and set(req) <= KNOWN_FIELDS and isinstance(req.get("questions"), dict)
                and req["questions"] and isinstance(req.get("model"), str) and "state" in req and key):
            self.stats["passthrough"] += 1
            return self.relay(await self.forward(request, body, kh))   # let Jev judge what we don't understand

        tenant = self.settings.tenant_for(kh)
        routings: dict[str, Routing] = {}
        others: list[str] = []
        try:
            for name, q in req["questions"].items():
                if not (isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), dict)):
                    others.append(name)
                    continue
                try:
                    routings[name] = await anyio.to_thread.run_sync(
                        self.manager.route, tenant, q.get("instructions"), q["criteria"], [req["state"]],
                        req["model"])
                except ValueError:
                    others.append(name)                 # not a valid task (e.g. one class): Jev decides
        except Exception:
            log.exception("routing failed; forwarding")
            self._defer_all(routings, "proxy_error")
            await self._finish(routings, None, None, 0.0)       # releases; failed items are not recorded
            self.stats["errors"] += 1
            return self.relay(await self.forward(request, body, kh))

        verified = self.keys.verified(kh)
        if not others and verified and routings and all(r.local for r in routings.values()):
            try:
                resp = await self._local_response(req, routings)
                self.stats["local"] += 1
                return resp
            except Exception:
                log.exception("local answer failed; forwarding")
                self.stats["errors"] += 1
                # engines were released by _local_response's cleanup; route again is not worth it
                return self.relay(await self.forward(request, body, kh))

        # to the teacher: the whole request, with the caller's key
        reason = "co_deferred" if verified else "key_unverified"
        self._defer_all(routings, reason)
        t0 = time.perf_counter()
        resp = await self.forward(request, body, kh)
        latency_ms = (time.perf_counter() - t0) * 1000
        answers = None
        if isinstance(resp, httpx.Response) and resp.is_success:
            try:
                answers = resp.json()
            except ValueError:
                answers = None
        await self._finish(routings, answers, resp, latency_ms)
        self.stats["forwarded"] += 1
        sources = {n: (r.reason or reason) for n, r in routings.items()}
        return self.relay(resp, {"x-jevstiller-source": "upstream",
                                 "x-jevstiller-detail": json.dumps(sources, separators=(",", ":"))})

    @staticmethod
    def _defer_all(routings: Mapping[str, Routing], reason: str) -> None:
        """The request goes to Jev as a whole: nothing in it is answered by a student."""
        for r in routings.values():
            if r.engine is not None:
                for i in range(len(r.routed.recs)):
                    r.engine.defer(r.routed, i, reason)

    async def _finish(self, routings: Mapping[str, Routing], answers: dict | None,
                      resp: httpx.Response | Response | None, latency_ms: float) -> None:
        """Complete every routing with Jev's answer to its question, or an error (not recorded)."""
        n_choice = max(1, len(routings))
        usage = (answers or {}).get("usage") or {}
        tokens = int(usage.get("input_tokens") or 0) // n_choice
        model = (answers or {}).get("model")
        rid = resp.headers.get(REQUEST_ID) if resp is not None else None
        for name, r in routings.items():
            if r.engine is None:
                continue
            out: TeacherOutput | Exception
            a = ((answers or {}).get("answers") or {}).get(name)
            if isinstance(a, dict) and a.get("type") == "choice" and isinstance(a.get("probabilities"), dict):
                probs = {c: float(a["probabilities"].get(c, 0.0)) for c in r.task.labels}
                z = sum(probs.values()) or 1.0
                out = TeacherOutput(label=str(a.get("choice")), probs={c: p / z for c, p in probs.items()},
                                    confidence=float(a.get("confidence", 0.0)), input_tokens=tokens,
                                    cost_usd=tokens * self.settings.price_per_mtok / 1e6, latency_ms=latency_ms,
                                    request_id=rid, model=f"jev:{model}" if model else None)
            else:
                status = getattr(resp, "status_code", None)
                out = RuntimeError(f"no teacher answer (upstream status {status})")
            outs = [out] * len(r.routed.to_teacher)
            try:
                await anyio.to_thread.run_sync(self.manager.complete, r, outs, "jev")
            except Exception:
                log.exception("recording task %s failed", r.key)

    async def _local_response(self, req: dict, routings: Mapping[str, Routing]) -> Response:
        answers: dict[str, Any] = {}
        detail: dict[str, str] = {}
        model = None
        released: set[str] = set()
        try:
            for name in req["questions"]:
                r = routings[name]
                released.add(name)                       # complete() releases the engine even if it raises
                [res] = await anyio.to_thread.run_sync(self.manager.complete, r, [], "jev")
                answers[name] = {"type": "choice", "choice": res.label, "confidence": _peakedness(res.probs),
                                 "probabilities": res.probs}
                detail[name] = res.source
                lineage = r.engine._teacher_model or ""
                model = model or (lineage[4:] if lineage.startswith("jev:") else None)
        finally:
            for name, r in routings.items():             # a failure part-way: release the rest unrecorded
                if name not in released and r.engine is not None:
                    self.manager._release(r.key)
        body = {"model": model or req["model"], "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}
        return JSONResponse(body, headers={REQUEST_ID: f"jvs_{uuid.uuid4().hex}", "x-jevstiller-source": "local",
                                           "x-jevstiller-detail": json.dumps(detail, separators=(",", ":"))})

    async def healthz(self, request: Request) -> Response:
        """Liveness: the process answers. No counters here (it is usually unauthenticated)."""
        return JSONResponse({"ok": True})


class _TooLarge(Exception):
    pass


class AccessMiddleware:
    """Who may use the proxy at all, checked before anything else: client network, shared token, body size.
    `/healthz` is exempt from the network and token checks (load-balancer probes)."""

    def __init__(self, app, settings: ProxySettings):
        import ipaddress
        self.app, self.settings = app, settings
        self.networks = [ipaddress.ip_network(n, strict=False) for n in settings.allow_networks]

    def _client_allowed(self, scope) -> bool:
        if not self.networks:
            return True
        import ipaddress
        host = (scope.get("client") or ("",))[0]
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return any(ip in n for n in self.networks)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in ("/healthz", "/readyz"):
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if not self._client_allowed(scope):
            return await _error(403, "client network not allowed (jevstiller)")(scope, receive, send)
        token = self.settings.access_token
        if token and not hmac.compare_digest(headers.get(PROXY_TOKEN_HEADER, "").encode(), token.encode()):
            return await _error(401, "missing or invalid x-jevstiller-token (jevstiller)")(scope, receive, send)
        try:
            if int(headers.get("content-length", "0")) > self.settings.max_body_bytes:
                return await _error(413, "request body too large (jevstiller)")(scope, receive, send)
        except ValueError:
            return await _error(400, "invalid content-length (jevstiller)")(scope, receive, send)
        try:
            await self.app(scope, receive, send)
        except _TooLarge:
            await _error(413, "request body too large (jevstiller)")(scope, receive, send)


def create_app(manager: TaskManager, settings: ProxySettings | None = None, keys: KeyRegistry | None = None,
               client: httpx.AsyncClient | None = None, closers: Sequence[Callable[[], Any]] = ()) -> Starlette:
    """The proxy as an ASGI app. On shutdown it closes the upstream client and the manager, then calls
    `closers` (e.g. the training scheduler's shutdown), in order."""
    settings = settings or ProxySettings()
    keys = keys or KeyRegistry(secrets.token_bytes(32), settings.key_ttl_s)
    proxy = Proxy(manager, settings, keys, client)
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        await proxy.client.aclose()
        await anyio.to_thread.run_sync(manager.close)
        for close in closers:
            await anyio.to_thread.run_sync(close)

    app = Starlette(routes=[
        Route(SYSTEM_ONE, proxy.system_one, methods=["POST"]),
        Route("/healthz", proxy.healthz, methods=["GET"]),
        Route("/{path:path}", proxy.passthrough, methods=methods),
    ], lifespan=lifespan)
    app.state.proxy = proxy
    app.add_middleware(AccessMiddleware, settings=settings)
    return app
