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

Security properties (security audit run 1, 2026-09-24; see SECURITY.md):
- a key counts as accepted only after Jev answered a /v1/systemone request with it (2xx, parseable), and
  requests with an unaccepted key are forwarded first and only recorded afterwards, so callers without a
  working Jev key can neither get local answers nor create tasks;
- /healthz, /readyz, /metrics and /jevstiller/* are served locally and never forwarded;
- the access token, network allowlist, body limit (enforced while streaming), a single Authorization
  header and dot-free paths are checked before anything else;
- the upstream client stores no cookies, so nothing leaks between callers.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Any

import anyio
import httpx
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .admin import Admin
from .manager import Routing, TaskManager
from .metrics import Registry as MetricsRegistry
from .task import canonical_json, state_text
from .teachers import TeacherOutput

log = logging.getLogger("jevstiller.server")
access_log = logging.getLogger("jevstiller.access")

SYSTEM_ONE = "/v1/systemone"
REQUEST_ID = "x-typesafe-request-id"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}
RESPONSE_DROP = HOP_BY_HOP | {"content-encoding",         # httpx hands us the decoded body
                              "server", "date"}           # uvicorn sets its own: no duplicate headers
PROXY_TOKEN_HEADER = "x-jevstiller-token"
FORWARD_DROP = HOP_BY_HOP | {PROXY_TOKEN_HEADER}          # the proxy's own credential never reaches Jev
KNOWN_FIELDS = {"state", "model", "questions"}
LOCAL_PATHS = ("/healthz", "/readyz", "/metrics")          # served by the proxy itself, never forwarded
PROBE_PATHS = ("/healthz", "/readyz")                     # GET/HEAD need no access token (load balancers)
ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
MAX_DETAIL_QUESTIONS = 64                                 # beyond this, x-jevstiller-detail is a summary
UPSTREAM_POOL = 16                                        # connections per upstream client (see Proxy)
MAX_RETRY_AFTER_S = 3600.0                                # an upstream back-off longer than this is capped


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
    allow_networks: list[str] = field(default_factory=list)    # if set, only these client CIDRs (the peer as
                                                           # uvicorn reports it; see trust_forwarded_for in cli)
    max_body_bytes: int = 4 * 2**20                        # larger requests get 413 (Jev's own limit is ~64k tokens)
    max_questions: int = 32                                # distinct choice questions routed per request
    max_encoder_wait_ms: float = 200.0                     # encoder busier than this: forward (0 = never)

    def tenant_for(self, kh: str | None) -> str:
        if kh and kh in self.tenant_map:
            return self.tenant_map[kh]
        return "default" if self.tenancy == "shared" else f"key:{kh}"


class KeyRegistry:
    """Which API keys Jev has accepted, by salted hash only; raw keys are never stored or logged.

    A key must have been accepted by Jev (a successful /v1/systemone answer) within `ttl_s` before the proxy
    answers anything for it locally; a 401/403 from Jev on any path revokes it at once. Also remembers per-key
    upstream back-off from 429 `retry-after`.
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
            t = self._verified.get(kh) if kh is not None else None
            return t is not None and time.monotonic() - t < self.ttl_s

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


def load_salt(data_dir: str | Path, create: bool = True) -> bytes:
    """A per-deployment secret for hashing API keys, created on first start (mode 0600). With
    `create=False`, a missing salt is an error (e.g. `jevstiller key-hash` pointed at the wrong directory)."""
    p = Path(data_dir) / "key-salt"
    if not p.exists():
        if not create:
            raise FileNotFoundError(f"no key-salt in {Path(data_dir).resolve()}: is this the server's data dir?")
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(32))
    salt = p.read_bytes()
    if len(salt) < 16:
        raise ValueError(f"{p} is shorter than 16 bytes; refusing to use it as a salt")
    return salt


def _retry_after_s(headers: Mapping[str, str]) -> float:
    """The upstream's requested back-off in seconds: 0 if absent or unparseable, at most MAX_RETRY_AFTER_S."""
    try:
        if "retry-after-ms" in headers:
            s = float(headers["retry-after-ms"]) / 1000
        else:
            s = float(headers.get("retry-after", 0))
    except ValueError:
        return 0.0
    return min(s, MAX_RETRY_AFTER_S) if math.isfinite(s) and s > 0 else 0.0


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


def _under(path: str, base: str) -> bool:
    """`path` is `base` or below it (no dot segments can remain in an httpx-merged path)."""
    return not base or path == base or path.startswith(base + "/")


def _usable(a: Any) -> bool:
    """Jev's answer to one question is a usable choice answer."""
    return isinstance(a, dict) and a.get("type") == "choice" and isinstance(a.get("probabilities"), dict)


def _answered(answer: Mapping, groups: Mapping[str, list[str]]) -> set[str]:
    """The specs whose question Jev answered usably."""
    answers = answer.get("answers") or {}
    return {spec for spec, names in groups.items() if _usable(answers.get(names[0]))}


def _no_cookies() -> CookieJar:
    """A cookie jar that stores nothing: the proxy must not carry one caller's upstream cookies into another
    caller's request. (httpx keeps a raw CookieJar as is; wrapping it in httpx.Cookies would copy it into a
    default jar and lose the policy.)"""
    return CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))


def _systemone_answer(resp: httpx.Response | Response | None) -> dict | None:
    """Jev's parsed answer when `resp` is a successful, well-formed /v1/systemone response, else None."""
    if not isinstance(resp, httpx.Response) or not resp.is_success:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("answers"), dict) and isinstance(data.get("model"), str):
        return data
    return None


class ProxyMetrics:
    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self.requests = registry.counter("jevstiller_requests_total", "Requests handled, by route and outcome",
                                         ["route", "source"])
        self.latency = registry.histogram("jevstiller_request_duration_seconds",
                                          "Time to answer a /v1/systemone request", ["source"])
        self.questions = registry.counter("jevstiller_questions_total", "Choice questions, by what answered them",
                                          ["outcome"])
        self.upstream = registry.counter("jevstiller_upstream_responses_total", "Upstream responses by status class",
                                         ["status"])
        self.upstream_latency = registry.histogram("jevstiller_upstream_duration_seconds", "Upstream call time")
        self.rejected = registry.counter("jevstiller_rejected_total", "Requests refused by the proxy itself",
                                         ["reason"])


class Proxy:
    def __init__(self, manager: TaskManager, settings: ProxySettings, keys: KeyRegistry,
                 client: httpx.AsyncClient | None = None, metrics: ProxyMetrics | None = None):
        self.manager, self.settings, self.keys = manager, settings, keys
        # Connections to Jev are split over several small clients, each request going to the least busy one.
        # httpcore's pool does work per connection per waiting request on every event: one pool of 256 forwarded
        # ~90 req/s at 256 callers, 16 pools of 16 ~840 (Jev's own ceiling with 290 ms answers: ~880).
        n = max(1, settings.max_upstream_inflight)
        self._base_path = httpx.URL(settings.upstream).path.rstrip("/")   # forwarded paths must stay under it
        self.clients = [client] if client is not None else [
            httpx.AsyncClient(base_url=settings.upstream.rstrip("/"), timeout=settings.upstream_timeout_s,
                              cookies=_no_cookies(), limits=httpx.Limits(max_connections=UPSTREAM_POOL,
                                                                         max_keepalive_connections=UPSTREAM_POOL))
            for _ in range(-(-n // UPSTREAM_POOL))]
        self.client = self.clients[0]
        self._client_busy = [0] * len(self.clients)             # event-loop only
        self.metrics = metrics or ProxyMetrics(MetricsRegistry())
        self._upstream_slots = settings.max_upstream_inflight   # event-loop only: no lock needed
        self._routing = 0                                       # requests inside _route (event loop only)
        self.stats = {"local": 0, "forwarded": 0, "passthrough": 0, "errors": 0}

    # ---- upstream -----------------------------------------------------------------
    async def forward(self, request: Request, body: bytes, kh: str | None) -> httpx.Response | Response:
        """Send the caller's request to Jev unchanged. Returns Jev's response, or a Jev-style error response
        of our own (back-off, overload, timeout, unreachable). Never marks a key as accepted: only a parsed
        /v1/systemone answer does (system_one)."""
        wait = self.keys.blocked_for(kh)
        if wait > 0:
            self.metrics.upstream.inc("backoff")
            return _error(429, "rate limited by the upstream API; retry later (jevstiller)", wait)
        if self._upstream_slots <= 0:
            self.metrics.upstream.inc("overload")
            return _error(503, "jevstiller is overloaded; retry shortly", 1)
        self._upstream_slots -= 1
        t0 = time.perf_counter()
        try:
            i = min(range(len(self.clients)), key=self._client_busy.__getitem__)
            self._client_busy[i] += 1
            try:
                # raw bytes: header values may hold any byte h11 accepts (obs-text); httpx would encode str
                # values as ASCII and fail on them
                headers = [(k, v) for k, v in request.headers.raw if k.decode("latin-1").lower() not in FORWARD_DROP]
                for attempt in (0, 1):
                    req = self.clients[i].build_request(request.method, request.url.path,
                                                        params=request.query_params, headers=headers, content=body)
                    if not _under(req.url.path, self._base_path):
                        return _error(400, "invalid path (jevstiller)")
                    try:
                        resp = await self.clients[i].send(req)
                        break
                    except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError):
                        # Typically a pooled connection the upstream had just closed as idle (both sides
                        # time out around 5 s by default): 7 in 1.15M requests in the soak. Once, at once.
                        if attempt:
                            raise
                        self.metrics.upstream.inc("retried")
            finally:
                self._client_busy[i] -= 1
        except httpx.TimeoutException:
            self.metrics.upstream.inc("timeout")
            return _error(504, "upstream API timed out (jevstiller)")
        except httpx.HTTPError as e:
            log.warning("upstream unreachable: %s", type(e).__name__)
            self.metrics.upstream.inc("unreachable")
            return _error(502, "upstream API unreachable (jevstiller)")
        except Exception:
            # e.g. httpx.InvalidURL: a request the proxy could not send is a bad gateway, never a 500 that
            # skips the caller's cleanup
            log.exception("forwarding failed")
            self.metrics.upstream.inc("error")
            return _error(502, "could not forward the request (jevstiller)")
        finally:
            self._upstream_slots += 1
        self.metrics.upstream_latency.observe(time.perf_counter() - t0)
        self.metrics.upstream.inc(f"{resp.status_code // 100}xx")
        if resp.status_code == 429:
            self.keys.block(kh, _retry_after_s(resp.headers))
        elif resp.status_code in (401, 403):
            self.keys.revoke(kh)
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
        body = await request.body()                     # bounded while streaming by AccessMiddleware
        if len(body) > self.settings.max_body_bytes:
            raise _TooLarge()
        return body

    async def passthrough(self, request: Request) -> Response:
        body = await self._body(request)
        kh = self.keys.hash(_bearer(request))
        self.stats["passthrough"] += 1
        self.metrics.requests.inc("passthrough", "upstream")
        return self.relay(await self.forward(request, body, kh))

    async def system_one(self, request: Request) -> Response:
        t0 = time.perf_counter()
        body = await self._body(request)
        key = _bearer(request)
        kh = self.keys.hash(key)
        log_rec: dict[str, Any] = {"path": SYSTEM_ONE}
        try:
            req = json.loads(body)
        except Exception:                               # includes RecursionError on absurd nesting
            req = None
        if not (isinstance(req, dict) and set(req) <= KNOWN_FIELDS and isinstance(req.get("questions"), dict)
                and req["questions"] and isinstance(req.get("model"), str) and "state" in req and key):
            return await self._plain_forward(request, body, kh, "not_understood", t0, log_rec)

        # one routing per distinct choice question; duplicates under other names share it
        groups: dict[str, list[str]] = {}
        specs: dict[str, dict] = {}
        others: list[str] = []
        for name, q in req["questions"].items():
            if not (isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), dict)):
                others.append(name)
                continue
            try:
                spec = canonical_json([q.get("instructions"), q["criteria"]])
            except Exception:
                others.append(name)
                continue
            groups.setdefault(spec, []).append(name)
            specs[spec] = q
        log_rec["questions"] = len(req["questions"])
        if not groups or len(groups) > self.settings.max_questions:
            return await self._plain_forward(request, body, kh, "too_many_questions" if groups else "no_choice",
                                             t0, log_rec)
        try:
            tenant = self.settings.tenant_for(kh)
        except Exception:
            log.exception("tenant lookup failed; forwarding")
            return await self._plain_forward(request, body, kh, "proxy_error", t0, log_rec)
        log_rec["tenant"] = tenant

        if not self.keys.verified(kh):
            # Never route, admit or answer for a key Jev hasn't accepted: forward first, learn afterwards.
            resp = await self.forward(request, body, kh)
            answer = _systemone_answer(resp)
            if answer is not None:
                self.keys.accept(kh)
            if answer is not None and not self._overloaded():       # record it, unless the encoder is saturated
                routings, _ = await self._route(groups, specs, tenant, req, answered=_answered(answer, groups))
                if routings is not None:                # None: routing failed and released everything
                    try:
                        self._defer_all(routings, "key_unverified")
                    finally:
                        await self._finish(routings, groups, answer, resp, (time.perf_counter() - t0) * 1000)
                    log_rec["tasks"] = [r.key for r in routings.values()][:8]
            return self._upstream_reply(resp, {n: "key_unverified" for g in groups.values() for n in g}, t0,
                                        log_rec)

        if self._overloaded():                          # the local path would be slower than Jev: forward
            return await self._plain_forward(request, body, kh, "overloaded", t0, log_rec)
        routings, unsupported = await self._route(groups, specs, tenant, req)
        if routings is None:                            # routing itself failed: fail open
            return await self._plain_forward(request, body, kh, "proxy_error", t0, log_rec)
        log_rec["tasks"] = [r.key for r in routings.values()][:8]
        if not others and not unsupported and all(r.local for r in routings.values()):
            try:
                resp = await self._local_response(req, groups, routings)
            except Exception:
                log.exception("local answer failed; forwarding")
                self.stats["errors"] += 1
                return await self._plain_forward(request, body, kh, "proxy_error", t0, log_rec)
            self.stats["local"] += 1
            self.metrics.requests.inc("systemone", "local")
            self.metrics.questions.inc("local", by=sum(len(v) for v in groups.values()))
            self.metrics.latency.observe(time.perf_counter() - t0, "local")
            self._log(log_rec, "local", 200, t0, resp.headers.get(REQUEST_ID))
            return resp

        # to the teacher: the whole request, with the caller's key
        t_up = time.perf_counter()
        answer = resp = None
        try:                                            # whatever happens, the routed engines are released
            self._defer_all(routings, "co_deferred")
            resp = await self.forward(request, body, kh)
            answer = _systemone_answer(resp)
            if answer is not None:
                self.keys.accept(kh)                    # refresh
        finally:
            admitted = await self._finish(routings, groups, answer, resp, (time.perf_counter() - t_up) * 1000)
        if admitted and answer is not None:             # this answer made them tasks: record it as their first row
            more, _ = await self._route({s: groups[s] for s in admitted}, specs, tenant, req, answered=set(admitted))
            if more:
                try:
                    self._defer_all(more, "co_deferred")
                finally:
                    await self._finish(more, groups, answer, resp, (time.perf_counter() - t_up) * 1000)
        reasons = {}
        for spec, names in groups.items():
            r = routings.get(spec)
            for n in names:
                reasons[n] = (r.reason or "co_deferred") if r is not None else "unsupported"
        for n in others:
            reasons[n] = "unsupported"
        return self._upstream_reply(resp, reasons, t0, log_rec)

    def _overloaded(self) -> bool:
        """The shared encoder is so backed up that a local answer would take longer than `max_encoder_wait_ms`.
        Such requests go to Jev as they are: not routed, encoded or recorded. Without this, a burst beyond the
        encoder's capacity queued without bound, and callers hit their timeouts (the P6.4 load test)."""
        limit = self.settings.max_encoder_wait_ms
        enc = self.manager.encoder
        wait = getattr(enc, "expected_wait_s", None)
        per = getattr(enc, "time_per_text_s", None)
        pending = getattr(enc, "pending", None)
        if not limit or not callable(wait):
            return False
        if callable(pending) and pending() == 0 and self._routing == 0:
            # An idle encoder is not backed up, whatever it measured last. Letting this request through also
            # refreshes the estimate: one slow batch used to latch the gate shut until restart (audit run 2).
            return False
        # requests already routing wait for a worker thread before they reach the encoder's queue
        ahead = max(wait(), (self._routing + 1) * per()) if callable(per) else wait()
        if ahead * 1000 <= limit:
            return False
        self.metrics.questions.inc("overloaded")
        return True

    async def _plain_forward(self, request: Request, body: bytes, kh: str | None, reason: str, t0: float,
                             log_rec: dict) -> Response:
        """Forward without routing or recording anything."""
        self.stats["passthrough"] += 1
        resp = await self.forward(request, body, kh)
        log_rec["reason"] = reason
        return self._upstream_reply(resp, None, t0, log_rec)

    def _upstream_reply(self, resp: httpx.Response | Response, reasons: dict[str, str] | None, t0: float,
                        log_rec: dict) -> Response:
        self.stats["forwarded"] += 1
        self.metrics.requests.inc("systemone", "upstream")
        self.metrics.latency.observe(time.perf_counter() - t0, "upstream")
        extra = {"x-jevstiller-source": "upstream"}
        if reasons:
            for r in reasons.values():
                self.metrics.questions.inc(r)
            detail: Any = reasons if len(reasons) <= MAX_DETAIL_QUESTIONS else {"questions": len(reasons)}
            extra["x-jevstiller-detail"] = json.dumps(detail, separators=(",", ":"))
        out = self.relay(resp, extra)
        self._log(log_rec, "upstream", out.status_code, t0, out.headers.get(REQUEST_ID), reasons)
        return out

    def _log(self, rec: dict, source: str, status: int, t0: float, request_id: str | None,
             reasons: dict[str, str] | None = None) -> None:
        if not access_log.isEnabledFor(logging.INFO):
            return
        rec.update(source=source, status=status, request_id=request_id,
                   latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        if reasons:
            counts: dict[str, int] = {}
            for r in reasons.values():
                counts[r] = counts.get(r, 0) + 1
            rec["reasons"] = counts
        access_log.info(json.dumps(rec, separators=(",", ":")))

    async def _route(self, groups: Mapping[str, list[str]], specs: Mapping[str, dict], tenant: str,
                     req: dict, answered: set[str] | None = None) -> tuple[dict[str, Routing] | None, list[str]]:
        """Route each distinct question once. Returns (routings by spec, specs that can't be tasks), or
        (None, []) if routing failed unexpectedly (everything routed so far is released).

        `answered`: the specs Jev has already answered usably (the unverified-key path routes after Jev's
        answer). Otherwise Jev hasn't answered yet, and questions that aren't tasks yet are not counted towards
        admission here: `_finish` counts those Jev answers (a question Jev rejects never becomes a task)."""
        routings: dict[str, Routing] = {}
        unsupported: list[str] = []
        self._routing += 1
        try:
            # the state is the same for every question: canonical text and embedding once per request, and the
            # embedding only if some question is a task (requests that are only forwarded never pay for it)
            stexts = [state_text(req["state"])]
            memo: list = []

            def X():
                if not memo:
                    memo.append(self.manager.encoder.encode(stexts))
                return memo[0]
            for spec in groups:
                q = specs[spec]
                try:
                    admit = answered is not None and spec in answered
                    routings[spec] = await anyio.to_thread.run_sync(
                        lambda q=q, admit=admit: self.manager.route(
                            tenant, q.get("instructions"), q["criteria"], [req["state"]], req["model"],
                            stexts=stexts, X=X, admit=admit))
                except ValueError:
                    unsupported.append(spec)            # not a valid task (e.g. one class): Jev decides
        except Exception:
            log.exception("routing failed; forwarding")
            self._defer_all(routings, "proxy_error")
            await self._finish(routings, groups, None, None, 0.0)   # releases; nothing is recorded
            self.stats["errors"] += 1
            return None, []
        finally:
            self._routing -= 1
        return routings, unsupported

    @staticmethod
    def _defer_all(routings: Mapping[str, Routing], reason: str) -> None:
        """The request goes to Jev as a whole: nothing in it is answered by a student."""
        for r in routings.values():
            if r.engine is not None:
                for i in range(len(r.routed.recs)):
                    r.engine.defer(r.routed, i, reason)

    def _teacher_output(self, a: Any, labels: Sequence[str], tokens: int, latency_ms: float, rid: str | None,
                        model: Any) -> TeacherOutput | Exception:
        try:
            if not (isinstance(a, dict) and a.get("type") == "choice" and isinstance(a.get("probabilities"), dict)):
                raise ValueError("no choice answer")
            probs = {c: float(a["probabilities"].get(c, 0.0)) for c in labels}
            z = sum(probs.values()) or 1.0
            return TeacherOutput(label=str(a.get("choice")), probs={c: p / z for c, p in probs.items()},
                                 confidence=float(a.get("confidence", 0.0)), input_tokens=tokens,
                                 cost_usd=tokens * self.settings.price_per_mtok / 1e6, latency_ms=latency_ms,
                                 request_id=rid, model=f"jev:{model}" if isinstance(model, str) and model else None)
        except Exception as e:
            return RuntimeError(f"no usable teacher answer: {e}")

    async def _finish(self, routings: Mapping[str, Routing], groups: Mapping[str, list[str]], answer: dict | None,
                      resp: httpx.Response | Response | None, latency_ms: float) -> list[str]:
        """Complete every routing (always: it releases the engine) with Jev's answer to its question, or an
        error, which is not recorded. Questions not admitted yet count towards admission only if Jev answered
        them usably; returns the specs that became tasks just now. Shielded from cancellation: an engine that is
        never released stays loaded and undeletable."""
        with anyio.CancelScope(shield=True):
            return await self._finish_all(routings, groups, answer, resp, latency_ms)

    async def _finish_all(self, routings: Mapping[str, Routing], groups: Mapping[str, list[str]],
                          answer: dict | None, resp: httpx.Response | Response | None,
                          latency_ms: float) -> list[str]:
        admitted: list[str] = []
        try:
            tokens = int((answer or {}).get("usage", {}).get("input_tokens") or 0) // max(1, len(routings))
        except Exception:
            tokens = 0
        model = (answer or {}).get("model")
        rid = resp.headers.get(REQUEST_ID) if resp is not None else None
        answers = (answer or {}).get("answers") or {}
        for spec, r in routings.items():
            first = groups[spec][0]
            if r.engine is None:
                if r.uncounted and _usable(answers.get(first)):
                    try:
                        if await anyio.to_thread.run_sync(self.manager.observe, r):
                            admitted.append(spec)
                    except Exception:
                        log.exception("admission count for task %s failed", r.key)
                continue
            out = self._teacher_output(answers.get(first), r.task.labels, tokens, latency_ms, rid, model)
            try:
                await anyio.to_thread.run_sync(self.manager.complete, r, [out] * len(r.routed.to_teacher), "jev")
            except Exception:
                log.exception("recording task %s failed", r.key)
        return admitted

    async def _local_response(self, req: dict, groups: Mapping[str, list[str]],
                              routings: Mapping[str, Routing]) -> Response:
        answers: dict[str, Any] = {}
        detail: dict[str, str] = {}
        model = None
        released: set[str] = set()
        try:
            for spec, names in groups.items():
                r = routings[spec]
                released.add(spec)                       # complete() releases the engine even if it raises
                [res] = await anyio.to_thread.run_sync(self.manager.complete, r, [], "jev")
                for name in names:
                    answers[name] = {"type": "choice", "choice": res.label,
                                     "confidence": _peakedness(res.probs), "probabilities": res.probs}
                    detail[name] = res.source
                lineage = r.engine._teacher_model or ""
                model = model or (lineage[4:] if lineage.startswith("jev:") else None)
        finally:
            for spec, r in routings.items():             # a failure part-way: release the rest unrecorded
                if spec not in released and r.engine is not None:
                    self.manager._release(r.key)
        answers = {name: answers[name] for name in req["questions"] if name in answers}   # request order
        body = {"model": model or req["model"], "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}
        d: Any = detail if len(detail) <= MAX_DETAIL_QUESTIONS else {"questions": len(detail)}
        return JSONResponse(body, headers={REQUEST_ID: f"jvs_{uuid.uuid4().hex}", "x-jevstiller-source": "local",
                                           "x-jevstiller-detail": json.dumps(d, separators=(",", ":"))})

    # ---- local endpoints ------------------------------------------------------------
    async def healthz(self, request: Request) -> Response:
        """Liveness: the process answers. No counters here (it is usually unauthenticated)."""
        if request.method not in ("GET", "HEAD"):
            return _error(405, "method not allowed")
        return JSONResponse({"ok": True})


class _TooLarge(Exception):
    pass


_BAD_PATH_CHARS = re.compile(r"[\x00-\x1f\x7f?#]")


class AccessMiddleware:
    """Who may use the proxy at all, checked before anything else.

    GET/HEAD /healthz and /readyz (probes) skip the network and token checks; every other request must pass:
    client network (`allow_networks`, against the peer uvicorn reports), the shared token, a single
    Authorization header, no `.`/`..` path segments, and the body limit, enforced on the declared
    Content-Length and again while the body streams in (chunked bodies included).
    """

    def __init__(self, app, settings: ProxySettings, metrics: ProxyMetrics | None = None):
        import ipaddress
        self.app, self.settings, self.metrics = app, settings, metrics
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

    async def _refuse(self, reason: str, status: int, detail: str, scope, receive, send) -> None:
        if self.metrics is not None:
            self.metrics.rejected.inc(reason)
        await _error(status, f"{detail} (jevstiller)")(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope.get("path", ""), scope.get("method", "")
        if path in PROBE_PATHS and method in ("GET", "HEAD"):
            return await self.app(scope, receive, send)
        raw = scope.get("headers", [])
        if not self._client_allowed(scope):
            return await self._refuse("network", 403, "client network not allowed", scope, receive, send)
        token = self.settings.access_token
        if token:
            given = [v for k, v in raw if k.lower() == PROXY_TOKEN_HEADER.encode()]
            if len(given) != 1 or not hmac.compare_digest(given[0], token.encode()):
                return await self._refuse("token", 401, "missing or invalid x-jevstiller-token",
                                          scope, receive, send)
        if sum(1 for k, _ in raw if k.lower() == b"authorization") > 1:
            return await self._refuse("duplicate_auth", 400, "more than one Authorization header",
                                      scope, receive, send)
        if any(seg in (".", "..") for seg in path.split("/")) or _BAD_PATH_CHARS.search(path):
            # control characters, ? and # (decoded from %xx) would be dropped or split off when the path is
            # rebuilt into a URL, turning a harmless-looking segment into ".." upstream (audit run 2)
            return await self._refuse("path", 400, "invalid path", scope, receive, send)
        limit = self.settings.max_body_bytes
        lengths = [v for k, v in raw if k.lower() == b"content-length"]
        try:
            if lengths and int(lengths[-1]) > limit:
                return await self._refuse("body", 413, "request body too large", scope, receive, send)
        except ValueError:
            return await self._refuse("body", 400, "invalid content-length", scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _TooLarge()
            return message
        try:
            await self.app(scope, limited_receive, send)
        except _TooLarge:
            await self._refuse("body", 413, "request body too large", scope, receive, send)


def create_app(manager: TaskManager, settings: ProxySettings | None = None, keys: KeyRegistry | None = None,
               client: httpx.AsyncClient | None = None, closers: Sequence[Callable[[], Any]] = (),
               admin_token: str | None = None, metrics_public: bool = False,
               ready: Callable[[], dict[str, bool]] | None = None) -> Starlette:
    """The proxy as an ASGI app.

    - `admin_token` enables the admin API under /jevstiller/v1 and protects GET /metrics (unless
      `metrics_public`). Neither is ever forwarded upstream.
    - `ready()` returns named readiness checks for GET /readyz (all must be True); default: manager open.
    - On shutdown: close the upstream client and the manager, then call `closers` in order.
    """
    settings = settings or ProxySettings()
    keys = keys or KeyRegistry(secrets.token_bytes(32), settings.key_ttl_s)
    registry = MetricsRegistry()
    metrics = ProxyMetrics(registry)
    proxy = Proxy(manager, settings, keys, client, metrics)
    admin = Admin(manager, admin_token, stats=lambda: {"proxy": dict(proxy.stats)})
    _task_gauges(registry, manager, keys)

    def _ready() -> dict[str, bool]:
        checks = {"manager": not manager._stop.is_set()}
        if ready is not None:
            checks.update(ready())
        return checks

    async def readyz(request: Request) -> Response:
        if request.method not in ("GET", "HEAD"):
            return _error(405, "method not allowed")
        checks = await anyio.to_thread.run_sync(_ready)
        ok = all(checks.values())
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    async def metrics_endpoint(request: Request) -> Response:
        if not metrics_public:
            if (denied := admin.authorized(request)) is not None:
                return denied
        if request.method != "GET":
            return _error(405, "method not allowed")
        return Response(await anyio.to_thread.run_sync(registry.render),
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    async def local_not_found(request: Request) -> Response:
        return _error(404, "not found")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        for c in proxy.clients:
            await c.aclose()
        await anyio.to_thread.run_sync(manager.close)
        for close in closers:
            await anyio.to_thread.run_sync(close)

    async def client_gone(request: Request, exc: Exception) -> Response:
        # the caller hung up while sending its request: nobody to answer, and not the proxy's error
        return Response(status_code=499)

    app = Starlette(exception_handlers={ClientDisconnect: client_gone}, routes=[
        Route(SYSTEM_ONE, proxy.system_one, methods=["POST"]),
        Route("/healthz", proxy.healthz, methods=ALL_METHODS),
        Route("/readyz", readyz, methods=ALL_METHODS),
        Route("/metrics", metrics_endpoint, methods=ALL_METHODS),
        *admin.routes(),
        Route("/jevstiller/{rest:path}", local_not_found, methods=ALL_METHODS),   # never forwarded
        Route("/{path:path}", proxy.passthrough, methods=ALL_METHODS),
    ], lifespan=lifespan)
    app.state.proxy = proxy
    app.state.metrics = registry
    app.add_middleware(AccessMiddleware, settings=settings, metrics=metrics)
    return app


def _task_gauges(registry: MetricsRegistry, manager: TaskManager, keys: KeyRegistry) -> None:
    def tasks():
        return [((), len(manager.tasks()))]

    def loaded():
        return [((), len(manager.loaded()))]

    def memory():
        return [((), manager.memory_mb() * 2**20)]

    def training():
        stats = getattr(manager.train_executor, "stats", None)
        if not callable(stats):
            return []
        s = stats()
        return [(("running",), s["running"]), (("queued",), s["queued"])]

    def verified_keys():
        now = time.monotonic()
        with keys._lock:
            return [((), sum(1 for t in keys._verified.values() if now - t < keys.ttl_s))]

    def per_task():
        out = []
        for key in manager.loaded():
            with manager._lock:
                e = manager._engines.get(key)
            if e is not None:
                out.append(((key, e._prod.name if e._prod else "-", e.mode), e.training_priority()))
        return out

    def dropped():
        from .store import dropped_records
        return [((), dropped_records())]

    def encoder_wait():
        wait = getattr(manager.encoder, "expected_wait_s", None)
        return [((), wait())] if callable(wait) else []

    registry.gauge("jevstiller_tasks", "Registered tasks", fn=tasks)
    registry.gauge("jevstiller_encoder_wait_seconds", "How long a request would now wait for the shared encoder",
                   fn=encoder_wait)
    registry.counter_fn("jevstiller_store_dropped_records_total",
                        "Records the sample stores failed to write (disk full, I/O errors)", fn=dropped)
    registry.gauge("jevstiller_tasks_loaded", "Tasks loaded in memory", fn=loaded)
    registry.gauge("jevstiller_student_memory_bytes", "Memory held by loaded students", fn=memory)
    registry.gauge("jevstiller_training_jobs", "Training jobs by state", ["state"], fn=training)
    registry.gauge("jevstiller_verified_keys", "API keys accepted by Jev within the TTL", fn=verified_keys)
    registry.gauge("jevstiller_task_teacher_calls_per_minute", "Recent teacher-call rate per loaded task",
                   ["task", "production", "mode"], fn=per_task)
