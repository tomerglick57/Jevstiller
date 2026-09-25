"""Admin API, under `/jevstiller/v1/` (never forwarded to Jev). Off unless an admin token is configured; every
call needs `Authorization: Bearer <admin token>`.

    GET    /jevstiller/v1/tasks[?tenant=]            list tasks
    GET    /jevstiller/v1/tasks/{key}                status (+ text report) of one task
    GET    /jevstiller/v1/tasks/{key}/versions       its student versions
    POST   /jevstiller/v1/tasks/{key}/mode           {"mode": "auto" | "teacher_only" | "cascade"}   (persisted)
    POST   /jevstiller/v1/tasks/{key}/target         {"target_agreement": 0.99}                      (persisted)
    POST   /jevstiller/v1/tasks/{key}/train          train a candidate now (waits for it)
    POST   /jevstiller/v1/tasks/{key}/promote        {"version": "student:v3"}
    POST   /jevstiller/v1/tasks/{key}/rollback
    DELETE /jevstiller/v1/tasks/{key}                delete the task and all its data
    DELETE /jevstiller/v1/tenants/{tenant}           delete every task of a tenant
    GET    /jevstiller/v1/stats                      proxy, manager and scheduler counters

    GET    /jevstiller/status                        a read-only HTML page (browsers: log in with any user name
                                                     and the admin token as the password)
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import hmac
import math
import re
import time
from collections.abc import Callable
from typing import Any

import anyio
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from ._manager import TaskManager

PREFIX = "/jevstiller/v1"
PAGE_CACHE_S = 5.0                                      # the status page is redrawn at most this often
_KEY = re.compile(r"^[0-9a-f]{20}$")


def _json_safe(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _json_safe(dataclasses.asdict(x))
    return x


def _ok(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(_json_safe(data), status_code=status)


def _err(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


class Admin:
    def __init__(self, manager: TaskManager, token: str | None, stats: Callable[[], dict] | None = None,
                 version: str = ""):
        self.manager, self.token, self.stats_fn, self.version = manager, token, stats, version
        self._page: tuple[float, str, str] | None = None   # (rendered at, html, style nonce)

    def authorized(self, request: Request, basic: bool = False) -> Response | None:
        """None when the call may proceed, else the response to send. `basic`: also accept the admin token as the
        password of HTTP Basic auth (any user name), for browsers."""
        if not self.token:
            return _err(404, "not found")                    # the admin API is off
        auth = request.headers.get("authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if basic and auth.lower().startswith("basic "):
            try:
                given = base64.b64decode(auth[6:].strip(), validate=True).decode().partition(":")[2]
            except (binascii.Error, UnicodeDecodeError):
                given = ""
        if not hmac.compare_digest(given.encode(), self.token.encode()):
            if basic:
                return Response("admin token required (as the password)\n", 401, media_type="text/plain",
                                headers={"www-authenticate": 'Basic realm="jevstiller", charset="UTF-8"'})
            return _err(401, "admin token required")
        return None

    def _key(self, request: Request) -> str | None:
        key = request.path_params.get("key", "")
        return key if _KEY.match(key) and key in {i.key for i in self.manager.tasks()} else None

    async def _with_engine(self, key: str, fn: Callable[[Any], Any]) -> Any:
        """`fn(engine)` in a worker thread with the task's engine held open throughout (an unload or delete
        closing it under a running call crashed the process: security audit run 2). None if the task is gone."""
        def run():
            try:
                with self.manager.hold(key) as engine:
                    return fn(engine)
            except KeyError:                                # deleted meanwhile
                return None
        return await anyio.to_thread.run_sync(run)

    async def _body(self, request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    # ---- handlers -----------------------------------------------------------------
    async def tasks(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        loaded = set(self.manager.loaded())
        tenant = request.query_params.get("tenant")
        return _ok([{"key": i.key, "tenant": i.tenant, "model": i.model, "classes": len(i.classes),
                     "target_agreement": i.target_agreement, "mode": i.mode, "created": i.created,
                     "last_seen": i.last_seen, "loaded": i.key in loaded} for i in self.manager.tasks(tenant)])

    async def task(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        st = await self._with_engine(key, lambda e: e.status())
        info = next((i for i in self.manager.tasks() if i.key == key), None)
        if st is None or info is None:
            return _err(404, "unknown task")
        return _ok({"info": info, "status": st, "report": st.report()})

    async def versions(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        versions = await self._with_engine(key, lambda e: e.versions())
        return _ok(versions) if versions is not None else _err(404, "unknown task")

    async def mode(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        mode = (await self._body(request)).get("mode")
        try:
            await anyio.to_thread.run_sync(self.manager.set_mode, key, mode)
        except ValueError as e:
            return _err(422, str(e))
        return _ok({"key": key, "mode": mode})

    async def target(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        value = (await self._body(request)).get("target_agreement")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return _err(422, "target_agreement must be a number in [0.5, 1)")
        try:
            await anyio.to_thread.run_sync(self.manager.set_target, key, float(value))
        except ValueError as e:
            return _err(422, str(e))
        return _ok({"key": key, "target_agreement": float(value)})

    async def train(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        try:
            report = await self._with_engine(key, lambda e: e.train_now())
        except Exception as e:
            return _err(500, f"training failed: {type(e).__name__}")
        return _ok(report) if report is not None else _err(404, "unknown task")

    async def promote(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        version = (await self._body(request)).get("version")
        try:
            if await self._with_engine(key, lambda e: e.promote(str(version)) or True) is None:
                return _err(404, "unknown task")
        except ValueError as e:
            return _err(422, str(e))
        return _ok({"key": key, "production": version})

    async def rollback(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        target = await self._with_engine(key, lambda e: (e.rollback(),))
        if target is None:
            return _err(404, "unknown task")
        return _ok({"key": key, "production": target[0]})

    async def delete_task(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        if (key := self._key(request)) is None:
            return _err(404, "unknown task")
        if not await anyio.to_thread.run_sync(self.manager.delete, key, "admin API"):
            return _err(409, "a request is in flight for this task; retry")
        return _ok({"deleted": key})

    async def delete_tenant(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        tenant = request.path_params["tenant"]
        deleted = await anyio.to_thread.run_sync(self.manager.delete_tenant, tenant)
        left = [i.key for i in self.manager.tasks(tenant)]
        return _ok({"deleted": deleted, "remaining": left}, 200 if not left else 409)

    async def stats(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        m = self.manager
        data = {"tasks": len(m.tasks()), "loaded": len(m.loaded()), "loads": m.loads, "unloads": m.unloads,
                "student_memory_mb": m.memory_mb()}
        sched = getattr(m.train_executor, "stats", None)
        if callable(sched):
            data["training"] = sched()
        if self.stats_fn:
            data.update(self.stats_fn())
        return _ok(data)

    async def status_page(self, request: Request) -> Response:
        if (denied := self.authorized(request, basic=True)) is not None:
            return denied
        if request.method not in ("GET", "HEAD"):
            return _err(405, "method not allowed")
        now = time.monotonic()
        if self._page is None or now - self._page[0] > PAGE_CACHE_S:   # drawing it reads every loaded task's store
            from ._status_page import render
            stats = (self.stats_fn() if self.stats_fn else {}).get("proxy")
            page, nonce = await anyio.to_thread.run_sync(lambda: render(self.manager, stats, self.version))
            self._page = (now, page, nonce)
        _, page, nonce = self._page
        return HTMLResponse(page, headers={
            "content-security-policy": (f"default-src 'none'; style-src 'nonce-{nonce}'; base-uri 'none'; "
                                        "form-action 'none'; frame-ancestors 'none'"),
            "x-frame-options": "DENY", "x-content-type-options": "nosniff", "referrer-policy": "no-referrer",
            "cache-control": "no-store"})

    def routes(self) -> list[Route]:
        p = PREFIX
        return [
            Route("/jevstiller/status", self.status_page, methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE",
                                                                   "OPTIONS"]),
            Route(f"{p}/tasks", self.tasks, methods=["GET"]),
            Route(f"{p}/tasks/{{key}}", self.task, methods=["GET"]),
            Route(f"{p}/tasks/{{key}}", self.delete_task, methods=["DELETE"]),
            Route(f"{p}/tasks/{{key}}/versions", self.versions, methods=["GET"]),
            Route(f"{p}/tasks/{{key}}/mode", self.mode, methods=["POST"]),
            Route(f"{p}/tasks/{{key}}/target", self.target, methods=["POST"]),
            Route(f"{p}/tasks/{{key}}/train", self.train, methods=["POST"]),
            Route(f"{p}/tasks/{{key}}/promote", self.promote, methods=["POST"]),
            Route(f"{p}/tasks/{{key}}/rollback", self.rollback, methods=["POST"]),
            Route(f"{p}/tenants/{{tenant}}", self.delete_tenant, methods=["DELETE"]),
            Route(f"{p}/stats", self.stats, methods=["GET"]),
            Route(f"{p}/{{rest:path}}", self._not_found,
                  methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]),
        ]

    async def _not_found(self, request: Request) -> Response:
        if (denied := self.authorized(request)) is not None:
            return denied
        return _err(404, "not found")
