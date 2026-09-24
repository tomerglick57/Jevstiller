"""Jev (TypeSafe) adapter over `typesafe-sdk`. Requires TYPESAFE_API_KEY.

One request per text: state=text, one Choice question whose criteria are the task's
class descriptions. The full probability distribution is the training target.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from ..task import Task
from . import TeacherOutput


class _TokenBucket:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_at)
            self.next_at = t + self.interval
        if t > now:
            time.sleep(t - now)


class JevTeacher:
    name = "jev"

    def __init__(self, model: str = "jev-1.13.0", api_key: str | None = None, rpm: int = 1100,
                 concurrency: int = 8, price_per_mtok: float = 0.042, question_id: str = "label",
                 max_retries: int = 5, timeout: float = 10.0):
        try:
            from typesafe_sdk import RetryPolicy, TypeSafeClient  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'jevstiller[jev]' to use JevTeacher") from e
        self.model = model
        self.name = f"jev:{model}"
        self.price_per_mtok = price_per_mtok
        self.question_id = question_id
        self.concurrency = concurrency
        self.bucket = _TokenBucket(rpm)
        self.client = TypeSafeClient(api_key=api_key, model=model, retry=RetryPolicy(max_retries=max_retries),
                                     timeout=timeout)
        self._Choice = __import__("typesafe_sdk").Choice

    def _one(self, text: str, task: Task) -> TeacherOutput:
        self.bucket.wait()
        q = {self.question_id: self._Choice(instructions=task.instructions, criteria=dict(task.classes))}
        t0 = time.perf_counter()
        res = self.client.system_one(state=text, questions=q)
        dt = (time.perf_counter() - t0) * 1000
        ans = res.choices[self.question_id]
        probs = {c: float(ans.probabilities.get(c, 0.0)) for c in task.labels}
        z = sum(probs.values()) or 1.0
        probs = {c: p / z for c, p in probs.items()}
        toks = 0
        usage = getattr(res, "usage", None)
        if usage is not None and getattr(usage, "input_tokens", None) is not None:
            toks = int(usage.input_tokens)
        raw = None
        try:
            raw = res.raw_http_response.json()
            toks = toks or int(raw.get("usage", {}).get("input_tokens", 0))
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            request_id = res.request_id       # the SDK raises TypeSafeError when the header is missing
        except Exception:
            request_id = None
        return TeacherOutput(label=str(ans.choice), probs=probs, confidence=float(ans.confidence),
                             input_tokens=toks, cost_usd=toks * self.price_per_mtok / 1e6,
                             latency_ms=dt, request_id=request_id, raw=raw)

    def _safe(self, text: str, task: Task) -> TeacherOutput | Exception:
        try:
            return self._one(text, task)
        except Exception as e:                # one failed request fails only its own item
            return e

    def classify(self, texts: Sequence[str], task: Task) -> list[TeacherOutput | Exception]:
        if len(texts) == 1:
            return [self._safe(texts[0], task)]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(lambda t: self._safe(t, task), texts))
