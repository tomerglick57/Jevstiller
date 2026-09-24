"""Serve previously recorded teacher answers (from a sample store or a dict). Free and offline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..task import Task
from . import TeacherOutput


class ReplayTeacher:
    name = "replay"

    def __init__(self, answers: Mapping[str, TeacherOutput], fallback=None):
        """`answers` maps text -> TeacherOutput. `fallback` is an optional live teacher for misses."""
        self.answers = dict(answers)
        self.fallback = fallback
        self.misses = 0

    def classify(self, texts: Sequence[str], task: Task) -> list[TeacherOutput | Exception]:
        out: list[TeacherOutput | None] = [self.answers.get(t) for t in texts]
        missing = [i for i, o in enumerate(out) if o is None]
        if missing:
            if self.fallback is None:
                raise KeyError(f"{len(missing)} texts not in replay and no fallback teacher")
            self.misses += len(missing)
            fresh = self.fallback.classify([texts[i] for i in missing], task)
            for i, o in zip(missing, fresh, strict=True):
                out[i] = o
                if not isinstance(o, Exception):
                    self.answers[texts[i]] = o
        return out  # type: ignore[return-value]


class CachedTeacher:
    """Wraps a live teacher; persists every answer to a JSONL file keyed by (task version, text).

    Re-running an experiment costs nothing after the first pass.
    """

    def __init__(self, inner, path):
        import json
        from pathlib import Path
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, TeacherOutput] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    d = json.loads(line)
                    self.cache[d["key"]] = TeacherOutput(d["label"], d["probs"], d["confidence"], d["input_tokens"],
                                                         d["cost_usd"], d["latency_ms"], d.get("request_id"))

    def classify(self, texts: Sequence[str], task: Task) -> list[TeacherOutput | Exception]:
        import json
        keys = [f"{task.version}\t{t}" for t in texts]
        out: list[TeacherOutput | None] = [self.cache.get(k) for k in keys]
        missing = [i for i, o in enumerate(out) if o is None]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = self.inner.classify([texts[i] for i in missing], task)
            with open(self.path, "a") as f:
                for i, o in zip(missing, fresh, strict=True):
                    out[i] = o
                    if isinstance(o, Exception):
                        continue
                    self.cache[keys[i]] = o
                    f.write(json.dumps({"key": keys[i], "label": o.label, "probs": o.probs, "confidence": o.confidence,
                                        "input_tokens": o.input_tokens, "cost_usd": o.cost_usd,
                                        "latency_ms": o.latency_ms, "request_id": o.request_id}) + "\n")
        return out  # type: ignore[return-value]
