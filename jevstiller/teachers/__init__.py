"""Teacher protocol and adapters."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..task import Task


@dataclass
class TeacherOutput:
    label: str
    probs: dict[str, float]          # full distribution over task.labels
    confidence: float                # teacher's own scalar
    input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    request_id: str | None = None
    raw: Any = None


@runtime_checkable
class Teacher(Protocol):
    """`classify` returns one entry per text, in order. An entry may be an Exception instead of a
    TeacherOutput when only that item failed; raising fails every item of the call."""

    name: str

    def classify(self, texts: Sequence[str], task: Task) -> list[TeacherOutput | Exception]: ...


def peakedness(probs: dict[str, float]) -> float:
    """Jev-style confidence: (K*max - 1) / (K - 1); 0 for uniform, 1 for one-hot."""
    k = len(probs)
    if k < 2:
        return 1.0
    return max(0.0, (k * max(probs.values()) - 1.0) / (k - 1.0))


from .replay import CachedTeacher, ReplayTeacher  # noqa: E402
from .synthetic import SyntheticTeacher, SyntheticWorld  # noqa: E402

__all__ = ["Teacher", "TeacherOutput", "peakedness", "SyntheticTeacher", "SyntheticWorld", "ReplayTeacher",
           "CachedTeacher"]
