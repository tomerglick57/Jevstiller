"""Prometheus metrics in the text exposition format, without a dependency.

Counters and histograms are updated on the request path (a lock and a few additions); gauges are computed when
`/metrics` is scraped (callbacks). Label values must come from small, known sets (sources, reasons, statuses, task
keys of loaded tasks) — never from request content.
"""
from __future__ import annotations

import bisect
import math
import threading
from collections.abc import Callable, Iterable, Sequence

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _escape(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(names: Sequence[str], values: Sequence[str], extra: str = "") -> str:
    parts = [f'{n}="{_escape(v)}"' for n, v in zip(names, values, strict=True)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


def _num(x: float) -> str:
    if math.isinf(x):
        return "+Inf" if x > 0 else "-Inf"
    return repr(float(x)) if not float(x).is_integer() else str(int(x))


class _Metric:
    kind = ""

    def __init__(self, name: str, help_: str, labels: Sequence[str] = ()):
        self.name, self.help, self.labelnames = name, help_, tuple(labels)
        self._lock = threading.Lock()

    def header(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]


class Counter(_Metric):
    kind = "counter"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, *labels: str, by: float = 1.0) -> None:
        with self._lock:
            self._values[labels] = self._values.get(labels, 0.0) + by

    def value(self, *labels: str) -> float:
        with self._lock:
            return self._values.get(labels, 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        return self.header() + [f"{self.name}{_labels(self.labelnames, k)} {_num(v)}" for k, v in items]


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, name: str, help_: str, labels: Sequence[str] = (), buckets: Sequence[float] = LATENCY_BUCKETS):
        super().__init__(name, help_, labels)
        self.buckets = tuple(sorted(buckets))
        self._data: dict[tuple[str, ...], list] = {}          # labels -> [bucket counts..., count, sum]

    def observe(self, value: float, *labels: str) -> None:
        i = bisect.bisect_left(self.buckets, value)
        with self._lock:
            d = self._data.setdefault(labels, [0] * len(self.buckets) + [0, 0.0])
            if i < len(self.buckets):
                d[i] += 1
            d[-2] += 1
            d[-1] += value

    def render(self) -> list[str]:
        with self._lock:
            items = sorted((k, list(v)) for k, v in self._data.items())
        out = self.header()
        for k, d in items:
            acc = 0
            for b, c in zip(self.buckets, d[:len(self.buckets)], strict=True):
                acc += c
                le = 'le="' + _num(b) + '"'
                out.append(f"{self.name}_bucket{_labels(self.labelnames, k, le)} {acc}")
            inf = 'le="+Inf"'
            out.append(f"{self.name}_bucket{_labels(self.labelnames, k, inf)} {d[-2]}")
            out.append(f"{self.name}_count{_labels(self.labelnames, k)} {d[-2]}")
            out.append(f"{self.name}_sum{_labels(self.labelnames, k)} {_num(d[-1])}")
        return out


class Gauge(_Metric):
    """Computed at scrape time: `fn()` returns an iterable of (label values, value)."""

    kind = "gauge"

    def __init__(self, name: str, help_: str, labels: Sequence[str] = (),
                 fn: Callable[[], Iterable[tuple[Sequence[str], float]]] | None = None):
        super().__init__(name, help_, labels)
        self.fn = fn

    def render(self) -> list[str]:
        try:
            items = list(self.fn()) if self.fn else []
        except Exception:                                    # a broken callback must not break the scrape
            items = []
        return self.header() + [f"{self.name}{_labels(self.labelnames, tuple(k))} {_num(v)}" for k, v in items]


class ComputedCounter(Gauge):
    """A counter read at scrape time from something that already counts (`fn` as for Gauge)."""

    kind = "counter"


class Registry:
    def __init__(self):
        self.metrics: list[_Metric] = []

    def add(self, m: _Metric) -> _Metric:
        self.metrics.append(m)
        return m

    def counter(self, name: str, help_: str, labels: Sequence[str] = ()) -> Counter:
        return self.add(Counter(name, help_, labels))            # type: ignore[return-value]

    def histogram(self, name: str, help_: str, labels: Sequence[str] = (),
                  buckets: Sequence[float] = LATENCY_BUCKETS) -> Histogram:
        return self.add(Histogram(name, help_, labels, buckets))  # type: ignore[return-value]

    def gauge(self, name: str, help_: str, labels: Sequence[str] = (),
              fn: Callable[[], Iterable[tuple[Sequence[str], float]]] | None = None) -> Gauge:
        return self.add(Gauge(name, help_, labels, fn))           # type: ignore[return-value]

    def counter_fn(self, name: str, help_: str, labels: Sequence[str] = (),
                   fn: Callable[[], Iterable[tuple[Sequence[str], float]]] | None = None) -> ComputedCounter:
        return self.add(ComputedCounter(name, help_, labels, fn))  # type: ignore[return-value]

    def render(self) -> str:
        lines: list[str] = []
        for m in self.metrics:
            lines += m.render()
        return "\n".join(lines) + "\n"
