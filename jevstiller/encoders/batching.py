"""One encoder shared by many tasks and threads, with concurrent calls merged into batches."""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future

import numpy as np

MAX_CHARS = 32_768        # characters of each text the encoder sees: far beyond a 256-token window


class BatchingEncoder:
    """Wraps an encoder so that concurrent `encode` calls run as one batch.

    A single worker thread runs the inner encoder. While it works, new calls queue up; the next batch takes
    everything queued (up to `max_batch` texts). Idle traffic sees no added latency; under load, batches grow
    by themselves, which is what a GPU (or ONNX Runtime's threaded kernels) needs to be efficient.
    `max_wait_ms` > 0 additionally holds a batch open that long to collect more calls.

    Calls with at least `max_batch` texts bypass the queue. Same `id` and `dim` as the inner encoder, so
    stored embeddings stay in one lineage.

    Texts are cut to `max_chars` first. Model encoders read only the first 256 tokens (~1-2k characters of
    prose) but tokenize the whole text before truncating, and the hash encoder reads all of it, so without a
    cap one 4 MB request took seconds of the shared encoder's time (security audit run 2).

    `expected_wait_s()` estimates how long a new call would wait (texts queued or being encoded, times the
    measured time per text), so a caller can do something else when the encoder is saturated (the proxy
    forwards such requests to the teacher). `pending()` is the number of texts queued or being encoded.
    """

    def __init__(self, inner, max_batch: int = 256, max_wait_ms: float = 0.0, max_chars: int = MAX_CHARS):
        self.inner = inner
        self.id, self.dim = inner.id, inner.dim
        self.max_batch, self.max_wait_s, self.max_chars = max_batch, max_wait_ms / 1000, max_chars
        self.batches = self.calls = 0                  # counters, for tests and metrics
        self._inflight = 0                             # texts in the batch being encoded
        self._per_text_s = 0.0                         # moving average of encode time per text
        self._q: deque[tuple[list[str], Future]] = deque()
        self._queued = 0
        self._cv = threading.Condition()
        self._closed = False
        self._worker: threading.Thread | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t[:self.max_chars] for t in texts]
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        if len(texts) >= self.max_batch:
            return self.inner.encode(texts)
        fut: Future = Future()
        with self._cv:
            if self._closed:
                raise RuntimeError("encoder is closed")
            self._q.append((texts, fut))
            self._queued += len(texts)
            self.calls += 1
            if self._worker is None:
                self._worker = threading.Thread(target=self._loop, daemon=True, name="jevstiller-encoder")
                self._worker.start()
            self._cv.notify_all()
        return fut.result()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._closed:
                    self._cv.wait()
                if not self._q:
                    return
                if self.max_wait_s > 0 and self._queued < self.max_batch:
                    self._cv.wait_for(lambda: self._queued >= self.max_batch or self._closed, self.max_wait_s)
                batch: list[tuple[list[str], Future]] = []
                n = 0
                while self._q and (not batch or n + len(self._q[0][0]) <= self.max_batch):
                    texts, fut = self._q.popleft()
                    batch.append((texts, fut))
                    n += len(texts)
                self._queued -= n
                self._inflight = n
                self.batches += 1
            t0 = time.perf_counter()
            try:
                X = self.inner.encode([t for texts, _ in batch for t in texts])
            except BaseException as e:
                for _, fut in batch:
                    fut.set_exception(e)
                continue
            finally:
                with self._cv:
                    self._inflight = 0
            per = (time.perf_counter() - t0) / n
            self._per_text_s = per if not self._per_text_s else 0.8 * self._per_text_s + 0.2 * per
            i = 0
            for texts, fut in batch:
                fut.set_result(X[i:i + len(texts)])
                i += len(texts)

    def time_per_text_s(self) -> float:
        """Measured encode time per text (moving average); 0 until the first batch."""
        return self._per_text_s

    def pending(self) -> int:
        """Texts queued or being encoded right now (0: idle)."""
        return self._queued + self._inflight

    def expected_wait_s(self) -> float:
        """Roughly how long a call made now would take: everything ahead of it, then itself."""
        return (self._queued + self._inflight + 1) * self._per_text_s

    def close(self) -> None:
        """Finish queued calls, then stop the worker."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join()
