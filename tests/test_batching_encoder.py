"""P2.2: one encoder shared by many threads; concurrent calls merge into batches."""
import threading
import time

import numpy as np
import pytest

from jevstiller.encoders import BatchingEncoder, HashEncoder


class SlowEncoder:
    """A HashEncoder that takes a fixed time per call, like a GPU kernel launch."""

    def __init__(self, delay_s=0.02, fail_on=None):
        self.inner, self.delay_s, self.fail_on = HashEncoder(dim=64), delay_s, fail_on
        self.id, self.dim, self.sizes = self.inner.id, self.inner.dim, []

    def encode(self, texts):
        self.sizes.append(len(texts))
        time.sleep(self.delay_s)
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise ValueError("bad input")
        return self.inner.encode(texts)


def test_concurrent_calls_are_batched_and_rows_match():
    inner = SlowEncoder()
    enc = BatchingEncoder(inner, max_batch=64)
    out, errors = {}, []

    def call(k):
        try:
            texts = [f"text {k} {j}" for j in range(1 + k % 3)]
            out[k] = (texts, enc.encode(texts))
        except Exception as e:  # pragma: no cover
            errors.append(e)
    ts = [threading.Thread(target=call, args=(k,)) for k in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(out) == 40
    ref = HashEncoder(dim=64)
    for texts, X in out.values():
        assert np.allclose(X, ref.encode(texts))
    assert enc.batches < 40 and max(inner.sizes) > 3            # calls were merged
    assert all(s <= 64 for s in inner.sizes)
    enc.close()


def test_idle_call_is_not_delayed():
    enc = BatchingEncoder(SlowEncoder(delay_s=0.0))
    t0 = time.perf_counter()
    enc.encode(["one"])
    assert time.perf_counter() - t0 < 0.5
    assert enc.id == HashEncoder(dim=64).id and enc.dim == 64
    enc.close()


def test_errors_reach_every_caller_in_the_batch_and_the_encoder_recovers():
    enc = BatchingEncoder(SlowEncoder(fail_on="bad"))
    with pytest.raises(ValueError):
        enc.encode(["bad text"])
    assert enc.encode(["good text"]).shape == (1, 64)
    enc.close()
    with pytest.raises(RuntimeError):
        enc.encode(["after close"])


def test_large_calls_bypass_the_queue():
    inner = SlowEncoder(delay_s=0.0)
    enc = BatchingEncoder(inner, max_batch=8)
    assert enc.encode([f"t{i}" for i in range(20)]).shape == (20, 64)
    assert enc.batches == 0 and inner.sizes == [20]
    enc.close()
