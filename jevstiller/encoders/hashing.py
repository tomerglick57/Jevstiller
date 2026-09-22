"""Hashed bag of words + bigrams. No model download; deterministic; a real (weak) baseline."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")


class HashEncoder:
    def __init__(self, dim: int = 512, seed: int = 0, bigrams: bool = True):
        self.dim = dim
        self.seed = seed
        self.bigrams = bigrams
        self.id = f"hash-{dim}-{seed}-{'bi' if bigrams else 'uni'}"
        self._key = seed.to_bytes(8, "little")
        self._cache: dict[str, tuple[int, float]] = {}

    def _slot(self, feat: str) -> tuple[int, float]:
        s = self._cache.get(feat)
        if s is None:
            h = hashlib.blake2b(feat.encode(), digest_size=8, key=self._key).digest()
            s = (int.from_bytes(h[:4], "little") % self.dim, 1.0 if h[4] & 1 else -1.0)
            if len(self._cache) < 200_000:
                self._cache[feat] = s
        return s

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = _TOKEN.findall(t.lower())
            feats = list(toks)
            if self.bigrams:
                feats += [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]
            for f in feats:
                j, sgn = self._slot(f)
                X[i, j] += sgn
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        np.divide(X, norms, out=X, where=norms > 0)
        return X
