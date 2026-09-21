"""Out-of-distribution score: 1 - mean cosine similarity to the k nearest training embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class KnnOOD:
    def __init__(self, k: int = 10):
        self.k = k
        self.X: np.ndarray | None = None

    def fit(self, X: np.ndarray, max_ref: int = 50_000, seed: int = 0) -> None:
        X = X.astype(np.float32)
        if X.shape[0] > max_ref:
            idx = np.random.default_rng(seed).choice(X.shape[0], max_ref, replace=False)
            X = X[idx]
        self.X = X

    def score(self, Q: np.ndarray, chunk: int = 2048) -> np.ndarray:
        assert self.X is not None, "fit first"
        k = min(self.k, self.X.shape[0])
        out = np.empty(Q.shape[0], dtype=np.float32)
        Q = Q.astype(np.float32)
        for s in range(0, Q.shape[0], chunk):
            sims = Q[s:s + chunk] @ self.X.T
            top = -np.partition(-sims, k - 1, axis=1)[:, :k]
            out[s:s + chunk] = 1.0 - top.mean(axis=1)
        return out

    def save(self, path: Path) -> None:
        np.savez(path, X=self.X, k=self.k)

    @classmethod
    def load(cls, path: Path) -> "KnnOOD":
        d = np.load(path)
        o = cls(int(d["k"]))
        o.X = d["X"]
        return o
