"""Out-of-distribution score: 1 - mean cosine similarity to the k nearest training embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class KnnOOD:
    def __init__(self, k: int = 10):
        self.k = k
        self.X: np.ndarray | None = None

    def fit(self, X: np.ndarray, max_ref: int = 5_000, seed: int = 0, y: np.ndarray | None = None) -> None:
        """Keep at most `max_ref` reference embeddings. With labels `y`, the sample is stratified: every class
        keeps up to an equal share (all of its rows if it has fewer), and the rest of the budget is filled at
        random, so small classes stay represented. Memory and per-request cost are ~ max_ref x dim."""
        X = X.astype(np.float32)
        n = X.shape[0]
        if n > max_ref:
            rng = np.random.default_rng(seed)
            if y is None:
                idx = rng.choice(n, max_ref, replace=False)
            else:
                y = np.asarray(y)
                classes = np.unique(y)
                share = max_ref // len(classes)
                keep = [rng.permutation(np.flatnonzero(y == c))[:share] for c in classes]
                idx = np.concatenate(keep)
                rest = np.setdiff1d(np.arange(n), idx, assume_unique=False)
                idx = np.concatenate([idx, rng.choice(rest, max_ref - len(idx), replace=False)])
            X = X[np.sort(idx)]
        self.X = np.ascontiguousarray(X)

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
    def load(cls, path: Path) -> KnnOOD:
        d = np.load(path)
        o = cls(int(d["k"]))
        o.X = d["X"]
        return o
