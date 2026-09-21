"""Student heads on frozen embeddings. numpy only."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LinearStudent:
    """Multinomial logistic regression trained with cross-entropy against soft targets."""

    kind = "linear"

    def __init__(self, dim: int, n_classes: int):
        self.W = np.zeros((dim, n_classes), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 2000, lr: float = 0.05,
            l2: float = 1e-6, seed: int = 0, sample_weight: np.ndarray | None = None,
            val_fraction: float = 0.1, patience: int = 4, eval_every: int = 25) -> dict:
        """Full-batch Adam on weighted soft-target cross-entropy, with early stopping.

        X (n, d) embeddings; Y (n, K) target distributions (rows sum to 1); sample_weight (n,) importance
        weights. A `val_fraction` slice (seeded) is held out; training stops when its weighted loss has not
        improved for `patience` evaluations and the best parameters are restored.
        """
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        n = X.shape[0]
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_val = int(n * val_fraction) if n >= 50 else 0
        vi, ti = idx[:n_val], idx[n_val:]
        Xt, Yt, wt = X[ti], Y[ti], w[ti]
        wt = (wt / wt.sum() * len(ti))[:, None]          # mean weight 1, so lr/l2 keep their meaning
        if n_val:
            Xv, Yv, wv = X[vi], Y[vi], (w[vi] / w[vi].sum())[:, None]
        self.W = rng.normal(0, 0.01, self.W.shape)
        self.b[:] = 0
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best = (np.inf, self.W.copy(), self.b.copy(), 0)
        bad = 0
        loss = float("nan")
        for t in range(1, epochs + 1):
            P = _softmax(Xt @ self.W + self.b)
            G = wt * (P - Yt) / len(ti)
            gW = Xt.T @ G + l2 * self.W
            gb = G.sum(axis=0)
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW * gW
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            self.W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
            self.b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
            if n_val and t % eval_every == 0:
                Pv = _softmax(Xv @ self.W + self.b)
                vloss = float(-(wv * Yv * np.log(Pv + 1e-12)).sum())
                if vloss < best[0] - 1e-5:
                    best, bad = (vloss, self.W.copy(), self.b.copy(), t), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if n_val:
            _, self.W, self.b, stopped = best
        else:
            stopped = epochs
        P = _softmax(Xt @ self.W + self.b)
        loss = float(-(wt * Yt * np.log(P + 1e-12)).sum() / len(ti))
        return {"epochs": stopped, "loss": loss, "n": n, "n_val": n_val, "val_loss": best[0] if n_val else None}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(X.astype(np.float64) @ self.W + self.b)

    def save(self, path: Path) -> None:
        np.savez(path, W=self.W, b=self.b)

    @classmethod
    def load(cls, path: Path) -> "LinearStudent":
        d = np.load(path)
        s = cls(d["W"].shape[0], d["W"].shape[1])
        s.W, s.b = d["W"], d["b"]
        return s
