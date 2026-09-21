import numpy as np

from jevstiller.student import LinearStudent
from jevstiller.ood import KnnOOD


def test_linear_student_learns_soft_targets():
    rng = np.random.default_rng(0)
    n, d, K = 2000, 32, 4
    W = rng.normal(size=(d, K))
    X = rng.normal(size=(n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    Z = X @ W * 4
    Y = np.exp(Z - Z.max(1, keepdims=True)); Y /= Y.sum(1, keepdims=True)
    s = LinearStudent(d, K)
    r = s.fit(X[:1500], Y[:1500])
    assert r['n_val'] == 150 and 0 < r['epochs'] <= 2000
    P = s.predict_proba(X[1500:])
    assert (P.argmax(1) == Y[1500:].argmax(1)).mean() > 0.95


def test_knn_ood_scores_novel_higher():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 16)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    o = KnnOOD(k=5); o.fit(X)
    near = X[:50] + rng.normal(scale=0.05, size=(50, 16)); near /= np.linalg.norm(near, axis=1, keepdims=True)
    far = rng.normal(size=(50, 16)) + 3; far /= np.linalg.norm(far, axis=1, keepdims=True)
    assert o.score(near).mean() < o.score(far).mean()
