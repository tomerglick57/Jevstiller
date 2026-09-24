"""A deterministic fake world + teacher, for tests and demos. Never touches the network."""
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence

from ..task import State, Task, state_text
from . import TeacherOutput, peakedness


class SyntheticWorld:
    """Generates texts from per-class vocabularies with a shared, ambiguous vocabulary.

    Each class owns `vocab_size` words; `shared_size` words belong to no class.
    A text of class c draws `signal` of its words from c's vocabulary and the rest
    from the shared pool, so classes are learnable but not trivially separable.
    """

    def __init__(self, labels: Sequence[str], seed: int = 0, vocab_size: int = 30,
                 shared_size: int = 60, signal: float = 0.6, length: tuple[int, int] = (5, 12)):
        self.labels = list(labels)
        self.signal = signal
        self.length = length
        rng = random.Random(seed)
        words = [f"w{i}" for i in range(vocab_size * len(labels) + shared_size)]
        rng.shuffle(words)
        self.class_vocab = {c: words[i * vocab_size:(i + 1) * vocab_size] for i, c in enumerate(labels)}
        self.shared = words[vocab_size * len(labels):]
        self.word_class = {w: c for c, ws in self.class_vocab.items() for w in ws}
        self._rng = random.Random(seed + 1)

    def sample(self, n: int, priors: Sequence[float] | None = None, drift: float = 0.0) -> list[tuple[str, str]]:
        """Return (text, true_class) pairs. `drift` > 0 mixes in words from a foreign vocabulary."""
        out = []
        for _ in range(n):
            c = self._rng.choices(self.labels, weights=priors)[0]
            k = self._rng.randint(*self.length)
            words = []
            for _ in range(k):
                r = self._rng.random()
                if r < self.signal:
                    words.append(self._rng.choice(self.class_vocab[c]))
                elif r < self.signal + drift:
                    words.append(f"x{self._rng.randint(0, 200)}")   # out-of-vocabulary noise
                else:
                    words.append(self._rng.choice(self.shared))
            out.append((" ".join(words), c))
        return out


class SyntheticTeacher:
    """Scores each class by vocabulary overlap, adds text-keyed deterministic noise, softmaxes.

    Deterministic per text (same text -> same answer), like Jev.
    """

    name = "synthetic"

    def __init__(self, world: SyntheticWorld, temperature: float = 0.7, noise: float = 0.5,
                 price_per_mtok: float = 0.042):
        self.world = world
        self.temperature = temperature
        self.noise = noise
        self.price_per_mtok = price_per_mtok

    def _scores(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        counts = {c: 0.0 for c in labels}
        for w in text.split():
            c = self.world.word_class.get(w)
            if c in counts:
                counts[c] += 1.0
        h = hashlib.blake2b(text.encode(), digest_size=16).digest()
        for i, c in enumerate(labels):
            u = int.from_bytes(h[i % 16:i % 16 + 2], "little") / 65535.0
            counts[c] += (u - 0.5) * 2 * self.noise
        return counts

    def answer(self, text: str, task: Task) -> TeacherOutput:
        labels = task.labels
        s = self._scores(text, labels)
        m = max(s.values())
        exps = {c: math.exp((v - m) / self.temperature) for c, v in s.items()}
        z = sum(exps.values())
        probs = {c: e / z for c, e in exps.items()}
        label = max(probs, key=probs.get)
        toks = len(text.split()) + 40
        return TeacherOutput(label=label, probs=probs, confidence=peakedness(probs),
                             input_tokens=toks, cost_usd=toks * self.price_per_mtok / 1e6,
                             latency_ms=0.0)

    def classify(self, texts: Sequence[State], task: Task) -> list[TeacherOutput]:
        return [self.answer(state_text(t), task) for t in texts]
