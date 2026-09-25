"""Task definition and runtime configuration."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

State = str | dict | list
"""What is classified: text, or a JSON object/array (Jev's `state`). Non-text states are encoded as
canonical JSON (see `state_text`); the teacher always receives the original value."""

MAX_CLASSES = 255                       # Jev's limit for a Choice question


def canonical_json(value: Any) -> str:
    """Stable text for a JSON value: sorted keys, no insignificant whitespace, UTF-8 kept as is."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


MAX_TEXT_CHARS = 32_768  # of a state's text: what the encoder reads and what the store keeps (far beyond 256 tokens)


def state_text(state: State) -> str:
    """The text the encoder sees and the store keeps. Strings are unchanged."""
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return canonical_json(state)
    raise TypeError(f"state must be str, dict or list, got {type(state).__name__}")


@dataclass(frozen=True)
class Task:
    """What is being classified. Everything here defines the *teacher's* behaviour.

    `classes` maps class name -> description. Descriptions are sent to the teacher verbatim (Jev's
    `criteria`), so they are part of the task version. Like `instructions`, a description may be text, a
    JSON object or array, or None ("interpreted by its name alone"). A plain list of names means empty
    descriptions.
    """

    name: str
    instructions: Any
    classes: Mapping[str, Any] | Sequence[str]
    target_agreement: float = 0.98

    def __post_init__(self) -> None:
        if not isinstance(self.classes, Mapping):
            if isinstance(self.classes, str):
                raise ValueError("classes must be a mapping or a list of names, not a string")
            object.__setattr__(self, "classes", {c: "" for c in self.classes})
        else:
            object.__setattr__(self, "classes", dict(self.classes))
        if not all(isinstance(c, str) and c for c in self.classes):
            raise ValueError("class names must be non-empty strings")
        if not 2 <= len(self.classes) <= MAX_CLASSES:
            raise ValueError(f"a task needs between 2 and {MAX_CLASSES} classes, got {len(self.classes)}")
        if not 0.5 <= self.target_agreement < 1.0:
            raise ValueError("target_agreement must be in [0.5, 1)")
        try:
            json.dumps({"i": self.instructions, "c": self.classes})
        except (TypeError, ValueError) as e:
            raise ValueError(f"instructions and class descriptions must be JSON values: {e}") from None

    @property
    def labels(self) -> list[str]:
        return list(self.classes.keys())

    @property
    def budget(self) -> float:
        """Disagreement budget: fraction of all requests allowed to differ from the teacher."""
        return 1.0 - self.target_agreement

    @property
    def version(self) -> str:
        """Short hash of what the teacher is asked, used to tag stored rows. Independent of class order (a
        loaded student is reordered to the task's labels), and unchanged from 0.1.0 for text tasks."""
        return self.fingerprint[:12]

    @property
    def fingerprint(self) -> str:
        """Full SHA-256 of what the teacher is asked (the identity the task manager keys tasks on: 48 bits
        of `version` are within reach of a second-preimage search)."""
        blob = json.dumps({"i": self.instructions, "c": self.classes}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


MODES = ("auto", "teacher_only", "cascade")
TRAINING = ("background", "inline", "manual")
TEACHER_CHANGE = ("fallback", "audit")
RARE_CLASSES = ("wait", "defer")


@dataclass
class Config:
    """How the loop runs. Every field is a default that can be overridden."""

    confidence: float = 0.95            # 1 - delta for every bound
    audit_rate: float = 0.02            # share of all traffic always sent to the teacher
    audit_rate_shadow: float = 0.10     # audit rate while a candidate is in shadow (judge it on fresh traffic quickly)
    audit_rate_elevated: float = 0.10   # audit rate while the drift monitor is suspicious
    calib_fraction: float = 0.20        # share of IID samples hashed into the calibration split
    min_train_samples: int = 1000
    min_samples_per_class: int = 50
    min_calib_samples: int = 500
    min_new_samples: int = 2000         # retrain trigger
    shadow_min_samples: int = 1000      # requests a shadow candidate must run alongside before judgement
    ood_quantile: float = 0.99
    ood_k: int = 10
    ood_max_ref: int = 5_000            # reference embeddings kept per version (memory, per-request cost)
    max_train_samples: int = 50_000     # a fit uses at most this many of the most recent training rows (0 = all):
                                        # without a cap, each retrain of a busy task reads its whole history
    max_calib_samples: int = 20_000     # ... and calibration rows
    drift_window: int = 500             # audit records in the rolling agreement check
    drift_min_samples: int = 200
    drift_margin: float = 0.0           # ub(agreement) < target - margin -> hard fallback
    drift_check_every: int = 20         # re-check drift after this many new audit answers
    mode: str = "auto"                  # auto | teacher_only | cascade
    training: str = "background"        # background: a worker thread trains off the request path
                                        # inline: train inside classify_batch (deterministic replays, tests)
                                        # manual: never automatically; call maintain() / train_now()
    maintenance_interval_s: float = 1.0  # background: at most one maintenance pass per interval
    train_threads: int = 2              # BLAS threads per fit (0 = library default, i.e. every core)
    teacher_change: str = "fallback"    # the teacher's resolved model changed (e.g. jev-latest moved):
                                        # fallback: all traffic to the teacher until a student of the new
                                        #   model passes shadow; audit: keep serving, raise the audit rate,
                                        #   retrain, and let the drift monitor decide
    teacher_change_confirm: int = 20    # consecutive answers from a new model before switching lineage
    rare_classes: str = "wait"          # a class below min_samples_per_class:
                                        # wait: no first student until every class has enough samples
                                        # defer: train without waiting; a prediction of a rare class goes
                                        #   to the teacher until the class has enough samples
    store_text: bool = True             # False: keep only the hash + embedding (no raw text in the store)
    seed: int = 0
    student_epochs: int = 2000          # upper bound; early stopping on a validation slice decides
    label_target: str = "probs"        # train on the teacher's distribution, or 'hard' (argmax)
    importance_weighting: bool = True   # audit rows weigh 1/audit_rate in training, so the training set tracks traffic
    weight_power: float = 0.5           # tempering: weight = (1/audit_rate) ** power; 1.0 = exact Horvitz-Thompson
    max_weight: float = 20.0
    fit_headroom: float = 0.15          # fit policies at (1 - headroom) * budget; shadow judges at the full budget
    student_l2: float = 1e-6
    student_patience: int = 4

    def __post_init__(self) -> None:
        for name, allowed in (("mode", MODES), ("training", TRAINING), ("label_target", ("probs", "hard")),
                              ("teacher_change", TEACHER_CHANGE), ("rare_classes", RARE_CLASSES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        for name, floor in (("max_train_samples", "min_train_samples"), ("max_calib_samples", "min_calib_samples")):
            cap = getattr(self, name)
            if cap < 0 or 0 < cap < getattr(self, floor):
                raise ValueError(f"{name} must be 0 (no cap) or at least {floor} ({getattr(self, floor)}), got {cap}")

    def to_dict(self) -> dict:
        return asdict(self)
