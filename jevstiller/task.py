"""Task definition and runtime configuration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Task:
    """What is being classified. Everything here defines the *teacher's* behaviour.

    `classes` maps class name -> description. Descriptions are sent to the teacher
    verbatim (Jev's `criteria`), so they are part of the task version.
    """

    name: str
    instructions: str
    classes: Mapping[str, str] | Sequence[str]
    target_agreement: float = 0.98

    def __post_init__(self) -> None:
        if not isinstance(self.classes, Mapping):
            object.__setattr__(self, "classes", {c: "" for c in self.classes})
        else:
            object.__setattr__(self, "classes", dict(self.classes))
        if len(self.classes) < 2:
            raise ValueError("a task needs at least two classes")
        if not 0.5 <= self.target_agreement < 1.0:
            raise ValueError("target_agreement must be in [0.5, 1)")

    @property
    def labels(self) -> list[str]:
        return list(self.classes.keys())

    @property
    def budget(self) -> float:
        """Disagreement budget: fraction of all requests allowed to differ from the teacher."""
        return 1.0 - self.target_agreement

    @property
    def version(self) -> str:
        blob = json.dumps({"i": self.instructions, "c": self.classes}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


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
    drift_window: int = 500             # audit records in the rolling agreement check
    drift_min_samples: int = 200
    drift_margin: float = 0.0           # ub(agreement) < target - margin -> hard fallback
    mode: str = "auto"                  # auto | teacher_only | cascade
    seed: int = 0
    student_epochs: int = 2000          # upper bound; early stopping on a validation slice decides
    label_target: str = "probs"        # train on the teacher's distribution, or 'hard' (argmax)
    importance_weighting: bool = True   # audit rows weigh 1/audit_rate in training, so the training set tracks traffic
    weight_power: float = 0.5           # tempering: weight = (1/audit_rate) ** power; 1.0 = exact Horvitz-Thompson
    max_weight: float = 20.0
    fit_headroom: float = 0.15          # fit policies at (1 - headroom) * budget; shadow judges at the full budget
    student_l2: float = 1e-6
    student_patience: int = 4

    def to_dict(self) -> dict:
        return asdict(self)
