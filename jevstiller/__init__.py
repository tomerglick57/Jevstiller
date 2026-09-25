"""Jevstiller: a repeated Jev classification task, distilled into a local model on the fly.

The public API is what this package exports (`__all__` below), plus `jevstiller.server` (the proxy as a library),
`jevstiller.encoders`, `jevstiller.teachers` and `jevstiller.teachers.jev`. Modules and names that start with an
underscore are internal and may change in any release (docs/compatibility.md).
"""
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from ._core import Jevstiller, Result, Routed, Status, TeacherError, TrainReport
from ._manager import Admission, Routing, TaskInfo, TaskManager
from ._scheduler import TrainScheduler
from ._task import Config, State, Task
from ._training import train_pool
from .encoders import BatchingEncoder, Encoder, HashEncoder, load_encoder
from .teachers import CachedTeacher, ReplayTeacher, SyntheticTeacher, SyntheticWorld, Teacher, TeacherOutput

__all__ = [
    # one task
    "Task", "Config", "State", "Jevstiller", "Result", "Routed", "Status", "TrainReport", "TeacherError",
    # many tasks, shared training
    "TaskManager", "Admission", "Routing", "TaskInfo", "TrainScheduler", "train_pool",
    # teachers and encoders
    "Teacher", "TeacherOutput", "SyntheticTeacher", "SyntheticWorld", "ReplayTeacher", "CachedTeacher",
    "Encoder", "HashEncoder", "BatchingEncoder", "load_encoder",
]

try:
    __version__ = _version("jevstiller")
except _PackageNotFoundError:  # pragma: no cover - running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
