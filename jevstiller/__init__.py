from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .core import Jevstiller, Result, Status, TeacherError, TrainReport
from .encoders import BatchingEncoder, Encoder, HashEncoder, load_encoder
from .manager import Admission, TaskManager
from .scheduler import TrainScheduler
from .task import Config, Task
from .teachers import CachedTeacher, ReplayTeacher, SyntheticTeacher, SyntheticWorld, Teacher, TeacherOutput

__all__ = ["Task", "Config", "Jevstiller", "Result", "Status", "TrainReport", "TeacherError", "Teacher",
           "TeacherOutput", "SyntheticTeacher", "SyntheticWorld", "ReplayTeacher", "CachedTeacher", "Encoder",
           "HashEncoder", "BatchingEncoder", "load_encoder", "TaskManager", "Admission",
           "TrainScheduler"]

try:
    __version__ = _version("jevstiller")
except PackageNotFoundError:  # pragma: no cover - running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
