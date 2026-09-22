import pytest

from jevstiller import SyntheticTeacher, SyntheticWorld, Task

LABELS = ["billing", "technical", "cancellation", "sales", "other"]


@pytest.fixture
def task():
    return Task(name="support", instructions="Which team handles this?", classes=LABELS, target_agreement=0.95)


@pytest.fixture
def world():
    return SyntheticWorld(LABELS, seed=1)


@pytest.fixture
def teacher(world):
    return SyntheticTeacher(world, temperature=0.7, noise=0.5)
