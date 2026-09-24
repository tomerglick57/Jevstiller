"""JevTeacher against a fake typesafe_sdk, so the adapter is tested without a key or network."""
import sys
import types

import pytest

from jevstiller import Task


class _Answer:
    def __init__(self, choice, probs, conf):
        self.choice, self.probabilities, self.confidence = choice, probs, conf


class _Result:
    def __init__(self, ans, tokens):
        self.choices = {"label": ans}
        self.request_id = "req_test"
        self.model = "jev-1.13.0"                 # what `jev-latest` resolved to

        class _Raw:
            def json(_self):
                return {"usage": {"input_tokens": tokens}}
        self.raw_http_response = _Raw()


class _Client:
    calls = []

    def __init__(self, api_key=None, model=None, retry=None, timeout=None):
        self.model, self.timeout = model, timeout

    def system_one(self, state, questions):
        _Client.calls.append((state, questions))
        if isinstance(state, dict):
            state = " ".join(f"{k} {v}" for k, v in state.items())
        if state == "boom":
            raise RuntimeError("529 overloaded")
        q = questions["label"]
        first = next(iter(q.criteria))
        probs = {c: (0.7 if c == first else 0.3 / (len(q.criteria) - 1)) for c in q.criteria}
        return _Result(_Answer(first, probs, 0.55), tokens=len(state.split()) + 30)


class _Choice:
    def __init__(self, instructions, criteria):
        self.instructions, self.criteria = instructions, criteria


@pytest.fixture
def fake_sdk(monkeypatch):
    mod = types.ModuleType("typesafe_sdk")
    mod.TypeSafeClient, mod.Choice, mod.RetryPolicy = _Client, _Choice, lambda **kw: kw
    monkeypatch.setitem(sys.modules, "typesafe_sdk", mod)
    _Client.calls.clear()
    return mod


def test_jev_adapter_maps_answers(fake_sdk):
    from jevstiller.teachers.jev import JevTeacher
    task = Task(name="t", instructions="Which team?", classes={"billing": "money", "tech": "bugs"})
    t = JevTeacher(rpm=100000, concurrency=2)
    outs = t.classify(["charged twice", "app crashes", "x"], task)
    assert [o.label for o in outs] == ["billing"] * 3
    assert all(abs(sum(o.probs.values()) - 1) < 1e-9 and set(o.probs) == {"billing", "tech"} for o in outs)
    assert outs[0].input_tokens == 32 and abs(outs[0].cost_usd - 32 * 0.042 / 1e6) < 1e-12
    assert outs[0].confidence == 0.55 and outs[0].request_id == "req_test"
    state, q = _Client.calls[0]
    assert q["label"].instructions == "Which team?" and q["label"].criteria == task.classes
    assert t.name == "jev:jev-1.13.0"


def test_jev_adapter_isolates_failed_items(fake_sdk):
    from jevstiller.teachers.jev import JevTeacher
    task = Task(name="t", instructions="Which team?", classes={"billing": "money", "tech": "bugs"})
    t = JevTeacher(rpm=100000, concurrency=2, timeout=3.0)
    assert t.client.timeout == 3.0
    outs = t.classify(["charged twice", "boom", "app crashes"], task)
    assert outs[0].label == "billing" and outs[2].label == "billing"
    assert isinstance(outs[1], RuntimeError)
    assert isinstance(t.classify(["boom"], task)[0], RuntimeError)


def test_jev_adapter_reports_resolved_model_and_passes_objects(fake_sdk):
    from jevstiller.teachers.jev import JevTeacher
    task = Task(name="t", instructions={"q": "Which team?"}, classes={"billing": {"covers": "money"}, "tech": None})
    t = JevTeacher(model="jev-latest", rpm=100000)
    state = {"subject": "charged twice", "body": "please refund"}
    out = t.classify([state], task)[0]
    assert out.model == "jev:jev-1.13.0" and t.name == "jev:jev-latest"
    sent_state, q = _Client.calls[-1]
    assert sent_state is state and q["label"].criteria == {"billing": {"covers": "money"}, "tech": None}
