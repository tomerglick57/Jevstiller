"""P1.6-P1.9: JSON tasks and states, label order, teacher lineage, rare classes."""
import json

import numpy as np
import pytest

from jevstiller import Config, Jevstiller, ReplayTeacher, SyntheticTeacher, SyntheticWorld, Task
from jevstiller._task import state_text

LABELS = ["billing", "technical", "cancellation", "sales", "other"]


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0, training="inline")
    base.update(kw)
    return Config(**base)


def _until_production(js, world, batches=40, priors=None):
    for _ in range(batches):
        js.classify_batch([t for t, _ in world.sample(200, priors=priors)])
        st = js.status()
        if st.production and st.shadow is None and st.mode == "cascade":
            return st
    raise AssertionError(js.status().report())


class Versioned:
    """A teacher whose answers name the model that produced them, like Jev's resolved `model`."""

    name = "versioned"

    def __init__(self, inner, model):
        self.inner, self.model, self.seen = inner, model, []

    def classify(self, texts, task):
        self.seen.extend(texts)
        outs = self.inner.classify(texts, task)
        for o in outs:
            o.model = self.model
        return outs


# ---- P1.6: tasks -------------------------------------------------------------------------------------------
def test_text_task_versions_are_unchanged_from_0_1_0():
    assert Task("a", "Which team?", ["x", "y"]).version == "768d0081e0e0"


def test_json_instructions_and_criteria():
    t = Task("t", {"question": "Which team?", "examples": ["refund please"]},
             {"billing": {"covers": ["refunds", "invoices"]}, "tech": ["bugs", "errors"], "other": None})
    assert t.labels == ["billing", "tech", "other"]
    same = Task("t", {"examples": ["refund please"], "question": "Which team?"},
                {"other": None, "tech": ["bugs", "errors"], "billing": {"covers": ["refunds", "invoices"]}})
    assert same.version == t.version                     # key and class order do not matter
    other = Task("t", {"question": "Which team?", "examples": ["refund please"]},
                 {"billing": {"covers": ["refunds"]}, "tech": ["bugs", "errors"], "other": None})
    assert other.version != t.version                    # any change to what the teacher reads does


@pytest.mark.parametrize("classes", [["only"], [f"c{i}" for i in range(256)], "ab", ["a", ""], {"a": 1, 2: "b"},
                                     {"a": {1, 2}, "b": ""}])
def test_invalid_tasks(classes):
    with pytest.raises(ValueError):
        Task("t", "x", classes)


def test_255_classes_allowed():
    assert len(Task("t", "x", [f"c{i}" for i in range(255)]).labels) == 255


def test_student_is_reordered_to_the_task_label_order(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg())
    _until_production(js, world)
    probe = [t for t, _ in world.sample(300)]
    before = js.evaluate(probe, [teacher.answer(t, task).label for t in probe])
    js.close()
    flipped = Task(task.name, task.instructions, list(reversed(task.labels)), task.target_agreement)
    assert flipped.version == task.version
    js2 = Jevstiller(flipped, teacher, tmp_path, config=_cfg(training="manual"))
    after = js2.evaluate(probe, [teacher.answer(t, task).label for t in probe])
    assert list(after["student_label"]) == list(before["student_label"])
    assert (after["accepted"] == before["accepted"]).all()
    js2.close()


# ---- P1.7: states ------------------------------------------------------------------------------------------
def test_object_states(tmp_path, task, world, teacher):
    spy = Versioned(teacher, "m1")
    js = Jevstiller(task, spy, tmp_path, config=_cfg(training="manual"))
    a = {"subject": "w1 w2", "body": "w3", "tags": ["x", "y"]}
    b = {"tags": ["x", "y"], "body": "w3", "subject": "w1 w2"}
    ra, rb = js.classify(a), js.classify(b)
    assert ra.label == rb.label
    assert spy.seen[0] is a                               # the teacher gets the object itself
    rows = js.store.db.execute("SELECT text, text_hash, state_type FROM samples ORDER BY id").fetchall()
    assert rows[0][0] == state_text(a) == state_text(b) and json.loads(rows[0][0]) == a
    assert rows[0][1] == rows[1][1] and rows[0][2] == "json"
    js.classify("plain w1")
    assert js.store.db.execute("SELECT state_type FROM samples ORDER BY id DESC").fetchone()[0] == "text"
    with pytest.raises(TypeError):
        js.classify(42)
    js.close()


def test_replay_teacher_keys_object_states(task, teacher):
    state = {"b": "w1", "a": "w2"}
    replay = ReplayTeacher({state_text(state): teacher.answer(state_text(state), task)})
    assert replay.classify([{"a": "w2", "b": "w1"}], task)[0].label == teacher.answer(state_text(state), task).label


def test_old_stores_are_migrated(tmp_path, task, teacher):
    import sqlite3
    d = tmp_path / task.name
    d.mkdir()
    db = sqlite3.connect(d / "samples.sqlite")
    from jevstiller._store import _SCHEMA
    db.executescript(_SCHEMA)                            # the 0.1.0 schema, without state_type
    db.close()
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(training="manual"))
    js.classify({"k": "w1"})
    assert js.store.db.execute("SELECT state_type FROM samples").fetchone()[0] == "json"
    js.close()


# ---- P1.8: teacher lineage ---------------------------------------------------------------------------------
def test_teacher_model_change_starts_a_new_lineage(tmp_path, task, world, teacher):
    v1 = Versioned(teacher, "jev:1.13.0")
    js = Jevstiller(task, v1, tmp_path, config=_cfg(teacher_change_confirm=5))
    _until_production(js, world)
    prod_v1 = js.status().production
    assert js.status().teacher_model == "jev:1.13.0"
    assert js.registry.load(prod_v1).meta["teacher_model"] == "jev:1.13.0"

    js.teacher = Versioned(SyntheticTeacher(SyntheticWorld(task.labels, seed=7)), "jev:1.14.0")
    js.cfg.training = "manual"                          # hold the new lineage untrained for a moment
    js.set_mode("teacher_only")                         # every request reaches the teacher
    js.classify_batch([t for t, _ in world.sample(4)])  # 4 answers < confirm: no switch yet
    assert js.status().teacher_model == "jev:1.13.0"
    js.classify_batch([t for t, _ in world.sample(10)])
    js.cfg.mode = "auto"                                # hand routing back (set_mode would clear a fallback)
    st = js.status()
    assert st.teacher_model == "jev:1.14.0" and st.mode == "teacher_only", st.report()
    assert any(e["kind"] == "teacher_changed" and e["current"] == "jev:1.14.0" for e in st.events)
    js.close()

    js = Jevstiller(task, js.teacher, tmp_path, config=_cfg())   # restart: still falls back
    assert js.mode == "teacher_only"
    st = _until_production(js, world)
    assert st.production != prod_v1 and st.teacher_model == "jev:1.14.0"
    assert js.registry.load(st.production).meta["teacher_model"] == "jev:1.14.0"
    lineage = js.registry.load(st.production).meta["n_train"]
    n_new = js.store.db.execute("SELECT COUNT(*) FROM samples WHERE teacher_model='jev:1.14.0' AND split='train' "
                                "AND teacher_label IS NOT NULL").fetchone()[0]
    assert lineage <= n_new                              # trained on the new model's answers only
    js.close()


def test_alternating_models_do_not_flap(tmp_path, task, world, teacher):
    js = Jevstiller(task, Versioned(teacher, "a"), tmp_path, config=_cfg(training="manual", teacher_change_confirm=5))
    for i in range(20):
        js.teacher = Versioned(teacher, "b" if i % 2 else "a")    # a, b, a, b ...: 3 in a row at most
        js.classify_batch([t for t, _ in world.sample(3)])
    assert js.status().teacher_model == "a"
    assert not any(e["kind"] == "teacher_changed" for e in js.status().events)
    js.close()


def test_teacher_change_in_audit_mode_keeps_serving(tmp_path, task, world, teacher):
    js = Jevstiller(task, Versioned(teacher, "m1"), tmp_path,
                    config=_cfg(teacher_change="audit", teacher_change_confirm=3))
    _until_production(js, world)
    js.cfg.training = "manual"
    js.teacher = Versioned(teacher, "m2")
    base_rate = js.audit_rate
    for _ in range(10):
        js.classify_batch([t for t, _ in world.sample(100)])
        if js.status().teacher_model == "m2":
            break
    assert js.status().teacher_model == "m2" and js.mode == "cascade"
    assert js.audit_rate > base_rate
    js.close()


# ---- P1.9: rare classes ------------------------------------------------------------------------------------
PRIORS = [1, 1, 1, 1, 0.004]                             # "other" is rare


def test_readiness_names_the_blocking_class(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg())
    for _ in range(15):
        js.classify_batch([t for t, _ in world.sample(200, priors=PRIORS)])
    st = js.status()
    assert st.production is None and st.readiness and st.readiness["blocked_by_rare"]
    assert "other" in st.readiness["rare"] and st.readiness["rare"]["other"] < 20
    assert "rare_classes='defer'" in st.report()
    js.close()


def test_defer_rare_classes_trains_and_never_serves_them(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(rare_classes="defer"))
    st = _until_production(js, world, priors=PRIORS)
    policy = js.registry.load(st.production).meta["deferred_labels"]
    assert policy == ["other"] and st.policy["deferred_labels"] == ["other"]
    tail = [t for t, _ in world.sample(3000, priors=[1, 1, 1, 1, 1])]
    res = js.classify_batch(tail)
    served = [(t, r) for t, r in zip(tail, res, strict=True) if r.source != "teacher"]
    assert served and all(r.label != "other" for _, r in served)
    assert any(r.routing_reason == "rare_class" for r in res)
    disagree = sum(r.label != teacher.answer(t, task).label for t, r in served)
    assert disagree / len(tail) <= task.budget + 0.01, js.status().report()
    assert "always sent to the teacher: other" in js.status().report()
    js.close()


def test_policy_without_deferred_labels_loads_from_old_files(tmp_path):
    from jevstiller._calibrate import RoutingPolicy
    old = {"conf_threshold": 0.5, "ood_threshold": 0.9, "expected_coverage": 0.8, "disagreement_ub": 0.01,
           "expected_system_disagreement": 0.008, "budget": 0.02, "delta": 0.05, "n_calib": 100}
    (tmp_path / "policy.json").write_text(json.dumps(old))
    p = RoutingPolicy.load(tmp_path / "policy.json")
    assert p.deferred_labels == [] and p.accepts(np.array([0.6]), np.array([0.1])).all()
