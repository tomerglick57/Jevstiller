
from jevstiller import SyntheticTeacher, SyntheticWorld, Task
from jevstiller.teachers import CachedTeacher, ReplayTeacher


def test_cached_teacher_persists_and_replays(tmp_path):
    labels = ["a", "b", "c"]
    world = SyntheticWorld(labels, seed=3)
    live = SyntheticTeacher(world)
    task = Task(name="t", instructions="?", classes=labels)
    texts = [t for t, _ in world.sample(20)]
    c1 = CachedTeacher(live, tmp_path / "cache.jsonl")
    a = c1.classify(texts, task)
    assert c1.misses == 20 and c1.hits == 0
    c2 = CachedTeacher(live, tmp_path / "cache.jsonl")       # fresh process: reads the file
    b = c2.classify(texts, task)
    assert c2.hits == 20 and c2.misses == 0
    assert [o.label for o in a] == [o.label for o in b] and a[0].probs == b[0].probs
    # a different task version must not hit the cache
    task2 = Task(name="t", instructions="something else", classes=labels)
    c2.classify(texts[:5], task2)
    assert c2.misses == 5
    assert len(open(tmp_path / "cache.jsonl").readlines()) == 25


def test_replay_teacher_fallback():
    labels = ["a", "b"]
    world = SyntheticWorld(labels, seed=0)
    live = SyntheticTeacher(world)
    task = Task(name="t", instructions="?", classes=labels)
    r = ReplayTeacher({}, fallback=live)
    out = r.classify(["w1 w2"], task)
    assert out[0].label in labels and r.misses == 1
    r.classify(["w1 w2"], task)
    assert r.misses == 1
