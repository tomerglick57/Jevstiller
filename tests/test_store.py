import numpy as np

from jevstiller.store import SampleStore, Record, split_for, text_hash


def _rec(text, channel, **kw):
    return Record(text=text, task_version="tv", encoder_id="e", embedding=np.ones(4, np.float32),
                  served_by="teacher", routing_reason=channel, channel=channel,
                  teacher_label="a", teacher_probs={"a": 0.7, "b": 0.3}, **kw)


def test_split_is_deterministic_and_proportional():
    hs = [text_hash(f"text {i}") for i in range(20000)]
    frac = np.mean([split_for(h, 0.2) == "calib" for h in hs])
    assert 0.18 < frac < 0.22
    assert split_for(text_hash("x"), 0.2) == split_for(text_hash("x"), 0.2)


def test_deferred_never_enters_calib(tmp_path):
    s = SampleStore(tmp_path / "s.sqlite", calib_fraction=0.5)
    s.insert([_rec(f"t{i}", "deferred") for i in range(200)] + [_rec(f"u{i}", "audit") for i in range(200)])
    c = s.counts("tv")
    assert c["labelled_calib"] > 50
    rows = s.db.execute("SELECT COUNT(*) FROM samples WHERE channel='deferred' AND split='calib'").fetchone()[0]
    assert rows == 0
    X, Y, y, w = s.calib_set("tv", "e", ["a", "b"], 4)
    assert X.shape == (c["labelled_calib"], 4) and np.allclose(Y.sum(1), 1) and (y == 0).all() and (w == 1).all()


def test_weights_round_trip(tmp_path):
    s = SampleStore(tmp_path / "s.sqlite", calib_fraction=0.0)
    s.insert([_rec("a1", "audit", weight=25.0), _rec("d1", "deferred")])
    X, Y, y, w = s.training_set("tv", "e", ["a", "b"], 4)
    assert sorted(w.tolist()) == [1.0, 25.0]
