import numpy as np

from jevstiller._store import Record, SampleStore, split_for


def _rec(text, channel, **kw):
    return Record(text=text, task_version="tv", encoder_id="e", embedding=np.ones(4, np.float32),
                  served_by="teacher", routing_reason=channel, channel=channel,
                  teacher_label="a", teacher_probs={"a": 0.7, "b": 0.3}, **kw)


def test_split_is_per_request_and_proportional(tmp_path):
    """Copies of one text are split independently, like any other requests (security audit run 3)."""
    s = SampleStore(tmp_path / "s.sqlite", calib_fraction=0.2)
    s.insert([_rec("the same text", "audit") for _ in range(5000)])
    frac = s.counts("tv")["labelled_calib"] / 5000
    assert 0.17 < frac < 0.23
    assert split_for(0.1, 0.2) == "calib" and split_for(0.3, 0.2) == "train"


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
