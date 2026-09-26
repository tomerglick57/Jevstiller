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


def _answered(text, channel, model="m1", **kw):
    return _rec(text, channel, teacher_model=model, teacher_cost_usd=0.01, **kw)


def _local(text):
    return Record(text=text, task_version="tv", encoder_id="e", embedding=np.ones(4, np.float32),
                  served_by="student", routing_reason="confident", channel="student", student_label="a")


def _groups(s):
    """Row ids per (lineage, split, answered), oldest first."""
    g = {}
    for i, tm, sp, lab in s.db.execute("SELECT id, teacher_model, split, teacher_label IS NOT NULL FROM samples "
                                       "ORDER BY id"):
        g.setdefault((tm, sp, lab), []).append(i)
    return g


def _totals(c):
    return {k: c[k] for k in ("total", "served_by", "channel", "split", "teacher_calls")} | {
        "cost": round(c["teacher_cost_usd"], 6)}


def _mixed(s, rounds=6):
    for r in range(rounds):                          # interleaved, like traffic: answers of two lineages, local rows
        s.insert([_answered(f"a{r}.{i}", "audit") for i in range(40)] + [_local(f"l{r}.{i}") for i in range(100)]
                 + [_answered(f"d{r}.{i}", "deferred", model="m2") for i in range(10)])


def test_prune_keeps_the_newest_rows_of_each_kind_and_the_totals(tmp_path):
    """24-hour soak: every request stayed in the store for good (31.7 GB). A reader takes at most the newest rows of
    one lineage, so the rest can go; the status totals must not notice."""
    s = SampleStore(tmp_path / "s.sqlite", calib_fraction=0.3, write_behind=False)
    _mixed(s)
    before, totals = _groups(s), _totals(s.counts("tv"))
    train = s.training_set("tv", "e", ["a", "b"], 4, teacher_model="m1", limit=50)
    calib = s.calib_set("tv", "e", ["a", "b"], 4, teacher_model="m1", limit=20)
    assert s.prune(keep_train=50, keep_calib=20, keep_unanswered=150) == (240 - 50 - 20 + 60 - 50 + 600 - 150, False)
    after = _groups(s)
    for (tm, sp, lab), ids in before.items():
        keep = 150 if not lab else 20 if sp == "calib" else 50
        assert after[(tm, sp, lab)] == ids[-keep:], (tm, sp, lab)
    assert _totals(s.counts("tv")) == totals
    for got, want in zip(s.training_set("tv", "e", ["a", "b"], 4, teacher_model="m1", limit=50), train, strict=True):
        assert np.array_equal(got, want)
    for got, want in zip(s.calib_set("tv", "e", ["a", "b"], 4, teacher_model="m1", limit=20), calib, strict=True):
        assert np.array_equal(got, want)
    assert s.prune(keep_train=50, keep_calib=20, keep_unanswered=150) == (0, False)
    _mixed(s, rounds=1)                              # deleted rows keep adding up
    s.prune(keep_train=50, keep_calib=20, keep_unanswered=150)
    assert s.counts("tv")["total"] == 7 * 150 and s.counts("tv")["served_by"] == {"teacher": 7 * 50, "student": 700}


def test_prune_keeps_every_row_of_a_kind_at_zero_and_rows_after_keep_after_id(tmp_path):
    s = SampleStore(tmp_path / "s.sqlite", calib_fraction=0.3, write_behind=False)
    _mixed(s)
    s.insert([_rec("no model", "deferred")])         # an answer without its model: never deleted
    assert s.prune(keep_train=0, keep_calib=0, keep_unanswered=0) == (0, False)
    local = _groups(s)[(None, "train", False)]
    shadow_from = local[-1] - 350                     # a shadow started here: its rows stay until it is judged
    n, more = s.prune(keep_train=0, keep_calib=0, keep_unanswered=10, keep_after_id=shadow_from)
    kept = _groups(s)[(None, "train", False)]
    assert kept == [i for i in local if i > shadow_from] and len(kept) > 10 and (n, more) == (600 - len(kept), False)
    assert (None, "train", True) in _groups(s)


def test_a_backlog_is_pruned_over_several_calls(tmp_path):
    s = SampleStore(tmp_path / "s.sqlite", write_behind=False)
    s.insert([_local(f"l{i}") for i in range(1000)])
    assert s.prune(0, 0, keep_unanswered=100, max_rows=400, batch=150) == (400, True)
    assert s.prune(0, 0, keep_unanswered=100, max_rows=400, batch=150) == (400, True)
    assert s.prune(0, 0, keep_unanswered=100, max_rows=400, batch=150) == (100, False)
    assert s.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 100 and s.counts("tv")["total"] == 1000


def _size(s):
    s.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return s.path.stat().st_size


def test_a_new_store_hands_deleted_space_back(tmp_path):
    s = SampleStore(tmp_path / "s.sqlite", write_behind=False)
    assert s.db.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
    s.insert([_local("x" * 4000 + str(i)) for i in range(2000)])
    full = _size(s)
    s.prune(0, 0, keep_unanswered=100)
    assert _size(s) < full / 5


def test_an_older_store_is_rebuilt_once_most_of_it_is_free(tmp_path):
    import sqlite3
    old = sqlite3.connect(tmp_path / "s.sqlite")      # a store from before auto_vacuum: tables already there
    old.execute("CREATE TABLE placeholder (x)")
    old.commit()
    old.close()
    s = SampleStore(tmp_path / "s.sqlite", write_behind=False)
    assert s.db.execute("PRAGMA auto_vacuum").fetchone()[0] == 0
    assert s.db.execute("SELECT COUNT(*) FROM deleted_rows").fetchone()[0] == 0     # migrated
    s.insert([_local("x" * 4000 + str(i)) for i in range(2000)])
    full = _size(s)
    assert s.prune(0, 0, keep_unanswered=1500, max_rows=100) == (100, True)
    assert s.db.execute("PRAGMA auto_vacuum").fetchone()[0] == 0                   # not while a backlog is left
    s.prune(0, 0, keep_unanswered=1500)
    assert s.db.execute("PRAGMA auto_vacuum").fetchone()[0] == 0 and _size(s) >= full  # a quarter free: no rebuild
    s.prune(0, 0, keep_unanswered=100)
    assert s.db.execute("PRAGMA auto_vacuum").fetchone()[0] == 2 and _size(s) < full / 5
    assert s.counts("tv")["total"] == 2000
