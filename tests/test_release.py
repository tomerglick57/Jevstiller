"""store_text=False, thread-safety, versions() and export()."""
import json
import threading

from jevstiller import Config, Jevstiller


def _cfg(**kw):
    base = dict(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                min_new_samples=1500, shadow_min_samples=150, drift_min_samples=100, seed=0, training="inline")
    base.update(kw)
    return Config(**base)


def test_store_text_false_keeps_hash_only(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg(store_text=False))
    texts = [t for t, _ in world.sample(50)]
    js.classify_batch(texts)
    rows = js.store.db.execute("SELECT text, text_hash, embedding IS NOT NULL FROM samples").fetchall()
    assert all(r[0] == "" and len(r[1]) == 16 and r[2] == 1 for r in rows)
    assert js.store.counts(task.version)["total"] == 50


def test_concurrent_classify(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path, config=_cfg())
    errors = []

    def worker():
        try:
            for _ in range(5):
                js.classify_batch([t for t, _ in world.sample(40)])
        except Exception as e:  # pragma: no cover
            errors.append(e)
    ts = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and js.store.counts(task.version)["total"] == 6 * 5 * 40


def test_versions_and_export(tmp_path, task, world, teacher):
    js = Jevstiller(task, teacher, tmp_path / "d", config=_cfg())
    for _ in range(20):
        js.classify_batch([t for t, _ in world.sample(200)])
    vs = js.versions()
    assert vs and any(v["state"] == "production" for v in vs)
    out = js.export(tmp_path / "bundle")
    names = {p.name for p in out.iterdir()}
    assert {"head.npz", "ood.npz", "policy.json", "meta.json", "task.json"} <= names
    meta = json.loads((out / "task.json").read_text())
    assert meta["version"] == js.status().production and meta["encoder_id"] == js.encoder.id
