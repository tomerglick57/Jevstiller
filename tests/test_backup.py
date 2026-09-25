"""P5.10: backup while serving, restore into a fresh data directory."""
import os
import stat
import threading

import pytest

from jevstiller import Admission, Config, HashEncoder, TaskManager
from jevstiller._backup import backup, restore
from jevstiller.server import load_salt

LABELS = ["billing", "technical", "cancellation", "sales", "other"]


def test_backup_while_writing_and_restore(tmp_path, teacher, world):
    src = tmp_path / "data"
    load_salt(src)
    cfg = Config(audit_rate=0.05, min_train_samples=400, min_samples_per_class=20, min_calib_samples=150,
                 min_new_samples=10**9, shadow_min_samples=150, training="inline")
    m = TaskManager(src, teacher, HashEncoder(dim=64), cfg, admission=Admission(min_requests=1),
                    target_agreement=0.95, janitor_interval_s=3600)
    for _ in range(30):
        m.classify("acme", "Q?", LABELS, [t for t, _ in world.sample(200)])
    key = m.tasks()[0].key
    prod = m.engine(key).status().production
    assert prod
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            m.classify("acme", "Q?", LABELS, [t for t, _ in world.sample(20)])
    t = threading.Thread(target=writer)
    t.start()
    manifest = backup(src, tmp_path / "bk")                      # while requests are being recorded
    stop.set()
    t.join()
    m.close()
    assert manifest["tasks"] == 1
    assert stat.S_IMODE(os.stat(tmp_path / "bk" / "key-salt").st_mode) == 0o600
    with pytest.raises(FileExistsError):
        backup(src, tmp_path / "bk")

    dst = tmp_path / "restored"
    restore(tmp_path / "bk", dst)
    m2 = TaskManager(dst, teacher, HashEncoder(dim=64), cfg, admission=Admission(min_requests=100),
                     target_agreement=0.95, janitor_interval_s=3600)
    e = m2.engine(key)
    assert e.status().production == prod and e.store.counts(e.task.version)["total"] >= 6000
    assert (dst / "key-salt").read_bytes() == (src / "key-salt").read_bytes()
    res = m2.classify("acme", "Q?", LABELS, [t for t, _ in world.sample(200)])
    assert any(r.source.startswith("student") for r in res)     # serves from the restored model
    m2.close()
    with pytest.raises(FileExistsError):
        restore(tmp_path / "bk", dst)
