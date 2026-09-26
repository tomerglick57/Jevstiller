"""P1.4: registry.json is always readable, whatever point a save or promote dies at."""
import json
import os
import random
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import jevstiller._registry as registry_mod
from jevstiller._calibrate import RoutingPolicy
from jevstiller._ood import KnnOOD
from jevstiller._registry import Registry
from jevstiller._student import LinearStudent


def _parts():
    ood = KnnOOD(2)
    ood.fit(np.eye(4, dtype=np.float32))
    return LinearStudent(4, 2), ood, RoutingPolicy(0.5, 0.9, 0.8, 0.01, 0.008, 0.02, 0.05, 100)


def _save(reg, state="shadow"):
    return reg.save(*_parts(), meta={}, state=state)


def test_failed_index_write_leaves_the_old_index(tmp_path, monkeypatch):
    reg = Registry(tmp_path)
    v1 = _save(reg)
    reg.promote(v1)
    v2 = _save(reg)
    before = reg.index.read_text()

    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(registry_mod.os, "replace", boom)
    with pytest.raises(OSError):
        reg.promote(v2)
    monkeypatch.undo()
    assert reg.index.read_text() == before and reg.production == v1
    assert Registry(tmp_path).production == v1
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_orphan_version_dir_is_replaced(tmp_path):
    reg = Registry(tmp_path)
    (tmp_path / "student-v1").mkdir()                        # a crash after the copy, before the index write
    (tmp_path / "student-v1" / "junk").write_text("x")
    name = _save(reg)
    assert name == "student:v1" and not (tmp_path / "student-v1" / "junk").exists()
    reg.load(name)


def test_unknown_version_and_state_are_rejected(tmp_path):
    reg = Registry(tmp_path)
    with pytest.raises(ValueError):
        reg.promote("student:v1")
    name = _save(reg)
    with pytest.raises(ValueError):
        reg.set_state(name, "live")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_kill_during_save_and_promote(tmp_path):
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from test_registry import _save
        from jevstiller._registry import Registry
        reg = Registry({str(tmp_path)!r}, keep=2)
        print("ready", flush=True)
        while True:
            reg.promote(_save(reg))
    """)
    rng = random.Random(0)
    for _ in range(5):
        p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        assert p.stdout.readline().strip() == "ready"
        time.sleep(rng.uniform(0.05, 0.5))
        p.send_signal(signal.SIGKILL)
        p.wait()
        d = json.loads((tmp_path / "registry.json").read_text())      # never partial
        reg = Registry(tmp_path, keep=2)
        for v in d["versions"]:
            if not v.get("deleted"):
                reg.load(v["name"])                                    # every version not deleted is complete
        if d["production"]:
            assert reg.state(d["production"]) == "production"
        reg.prune()                                                    # a kill between marking and deleting
        assert not [v for v in d["versions"] if v.get("deleted") and reg._dir(v["name"]).exists()]
    assert d["versions"], "the writer never got to save anything"


def _on_disk(reg):
    return sorted((p.name for p in reg.root.iterdir() if p.is_dir()), key=lambda n: int(n.split("-v")[1]))


def test_only_the_newest_finished_versions_keep_their_files(tmp_path):
    """24-hour soak: every superseded version stayed on disk (~10 MB each, 645 in one task by the end)."""
    reg = Registry(tmp_path, keep=2)
    for _ in range(5):                                         # v1..v5 promoted in turn: v1..v4 superseded
        reg.promote(_save(reg))
    for _ in range(3):                                         # v6..v8 rejected
        _save(reg, state="rejected")
    reg.promote(shadow := _save(reg))                          # v9 production, v5 superseded
    _save(reg, state="shadow")                                 # v10 shadow, the newest
    assert _on_disk(reg) == ["student-v4", "student-v5", "student-v7", "student-v8", "student-v9", "student-v10"]
    listed = {v["name"]: v for v in reg.versions()}
    assert len(listed) == 10 and listed["student:v1"]["state"] == "superseded" and listed["student:v1"]["deleted"]
    assert _save(reg, state="rejected") == "student:v11"       # names are never reused
    with pytest.raises(ValueError, match="deleted"):
        reg.load("student:v1")
    with pytest.raises(ValueError, match="deleted"):
        reg.promote("student:v6")
    assert reg.production == shadow and reg.rollback() == "student:v5"
    assert reg.rollback() == "student:v4" and reg.rollback() is None     # v3 and older are gone


def test_the_newest_version_is_kept_whatever_its_state(tmp_path):
    """It holds the last training's position in the store (trained_to_id), read back on a restart."""
    reg = Registry(tmp_path, keep=0)
    reg.promote(_save(reg))
    reg.promote(_save(reg))
    _save(reg, state="rejected")
    _save(reg, state="rejected")
    assert _on_disk(reg) == ["student-v2", "student-v4"]
    assert Registry(tmp_path).meta("student:v4") == {}


def test_versions_kept_before_a_limit_are_deleted_by_prune(tmp_path):
    reg = Registry(tmp_path)                                   # keep=None: every version stays
    for _ in range(4):
        reg.promote(_save(reg))
    assert len(_on_disk(reg)) == 4
    reg = Registry(tmp_path, keep=1)
    assert _on_disk(reg) == ["student-v1", "student-v2", "student-v3", "student-v4"]   # opening deletes nothing
    assert reg.prune() == 2 and _on_disk(reg) == ["student-v3", "student-v4"]
    assert reg.prune() == 0
    with pytest.raises(ValueError, match="keep"):
        Registry(tmp_path, keep=-1)


def test_a_crash_before_the_delete_leaves_nothing_behind(tmp_path, monkeypatch):
    reg = Registry(tmp_path, keep=1)
    reg.promote(_save(reg))
    reg.promote(_save(reg))
    monkeypatch.setattr(registry_mod.shutil, "rmtree", lambda *a, **kw: None)     # the process dies here
    reg.promote(_save(reg))
    monkeypatch.undo()
    assert _on_disk(reg) == ["student-v1", "student-v2", "student-v3"]
    assert [v["name"] for v in reg.versions() if v.get("deleted")] == ["student:v1"]
    assert reg.prune() == 1 and _on_disk(reg) == ["student-v2", "student-v3"]
