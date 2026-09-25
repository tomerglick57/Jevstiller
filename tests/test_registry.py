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
        reg = Registry({str(tmp_path)!r})
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
        reg = Registry(tmp_path)
        for v in d["versions"]:
            reg.load(v["name"])                                        # every listed version is complete
        if d["production"]:
            assert reg.state(d["production"]) == "production"
    assert d["versions"], "the writer never got to save anything"
