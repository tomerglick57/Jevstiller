"""Immutable student versions on disk + a pointer to production."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .calibrate import RoutingPolicy
from .ood import KnnOOD
from .student import LinearStudent

STATES = ("candidate", "shadow", "production", "superseded", "rejected", "rolled_back")


@dataclass
class Bundle:
    name: str
    student: LinearStudent
    ood: KnnOOD
    policy: RoutingPolicy
    meta: dict


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms that cannot open directories (Windows)
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temp file in the same directory, fsync, then rename over `path`. A crash at any point
    leaves either the old file or the new one, never a partial one."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(path.parent)


def write_bundle(vdir: Path, student: LinearStudent, ood: KnnOOD, policy: RoutingPolicy) -> None:
    student.save(vdir / "head.npz")
    ood.save(vdir / "ood.npz")
    policy.save(vdir / "policy.json")


class Registry:
    """`registry.json` is the single source of truth: a version directory not listed there does not exist
    (it is a leftover from a crash and is removed on the next save)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"
        self._lock = threading.RLock()
        if not self.index.exists():
            self._write({"production": None, "versions": []})
        for leftover in self.root.glob(".*"):             # temp files/dirs of a save interrupted by a crash
            shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()

    @staticmethod
    def _require(d: dict, name: str) -> None:
        if not any(v["name"] == name for v in d["versions"]):
            raise ValueError(f"unknown version {name!r}")

    def _read(self) -> dict:
        return json.loads(self.index.read_text())

    def _write(self, d: dict) -> None:
        atomic_write_text(self.index, json.dumps(d, indent=2))

    @property
    def production(self) -> str | None:
        return self._read()["production"]

    def versions(self) -> list[dict]:
        return self._read()["versions"]

    def state(self, name: str) -> str | None:
        for v in self.versions():
            if v["name"] == name:
                return v["state"]
        return None

    def in_state(self, state: str) -> list[str]:
        return [v["name"] for v in self.versions() if v["state"] == state]

    def save(self, student: LinearStudent, ood: KnnOOD, policy: RoutingPolicy, meta: dict,
             state: str = "candidate") -> str:
        tmp = self.staging_dir()
        try:
            write_bundle(tmp, student, ood, policy)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self.adopt(tmp, meta, state)

    def staging_dir(self) -> Path:
        """A fresh directory to build a version in. Hidden, so a crash leaves nothing a restart keeps."""
        return Path(tempfile.mkdtemp(dir=self.root, prefix=".staging-"))

    def adopt(self, staged: Path, meta: dict, state: str = "candidate") -> str:
        """Register a directory holding head.npz, ood.npz and policy.json (e.g. written by a training
        worker) as the next version. The directory is moved, not copied."""
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {state!r}")
        staged = Path(staged)
        try:
            (staged / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
            for f in staged.iterdir():
                with open(f, "rb") as fh:
                    os.fsync(fh.fileno())
            with self._lock:
                d = self._read()
                name = f"student:v{len(d['versions']) + 1}"
                vdir = self.root / name.replace(":", "-")
                if vdir.exists():                        # orphan from a crash between move and index write
                    shutil.rmtree(vdir)
                os.replace(staged, vdir)
                _fsync_dir(self.root)
                d["versions"].append({"name": name, "state": state, "created": time.time(), "dir": str(vdir)})
                self._write(d)
                return name
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise

    def load(self, name: str) -> Bundle:
        vdir = self.root / name.replace(":", "-")
        return Bundle(name, LinearStudent.load(vdir / "head.npz"), KnnOOD.load(vdir / "ood.npz"),
                      RoutingPolicy.load(vdir / "policy.json"), json.loads((vdir / "meta.json").read_text()))

    def set_state(self, name: str, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {state!r}")
        with self._lock:
            d = self._read()
            self._require(d, name)
            for v in d["versions"]:
                if v["name"] == name:
                    v["state"] = state
            self._write(d)

    def promote(self, name: str) -> str | None:
        with self._lock:
            d = self._read()
            self._require(d, name)
            prev = d["production"]
            for v in d["versions"]:
                if v["name"] == prev:
                    v["state"] = "superseded"
                if v["name"] == name:
                    v["state"] = "production"
            d["production"] = name
            self._write(d)
            return prev

    def rollback(self) -> str | None:
        """Production -> rolled_back; the most recent superseded version becomes production."""
        with self._lock:
            d = self._read()
            cur = d["production"]
            prev = [v for v in d["versions"] if v["state"] == "superseded"]
            target = prev[-1]["name"] if prev else None
            for v in d["versions"]:
                if v["name"] == cur:
                    v["state"] = "rolled_back"
                if v["name"] == target:
                    v["state"] = "production"
            d["production"] = target
            self._write(d)
            return target
