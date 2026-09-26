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

from ._calibrate import RoutingPolicy
from ._ood import KnnOOD
from ._student import LinearStudent

STATES = ("candidate", "shadow", "production", "superseded", "rejected", "rolled_back")
LIVE = ("candidate", "shadow", "production")    # never deleted by `keep`


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
    (it is a leftover from a crash and is removed on the next save).

    `keep`: the files of only the newest `keep` versions of each finished state (superseded, rejected,
    rolled_back) are kept; older ones are deleted on the next write (or `prune()`). Candidate, shadow and
    production versions and the newest version are always kept. A deleted version stays listed, marked
    `deleted`, so its name is never reused. None keeps every version."""

    def __init__(self, root: Path, keep: int | None = None):
        if keep is not None and keep < 0:
            raise ValueError(f"keep must be >= 0 or None, got {keep}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"
        self.keep = keep
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

    def _dir(self, name: str) -> Path:
        return self.root / name.replace(":", "-")

    def _write(self, d: dict) -> None:
        """Write the index, then delete the files of the versions past `keep`: marked first, so a crash in
        between leaves a directory the next write removes, never a listed version with missing files."""
        doomed = self._expire(d)
        atomic_write_text(self.index, json.dumps(d, indent=2))
        for vdir in doomed:
            shutil.rmtree(vdir, ignore_errors=True)

    def _expire(self, d: dict) -> list[Path]:
        if self.keep is None:
            return []
        seen: dict[str, int] = {}
        for i, v in enumerate(reversed(d["versions"])):
            if i == 0 or v.get("deleted") or v["state"] in LIVE or v["name"] == d["production"]:
                continue
            seen[v["state"]] = seen.get(v["state"], 0) + 1
            if seen[v["state"]] > self.keep:
                v["deleted"] = time.time()
        return [p for v in d["versions"] if v.get("deleted") and (p := self._dir(v["name"])).exists()]

    def prune(self) -> int:
        """Delete the files of the versions past `keep` now (otherwise the next write does). Returns how many
        version directories were removed."""
        with self._lock:
            d = self._read()
            doomed = self._expire(d)
            if doomed:
                self._write(d)
            return len(doomed)

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
                vdir = self._dir(name)
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

    def _files(self, name: str) -> Path:
        vdir = self._dir(name)
        if not vdir.exists() and any(v["name"] == name and v.get("deleted") for v in self.versions()):
            raise ValueError(f"{name} was deleted: only the newest {self.keep} versions of each finished state "
                             f"are kept (keep_versions)")
        return vdir

    def meta(self, name: str) -> dict:
        """A version's metadata alone (no arrays)."""
        return json.loads((self._files(name) / "meta.json").read_text())

    def load(self, name: str) -> Bundle:
        vdir = self._files(name)
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
            self._files(name)                           # refuses a deleted version
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
        """Production -> rolled_back; the most recent superseded version still on disk becomes production."""
        with self._lock:
            d = self._read()
            cur = d["production"]
            prev = [v for v in d["versions"] if v["state"] == "superseded" and not v.get("deleted")]
            target = prev[-1]["name"] if prev else None
            for v in d["versions"]:
                if v["name"] == cur:
                    v["state"] = "rolled_back"
                if v["name"] == target:
                    v["state"] = "production"
            d["production"] = target
            self._write(d)
            return target
