"""Immutable student versions on disk + a pointer to production."""
from __future__ import annotations

import json
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


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"
        if not self.index.exists():
            self._write({"production": None, "versions": []})

    def _read(self) -> dict:
        return json.loads(self.index.read_text())

    def _write(self, d: dict) -> None:
        self.index.write_text(json.dumps(d, indent=2))

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
        d = self._read()
        name = f"student:v{len(d['versions']) + 1}"
        vdir = self.root / name.replace(":", "-")
        vdir.mkdir()
        student.save(vdir / "head.npz")
        ood.save(vdir / "ood.npz")
        policy.save(vdir / "policy.json")
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        d["versions"].append({"name": name, "state": state, "created": time.time(), "dir": str(vdir)})
        self._write(d)
        return name

    def load(self, name: str) -> Bundle:
        vdir = self.root / name.replace(":", "-")
        return Bundle(name, LinearStudent.load(vdir / "head.npz"), KnnOOD.load(vdir / "ood.npz"),
                      RoutingPolicy.load(vdir / "policy.json"), json.loads((vdir / "meta.json").read_text()))

    def set_state(self, name: str, state: str) -> None:
        assert state in STATES
        d = self._read()
        for v in d["versions"]:
            if v["name"] == name:
                v["state"] = state
        self._write(d)

    def promote(self, name: str) -> str | None:
        d = self._read()
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
