"""`jevstiller backup` / `jevstiller restore`: a consistent copy of a data directory, safe while serving.

Per task: SQLite's online backup API for `samples.sqlite` (a consistent snapshot even with the writer running),
`task.json`, and `versions/` (the registry index is copied first, so every version it lists is present in the
copy). The deployment's `key-salt` is copied too: without it, key hashes in a tenants file would no longer match.

Backups hold request text like the data dir, so they get the same modes (files 0600, directories 0700), and a
restored data dir is set to them too, whatever the source's modes (security audit run 2).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def _private() -> Iterator[None]:
    """Everything created inside is owner-only (umask 077), as `jevstiller serve` does."""
    old = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(old)


def _lock_down(root: Path) -> None:
    """Owner-only modes for `root` and everything below it (copies keep their source's modes)."""
    if not root.exists():
        return
    os.chmod(root, 0o700 if root.is_dir() else 0o600)
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            os.chmod(os.path.join(dirpath, d), 0o700)
        for f in filenames:
            os.chmod(os.path.join(dirpath, f), 0o600)


def _copy_versions(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "registry.json").exists():
        shutil.copy2(src / "registry.json", dst / "registry.json")
    for d in src.iterdir():
        if d.is_dir() and not d.name.startswith("."):          # skip staging dirs of running jobs
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True)


def backup(data_dir: str | Path, out: str | Path) -> dict:
    with _private():
        manifest = _backup(Path(data_dir), Path(out))
        _lock_down(Path(out))
    return manifest


def _backup(src: Path, dst: Path) -> dict:
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} is not empty")
    dst.mkdir(parents=True, exist_ok=True)
    tasks = 0
    for tdir in sorted((src / "tasks").glob("*")) if (src / "tasks").exists() else []:
        if not (tdir / "task.json").exists():
            continue
        out_t = dst / "tasks" / tdir.name
        out_t.mkdir(parents=True)
        shutil.copy2(tdir / "task.json", out_t / "task.json")
        if (tdir / "samples.sqlite").exists():
            s = sqlite3.connect(f"{(tdir / 'samples.sqlite').resolve().as_uri()}?mode=ro", uri=True)
            d = sqlite3.connect(out_t / "samples.sqlite")
            try:
                s.backup(d)
            finally:
                d.close()
                s.close()
        if (tdir / "versions").exists():
            _copy_versions(tdir / "versions", out_t / "versions")
        tasks += 1
    if (src / "key-salt").exists():
        shutil.copy2(src / "key-salt", dst / "key-salt")
    manifest = {"created": time.time(), "source": str(src.resolve()), "tasks": tasks}
    (dst / "backup.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def restore(backup_dir: str | Path, data_dir: str | Path, force: bool = False) -> dict:
    """Copy a backup into an empty data directory (or over an existing one with `force`). Stop the server first."""
    with _private():
        manifest = _restore(Path(backup_dir), Path(data_dir), force)
        dst = Path(data_dir)
        os.chmod(dst, 0o700)
        _lock_down(dst / "tasks")
        _lock_down(dst / "key-salt")
    return manifest


def _restore(src: Path, dst: Path, force: bool) -> dict:
    if not (src / "backup.json").exists():
        raise FileNotFoundError(f"{src} is not a jevstiller backup (no backup.json)")
    if dst.exists() and any(dst.iterdir()) and not force:
        raise FileExistsError(f"{dst} is not empty (use force to overwrite)")
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "tasks").exists() and force:
        shutil.rmtree(dst / "tasks")
    if (src / "tasks").exists():
        shutil.copytree(src / "tasks", dst / "tasks")
    if (src / "key-salt").exists():
        shutil.copy2(src / "key-salt", dst / "key-salt")
    return json.loads((src / "backup.json").read_text())
