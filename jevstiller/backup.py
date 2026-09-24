"""`jevstiller backup` / `jevstiller restore`: a consistent copy of a data directory, safe while serving.

Per task: SQLite's online backup API for `samples.sqlite` (a consistent snapshot even with the writer running),
`task.json`, and `versions/` (the registry index is copied first, so every version it lists is present in the
copy). The deployment's `key-salt` is copied too (mode 0600): without it, key hashes in a tenants file would
no longer match.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path


def _copy_versions(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "registry.json").exists():
        shutil.copy2(src / "registry.json", dst / "registry.json")
    for d in src.iterdir():
        if d.is_dir() and not d.name.startswith("."):          # skip staging dirs of running jobs
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True)


def backup(data_dir: str | Path, out: str | Path) -> dict:
    src, dst = Path(data_dir), Path(out)
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
        os.chmod(dst / "key-salt", 0o600)
    manifest = {"created": time.time(), "source": str(src.resolve()), "tasks": tasks}
    (dst / "backup.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def restore(backup_dir: str | Path, data_dir: str | Path, force: bool = False) -> dict:
    """Copy a backup into an empty data directory (or over an existing one with `force`). Stop the server first."""
    src, dst = Path(backup_dir), Path(data_dir)
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
        os.chmod(dst / "key-salt", 0o600)
    return json.loads((src / "backup.json").read_text())
