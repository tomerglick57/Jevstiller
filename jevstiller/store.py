"""Append-only sample store on SQLite. One row per request, whoever served it."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("jevstiller")

IID_CHANNELS = ("bootstrap", "audit", "fallback")   # only these may feed calibration / evaluation
TEACHER_CHANNELS = IID_CHANNELS + ("deferred",)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def split_for(text_h: str, calib_fraction: float) -> str:
    return "calib" if int(text_h[:8], 16) % 10_000 < calib_fraction * 10_000 else "train"


@dataclass
class Record:
    text: str
    task_version: str
    encoder_id: str
    embedding: np.ndarray | None
    served_by: str                      # teacher | student
    routing_reason: str
    channel: str                        # bootstrap | audit | fallback | deferred | student
    state_type: str = "text"            # text | json (then `text` is the canonical JSON of the state)
    ts: float = field(default_factory=time.time)
    weight: float = 1.0                 # importance weight: how many requests this row stands for
    text_hash: str | None = None        # set by the caller when text is not stored
    latency_ms: float = 0.0
    teacher_label: str | None = None
    teacher_probs: dict | None = None
    teacher_confidence: float | None = None
    teacher_model: str | None = None
    teacher_input_tokens: int | None = None
    teacher_cost_usd: float | None = None
    teacher_request_id: str | None = None
    teacher_latency_ms: float | None = None
    student_version: str | None = None
    student_label: str | None = None
    student_probs: dict | None = None
    student_confidence: float | None = None
    ood_score: float | None = None
    shadow_version: str | None = None
    shadow_label: str | None = None
    shadow_confidence: float | None = None
    shadow_ood: float | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY, ts REAL, task_version TEXT, text TEXT, text_hash TEXT,
  encoder_id TEXT, embedding BLOB,
  served_by TEXT, routing_reason TEXT, channel TEXT, split TEXT, latency_ms REAL, weight REAL,
  teacher_label TEXT, teacher_probs TEXT, teacher_confidence REAL, teacher_model TEXT,
  teacher_input_tokens INTEGER, teacher_cost_usd REAL, teacher_request_id TEXT, teacher_latency_ms REAL,
  student_version TEXT, student_label TEXT, student_probs TEXT, student_confidence REAL, ood_score REAL,
  shadow_version TEXT, shadow_label TEXT, shadow_confidence REAL, shadow_ood REAL
);
CREATE INDEX IF NOT EXISTS ix_samples_tv_split ON samples(task_version, split, channel);
CREATE INDEX IF NOT EXISTS ix_samples_shadow ON samples(shadow_version);
CREATE INDEX IF NOT EXISTS ix_samples_ts ON samples(ts);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, detail TEXT);
"""

SCHEMA_VERSION = 1                     # PRAGMA user_version; 0 = a 0.1.0 store (or a new file)

# columns added after 0.1.0: (name, type). Existing stores get them via ALTER TABLE on open.
_ADDED_COLUMNS = (("state_type", "TEXT"),)
_INDEXES_AFTER_MIGRATION = """
CREATE INDEX IF NOT EXISTS ix_samples_lineage ON samples(task_version, teacher_model, split);
"""


def _lineage(teacher_model: str | None) -> tuple[str, tuple]:
    """SQL filter for one teacher lineage; None = every teacher."""
    return (" AND teacher_model=?", (teacher_model,)) if teacher_model is not None else ("", ())


@runtime_checkable
class Store(Protocol):
    """What the engine and the training job need from a sample store. `SampleStore` (one SQLite file per
    task) is the implementation; a shared database (e.g. Postgres, for several replicas) would implement the
    same methods, plus a way for `training.run_fit_job` to open it read-only from a worker process (today it
    reopens `SampleStore(path, read_only=True)`)."""

    path: Path

    def insert(self, recs: Sequence[Record]) -> None: ...
    def event(self, kind: str, **detail) -> None: ...
    def flush(self, timeout: float | None = None) -> bool: ...
    def counts(self, task_version: str, teacher_model: str | None = None) -> dict: ...
    def training_set(self, task_version, encoder_id, labels, dim, teacher_model=None): ...
    def calib_set(self, task_version, encoder_id, labels, dim, teacher_model=None): ...
    def shadow_records(self, shadow_version: str, task_version: str, teacher_model: str | None = None): ...
    def audit_window(self, task_version: str, encoder_id: str, dim: int, limit: int,
                     teacher_model: str | None = None): ...
    def latest_teacher_model(self, task_version: str) -> str | None: ...
    def max_id(self) -> int: ...
    def events(self, limit: int = 20) -> list[dict]: ...
    def close(self) -> None: ...


class SampleStore:
    """Append-only store. Thread-safe.

    Writes are write-behind by default: `insert` queues records and returns; one writer thread commits them
    in batches, so no request waits on the disk. Every read through this class (including `.db`) first
    waits for records queued before it, so reads always see earlier inserts. A crash loses at most the
    records still queued (typically a few milliseconds of traffic). `write_behind=False` commits in `insert`.

    Each thread gets its own SQLite connection (WAL), so reads never block the writer.
    """

    def __init__(self, path: Path, calib_fraction: float = 0.2, busy_timeout_s: float = 30.0,
                 write_behind: bool = True, max_pending: int = 100_000, max_batch: int = 5_000,
                 read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        self.calib_fraction = calib_fraction
        self.busy_timeout_s = busy_timeout_s
        self.write_behind = write_behind
        self.max_pending, self.max_batch = max_pending, max_batch
        self.write_errors = 0
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False
        self._write_lock = threading.Lock()   # writers queue here, not in SQLite's sleeping busy handler
        self._wcv = threading.Condition()
        self._pending: deque[Record] = deque()
        self._enqueued = self._written = 0
        self._stopping = False
        self._writer: threading.Thread | None = None
        if not read_only:
            conn = self._conn()
            if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:   # new or older store
                conn.execute("PRAGMA journal_mode=WAL")                            # persists in the file
                conn.executescript(_SCHEMA)
                have = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
                with conn:
                    for col, typ in _ADDED_COLUMNS:
                        if col not in have:
                            conn.execute(f"ALTER TABLE samples ADD COLUMN {col} {typ}")
                conn.executescript(_INDEXES_AFTER_MIGRATION)
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _conn(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            with self._conns_lock:
                if self._closed:
                    raise sqlite3.ProgrammingError("the sample store is closed")
                if self.read_only:                              # e.g. a training worker process
                    conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True,
                                           timeout=self.busy_timeout_s, check_same_thread=False)
                else:
                    conn = sqlite3.connect(self.path, timeout=self.busy_timeout_s, check_same_thread=False)
                    conn.execute("PRAGMA synchronous=NORMAL")  # durable across crashes in WAL mode
                self._conns.append(conn)
            self._local.conn = conn
        return conn

    @property
    def db(self) -> sqlite3.Connection:
        """This thread's connection, once every record queued so far is committed."""
        self.flush()
        return self._conn()

    # ---- writes -------------------------------------------------------------
    def insert(self, recs: Sequence[Record]) -> None:
        if not recs:
            return
        if self.read_only:
            raise sqlite3.ProgrammingError("the sample store is read-only")
        if not self.write_behind:
            self._write(recs)
            return
        with self._wcv:
            if self._stopping:
                raise sqlite3.ProgrammingError("the sample store is closed")
            while len(self._pending) >= self.max_pending:  # backpressure if the disk falls behind
                self._wcv.wait()
            self._pending.extend(recs)
            self._enqueued += len(recs)
            if self._writer is None:
                self._writer = threading.Thread(target=self._writer_loop, daemon=True,
                                                name=f"jevstiller-writer-{self.path.parent.name}")
                self._writer.start()
            self._wcv.notify_all()

    def _writer_loop(self) -> None:
        while True:
            with self._wcv:
                while not self._pending and not self._stopping:
                    self._wcv.wait()
                if not self._pending:
                    return
                batch = [self._pending.popleft() for _ in range(min(len(self._pending), self.max_batch))]
                self._wcv.notify_all()                       # room again for back-pressured inserts
            failed = False
            try:
                self._write(batch)
            except Exception:
                failed = True
                log.exception("sample store %s: dropped %d records", self.path, len(batch))
            with self._wcv:
                self._written += len(batch)
                self.write_errors += len(batch) if failed else 0
                self._wcv.notify_all()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until every record queued before this call is committed (or dropped on error)."""
        if threading.current_thread() is self._writer:
            return True
        with self._wcv:
            target = self._enqueued
            return self._wcv.wait_for(lambda: self._written >= target, timeout)

    def _write(self, recs: Sequence[Record]) -> None:
        rows = []
        for r in recs:
            h = r.text_hash or text_hash(r.text)
            split = split_for(h, self.calib_fraction) if r.channel in IID_CHANNELS else "train"
            emb = r.embedding.astype(np.float32).tobytes() if r.embedding is not None else None
            rows.append((r.ts, r.task_version, r.text, h, r.encoder_id, emb,
                         r.served_by, r.routing_reason, r.channel, split, r.latency_ms, r.weight,
                         r.teacher_label, json.dumps(r.teacher_probs) if r.teacher_probs else None,
                         r.teacher_confidence, r.teacher_model, r.teacher_input_tokens, r.teacher_cost_usd,
                         r.teacher_request_id, r.teacher_latency_ms,
                         r.student_version, r.student_label,
                         json.dumps(r.student_probs) if r.student_probs else None,
                         r.student_confidence, r.ood_score,
                         r.shadow_version, r.shadow_label, r.shadow_confidence, r.shadow_ood, r.state_type))
        conn = self._conn()
        with self._write_lock, conn:
            conn.executemany(
                "INSERT INTO samples (ts,task_version,text,text_hash,encoder_id,embedding,"
                "served_by,routing_reason,channel,split,latency_ms,weight,"
                "teacher_label,teacher_probs,teacher_confidence,teacher_model,teacher_input_tokens,"
                "teacher_cost_usd,teacher_request_id,teacher_latency_ms,"
                "student_version,student_label,student_probs,student_confidence,ood_score,"
                "shadow_version,shadow_label,shadow_confidence,shadow_ood,state_type) "
                "VALUES (" + ",".join("?" * 30) + ")", rows)

    def event(self, kind: str, **detail) -> None:
        conn = self._conn()
        with self._write_lock, conn:
            conn.execute("INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
                            (time.time(), kind, json.dumps(detail, default=str)))

    # ---- reads --------------------------------------------------------------
    def _matrix(self, rows: list, labels: Sequence[str], dim: int):
        X = np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32).reshape(len(rows), dim) if rows \
            else np.zeros((0, dim), np.float32)
        idx = {c: i for i, c in enumerate(labels)}
        Y = np.zeros((len(rows), len(labels)), np.float32)
        y = np.zeros(len(rows), np.int64)
        for i, r in enumerate(rows):
            probs = json.loads(r[1])
            for c, p in probs.items():
                if c in idx:
                    Y[i, idx[c]] = p
            y[i] = idx.get(r[2], -1)
        s = Y.sum(axis=1, keepdims=True)
        np.divide(Y, s, out=Y, where=s > 0)
        w = np.array([r[3] if len(r) > 3 and r[3] is not None else 1.0 for r in rows], np.float64)
        return X, Y, y, w

    def labelled(self, task_version: str, encoder_id: str, labels: Sequence[str], dim: int,
                 split: str, channels: Iterable[str] = TEACHER_CHANNELS, teacher_model: str | None = None):
        """Embeddings X, teacher distributions Y, teacher argmax y, importance weights w for teacher-labelled rows
        (of one teacher lineage when `teacher_model` is given)."""
        ch = tuple(channels)
        lin, lin_args = _lineage(teacher_model)
        rows = self.db.execute(
            f"SELECT embedding, teacher_probs, teacher_label, weight FROM samples "
            f"WHERE task_version=? AND encoder_id=?{lin} "
            f"AND split=? AND teacher_label IS NOT NULL AND channel IN ({','.join('?' * len(ch))}) ORDER BY id",
            (task_version, encoder_id, *lin_args, split, *ch)).fetchall()
        return self._matrix(rows, labels, dim)

    def training_set(self, task_version, encoder_id, labels, dim, teacher_model=None):
        return self.labelled(task_version, encoder_id, labels, dim, "train", teacher_model=teacher_model)

    def calib_set(self, task_version, encoder_id, labels, dim, teacher_model=None):
        return self.labelled(task_version, encoder_id, labels, dim, "calib", IID_CHANNELS, teacher_model)

    def shadow_records(self, shadow_version: str, task_version: str, teacher_model: str | None = None):
        """IID rows where the shadow candidate ran alongside a teacher answer."""
        ch = IID_CHANNELS
        lin, lin_args = _lineage(teacher_model)
        rows = self.db.execute(
            f"SELECT shadow_label, shadow_confidence, shadow_ood, teacher_label, student_label, student_confidence, "
            f"ood_score FROM samples WHERE shadow_version=? AND task_version=?{lin} AND teacher_label IS NOT NULL "
            f"AND channel IN ({','.join('?' * len(ch))})", (shadow_version, task_version, *lin_args, *ch)).fetchall()
        return rows

    def audit_window(self, task_version: str, encoder_id: str, dim: int, limit: int, teacher_model: str | None = None):
        """Embeddings and teacher labels of the most recent audit rows, whichever student scored them."""
        lin, lin_args = _lineage(teacher_model)
        rows = self.db.execute(
            f"SELECT embedding, teacher_label FROM samples WHERE task_version=? AND encoder_id=?{lin} "
            f"AND channel='audit' AND teacher_label IS NOT NULL AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?",
            (task_version, encoder_id, *lin_args, limit)).fetchall()
        if not rows:
            return np.zeros((0, dim), np.float32), np.array([], dtype=object)
        X = np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        return X, np.array([r[1] for r in rows], dtype=object)

    def counts(self, task_version: str, teacher_model: str | None = None) -> dict:
        """Totals over every teacher; the labelled counts (what training can use) over one lineage if given."""
        lin, lin_args = _lineage(teacher_model)
        c = {}
        c["total"] = self.db.execute("SELECT COUNT(*) FROM samples WHERE task_version=?", (task_version,)).fetchone()[0]
        for k, in_ in (("served_by", "served_by"), ("channel", "channel"), ("split", "split")):
            c[k] = dict(self.db.execute(f"SELECT {in_}, COUNT(*) FROM samples WHERE task_version=? GROUP BY {in_}",
                                        (task_version,)).fetchall())
        c["per_class_train"] = dict(self.db.execute(
            f"SELECT teacher_label, COUNT(*) FROM samples WHERE task_version=?{lin} AND split='train' "
            f"AND teacher_label IS NOT NULL GROUP BY teacher_label", (task_version, *lin_args)).fetchall())
        c["labelled_train"] = sum(c["per_class_train"].values())
        c["labelled_calib"] = self.db.execute(
            f"SELECT COUNT(*) FROM samples WHERE task_version=?{lin} AND split='calib' AND teacher_label IS NOT NULL",
            (task_version, *lin_args)).fetchone()[0]
        c["teacher_cost_usd"] = self.db.execute(
            "SELECT COALESCE(SUM(teacher_cost_usd),0) FROM samples WHERE task_version=?", (task_version,)).fetchone()[0]
        c["teacher_calls"] = self.db.execute(
            "SELECT COUNT(*) FROM samples WHERE task_version=? AND teacher_label IS NOT NULL",
            (task_version,)).fetchone()[0]
        return c

    def latest_teacher_model(self, task_version: str) -> str | None:
        """The teacher model of the most recent teacher-labelled row: the lineage the store was last in."""
        row = self.db.execute(
            "SELECT teacher_model FROM samples WHERE task_version=? AND teacher_label IS NOT NULL "
            "ORDER BY id DESC LIMIT 1", (task_version,)).fetchone()
        return row[0] if row else None

    def max_id(self) -> int:
        return self.db.execute("SELECT COALESCE(MAX(id),0) FROM samples").fetchone()[0]

    def events(self, limit: int = 20) -> list[dict]:
        rows = self.db.execute("SELECT ts, kind, detail FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": ts, "kind": k, **json.loads(d)} for ts, k, d in rows][::-1]

    def replay_answers(self, task_version: str) -> dict:
        """text -> TeacherOutput for every teacher-labelled row (feeds ReplayTeacher)."""
        from .teachers import TeacherOutput
        rows = self.db.execute(
            "SELECT text, teacher_label, teacher_probs, teacher_confidence, teacher_input_tokens, teacher_cost_usd "
            "FROM samples WHERE task_version=? AND teacher_label IS NOT NULL", (task_version,)).fetchall()
        return {t: TeacherOutput(l, json.loads(p), c or 0.0, tok or 0, cost or 0.0) for t, l, p, c, tok, cost in rows}

    def close(self) -> None:
        """Commit everything queued, stop the writer, close every connection."""
        with self._wcv:
            self._stopping = True
            self._wcv.notify_all()
            writer = self._writer
        if writer is not None:
            writer.join()
        with self._conns_lock:
            self._closed = True
            conns, self._conns = self._conns, []
        for c in conns:
            c.close()
