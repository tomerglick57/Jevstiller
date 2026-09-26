"""Append-only sample store on SQLite. One row per request, whoever served it."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
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
LOG_EVERY_S = 60.0                                  # write failures: one log line per store per minute

_dropped = 0                                        # records every store in this process failed to write
_dropped_lock = threading.Lock()


def dropped_records() -> int:
    """Records the write-behind stores of this process could not commit (disk full, I/O errors), since start."""
    return _dropped
TEACHER_CHANNELS = IID_CHANNELS + ("deferred",)


def text_hash(text: str, key: bytes | None = None) -> str:
    """64-bit id of a text: keys replays and dedup. With `key` (the deployment salt)
    it is an HMAC, so a store kept with store_text=False holds no plain hash of the text."""
    if key:
        return hmac.new(key, text.encode(), hashlib.sha256).hexdigest()[:16]
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def split_for(draw: float, calib_fraction: float) -> str:
    """The split of one request, from a uniform draw in [0, 1). Per request, not per text: the guarantee is over
    requests, and splitting by text made a heavily repeated input dominate the calibration set, which broke the
    bound on the real traffic mix (security audit run 3)."""
    return "calib" if draw < calib_fraction else "train"


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
CREATE TABLE IF NOT EXISTS deleted_rows (
  task_version TEXT NOT NULL, served_by TEXT NOT NULL, channel TEXT NOT NULL, split TEXT NOT NULL,
  n INTEGER NOT NULL, teacher_calls INTEGER NOT NULL, teacher_cost_usd REAL NOT NULL,
  PRIMARY KEY (task_version, served_by, channel, split)
);
"""

SCHEMA_VERSION = 2                     # PRAGMA user_version; 0 = a 0.1.0 store (or a new file); 2 adds deleted_rows

# columns added after 0.1.0: (name, type). Existing stores get them via ALTER TABLE on open.
_ADDED_COLUMNS = (("state_type", "TEXT"),)
_INDEXES_AFTER_MIGRATION = """
CREATE INDEX IF NOT EXISTS ix_samples_lineage ON samples(task_version, teacher_model, split);
"""


def _lineage(teacher_model: str | None, since_id: int = 0) -> tuple[str, tuple]:
    """SQL filter for one teacher lineage (None = every teacher), from row `since_id` on (0 = all rows)."""
    sql, args = (" AND teacher_model=?", (teacher_model,)) if teacher_model is not None else ("", ())
    return (sql + " AND id>?", (*args, since_id)) if since_id else (sql, args)


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
    def counts(self, task_version: str, teacher_model: str | None = None, since_id: int = 0) -> dict: ...
    def training_set(self, task_version, encoder_id, labels, dim, teacher_model=None, since_id=0, limit=0): ...
    def calib_set(self, task_version, encoder_id, labels, dim, teacher_model=None, since_id=0, limit=0): ...
    def shadow_records(self, shadow_version: str, task_version: str, teacher_model: str | None = None,
                       since_id: int = 0): ...
    def audit_served(self, task_version: str, version: str, limit: int, teacher_model: str | None = None,
                     since_id: int = 0): ...
    def redact_text(self, older_than_ts: float) -> int: ...
    def latest_teacher_model(self, task_version: str) -> str | None: ...
    def max_id(self) -> int: ...
    def prune(self, keep_train: int, keep_calib: int, keep_unanswered: int, keep_after_id: int | None = None,
              max_rows: int = 20_000) -> tuple[int, bool]: ...
    def answered(self, task_version: str, teacher_model: str, after_id: int, upto_id: int | None = None) -> int: ...
    def events(self, limit: int = 20) -> list[dict]: ...
    def last_event(self, kinds: Sequence[str]) -> dict | None: ...
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
                 read_only: bool = False, split_seed: str | None = None):
        self.path = Path(path)
        self.read_only = read_only
        self.calib_fraction = calib_fraction
        # the per-request calibration draw: seeded from `split_seed` (the loop passes its config seed and the task
        # version, so a replay splits the same way on any machine), else from the path
        seed = split_seed if split_seed is not None else str(self.path)
        self._split_rng = random.Random(hashlib.sha256(seed.encode()).digest())
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
        self._last_error_log = -LOG_EVERY_S
        self._writer: threading.Thread | None = None
        if not read_only:
            conn = self._conn()
            if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:   # new or older store
                if not conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]:
                    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")  # a new file: `prune` can hand space back
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
                    conn.execute("PRAGMA secure_delete=ON")    # overwritten text is zeroed, not left in pages
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
            except Exception as e:
                failed = True
                now = time.monotonic()
                if now - self._last_error_log >= LOG_EVERY_S:          # a full disk fails every batch
                    self._last_error_log = now
                    log.error("sample store %s: dropped %d records (%d so far): %r", self.path, len(batch),
                              self.write_errors + len(batch), e)
            if failed:
                global _dropped
                with _dropped_lock:
                    _dropped += len(batch)
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
            split = split_for(self._split_rng.random(), self.calib_fraction) if r.channel in IID_CHANNELS else "train"
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
                 split: str, channels: Iterable[str] = TEACHER_CHANNELS, teacher_model: str | None = None,
                 since_id: int = 0, limit: int = 0):
        """Embeddings X, teacher distributions Y, teacher argmax y, importance weights w for teacher-labelled rows
        (of one teacher lineage when `teacher_model` is given, after row `since_id`), oldest first. With `limit`,
        only the most recent `limit` rows."""
        ch = tuple(channels)
        lin, lin_args = _lineage(teacher_model, since_id)
        rows = self.db.execute(
            f"SELECT embedding, teacher_probs, teacher_label, weight FROM samples "
            f"WHERE task_version=? AND encoder_id=?{lin} "
            f"AND split=? AND teacher_label IS NOT NULL AND channel IN ({','.join('?' * len(ch))}) "
            + ("ORDER BY id DESC LIMIT ?" if limit else "ORDER BY id"),
            (task_version, encoder_id, *lin_args, split, *ch, *((limit,) if limit else ()))).fetchall()
        return self._matrix(rows[::-1] if limit else rows, labels, dim)

    def training_set(self, task_version, encoder_id, labels, dim, teacher_model=None, since_id=0, limit=0):
        return self.labelled(task_version, encoder_id, labels, dim, "train", teacher_model=teacher_model,
                             since_id=since_id, limit=limit)

    def calib_set(self, task_version, encoder_id, labels, dim, teacher_model=None, since_id=0, limit=0):
        return self.labelled(task_version, encoder_id, labels, dim, "calib", IID_CHANNELS, teacher_model, since_id,
                             limit)

    def shadow_records(self, shadow_version: str, task_version: str, teacher_model: str | None = None,
                       since_id: int = 0):
        """IID rows where the shadow candidate ran alongside a teacher answer."""
        ch = IID_CHANNELS
        lin, lin_args = _lineage(teacher_model, since_id)
        rows = self.db.execute(
            f"SELECT shadow_label, shadow_confidence, shadow_ood, teacher_label, student_label, student_confidence, "
            f"ood_score FROM samples WHERE shadow_version=? AND task_version=?{lin} AND teacher_label IS NOT NULL "
            f"AND channel IN ({','.join('?' * len(ch))})", (shadow_version, task_version, *lin_args, *ch)).fetchall()
        return rows

    def audit_served(self, task_version: str, version: str, limit: int, teacher_model: str | None = None,
                     since_id: int = 0):
        """The most recent audit rows served while `version` was in production: what that student said at the
        time (label, confidence, OOD score) and the teacher's label. Scoring these rows as served, not re-scored by
        a model that may since have trained on them, keeps the audit an out-of-sample estimate."""
        lin, lin_args = _lineage(teacher_model, since_id)
        rows = self.db.execute(
            f"SELECT student_label, student_confidence, ood_score, teacher_label FROM samples "
            f"WHERE task_version=? AND student_version=?{lin} AND channel='audit' AND teacher_label IS NOT NULL "
            f"AND student_label IS NOT NULL ORDER BY id DESC LIMIT ?",
            (task_version, version, *lin_args, limit)).fetchall()
        return (np.array([r[0] for r in rows], dtype=object),
                np.array([r[1] if r[1] is not None else 0.0 for r in rows], np.float64),
                np.array([r[2] if r[2] is not None else np.inf for r in rows], np.float64),
                np.array([r[3] for r in rows], dtype=object))

    def counts(self, task_version: str, teacher_model: str | None = None, since_id: int = 0) -> dict:
        """Totals over every teacher, deleted rows included (`prune`); the labelled counts (what training can use)
        over one lineage if given, of the rows still stored."""
        lin, lin_args = _lineage(teacher_model, since_id)
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
        for sb, ch, sp, n, calls, cost in self.db.execute(
                "SELECT served_by, channel, split, n, teacher_calls, teacher_cost_usd FROM deleted_rows "
                "WHERE task_version=?", (task_version,)).fetchall():
            c["total"] += n
            for k, v in (("served_by", sb), ("channel", ch), ("split", sp)):
                v = v or None                                # stored as '' for NULL (a primary key column)
                c[k][v] = c[k].get(v, 0) + n
            c["teacher_calls"] += calls
            c["teacher_cost_usd"] += cost
        return c

    def prune(self, keep_train: int, keep_calib: int, keep_unanswered: int, keep_after_id: int | None = None,
              max_rows: int = 20_000, batch: int = 5_000) -> tuple[int, bool]:
        """Delete the rows no reader can use any more, oldest first. For each task version and teacher lineage,
        keep the newest `keep_train` training rows and the newest `keep_calib` calibration rows. Of the rows
        without a teacher answer, keep the newest `keep_unanswered`. 0 keeps every row of that kind. Rows after
        `keep_after_id` are all kept (e.g. those of a shadow not judged yet).

        Training and calibration take at most the newest `max_train_samples` / `max_calib_samples` rows of one
        lineage and split, so with those as `keep_train` / `keep_calib` nothing they read is deleted (the audit
        window and a shadow's rows: DESIGN §7.2). Deleted rows still count in `counts()` (table `deleted_rows`);
        secure_delete zeroes their text. Deletes at most `max_rows`,
        in transactions of `batch`, so the writer is never held up for long and a backlog goes over several calls.
        Returns (rows deleted, whether more may remain)."""
        if self.read_only:
            raise sqlite3.ProgrammingError("the sample store is read-only")
        self.flush()
        conn = self._conn()
        groups = conn.execute("SELECT DISTINCT task_version, teacher_model, split FROM samples").fetchall()
        where = "task_version=? AND teacher_model IS ? AND split=?"
        deleted = 0
        for tv, tm, split in groups:
            keep = keep_unanswered if tm is None else keep_calib if split == "calib" else keep_train
            row = conn.execute(f"SELECT id FROM samples WHERE {where} ORDER BY id DESC LIMIT 1 OFFSET ?",
                               (tv, tm, split, keep - 1)).fetchone() if keep else None
            if row is None:
                continue
            oldest_kept = row[0] if keep_after_id is None else min(row[0], keep_after_id + 1)
            # a teacher answer always comes with its model (Jevstiller.complete); a row inserted without one is kept
            only = " AND teacher_label IS NULL" if tm is None else ""
            while True:
                n = min(batch, max_rows - deleted)
                if n <= 0:
                    self._release_space(conn, finished=False)
                    return deleted, True
                with self._write_lock, conn:
                    ids = conn.execute(f"SELECT id FROM samples WHERE {where} AND id<?{only} ORDER BY id LIMIT ?",
                                       (tv, tm, split, oldest_kept, n)).fetchall()
                    if not ids:
                        break
                    rng = (tv, tm, split, ids[0][0], ids[-1][0])
                    conn.execute(
                        "INSERT INTO deleted_rows SELECT task_version, COALESCE(served_by, ''), COALESCE(channel, ''), "
                        "COALESCE(split, ''), COUNT(*), COUNT(teacher_label), COALESCE(SUM(teacher_cost_usd), 0) "
                        f"FROM samples WHERE {where} AND id BETWEEN ? AND ?{only} GROUP BY 1, 2, 3, 4 "
                        "ON CONFLICT DO UPDATE SET n = n + excluded.n, teacher_calls = teacher_calls + "
                        "excluded.teacher_calls, teacher_cost_usd = teacher_cost_usd + excluded.teacher_cost_usd", rng)
                    conn.execute(f"DELETE FROM samples WHERE {where} AND id BETWEEN ? AND ?{only}", rng)
                deleted += len(ids)
                if len(ids) < n:
                    break
        if deleted:
            self._release_space(conn, finished=True)
        return deleted, False

    def _release_space(self, conn: sqlite3.Connection, finished: bool) -> None:
        """Hand the pages of deleted rows back to the file system. A store created with auto_vacuum does that
        after every prune. An older one (0.3.3 and before) is rebuilt once, after the prune that clears its
        backlog, if more than half of the file is then free: VACUUM copies only the rows left and turns
        auto_vacuum on (the 24-hour soak's biggest store: 8.6 GB down to 294 MB in 8–15 s)."""
        with self._write_lock:
            mode = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
            if mode == 2:                                                 # INCREMENTAL
                conn.execute("PRAGMA incremental_vacuum").fetchall()     # stepped to the end
            elif mode == 0 and finished:
                free = conn.execute("PRAGMA freelist_count").fetchone()[0]
                if free > conn.execute("PRAGMA page_count").fetchone()[0] // 2:
                    t0 = time.monotonic()
                    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                    conn.execute("VACUUM")
                    log.info("sample store %s: rebuilt without %d free pages in %.1f s", self.path, free,
                             time.monotonic() - t0)

    def redact_text(self, older_than_ts: float) -> int:
        """Blank the stored text of rows older than `older_than_ts` (hash, embedding and answers stay, so
        they still train and calibrate), and make sure the old text is gone from the files: secure_delete
        zeroes freed page content, and a TRUNCATE checkpoint empties the WAL (retried while readers hold
        it). Returns the number of rows changed."""
        self.flush()
        conn = self._conn()
        with self._write_lock, conn:
            cur = conn.execute("UPDATE samples SET text='' WHERE ts < ? AND text IS NOT NULL AND text != ''",
                               (older_than_ts,))
        if cur.rowcount:
            with self._write_lock:
                for _ in range(20):
                    busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    if not busy:
                        break
                    time.sleep(0.05)
                else:
                    log.warning("sample store %s: WAL still in use; redacted text leaves the WAL at the next "
                                "checkpoint", self.path)
        return cur.rowcount

    def latest_teacher_model(self, task_version: str) -> str | None:
        """The teacher model of the most recent teacher-labelled row: the lineage the store was last in."""
        row = self.db.execute(
            "SELECT teacher_model FROM samples WHERE task_version=? AND teacher_label IS NOT NULL "
            "ORDER BY id DESC LIMIT 1", (task_version,)).fetchone()
        return row[0] if row else None

    def max_id(self) -> int:
        return self.db.execute("SELECT COALESCE(MAX(id),0) FROM samples").fetchone()[0]

    def answered(self, task_version: str, teacher_model: str, after_id: int, upto_id: int | None = None) -> int:
        """Rows with an answer from `teacher_model` after row `after_id` (up to `upto_id`): what a retrain has
        new to learn from. A row has a teacher model only if the teacher answered it. The split filter lets
        the lineage index seek straight to `after_id`, so the cost is the number of answers, not of rows."""
        sql = ("SELECT COUNT(*) FROM samples WHERE task_version=? AND teacher_model=? AND split IN ('train', 'calib') "
               "AND id>?")
        args: tuple = (task_version, teacher_model, after_id)
        if upto_id is not None:
            sql, args = sql + " AND id<=?", (*args, upto_id)
        return self.db.execute(sql, args).fetchone()[0]

    def events(self, limit: int = 20) -> list[dict]:
        rows = self.db.execute("SELECT ts, kind, detail FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": ts, "kind": k, **json.loads(d)} for ts, k, d in rows][::-1]

    def last_event(self, kinds: Sequence[str]) -> dict | None:
        """The most recent event of any of these kinds."""
        row = self.db.execute(f"SELECT ts, kind, detail FROM events WHERE kind IN ({','.join('?' * len(kinds))}) "
                              "ORDER BY id DESC LIMIT 1", tuple(kinds)).fetchone()
        return {"ts": row[0], "kind": row[1], **json.loads(row[2])} if row else None

    def replay_answers(self, task_version: str) -> dict:
        """text -> TeacherOutput for every teacher-labelled row (feeds ReplayTeacher)."""
        from .teachers import TeacherOutput
        rows = self.db.execute(
            "SELECT text, teacher_label, teacher_probs, teacher_confidence, teacher_input_tokens, teacher_cost_usd "
            "FROM samples WHERE task_version=? AND teacher_label IS NOT NULL", (task_version,)).fetchall()
        return {t: TeacherOutput(l, json.loads(p), c or 0.0, tok or 0, cost or 0.0) for t, l, p, c, tok, cost in rows}

    def close(self) -> None:
        """Commit everything queued, stop the writer, close every connection. Waits for writes in progress
        (e.g. text retention): closing a connection another thread is using crashes the process. Readers must
        not overlap a close (the task manager holds an engine while anything uses it)."""
        with self._wcv:
            self._stopping = True
            self._wcv.notify_all()
            writer = self._writer
        if writer is not None:
            writer.join()
        with self._write_lock, self._conns_lock:
            self._closed = True
            conns, self._conns = self._conns, []
        for c in conns:
            c.close()
