"""Append-only sample store on SQLite. One row per request, whoever served it."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

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
    ts: float = field(default_factory=time.time)
    weight: float = 1.0                 # importance weight: how many requests this row stands for
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


class SampleStore:
    def __init__(self, path: Path, calib_fraction: float = 0.2):
        self.path = Path(path)
        self.calib_fraction = calib_fraction
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)

    # ---- writes -------------------------------------------------------------
    def insert(self, recs: Sequence[Record]) -> None:
        rows = []
        for r in recs:
            h = text_hash(r.text)
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
                         r.shadow_version, r.shadow_label, r.shadow_confidence, r.shadow_ood))
        with self.db:
            self.db.executemany(
                "INSERT INTO samples (ts,task_version,text,text_hash,encoder_id,embedding,"
                "served_by,routing_reason,channel,split,latency_ms,weight,"
                "teacher_label,teacher_probs,teacher_confidence,teacher_model,teacher_input_tokens,"
                "teacher_cost_usd,teacher_request_id,teacher_latency_ms,"
                "student_version,student_label,student_probs,student_confidence,ood_score,"
                "shadow_version,shadow_label,shadow_confidence,shadow_ood) "
                "VALUES (" + ",".join("?" * 29) + ")", rows)

    def event(self, kind: str, **detail) -> None:
        with self.db:
            self.db.execute("INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
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
                 split: str, channels: Iterable[str] = TEACHER_CHANNELS):
        """Embeddings X, teacher distributions Y, teacher argmax y, importance weights w for teacher-labelled rows."""
        ch = tuple(channels)
        rows = self.db.execute(
            f"SELECT embedding, teacher_probs, teacher_label, weight FROM samples WHERE task_version=? AND encoder_id=? "
            f"AND split=? AND teacher_label IS NOT NULL AND channel IN ({','.join('?' * len(ch))}) ORDER BY id",
            (task_version, encoder_id, split, *ch)).fetchall()
        return self._matrix(rows, labels, dim)

    def training_set(self, task_version, encoder_id, labels, dim):
        return self.labelled(task_version, encoder_id, labels, dim, "train")

    def calib_set(self, task_version, encoder_id, labels, dim):
        return self.labelled(task_version, encoder_id, labels, dim, "calib", IID_CHANNELS)

    def shadow_records(self, shadow_version: str, task_version: str):
        """IID rows where the shadow candidate ran alongside a teacher answer."""
        ch = IID_CHANNELS
        rows = self.db.execute(
            f"SELECT shadow_label, shadow_confidence, shadow_ood, teacher_label, student_label, student_confidence, "
            f"ood_score FROM samples WHERE shadow_version=? AND task_version=? AND teacher_label IS NOT NULL "
            f"AND channel IN ({','.join('?' * len(ch))})", (shadow_version, task_version, *ch)).fetchall()
        return rows

    def audit_window(self, task_version: str, encoder_id: str, dim: int, limit: int):
        """Embeddings and teacher labels of the most recent audit rows, whichever student scored them."""
        rows = self.db.execute(
            "SELECT embedding, teacher_label FROM samples WHERE task_version=? AND encoder_id=? AND channel='audit' "
            "AND teacher_label IS NOT NULL AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?",
            (task_version, encoder_id, limit)).fetchall()
        if not rows:
            return np.zeros((0, dim), np.float32), np.array([], dtype=object)
        X = np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        return X, np.array([r[1] for r in rows], dtype=object)

    def counts(self, task_version: str) -> dict:
        c = {}
        c["total"] = self.db.execute("SELECT COUNT(*) FROM samples WHERE task_version=?", (task_version,)).fetchone()[0]
        for k, in_ in (("served_by", "served_by"), ("channel", "channel"), ("split", "split")):
            c[k] = dict(self.db.execute(f"SELECT {in_}, COUNT(*) FROM samples WHERE task_version=? GROUP BY {in_}",
                                        (task_version,)).fetchall())
        c["per_class_train"] = dict(self.db.execute(
            "SELECT teacher_label, COUNT(*) FROM samples WHERE task_version=? AND split='train' "
            "AND teacher_label IS NOT NULL GROUP BY teacher_label", (task_version,)).fetchall())
        c["labelled_train"] = sum(c["per_class_train"].values())
        c["labelled_calib"] = self.db.execute(
            "SELECT COUNT(*) FROM samples WHERE task_version=? AND split='calib' AND teacher_label IS NOT NULL",
            (task_version,)).fetchone()[0]
        c["teacher_cost_usd"] = self.db.execute(
            "SELECT COALESCE(SUM(teacher_cost_usd),0) FROM samples WHERE task_version=?", (task_version,)).fetchone()[0]
        c["teacher_calls"] = self.db.execute(
            "SELECT COUNT(*) FROM samples WHERE task_version=? AND teacher_label IS NOT NULL", (task_version,)).fetchone()[0]
        return c

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
        self.db.close()
