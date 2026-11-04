"""Persistent external memory backed by SQLite."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("cognitive.memory")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    id: str
    mem_type: str
    content: str
    strength: float
    last_accessed: str
    access_count: int
    tags: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class PersistentMemory:
    """Thread-safe SQLite external memory store."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory (
                        id TEXT PRIMARY KEY,
                        mem_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        strength REAL NOT NULL DEFAULT 0.5,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        tags TEXT NOT NULL DEFAULT '[]'
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_strength ON memory(strength DESC)"
                )
                conn.commit()
        log.info("Persistent memory ready at %s", self.db_path)

    def seed_if_empty(self, seeds: List[tuple]) -> None:
        with self._lock:
            with self._conn() as conn:
                n = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
                if n > 0:
                    return
                for mtype, content, strength, tags in seeds:
                    mid = str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO memory (id, mem_type, content, strength, last_accessed, access_count, tags) VALUES (?,?,?,?,?,?,?)",
                        (mid, mtype, content, strength, _now(), 1, json.dumps(tags)),
                    )
                conn.commit()
                log.info("Seeded %d memory records", len(seeds))

    def write(self, mem_type: str, content: str, strength: float, tags: List[str]) -> MemoryRecord:
        mid = str(uuid.uuid4())
        rec = MemoryRecord(
            id=mid, mem_type=mem_type, content=content,
            strength=min(1.0, max(0.0, strength)),
            last_accessed=_now(), access_count=1, tags=tags,
        )
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO memory (id, mem_type, content, strength, last_accessed, access_count, tags) VALUES (?,?,?,?,?,?,?)",
                    (rec.id, rec.mem_type, rec.content, rec.strength, rec.last_accessed, rec.access_count, json.dumps(rec.tags)),
                )
                conn.commit()
        return rec

    def list(self, limit: int = 50) -> List[MemoryRecord]:
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM memory ORDER BY strength DESC, last_accessed DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            with self._conn() as conn:
                return int(conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0])

    def recall(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        q = (query or "").lower().strip()
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute("SELECT * FROM memory").fetchall()
                scored: List[tuple] = []
                for r in rows:
                    content = (r["content"] or "").lower()
                    tags = json.loads(r["tags"] or "[]")
                    score = float(r["strength"])
                    if q and any(tok in content for tok in q.split() if tok):
                        score += 0.35
                    if q and any(tag in q for tag in tags):
                        score += 0.2
                    scored.append((score, r))
                scored.sort(key=lambda x: -x[0])
                results = []
                for score, r in scored[:limit]:
                    conn.execute(
                        "UPDATE memory SET access_count = access_count + 1, strength = MIN(1.0, strength + 0.03), last_accessed = ? WHERE id = ?",
                        (_now(), r["id"]),
                    )
                    results.append(self._row(r, bump=True))
                conn.commit()
        return results

    def _row(self, r: sqlite3.Row, bump: bool = False) -> MemoryRecord:
        return MemoryRecord(
            id=r["id"],
            mem_type=r["mem_type"],
            content=r["content"],
            strength=min(1.0, float(r["strength"]) + (0.03 if bump else 0.0)),
            last_accessed=r["last_accessed"],
            access_count=int(r["access_count"]) + (1 if bump else 0),
            tags=json.loads(r["tags"] or "[]"),
        )
