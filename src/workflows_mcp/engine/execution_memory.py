"""Ephemeral per-execution memory backed by SQLite.

Provides key-value storage and conversation history that survives
across blocks within a single workflow execution, including sub-workflows.
The SQLite file lives in the worker pod workspace and is discarded
on pod termination — no persistent state, no network overhead.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS execution_context (
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    scope   TEXT NOT NULL DEFAULT 'global',
    PRIMARY KEY (key, scope)
);

CREATE TABLE IF NOT EXISTS execution_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True, slots=True)
class Turn:
    """A single conversation turn."""

    id: int
    block_id: str
    role: str
    content: str
    created_at: str


class ExecutionMemory:
    """Ephemeral memory for the current workflow execution.

    Thread-safe via WAL mode and ``check_same_thread=False``.
    All public methods are async, wrapping synchronous sqlite3
    calls with ``asyncio.to_thread``.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the database and create tables."""
        self._conn = await asyncio.to_thread(self._open)
        await asyncio.to_thread(self._conn.executescript, _SCHEMA_SQL)
        logger.debug("Execution memory initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    # ------------------------------------------------------------------
    # Key-Value Context
    # ------------------------------------------------------------------

    async def get(self, key: str, scope: str = "global") -> str | None:
        """Retrieve a value by key and scope."""
        assert self._conn is not None, "ExecutionMemory not initialized"
        row = await asyncio.to_thread(
            self._conn.execute,
            "SELECT value FROM execution_context WHERE key = ? AND scope = ?",
            (key, scope),
        )
        result = row.fetchone()
        return result[0] if result else None

    async def set(self, key: str, value: str, scope: str = "global") -> None:
        """Store a value by key and scope (upsert)."""
        assert self._conn is not None, "ExecutionMemory not initialized"
        await asyncio.to_thread(
            self._conn.execute,
            "INSERT INTO execution_context (key, value, scope) VALUES (?, ?, ?) "
            "ON CONFLICT(key, scope) DO UPDATE SET value = excluded.value",
            (key, value, scope),
        )
        await asyncio.to_thread(self._conn.commit)

    # ------------------------------------------------------------------
    # Conversation Turns
    # ------------------------------------------------------------------

    async def add_turn(self, block_id: str, role: str, content: str) -> None:
        """Append a conversation turn."""
        assert self._conn is not None, "ExecutionMemory not initialized"
        await asyncio.to_thread(
            self._conn.execute,
            "INSERT INTO execution_turns (block_id, role, content) VALUES (?, ?, ?)",
            (block_id, role, content),
        )
        await asyncio.to_thread(self._conn.commit)

    async def get_turns(self, block_id: str | None = None) -> list[Turn]:
        """Retrieve conversation turns, optionally filtered by block_id."""
        assert self._conn is not None, "ExecutionMemory not initialized"
        if block_id is not None:
            cursor = await asyncio.to_thread(
                self._conn.execute,
                "SELECT id, block_id, role, content, created_at "
                "FROM execution_turns WHERE block_id = ? ORDER BY id",
                (block_id,),
            )
        else:
            cursor = await asyncio.to_thread(
                self._conn.execute,
                "SELECT id, block_id, role, content, created_at FROM execution_turns ORDER BY id",
            )
        rows = cursor.fetchall()
        return [
            Turn(id=r[0], block_id=r[1], role=r[2], content=r[3], created_at=r[4]) for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn


__all__ = ["ExecutionMemory", "Turn"]
