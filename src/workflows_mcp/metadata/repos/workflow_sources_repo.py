from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection


class WorkflowSourceRepositoryError(RuntimeError):
    """Base class for deterministic workflow source repository errors."""


class DuplicateWorkflowSourceError(WorkflowSourceRepositoryError):
    """Raised when a workflow source with the same normalized path already exists globally."""


class WorkflowSourceNotFoundError(WorkflowSourceRepositoryError):
    """Raised when an operation targets a missing workflow source id."""


class InvalidWorkflowSourcePathError(WorkflowSourceRepositoryError):
    """Raised when a source path does not resolve to an existing directory."""


@dataclass(frozen=True)
class WorkflowSourceCreate:
    source_path: str
    checksum: str | None = None


@dataclass(frozen=True)
class WorkflowSourceRecord:
    source_id: str
    source_path: str
    checksum: str | None
    discovered_at: str
    last_loaded_at: str | None
    status: str | None
    error_message: str | None


def _normalize_source_path(raw: str) -> str:
    return str(Path(raw).expanduser().resolve(strict=False))


class SQLiteWorkflowSourcesRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def create(self, data: WorkflowSourceCreate) -> WorkflowSourceRecord:
        normalized_path = _normalize_source_path(data.source_path)
        normalized_path_obj = Path(normalized_path)
        if not normalized_path_obj.exists() or not normalized_path_obj.is_dir():
            raise InvalidWorkflowSourcePathError(
                f"workflow source path must exist and be a directory: {normalized_path}"
            )

        source_id = str(uuid.uuid4())

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._assert_not_duplicate_path(normalized_source_path=normalized_path)
            self._conn.execute(
                """
                INSERT INTO workflow_sources (
                    source_id,
                    source_path,
                    checksum
                ) VALUES (?, ?, ?)
                """,
                (source_id, normalized_path, data.checksum),
            )
            record = self._get_for_update(source_id)
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            if self._is_unique_source_path_violation(exc):
                raise DuplicateWorkflowSourceError("workflow source path already exists") from exc
            raise
        except Exception:
            self._conn.rollback()
            raise

        if record is None:
            raise WorkflowSourceRepositoryError(
                f"workflow source create persisted but row reload failed for id={source_id}"
            )
        return record

    def get_by_path(self, source_path: str) -> WorkflowSourceRecord | None:
        """Return the workflow source record for a normalized path."""
        normalized = _normalize_source_path(source_path)
        row = self._conn.execute(
            """
            SELECT ws.source_id, ws.source_path, ws.checksum,
                   ws.discovered_at, wrs.last_loaded_at, wrs.status, wrs.error_message
            FROM workflow_sources ws
            LEFT JOIN workflow_reload_state wrs ON wrs.source_id = ws.source_id
            WHERE ws.source_path = ?
            """,
            (normalized,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list(self) -> list[WorkflowSourceRecord]:
        rows = self._conn.execute(
            """
            SELECT ws.source_id, ws.source_path, ws.checksum,
                   ws.discovered_at, wrs.last_loaded_at, wrs.status, wrs.error_message
            FROM workflow_sources ws
            LEFT JOIN workflow_reload_state wrs ON wrs.source_id = ws.source_id
            ORDER BY ws.discovered_at ASC, ws.source_id ASC
            """
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, source_id: str) -> WorkflowSourceRecord:
        record = self._get_for_update(source_id)
        if record is None:
            raise WorkflowSourceNotFoundError(f"workflow source not found: {source_id}")
        return record

    def delete(self, source_id: str) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT 1 FROM workflow_sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if row is None:
                raise WorkflowSourceNotFoundError(f"workflow source not found: {source_id}")
            self._conn.execute(
                "DELETE FROM workflow_sources WHERE source_id = ?",
                (source_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def update_reload_state(
        self,
        source_id: str,
        *,
        status: str,
        error_message: str | None = None,
    ) -> WorkflowSourceRecord:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._assert_source_exists(source_id)
            self._conn.execute(
                """
                INSERT INTO workflow_reload_state (
                    source_id,
                    last_loaded_at,
                    status,
                    error_message
                ) VALUES (?, CURRENT_TIMESTAMP, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    last_loaded_at = CURRENT_TIMESTAMP,
                    status = excluded.status,
                    error_message = excluded.error_message
                """,
                (source_id, status, error_message),
            )
            record = self._get_for_update(source_id)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        if record is None:
            raise WorkflowSourceNotFoundError(f"workflow source not found: {source_id}")
        return record

    def _assert_source_exists(self, source_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM workflow_sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            raise WorkflowSourceNotFoundError(f"workflow source not found: {source_id}")

    def _assert_not_duplicate_path(self, *, normalized_source_path: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM workflow_sources WHERE source_path = ?",
            (normalized_source_path,),
        ).fetchone()
        if row is not None:
            raise DuplicateWorkflowSourceError("workflow source path already exists")

    def _get_for_update(self, source_id: str) -> WorkflowSourceRecord | None:
        row = self._conn.execute(
            """
            SELECT ws.source_id, ws.source_path, ws.checksum,
                   ws.discovered_at, wrs.last_loaded_at, wrs.status, wrs.error_message
            FROM workflow_sources ws
            LEFT JOIN workflow_reload_state wrs ON wrs.source_id = ws.source_id
            WHERE ws.source_id = ?
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    @staticmethod
    def _is_unique_source_path_violation(exc: sqlite3.IntegrityError) -> bool:
        message = str(exc).lower()
        return "workflow_sources.source_path" in message or "unique" in message

    @staticmethod
    def _row_to_record(row: tuple[object, ...]) -> WorkflowSourceRecord:
        return WorkflowSourceRecord(
            source_id=str(row[0]),
            source_path=str(row[1]),
            checksum=str(row[2]) if row[2] is not None else None,
            discovered_at=str(row[3]),
            last_loaded_at=str(row[4]) if row[4] is not None else None,
            status=str(row[5]) if row[5] is not None else None,
            error_message=str(row[6]) if row[6] is not None else None,
        )
