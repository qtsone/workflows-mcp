from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from sqlite3 import Connection


class WatcherRepositoryError(RuntimeError):
    """Base class for deterministic watcher repository errors."""


class UnknownProjectError(WatcherRepositoryError):
    """Raised when watcher operations reference a missing project."""


class InvalidWatcherStateError(WatcherRepositoryError):
    """Raised when attempting to persist an unsupported watcher state."""


_ALLOWED_WATCHER_STATES = {"enabled", "paused", "disabled"}


@dataclass(frozen=True)
class WatcherStatusRecord:
    project_id: str
    state: str
    last_event_at: str | None
    updated_at: str


@dataclass(frozen=True)
class DirtyQueueEntry:
    id: int
    project_id: str
    path: str
    event_type: str
    reason: str
    enqueued_at: str
    updated_at: str
    processed_at: str | None


@dataclass(frozen=True)
class WatcherQueueSummary:
    project_id: str
    dirty_count: int
    requires_reconciliation: bool


class SQLiteWatcherRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def set_status(self, *, project_id: str, state: str) -> WatcherStatusRecord:
        self._assert_project_exists(project_id)
        if state not in _ALLOWED_WATCHER_STATES:
            raise InvalidWatcherStateError(
                f"unsupported watcher state: {state}; expected one of enabled|paused|disabled"
            )

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                INSERT INTO watcher_status (project_id, state, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(project_id) DO UPDATE SET
                    state = excluded.state,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (project_id, state),
            )
            row = self._conn.execute(
                """
                SELECT project_id, state, last_event_at, updated_at
                FROM watcher_status
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        if row is None:
            raise WatcherRepositoryError(
                f"watcher status persistence failed for project_id={project_id}"
            )
        return self._status_row_to_record(row)

    def get_status(self, project_id: str) -> WatcherStatusRecord | None:
        row = self._conn.execute(
            """
            SELECT project_id, state, last_event_at, updated_at
            FROM watcher_status
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        if row is None:
            return None
        return self._status_row_to_record(row)

    def list_statuses(self, *, state: str | None = None) -> list[WatcherStatusRecord]:
        params: tuple[object, ...] = ()
        query = (
            "SELECT project_id, state, last_event_at, updated_at "
            "FROM watcher_status "
        )
        if state is not None:
            if state not in _ALLOWED_WATCHER_STATES:
                raise InvalidWatcherStateError(
                    "unsupported watcher state filter: "
                    f"{state}; expected one of enabled|paused|disabled"
                )
            query += "WHERE state = ? "
            params = (state,)
        query += "ORDER BY project_id ASC"

        rows = self._conn.execute(query, params).fetchall()
        return [self._status_row_to_record(row) for row in rows]

    def enqueue_dirty(
        self,
        *,
        project_id: str,
        path: str,
        event_type: str,
        reason: str,
    ) -> DirtyQueueEntry:
        self._assert_project_exists(project_id)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                INSERT INTO watcher_queue (
                    project_id,
                    path,
                    event_type,
                    reason,
                    enqueued_at,
                    updated_at,
                    processed_at
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
                ON CONFLICT(project_id, path, event_type)
                WHERE processed_at IS NULL
                DO UPDATE SET
                    reason = excluded.reason,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (project_id, path, event_type, reason),
            )
            row = self._conn.execute(
                """
                SELECT
                    id,
                    project_id,
                    path,
                    event_type,
                    reason,
                    enqueued_at,
                    updated_at,
                    processed_at
                FROM watcher_queue
                WHERE project_id = ?
                  AND path = ?
                  AND event_type = ?
                  AND processed_at IS NULL
                LIMIT 1
                """,
                (project_id, path, event_type),
            ).fetchone()
            self._conn.execute(
                """
                UPDATE watcher_status
                SET last_event_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE project_id = ?
                """,
                (project_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        if row is None:
            raise WatcherRepositoryError(
                "dirty queue persistence failed for "
                f"project_id={project_id}, path={path}, event_type={event_type}"
            )
        return self._queue_row_to_record(row)

    def list_active_dirty(self, *, project_id: str) -> list[DirtyQueueEntry]:
        rows = self._conn.execute(
            """
            SELECT id, project_id, path, event_type, reason, enqueued_at, updated_at, processed_at
            FROM watcher_queue
            WHERE project_id = ?
              AND processed_at IS NULL
            ORDER BY enqueued_at ASC, id ASC
            """,
            (project_id,),
        ).fetchall()
        return [self._queue_row_to_record(row) for row in rows]

    def list_dirty_history(self, *, project_id: str, limit: int = 25) -> list[DirtyQueueEntry]:
        normalized_limit = max(1, min(limit, 100))
        rows = self._conn.execute(
            """
            SELECT id, project_id, path, event_type, reason, enqueued_at, updated_at, processed_at
            FROM watcher_queue
            WHERE project_id = ?
            ORDER BY
                CASE WHEN processed_at IS NULL THEN 0 ELSE 1 END ASC,
                COALESCE(processed_at, updated_at, enqueued_at) DESC,
                id DESC
            LIMIT ?
            """,
            (project_id, normalized_limit),
        ).fetchall()
        return [self._queue_row_to_record(row) for row in rows]

    def count_active_dirty(self, *, project_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(1)
            FROM watcher_queue
            WHERE project_id = ?
              AND processed_at IS NULL
            """,
            (project_id,),
        ).fetchone()
        raw_count = 0 if row is None else row[0]
        return int(raw_count)

    def mark_active_dirty_processed(self, *, project_id: str) -> int:
        self._assert_project_exists(project_id)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                """
                UPDATE watcher_queue
                SET processed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE project_id = ?
                  AND processed_at IS NULL
                """,
                (project_id,),
            )
            self._conn.execute(
                """
                UPDATE watcher_status
                SET updated_at = CURRENT_TIMESTAMP
                WHERE project_id = ?
                """,
                (project_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        return cursor.rowcount

    def requires_reconciliation(self, *, project_id: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM watcher_queue
            WHERE project_id = ?
              AND processed_at IS NULL
              AND (
                    reason LIKE 'reconciliation%'
                 OR reason LIKE 'rebuild%'
              )
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()
        return row is not None

    def list_queue_summaries(self) -> list[WatcherQueueSummary]:
        rows = self._conn.execute(
            """
            SELECT
                project_id,
                COUNT(1) AS dirty_count,
                MAX(
                    CASE
                        WHEN reason LIKE 'reconciliation%' OR reason LIKE 'rebuild%'
                            THEN 1
                        ELSE 0
                    END
                ) AS requires_reconciliation
            FROM watcher_queue
            WHERE processed_at IS NULL
            GROUP BY project_id
            ORDER BY project_id ASC
            """
        ).fetchall()
        summaries: list[WatcherQueueSummary] = []
        for row in rows:
            raw_dirty_count = row[1]
            raw_requires = row[2]
            summaries.append(
                WatcherQueueSummary(
                    project_id=str(row[0]),
                    dirty_count=int(raw_dirty_count),
                    requires_reconciliation=bool(int(raw_requires)),
                )
            )
        return summaries

    def project_exists(self, project_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        return row is not None

    def _assert_project_exists(self, project_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            raise UnknownProjectError(f"project not found: {project_id}")

    @staticmethod
    def _status_row_to_record(
        row: sqlite3.Row | tuple[object, ...],
    ) -> WatcherStatusRecord:
        return WatcherStatusRecord(
            project_id=str(row[0]),
            state=str(row[1]),
            last_event_at=str(row[2]) if row[2] is not None else None,
            updated_at=str(row[3]),
        )

    @staticmethod
    def _queue_row_to_record(row: sqlite3.Row | tuple[object, ...]) -> DirtyQueueEntry:
        raw_id = row[0]
        if not isinstance(raw_id, int):
            raise WatcherRepositoryError("invalid watcher queue row id type")
        return DirtyQueueEntry(
            id=raw_id,
            project_id=str(row[1]),
            path=str(row[2]),
            event_type=str(row[3]),
            reason=str(row[4]),
            enqueued_at=str(row[5]),
            updated_at=str(row[6]),
            processed_at=str(row[7]) if row[7] is not None else None,
        )
