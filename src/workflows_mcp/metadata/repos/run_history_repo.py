from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from sqlite3 import Connection
from typing import SupportsInt, cast

DEFAULT_SUMMARY_LIMIT = 512
SUMMARY_TRUNCATION_MARKER = "[...]"


class RunHistoryRepositoryError(RuntimeError):
    """Base class for deterministic run history repository errors."""


class RunNotFoundError(RunHistoryRepositoryError):
    """Raised when mutating a run id that does not exist."""


class RunHistoryTransactionOwnershipError(RunHistoryRepositoryError):
    """Raised when repository writes are invoked under caller-owned transactions."""


class InvalidRunHistoryPaginationError(RunHistoryRepositoryError):
    """Raised when list pagination parameters are out of supported bounds."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    project_id: str | None
    token_id: str | None
    workflow_name: str
    status: str
    cancellable: bool
    result_summary: str | None
    error_summary: str | None
    execution_state_json: str | None
    timeout_seconds: int
    created_at: str
    started_at: str | None
    updated_at: str
    finished_at: str | None
    inputs_json: str | None


@dataclass(frozen=True)
class RunningRunForStaleCheck:
    run_id: str
    timeout_seconds: int
    updated_at: str


def _compact_summary(value: str | None, *, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    if limit <= len(SUMMARY_TRUNCATION_MARKER):
        return SUMMARY_TRUNCATION_MARKER[:limit]
    keep = limit - len(SUMMARY_TRUNCATION_MARKER)
    return f"{value[:keep]}{SUMMARY_TRUNCATION_MARKER}"


class SQLiteRunHistoryRepository:
    def __init__(self, conn: Connection, *, summary_limit: int = DEFAULT_SUMMARY_LIMIT) -> None:
        if summary_limit <= 0:
            raise ValueError("summary_limit must be positive")
        self._conn = conn
        self._summary_limit = summary_limit

    def create_run(
        self,
        *,
        run_id: str,
        workflow_name: str,
        status: str,
        timeout_seconds: int,
        created_at: str,
        started_at: str | None = None,
        updated_at: str,
        inputs_json: str | None = None,
        project_id: str | None = None,
        token_id: str | None = None,
        cancellable: bool = False,
        result_summary: str | None = None,
        error_summary: str | None = None,
        execution_state_json: str | None = None,
    ) -> RunRecord:
        self._assert_write_transaction_ownership()
        compact_result = _compact_summary(result_summary, limit=self._summary_limit)
        compact_error = _compact_summary(error_summary, limit=self._summary_limit)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                INSERT INTO job_runs (
                    run_id,
                    project_id,
                    token_id,
                    workflow_name,
                    status,
                    cancellable,
                    result_summary,
                    error_summary,
                    execution_state_json,
                    timeout_seconds,
                    created_at,
                    started_at,
                    updated_at,
                    inputs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    project_id,
                    token_id,
                    workflow_name,
                    status,
                    1 if cancellable else 0,
                    compact_result,
                    compact_error,
                    execution_state_json,
                    timeout_seconds,
                    created_at,
                    started_at,
                    updated_at,
                    inputs_json,
                ),
            )
            row = self._conn.execute(
                """
                SELECT run_id, project_id, token_id, workflow_name, status,
                       cancellable, result_summary, error_summary,
                       execution_state_json,
                       timeout_seconds, created_at, started_at, updated_at, finished_at, inputs_json
                FROM job_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        if row is None:
            raise RunHistoryRepositoryError(f"run create persisted but row reload failed: {run_id}")
        return self._row_to_record(row)

    def update_run(
        self,
        *,
        run_id: str,
        status: str,
        updated_at: str,
        finished_at: str | None = None,
        started_at: str | None = None,
        cancellable: bool | None = None,
        result_summary: str | None = None,
        error_summary: str | None = None,
        execution_state_json: str | None = None,
        inputs_json: str | None = None,
    ) -> RunRecord:
        self._assert_write_transaction_ownership()
        compact_result = _compact_summary(result_summary, limit=self._summary_limit)
        compact_error = _compact_summary(error_summary, limit=self._summary_limit)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT run_id, project_id, token_id, workflow_name, status,
                       cancellable, result_summary, error_summary,
                       execution_state_json,
                       timeout_seconds, created_at, started_at, updated_at, finished_at, inputs_json
                FROM job_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                self._conn.rollback()
                raise RunNotFoundError(f"run not found: {run_id}")

            current = self._row_to_record(row)
            final_finished_at = finished_at if finished_at is not None else current.finished_at
            final_started_at = started_at if started_at is not None else current.started_at
            final_cancellable = cancellable if cancellable is not None else current.cancellable
            final_result = compact_result if result_summary is not None else current.result_summary
            final_error = compact_error if error_summary is not None else current.error_summary
            final_execution_state = (
                execution_state_json
                if execution_state_json is not None
                else current.execution_state_json
            )
            final_inputs = inputs_json if inputs_json is not None else current.inputs_json

            self._conn.execute(
                """
                UPDATE job_runs
                SET status = ?,
                    started_at = ?,
                    finished_at = ?,
                    cancellable = ?,
                    result_summary = ?,
                    error_summary = ?,
                    execution_state_json = ?,
                    updated_at = ?,
                    inputs_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    final_started_at,
                    final_finished_at,
                    1 if final_cancellable else 0,
                    final_result,
                    final_error,
                    final_execution_state,
                    updated_at,
                    final_inputs,
                    run_id,
                ),
            )
            updated = self._conn.execute(
                """
                SELECT run_id, project_id, token_id, workflow_name, status,
                       cancellable, result_summary, error_summary,
                       execution_state_json,
                       timeout_seconds, created_at, started_at, updated_at, finished_at, inputs_json
                FROM job_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            self._conn.commit()
        except RunNotFoundError:
            raise
        except Exception:
            self._conn.rollback()
            raise

        if updated is None:
            raise RunHistoryRepositoryError(f"run update persisted but row reload failed: {run_id}")
        return self._row_to_record(updated)

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            """
            SELECT run_id, project_id, token_id, workflow_name, status,
                   cancellable, result_summary, error_summary,
                   execution_state_json,
                   timeout_seconds, created_at, started_at, updated_at, finished_at, inputs_json
            FROM job_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_runs(
        self,
        *,
        limit: int,
        offset: int = 0,
        status: str | None = None,
    ) -> list[RunRecord]:
        if limit <= 0:
            raise InvalidRunHistoryPaginationError("limit must be > 0")
        if offset < 0:
            raise InvalidRunHistoryPaginationError("offset must be >= 0")

        if status is None:
            rows = self._conn.execute(
                """
                SELECT run_id, project_id, token_id, workflow_name, status,
                       cancellable, result_summary, error_summary,
                       execution_state_json,
                       timeout_seconds, created_at, started_at, updated_at, finished_at, inputs_json
                FROM job_runs
                ORDER BY created_at ASC, run_id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT run_id, project_id, token_id, workflow_name, status,
                       cancellable, result_summary, error_summary,
                       execution_state_json,
                       timeout_seconds, created_at, started_at, updated_at, finished_at, inputs_json
                FROM job_runs
                WHERE status = ?
                ORDER BY created_at ASC, run_id ASC
                LIMIT ? OFFSET ?
                """,
                (status, limit, offset),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_running_runs_for_stale_check(self) -> list[RunningRunForStaleCheck]:
        rows = self._conn.execute(
            """
            SELECT run_id, timeout_seconds, updated_at
            FROM job_runs
            WHERE status = 'running'
            ORDER BY updated_at ASC, run_id ASC
            """
        ).fetchall()
        return [
            RunningRunForStaleCheck(
                run_id=str(row[0]),
                timeout_seconds=int(row[1]),
                updated_at=str(row[2]),
            )
            for row in rows
        ]

    def _assert_write_transaction_ownership(self) -> None:
        if self._conn.in_transaction:
            raise RunHistoryTransactionOwnershipError(
                "caller-owned transaction detected; run history "
                "repository write methods own transactions"
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row | tuple[object, ...]) -> RunRecord:
        cancellable_raw = cast(str | bytes | bytearray | SupportsInt, row[5])
        timeout_seconds_raw = cast(str | bytes | bytearray | SupportsInt, row[9])
        return RunRecord(
            run_id=str(row[0]),
            project_id=str(row[1]) if row[1] is not None else None,
            token_id=str(row[2]) if row[2] is not None else None,
            workflow_name=str(row[3]),
            status=str(row[4]),
            cancellable=bool(int(cancellable_raw)),
            result_summary=str(row[6]) if row[6] is not None else None,
            error_summary=str(row[7]) if row[7] is not None else None,
            execution_state_json=str(row[8]) if row[8] is not None else None,
            timeout_seconds=int(timeout_seconds_raw),
            created_at=str(row[10]),
            started_at=str(row[11]) if row[11] is not None else None,
            updated_at=str(row[12]),
            finished_at=str(row[13]) if row[13] is not None else None,
            inputs_json=str(row[14]) if row[14] is not None else None,
        )
