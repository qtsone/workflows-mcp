"""Persistent job storage using SQLite metadata + run history summaries."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.run_history_repo import SQLiteRunHistoryRepository

from .state_config import StateConfig

if TYPE_CHECKING:
    from .job_queue import Job, WorkflowStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JobStore:
    """Persistent storage for job queue with compact SQLite-only persistence."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize job store with configurable database path."""
        self._db_path = Path(db_path) if db_path is not None else StateConfig.get_db_path()

    async def init(self) -> None:
        """Initialize database schema and load existing stats.

        Must be called before using the store.
        Creates tables and indexes if they don't exist.
        """
        # Initialize database schema
        await self._run_in_executor(self._init_db)

        logger.info(f"JobStore initialized: db={self._db_path}")

    def _init_db(self) -> None:
        """Initialize SQLite database with schema (runs in thread pool).

        Creates stats table and metadata schema.
        """
        conn = connect_metadata_db(self._db_path)

        migrate_metadata_db(conn)

        # Enable WAL mode for concurrent access (multiple MCP instances)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster, still safe with WAL

        # Create stats table for persistent statistics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
        """)

        # Initialize stats if not present
        conn.execute("INSERT OR IGNORE INTO stats VALUES ('total_jobs', 0)")
        conn.execute("INSERT OR IGNORE INTO stats VALUES ('completed_jobs', 0)")
        conn.execute("INSERT OR IGNORE INTO stats VALUES ('failed_jobs', 0)")
        conn.execute("INSERT OR IGNORE INTO stats VALUES ('cancelled_jobs', 0)")

        conn.commit()
        conn.close()

        logger.debug("Database schema initialized with WAL mode")

    async def save_job(self, job: Job) -> None:
        """Save job to SQLite metadata and compact run-history summaries."""
        # Update job's updated_at timestamp
        job.updated_at = datetime.now()

        await self._save_job_metadata(job)

    async def _save_job_metadata(self, job: Job) -> None:
        """Save job metadata to SQLite database.

        Args:
            job: Job instance to save metadata for
        """

        def _write() -> None:
            result_summary = self._build_result_summary(job)
            execution_state_json = self._build_execution_state_json(job)
            cancellable = job.status.value in {"queued", "running"}
            inputs_json = json.dumps(job.inputs, separators=(",", ":"), default=str)
            created_at = job.created_at.isoformat()
            started_at = job.started_at.isoformat() if job.started_at else None
            updated_at = job.updated_at.isoformat()
            finished_at = job.completed_at.isoformat() if job.completed_at else None

            conn = connect_metadata_db(self._db_path)
            run_repo = SQLiteRunHistoryRepository(conn)
            existing = run_repo.get_run(job.id)
            if existing is None:
                run_repo.create_run(
                    run_id=job.id,
                    project_id=job.project_id,
                    token_id=job.token_id,
                    workflow_name=job.workflow,
                    status=job.status.value,
                    created_at=created_at,
                    started_at=started_at,
                    timeout_seconds=job.timeout,
                    updated_at=updated_at,
                    inputs_json=inputs_json,
                    cancellable=cancellable,
                    result_summary=result_summary,
                    error_summary=job.error,
                    execution_state_json=execution_state_json,
                )
            else:
                run_repo.update_run(
                    run_id=job.id,
                    status=job.status.value,
                    started_at=started_at,
                    updated_at=updated_at,
                    finished_at=finished_at,
                    cancellable=cancellable,
                    result_summary=result_summary,
                    error_summary=job.error,
                    execution_state_json=execution_state_json,
                    inputs_json=inputs_json,
                )
            conn.close()

        await self._run_in_executor(_write)

    async def load_job(self, job_id: str) -> dict[str, Any]:
        """Load job data from compact SQLite metadata.

        Args:
            job_id: Job ID to load

        Returns:
            Job data as dict

        Raises:
            KeyError: If job not found in database
            FileNotFoundError: no longer raised (SQLite-only persistence)
        """
        # Check existence in run-history first
        exists = await self._job_exists(job_id)
        if not exists:
            raise KeyError(f"Job not found: {job_id}")

        def _read() -> dict[str, Any]:
            conn = connect_metadata_db(self._db_path)
            run_record = SQLiteRunHistoryRepository(conn).get_run(job_id)
            conn.close()

            if run_record is None:
                raise KeyError(f"Job not found: {job_id}")

            result_payload = self._parse_result_summary(run_record.result_summary)
            execution_state_payload = self._parse_execution_state_json(
                run_record.execution_state_json
            )
            if execution_state_payload is not None:
                if result_payload is None:
                    result_payload = {}
                result_payload["execution_state"] = execution_state_payload
            inputs_payload = self._parse_inputs_json(run_record.inputs_json)
            return {
                "id": run_record.run_id,
                "workflow": run_record.workflow_name,
                "status": run_record.status,
                "timeout": run_record.timeout_seconds,
                "created_at": run_record.created_at,
                "started_at": run_record.started_at,
                "completed_at": run_record.finished_at,
                "updated_at": run_record.updated_at,
                "error": run_record.error_summary,
                "result": result_payload,
                "inputs": inputs_payload,
                "cancellable": run_record.cancellable,
            }

        return await self._run_in_executor(_read)

    async def _job_exists(self, job_id: str) -> bool:
        """Check if job exists in database.

        Args:
            job_id: Job ID to check

        Returns:
            True if job exists, False otherwise
        """

        def _check() -> bool:
            conn = connect_metadata_db(self._db_path)
            cursor = conn.execute("SELECT 1 FROM job_runs WHERE run_id = ?", (job_id,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists

        return await self._run_in_executor(_check)

    async def list_jobs(
        self, status: WorkflowStatus | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List jobs metadata from SQLite.

        Fast query using SQLite metadata only.

        Args:
            status: Filter by status (None for all)
            limit: Maximum number of jobs to return

        Returns:
            List of job metadata dicts (most recent first)
        """

        def _query() -> list[dict[str, Any]]:
            conn = connect_metadata_db(self._db_path)
            conn.row_factory = sqlite3.Row

            if status:
                cursor = conn.execute(
                    """
                    SELECT run_id AS id,
                           workflow_name AS workflow,
                           status,
                           timeout_seconds AS timeout,
                           created_at,
                           started_at,
                           finished_at AS completed_at,
                           error_summary,
                           cancellable
                    FROM job_runs
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (status.value if hasattr(status, "value") else status, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT run_id AS id,
                           workflow_name AS workflow,
                           status,
                           timeout_seconds AS timeout,
                           created_at,
                           started_at,
                           finished_at AS completed_at,
                           error_summary,
                           cancellable
                    FROM job_runs
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            rows = [dict(row) for row in cursor.fetchall()]
            for row in rows:
                row["cancellable"] = bool(int(row["cancellable"]))
            conn.close()
            return rows

        return await self._run_in_executor(_query)

    async def delete_job(self, job_id: str) -> None:
        """Delete job from SQLite metadata.

        Args:
            job_id: Job ID to delete
        """

        def _delete() -> None:
            conn = connect_metadata_db(self._db_path)
            conn.execute("DELETE FROM job_runs WHERE run_id = ?", (job_id,))
            conn.commit()
            conn.close()

        await self._run_in_executor(_delete)

    async def get_stale_jobs(self, grace_period: int = 600) -> list[str]:
        """Find stale RUNNING jobs based on timeout + grace period.

        A job is stale if:
            - Status is RUNNING
            - Last update was more than (timeout + grace_period) ago

        Args:
            grace_period: Additional seconds beyond timeout (default: 600 = 10 min)

        Returns:
            List of stale job IDs
        """

        def _query() -> list[str]:
            conn = connect_metadata_db(self._db_path)
            runs = SQLiteRunHistoryRepository(conn).list_running_runs_for_stale_check()

            now = datetime.now()
            stale_ids = []

            for run in runs:
                updated_at = datetime.fromisoformat(run.updated_at)

                # Check if job hasn't been updated in (timeout + grace) seconds
                elapsed = (now - updated_at).total_seconds()
                if elapsed > (run.timeout_seconds + grace_period):
                    stale_ids.append(run.run_id)

            conn.close()
            return stale_ids

        return await self._run_in_executor(_query)

    async def increment_stat(self, key: str) -> None:
        """Increment a statistics counter.

        Args:
            key: Stat key (total_jobs, completed_jobs, failed_jobs, cancelled_jobs)
        """

        def _increment() -> None:
            conn = connect_metadata_db(self._db_path)
            conn.execute(
                "UPDATE stats SET value = value + 1 WHERE key = ?",
                (key,),
            )
            conn.commit()
            conn.close()

        await self._run_in_executor(_increment)

    async def get_stats(self) -> dict[str, int]:
        """Get all statistics from database.

        Returns:
            Dict of stat keys to values
        """

        def _query() -> dict[str, int]:
            conn = connect_metadata_db(self._db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT key, value FROM stats")
            stats = dict(cursor.fetchall())
            conn.close()
            return stats

        return await self._run_in_executor(_query)

    async def _run_in_executor(self, func: Callable[[], T]) -> T:
        """Run blocking function in thread pool executor.

        Args:
            func: Synchronous function to run

        Returns:
            Result from function
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)

    @staticmethod
    def _build_result_summary(job: Job) -> str | None:
        if not isinstance(job.result, dict):
            return None
        outputs = job.result.get("outputs")
        prompt = job.result.get("prompt")
        payload: dict[str, Any] = {}
        if outputs is not None:
            payload["outputs"] = outputs
        if prompt is not None:
            payload["prompt"] = prompt
        if not payload:
            return None
        import json

        return json.dumps(payload, separators=(",", ":"), default=str)

    @staticmethod
    def _build_execution_state_json(job: Job) -> str | None:
        if not isinstance(job.result, dict):
            return None
        execution_state = job.result.get("execution_state")
        if not isinstance(execution_state, dict):
            return None
        return json.dumps(execution_state, separators=(",", ":"), default=str)

    @staticmethod
    def _parse_result_summary(summary: str | None) -> dict[str, Any] | None:
        if summary is None:
            return None
        import json

        try:
            parsed = json.loads(summary)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _parse_inputs_json(inputs_json: str | None) -> dict[str, Any]:
        if inputs_json is None:
            return {}
        try:
            parsed = json.loads(inputs_json)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _parse_execution_state_json(execution_state_json: str | None) -> dict[str, Any] | None:
        if execution_state_json is None:
            return None
        try:
            parsed = json.loads(execution_state_json)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


__all__ = ["JobStore"]
