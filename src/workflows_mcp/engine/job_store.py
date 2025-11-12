"""Persistent job storage using SQLite + JSON files.

Architecture:
    - SQLite (state.db): Job metadata for fast queries
    - JSON files (jobs/*.json): Full job data including large inputs/results
    - Write-through pattern: All writes immediately persisted
    - Load-on-demand: No in-memory cache, always read from filesystem

Storage Layout:
    ~/.workflows/states/<hash-of-cwd>/
      state.db          # SQLite database
      jobs/
        job_abc123.json
        job_def456.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar, cast

from .state_config import StateConfig

if TYPE_CHECKING:
    from .job_queue import Job, WorkflowStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JobStore:
    """Persistent storage for job queue using SQLite + JSON files.

    Provides atomic, durable storage with support for concurrent access
    across multiple MCP server instances via SQLite WAL mode.

    Architecture:
        - state.db: Job metadata (id, workflow, status, timestamps)
        - jobs/*.json: Full job data (inputs, result, error)
        - No in-memory cache (load on demand)
        - Write-through (immediate persistence)

    Example:
        store = JobStore()
        await store.init()

        # Save job
        await store.save_job(job)

        # Load job
        job_data = await store.load_job("job_abc123")

        # List metadata only (fast)
        jobs = await store.list_jobs(status="completed", limit=100)
    """

    def __init__(self) -> None:
        """Initialize job store with path-based isolation."""
        self._db_path = StateConfig.get_db_path()
        self._jobs_dir = StateConfig.get_jobs_dir()

    async def init(self) -> None:
        """Initialize database schema and load existing stats.

        Must be called before using the store.
        Creates tables and indexes if they don't exist.
        """
        # Initialize database schema
        await self._run_in_executor(self._init_db)

        logger.info(f"JobStore initialized: db={self._db_path}, jobs_dir={self._jobs_dir}")

    def _init_db(self) -> None:
        """Initialize SQLite database with schema (runs in thread pool).

        Creates:
            - jobs table with metadata and indexes
            - stats table for persistent statistics
            - WAL mode for concurrent access
        """
        conn = sqlite3.connect(self._db_path)

        # Enable WAL mode for concurrent access (multiple MCP instances)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster, still safe with WAL

        # Create jobs metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                workflow TEXT NOT NULL,
                status TEXT NOT NULL,
                timeout INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL,
                error_summary TEXT
            )
        """)

        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON jobs(updated_at)")

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
        """Save job to both SQLite metadata and JSON file.

        Atomic write pattern: JSON file written first, then metadata updated.
        Uses temp file + rename for atomic JSON writes.

        Args:
            job: Job instance to save
        """
        # Update job's updated_at timestamp
        job.updated_at = datetime.now()

        # Write to JSON file (atomic via temp file)
        await self._save_job_file(job)

        # Write metadata to SQLite
        await self._save_job_metadata(job)

    async def _save_job_file(self, job: Job) -> None:
        """Save full job data to JSON file (atomic write).

        Uses temp file + rename pattern for atomic writes.

        Args:
            job: Job instance to save
        """

        def _write() -> None:
            job_file = self._jobs_dir / f"{job.id}.json"
            temp_file = job_file.with_suffix(".json.tmp")

            # Write to temp file
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(job.model_dump(), f, indent=2, default=str, ensure_ascii=False)

            # Atomic rename (POSIX guarantee)
            temp_file.rename(job_file)

        await self._run_in_executor(_write)

    async def _save_job_metadata(self, job: Job) -> None:
        """Save job metadata to SQLite database.

        Args:
            job: Job instance to save metadata for
        """

        def _write() -> None:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    job.id,
                    job.workflow,
                    job.status.value,
                    job.timeout,
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.updated_at.isoformat(),
                    job.error[:200] if job.error else None,  # Truncate error summary
                ),
            )
            conn.commit()
            conn.close()

        await self._run_in_executor(_write)

    async def load_job(self, job_id: str) -> dict[str, Any]:
        """Load full job data from JSON file.

        Args:
            job_id: Job ID to load

        Returns:
            Job data as dict

        Raises:
            KeyError: If job not found in database
            FileNotFoundError: If JSON file missing (corrupted state)
        """
        # Check existence in database first
        exists = await self._job_exists(job_id)
        if not exists:
            raise KeyError(f"Job not found: {job_id}")

        # Load from JSON file
        def _read() -> dict[str, Any]:
            job_file = self._jobs_dir / f"{job_id}.json"
            if not job_file.exists():
                raise FileNotFoundError(f"Job file missing (corrupted state): {job_id}")

            with open(job_file, encoding="utf-8") as f:
                return cast(dict[str, Any], json.load(f))

        return await self._run_in_executor(_read)

    async def _job_exists(self, job_id: str) -> bool:
        """Check if job exists in database.

        Args:
            job_id: Job ID to check

        Returns:
            True if job exists, False otherwise
        """

        def _check() -> bool:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists

        return await self._run_in_executor(_check)

    async def list_jobs(
        self, status: WorkflowStatus | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List jobs metadata without loading full JSON files.

        Fast query using SQLite metadata only.

        Args:
            status: Filter by status (None for all)
            limit: Maximum number of jobs to return

        Returns:
            List of job metadata dicts (most recent first)
        """

        def _query() -> list[dict[str, Any]]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row

            if status:
                cursor = conn.execute(
                    """
                    SELECT id, workflow, status, timeout,
                           created_at, started_at, completed_at, error_summary
                    FROM jobs
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (status.value if hasattr(status, "value") else status, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, workflow, status, timeout,
                           created_at, started_at, completed_at, error_summary
                    FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows

        return await self._run_in_executor(_query)

    async def delete_job(self, job_id: str) -> None:
        """Delete job from both SQLite and JSON file.

        Args:
            job_id: Job ID to delete
        """

        def _delete() -> None:
            # Delete from database
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            conn.close()

            # Delete JSON file
            job_file = self._jobs_dir / f"{job_id}.json"
            job_file.unlink(missing_ok=True)

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
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute(
                """
                SELECT id, timeout, updated_at
                FROM jobs
                WHERE status = 'running'
            """
            )

            now = datetime.now()
            stale_ids = []

            for row in cursor:
                job_id, timeout, updated_at_str = row
                updated_at = datetime.fromisoformat(updated_at_str)

                # Check if job hasn't been updated in (timeout + grace) seconds
                elapsed = (now - updated_at).total_seconds()
                if elapsed > (timeout + grace_period):
                    stale_ids.append(job_id)

            conn.close()
            return stale_ids

        return await self._run_in_executor(_query)

    async def increment_stat(self, key: str) -> None:
        """Increment a statistics counter.

        Args:
            key: Stat key (total_jobs, completed_jobs, failed_jobs, cancelled_jobs)
        """

        def _increment() -> None:
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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


__all__ = ["JobStore"]
