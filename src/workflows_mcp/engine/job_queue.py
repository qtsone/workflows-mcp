"""Job Queue for async workflow execution.

Enables non-blocking workflow execution with status tracking for long-running workflows.
Uses worker pool pattern for parallel job processing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .job_store import JobStore
from .workflow_runner import WorkflowRunner

if TYPE_CHECKING:
    from ..context import AppContext

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Canonical workflow execution status used across all execution modes.

    This unified status enum is used by:
    - Job model (async execution)
    - ExecutionResult (sync execution)
    - Paused workflows (Prompt blocks)

    Status Lifecycle:
    - QUEUED: Job submitted, waiting for worker
    - RUNNING: Currently executing
    - PAUSED: Waiting for user input (Prompt block)
    - COMPLETED: Successfully finished
    - FAILED: Execution error
    - CANCELLED: User cancelled
    """

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Job metadata and result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Unique job ID")
    workflow: str = Field(description="Workflow name")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Workflow inputs")
    status: WorkflowStatus = Field(default=WorkflowStatus.QUEUED, description="Current status")
    timeout: int = Field(default=3600, description="Job timeout in seconds")
    result: dict[str, Any] | None = Field(default=None, description="Workflow result")
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    started_at: datetime | None = Field(default=None, description="Start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update time (for stale detection)"
    )


class JobQueue:
    """Async workflow execution queue with worker pool.

    Architecture:
    - Multiple worker coroutines process jobs in parallel
    - Jobs tracked by ID for status queries
    - Non-blocking submission returns job_id immediately

    Usage:
        queue = JobQueue(app_context, num_workers=3)
        await queue.start()

        # Submit job, get immediate response
        job_id = await queue.submit_job("workflow-name", {"input": "value"})

        # Check status later
        status = queue.get_status(job_id)
    """

    def __init__(self, app_context: AppContext, num_workers: int = 3):
        """Initialize job queue.

        Args:
            app_context: AppContext instance for creating execution contexts
            num_workers: Number of parallel workers
        """
        self._app_context = app_context
        self._num_workers = num_workers
        self._store = JobStore()  # Persistent storage (SQLite + JSON)
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._job_tasks: dict[str, asyncio.Task[Any]] = {}  # Track running tasks for cancellation
        self._running = False

        # Job retention configuration
        self._max_jobs = int(os.getenv("WORKFLOWS_JOB_HISTORY_MAX", "1000"))
        self._job_ttl = int(os.getenv("WORKFLOWS_JOB_HISTORY_TTL", "86400"))

        # Job timeout configuration
        self._default_job_timeout = int(os.getenv("WORKFLOWS_JOB_TIMEOUT", "3600"))
        self._max_job_timeout = 86400  # 24 hours hard limit

        # Backpressure configuration (soft limit on concurrent jobs)
        self._max_concurrent_jobs = int(os.getenv("WORKFLOWS_MAX_CONCURRENT_JOBS", "500"))

        # Lock for cleanup operations
        self._cleanup_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start worker pool and initialize persistent storage."""
        if self._running:
            logger.warning("JobQueue already running")
            return

        # Initialize persistent storage
        await self._store.init()

        # Detect and mark stale jobs (from previous MCP instance crashes)
        await self._cleanup_stale_jobs()

        # Start workers
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(worker_id=i)) for i in range(self._num_workers)
        ]

        # Load stats from database
        stats = await self._store.get_stats()
        logger.info(
            f"JobQueue started with {self._num_workers} workers. "
            f"Stats: {stats['total_jobs']} total, "
            f"{stats['completed_jobs']} completed, "
            f"{stats['failed_jobs']} failed, "
            f"{stats['cancelled_jobs']} cancelled"
        )

    async def stop(self, wait_for_completion: bool = True) -> None:
        """Stop worker pool.

        Args:
            wait_for_completion: If True, wait for running jobs to complete
        """
        if not self._running:
            return

        self._running = False

        if wait_for_completion:
            # Wait for queue to empty
            await self._queue.join()

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        # Load final stats from database
        stats = await self._store.get_stats()
        logger.info(
            f"JobQueue stopped. Stats: {stats['total_jobs']} total, "
            f"{stats['completed_jobs']} completed, "
            f"{stats['failed_jobs']} failed, "
            f"{stats['cancelled_jobs']} cancelled"
        )

    async def submit_job(
        self, workflow: str, inputs: dict[str, Any] | None = None, timeout: int | None = None
    ) -> str:
        """Submit workflow for async execution.

        Args:
            workflow: Workflow name
            inputs: Workflow inputs
            timeout: Optional job timeout in seconds (default: WORKFLOWS_JOB_TIMEOUT env var)
                     Maximum allowed: 86400 seconds (24 hours)

        Returns:
            Job ID for status tracking

        Raises:
            RuntimeError: If queue not started or at capacity
            ValueError: If timeout exceeds maximum limit
        """
        if not self._running:
            raise RuntimeError("JobQueue not started. Call start() first.")

        # Validate and set timeout
        if timeout is not None:
            if timeout <= 0:
                raise ValueError(f"Timeout must be positive, got {timeout}")
            if timeout > self._max_job_timeout:
                raise ValueError(
                    f"Timeout {timeout}s exceeds maximum allowed "
                    f"{self._max_job_timeout}s (24 hours)"
                )
            job_timeout = timeout
        else:
            job_timeout = self._default_job_timeout

        # Check active job limit (backpressure - soft limit)
        stats = await self.get_stats()
        active_jobs = stats["queue_size"] + len(self._job_tasks)

        if active_jobs >= self._max_concurrent_jobs:
            raise RuntimeError(
                f"Job queue at capacity: {active_jobs}/{self._max_concurrent_jobs} active jobs. "
                f"Current: {stats['queue_size']} queued, {len(self._job_tasks)} running. "
                f"Use get_queue_stats() to monitor or cancel_job() to free slots. "
                f"Adjust via WORKFLOWS_MAX_CONCURRENT_JOBS environment variable."
            )

        # Create job with timeout
        job = Job(
            id=f"job_{uuid.uuid4().hex[:8]}",
            workflow=workflow,
            inputs=inputs or {},
            timeout=job_timeout,
        )

        # Persist job to storage (state.db + JSON file)
        await self._store.save_job(job)

        # Increment stats in database
        await self._store.increment_stat("total_jobs")

        # Get current total for cleanup trigger
        stats = await self._store.get_stats()
        if stats["total_jobs"] % 10 == 0:
            asyncio.create_task(self._cleanup_old_jobs())

        # Queue for execution (ephemeral, not persisted)
        await self._queue.put(job)
        logger.info(f"Job submitted: {job.id} (workflow={workflow}, timeout={job_timeout}s)")

        return job.id

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """Get job status.

        Returns essential job information only:
        - Job status and timing
        - Workflow-level outputs (not block internals)
        - Error message if failed
        - Path to full result file for debugging

        This minimal response prevents token waste by excluding block execution
        details, stdout/stderr, and deep execution trees (which can exceed 30K tokens).
        Full job data is stored in result_file and can be accessed for debugging.

        Args:
            job_id: Job ID

        Returns:
            {
                "id": "job_abc123",
                "workflow": "workflow-name",
                "status": "completed" | "running" | "failed" | "queued" | "cancelled",
                "outputs": {...} | null,  # Workflow outputs only (if completed)
                "error": "..." | null,    # Error message (if failed)
                "timeout": 3600,          # Job timeout in seconds
                "created_at": "2025-11-11T12:00:00",
                "started_at": "2025-11-11T12:00:01" | null,
                "completed_at": "2025-11-11T12:00:05" | null,
                "result_file": "~/.workflows/states/<hash>/jobs/job_abc123.json"
            }

        Raises:
            KeyError: If job not found in database
            FileNotFoundError: If JSON file missing (corrupted state)
        """
        from .state_config import StateConfig

        # Load full job data from JSON file (includes inputs, result, all execution details)
        job_data = await self._store.load_job(job_id)

        # Extract workflow-level outputs from result (if available)
        result = job_data.get("result") or {}
        outputs = result.get("outputs")

        # Extract pause prompt if job is paused (critical for resume)
        prompt = result.get("prompt") if job_data["status"] == "paused" else None

        # Build result file path (full job data location for debugging)
        state_dir = StateConfig.get_state_dir()
        result_file = str(state_dir / "jobs" / f"{job_data['id']}.json")

        # Return minimal response optimized for LLM callers
        response = {
            "id": job_data["id"],
            "workflow": job_data["workflow"],
            "status": job_data["status"],
            "outputs": outputs,  # Workflow outputs only, not block internals
            "error": job_data.get("error"),
            "timeout": job_data.get("timeout", 3600),  # Job timeout in seconds
            "created_at": job_data["created_at"],
            "started_at": job_data.get("started_at"),
            "completed_at": job_data.get("completed_at"),
            "result_file": result_file,
        }

        # Add prompt field when paused (so LLM/user knows what response is expected)
        if prompt:
            response["prompt"] = prompt

        return response

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel pending or running job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled, False if already completed/failed

        Raises:
            KeyError: If job not found
        """
        # Load job from storage
        job_data = await self._store.load_job(job_id)

        # Check if job can be cancelled
        status = WorkflowStatus(job_data["status"])
        if status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
            return False

        # Create Job model from data
        job = Job(**job_data)

        # Update job status
        job.status = WorkflowStatus.CANCELLED
        job.completed_at = datetime.now()
        job.updated_at = datetime.now()

        # Save to storage
        await self._store.save_job(job)
        await self._store.increment_stat("cancelled_jobs")

        # Cancel running task if exists
        if job_id in self._job_tasks:
            task = self._job_tasks[job_id]
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled running task for job: {job_id}")
            del self._job_tasks[job_id]
        else:
            logger.info(f"Job cancelled (not yet started): {job_id}")

        return True

    async def list_jobs(
        self, status: WorkflowStatus | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List jobs metadata from database (fast query, no JSON file loading).

        Args:
            status: Filter by status (None = all jobs)
            limit: Maximum number of jobs to return

        Returns:
            List of job metadata dicts (most recent first)
            Does NOT include full inputs/result data
        """
        # Query database only (fast, no JSON file access)
        return await self._store.list_jobs(status=status, limit=limit)

    async def get_stats(self) -> dict[str, int]:
        """Get queue statistics from database and runtime state.

        Returns:
            Dict with total_jobs, completed_jobs, failed_jobs, cancelled_jobs,
            queue_size, and active_workers
        """
        # Get persistent stats from database
        db_stats = await self._store.get_stats()

        # Add runtime stats (ephemeral state)
        return {
            **db_stats,
            "queue_size": self._queue.qsize(),
            "active_workers": len([w for w in self._workers if not w.done()]),
        }

    async def _cleanup_stale_jobs(self) -> None:
        """Detect and mark stale RUNNING jobs as FAILED on startup.

        A job is stale if:
            - Status is RUNNING
            - Last update was more than (timeout + grace_period) ago

        This handles MCP server crashes where jobs remain in RUNNING state forever.
        """
        grace_period = 600  # 10 minutes beyond timeout

        # Get stale jobs from store
        stale_job_ids = await self._store.get_stale_jobs(grace_period=grace_period)

        if not stale_job_ids:
            return

        # Mark each stale job as failed
        for job_id in stale_job_ids:
            try:
                # Load job
                job_data = await self._store.load_job(job_id)
                job = Job(**job_data)

                # Mark as failed
                job.status = WorkflowStatus.FAILED
                job.error = "Job marked as failed (server crash or timeout)"
                job.completed_at = datetime.now()
                job.updated_at = datetime.now()

                # Save back to storage
                await self._store.save_job(job)
                await self._store.increment_stat("failed_jobs")

                logger.warning(
                    f"Marked stale job as failed: {job_id} "
                    f"(workflow={job.workflow}, timeout={job.timeout}s)"
                )

            except Exception as e:
                logger.error(f"Failed to mark stale job {job_id} as failed: {e}")

        logger.info(f"Stale job cleanup: marked {len(stale_job_ids)} jobs as failed")

    async def _cleanup_old_jobs(self) -> None:
        """Remove completed jobs older than TTL from storage.

        Enforces job history limits by:
            1. Removing jobs older than TTL (WORKFLOWS_JOB_HISTORY_TTL)
            2. Keeping only MAX jobs (WORKFLOWS_JOB_HISTORY_MAX)
        """
        # Prevent concurrent cleanup runs
        async with self._cleanup_lock:
            # Get all jobs sorted by creation time (newest first)
            all_jobs = await self._store.list_jobs(status=None, limit=10000)

            # Filter jobs to delete
            cutoff = datetime.now() - timedelta(seconds=self._job_ttl)
            to_delete = []

            for idx, job_metadata in enumerate(all_jobs):
                # Delete if too old
                if job_metadata.get("completed_at"):
                    completed_at = datetime.fromisoformat(job_metadata["completed_at"])
                    if completed_at < cutoff:
                        to_delete.append(job_metadata["id"])
                        continue

                # Delete if beyond max limit (keep newest max_jobs)
                if idx >= self._max_jobs:
                    to_delete.append(job_metadata["id"])

            # Delete jobs from storage (state.db + JSON files)
            if to_delete:
                for job_id in to_delete:
                    try:
                        await self._store.delete_job(job_id)
                    except Exception as e:
                        logger.error(f"Failed to delete job {job_id}: {e}")

                logger.info(
                    f"Job cleanup: removed {len(to_delete)} jobs, "
                    f"remaining: {len(all_jobs) - len(to_delete)}"
                )

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine - processes jobs from queue.

        Args:
            worker_id: Worker identifier for logging
        """
        logger.info(f"JobQueue worker {worker_id} started")

        while True:
            try:
                # Get next job (with timeout to check _running flag)
                timeout = 1.0 if self._running else 0.1
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except TimeoutError:
                    if not self._running:
                        break  # Exit when stopping and no more items
                    continue  # Keep waiting when running

                # Skip if cancelled while queued
                if job.status == WorkflowStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                # Update status to RUNNING and persist
                job.status = WorkflowStatus.RUNNING
                job.started_at = datetime.now()
                job.updated_at = datetime.now()
                await self._store.save_job(job)

                # Store task reference for cancellation support
                current_task = asyncio.current_task()
                if current_task:
                    self._job_tasks[job.id] = current_task

                logger.info(
                    f"Worker {worker_id} executing job: {job.id} "
                    f"(workflow={job.workflow}, timeout={job.timeout}s)"
                )

                try:
                    # Create execution context and runner for this job
                    workflow_schema = self._app_context.registry.get(job.workflow)
                    if not workflow_schema:
                        raise ValueError(f"Workflow not found: {job.workflow}")

                    execution_context = self._app_context.create_execution_context()
                    runner = WorkflowRunner()

                    # Execute workflow with timeout wrapper
                    result = await asyncio.wait_for(
                        runner.execute(workflow_schema, job.inputs, execution_context),
                        timeout=job.timeout,
                    )

                    # Update job with result and persist (unified Job architecture)
                    # Check if workflow paused (Prompt block)
                    if hasattr(result, "status") and result.status == "paused":
                        # Workflow paused - set PAUSED status
                        job.status = WorkflowStatus.PAUSED
                        # Store execution_state in result for resume
                        if hasattr(result, "_build_debug_data"):
                            job.result = result._build_debug_data()
                        else:
                            job.result = {"status": "paused", "error": "Missing execution state"}
                        job.updated_at = datetime.now()
                        # Note: No completed_at - workflow not finished
                        await self._store.save_job(job)

                        prompt_preview = (
                            result.pause_data.prompt[:50] if result.pause_data else "N/A"
                        )
                        logger.info(
                            f"Worker {worker_id} paused job: {job.id} "
                            f"(workflow={job.workflow}, prompt={prompt_preview}...)"
                        )
                    else:
                        # Workflow completed successfully
                        job.status = WorkflowStatus.COMPLETED
                        # Use clean debug format (same as /tmp/ debug files)
                        # This prevents secret_redactor serialization and provides consistent format
                        if hasattr(result, "_build_debug_data"):
                            job.result = result._build_debug_data()
                        elif isinstance(result, dict):
                            job.result = dict(result)
                        else:
                            job.result = {"value": result}
                        job.completed_at = datetime.now()
                        job.updated_at = datetime.now()
                        await self._store.save_job(job)
                        await self._store.increment_stat("completed_jobs")

                        duration = (job.completed_at - job.started_at).total_seconds()
                        logger.info(
                            f"Worker {worker_id} completed job: {job.id} "
                            f"(status={result.status if hasattr(result, 'status') else 'unknown'}, "
                            f"duration={duration:.1f}s)"
                        )

                except TimeoutError:
                    # Job exceeded timeout - persist failure
                    job.status = WorkflowStatus.FAILED
                    job.error = f"Job exceeded timeout limit ({job.timeout} seconds)"
                    job.completed_at = datetime.now()
                    job.updated_at = datetime.now()
                    await self._store.save_job(job)
                    await self._store.increment_stat("failed_jobs")

                    duration = (job.completed_at - job.started_at).total_seconds()
                    logger.error(
                        f"Worker {worker_id} job timeout: {job.id} "
                        f"(workflow={job.workflow}, timeout={job.timeout}s, "
                        f"duration={duration:.1f}s)"
                    )

                except Exception as e:
                    # Job failed - persist failure
                    job.status = WorkflowStatus.FAILED
                    job.error = str(e)
                    job.completed_at = datetime.now()
                    job.updated_at = datetime.now()
                    await self._store.save_job(job)
                    await self._store.increment_stat("failed_jobs")

                    logger.error(f"Worker {worker_id} failed job: {job.id} - {e}", exc_info=True)

                finally:
                    # Clean up task reference
                    if job.id in self._job_tasks:
                        del self._job_tasks[job.id]

                    self._queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"JobQueue worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"JobQueue worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"JobQueue worker {worker_id} stopped")


__all__ = ["Job", "WorkflowStatus", "JobQueue"]
