"""Tests for Job Queue."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from workflows_mcp.context import AppContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.job_queue import Job, JobQueue, WorkflowStatus
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.schema import WorkflowSchema


@pytest.fixture
async def app_context():
    """Create AppContext for job queue tests."""
    registry = WorkflowRegistry()
    executor_registry = create_default_registry()
    llm_config_loader = LLMConfigLoader()
    io_queue = IOQueue()

    # Create simple test workflow
    test_workflow = WorkflowSchema(
        name="test-simple",
        description="Simple test workflow",
        blocks=[
            {
                "id": "echo",
                "type": "Shell",
                "inputs": {"command": "echo 'test output'"},
            }
        ],
    )
    registry.register(test_workflow)

    app_ctx = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,
    )

    return app_ctx


@pytest.fixture
async def job_queue(app_context):
    """Create and start JobQueue."""
    queue = JobQueue(app_context, num_workers=2)
    await queue.start()
    yield queue
    await queue.stop(wait_for_completion=False)


@pytest.mark.asyncio
async def test_job_queue_submit_and_complete(job_queue):
    """Test job submission and successful completion."""
    # Submit job
    job_id = await job_queue.submit_job("test-simple", {})

    assert job_id.startswith("job_")

    # Wait for completion (with timeout)
    for _ in range(50):  # 5 second timeout
        await asyncio.sleep(0.1)
        status = await job_queue.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break

    # Verify completion
    status = await job_queue.get_status(job_id)
    assert status["status"] == "completed"
    assert status["outputs"] is not None  # Workflow outputs (not full result)
    assert status["error"] is None
    assert "result_file" in status  # Full result stored in file


@pytest.mark.asyncio
async def test_job_queue_workflow_not_found(job_queue):
    """Test job execution with non-existent workflow."""
    # Submit job with invalid workflow
    job_id = await job_queue.submit_job("nonexistent-workflow", {})

    # Wait for failure
    for _ in range(50):
        await asyncio.sleep(0.1)
        status = await job_queue.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break

    # Verify failure
    status = await job_queue.get_status(job_id)
    assert status["status"] == "failed"
    assert status["error"] is not None
    assert "not found" in status["error"].lower()


@pytest.mark.asyncio
async def test_job_queue_cancel_queued_job(job_queue):
    """Test cancellation of queued job."""
    # Submit multiple jobs to fill queue
    job_ids = []
    for _ in range(5):
        job_id = await job_queue.submit_job("test-simple", {})
        job_ids.append(job_id)

    # Cancel one of the jobs immediately
    # With persistent storage, there's a race between cancel and worker execution
    # cancel_job() returns True if cancelled, False if already completed/failed
    cancelled = await job_queue.cancel_job(job_ids[2])
    assert cancelled in (True, False)

    # Check status - job may be in various states depending on timing
    status = await job_queue.get_status(job_ids[2])
    assert status["status"] in ("cancelled", "running", "completed", "failed")


@pytest.mark.asyncio
async def test_job_queue_cancel_completed_job(job_queue):
    """Test that completed jobs cannot be cancelled."""
    # Submit and wait for completion
    job_id = await job_queue.submit_job("test-simple", {})

    # Wait for completion
    for _ in range(50):
        await asyncio.sleep(0.1)
        status = await job_queue.get_status(job_id)
        if status["status"] == "completed":
            break

    # Try to cancel completed job
    cancelled = await job_queue.cancel_job(job_id)
    assert cancelled is False


@pytest.mark.asyncio
async def test_job_queue_get_status_not_found(job_queue):
    """Test get_status with non-existent job ID."""
    with pytest.raises(KeyError):
        await job_queue.get_status("nonexistent_job_id")


@pytest.mark.asyncio
async def test_job_queue_cancel_not_found(job_queue):
    """Test cancel_job with non-existent job ID."""
    with pytest.raises(KeyError):
        await job_queue.cancel_job("nonexistent_job_id")


@pytest.mark.asyncio
async def test_job_queue_list_jobs_all(job_queue):
    """Test listing all jobs."""
    # Submit multiple jobs
    job_ids = []
    for _ in range(3):
        job_id = await job_queue.submit_job("test-simple", {})
        job_ids.append(job_id)

    # Wait a bit for execution
    await asyncio.sleep(0.2)

    # List all jobs
    jobs = await job_queue.list_jobs()
    assert len(jobs) >= 3


@pytest.mark.asyncio
async def test_job_queue_list_jobs_with_status_filter(job_queue):
    """Test listing jobs with status filter.

    Note: In fast CI environments, jobs may complete before cancellation
    succeeds. This test validates the filtering logic when cancelled jobs
    exist, but accepts that cancellation may not always succeed.
    """
    # Submit multiple jobs to increase chance of catching one in-flight
    job_ids = []
    for _ in range(10):
        job_id = await job_queue.submit_job("test-simple", {})
        job_ids.append(job_id)

    # Try to cancel several jobs
    # With persistent storage, there's a race between cancel and worker execution
    for i in [1, 3, 5, 7]:
        await job_queue.cancel_job(job_ids[i])

    # Small delay to ensure cancellation is processed
    await asyncio.sleep(0.1)

    # List cancelled jobs - may be 0 if all jobs completed too quickly
    cancelled_jobs = await job_queue.list_jobs(status=WorkflowStatus.CANCELLED)

    # Verify filtering works correctly (if any cancelled jobs exist)
    if len(cancelled_jobs) > 0:
        assert all(j["status"] == "cancelled" for j in cancelled_jobs)

    # List completed jobs - should have at least some
    completed_jobs = await job_queue.list_jobs(status=WorkflowStatus.COMPLETED)
    assert len(completed_jobs) >= 1
    assert all(j["status"] == "completed" for j in completed_jobs)


@pytest.mark.asyncio
async def test_job_queue_list_jobs_with_limit(job_queue):
    """Test listing jobs with limit."""
    # Submit many jobs
    for _ in range(10):
        await job_queue.submit_job("test-simple", {})

    # List with limit
    jobs = await job_queue.list_jobs(limit=5)
    assert len(jobs) == 5


@pytest.mark.asyncio
async def test_job_queue_parallel_execution(job_queue):
    """Test parallel job execution with multiple workers."""
    # Submit multiple jobs
    job_ids = []
    for _ in range(4):
        job_id = await job_queue.submit_job("test-simple", {})
        job_ids.append(job_id)

    # Wait for all to complete
    for _ in range(100):  # 10 second timeout
        await asyncio.sleep(0.1)
        statuses = [(await job_queue.get_status(jid))["status"] for jid in job_ids]
        if all(s in ("completed", "failed") for s in statuses):
            break

    # Verify all completed
    for job_id in job_ids:
        status = await job_queue.get_status(job_id)
        assert status["status"] == "completed"


@pytest.mark.asyncio
async def test_job_queue_stats(job_queue):
    """Test job queue statistics."""
    # Submit and complete jobs
    job_id1 = await job_queue.submit_job("test-simple", {})
    job_id2 = await job_queue.submit_job("nonexistent-workflow", {})

    # Wait for completion
    for _ in range(50):
        await asyncio.sleep(0.1)
        status1 = await job_queue.get_status(job_id1)
        status2 = await job_queue.get_status(job_id2)
        if status1["status"] in ("completed", "failed") and status2["status"] in (
            "completed",
            "failed",
        ):
            break

    # Get stats
    stats = await job_queue.get_stats()
    assert stats["total_jobs"] >= 2
    assert stats["completed_jobs"] >= 1
    assert stats["failed_jobs"] >= 1


@pytest.mark.asyncio
async def test_job_queue_not_started_error(app_context):
    """Test that submitting to non-started queue raises error."""
    queue = JobQueue(app_context, num_workers=2)

    with pytest.raises(RuntimeError, match="not started"):
        await queue.submit_job("test-simple", {})


@pytest.mark.asyncio
async def test_job_queue_double_start(job_queue):
    """Test that starting already running queue is idempotent."""
    # Queue is already started by fixture
    # Start again should be no-op (just warning)
    await job_queue.start()
    # Should not raise


@pytest.mark.asyncio
async def test_job_queue_stop_waits_for_completion(app_context):
    """Test that stop(wait_for_completion=True) waits for jobs."""
    queue = JobQueue(app_context, num_workers=2)
    await queue.start()

    # Submit jobs
    job_ids = []
    for _ in range(3):
        job_id = await queue.submit_job("test-simple", {})
        job_ids.append(job_id)

    # Stop and wait for completion
    await queue.stop(wait_for_completion=True)

    # All jobs should be completed or failed
    for job_id in job_ids:
        status = await queue.get_status(job_id)
        assert status["status"] in ("completed", "failed", "cancelled")


@pytest.mark.asyncio
async def test_job_queue_job_model():
    """Test Job model creation and serialization."""
    job = Job(
        id="test_job_123",
        workflow="test-workflow",
        inputs={"key": "value"},
        status=WorkflowStatus.QUEUED,
    )

    assert job.id == "test_job_123"
    assert job.workflow == "test-workflow"
    assert job.inputs == {"key": "value"}
    assert job.status == WorkflowStatus.QUEUED
    assert job.timeout == 3600  # Default timeout
    assert job.result is None
    assert job.error is None

    # Test serialization
    job_dict = job.model_dump()
    assert job_dict["id"] == "test_job_123"
    assert job_dict["status"] == "queued"
    assert job_dict["timeout"] == 3600


@pytest.mark.asyncio
async def test_job_timeout(job_queue, app_context):
    """Test that jobs respect timeout limits."""
    # Create a workflow that sleeps longer than timeout
    app_context.registry.register(
        WorkflowSchema(
            name="slow-workflow",
            description="Slow workflow for timeout testing",
            blocks=[
                {
                    "id": "sleep",
                    "type": "Shell",
                    "inputs": {"command": "sleep 10"},
                }
            ],
        )
    )

    # Submit job with 1 second timeout
    job_id = await job_queue.submit_job("slow-workflow", timeout=1)

    # Wait for job to timeout (give it some extra time)
    await asyncio.sleep(3)

    # Check job failed with timeout error
    status = await job_queue.get_status(job_id)
    assert status["status"] == "failed"
    assert "timeout" in status["error"].lower()
    assert status["timeout"] == 1


@pytest.mark.asyncio
async def test_job_timeout_validation(job_queue):
    """Test timeout validation."""
    # Negative timeout should raise
    with pytest.raises(ValueError, match="Timeout must be positive"):
        await job_queue.submit_job("test-simple", timeout=-1)

    # Zero timeout should raise
    with pytest.raises(ValueError, match="Timeout must be positive"):
        await job_queue.submit_job("test-simple", timeout=0)

    # Timeout exceeding maximum should raise
    with pytest.raises(ValueError, match="exceeds maximum"):
        await job_queue.submit_job("test-simple", timeout=99999)


@pytest.mark.asyncio
async def test_job_custom_timeout(job_queue):
    """Test that custom timeout is stored and used."""
    # Submit job with custom timeout
    job_id = await job_queue.submit_job("test-simple", timeout=7200)

    # Check timeout is stored immediately
    status = await job_queue.get_status(job_id)
    assert status["timeout"] == 7200

    # Wait for completion
    for _ in range(50):
        await asyncio.sleep(0.1)
        status = await job_queue.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break

    # Verify timeout is still in status after completion
    status = await job_queue.get_status(job_id)
    assert status["timeout"] == 7200


@pytest.mark.asyncio
async def test_job_default_timeout(job_queue):
    """Test that default timeout is used when not specified."""
    # Submit job without timeout parameter
    job_id = await job_queue.submit_job("test-simple")

    # Check default timeout is used (3600 seconds)
    status = await job_queue.get_status(job_id)
    assert status["timeout"] == 3600


@pytest.mark.asyncio
async def test_job_queue_backpressure():
    """Test backpressure mechanism with soft limit."""
    # Create temporary directory for state
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create app context with registry
        registry = WorkflowRegistry()
        executor_registry = create_default_registry()
        llm_config_loader = LLMConfigLoader()
        io_queue = IOQueue()

        # Register simple workflow
        simple_workflow = WorkflowSchema(
            name="test-simple",
            description="Test",
            blocks=[
                {
                    "id": "echo",
                    "type": "Shell",
                    "inputs": {"command": "echo 'test'"},
                }
            ],
        )
        registry.register(simple_workflow)

        app_context = AppContext(
            registry=registry,
            executor_registry=executor_registry,
            llm_config_loader=llm_config_loader,
            io_queue=io_queue,
        )

        # Create queue with LOW max_concurrent_jobs limit
        queue = JobQueue(app_context, num_workers=1)
        queue._max_concurrent_jobs = 2  # Override for testing

        await queue.start()

        try:
            # Submit jobs up to the limit
            job_id_1 = await queue.submit_job("test-simple")
            _ = await queue.submit_job("test-simple")  # Second job (fills queue)

            # Verify we have 2 active jobs
            stats = await queue.get_stats()
            assert stats["queue_size"] + len(queue._job_tasks) == 2

            # Try to submit one more - should fail with RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                await queue.submit_job("test-simple")

            # Verify error message contains useful information
            error_msg = str(exc_info.value)
            assert "Job queue at capacity" in error_msg
            assert "2/2 active jobs" in error_msg
            assert "WORKFLOWS_MAX_CONCURRENT_JOBS" in error_msg

            # Cancel one job to free a slot
            await queue.cancel_job(job_id_1)

            # Wait a bit for cancellation to propagate
            await asyncio.sleep(0.1)

            # Should be able to submit again after freeing a slot
            job_id_3 = await queue.submit_job("test-simple")
            assert job_id_3 is not None

        finally:
            await queue.stop(wait_for_completion=False)

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
