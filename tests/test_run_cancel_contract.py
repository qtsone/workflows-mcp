"""Stable cancel contract tests for async run queue."""

import asyncio

import pytest

from workflows_mcp.context import AppContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.job_queue import WorkflowStatus
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.schema import WorkflowSchema


@pytest.fixture
async def app_context():
    registry = WorkflowRegistry()
    executor_registry = create_default_registry()
    llm_config_loader = LLMConfigLoader()
    io_queue = IOQueue()

    registry.register(
        WorkflowSchema(
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
    )
    registry.register(
        WorkflowSchema(
            name="test-slow-cancel",
            description="Slow workflow to test cancel semantics",
            blocks=[
                {
                    "id": "sleep",
                    "type": "Shell",
                    "inputs": {"command": "sleep 2"},
                }
            ],
        )
    )

    return AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,
    )


@pytest.fixture
async def job_queue(app_context):
    from workflows_mcp.engine.job_queue import JobQueue

    queue = JobQueue(app_context, num_workers=1)
    await queue.start()
    yield queue
    await queue.stop(wait_for_completion=False)


@pytest.mark.asyncio
async def test_get_status_exposes_deterministic_cancellable(job_queue):
    """Status exposes cancellable bool for queued/running/terminal states."""
    job_id = await job_queue.submit_job("test-slow-cancel", {})

    queued_or_running = await job_queue.get_status(job_id)
    assert isinstance(queued_or_running["cancellable"], bool)

    for _ in range(50):
        await asyncio.sleep(0.1)
        status = await job_queue.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break

    terminal_status = await job_queue.get_status(job_id)
    assert terminal_status["status"] in ("completed", "failed")
    assert terminal_status["cancellable"] is False


@pytest.mark.asyncio
async def test_cancel_returns_non_cancellable_when_underlying_task_not_interruptible(job_queue):
    """Cancel reports stable non-cancellable contract when no active task exists."""
    job_id = await job_queue.submit_job("test-slow-cancel", {})

    # Wait until worker picks the task to reduce false positives from queued state.
    for _ in range(20):
        await asyncio.sleep(0.05)
        status = await job_queue.get_status(job_id)
        if status["status"] == WorkflowStatus.RUNNING.value:
            break

    # Simulate non-interruptible backend: status says active, but no cancellable runtime handle.
    job_queue._job_tasks.pop(job_id, None)

    result = await job_queue.cancel_job(job_id)
    assert result == {
        "job_id": job_id,
        "cancelled": False,
        "outcome": "non_cancellable",
        "error_code": "JOB_NON_CANCELLABLE",
        "message": "Job cannot be interrupted in its current execution state",
    }
