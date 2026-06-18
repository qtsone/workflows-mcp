#!/usr/bin/env python3
"""Test nested Workflow blocks in for_each with pause/resume.

This test reproduces a bug where:
1. Parent workflow has for_each: sequential with Workflow blocks
2. Child workflow pauses (Prompt block)
3. User provides feedback/response
4. Response is incorrectly routed to parent instead of child

Test Scenario:
1. Parent workflow with for_each: sequential calling child workflows
2. Child workflows have Prompt blocks that pause for approval
3. Resume first child with "yes"
4. Verify first child completes with approval
5. Second child should pause for its approval
6. Resume second child with "yes"
7. Verify both children completed successfully

Transport: Direct MCP tool calls via mock AppContext (ADR-013).
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from workflows_mcp.context import AppContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.job_queue import JobQueue
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.tools import execute_workflow, resume_workflow

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture
def full_registry() -> WorkflowRegistry:
    """WorkflowRegistry loaded from the full test workflows directory tree."""
    registry = WorkflowRegistry()
    registry.load_from_directory(WORKFLOWS_DIR)
    return registry


@pytest.fixture
async def full_context(full_registry: WorkflowRegistry) -> MagicMock:
    """AppContext with all test workflows, a started IOQueue, and a live JobStore."""
    executor_registry = create_default_registry()
    llm_config_loader = LLMConfigLoader()
    io_queue = IOQueue()

    app_context = AppContext(
        registry=full_registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,
    )

    job_queue = JobQueue(app_context, num_workers=2)
    await job_queue._store.init()

    app_context = AppContext(
        registry=full_registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=job_queue,
    )

    await io_queue.start()
    try:
        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = app_context
        yield mock_ctx
    finally:
        await io_queue.stop()


# =============================================================================
# Tests
# =============================================================================


class TestNestedWorkflowPauseResume:
    """Test pause/resume for nested Workflow blocks in for_each loops."""

    @pytest.mark.asyncio
    async def test_nested_workflow_in_foreach_sequential_pause_resume(
        self, full_context: MagicMock
    ) -> None:
        """Test that nested workflow pause/resume works correctly in for_each sequential.

        This is the core bug reproduction test:
        1. Start parent workflow with 2 items
        2. First child pauses for approval
        3. Resume with "yes" -> first child should complete
        4. Second child should pause for approval (not parent!)
        5. Resume with "yes" -> second child should complete
        6. Parent workflow should complete with all items processed
        """
        # Step 1: Start parent workflow
        exec_result = await execute_workflow(
            workflow="nested-workflow-in-foreach-parent",
            inputs={
                "work_items": [
                    {"name": "item1", "value": "value1"},
                    {"name": "item2", "value": "value2"},
                ]
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert exec_response["status"] == "paused", (
            f"Expected workflow to pause, got status: {exec_response.get('status')}"
        )
        assert "job_id" in exec_response
        job_id = exec_response["job_id"]

        prompt = exec_response.get("prompt", "")
        assert "item1" in prompt, f"Expected prompt to mention 'item1' (first child), got: {prompt}"

        # Step 2: Resume first child with "yes" — should pause for second child
        resume1_result = await resume_workflow(
            job_id=job_id, response="yes", debug=True, ctx=full_context
        )
        resume1_response: dict[str, Any] = resume1_result.structuredContent
        assert resume1_response["status"] == "paused", (
            f"Expected workflow to pause for item2, got: {resume1_response.get('status')}. "
            f"Resume response was incorrectly routed to parent workflow. "
            f"Full response: {resume1_response}"
        )

        prompt2 = resume1_response.get("prompt", "")
        assert "item2" in prompt2, (
            f"Expected prompt to mention 'item2' (second child), got: {prompt2}"
        )

        job_id_2 = resume1_response.get("job_id", job_id)

        # Step 3: Resume second child — should complete
        resume2_result = await resume_workflow(
            job_id=job_id_2, response="yes", debug=True, ctx=full_context
        )
        resume2_response: dict[str, Any] = resume2_result.structuredContent
        assert resume2_response["status"] == "success", (
            f"Expected workflow to complete after both children approved, "
            f"got status: {resume2_response.get('status')}. "
            f"Error: {resume2_response.get('error')}"
        )

        outputs = resume2_response.get("outputs", {})
        assert outputs.get("setup_completed") is True
        assert outputs.get("all_items_processed") is True
        assert outputs.get("finalize_completed") is True

    @pytest.mark.asyncio
    async def test_nested_workflow_single_item_pause_resume(self, full_context: MagicMock) -> None:
        """Test simpler case: single item for_each with nested workflow pause."""
        exec_result = await execute_workflow(
            workflow="nested-workflow-in-foreach-parent",
            inputs={
                "work_items": [
                    {"name": "single_item", "value": "single_value"},
                ]
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert exec_response["status"] == "paused"
        job_id = exec_response["job_id"]

        prompt = exec_response.get("prompt", "")
        assert "single_item" in prompt

        resume_result = await resume_workflow(
            job_id=job_id, response="yes", debug=True, ctx=full_context
        )
        resume_response: dict[str, Any] = resume_result.structuredContent
        assert resume_response["status"] == "success", (
            f"Expected success after single item approval, got: {resume_response}"
        )

        outputs = resume_response.get("outputs", {})
        assert outputs.get("all_items_processed") is True

    @pytest.mark.asyncio
    async def test_nested_workflow_feedback_iteration(self, full_context: MagicMock) -> None:
        """Test that a denial response completes the workflow via the denied branch."""
        exec_result = await execute_workflow(
            workflow="nested-workflow-in-foreach-parent",
            inputs={
                "work_items": [
                    {"name": "test_item", "value": "test_value"},
                ]
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert exec_response["status"] == "paused"
        job_id = exec_response["job_id"]

        resume_result = await resume_workflow(
            job_id=job_id, response="no", debug=True, ctx=full_context
        )
        resume_response: dict[str, Any] = resume_result.structuredContent
        assert resume_response["status"] == "success", (
            f"Expected success after denial, got: {resume_response}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
