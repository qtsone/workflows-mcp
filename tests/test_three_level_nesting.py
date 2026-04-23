#!/usr/bin/env python3
"""Test 3-level nested workflow pause/resume.

This test reproduces a bug where:
1. Grandparent workflow has for_each calling parent workflows
2. Parent workflow has for_each calling child workflows AND blocks AFTER for_each
3. Child workflow has Prompt block that pauses
4. After child resumes, parent's remaining blocks (after for_each) don't run

Test Scenario:
    Grandparent
      ├── grandparent_setup (Shell)
      ├── process_parents (for_each → Parent)
      │     └── Parent
      │           ├── parent_setup (Shell)
      │           ├── process_children (for_each → Child)
      │           │     └── Child
      │           │           ├── child_setup (Shell)
      │           │           ├── child_approval (Prompt) ← PAUSES HERE
      │           │           └── child_complete (Shell)
      │           └── parent_complete (Shell) ← BUG: This doesn't run!
      └── grandparent_complete (Shell)

The bug causes parent_complete to never execute after child workflow resumes.

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


class TestThreeLevelNestedWorkflow:
    """Test pause/resume for 3-level nested workflows."""

    @pytest.mark.asyncio
    async def test_three_level_nesting_blocks_after_foreach_run(
        self, full_context: MagicMock
    ) -> None:
        """Test that blocks AFTER for_each in middle workflow run after child resumes.

        This is the core 3-level nesting bug reproduction test.
        """
        # Step 1: Start grandparent workflow with minimal input
        exec_result = await execute_workflow(
            workflow="three-level-grandparent",
            inputs={
                "parents": [
                    {
                        "name": "parent1",
                        "children": ["child1"],
                    }
                ]
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert exec_response["status"] == "paused", (
            f"Expected workflow to pause at child Prompt, got: {exec_response}"
        )
        assert "job_id" in exec_response
        job_id = exec_response["job_id"]

        prompt = exec_response.get("prompt", "")
        assert "child1" in prompt, f"Expected prompt to mention 'child1', got: {prompt}"

        # Step 2: Resume child with approval — should complete
        resume_result = await resume_workflow(
            job_id=job_id, response="approved", debug=True, ctx=full_context
        )
        resume_response: dict[str, Any] = resume_result.structuredContent
        assert resume_response["status"] == "success", (
            f"Expected workflow to complete after child approval, "
            f"got status: {resume_response.get('status')}. "
            f"Error: {resume_response.get('error')}"
        )

        outputs = resume_response.get("outputs", {})
        assert outputs.get("grandparent_setup_completed") is True, (
            "grandparent_setup should have run"
        )
        assert outputs.get("all_parents_completed") is True, "all parents should have completed"
        assert outputs.get("grandparent_complete_ran") is True, (
            "grandparent_complete should have run"
        )

    @pytest.mark.asyncio
    async def test_three_level_multiple_children_sequential(self, full_context: MagicMock) -> None:
        """Test 3-level nesting with multiple children (sequential pause/resume)."""
        exec_result = await execute_workflow(
            workflow="three-level-grandparent",
            inputs={
                "parents": [
                    {
                        "name": "parent1",
                        "children": ["childA", "childB"],
                    }
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
        assert "childA" in prompt, f"Expected childA in prompt, got: {prompt}"

        # Resume first child
        resume1_result = await resume_workflow(
            job_id=job_id, response="approved A", debug=True, ctx=full_context
        )
        resume1_response: dict[str, Any] = resume1_result.structuredContent
        assert resume1_response["status"] == "paused", (
            f"Expected pause for childB, got: {resume1_response}"
        )
        job_id_2 = resume1_response.get("job_id", job_id)
        prompt2 = resume1_response.get("prompt", "")
        assert "childB" in prompt2, f"Expected childB in prompt, got: {prompt2}"

        # Resume second child
        resume2_result = await resume_workflow(
            job_id=job_id_2, response="approved B", debug=True, ctx=full_context
        )
        resume2_response: dict[str, Any] = resume2_result.structuredContent
        assert resume2_response["status"] == "success", (
            f"Expected success after both children, got: {resume2_response}"
        )

        outputs = resume2_response.get("outputs", {})
        assert outputs.get("grandparent_complete_ran") is True

    @pytest.mark.asyncio
    async def test_three_level_multiple_parents_sequential(self, full_context: MagicMock) -> None:
        """Test 3-level nesting with multiple parents (each with children)."""
        exec_result = await execute_workflow(
            workflow="three-level-grandparent",
            inputs={
                "parents": [
                    {"name": "parent1", "children": ["child1"]},
                    {"name": "parent2", "children": ["child2"]},
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
        assert "child1" in prompt and "parent1" in prompt

        # Resume parent1's child
        resume1_result = await resume_workflow(
            job_id=job_id, response="approved 1", debug=True, ctx=full_context
        )
        resume1_response: dict[str, Any] = resume1_result.structuredContent
        assert resume1_response["status"] == "paused", (
            f"Expected pause for parent2's child, got: {resume1_response}"
        )
        job_id_2 = resume1_response.get("job_id", job_id)
        prompt2 = resume1_response.get("prompt", "")
        assert "child2" in prompt2 and "parent2" in prompt2, (
            f"Expected parent2/child2 in prompt, got: {prompt2}"
        )

        # Resume parent2's child
        resume2_result = await resume_workflow(
            job_id=job_id_2, response="approved 2", debug=True, ctx=full_context
        )
        resume2_response: dict[str, Any] = resume2_result.structuredContent
        assert resume2_response["status"] == "success", (
            f"Expected success, got: {resume2_response}"
        )

        outputs = resume2_response.get("outputs", {})
        assert outputs.get("grandparent_complete_ran") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
