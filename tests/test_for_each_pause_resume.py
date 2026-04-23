#!/usr/bin/env python3
"""Integration tests for for_each pause/resume functionality (ADR-010).

Tests pause/resume behavior for for_each blocks with Prompt executors,
including:
1. Sequential mode pause/resume (iteration by iteration)
2. Multiple pause/resume cycles
3. Parallel mode rejection (NotImplementedError)
4. Bracket notation access after resume

Transport: Direct MCP tool calls via mock AppContext (ADR-013).
"""

import json
import os
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


class TestForEachPauseResume:
    """Test for_each pause/resume functionality (ADR-010)."""

    @pytest.mark.asyncio
    async def test_sequential_for_each_pause_first_iteration(
        self, full_context: MagicMock
    ) -> None:
        """Test that sequential for_each pauses on first Prompt iteration."""
        result = await execute_workflow(
            workflow="for-each-pause-sequential",
            inputs={
                "questions": {
                    "name": "What is your name?",
                    "email": "What is your email?",
                }
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert response["status"] == "paused", "Workflow should pause on first iteration"
        assert "job_id" in response, "Should return job_id for resume"
        assert "prompt" in response, "Should include pause prompt"
        assert response["prompt"] == "What is your name?"

    @pytest.mark.asyncio
    async def test_sequential_for_each_complete_pause_resume_cycle(
        self, full_context: MagicMock
    ) -> None:
        """Test complete pause/resume cycle for sequential for_each."""
        # Step 1: Execute until first pause
        exec_result = await execute_workflow(
            workflow="for-each-pause-sequential",
            inputs={
                "questions": {
                    "name": "What is your name?",
                    "email": "What is your email?",
                    "role": "What is your role?",
                }
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert exec_response["status"] == "paused"
        job_id = exec_response["job_id"]
        assert exec_response["prompt"] == "What is your name?"

        # Step 2: Resume with first answer — should pause on second iteration
        resume1_result = await resume_workflow(
            job_id=job_id, response="Alice", debug=True, ctx=full_context
        )
        resume1_response: dict[str, Any] = resume1_result.structuredContent
        assert resume1_response["status"] == "paused", "Should pause on second iteration"
        assert resume1_response["prompt"] == "What is your email?"

        # Step 3: Resume with second answer — should pause on third iteration
        resume2_result = await resume_workflow(
            job_id=job_id, response="alice@example.com", debug=True, ctx=full_context
        )
        resume2_response: dict[str, Any] = resume2_result.structuredContent
        assert resume2_response["status"] == "paused", "Should pause on third iteration"
        assert resume2_response["prompt"] == "What is your role?"

        # Step 4: Resume with third answer (final) — should complete
        resume3_result = await resume_workflow(
            job_id=job_id, response="Engineer", debug=True, ctx=full_context
        )
        resume3_response: dict[str, Any] = resume3_result.structuredContent
        assert resume3_response["status"] == "success", (
            "Workflow should complete after all iterations"
        )

        # Read debug logfile to verify block execution details
        assert "logfile" in resume3_response, "Debug mode should include logfile"
        logfile_path = resume3_response["logfile"]
        assert os.path.exists(logfile_path), f"Logfile should exist at {logfile_path}"

        with open(logfile_path, encoding="utf-8") as f:
            debug_data: dict[str, Any] = json.load(f)

        # Verify all iterations completed with correct responses
        assert "gather_answers" in debug_data["blocks"]
        gather_answers = debug_data["blocks"]["gather_answers"]

        assert "name" in gather_answers["blocks"]
        assert gather_answers["blocks"]["name"]["outputs"]["response"] == "Alice"

        assert "email" in gather_answers["blocks"]
        assert gather_answers["blocks"]["email"]["outputs"]["response"] == "alice@example.com"

        assert "role" in gather_answers["blocks"]
        assert gather_answers["blocks"]["role"]["outputs"]["response"] == "Engineer"

        # Verify summarize block executed with correct outputs
        assert "summarize" in debug_data["blocks"]
        summary_output = debug_data["blocks"]["summarize"]["outputs"]["stdout"]
        assert "Name: Alice" in summary_output
        assert "Email: alice@example.com" in summary_output
        assert "Role: Engineer" in summary_output

    @pytest.mark.asyncio
    async def test_parallel_for_each_with_prompt_raises_not_implemented(
        self, full_context: MagicMock
    ) -> None:
        """Test that parallel mode with Prompt raises NotImplementedError."""
        result = await execute_workflow(
            workflow="for-each-pause-parallel-error",
            inputs={
                "questions": {
                    "q1": "Question 1?",
                    "q2": "Question 2?",
                }
            },
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert response["status"] == "failure", "Workflow should fail in parallel mode"
        assert "error" in response
        error_msg = response["error"]
        assert "parallel mode" in error_msg.lower(), "Error should mention parallel mode"
        assert "sequential" in error_msg.lower(), "Error should suggest sequential mode"
        assert "Pause/resume is only supported with for_each_mode: sequential" in error_msg

    @pytest.mark.asyncio
    async def test_for_each_pause_checkpoint_contains_required_fields(
        self, full_context: MagicMock
    ) -> None:
        """Test that for_each checkpoint contains all required fields for resume."""
        exec_result = await execute_workflow(
            workflow="for-each-pause-sequential",
            inputs={
                "questions": {
                    "q1": "Question 1?",
                    "q2": "Question 2?",
                    "q3": "Question 3?",
                }
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert exec_response["status"] == "paused"
        assert "job_id" in exec_response

        # Read debug logfile to inspect checkpoint structure
        assert "logfile" in exec_response
        logfile_path = exec_response["logfile"]

        with open(logfile_path, encoding="utf-8") as f:
            debug_data: dict[str, Any] = json.load(f)

        # Verify execution_state structure
        assert "execution_state" in debug_data
        execution_state = debug_data["execution_state"]

        # Verify pause_metadata structure
        assert "pause_metadata" in execution_state
        pause_metadata = execution_state["pause_metadata"]

        # Validate required checkpoint fields
        assert pause_metadata["type"] == "for_each_iteration"
        assert pause_metadata["for_each_block_id"] == "gather_answers"
        assert pause_metadata["current_iteration_key"] == "q1"
        assert pause_metadata["current_iteration_index"] == 0
        assert pause_metadata["completed_iterations"] == []
        assert set(pause_metadata["remaining_iteration_keys"]) == {"q2", "q3"}
        assert pause_metadata["all_iterations"] == {
            "q1": "Question 1?",
            "q2": "Question 2?",
            "q3": "Question 3?",
        }
        assert pause_metadata["executor_type"] == "Prompt"
        assert "inputs_template" in pause_metadata
        assert pause_metadata["mode"] == "sequential"
        assert "paused_iteration_checkpoint" in pause_metadata

    @pytest.mark.asyncio
    async def test_for_each_resume_updates_checkpoint_correctly(
        self, full_context: MagicMock
    ) -> None:
        """Test that checkpoint updates correctly after each resume."""
        exec_result = await execute_workflow(
            workflow="for-each-pause-sequential",
            inputs={
                "questions": {
                    "q1": "Q1",
                    "q2": "Q2",
                    "q3": "Q3",
                }
            },
            debug=True,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        job_id = exec_response["job_id"]

        # First resume
        resume1_result = await resume_workflow(
            job_id=job_id, response="A1", debug=True, ctx=full_context
        )
        resume1_response: dict[str, Any] = resume1_result.structuredContent
        assert resume1_response["status"] == "paused"

        with open(resume1_response["logfile"], encoding="utf-8") as f:
            debug1 = json.load(f)

        pause1 = debug1["execution_state"]["pause_metadata"]
        assert pause1["current_iteration_key"] == "q2"
        assert pause1["current_iteration_index"] == 1
        assert pause1["completed_iterations"] == ["q1"]
        assert pause1["remaining_iteration_keys"] == ["q3"]

        # Second resume
        resume2_result = await resume_workflow(
            job_id=job_id, response="A2", debug=True, ctx=full_context
        )
        resume2_response: dict[str, Any] = resume2_result.structuredContent
        assert resume2_response["status"] == "paused"

        with open(resume2_response["logfile"], encoding="utf-8") as f:
            debug2 = json.load(f)

        pause2 = debug2["execution_state"]["pause_metadata"]
        assert pause2["current_iteration_key"] == "q3"
        assert pause2["current_iteration_index"] == 2
        assert pause2["completed_iterations"] == ["q1", "q2"]
        assert pause2["remaining_iteration_keys"] == []
