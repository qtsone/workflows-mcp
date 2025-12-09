#!/usr/bin/env python3
"""Integration tests for for_each pause/resume functionality (ADR-010).

Tests pause/resume behavior for for_each blocks with Prompt executors,
including:
1. Sequential mode pause/resume (iteration by iteration)
2. Multiple pause/resume cycles
3. Parallel mode rejection (NotImplementedError)
4. Bracket notation access after resume
"""

import json
import os
from typing import Any

import pytest
from mcp.types import TextContent
from test_mcp_client import get_mcp_client


class TestForEachPauseResume:
    """Test for_each pause/resume functionality (ADR-010)."""

    @pytest.mark.asyncio
    async def test_sequential_for_each_pause_first_iteration(self) -> None:
        """Test that sequential for_each pauses on first Prompt iteration.

        Validates:
        - Workflow pauses when first iteration encounters Prompt
        - Pause prompt contains correct question
        - Job ID is returned for resume
        """
        async with get_mcp_client() as client:
            # Execute workflow until first pause
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "for-each-pause-sequential",
                    "inputs": {
                        "questions": {
                            "name": "What is your name?",
                            "email": "What is your email?",
                        }
                    },
                    "debug": True,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Verify workflow paused
            assert response["status"] == "paused", "Workflow should pause on first iteration"
            assert "job_id" in response, "Should return job_id for resume"
            assert "prompt" in response, "Should include pause prompt"
            assert response["prompt"] == "What is your name?"

    @pytest.mark.asyncio
    async def test_sequential_for_each_complete_pause_resume_cycle(self) -> None:
        """Test complete pause/resume cycle for sequential for_each.

        Validates:
        - First iteration pauses correctly
        - Resume continues with second iteration
        - Second iteration pauses correctly
        - Final resume completes workflow
        - All outputs accessible via bracket notation
        """
        async with get_mcp_client() as client:
            # Step 1: Execute until first pause
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "for-each-pause-sequential",
                    "inputs": {
                        "questions": {
                            "name": "What is your name?",
                            "email": "What is your email?",
                            "role": "What is your role?",
                        }
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            assert exec_response["status"] == "paused"
            job_id = exec_response["job_id"]
            assert exec_response["prompt"] == "What is your name?"

            # Step 2: Resume with first answer
            resume1_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "Alice",
                    "debug": True,
                },
            )

            resume1_content = resume1_result.content[0]
            assert isinstance(resume1_content, TextContent)
            resume1_response: dict[str, Any] = json.loads(resume1_content.text)

            # Should pause again on second iteration
            assert resume1_response["status"] == "paused", "Should pause on second iteration"
            assert resume1_response["prompt"] == "What is your email?"

            # Step 3: Resume with second answer
            resume2_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "alice@example.com",
                    "debug": True,
                },
            )

            resume2_content = resume2_result.content[0]
            assert isinstance(resume2_content, TextContent)
            resume2_response: dict[str, Any] = json.loads(resume2_content.text)

            # Should pause again on third iteration
            assert resume2_response["status"] == "paused", "Should pause on third iteration"
            assert resume2_response["prompt"] == "What is your role?"

            # Step 4: Resume with third answer (final)
            resume3_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "Engineer",
                    "debug": True,
                },
            )

            resume3_content = resume3_result.content[0]
            assert isinstance(resume3_content, TextContent)
            resume3_response: dict[str, Any] = json.loads(resume3_content.text)

            # Should complete successfully
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

            # Check bracket notation access works
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
    async def test_parallel_for_each_with_prompt_raises_not_implemented(self) -> None:
        """Test that parallel mode with Prompt raises NotImplementedError.

        Validates:
        - Parallel mode detects pause
        - Raises NotImplementedError with clear message
        - Error message guides user to use sequential mode
        """
        async with get_mcp_client() as client:
            # Execute workflow that should fail
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "for-each-pause-parallel-error",
                    "inputs": {
                        "questions": {
                            "q1": "Question 1?",
                            "q2": "Question 2?",
                        }
                    },
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Verify workflow failed with NotImplementedError
            assert response["status"] == "failure", "Workflow should fail in parallel mode"
            assert "error" in response
            error_msg = response["error"]
            assert "parallel mode" in error_msg.lower(), "Error should mention parallel mode"
            assert "sequential" in error_msg.lower(), "Error should suggest sequential mode"
            assert "Pause/resume is only supported with for_each_mode: sequential" in error_msg

    @pytest.mark.asyncio
    async def test_for_each_pause_checkpoint_contains_required_fields(self) -> None:
        """Test that for_each checkpoint contains all required fields for resume.

        Validates:
        - Checkpoint has type="for_each_iteration"
        - Contains iteration state (current, completed, remaining)
        - Contains executor configuration
        - Contains nested iteration checkpoint
        """
        async with get_mcp_client() as client:
            # Execute workflow until first pause
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "for-each-pause-sequential",
                    "inputs": {
                        "questions": {
                            "q1": "Question 1?",
                            "q2": "Question 2?",
                            "q3": "Question 3?",
                        }
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            assert exec_response["status"] == "paused"
            # Verify job_id is present
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
            assert pause_metadata["completed_iterations"] == []  # First iteration
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
    async def test_for_each_resume_updates_checkpoint_correctly(self) -> None:
        """Test that checkpoint updates correctly after each resume.

        Validates:
        - completed_iterations grows on each resume
        - remaining_iteration_keys shrinks on each resume
        - current_iteration_key/index advance correctly
        """
        async with get_mcp_client() as client:
            # Execute workflow
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "for-each-pause-sequential",
                    "inputs": {
                        "questions": {
                            "q1": "Q1",
                            "q2": "Q2",
                            "q3": "Q3",
                        }
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)
            job_id = exec_response["job_id"]

            # First resume
            resume1_result = await client.call_tool(
                "resume_workflow",
                arguments={"job_id": job_id, "response": "A1", "debug": True},
            )

            resume1_content = resume1_result.content[0]
            assert isinstance(resume1_content, TextContent)
            resume1_response: dict[str, Any] = json.loads(resume1_content.text)

            assert resume1_response["status"] == "paused"

            # Check checkpoint after first resume
            with open(resume1_response["logfile"], encoding="utf-8") as f:
                debug1 = json.load(f)

            pause1 = debug1["execution_state"]["pause_metadata"]
            assert pause1["current_iteration_key"] == "q2"
            assert pause1["current_iteration_index"] == 1
            assert pause1["completed_iterations"] == ["q1"]
            assert pause1["remaining_iteration_keys"] == ["q3"]

            # Second resume
            resume2_result = await client.call_tool(
                "resume_workflow",
                arguments={"job_id": job_id, "response": "A2", "debug": True},
            )

            resume2_content = resume2_result.content[0]
            if not isinstance(resume2_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(resume2_content)}")
            resume2_response: dict[str, Any] = json.loads(resume2_content.text)

            assert resume2_response["status"] == "paused"

            # Check checkpoint after second resume
            with open(resume2_response["logfile"], encoding="utf-8") as f:
                debug2 = json.load(f)

            pause2 = debug2["execution_state"]["pause_metadata"]
            assert pause2["current_iteration_key"] == "q3"
            assert pause2["current_iteration_index"] == 2
            assert pause2["completed_iterations"] == ["q1", "q2"]
            assert pause2["remaining_iteration_keys"] == []
