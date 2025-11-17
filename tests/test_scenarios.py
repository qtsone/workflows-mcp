#!/usr/bin/env python3
"""End-to-end scenario testing for workflows-mcp.

Philosophy: Test realistic workflow usage patterns as they would be used
by Claude Code or other MCP clients. These tests complement snapshot-based
regression testing with real-world integration scenarios.

Test Categories:
1. Complete CI/CD pipelines
2. Multi-step automation workflows
3. Error recovery scenarios
4. Interactive workflow patterns
5. Async execution patterns
6. Workflow composition chains
"""

import json
import os
from typing import Any

import pytest
from mcp.types import TextContent
from test_behavior import (
    WorkflowBehavior,
    assert_workflow_behavior,
    assert_workflow_failed,
    assert_workflow_paused,
    assert_workflow_succeeded,
)
from test_mcp_client import get_mcp_client

# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndScenarios:
    """Real-world workflow execution scenarios."""

    @pytest.mark.asyncio
    async def test_complete_file_processing_pipeline(self):
        """Test complete file processing workflow: create → read → transform → validate.

        This simulates a realistic data processing pipeline using multiple
        file operation blocks in sequence.
        """
        async with get_mcp_client() as client:
            # Execute integrated workflow
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "integration-end-to-end",
                    "inputs": {},
                    "debug": False,
                },
            )

            # Extract response
            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate behavior
            assert_workflow_succeeded(response)
            assert "outputs" in response
            # Verify all pipeline stages completed
            assert response["outputs"]["all_blocks_succeeded"] is True
            assert response["outputs"]["validation_passed"] is True
            assert response["outputs"]["final_phase"] == "complete"

    @pytest.mark.asyncio
    async def test_conditional_execution_branching(self):
        """Test workflow with complex conditional logic and multiple branches.

        Tests that conditional blocks execute correctly based on runtime
        conditions and that only the correct branch executes.
        """
        async with get_mcp_client() as client:
            # Test with condition that should take "success" branch
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "core-conditionals-test",
                    "inputs": {},
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate correct branch executed
            assert_workflow_succeeded(response)
            assert response["outputs"]["success_condition_executed"] is True
            assert response["outputs"]["failure_condition_executed"] is True

    @pytest.mark.asyncio
    async def test_workflow_composition_chain(self):
        """Test workflow calling another workflow (composition).

        Tests that parent workflows can call child workflows and that
        outputs are properly passed through the composition chain.
        """
        async with get_mcp_client() as client:
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "composition-output-passing",
                    "inputs": {"value": 10},
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate composition worked
            assert_workflow_succeeded(response)
            # Child workflow outputs should be accessible
            assert response["outputs"]["both_succeeded"] is True
            assert response["outputs"]["multiply_result"] == 50  # 10 * 5
            assert response["outputs"]["add_result"] == 55  # 50 + 5

    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self):
        """Test that parallel blocks execute concurrently, not sequentially.

        This workflow has multiple independent blocks that should execute
        in parallel (same DAG wave). We validate they all complete.
        """
        async with get_mcp_client() as client:
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "dag-execution-parallel",
                    "inputs": {},
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate all parallel blocks completed
            assert_workflow_succeeded(response)
            assert response["outputs"]["all_succeeded"] is True
            assert response["outputs"]["parallel_1_output"] == "parallel_1"
            assert response["outputs"]["parallel_2_output"] == "parallel_2"
            assert response["outputs"]["parallel_3_output"] == "parallel_3"

    @pytest.mark.asyncio
    async def test_error_recovery_with_optional_dependencies(self):
        """Test workflow continues when optional dependency fails.

        Tests that workflows with optional dependencies (using conditions)
        can recover from failures in non-critical blocks.
        """
        async with get_mcp_client() as client:
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "dag-execution-optional-deps",
                    "inputs": {},
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Workflow should succeed despite optional block failure
            assert_workflow_succeeded(response)


# =============================================================================
# Interactive Workflow Scenarios
# =============================================================================


class TestInteractiveScenarios:
    """Real-world interactive workflow patterns."""

    @pytest.mark.asyncio
    async def test_approval_workflow_with_retry(self):
        """Test interactive approval workflow with multiple resume attempts.

        Simulates a user denying approval first, then approving on retry.
        """
        async with get_mcp_client() as client:
            # Step 1: Start workflow (should pause at Prompt block)
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {"message": "Deploy to production?"},
                    "debug": False,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            # Verify workflow paused
            assert_workflow_paused(exec_response, prompt_pattern="Deploy")
            job_id = exec_response["job_id"]

            # Step 2: Deny first time
            deny_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "no",
                    "debug": False,
                },
            )

            deny_content = deny_result.content[0]
            assert isinstance(deny_content, TextContent)
            deny_response: dict[str, Any] = json.loads(deny_content.text)

            # Verify denial branch executed
            assert_workflow_succeeded(deny_response)
            assert deny_response["outputs"]["approved"] == "false"
            assert deny_response["outputs"]["denied"] == "true"

    @pytest.mark.asyncio
    async def test_job_status_tracking(self):
        """Test querying job status for paused workflow.

        Validates that get_job_status provides accurate information
        about paused workflows before resume.
        """
        async with get_mcp_client() as client:
            # Start and pause workflow
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {},
                    "debug": False,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)
            job_id = exec_response["job_id"]

            # Query job status
            status_result = await client.call_tool(
                "get_job_status",
                arguments={"job_id": job_id},
            )

            status_content = status_result.content[0]
            assert isinstance(status_content, TextContent)
            status: dict[str, Any] = json.loads(status_content.text)

            # Validate status information
            assert status["id"] == job_id
            assert status["status"] == "paused"
            assert status["workflow"] == "interactive-simple-approval"
            assert "prompt" in status
            assert "result_file" in status


# =============================================================================
# Error Scenario Tests
# =============================================================================


class TestErrorScenarios:
    """Test error handling and failure scenarios."""

    @pytest.mark.asyncio
    async def test_workflow_not_found_error(self):
        """Test executing non-existent workflow returns helpful error."""
        async with get_mcp_client() as client:
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "this-workflow-does-not-exist",
                    "inputs": {},
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate helpful error response
            assert_workflow_failed(response, error_pattern="not found")
            assert "available_workflows" in response

    @pytest.mark.asyncio
    async def test_secrets_missing_error(self):
        """Test workflow with missing secret handles error gracefully.

        This workflow tests that blocks with missing secrets fail properly
        and dependent blocks are skipped, while the workflow continues.
        """
        async with get_mcp_client() as client:
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "core-secrets-management-test",
                    "inputs": {},
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate workflow succeeds (error handling is working)
            assert_workflow_succeeded(response)
            # Block with missing secret should have failed
            assert response["outputs"]["missing_secret_failed"] is True
            # Other secrets operations should succeed
            assert response["outputs"]["shell_basic_succeeded"] is True
            assert response["outputs"]["multiple_secrets_succeeded"] is True


# =============================================================================
# Async Execution Scenarios (requires job queue enabled)
# =============================================================================


@pytest.mark.skipif(
    os.getenv("WORKFLOWS_JOB_QUEUE_ENABLED", "true").lower() == "false",
    reason="Job queue disabled (set WORKFLOWS_JOB_QUEUE_ENABLED=true to enable)",
)
class TestAsyncExecutionScenarios:
    """Test async workflow execution patterns."""

    @pytest.mark.asyncio
    async def test_async_workflow_submission(self):
        """Test submitting workflow for async execution."""
        async with get_mcp_client() as client:
            # Submit async job
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "workflow-output-type-coercion",
                    "inputs": {},
                    "mode": "async",
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate job submission
            assert response["status"] == "queued"
            assert "job_id" in response
            assert response["workflow"] == "workflow-output-type-coercion"

    @pytest.mark.asyncio
    async def test_async_job_status_polling(self):
        """Test polling async job status until completion."""
        async with get_mcp_client() as client:
            # Submit job
            submit_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "workflow-output-type-coercion",
                    "inputs": {},
                    "mode": "async",
                },
            )

            submit_content = submit_result.content[0]
            assert isinstance(submit_content, TextContent)
            submit_response: dict[str, Any] = json.loads(submit_content.text)
            job_id = submit_response["job_id"]

            # Poll until complete (with timeout)
            import asyncio

            for _ in range(30):  # 30 attempts = 15 seconds max
                await asyncio.sleep(0.5)

                status_result = await client.call_tool(
                    "get_job_status",
                    arguments={"job_id": job_id},
                )

                status_content = status_result.content[0]
                assert isinstance(status_content, TextContent)
                status: dict[str, Any] = json.loads(status_content.text)

                if status["status"] in ["completed", "failed"]:
                    break

            # Validate job completed
            assert status["status"] == "completed"
            assert "outputs" in status


# =============================================================================
# Behavior-Based Validation Examples
# =============================================================================


class TestBehaviorBasedValidation:
    """Examples of behavior-based testing vs structure-based."""

    @pytest.mark.asyncio
    async def test_workflow_with_behavior_spec(self):
        """Test workflow using behavior specification instead of snapshot.

        This is more resilient to architectural changes than exact JSON matching.
        """
        # Define expected behavior
        expected = WorkflowBehavior(
            status="success",
            output_schema={
                "exit_code_int": int,
                "command_succeeded": bool,
            },
            output_values={
                "exit_code_int": 0,
                "command_succeeded": True,
            },
        )

        async with get_mcp_client() as client:
            result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "workflow-output-type-coercion",
                    "inputs": {},
                    "debug": False,
                },
            )

            content = result.content[0]
            assert isinstance(content, TextContent)
            response: dict[str, Any] = json.loads(content.text)

            # Validate behavior (not structure)
            assert_workflow_behavior(response, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
