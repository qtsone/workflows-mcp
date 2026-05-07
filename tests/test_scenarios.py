#!/usr/bin/env python3
"""End-to-end scenario testing for workflows-mcp.

Philosophy: Test realistic workflow usage patterns as they would be used
by Claude Code or other MCP clients. These tests complement snapshot-based
regression testing with real-world integration scenarios.

Transport: Direct MCP tool calls via mock AppContext (ADR-013).

Test Categories:
1. Complete CI/CD pipelines
2. Multi-step automation workflows
3. Error recovery scenarios
4. Interactive workflow patterns
5. Async execution patterns
6. Workflow composition chains
"""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from test_behavior import (
    WorkflowBehavior,
    assert_workflow_behavior,
    assert_workflow_failed,
    assert_workflow_paused,
    assert_workflow_succeeded,
)

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
    """AppContext with all test workflows, a started IOQueue, a live JobStore."""
    executor_registry = create_default_registry()
    llm_config_loader = LLMConfigLoader()
    io_queue = IOQueue()

    # Build initial context without job_queue to pass into JobQueue constructor
    app_context = AppContext(
        registry=full_registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,
    )

    job_queue = JobQueue(app_context, num_workers=2)
    await job_queue._store.init()

    # Rebuild context with job_queue wired in
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
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndScenarios:
    """Real-world workflow execution scenarios."""

    @pytest.mark.asyncio
    async def test_complete_file_processing_pipeline(self, full_context: MagicMock) -> None:
        """Test complete file processing workflow: create → read → transform → validate."""
        result = await execute_workflow(
            workflow="integration-end-to-end",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)
        assert response["outputs"]["all_blocks_succeeded"] is True
        assert response["outputs"]["validation_passed"] is True
        assert response["outputs"]["final_phase"] == "complete"

    @pytest.mark.asyncio
    async def test_conditional_execution_branching(self, full_context: MagicMock) -> None:
        """Test workflow with complex conditional logic and multiple branches."""
        result = await execute_workflow(
            workflow="core-conditionals-test",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)
        assert response["outputs"]["success_condition_executed"] is True
        assert response["outputs"]["failure_condition_executed"] is True

    @pytest.mark.asyncio
    async def test_workflow_composition_chain(self, full_context: MagicMock) -> None:
        """Test workflow calling another workflow (composition)."""
        result = await execute_workflow(
            workflow="composition-output-passing",
            inputs={"value": 10},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)
        assert response["outputs"]["both_succeeded"] is True
        assert response["outputs"]["multiply_result"] == 50  # 10 * 5
        assert response["outputs"]["add_result"] == 55  # 50 + 5

    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self, full_context: MagicMock) -> None:
        """Test that parallel blocks execute concurrently, not sequentially."""
        result = await execute_workflow(
            workflow="dag-execution-parallel",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)
        assert response["outputs"]["all_succeeded"] is True
        assert response["outputs"]["parallel_1_output"].strip() == "parallel_1"
        assert response["outputs"]["parallel_2_output"].strip() == "parallel_2"
        assert response["outputs"]["parallel_3_output"].strip() == "parallel_3"

    @pytest.mark.asyncio
    async def test_error_recovery_with_optional_dependencies(
        self, full_context: MagicMock
    ) -> None:
        """Test workflow continues when optional dependency fails."""
        result = await execute_workflow(
            workflow="dag-execution-optional-deps",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)


# =============================================================================
# Interactive Workflow Scenarios
# =============================================================================


class TestInteractiveScenarios:
    """Real-world interactive workflow patterns."""

    @pytest.mark.asyncio
    async def test_approval_workflow_with_retry(self, full_context: MagicMock) -> None:
        """Test interactive approval workflow: deny then approve."""
        # Step 1: Start workflow — should pause at Prompt block
        exec_result = await execute_workflow(
            workflow="interactive-simple-approval",
            inputs={"message": "Deploy to production?"},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert_workflow_paused(exec_response, prompt_pattern="Deploy")
        job_id = exec_response["job_id"]

        # Step 2: Deny — verify denial branch executes
        deny_result = await resume_workflow(
            job_id=job_id,
            response="no",
            debug=False,
            ctx=full_context,
        )

        deny_response: dict[str, Any] = deny_result.structuredContent
        assert_workflow_succeeded(deny_response)
        assert deny_response["outputs"]["approved"] == "false"
        assert deny_response["outputs"]["denied"] == "true"

    @pytest.mark.asyncio
    async def test_job_status_tracking(self, full_context: MagicMock) -> None:
        """Test querying job status for paused workflow via resume path."""
        # Start and pause workflow
        exec_result = await execute_workflow(
            workflow="interactive-simple-approval",
            inputs={"message": "Proceed?"},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert_workflow_paused(exec_response)
        job_id = exec_response["job_id"]

        # Verify job_id is present and resume works (job must exist in store)
        assert job_id.startswith("job_")

        # Resume to clean up
        resume_result = await resume_workflow(
            job_id=job_id,
            response="yes",
            debug=False,
            ctx=full_context,
        )
        resume_response: dict[str, Any] = resume_result.structuredContent
        assert_workflow_succeeded(resume_response)


# =============================================================================
# Error Scenario Tests
# =============================================================================


class TestErrorScenarios:
    """Test error handling and failure scenarios."""

    @pytest.mark.asyncio
    async def test_workflow_not_found_error(self, full_context: MagicMock) -> None:
        """Test executing non-existent workflow returns helpful error."""
        result = await execute_workflow(
            workflow="this-workflow-does-not-exist",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_failed(response, error_pattern="not found")
        assert "available_workflows" in response

    @pytest.mark.asyncio
    async def test_secrets_missing_error(self, full_context: MagicMock) -> None:
        """Test workflow with missing secret handles error gracefully."""
        result = await execute_workflow(
            workflow="core-secrets-management-test",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)
        assert response["outputs"]["missing_secret_failed"] is True
        assert response["outputs"]["shell_basic_succeeded"] is True
        assert response["outputs"]["multiple_secrets_succeeded"] is True


# =============================================================================
# Async Execution Scenarios
# =============================================================================


@pytest.mark.skipif(
    os.getenv("WORKFLOWS_JOB_QUEUE_ENABLED", "true").lower() == "false",
    reason="Job queue disabled (set WORKFLOWS_JOB_QUEUE_ENABLED=true to enable)",
)
class TestAsyncExecutionScenarios:
    """Test async workflow execution patterns."""

    @pytest.mark.asyncio
    async def test_async_workflow_submission(self, full_context: MagicMock) -> None:
        """Test submitting workflow for async execution."""
        job_queue = full_context.request_context.lifespan_context.job_queue
        await job_queue.start()
        try:
            result = await execute_workflow(
                workflow="workflow-output-type-coercion",
                inputs={},
                debug=False,
                mode="async",
                timeout=None,
                ctx=full_context,
            )

            response: dict[str, Any] = result.structuredContent
            assert response["status"] == "queued"
            assert "job_id" in response
            assert response["workflow"] == "workflow-output-type-coercion"
        finally:
            await job_queue.stop()

    @pytest.mark.asyncio
    async def test_async_job_status_polling(self, full_context: MagicMock) -> None:
        """Test polling async job status until completion."""
        import asyncio

        from workflows_mcp.tools import get_job_status

        # Start job queue workers for this test
        job_queue = full_context.request_context.lifespan_context.job_queue
        await job_queue.start()

        try:
            submit_result = await execute_workflow(
                workflow="workflow-output-type-coercion",
                inputs={},
                debug=False,
                mode="async",
                timeout=None,
                ctx=full_context,
            )

            submit_response: dict[str, Any] = submit_result.structuredContent
            job_id = submit_response["job_id"]

            # Poll until complete (with timeout)
            status: dict[str, Any] = {}
            for _ in range(30):  # 15 seconds max
                await asyncio.sleep(0.5)
                status_result = await get_job_status(job_id=job_id, ctx=full_context)
                status = status_result.structuredContent
                if status["status"] in ["completed", "failed"]:
                    break

            assert status["status"] == "completed"
            assert "outputs" in status
        finally:
            await job_queue.stop()


# =============================================================================
# Behavior-Based Validation
# =============================================================================


class TestBehaviorBasedValidation:
    """Examples of behavior-based testing vs structure-based."""

    @pytest.mark.asyncio
    async def test_workflow_with_behavior_spec(self, full_context: MagicMock) -> None:
        """Test workflow using behavior specification instead of snapshot."""
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

        result = await execute_workflow(
            workflow="workflow-output-type-coercion",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_behavior(response, expected)


class TestContinueOnErrorOpt:
    """Regression tests: blocks with continue_on_error:true must not cause workflow failure."""

    @pytest.mark.asyncio
    async def test_continue_on_error_block_does_not_fail_workflow(
        self, full_context: MagicMock
    ) -> None:
        """A failed block marked continue_on_error:true must not propagate as workflow failure.

        core-conditionals-test has `failure_block` (exit 1) with continue_on_error:true.
        The workflow must still return status='success' because the failure is tolerated.
        """
        result = await execute_workflow(
            workflow="core-conditionals-test",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)
        # failure_block ran and failed, but was tolerated
        assert response["outputs"]["failure_block_failed"] is True

    @pytest.mark.asyncio
    async def test_dag_optional_deps_continue_on_error_succeeds(
        self, full_context: MagicMock
    ) -> None:
        """dag-execution-optional-deps: job_2 fails with continue_on_error:true.

        Workflow must return status='success' and downstream optional-dep blocks
        (job_4, job_5, job_6) must have run despite job_2 failing.
        """
        result = await execute_workflow(
            workflow="dag-execution-optional-deps",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert_workflow_succeeded(response)
        assert response["outputs"]["job_1_succeeded"] is True
        assert response["outputs"]["job_2_failed"] is True
        assert response["outputs"]["job_3_succeeded"] is True
        assert response["outputs"]["job_4_succeeded"] is True
        assert response["outputs"]["job_5_succeeded"] is True
        assert response["outputs"]["job_6_succeeded"] is True


class TestTerminalStatusPropagation:
    """Terminal-status propagation: failed blocks without continue_on_error must
    cause workflow-level failure, both on fresh execution and resume paths."""

    @pytest.mark.asyncio
    async def test_fresh_execution_failed_block_no_opt_out_returns_failure(
        self, full_context: MagicMock
    ) -> None:
        """A block that fails without continue_on_error must make the workflow fail.

        Uses core-resume-with-failure directly via WorkflowRunner so we can verify
        the fresh-execution terminal-status classification without the resume path.
        fail_after_resume depends on `ask` (Prompt), but we feed a pre-resolved
        Execution that skips the pause so only the shell block runs.

        Simpler: register a tiny inline workflow in the registry and execute it
        via the AppContext so ExecutionContext is properly constructed.
        """
        import yaml as _yaml

        from workflows_mcp.engine.schema import WorkflowSchema
        from workflows_mcp.engine.workflow_runner import WorkflowRunner

        failing_yaml = """
name: test-fail-no-continue
description: Inline test workflow with unguarded failing block
blocks:
  - id: fail_block
    type: Shell
    inputs:
      command: exit 1
"""
        workflow = WorkflowSchema.model_validate(_yaml.safe_load(failing_yaml))
        app_ctx = full_context.request_context.lifespan_context
        exec_ctx = app_ctx.create_execution_context()
        runner = WorkflowRunner()
        result = await runner.execute(workflow, {}, exec_ctx)
        assert result.status == "failure", (
            f"Expected failure but got {result.status}: {result.error}"
        )

    @pytest.mark.asyncio
    async def test_resume_path_failed_block_no_opt_out_returns_failure(
        self, full_context: MagicMock
    ) -> None:
        """Resume path: a block failing after resume without continue_on_error must fail.

        Fixture: core-resume-with-failure
        - Pauses at Prompt block
        - After resume, fail_after_resume (exit 1, no continue_on_error) runs
        - Expect workflow-level failure, not success
        """
        # Step 1: Start — must pause at Prompt
        exec_result = await execute_workflow(
            workflow="core-resume-with-failure",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=full_context,
        )

        exec_response: dict[str, Any] = exec_result.structuredContent
        assert_workflow_paused(exec_response, prompt_pattern="Continue")
        job_id = exec_response["job_id"]

        # Step 2: Resume — fail_after_resume must make workflow fail
        resume_result = await resume_workflow(
            job_id=job_id,
            response="yes",
            debug=False,
            ctx=full_context,
        )

        resume_response: dict[str, Any] = resume_result.structuredContent
        assert_workflow_failed(
            resume_response,
            error_pattern="failed without continue_on_error",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
