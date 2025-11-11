#!/usr/bin/env python3
"""Comprehensive MCP server testing.

Philosophy: Test the product AS IT'S MEANT TO BE USED - as an MCP server
communicating via the MCP protocol over stdio, exactly as Claude Code would.

This test suite validates:

1. **MCP Server Health & Protocol Compliance**
   - Server initialization and MCP protocol handshake
   - Tool discovery and capability exposition
   - Response format consistency

2. **MCP Tool Functionality**
   - Workflow execution (execute_workflow, execute_inline_workflow)
   - Workflow discovery (list_workflows, get_workflow_info)
   - Schema validation (get_workflow_schema, validate_workflow_yaml)
   - Checkpoint management (resume_workflow, list_checkpoints, etc.)
   - Input validation and error handling
   - Response format variations (minimal/detailed, json/markdown)

3. **Snapshot-Based Workflow Validation**
   - Real workflow execution against golden snapshots
   - Regression detection across code changes
   - Coverage validation (all workflows have snapshots)

Test Approach:
- Test via MCP protocol over stdio (real-world integration)
- Pydantic input validation throughout
- Actionable error messages
- Comprehensive edge case coverage
- Snapshot-based regression testing

Snapshot Management:
- Generate snapshots: `uv run python tests/generate_snapshots.py`
- Snapshots location: tests/snapshots/{workflow_name}.json
- Commit snapshots to git as source of truth
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from test_utils import format_diff, normalize_dynamic_fields

from workflows_mcp.context import AppContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.schema import WorkflowSchema
from workflows_mcp.tools import (
    execute_inline_workflow,
    execute_workflow,
    get_workflow_info,
    get_workflow_schema,
    list_workflows,
    validate_workflow_yaml,
)

# Test configuration
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
WORKFLOWS_DIR = Path(__file__).parent / "workflows"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_context():
    """Create mock MCP context with AppContext for unit testing MCP tools.

    This fixture creates an isolated test environment with:
    - WorkflowRegistry with test workflows
    - ExecutorRegistry with Shell executor
    - LLMConfigLoader with built-in defaults

    Returns:
        Mock context object with request_context.lifespan_context structure
    """
    registry = WorkflowRegistry()
    executor_registry = create_default_registry()
    llm_config_loader = LLMConfigLoader()

    # Register test workflow with Shell executor
    test_workflow = WorkflowSchema(
        name="test-workflow",
        description="Test workflow for unit tests",
        blocks=[
            {
                "id": "step1",
                "type": "Shell",
                "inputs": {"command": "echo 'Hello {{inputs.message}}'"},
            }
        ],
        inputs={
            "message": {
                "type": "str",
                "description": "Message to echo",
                "default": "World",
            }
        },
        outputs={"result": {"value": "{{blocks.step1.outputs.stdout}}"}},
    )
    registry.register(test_workflow)

    # Create IO queue (not started, fine for unit tests)
    io_queue = IOQueue()

    # Create mock context matching MCP server structure
    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
    )

    mock_ctx = MagicMock()
    mock_ctx.request_context.lifespan_context = app_context

    return mock_ctx


@asynccontextmanager
async def get_mcp_client() -> AsyncIterator[ClientSession]:
    """Context manager providing MCP client connected to server via stdio.

    Mimics exactly how Claude Code connects to MCP servers, ensuring
    we test the real integration path using MCP protocol over stdio.

    Configuration:
    - Uses WORKFLOWS_TEMPLATE_PATHS to point to tests/workflows
    - Sets WORKFLOWS_LOG_LEVEL=WARNING to suppress INFO logs
    - Executes server via `python -m workflows_mcp`

    Yields:
        ClientSession: Initialized MCP client session for tool calls
    """
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "workflows_mcp"],
        env={
            **os.environ,
            "WORKFLOWS_TEMPLATE_PATHS": str(WORKFLOWS_DIR),
            "WORKFLOWS_LOG_LEVEL": "WARNING",  # Suppress INFO logs
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# =============================================================================
# Test Classes - Part 1: MCP Server Health & Protocol Compliance
# =============================================================================


class TestMCPServerHealth:
    """MCP server health and protocol compliance validation.

    Smoke tests that validate the server is functioning correctly
    at a protocol level before running detailed tool tests.
    """

    async def test_server_initializes_successfully(self) -> None:
        """Test MCP server starts and responds to initialization."""
        async with get_mcp_client() as client:
            assert client is not None, "MCP client session should be established"

    async def test_server_exposes_all_required_tools(self) -> None:
        """Test server exposes all required workflow execution tools."""
        async with get_mcp_client() as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools.tools}

            required_tools = {
                "execute_workflow",
                "execute_inline_workflow",
                "list_workflows",
                "get_workflow_info",
                "get_workflow_schema",
                "validate_workflow_yaml",
                "resume_workflow",
                # Job queue tools (async execution)
                "get_job_status",
                "cancel_job",
                "list_jobs",
                "get_queue_stats",
            }

            missing = required_tools - tool_names
            if missing:
                pytest.fail(
                    f"Required MCP tools missing: {sorted(missing)}\n"
                    f"Available tools: {sorted(tool_names)}"
                )

    async def test_workflow_discovery_returns_test_workflows(self) -> None:
        """Test list_workflows discovers test workflows via MCP."""
        async with get_mcp_client() as client:
            result = await client.call_tool(
                "list_workflows", arguments={"tags": ["test"], "format": "json"}
            )

            content = result.content[0]
            if not isinstance(content, TextContent):
                pytest.fail(f"Expected TextContent, got {type(content)}")

            workflows: list[str] = json.loads(content.text)

            assert isinstance(workflows, list), "list_workflows should return a list"
            assert len(workflows) > 0, "No test workflows discovered"


# =============================================================================
# Test Classes - Part 2: Workflow Execution
# =============================================================================


class TestWorkflowExecution:
    """Workflow execution tests (execute_workflow, execute_inline_workflow).

    Tests both registered workflow execution and inline YAML execution,
    covering success cases, error handling, and response format variations.
    """

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, mock_context) -> None:
        """Test successful workflow execution."""
        result = await execute_workflow(
            workflow="test-workflow",
            inputs={"message": "Test"},
            debug=False,
            ctx=mock_context,
        )

        assert result["status"] == "success"
        assert "outputs" in result
        # Minimal format excludes blocks/metadata
        assert "blocks" not in result
        assert "metadata" not in result

    @pytest.mark.asyncio
    async def test_execute_workflow_debug_mode(self, mock_context) -> None:
        """Test workflow execution with debug mode (writes logfile)."""
        result = await execute_workflow(
            workflow="test-workflow",
            inputs={"message": "Test"},
            debug=True,
            ctx=mock_context,
        )

        assert result["status"] == "success"
        assert "outputs" in result
        # Debug mode includes logfile path instead of inline blocks/metadata
        assert "logfile" in result
        assert result["logfile"].startswith("/tmp/")
        assert result["logfile"].endswith(".json")

    @pytest.mark.asyncio
    async def test_execute_workflow_not_found(self, mock_context) -> None:
        """Test execute_workflow with non-existent workflow."""
        result = await execute_workflow(workflow="non-existent-workflow", ctx=mock_context)

        assert result["status"] == "failure"
        assert "not found" in result["error"].lower()
        # Should provide actionable guidance
        assert "available_workflows" in result
        assert isinstance(result["available_workflows"], list)

    @pytest.mark.asyncio
    async def test_execute_workflow_missing_required_inputs(self, mock_context) -> None:
        """Test execute_workflow with missing required inputs."""
        registry = mock_context.request_context.lifespan_context.registry

        required_workflow = WorkflowSchema(
            name="test-required-inputs",
            description="Workflow with required inputs",
            blocks=[
                {
                    "id": "echo1",
                    "type": "Shell",
                    "inputs": {"command": "echo {{inputs.required_param}}"},
                }
            ],
            inputs={
                "required_param": {
                    "type": "str",
                    "description": "Required parameter",
                    "required": True,
                }
            },
        )
        registry.register(required_workflow)

        result = await execute_workflow(
            workflow="test-required-inputs", inputs={}, ctx=mock_context
        )

        assert result["status"] == "failure"
        # Error about missing inputs or variable resolution
        assert "required" in result["error"].lower() or "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_workflow_with_custom_inputs(self, mock_context) -> None:
        """Test execute_workflow with runtime inputs."""
        result = await execute_workflow(
            workflow="test-workflow",
            inputs={"message": "CustomMessage"},
            debug=True,
            ctx=mock_context,
        )

        assert result["status"] == "success"
        assert "logfile" in result
        assert result["logfile"].startswith("/tmp/")

    # Inline workflow execution tests

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_success(self, mock_context) -> None:
        """Test successful inline workflow execution."""
        workflow_yaml = """
name: inline-test
description: Inline workflow test
blocks:
  - id: echo
    type: Shell
    inputs:
      command: echo 'Inline test'
outputs:
  result:
    value: "{{blocks.echo.outputs.stdout}}"
"""

        result = await execute_inline_workflow(workflow_yaml=workflow_yaml, ctx=mock_context)

        assert result["status"] == "success"
        assert "outputs" in result

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_empty_yaml(self, mock_context) -> None:
        """Test execute_inline_workflow with empty YAML."""
        result = await execute_inline_workflow(workflow_yaml="# empty yaml\n", ctx=mock_context)

        assert result["status"] == "failure"
        # Empty YAML parsed as None
        assert "dictionary" in result["error"].lower() or "nonetype" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_invalid_yaml(self, mock_context) -> None:
        """Test execute_inline_workflow with invalid YAML syntax."""
        result = await execute_inline_workflow(workflow_yaml="invalid: [unclosed", ctx=mock_context)

        assert result["status"] == "failure"
        assert "parse" in result["error"].lower() or "yaml" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_missing_required_fields(self, mock_context) -> None:
        """Test execute_inline_workflow with missing required fields."""
        workflow_yaml = """
name: incomplete-workflow
description: Missing blocks field
"""

        result = await execute_inline_workflow(workflow_yaml=workflow_yaml, ctx=mock_context)

        assert result["status"] == "failure"


# =============================================================================
# Test Classes - Part 3: Workflow Discovery & Metadata
# =============================================================================


class TestWorkflowDiscovery:
    """Workflow discovery, introspection, and validation tests.

    Tests workflow listing, metadata retrieval, schema generation,
    and YAML validation - all the tools for understanding and
    validating workflows before execution.
    """

    # Workflow listing tests

    @pytest.mark.asyncio
    async def test_list_workflows_json_format(self, mock_context) -> None:
        """Test list_workflows returns JSON list."""
        result = await list_workflows(format="json", ctx=mock_context)

        # list_workflows returns JSON string when called directly
        assert isinstance(result, str)
        workflows = json.loads(result)
        assert isinstance(workflows, list)
        assert len(workflows) > 0
        assert isinstance(workflows[0], str)
        assert "test-workflow" in workflows

    @pytest.mark.asyncio
    async def test_list_workflows_markdown_format(self, mock_context) -> None:
        """Test list_workflows markdown format."""
        result = await list_workflows(format="markdown", ctx=mock_context)

        assert isinstance(result, str)
        assert "Available Workflows" in result
        assert "test-workflow" in result

    @pytest.mark.asyncio
    async def test_list_workflows_with_tag_filter(self, mock_context) -> None:
        """Test workflow filtering by tags."""
        result = await list_workflows(tags=["nonexistent-tag"], format="json", ctx=mock_context)

        # list_workflows returns JSON string when called directly
        assert isinstance(result, str)
        workflows = json.loads(result)
        assert isinstance(workflows, list)
        # Should return empty list when no workflows match

    @pytest.mark.asyncio
    async def test_list_workflows_returns_all_workflows(self, mock_context) -> None:
        """Test list_workflows returns all registered workflows."""
        registry = mock_context.request_context.lifespan_context.registry

        another_workflow = WorkflowSchema(
            name="another-workflow",
            description="Another test workflow",
            blocks=[{"id": "echo", "type": "Shell", "inputs": {"command": "echo Hello"}}],
        )
        registry.register(another_workflow)

        result = await list_workflows(format="json", ctx=mock_context)

        # list_workflows returns JSON string when called directly
        assert isinstance(result, str)
        workflows = json.loads(result)
        assert isinstance(workflows, list)
        assert len(workflows) >= 2
        assert "test-workflow" in workflows
        assert "another-workflow" in workflows

    # Workflow metadata tests

    @pytest.mark.asyncio
    async def test_get_workflow_info_json_format(self, mock_context) -> None:
        """Test get_workflow_info returns structured data."""
        result = await get_workflow_info(workflow="test-workflow", format="json", ctx=mock_context)

        assert isinstance(result, dict)
        assert result["name"] == "test-workflow"
        assert "description" in result
        assert "blocks" in result
        assert "total_blocks" in result
        assert result["total_blocks"] > 0

    @pytest.mark.asyncio
    async def test_get_workflow_info_markdown_format(self, mock_context) -> None:
        """Test get_workflow_info markdown format."""
        result = await get_workflow_info(
            workflow="test-workflow", format="markdown", ctx=mock_context
        )

        assert isinstance(result, str)
        assert "# Workflow: test-workflow" in result
        assert "## Blocks" in result

    @pytest.mark.asyncio
    async def test_get_workflow_info_not_found(self, mock_context) -> None:
        """Test get_workflow_info with non-existent workflow."""
        result = await get_workflow_info(workflow="non-existent", format="json", ctx=mock_context)

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "available_workflows" in result

    # Schema generation tests

    @pytest.mark.asyncio
    async def test_get_workflow_schema_returns_valid_schema(self) -> None:
        """Test schema generation returns valid JSON Schema."""
        schema = await get_workflow_schema()

        assert isinstance(schema, dict)
        assert "$schema" in schema or "type" in schema
        assert "properties" in schema

    @pytest.mark.asyncio
    async def test_get_workflow_schema_includes_block_types(self) -> None:
        """Test schema includes block structure."""
        schema = await get_workflow_schema()

        assert "properties" in schema
        assert "blocks" in schema["properties"]

    # YAML validation tests

    @pytest.mark.asyncio
    async def test_validate_valid_workflow(self) -> None:
        """Test validation of valid workflow YAML."""
        valid_yaml = """
name: valid-workflow
description: A valid workflow
blocks:
  - id: step1
    type: Shell
    inputs:
      command: echo "test"
"""

        result = await validate_workflow_yaml(yaml_content=valid_yaml)

        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "block_types_used" in result

        if result["valid"]:
            assert len(result["errors"]) == 0
            assert "Shell" in result["block_types_used"]

    @pytest.mark.asyncio
    async def test_validate_invalid_yaml_syntax(self) -> None:
        """Test validation catches YAML syntax errors."""
        result = await validate_workflow_yaml(yaml_content="invalid: [yaml: syntax")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any(
            "yaml" in error.lower() and "syntax" in error.lower() for error in result["errors"]
        )

    @pytest.mark.asyncio
    async def test_validate_workflow_schema_error(self) -> None:
        """Test validation catches schema violations."""
        # Missing required 'name' field
        invalid_yaml = """
description: Missing name field
blocks: []
"""

        result = await validate_workflow_yaml(yaml_content=invalid_yaml)

        assert result["valid"] is False
        assert len(result["errors"]) > 0


# =============================================================================
# Test Classes - Part 4: Checkpoint Management
# =============================================================================


# =============================================================================
# Test Classes - Part 5: Interactive Workflows (with unified Job architecture)
# =============================================================================


class TestInteractiveWorkflows:
    """Tests for interactive workflow pause and resume functionality.

    Tests the complete pause/resume lifecycle:
    1. Execute workflow with Prompt block → pauses
    2. Checkpoint created automatically
    3. Resume with response → continues execution
    4. Response used in conditional blocks
    """

    @pytest.mark.asyncio
    async def test_workflow_pauses_on_prompt_block(self) -> None:
        """Test that workflow pauses when encountering Prompt block.

        Validates:
        - Workflow returns status="paused"
        - Checkpoint ID is provided
        - Prompt message is included
        - Workflow state is preserved
        """
        async with get_mcp_client() as mcp_client:
            # Execute workflow with Prompt block
            result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {"message": "Test approval?"},
                    "debug": True,
                },
            )

            content = result.content[0]
            if not isinstance(content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(content)}")

            response: dict[str, Any] = json.loads(content.text)

            # Verify paused state
            assert response["status"] == "paused", "Workflow should pause at Prompt block"
            assert "job_id" in response, "Should return job ID"
            assert response["job_id"].startswith("job_"), "Job ID should have job_ prefix"

            # Verify prompt information is included
            assert "prompt" in response, "Should include prompt message"
            assert "Test approval?" in response["prompt"], "Prompt should contain message"

            # Verify debug logfile was created
            assert "logfile" in response
            assert response["logfile"].startswith("/tmp/")

            # Optionally verify blocks in logfile (if needed for detailed validation)
            logfile_path = Path(response["logfile"])
            assert logfile_path.exists(), "Logfile should exist"
            debug_data = json.loads(logfile_path.read_text())
            assert "blocks" in debug_data
            assert "start" in debug_data["blocks"], "start block should have executed"
            assert debug_data["blocks"]["start"]["metadata"]["status"] == "completed", (
                "start should be completed (ADR-007: block status = completed)"
            )

    @pytest.mark.asyncio
    async def test_resume_paused_workflow_with_yes_response(self) -> None:
        """Test resuming paused workflow with 'yes' response.

        Validates:
        - Resume continues from checkpoint
        - Response is captured in block outputs
        - Conditional blocks execute based on response
        - Workflow completes successfully
        """
        async with get_mcp_client() as mcp_client:
            # Step 1: Execute workflow until it pauses
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {"message": "Approve deployment?"},
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)

            assert exec_response["status"] == "paused"
            job_id = exec_response["job_id"]

            # Step 2: Resume with 'yes' response
            resume_result = await mcp_client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "yes",
                    "debug": True,
                },
            )

            resume_content = resume_result.content[0]
            if not isinstance(resume_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(resume_content)}")

            # Debug: print resume content if empty or invalid
            if not resume_content.text:
                raise ValueError(f"resume_content.text is empty. Full result: {resume_result}")

            resume_response: dict[str, Any] = json.loads(resume_content.text)

            # Verify workflow completed successfully
            if resume_response["status"] != "success":
                error = resume_response.get("error", "No error message")
                print(f"Resume failed with error: {error}")
            assert resume_response["status"] == "success", "Workflow should complete after resume"

            # Read debug logfile to verify block execution details
            assert "logfile" in resume_response, "Debug mode should include logfile"
            logfile_path = resume_response["logfile"]
            assert os.path.exists(logfile_path), f"Logfile should exist at {logfile_path}"

            with open(logfile_path, encoding="utf-8") as f:
                debug_data: dict[str, Any] = json.load(f)

            # Verify approval block has response
            assert "approval" in debug_data["blocks"]
            assert debug_data["blocks"]["approval"]["outputs"]["response"] == "yes", (
                "Response should be captured"
            )

            # Verify approved_action executed (condition was true)
            assert "approved_action" in debug_data["blocks"], "approved_action should execute"
            assert debug_data["blocks"]["approved_action"]["metadata"]["status"] == "completed", (
                "approved_action should complete (ADR-007: block status = completed)"
            )

            # Verify denied_action was skipped (condition was false)
            assert "denied_action" in debug_data["blocks"], "denied_action should exist"
            assert debug_data["blocks"]["denied_action"]["metadata"]["status"] == "skipped", (
                "denied_action should be skipped"
            )

            # Verify outputs (variable resolution returns strings, not booleans)
            assert resume_response["outputs"]["approval_response"] == "yes"
            assert resume_response["outputs"]["approved"] == "true"
            assert resume_response["outputs"]["denied"] == "false"

    @pytest.mark.asyncio
    async def test_resume_paused_workflow_with_no_response(self) -> None:
        """Test resuming paused workflow with 'no' response.

        Validates:
        - Different response leads to different execution path
        - Conditional logic works correctly
        - Both branches are mutually exclusive
        """
        async with get_mcp_client() as mcp_client:
            # Step 1: Execute workflow until it pauses
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {},
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            job_id = exec_response["job_id"]

            # Step 2: Resume with 'no' response
            resume_result = await mcp_client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "no",
                    "debug": True,
                },
            )

            resume_content = resume_result.content[0]
            if not isinstance(resume_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(resume_content)}")

            resume_response: dict[str, Any] = json.loads(resume_content.text)

            # Verify workflow completed
            assert resume_response["status"] == "success"

            # Read debug logfile to verify block execution details
            assert "logfile" in resume_response, "Debug mode should include logfile"
            logfile_path = resume_response["logfile"]
            assert os.path.exists(logfile_path), f"Logfile should exist at {logfile_path}"

            with open(logfile_path, encoding="utf-8") as f:
                debug_data: dict[str, Any] = json.load(f)

            # Verify response captured
            assert debug_data["blocks"]["approval"]["outputs"]["response"] == "no"

            # Verify denied_action executed (condition was true)
            assert debug_data["blocks"]["denied_action"]["metadata"]["status"] == "completed", (
                "denied_action should complete when response is 'no' (ADR-007: block status)"
            )

            # Verify approved_action was skipped (condition was false)
            assert debug_data["blocks"]["approved_action"]["metadata"]["status"] == "skipped", (
                "approved_action should be skipped when response is 'no'"
            )

            # Verify outputs (variable resolution returns strings, not booleans)
            assert resume_response["outputs"]["approval_response"] == "no"
            assert resume_response["outputs"]["approved"] == "false"
            assert resume_response["outputs"]["denied"] == "true"

    @pytest.mark.asyncio
    async def test_checkpoint_persists_workflow_state(self) -> None:
        """Test that Job preserves complete workflow state for paused workflows.

        Validates:
        - Paused workflows are stored as Jobs
        - Job status shows paused state
        - Can retrieve Job info before resuming
        """
        async with get_mcp_client() as mcp_client:
            # Execute workflow until pause
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {"message": "Custom message for testing"},
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            job_id = exec_response["job_id"]

            # Retrieve job status
            status_result = await mcp_client.call_tool(
                "get_job_status",
                arguments={"job_id": job_id},
            )

            status_content = status_result.content[0]
            if not isinstance(status_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(status_content)}")

            job_status: dict[str, Any] = json.loads(status_content.text)

            # Verify job exists and is paused
            assert job_status["id"] == job_id, "Job ID should match"
            assert job_status["status"] == "paused", "Job should be in paused state"
            assert job_status["workflow"] == "interactive-simple-approval"

            # Verify job has result file (execution state is stored there)
            assert "result_file" in job_status, "Paused job should have result file"

    @pytest.mark.asyncio
    async def test_list_checkpoints_shows_paused_workflows(self) -> None:
        """Test that list_jobs returns paused workflows.

        Validates:
        - Paused workflows appear in job list
        - Job metadata is accessible
        - Can filter by status
        """
        async with get_mcp_client() as mcp_client:
            # Execute workflow until pause
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {},
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            job_id = exec_response["job_id"]

            # List paused jobs
            list_result = await mcp_client.call_tool("list_jobs", arguments={"status": "paused"})

            list_content = list_result.content[0]
            if not isinstance(list_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(list_content)}")

            jobs_response: dict[str, Any] = json.loads(list_content.text)

            # Verify job appears in list
            assert "jobs" in jobs_response
            assert "total" in jobs_response
            assert jobs_response["total"] > 0, "Should have at least one paused job"

            # Find our job
            our_job = None
            for job in jobs_response["jobs"]:
                if job["id"] == job_id:
                    our_job = job
                    break

            assert our_job is not None, "Our job should be in the list"
            assert our_job["workflow"] == "interactive-simple-approval", (
                "Workflow name should match"
            )
            assert our_job["status"] == "paused", "Status should be paused"

    @pytest.mark.asyncio
    async def test_delete_checkpoint_after_resume(self) -> None:
        """Test that paused job can be cancelled.

        Validates:
        - Paused jobs can be cancelled
        - Cancelled jobs show correct status
        - Cannot resume cancelled jobs
        """
        async with get_mcp_client() as mcp_client:
            # Execute and pause
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {},
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            job_id = exec_response["job_id"]

            # Verify job is paused
            status_before = await mcp_client.call_tool(
                "get_job_status",
                arguments={"job_id": job_id},
            )
            status_before_content = status_before.content[0]
            if not isinstance(status_before_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(status_before_content)}")
            status_before_data: dict[str, Any] = json.loads(status_before_content.text)
            assert status_before_data["status"] == "paused", "Job should be paused"

            # Cancel the paused job
            cancel_result = await mcp_client.call_tool("cancel_job", arguments={"job_id": job_id})

            cancel_content = cancel_result.content[0]
            if not isinstance(cancel_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(cancel_content)}")

            cancel_response: dict[str, Any] = json.loads(cancel_content.text)

            # Verify cancellation succeeded
            assert cancel_response["cancelled"] is True, "Job should be successfully cancelled"

            # Verify job status is now cancelled
            status_after = await mcp_client.call_tool(
                "get_job_status",
                arguments={"job_id": job_id},
            )
            status_after_content = status_after.content[0]
            if not isinstance(status_after_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(status_after_content)}")
            status_after_data: dict[str, Any] = json.loads(status_after_content.text)
            assert status_after_data["status"] == "cancelled", "Job status should be cancelled"


# =============================================================================
# Test Classes - Part 6: Quality Assurance
# =============================================================================


class TestQualityAssurance:
    """Response structure consistency and error handling validation.

    Tests that all MCP tools return consistent, well-structured responses
    and provide actionable, educational error messages.
    """

    # Response structure tests

    @pytest.mark.asyncio
    async def test_workflow_response_structure(self, mock_context) -> None:
        """Test workflow response structure consistency."""
        result = await execute_workflow(workflow="test-workflow", debug=True, ctx=mock_context)

        assert "status" in result
        assert result["status"] in ["success", "failure", "paused"]

        if result["status"] == "success":
            assert "outputs" in result
            assert "logfile" in result  # Debug mode writes to file
            assert result["logfile"].startswith("/tmp/")
        elif result["status"] == "failure":
            assert "error" in result

    # Error message quality tests

    @pytest.mark.asyncio
    async def test_workflow_not_found_includes_suggestions(self, mock_context) -> None:
        """Test workflow not found error includes helpful suggestions."""
        result = await execute_workflow(workflow="typo-workflow", ctx=mock_context)

        assert result["status"] == "failure"
        # Should provide list of available workflows
        assert "available_workflows" in result
        assert isinstance(result["available_workflows"], list)

    @pytest.mark.asyncio
    async def test_validation_error_provides_guidance(self) -> None:
        """Test YAML validation errors provide clear guidance."""
        result = await validate_workflow_yaml(yaml_content="invalid: yaml: [syntax")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("YAML" in error or "parsing" in error for error in result["errors"])


# =============================================================================
# Test Classes - Part 7: Regression Testing
# =============================================================================


class TestWorkflowSnapshots:
    """Snapshot-based workflow execution validation.

    Validates that workflow execution produces consistent, reproducible
    results by comparing actual execution against golden snapshots.

    Test Strategy:
    - Dynamic test discovery via MCP list_workflows (tag='test')
    - Parametrized tests - one test case per workflow
    - Detailed response format for comprehensive validation
    - Normalization of dynamic fields (timestamps, etc.)
    - Clear error reporting with actionable guidance

    Coverage Validation:
    - All discovered workflows must have snapshots
    - All snapshots must have corresponding workflows
    - No orphaned snapshots allowed
    """

    async def test_workflow_execution_matches_snapshot(
        self, workflow_name: str, workflow_inputs: dict[str, str]
    ) -> None:
        """Execute workflow and validate against snapshot.

        Test Flow:
        1. Execute workflow via MCP execute_workflow tool
        2. Extract response from MCP protocol
        3. Normalize dynamic fields in actual response
        4. Load expected snapshot (already normalized) from tests/snapshots/
        5. Compare normalized actual vs normalized expected (snapshot)
        6. Report precise differences on mismatch

        Note: Snapshots are pre-normalized by generate_snapshots.py, so we only
        normalize the actual response. This ensures stable snapshots and reliable tests.
        """
        # Execute workflow via MCP with minimal response format
        async with get_mcp_client() as mcp_client:
            result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": workflow_name,
                    "inputs": workflow_inputs,  # Inject base_url for HTTP workflows
                    "debug": False,  # Minimal response (status + outputs/error only)
                },
            )

            # Extract response from MCP protocol
            content = result.content[0]
            if not isinstance(content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(content)}")

            actual_response: dict[str, Any] = json.loads(content.text)

        # Load snapshot (already normalized during generation)
        snapshot_file = SNAPSHOTS_DIR / f"{workflow_name}.json"

        if not snapshot_file.exists():
            pytest.fail(
                f"\n{'=' * 80}\n"
                f"Snapshot missing for workflow: {workflow_name}\n"
                f"{'=' * 80}\n"
                f"Generate snapshot with:\n"
                f"  uv run python tests/generate_snapshots.py\n"
                f"{'=' * 80}\n"
                f"Actual response preview:\n"
                f"{json.dumps(actual_response, indent=2)[:500]}...\n"
                f"{'=' * 80}\n"
            )

        with open(snapshot_file) as f:
            expected_response: dict[str, Any] = json.load(f)

        # Normalize actual response before comparison
        # Even minimal responses can contain dynamic data in workflow outputs
        # Example: variable-resolution-metadata outputs {{metadata.start_time}}
        normalized_actual = normalize_dynamic_fields(actual_response)

        # Compare normalized actual vs expected (snapshot already normalized)
        if normalized_actual != expected_response:
            diff = format_diff(normalized_actual, expected_response, workflow_name)
            pytest.fail(diff)

    async def test_all_snapshots_have_corresponding_workflows(self) -> None:
        """Validate all snapshot files correspond to discoverable workflows."""
        # Get test workflows from MCP server
        async with get_mcp_client() as mcp_client:
            result = await mcp_client.call_tool(
                "list_workflows", arguments={"tags": ["test"], "format": "json"}
            )
            content = result.content[0]
            if not isinstance(content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(content)}")
            test_workflows: list[str] = json.loads(content.text)

        snapshot_files = list(SNAPSHOTS_DIR.glob("*.json"))
        snapshot_names = {f.stem for f in snapshot_files}

        workflow_names = set(test_workflows)
        orphaned = snapshot_names - workflow_names

        if orphaned:
            orphaned_list = "\n".join(f"  - {name}.json" for name in sorted(orphaned))
            pytest.fail(
                f"\n{'=' * 80}\n"
                f"Orphaned snapshots found (no corresponding workflow):\n"
                f"{'=' * 80}\n"
                f"{orphaned_list}\n"
                f"{'=' * 80}\n"
                f"Actions:\n"
                f"  1. Remove orphaned snapshots, OR\n"
                f"  2. Add corresponding workflows with tag='test'\n"
                f"{'=' * 80}\n"
            )

    async def test_all_workflows_have_snapshots(self) -> None:
        """Validate all non-interactive test workflows have corresponding snapshots.

        Excludes interactive workflows (tagged 'interactive') since they
        require resume_workflow and cannot complete via normal execution.
        """
        # Get test workflows from MCP server and exclude interactive ones
        async with get_mcp_client() as mcp_client:
            result = await mcp_client.call_tool(
                "list_workflows", arguments={"tags": ["test"], "format": "json"}
            )
            content = result.content[0]
            if not isinstance(content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(content)}")
            all_workflows: list[str] = json.loads(content.text)

            # Filter out interactive workflows
            non_interactive_workflows = []
            for wf_name in all_workflows:
                info_result = await mcp_client.call_tool(
                    "get_workflow_info", arguments={"workflow": wf_name, "format": "json"}
                )
                info_content = info_result.content[0]
                if not isinstance(info_content, TextContent):
                    continue

                workflow_info = json.loads(info_content.text)
                tags = workflow_info.get("tags", [])

                # Exclude workflows tagged as 'interactive'
                if "interactive" not in tags:
                    non_interactive_workflows.append(wf_name)

        snapshot_files = list(SNAPSHOTS_DIR.glob("*.json"))
        snapshot_names = {f.stem for f in snapshot_files}

        workflow_names = set(non_interactive_workflows)
        missing_snapshots = workflow_names - snapshot_names

        if missing_snapshots:
            missing_list = "\n".join(f"  - {name}" for name in sorted(missing_snapshots))
            pytest.fail(
                f"\n{'=' * 80}\n"
                f"Workflows missing snapshots:\n"
                f"{'=' * 80}\n"
                f"{missing_list}\n"
                f"{'=' * 80}\n"
                f"Generate missing snapshots with:\n"
                f"  uv run python tests/generate_snapshots.py\n"
                f"{'=' * 80}\n"
            )


# =============================================================================
# Pytest Hooks - Dynamic test parametrization
# =============================================================================


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamic test parametrization hook for workflow snapshot tests.

    Discovers workflows by executing list_workflows via MCP and injects
    them as test parameters for snapshot-based validation tests.

    Excludes interactive workflows (tagged 'interactive') since they
    require resume_workflow and cannot complete via normal execution.
    """
    if "workflow_name" in metafunc.fixturenames:
        import asyncio

        async def discover_workflows() -> list[str]:
            """Discover test workflows via MCP for parametrization."""
            async with get_mcp_client() as client:
                # Get all test workflows
                result = await client.call_tool(
                    "list_workflows", arguments={"tags": ["test"], "format": "json"}
                )

                content = result.content[0]
                if not isinstance(content, TextContent):
                    raise ValueError(f"Expected TextContent, got {type(content)}")

                all_workflows: list[str] = json.loads(content.text)

                # Exclude interactive workflows (they require resume_workflow)
                # Get workflow info for each to check tags
                non_interactive = []
                for wf_name in all_workflows:
                    info_result = await client.call_tool(
                        "get_workflow_info", arguments={"workflow": wf_name, "format": "json"}
                    )
                    info_content = info_result.content[0]
                    if not isinstance(info_content, TextContent):
                        continue

                    workflow_info = json.loads(info_content.text)
                    tags = workflow_info.get("tags", [])

                    # Exclude workflows tagged as 'interactive'
                    if "interactive" not in tags:
                        non_interactive.append(wf_name)

                return non_interactive

        workflows = asyncio.run(discover_workflows())
        metafunc.parametrize("workflow_name", workflows)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
