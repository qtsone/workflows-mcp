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
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from workflows_mcp.context import AppContext
from workflows_mcp.engine.checkpoint_store import InMemoryCheckpointStore
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.schema import WorkflowSchema
from workflows_mcp.tools import (
    delete_checkpoint,
    execute_inline_workflow,
    execute_workflow,
    get_checkpoint_info,
    get_workflow_info,
    get_workflow_schema,
    list_checkpoints,
    list_workflows,
    resume_workflow,
    validate_workflow_yaml,
)

# Test configuration
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
WORKFLOWS_DIR = Path(__file__).parent / "workflows"


# =============================================================================
# Utility Functions
# =============================================================================


def normalize_dynamic_fields(data: Any, path: str = "") -> Any:
    """Recursively normalize dynamic fields in workflow responses.

    Handles fields that change between test runs but don't affect
    functional correctness:
    - ISO 8601 timestamps → 'TIMESTAMP'
    - Execution times → 'EXECUTION_TIME'
    - Checkpoint IDs → 'CHECKPOINT_ID'
    - HTTP date headers → 'HTTP_DATE'
    - Amazon trace IDs → 'TRACE_ID'

    Args:
        data: Response data to normalize (dict, list, str, or primitive)
        path: Current path in data structure (for debugging)

    Returns:
        Normalized copy of data with dynamic fields replaced
    """
    if isinstance(data, dict):
        normalized = {}
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            # Normalize known dynamic field names
            if key in ("start_time", "end_time", "created_at", "timestamp", "completed_at"):
                normalized[key] = "TIMESTAMP"
            elif key in (
                "execution_time_ms",
                "duration_ms",
                "execution_time",
                "execution_time_seconds",
            ):
                normalized[key] = "EXECUTION_TIME"
            elif key == "checkpoint_id" and isinstance(value, str):
                normalized[key] = "CHECKPOINT_ID"
            elif key == "date" and isinstance(value, str):
                # Normalize HTTP date headers (e.g., "Sat, 01 Nov 2025 12:26:49 GMT")
                normalized[key] = "HTTP_DATE"
            else:
                normalized[key] = normalize_dynamic_fields(value, current_path)
        return normalized

    elif isinstance(data, list):
        return [normalize_dynamic_fields(item, f"{path}[{i}]") for i, item in enumerate(data)]

    elif isinstance(data, str):
        # Normalize ISO 8601 timestamps in string values
        normalized_str = re.sub(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2}",
            "TIMESTAMP",
            data,
        )
        # Normalize Amazon trace IDs (X-Amzn-Trace-Id)
        # Format: Root=1-6905fc89-71dbb5d47ab07eeb01ec09c5
        normalized_str = re.sub(
            r"Root=1-[a-f0-9]{8}-[a-f0-9]{24}",
            "Root=TRACE_ID",
            normalized_str,
        )
        # Normalize /private/tmp to /tmp (macOS vs Linux compatibility)
        # On macOS, /tmp is a symlink to /private/tmp
        # On Linux, /tmp is just /tmp
        normalized_str = normalized_str.replace("/private/tmp", "/tmp")
        return normalized_str

    else:
        # Primitives (int, float, bool, None) pass through
        return data


def format_diff(actual: dict[str, Any], expected: dict[str, Any], workflow_name: str) -> str:
    """Generate formatted diff between actual and expected responses.

    Args:
        actual: Normalized actual response
        expected: Normalized expected response
        workflow_name: Name of workflow being tested

    Returns:
        Formatted diff string with actionable guidance
    """
    import difflib

    actual_json = json.dumps(actual, indent=2, sort_keys=True)
    expected_json = json.dumps(expected, indent=2, sort_keys=True)

    diff = difflib.unified_diff(
        expected_json.splitlines(keepends=True),
        actual_json.splitlines(keepends=True),
        fromfile=f"expected/{workflow_name}.json",
        tofile=f"actual/{workflow_name}.json",
        lineterm="",
    )

    diff_text = "".join(diff)

    return (
        f"\n{'=' * 80}\n"
        f"Snapshot mismatch for workflow: {workflow_name}\n"
        f"{'=' * 80}\n"
        f"{diff_text}\n"
        f"{'=' * 80}\n"
        f"To update snapshot if this change is intentional:\n"
        f"  uv run python tests/generate_snapshots.py\n"
        f"{'=' * 80}\n"
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_context():
    """Create mock MCP context with AppContext for unit testing MCP tools.

    This fixture creates an isolated test environment with:
    - WorkflowRegistry with test workflows
    - ExecutorRegistry with Shell executor
    - InMemoryCheckpointStore for checkpoint tests

    Returns:
        Mock context object with request_context.lifespan_context structure
    """
    registry = WorkflowRegistry()
    executor_registry = create_default_registry()
    checkpoint_store = InMemoryCheckpointStore()

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
        outputs={"result": "{{blocks.step1.outputs.stdout}}"},
    )
    registry.register(test_workflow)

    # Create mock context matching MCP server structure
    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        checkpoint_store=checkpoint_store,
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
                "list_checkpoints",
                "get_checkpoint_info",
                "delete_checkpoint",
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
            response_format="minimal",
            ctx=mock_context,
        )

        assert result["status"] == "success"
        assert "outputs" in result
        # Minimal format excludes blocks/metadata
        assert "blocks" not in result
        assert "metadata" not in result

    @pytest.mark.asyncio
    async def test_execute_workflow_detailed_format(self, mock_context) -> None:
        """Test workflow execution with detailed response format."""
        result = await execute_workflow(
            workflow="test-workflow",
            inputs={"message": "Test"},
            response_format="detailed",
            ctx=mock_context,
        )

        assert result["status"] == "success"
        assert "outputs" in result
        # Detailed format includes blocks and metadata
        assert "blocks" in result
        assert "metadata" in result
        assert len(result["blocks"]) > 0

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
            response_format="detailed",
            ctx=mock_context,
        )

        assert result["status"] == "success"
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)

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
  result: "{{blocks.echo.outputs.stdout}}"
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


class TestCheckpointManagement:
    """Complete checkpoint lifecycle management tests.

    Tests checkpoint creation, listing, retrieval, resumption, and deletion.
    Note: Full interactive resume tests are in TestInteractiveWorkflows.
    These tests focus on error cases and basic checkpoint operations.
    """

    # Resume workflow error cases

    @pytest.mark.asyncio
    async def test_resume_workflow_invalid_checkpoint_id(self, mock_context) -> None:
        """Test resume_workflow with invalid checkpoint ID."""
        result = await resume_workflow(checkpoint_id="invalid", ctx=mock_context)

        assert result["status"] == "failure"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_workflow_checkpoint_not_found(self, mock_context) -> None:
        """Test resume_workflow with non-existent checkpoint."""
        result = await resume_workflow(checkpoint_id="non_existent_checkpoint", ctx=mock_context)

        assert result["status"] == "failure"

    # Checkpoint listing tests

    @pytest.mark.asyncio
    async def test_list_checkpoints_json_format(self, mock_context) -> None:
        """Test list_checkpoints returns structured data."""
        result = await list_checkpoints(format="json", ctx=mock_context)

        assert "checkpoints" in result
        assert "total" in result
        assert isinstance(result["checkpoints"], list)
        assert isinstance(result["total"], int)

    @pytest.mark.asyncio
    async def test_list_checkpoints_markdown_format(self, mock_context) -> None:
        """Test list_checkpoints markdown format."""
        result = await list_checkpoints(format="markdown", ctx=mock_context)

        assert isinstance(result, str)
        assert "checkpoints" in result.lower() or "no" in result.lower()

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_workflow_filter(self, mock_context) -> None:
        """Test list_checkpoints with workflow name filter."""
        result = await list_checkpoints(
            workflow_name="test-workflow", format="json", ctx=mock_context
        )

        assert "checkpoints" in result
        assert "total" in result

    # Checkpoint info retrieval tests

    @pytest.mark.asyncio
    async def test_get_checkpoint_info_json_format(self, mock_context) -> None:
        """Test get_checkpoint_info with non-existent checkpoint."""
        result = await get_checkpoint_info(
            checkpoint_id="nonexistent", format="json", ctx=mock_context
        )

        assert result["found"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_checkpoint_info_markdown_format(self, mock_context) -> None:
        """Test get_checkpoint_info markdown format."""
        result = await get_checkpoint_info(
            checkpoint_id="nonexistent", format="markdown", ctx=mock_context
        )

        assert isinstance(result, str)
        assert "not found" in result.lower() or "error" in result.lower()

    # Checkpoint deletion tests

    @pytest.mark.asyncio
    async def test_delete_checkpoint_not_found(self, mock_context) -> None:
        """Test delete_checkpoint with non-existent checkpoint."""
        result = await delete_checkpoint(checkpoint_id="nonexistent", ctx=mock_context)

        assert result["deleted"] is False
        assert "message" in result
        assert "not found" in result["message"].lower()


# =============================================================================
# Test Classes - Part 5: Interactive Workflows
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
                    "response_format": "detailed",
                },
            )

            content = result.content[0]
            if not isinstance(content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(content)}")

            response: dict[str, Any] = json.loads(content.text)

            # Verify paused state
            assert response["status"] == "paused", "Workflow should pause at Prompt block"
            assert "checkpoint_id" in response, "Should return checkpoint ID"
            assert response["checkpoint_id"].startswith("pause_"), (
                "Checkpoint ID should have pause_ prefix"
            )

            # Verify prompt information is included
            assert "prompt" in response, "Should include prompt message"
            assert "Test approval?" in response["prompt"], "Prompt should contain message"

            # Verify blocks executed before pause
            assert "blocks" in response
            assert "start" in response["blocks"], "start block should have executed"
            assert response["blocks"]["start"]["metadata"]["status"] == "completed", (
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
                    "response_format": "detailed",
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)

            assert exec_response["status"] == "paused"
            checkpoint_id = exec_response["checkpoint_id"]

            # Step 2: Resume with 'yes' response
            resume_result = await mcp_client.call_tool(
                "resume_workflow",
                arguments={
                    "checkpoint_id": checkpoint_id,
                    "response": "yes",
                    "response_format": "detailed",
                },
            )

            resume_content = resume_result.content[0]
            if not isinstance(resume_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(resume_content)}")

            resume_response: dict[str, Any] = json.loads(resume_content.text)

            # Verify workflow completed successfully
            assert resume_response["status"] == "success", "Workflow should complete after resume"

            # Verify approval block has response
            assert "approval" in resume_response["blocks"]
            assert resume_response["blocks"]["approval"]["outputs"]["response"] == "yes", (
                "Response should be captured"
            )

            # Verify approved_action executed (condition was true)
            assert "approved_action" in resume_response["blocks"], "approved_action should execute"
            assert (
                resume_response["blocks"]["approved_action"]["metadata"]["status"] == "completed"
            ), "approved_action should complete (ADR-007: block status = completed)"

            # Verify denied_action was skipped (condition was false)
            assert "denied_action" in resume_response["blocks"], "denied_action should exist"
            assert resume_response["blocks"]["denied_action"]["metadata"]["status"] == "skipped", (
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
                    "response_format": "detailed",
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            checkpoint_id = exec_response["checkpoint_id"]

            # Step 2: Resume with 'no' response
            resume_result = await mcp_client.call_tool(
                "resume_workflow",
                arguments={
                    "checkpoint_id": checkpoint_id,
                    "response": "no",
                    "response_format": "detailed",
                },
            )

            resume_content = resume_result.content[0]
            if not isinstance(resume_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(resume_content)}")

            resume_response: dict[str, Any] = json.loads(resume_content.text)

            # Verify workflow completed
            assert resume_response["status"] == "success"

            # Verify response captured
            assert resume_response["blocks"]["approval"]["outputs"]["response"] == "no"

            # Verify denied_action executed (condition was true)
            assert (
                resume_response["blocks"]["denied_action"]["metadata"]["status"] == "completed"
            ), "denied_action should complete when response is 'no' (ADR-007: block status)"

            # Verify approved_action was skipped (condition was false)
            assert (
                resume_response["blocks"]["approved_action"]["metadata"]["status"] == "skipped"
            ), "approved_action should be skipped when response is 'no'"

            # Verify outputs (variable resolution returns strings, not booleans)
            assert resume_response["outputs"]["approval_response"] == "no"
            assert resume_response["outputs"]["approved"] == "false"
            assert resume_response["outputs"]["denied"] == "true"

    @pytest.mark.asyncio
    async def test_checkpoint_persists_workflow_state(self) -> None:
        """Test that checkpoint preserves complete workflow state.

        Validates:
        - Completed blocks are preserved
        - Input parameters are preserved
        - Execution context is preserved
        - Can retrieve checkpoint info before resuming
        """
        async with get_mcp_client() as mcp_client:
            # Execute workflow until pause
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {"message": "Custom message for testing"},
                    "response_format": "detailed",
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            checkpoint_id = exec_response["checkpoint_id"]

            # Retrieve checkpoint info
            info_result = await mcp_client.call_tool(
                "get_checkpoint_info",
                arguments={"checkpoint_id": checkpoint_id, "format": "json"},
            )

            info_content = info_result.content[0]
            if not isinstance(info_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(info_content)}")

            checkpoint_info: dict[str, Any] = json.loads(info_content.text)

            # Verify checkpoint exists
            assert checkpoint_info["found"] is True, "Checkpoint should be found"
            assert checkpoint_info["checkpoint_id"] == checkpoint_id, "Checkpoint ID should match"

            # Verify checkpoint metadata
            assert checkpoint_info["workflow_name"] == "interactive-simple-approval"
            assert checkpoint_info["is_paused"] is True, "Should be a pause checkpoint"
            assert checkpoint_info["paused_block_id"] == "approval", "Paused at approval block"
            assert "pause_prompt" in checkpoint_info, "Should have pause prompt"

            # Verify completed blocks tracking
            assert "completed_blocks" in checkpoint_info
            assert "start" in checkpoint_info["completed_blocks"], (
                "start block should be tracked as completed"
            )

            # Verify execution progress
            assert "current_wave" in checkpoint_info
            assert "total_waves" in checkpoint_info
            assert "progress_percentage" in checkpoint_info

    @pytest.mark.asyncio
    async def test_list_checkpoints_shows_paused_workflows(self) -> None:
        """Test that list_checkpoints returns paused workflows.

        Validates:
        - Paused workflows appear in checkpoint list
        - Checkpoint metadata is accessible
        - Can filter by workflow name
        """
        async with get_mcp_client() as mcp_client:
            # Execute workflow until pause
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {},
                    "response_format": "detailed",
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            checkpoint_id = exec_response["checkpoint_id"]

            # List all checkpoints
            list_result = await mcp_client.call_tool(
                "list_checkpoints", arguments={"format": "json"}
            )

            list_content = list_result.content[0]
            if not isinstance(list_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(list_content)}")

            checkpoints_response: dict[str, Any] = json.loads(list_content.text)

            # Verify checkpoint appears in list
            assert "checkpoints" in checkpoints_response
            assert "total" in checkpoints_response
            assert checkpoints_response["total"] > 0, "Should have at least one checkpoint"

            # Find our checkpoint
            our_checkpoint = None
            for cp in checkpoints_response["checkpoints"]:
                if cp["checkpoint_id"] == checkpoint_id:
                    our_checkpoint = cp
                    break

            assert our_checkpoint is not None, "Our checkpoint should be in the list"
            assert our_checkpoint["workflow"] == "interactive-simple-approval", (
                "Workflow name should match"
            )
            assert our_checkpoint["type"] == "pause", "Type should be pause"
            assert our_checkpoint["is_paused"] is True, "Should be paused"

    @pytest.mark.asyncio
    async def test_delete_checkpoint_after_resume(self) -> None:
        """Test that checkpoint can be deleted after workflow resumes.

        Validates:
        - Checkpoint persists during pause
        - Can delete checkpoint after completion
        - Deleted checkpoint cannot be resumed
        """
        async with get_mcp_client() as mcp_client:
            # Execute and pause
            exec_result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "interactive-simple-approval",
                    "inputs": {},
                    "response_format": "detailed",
                },
            )

            exec_content = exec_result.content[0]
            if not isinstance(exec_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(exec_content)}")

            exec_response: dict[str, Any] = json.loads(exec_content.text)
            checkpoint_id = exec_response["checkpoint_id"]

            # Resume and complete
            await mcp_client.call_tool(
                "resume_workflow",
                arguments={
                    "checkpoint_id": checkpoint_id,
                    "response": "yes",
                    "response_format": "minimal",
                },
            )

            # Delete checkpoint
            delete_result = await mcp_client.call_tool(
                "delete_checkpoint", arguments={"checkpoint_id": checkpoint_id}
            )

            delete_content = delete_result.content[0]
            if not isinstance(delete_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(delete_content)}")

            delete_response: dict[str, Any] = json.loads(delete_content.text)

            # Verify deletion succeeded
            assert delete_response["deleted"] is True, "Checkpoint should be successfully deleted"

            # Verify checkpoint no longer exists
            info_result = await mcp_client.call_tool(
                "get_checkpoint_info",
                arguments={"checkpoint_id": checkpoint_id, "format": "json"},
            )

            info_content = info_result.content[0]
            if not isinstance(info_content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(info_content)}")

            info_response: dict[str, Any] = json.loads(info_content.text)

            assert info_response["found"] is False, "Deleted checkpoint should not be found"


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
        result = await execute_workflow(
            workflow="test-workflow", response_format="detailed", ctx=mock_context
        )

        assert "status" in result
        assert result["status"] in ["success", "failure", "paused"]

        if result["status"] == "success":
            assert "outputs" in result
            assert "blocks" in result
            assert "metadata" in result
        elif result["status"] == "failure":
            assert "error" in result

    @pytest.mark.asyncio
    async def test_checkpoint_response_structure(self, mock_context) -> None:
        """Test checkpoint-related response structures."""
        # List checkpoints
        list_result = await list_checkpoints(format="json", ctx=mock_context)
        assert "checkpoints" in list_result
        assert "total" in list_result

        # Get checkpoint info
        info_result = await get_checkpoint_info(
            checkpoint_id="test", format="json", ctx=mock_context
        )
        assert "found" in info_result

        # Delete checkpoint
        delete_result = await delete_checkpoint(checkpoint_id="test", ctx=mock_context)
        assert "deleted" in delete_result

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

    async def test_workflow_execution_matches_snapshot(self, workflow_name: str) -> None:
        """Execute workflow and validate against snapshot.

        Test Flow:
        1. Execute workflow via MCP execute_workflow tool
        2. Extract response from MCP protocol
        3. Normalize dynamic fields in response
        4. Load expected snapshot from tests/snapshots/
        5. Normalize expected snapshot
        6. Compare normalized responses
        7. Report precise differences on mismatch
        """
        # Execute workflow via MCP with detailed response format
        async with get_mcp_client() as mcp_client:
            result = await mcp_client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": workflow_name,
                    "inputs": {},
                    "response_format": "detailed",
                },
            )

            # Extract response from MCP protocol
            content = result.content[0]
            if not isinstance(content, TextContent):
                raise ValueError(f"Expected TextContent, got {type(content)}")

            actual_response: dict[str, Any] = json.loads(content.text)

        # Load snapshot
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

        # Normalize both responses to handle dynamic fields
        normalized_actual = normalize_dynamic_fields(actual_response)
        normalized_expected = normalize_dynamic_fields(expected_response)

        # Compare normalized responses
        if normalized_actual != normalized_expected:
            diff = format_diff(normalized_actual, normalized_expected, workflow_name)
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
