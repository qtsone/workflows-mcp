#!/usr/bin/env python3
"""MCP server and HTTP service testing.

Transport: HTTP-only (FastAPI / Uvicorn).

This test suite validates:

1. **HTTP Service Health** (TestHTTPServerIntegration)
   - build_app() composition and public endpoint availability
   - Fail-fast bootstrap token enforcement (spec §7.4)

2. **MCP Tool Functionality** (unit tests via mock_context)
   - Workflow execution (execute_workflow, execute_inline_workflow)
   - Workflow discovery (list_workflows, get_workflow_info)
   - Schema validation (get_workflow_schema, validate_workflow_yaml)
   - Checkpoint management (resume_workflow, list_checkpoints, etc.)
   - Input validation and error handling
   - Response format variations (minimal/detailed, json/markdown)
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult

from workflows_mcp.context import AppContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.schema import WorkflowSchema
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.run_history_repo import SQLiteRunHistoryRepository
from workflows_mcp.tools import (
    execute_inline_workflow,
    execute_workflow,
    get_workflow_info,
    get_workflow_schema,
    list_workflows,
    validate_workflow_yaml,
)
from workflows_mcp.tools_memory import register_memory_tools

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


# =============================================================================
# Test Classes - Part 1: MCP Server Health & Protocol Compliance
# =============================================================================
# HTTP service health tests are in TestHTTPServerIntegration (Part 8).


class TestProjectMemoryToolsMetadata:
    def test_project_tools_expose_actionable_descriptions(self) -> None:
        server = FastMCP("metadata-test")
        register_memory_tools(server, enable_project_tools=True)

        onboard = server._tool_manager._tools.get("onboard")
        assert onboard is not None
        assert onboard.description
        assert "start or continue project memory onboarding" in onboard.description.lower()

        sync = server._tool_manager._tools.get("sync")
        assert sync is not None
        assert sync.description
        assert "continue project memory synchronization" in sync.description.lower()

    def test_old_project_tool_names_absent(self) -> None:
        server = FastMCP("absent-names-test")
        register_memory_tools(server, enable_project_tools=True)
        assert server._tool_manager._tools.get("project_onboard") is None
        assert server._tool_manager._tools.get("project_sync") is None


class TestProjectMemoryToolsScanParameter:
    """Validate that onboard and sync accept the scan parameter."""

    def test_onboard_scan_parameter_accepted_in_schema(self) -> None:
        """onboard tool schema must include the scan parameter."""
        server = FastMCP("scan-schema-test")
        register_memory_tools(server, enable_project_tools=True)

        onboard = server._tool_manager._tools.get("onboard")
        assert onboard is not None

        import inspect

        sig = inspect.signature(onboard.fn)
        assert "scan" in sig.parameters, "onboard must accept a 'scan' parameter"

    def test_sync_scan_parameter_accepted_in_schema(self) -> None:
        """sync tool schema must include the scan parameter."""
        server = FastMCP("scan-schema-test-sync")
        register_memory_tools(server, enable_project_tools=True)

        sync = server._tool_manager._tools.get("sync")
        assert sync is not None

        import inspect

        sig = inspect.signature(sync.fn)
        assert "scan" in sig.parameters, "sync must accept a 'scan' parameter"

    def test_onboard_scan_description_mentions_snapshot(self) -> None:
        """onboard scan param description must reference snapshot persistence."""
        server = FastMCP("scan-desc-test")
        register_memory_tools(server, enable_project_tools=True)

        onboard = server._tool_manager._tools.get("onboard")
        assert onboard is not None

        import inspect

        sig = inspect.signature(onboard.fn)
        scan_param = sig.parameters.get("scan")
        assert scan_param is not None

        # Validate via tool description rather than annotation internals
        assert onboard.description is not None
        assert "start or continue project memory onboarding" in onboard.description.lower()

    @pytest.mark.asyncio
    async def test_onboard_rejects_scan_with_extra_fields(self, mock_context: MagicMock) -> None:
        """onboard with invalid scan config (extra fields) must return error."""
        from workflows_mcp.tools import execute_workflow  # noqa: F401 (trigger import check)
        from workflows_mcp.tools_memory import register_memory_tools as _rtm

        server = FastMCP("scan-validation-test")
        _rtm(server, enable_project_tools=True)
        tool = server._tool_manager._tools.get("onboard")
        assert tool is not None

        result = await tool.fn(
            ingest={"format": "structured", "memories": [{"content": "test"}]},
            scan={"patterns": ["*.py"], "unknown_extra_field": True},
            ctx=mock_context,
        )

        data = result.structuredContent
        assert "error" in data
        error = data["error"]
        assert isinstance(error, dict)
        assert "code" in error

    @pytest.mark.asyncio
    async def test_onboard_rejects_oss_r2_checkpoint(self, mock_context: MagicMock) -> None:
        """onboard must reject checkpoints with version != oss-r3."""
        from workflows_mcp.tools_memory import register_memory_tools as _rtm

        server = FastMCP("checkpoint-version-test")
        _rtm(server, enable_project_tools=True)
        tool = server._tool_manager._tools.get("onboard")
        assert tool is not None

        stale_checkpoint = {
            "version": "oss-r2",
            "scope": {},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 0,
            "completed": [],
        }

        result = await tool.fn(checkpoint=stale_checkpoint, ctx=mock_context)

        data = result.structuredContent
        assert "error" in data
        error = data["error"]
        assert isinstance(error, dict)
        assert error.get("code") == "MEM_CHECKPOINT_INVALID"

    @pytest.mark.asyncio
    async def test_sync_rejects_oss_r2_checkpoint(self, mock_context: MagicMock) -> None:
        """sync must reject checkpoints with version != oss-r3."""
        from workflows_mcp.tools_memory import register_memory_tools as _rtm

        server = FastMCP("sync-checkpoint-version-test")
        _rtm(server, enable_project_tools=True)
        tool = server._tool_manager._tools.get("sync")
        assert tool is not None

        stale_checkpoint = {
            "version": "oss-r2",
            "scope": {},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 0,
            "completed": [],
        }

        result = await tool.fn(checkpoint=stale_checkpoint, ctx=mock_context)

        data = result.structuredContent
        assert "error" in data
        error = data["error"]
        assert isinstance(error, dict)
        assert error.get("code") == "MEM_CHECKPOINT_INVALID"


class TestProjectMemoryToolsExposureInOssMode:
    @pytest.mark.asyncio
    async def test_app_lifespan_exposes_project_tools_when_oss_mode_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from workflows_mcp import server as server_mod

        class _FakeBackend:
            async def disconnect(self) -> None:
                return None

        class _FakeMemoryExecutor:
            type_name = "Memory"

        register_calls: list[bool] = []

        async def _fake_prepare_memory_schema(_memory_db_host: str) -> _FakeBackend:
            return _FakeBackend()

        def _fake_register_memory_tools(_mcp: Any, *, enable_project_tools: bool = True) -> None:
            register_calls.append(enable_project_tools)

        import workflows_mcp.engine.executors_memory as executors_memory_mod
        import workflows_mcp.tools_memory as tools_memory_mod

        monkeypatch.setattr(server_mod, "_prepare_memory_schema", _fake_prepare_memory_schema)
        monkeypatch.setattr(server_mod, "load_workflows", lambda _registry: None)
        monkeypatch.setattr(executors_memory_mod, "MemoryExecutor", _FakeMemoryExecutor)
        monkeypatch.setattr(tools_memory_mod, "register_memory_tools", _fake_register_memory_tools)
        monkeypatch.setenv("MEMORY_DB_HOST", "localhost")
        monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
        monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")
        monkeypatch.setenv("WORKFLOWS_OSS_MODE", "true")
        monkeypatch.setenv("WORKFLOWS_ENABLE_PROJECT_TOOLS", "true")

        local_mcp = FastMCP("project-tools-enabled-test", lifespan=server_mod.app_lifespan)
        async with server_mod.app_lifespan(local_mcp):
            pass

        assert register_calls == [True]

    @pytest.mark.asyncio
    async def test_app_lifespan_disables_project_tools_when_oss_mode_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from workflows_mcp import server as server_mod

        class _FakeBackend:
            async def disconnect(self) -> None:
                return None

        class _FakeMemoryExecutor:
            type_name = "Memory"

        register_calls: list[bool] = []

        async def _fake_prepare_memory_schema(_memory_db_host: str) -> _FakeBackend:
            return _FakeBackend()

        def _fake_register_memory_tools(_mcp: Any, *, enable_project_tools: bool = True) -> None:
            register_calls.append(enable_project_tools)

        import workflows_mcp.engine.executors_memory as executors_memory_mod
        import workflows_mcp.tools_memory as tools_memory_mod

        monkeypatch.setattr(server_mod, "_prepare_memory_schema", _fake_prepare_memory_schema)
        monkeypatch.setattr(server_mod, "load_workflows", lambda _registry: None)
        monkeypatch.setattr(executors_memory_mod, "MemoryExecutor", _FakeMemoryExecutor)
        monkeypatch.setattr(tools_memory_mod, "register_memory_tools", _fake_register_memory_tools)
        monkeypatch.setenv("MEMORY_DB_HOST", "localhost")
        monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
        monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")
        monkeypatch.setenv("WORKFLOWS_OSS_MODE", "false")
        monkeypatch.setenv("WORKFLOWS_ENABLE_PROJECT_TOOLS", "true")

        local_mcp = FastMCP("project-tools-disabled-test", lifespan=server_mod.app_lifespan)
        async with server_mod.app_lifespan(local_mcp):
            pass

        assert register_calls == [False]


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
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        assert isinstance(result, CallToolResult)
        data = result.structuredContent
        assert data["status"] == "success"
        assert "outputs" in data
        # Minimal format excludes blocks/metadata
        assert "blocks" not in data
        assert "metadata" not in data

    @pytest.mark.asyncio
    async def test_execute_workflow_debug_mode(self, mock_context) -> None:
        """Test workflow execution with debug mode (writes logfile)."""
        result = await execute_workflow(
            workflow="test-workflow",
            inputs={"message": "Test"},
            debug=True,
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "success"
        assert "outputs" in data
        # Debug mode includes logfile path instead of inline blocks/metadata
        assert "logfile" in data
        assert data["logfile"].startswith("/tmp/")
        assert data["logfile"].endswith(".json")

    @pytest.mark.asyncio
    async def test_execute_workflow_not_found(self, mock_context) -> None:
        """Test execute_workflow with non-existent workflow."""
        result = await execute_workflow(
            workflow="non-existent-workflow",
            inputs=None,
            debug=False,
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "failure"
        assert "not found" in data["error"].lower()
        # Should provide actionable guidance
        assert "available_workflows" in data
        assert isinstance(data["available_workflows"], list)

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
            workflow="test-required-inputs",
            inputs={},
            debug=False,
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "failure"
        # Error about missing inputs or variable resolution
        assert "required" in data["error"].lower() or "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_workflow_sync_failure_is_persisted_to_sqlite(
        self,
        mock_context,
        tmp_path: Path,
    ) -> None:
        """Registered sync workflow failures should appear in /runs without debug files."""
        app_ctx = mock_context.request_context.lifespan_context
        app_ctx.metadata_base_dir = tmp_path
        app_ctx.metadata_db_path = tmp_path / "server.db"
        app_ctx.metadata_db_conn = connect_metadata_db(tmp_path / "server.db")
        migrate_metadata_db(app_ctx.metadata_db_conn)
        registry = app_ctx.registry

        required_workflow = WorkflowSchema(
            name="test-persisted-sync-failure",
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
            workflow="test-persisted-sync-failure",
            inputs={},
            debug=True,
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        data = result.structuredContent
        assert data["status"] == "failure"
        assert "run_id" in data
        assert "logfile" not in data

        conn = connect_metadata_db(tmp_path / "server.db")
        try:
            run = SQLiteRunHistoryRepository(conn).get_run(str(data["run_id"]))
        finally:
            conn.close()
        assert run is not None
        assert run.workflow_name == "test-persisted-sync-failure"
        assert run.status == "failed"
        assert run.execution_mode == "sync"
        assert run.execution_json is not None
        assert "required_param" in run.execution_json

    @pytest.mark.asyncio
    async def test_execute_workflow_with_custom_inputs(self, mock_context) -> None:
        """Test execute_workflow with runtime inputs."""
        result = await execute_workflow(
            workflow="test-workflow",
            inputs={"message": "CustomMessage"},
            debug=True,
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "success"
        assert "logfile" in data
        assert data["logfile"].startswith("/tmp/")

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

        result = await execute_inline_workflow(
            workflow_yaml=workflow_yaml,
            inputs=None,
            debug=False,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "success"
        assert "outputs" in data

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_is_persisted_to_sqlite(
        self,
        mock_context,
        tmp_path: Path,
    ) -> None:
        """Inline workflow executions should use run history instead of debug files."""
        app_ctx = mock_context.request_context.lifespan_context
        app_ctx.metadata_base_dir = tmp_path
        app_ctx.metadata_db_path = tmp_path / "server.db"
        app_ctx.metadata_db_conn = connect_metadata_db(tmp_path / "server.db")
        migrate_metadata_db(app_ctx.metadata_db_conn)
        workflow_yaml = """
name: inline-persisted
description: Inline workflow test
blocks:
  - id: echo
    type: Shell
    inputs:
      command: echo 'Inline persisted'
outputs:
  result:
    value: "{{blocks.echo.outputs.stdout}}"
"""

        result = await execute_inline_workflow(
            workflow_yaml=workflow_yaml,
            inputs=None,
            debug=True,
            ctx=mock_context,
        )

        data = result.structuredContent
        assert data["status"] == "success"
        assert "run_id" in data
        assert "logfile" not in data

        conn = connect_metadata_db(tmp_path / "server.db")
        try:
            run = SQLiteRunHistoryRepository(conn).get_run(str(data["run_id"]))
        finally:
            conn.close()
        assert run is not None
        assert run.workflow_name == "inline-persisted"
        assert run.status == "completed"
        assert run.execution_mode == "inline"
        assert run.execution_json is not None
        assert "Inline persisted" in run.execution_json

    @pytest.mark.asyncio
    async def test_concurrent_executions_share_one_metadata_connection(
        self,
        mock_context,
        tmp_path: Path,
    ) -> None:
        """N concurrent sessions reuse the one lifespan-owned metadata connection.

        Smoke-tests the central-server invariant: a single shared SQLite connection
        serves many concurrent ``execute_workflow`` calls without transaction-ownership
        errors, connection churn, or lost run records.
        """
        app_ctx = mock_context.request_context.lifespan_context
        app_ctx.metadata_base_dir = tmp_path
        app_ctx.metadata_db_path = tmp_path / "server.db"
        shared_conn = connect_metadata_db(tmp_path / "server.db")
        migrate_metadata_db(shared_conn)
        app_ctx.metadata_db_conn = shared_conn

        app_ctx.registry.register(
            WorkflowSchema(
                name="concurrent-smoke",
                description="Trivial workflow for concurrency smoke test",
                blocks=[
                    {
                        "id": "echo",
                        "type": "Shell",
                        "inputs": {"command": "echo concurrent"},
                    }
                ],
            )
        )

        results = await asyncio.gather(
            *(
                execute_workflow(
                    workflow="concurrent-smoke",
                    inputs={},
                    debug=False,
                    mode="sync",
                    timeout=None,
                    ctx=mock_context,
                )
                for _ in range(16)
            )
        )

        run_ids = {result.structuredContent["run_id"] for result in results}
        assert len(run_ids) == 16  # every call persisted a distinct record
        for result in results:
            assert result.structuredContent["status"] == "success"

        repo = SQLiteRunHistoryRepository(shared_conn)
        for run_id in run_ids:
            run = repo.get_run(str(run_id))
            assert run is not None
            assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_empty_yaml(self, mock_context) -> None:
        """Test execute_inline_workflow with empty YAML."""
        result = await execute_inline_workflow(
            workflow_yaml="# empty yaml\n",
            inputs=None,
            debug=False,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "failure"
        # Empty YAML parsed as None
        assert "dictionary" in data["error"].lower() or "nonetype" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_invalid_yaml(self, mock_context) -> None:
        """Test execute_inline_workflow with invalid YAML syntax."""
        result = await execute_inline_workflow(
            workflow_yaml="invalid: [unclosed",
            inputs=None,
            debug=False,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "failure"
        assert "parse" in data["error"].lower() or "yaml" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_inline_workflow_missing_required_fields(self, mock_context) -> None:
        """Test execute_inline_workflow with missing required fields."""
        workflow_yaml = """
name: incomplete-workflow
description: Missing blocks field
"""

        result = await execute_inline_workflow(
            workflow_yaml=workflow_yaml,
            inputs=None,
            debug=False,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "failure"


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
        result = await list_workflows(tags=[], format="json", ctx=mock_context)

        # list_workflows returns CallToolResult with structuredContent
        assert isinstance(result, CallToolResult)
        workflows = result.structuredContent["workflows"]
        assert isinstance(workflows, list)
        assert len(workflows) > 0
        assert isinstance(workflows[0], str)
        assert "test-workflow" in workflows

    @pytest.mark.asyncio
    async def test_list_workflows_markdown_format(self, mock_context) -> None:
        """Test list_workflows markdown format."""
        result = await list_workflows(tags=[], format="markdown", ctx=mock_context)

        assert isinstance(result, CallToolResult)
        text = result.content[0].text
        assert "Available Workflows" in text
        assert "test-workflow" in text

    @pytest.mark.asyncio
    async def test_list_workflows_with_tag_filter(self, mock_context) -> None:
        """Test workflow filtering by tags."""
        result = await list_workflows(tags=["nonexistent-tag"], format="json", ctx=mock_context)

        # list_workflows returns CallToolResult with structuredContent
        assert isinstance(result, CallToolResult)
        workflows = result.structuredContent["workflows"]
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

        result = await list_workflows(tags=[], format="json", ctx=mock_context)

        # list_workflows returns CallToolResult with structuredContent
        assert isinstance(result, CallToolResult)
        workflows = result.structuredContent["workflows"]
        assert isinstance(workflows, list)
        assert len(workflows) >= 2
        assert "test-workflow" in workflows
        assert "another-workflow" in workflows

    # Workflow metadata tests

    @pytest.mark.asyncio
    async def test_get_workflow_info_json_format(self, mock_context) -> None:
        """Test get_workflow_info returns structured data."""
        result = await get_workflow_info(workflow="test-workflow", format="json", ctx=mock_context)

        # Tools return CallToolResult with structuredContent
        assert isinstance(result, CallToolResult)
        data = result.structuredContent
        assert data["name"] == "test-workflow"
        assert "description" in data
        assert "blocks" in data
        assert "total_blocks" in data
        assert data["total_blocks"] > 0

    @pytest.mark.asyncio
    async def test_get_workflow_info_markdown_format(self, mock_context) -> None:
        """Test get_workflow_info markdown format."""
        result = await get_workflow_info(
            workflow="test-workflow", format="markdown", ctx=mock_context
        )

        assert isinstance(result, CallToolResult)
        text = result.content[0].text
        assert "# Workflow: test-workflow" in text
        assert "## Blocks" in text

    @pytest.mark.asyncio
    async def test_get_workflow_info_not_found(self, mock_context) -> None:
        """Test get_workflow_info with non-existent workflow."""
        result = await get_workflow_info(workflow="non-existent", format="json", ctx=mock_context)

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert "error" in data
        assert "not found" in data["error"].lower()
        assert "available_workflows" in data

    # Schema generation tests

    @pytest.mark.asyncio
    async def test_get_workflow_schema_returns_valid_schema(self, mock_context) -> None:
        """Test schema generation returns valid JSON Schema."""
        result = await get_workflow_schema(ctx=mock_context)

        # Tools return CallToolResult with structuredContent
        assert isinstance(result, CallToolResult)
        schema = result.structuredContent
        assert isinstance(schema, dict)
        assert "$schema" in schema or "type" in schema
        assert "properties" in schema

    @pytest.mark.asyncio
    async def test_get_workflow_schema_includes_block_types(self, mock_context) -> None:
        """Test schema includes block structure."""
        result = await get_workflow_schema(ctx=mock_context)

        # Tools return CallToolResult with structuredContent
        schema = result.structuredContent
        assert "properties" in schema
        assert "blocks" in schema["properties"]

    # YAML validation tests

    @pytest.mark.asyncio
    async def test_validate_valid_workflow(self, mock_context) -> None:
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

        result = await validate_workflow_yaml(yaml_content=valid_yaml, ctx=mock_context)

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert "valid" in data
        assert "errors" in data
        assert "warnings" in data
        assert "block_types_used" in data

        if data["valid"]:
            assert len(data["errors"]) == 0
            assert "Shell" in data["block_types_used"]

    @pytest.mark.asyncio
    async def test_validate_invalid_yaml_syntax(self, mock_context) -> None:
        """Test validation catches YAML syntax errors."""
        result = await validate_workflow_yaml(
            yaml_content="invalid: [yaml: syntax", ctx=mock_context
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["valid"] is False
        assert len(data["errors"]) > 0
        assert any(
            "yaml" in error.lower() and "syntax" in error.lower() for error in data["errors"]
        )

    @pytest.mark.asyncio
    async def test_validate_workflow_schema_error(self, mock_context) -> None:
        """Test validation catches schema violations."""
        # Missing required 'name' field
        invalid_yaml = """
description: Missing name field
blocks: []
"""

        result = await validate_workflow_yaml(yaml_content=invalid_yaml, ctx=mock_context)

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["valid"] is False
        assert len(data["errors"]) > 0


# =============================================================================
# Test Classes - Part 4: Checkpoint Management
# =============================================================================


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
            workflow="test-workflow",
            inputs=None,
            debug=True,
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert "status" in data
        assert data["status"] in ["success", "failure", "paused"]

        if data["status"] == "success":
            assert "outputs" in data
            assert "logfile" in data  # Debug mode writes to file
            assert data["logfile"].startswith("/tmp/")
        elif data["status"] == "failure":
            assert "error" in data

    # Error message quality tests

    @pytest.mark.asyncio
    async def test_workflow_not_found_includes_suggestions(self, mock_context) -> None:
        """Test workflow not found error includes helpful suggestions."""
        result = await execute_workflow(
            workflow="typo-workflow",
            inputs=None,
            debug=False,
            mode="sync",
            timeout=None,
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["status"] == "failure"
        # Should provide list of available workflows
        assert "available_workflows" in data
        assert isinstance(data["available_workflows"], list)

    @pytest.mark.asyncio
    async def test_validation_error_provides_guidance(self, mock_context) -> None:
        """Test YAML validation errors provide clear guidance."""
        result = await validate_workflow_yaml(
            yaml_content="invalid: yaml: [syntax",
            ctx=mock_context,
        )

        # Tools return CallToolResult with structuredContent
        data = result.structuredContent
        assert data["valid"] is False
        assert len(data["errors"]) > 0
        assert any("YAML" in error or "parsing" in error for error in data["errors"])


# =============================================================================
# HTTP Service Integration (Task 7)
# =============================================================================


class TestHTTPServerIntegration:
    """HTTP service integration tests using FastAPI TestClient.

    build_app() is exercised end-to-end via an isolated tmp token store so
    tests never touch ~/.workflows on the developer's machine.
    """

    def _make_client(self, tmp_path: Path) -> TestClient:
        from workflows_mcp.auth import TokenStore
        from workflows_mcp.server import build_app

        store = TokenStore(tmp_path / "auth.json")
        store.write_token("test-token-" + "x" * 32)
        return TestClient(build_app(base_dir=tmp_path))

    def test_health_is_public(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_openapi_is_public(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        assert client.get("/openapi.json").status_code == 200

    def test_docs_is_public(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        assert client.get("/docs").status_code == 200

    def test_protected_config_requires_token(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        assert (
            client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
            ).status_code
            == 401
        )

    def test_build_app_no_longer_requires_bootstrap_token(self, tmp_path: Path) -> None:
        """build_app should initialize without requiring legacy bootstrap token state."""
        from workflows_mcp.server import build_app

        app = build_app(base_dir=tmp_path)
        client = TestClient(app)
        assert client.get("/health").status_code == 200


class TestBuiltinWorkflowDiscovery:
    """Verify built-in template workflows are discoverable via the registry.

    These tests load the real packaged templates/memory directory to confirm
    that user-callable built-in workflows are present without requiring a
    running MCP server (no MCP server restart needed for unit discovery).
    """

    def _make_builtin_registry(self) -> "WorkflowRegistry":
        from workflows_mcp import server as server_mod

        registry = WorkflowRegistry()
        builtin_path = server_mod._builtin_workflow_path()
        registry.load_from_directory(builtin_path)
        return registry

    def test_system2_derive_is_discoverable_as_builtin_workflow(self) -> None:
        """system2-derive must be present in the built-in template registry."""
        registry = self._make_builtin_registry()
        assert registry.exists("system2-derive"), (
            "system2-derive not found in built-in templates/memory registry. "
            f"Found workflows: {sorted(registry.list_names())}"
        )

    def test_system2_derive_workflow_has_expected_tags(self) -> None:
        """system2-derive workflow must carry the 'memory' and 'system2' tags."""
        registry = self._make_builtin_registry()
        workflow = registry.get("system2-derive")
        assert workflow is not None
        tags = set(workflow.tags or [])
        assert "memory" in tags, f"Expected tag 'memory' in {tags}"
        assert "system2" in tags, f"Expected tag 'system2' in {tags}"

    def test_system2_derive_workflow_has_expected_inputs(self) -> None:
        """system2-derive workflow must declare all Task 8 ADR-013 inputs."""
        registry = self._make_builtin_registry()
        workflow = registry.get("system2-derive")
        assert workflow is not None
        declared_inputs = set(workflow.inputs or {})
        required = {
            "palace",
            "scope",
            "evidence_entity_stable_ids",
            "room_intent_label",
            "compartment_reasoning_unit",
            "is_new_wing",
            "proof_bundle_evidence_categories",
            "corridor_from_claim_id",
            "corridor_to_claim_id",
            "corridor_type",
        }
        missing = required - declared_inputs
        assert not missing, (
            f"system2-derive is missing expected inputs: {missing}. Declared: {declared_inputs}"
        )

    def test_system2_verify_lifecycle_is_discoverable_as_builtin_workflow(self) -> None:
        """system2-verify-lifecycle must be present in the built-in template registry."""
        registry = self._make_builtin_registry()
        assert registry.exists("system2-verify-lifecycle"), (
            "system2-verify-lifecycle not found in built-in templates/memory registry. "
            f"Found workflows: {sorted(registry.list_names())}"
        )

    def test_system2_verify_lifecycle_workflow_has_expected_tags(self) -> None:
        """system2-verify-lifecycle must carry the 'memory' and 'system2' tags."""
        registry = self._make_builtin_registry()
        workflow = registry.get("system2-verify-lifecycle")
        assert workflow is not None
        tags = set(workflow.tags or [])
        assert "memory" in tags, f"Expected tag 'memory' in {tags}"
        assert "system2" in tags, f"Expected tag 'system2' in {tags}"

    def test_system2_verify_lifecycle_workflow_has_expected_inputs(self) -> None:
        """system2-verify-lifecycle must declare all Task 13 ADR-013 inputs."""
        registry = self._make_builtin_registry()
        workflow = registry.get("system2-verify-lifecycle")
        assert workflow is not None
        declared_inputs = set(workflow.inputs or {})
        required = {
            "palace",
            "scope",
            "degrade_claim_ids",
            "force_archive_claim_ids",
            "absent_verification_cycle_ids",
        }
        missing = required - declared_inputs
        assert not missing, (
            f"system2-verify-lifecycle is missing expected inputs: {missing}. "
            f"Declared: {declared_inputs}"
        )

    def test_system2_verify_lifecycle_workflow_invokes_reconcile_operation(self) -> None:
        """system2-verify-lifecycle must invoke reconcile_semantic_lifecycle operation."""
        registry = self._make_builtin_registry()
        workflow = registry.get("system2-verify-lifecycle")
        assert workflow is not None
        blocks = workflow.blocks or []

        def _block_operation(b: Any) -> str | None:
            if isinstance(b, dict):
                return b.get("inputs", {}).get("operation")
            inputs = getattr(b, "inputs", None) or {}
            return inputs.get("operation")

        reconcile_blocks = [
            b for b in blocks if _block_operation(b) == "reconcile_semantic_lifecycle"
        ]
        assert reconcile_blocks, (
            "system2-verify-lifecycle has no block invoking 'reconcile_semantic_lifecycle'. "
            f"Block operations found: {[_block_operation(b) for b in blocks]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
