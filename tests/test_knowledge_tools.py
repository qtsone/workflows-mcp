"""Tests for unified memory MCP tool."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.executors_file import _SENSITIVE_EXCLUDE_PATTERNS
from workflows_mcp.engine.memory_service import (
    ManageMemoryResult,
    MemoryContractError,
    MemoryResult,
    QueryMemoryResult,
)
from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_memory import (
    ScanConfig,
    ScanSnapshot,
    _compute_scan_delta,
    _get_scan_root,
    _hash_content,
    _validate_scan_path_within_workspace,
    register_memory_tools,
)

register_memory_tools(_mcp_server)


@pytest.fixture(autouse=True)
def _patch_memory_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch memory config resolution so ephemeral-backend tests bypass the
    production config guard (MEMORY_BACKEND_UNAVAILABLE) without touching any
    production code path."""
    sentinel = object()

    async def _noop_ensure_schema(_backend: Any) -> None:
        return

    monkeypatch.setattr(
        "workflows_mcp.tools_memory.memory_connection_config_from_metadata",
        lambda _app_ctx: None,
    )
    monkeypatch.setattr(
        "workflows_mcp.tools_memory.memory_connection_config_from_env",
        lambda: sentinel,
    )
    monkeypatch.setattr(
        "workflows_mcp.engine.knowledge.schema.ensure_schema",
        _noop_ensure_schema,
    )


def _get_tool_fn(name: str) -> Any:
    tool_manager = _mcp_server._tool_manager
    tool = tool_manager._tools.get(name)
    if tool is None:
        raise ValueError(f"Tool {name!r} not registered")
    return tool.fn


memory = _get_tool_fn("memory")
onboard = _get_tool_fn("onboard")
sync = _get_tool_fn("sync")


@pytest.fixture
def mock_ctx() -> MagicMock:
    ctx = MagicMock()
    app_ctx = MagicMock()
    app_ctx.memory_backend = None
    app_ctx.memory_backend_lock = None
    ctx.request_context.lifespan_context = app_ctx
    exec_context = MagicMock()
    exec_context.user_string_id = None
    app_ctx.create_execution_context.return_value = exec_context
    app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")
    app_ctx.get_active_context.return_value = None
    return ctx


def _make_backend_mock() -> MagicMock:
    backend = MagicMock()
    backend.connect = AsyncMock()
    backend.disconnect = AsyncMock()
    return backend


@pytest.fixture
def workspace_tmp() -> Path:
    """Create a temporary directory inside the workspace root (os.getcwd()).

    Required for scan tests because path safety validation rejects root
    values outside the workspace root.
    """
    workspace = Path(os.getcwd()).resolve()
    with tempfile.TemporaryDirectory(dir=workspace, prefix=".scan_test_") as td:
        yield Path(td)
    # TemporaryDirectory context manager handles cleanup


class TestMemoryToolRegistration:
    def test_memory_tool_registered(self) -> None:
        tool = _mcp_server._tool_manager._tools.get("memory")
        assert tool is not None
        assert tool.description
        assert "Query and update memory" in tool.description
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False

    def test_project_tools_registered_with_actionable_descriptions(self) -> None:
        onboard_tool = _mcp_server._tool_manager._tools.get("onboard")
        assert onboard_tool is not None
        assert onboard_tool.description
        assert "start or continue project memory onboarding" in onboard_tool.description.lower()

        sync_tool = _mcp_server._tool_manager._tools.get("sync")
        assert sync_tool is not None
        assert sync_tool.description
        assert "continue project memory synchronization" in sync_tool.description.lower()

    def test_old_project_tool_names_absent(self) -> None:
        assert _mcp_server._tool_manager._tools.get("project_onboard") is None
        assert _mcp_server._tool_manager._tools.get("project_sync") is None


class TestMemoryTool:
    @pytest.mark.asyncio
    async def test_project_onboard_and_sync_use_checkpointed_memory_flow(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            def _result_for_operation(operation: str) -> MemoryResult:
                if operation == "ingest":
                    return MemoryResult(
                        operation="ingest",
                        manage=ManageMemoryResult(
                            operation="store",
                            memory_ids=["m-1"],
                            stored_count=1,
                        ),
                    )
                if operation == "supersede":
                    return MemoryResult(
                        operation="supersede",
                        manage=ManageMemoryResult(
                            operation="supersede", superseded_ids=["m-legacy"], archived_count=1
                        ),
                    )
                if operation == "archive":
                    return MemoryResult(
                        operation="archive",
                        manage=ManageMemoryResult(operation="forget", archived_count=1),
                    )
                if operation == "maintain":
                    return MemoryResult(
                        operation="maintain",
                        manage=ManageMemoryResult(
                            operation="maintain", communities_updated=1, assessed_count=3
                        ),
                    )
                raise AssertionError(f"Unexpected operation: {operation}")

            async def _execute(request: Any) -> MemoryResult:
                return _result_for_operation(request.operation)

            mock_service.execute = AsyncMock(side_effect=_execute)

            onboard_result = await onboard(
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                ingest={"format": "raw", "content": "initial", "memory_tier": "direct"},
                supersede={"ids": ["m-legacy"], "superseded_by": "m-1"},
                archive={"ids": ["m-old"]},
                maintain={"mode": "community_refresh"},
                response={"mode": "compact", "debug": False},
                max_operations=1,
                ctx=mock_ctx,
            )

            onboard_payload = json.loads(onboard_result.content[0].text)
            assert onboard_payload["status"] == "checkpoint"
            assert onboard_payload["last_operation"] == "ingest"
            assert onboard_payload["checkpoint"]["next_index"] == 1
            assert onboard_payload["result"]["stored"] == 1

            sync_result = await sync(
                checkpoint=onboard_payload["checkpoint"],
                response={"mode": "compact", "debug": False},
                max_operations=3,
                ctx=mock_ctx,
            )
            sync_payload = json.loads(sync_result.content[0].text)
            assert sync_payload["status"] == "completed"
            assert sync_payload["completed_operations"] == [
                "ingest",
                "supersede",
                "archive",
                "maintain",
            ]

            executed_ops = [call.args[0].operation for call in mock_service.execute.call_args_list]
            assert executed_ops == ["ingest", "supersede", "archive", "maintain"]

    @pytest.mark.asyncio
    async def test_project_sync_rejects_invalid_checkpoint(self, mock_ctx: MagicMock) -> None:
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await sync(
                checkpoint={"version": "invalid"},
                response={"mode": "compact", "debug": False},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_CHECKPOINT_INVALID"

    @pytest.mark.asyncio
    async def test_project_onboard_rejects_checkpoint_without_ingest_first(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await onboard(
                checkpoint={
                    "version": "oss-r3",
                    "scope": {"palace": "acme"},
                    "plan": [{"operation": "supersede", "payload": {"ids": ["m-1"]}}],
                    "next_index": 0,
                    "completed": [],
                },
                response={"mode": "compact", "debug": False},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_CHECKPOINT_INVALID"

    @pytest.mark.asyncio
    async def test_project_sync_rejects_checkpoint_when_completed_mismatches_plan(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await sync(
                checkpoint={
                    "version": "oss-r3",
                    "scope": {"palace": "acme"},
                    "plan": [{"operation": "ingest", "payload": {"content": "x"}}],
                    "next_index": 1,
                    "completed": [{"operation": "archive", "result": {}}],
                },
                response={"mode": "compact", "debug": False},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_CHECKPOINT_INVALID"

    @pytest.mark.asyncio
    async def test_project_sync_rejects_checkpoint_when_completed_len_exceeds_next_index(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await sync(
                checkpoint={
                    "version": "oss-r3",
                    "scope": {"palace": "acme"},
                    "plan": [{"operation": "ingest", "payload": {"content": "x"}}],
                    "next_index": 0,
                    "completed": [{"operation": "ingest", "result": {}}],
                },
                response={"mode": "compact", "debug": False},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_CHECKPOINT_INVALID"

    @pytest.mark.asyncio
    async def test_project_sync_rejects_checkpoint_when_completed_len_less_than_next_index(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await sync(
                checkpoint={
                    "version": "oss-r3",
                    "scope": {"palace": "acme"},
                    "plan": [{"operation": "ingest", "payload": {"content": "x"}}],
                    "next_index": 1,
                    "completed": [],
                },
                response={"mode": "compact", "debug": False},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_CHECKPOINT_INVALID"

    @pytest.mark.asyncio
    async def test_uses_lifespan_backend_without_per_call_connect_disconnect(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_shared = MagicMock()
        backend_shared.connect = AsyncMock()
        backend_shared.disconnect = AsyncMock()
        mock_ctx.request_context.lifespan_context.memory_backend = backend_shared

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend") as mock_backend_cls,
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="query",
                    query=QueryMemoryResult(
                        facts=[{"content": "fact one"}],
                        memories=[],
                        communities=[],
                        paths=[],
                        evidence=[],
                        diagnostics={},
                    ),
                )
            )

            await memory(
                operation="query",
                query={"text": "find this"},
                scope={"palace": "test"},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )
            await memory(
                operation="query",
                query={"text": "find this too"},
                scope={"palace": "test"},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            assert mock_backend_cls.call_count == 0
            backend_shared.connect.assert_not_awaited()
            backend_shared.disconnect.assert_not_awaited()
            assert mock_service_cls.call_count == 2
            assert all(call.args[0] is backend_shared for call in mock_service_cls.call_args_list)

    @pytest.mark.asyncio
    async def test_serializes_shared_lifespan_backend_calls(self, mock_ctx: MagicMock) -> None:
        backend_shared = MagicMock()
        backend_shared.connect = AsyncMock()
        backend_shared.disconnect = AsyncMock()
        mock_ctx.request_context.lifespan_context.memory_backend = backend_shared
        mock_ctx.request_context.lifespan_context.memory_backend_lock = asyncio.Lock()

        in_flight = 0
        max_in_flight = 0

        async def _execute(_request: Any) -> MemoryResult:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return MemoryResult(
                operation="query",
                query=QueryMemoryResult(
                    facts=[{"content": "ok"}],
                    memories=[],
                    communities=[],
                    paths=[],
                    evidence=[],
                    diagnostics={},
                ),
            )

        with patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(side_effect=_execute)

            await asyncio.gather(
                memory(
                    operation="query",
                    query={"text": "find this"},
                    scope={"palace": "test"},
                    response={"mode": "compact", "debug": False},
                    ctx=mock_ctx,
                ),
                memory(
                    operation="query",
                    query={"text": "find this too"},
                    scope={"palace": "test"},
                    response={"mode": "compact", "debug": False},
                    ctx=mock_ctx,
                ),
            )

        assert max_in_flight == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_per_call_backend_when_lifespan_backend_missing(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="query",
                    query=QueryMemoryResult(
                        facts=[],
                        memories=[],
                        communities=[],
                        paths=[],
                        evidence=[],
                        diagnostics={},
                    ),
                )
            )

            await memory(
                operation="query",
                query={"text": "find this"},
                scope={"palace": "test"},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            backend_mock.connect.assert_awaited_once()
            backend_mock.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingest_structured_begin_failure_returns_non_empty_error_details(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        backend_mock.begin_transaction = AsyncMock(side_effect=RuntimeError(""))
        backend_mock.commit = AsyncMock()
        backend_mock.rollback = AsyncMock()

        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await memory(
                operation="ingest",
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                record={
                    "format": "structured",
                    "memories": [{"content": "fails before writes"}],
                },
                response={"mode": "compact", "debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_INTERNAL_ERROR"
        assert payload["error"]["message"] == "memory failed"
        assert payload["error"]["retryable"] is False
        assert payload["error"].get("correlation_id")
        backend_mock.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_query_search_compact_response(self, mock_ctx: MagicMock) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="query",
                    query=QueryMemoryResult(
                        facts=[{"content": "fact one", "path": "a.py", "source": "repo"}],
                        memories=[],
                        communities=[],
                        paths=[],
                        evidence=[],
                        diagnostics={},
                    ),
                )
            )

            result = await memory(
                operation="query",
                scope={"wing": "svc", "room": "component", "compartment": "topic"},
                query={"text": "find this", "mode": "search", "radius": 1, "precision": 0.5},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["facts"][0]["content"] == "fact one"

    @pytest.mark.asyncio
    async def test_query_communities_mode_is_preserved_through_contract_path(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _execute(request: Any) -> MemoryResult:
                assert request.query is not None
                assert request.query.mode == "communities"
                return MemoryResult(
                    operation="query",
                    query=QueryMemoryResult(
                        communities=[{"content": "cluster summary"}],
                        diagnostics={"effective_strategy": "communities"},
                    ),
                )

            mock_service.execute = AsyncMock(side_effect=_execute)

            result = await memory(
                operation="query",
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                query={"text": "find this", "mode": "communities"},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["communities"] == [{"content": "cluster summary"}]

    @pytest.mark.asyncio
    async def test_manage_failure_uses_machine_readable_error_envelope(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store",
                        success=False,
                        error="store transaction failed",
                    ),
                )
            )

            result = await memory(
                operation="ingest",
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                record={"format": "raw", "content": "test", "memory_tier": "direct"},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["error"]["code"] == "MEM_OPERATION_FAILED"
            assert payload["error"]["message"] == "store transaction failed"
            assert payload["error"]["retryable"] is False

    @pytest.mark.asyncio
    async def test_project_onboard_stops_on_error_envelope_and_returns_failed_checkpoint(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            def _result_for_operation(operation: str) -> MemoryResult:
                if operation == "ingest":
                    return MemoryResult(
                        operation="ingest",
                        manage=ManageMemoryResult(
                            operation="store",
                            memory_ids=["m-1"],
                            stored_count=1,
                        ),
                    )
                if operation == "supersede":
                    return MemoryResult(
                        operation="supersede",
                        manage=ManageMemoryResult(
                            operation="supersede",
                            success=False,
                            error="supersede failed",
                        ),
                    )
                raise AssertionError(f"Unexpected operation: {operation}")

            async def _execute(request: Any) -> MemoryResult:
                return _result_for_operation(request.operation)

            mock_service.execute = AsyncMock(side_effect=_execute)

            result = await onboard(
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                ingest={"format": "raw", "content": "initial", "memory_tier": "direct"},
                supersede={"ids": ["m-legacy"], "superseded_by": "m-1"},
                response={"mode": "compact", "debug": False},
                max_operations=5,
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["status"] == "checkpoint"
            assert payload["failed_operation"] == "supersede"
            assert payload["error"]["code"] == "MEM_OPERATION_FAILED"
            assert payload["completed_operations"] == ["ingest"]
            assert payload["checkpoint"]["next_index"] == 1
            assert payload["remaining_operations"] == ["supersede"]

    @pytest.mark.asyncio
    async def test_project_sync_stops_on_error_envelope_and_returns_failed_checkpoint(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="supersede",
                    manage=ManageMemoryResult(
                        operation="supersede",
                        success=False,
                        error="sync supersede failed",
                    ),
                )
            )

            result = await sync(
                checkpoint={
                    "version": "oss-r3",
                    "scope": {"palace": "acme"},
                    "plan": [
                        {"operation": "ingest", "payload": {"content": "seed"}},
                        {
                            "operation": "supersede",
                            "payload": {"ids": ["m-old"], "superseded_by": "m-new"},
                        },
                    ],
                    "next_index": 1,
                    "completed": [{"operation": "ingest", "result": {"stored": 1}}],
                },
                response={"mode": "compact", "debug": False},
                max_operations=3,
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["status"] == "checkpoint"
            assert payload["failed_operation"] == "supersede"
            assert payload["error"]["code"] == "MEM_OPERATION_FAILED"
            assert payload["completed_operations"] == []
            assert payload["checkpoint"]["next_index"] == 1
            assert payload["remaining_operations"] == ["supersede"]

    @pytest.mark.asyncio
    async def test_project_onboard_returns_checkpoint_payload_on_internal_step_exception(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory._execute_memory_request") as mock_execute,
        ):
            mock_execute.side_effect = [
                {"stored": 1},
                RuntimeError("boom"),
            ]

            result = await onboard(
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                ingest={"format": "raw", "content": "initial", "memory_tier": "direct"},
                supersede={"ids": ["m-legacy"], "superseded_by": "m-1"},
                response={"mode": "compact", "debug": False},
                max_operations=5,
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["status"] == "checkpoint"
            assert payload["failed_operation"] == "supersede"
            assert payload["error"]["code"] == "MEM_INTERNAL_ERROR"
            assert payload["error"]["message"] == "onboard failed"
            assert payload["error"]["retryable"] is False
            assert payload["error"].get("correlation_id")
            assert payload["completed_operations"] == ["ingest"]
            assert payload["checkpoint"]["next_index"] == 1
            assert payload["remaining_operations"] == ["supersede"]

    @pytest.mark.asyncio
    async def test_project_sync_returns_checkpoint_payload_on_internal_step_exception(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory._execute_memory_request") as mock_execute,
        ):
            mock_execute.side_effect = RuntimeError("sync-boom")

            result = await sync(
                checkpoint={
                    "version": "oss-r3",
                    "scope": {"palace": "acme"},
                    "plan": [
                        {
                            "operation": "supersede",
                            "payload": {"ids": ["m-old"], "superseded_by": "m-new"},
                        },
                    ],
                    "next_index": 0,
                    "completed": [],
                },
                response={"mode": "compact", "debug": False},
                max_operations=3,
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["status"] == "checkpoint"
            assert payload["failed_operation"] == "supersede"
            assert payload["error"]["code"] == "MEM_INTERNAL_ERROR"
            assert payload["error"]["message"] == "sync failed"
            assert payload["error"]["retryable"] is False
            assert payload["error"].get("correlation_id")
            assert payload["completed_operations"] == []
            assert payload["checkpoint"]["next_index"] == 0
            assert payload["remaining_operations"] == ["supersede"]

    @pytest.mark.asyncio
    async def test_memory_tool_rejects_legacy_hall_scope_key(self, mock_ctx: MagicMock) -> None:
        """Public memory tool contract must reject legacy hall taxonomy key."""
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await memory(
                operation="query",
                scope={"palace": "acme", "wing": "svc", "room": "comp", "hall": "legacy"},
                query={"text": "find this", "mode": "search"},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_INVALID_TAXONOMY_KEY"
        assert "compartment" in payload["error"]["message"]

    @pytest.mark.asyncio
    async def test_memory_tool_returns_deterministic_error_for_unknown_category_ingest(
        self, mock_ctx: MagicMock
    ) -> None:
        """Unknown categories with allow_create_categories=false stay deterministic."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                side_effect=MemoryContractError(
                    code="MEM_UNKNOWN_CATEGORY",
                    message=(
                        "MEM_UNKNOWN_CATEGORY: Unknown categories: 'unknown-cat'. "
                        "Set allow_create_categories=true to explicitly create missing categories."
                    ),
                    retryable=False,
                )
            )

            result = await memory(
                operation="ingest",
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                record={
                    "format": "raw",
                    "content": "test",
                    "memory_tier": "direct",
                    "categories": ["unknown-cat"],
                },
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_UNKNOWN_CATEGORY"
        assert payload["error"]["code"] != "MEM_INTERNAL_ERROR"
        assert "allow_create_categories=true" in payload["error"]["message"]

    @pytest.mark.asyncio
    async def test_query_graph_mode_shapes_paths(self, mock_ctx: MagicMock) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="query",
                    query=QueryMemoryResult(
                        facts=[],
                        memories=[],
                        communities=[],
                        paths=[{"nodes": ["a", "b"]}],
                        evidence=[{"nodes": [{"id": "a"}], "edges": []}],
                        diagnostics={},
                    ),
                )
            )

            result = await memory(
                operation="query",
                query={
                    "text": "graph",
                    "mode": "graph",
                    "graph": {"op": "path", "start": "a", "end": "b"},
                },
                scope={"palace": "test"},
                response={"mode": "graph"},
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert "paths" in payload
            assert "nodes" in payload

    @pytest.mark.asyncio
    async def test_maintain_community_refresh_shapes_communities_updated(
        self, mock_ctx: MagicMock
    ) -> None:
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="maintain",
                    manage=ManageMemoryResult(
                        operation="maintain",
                        communities_updated=2,
                        diagnostics={"mode": "community_refresh", "community_count": 2},
                    ),
                )
            )

            result = await memory(
                operation="maintain",
                maintenance={"mode": "community_refresh"},
                scope={"palace": "test"},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload == {"communities_updated": 2}


# ============================================================================
# Scan models unit tests
# ============================================================================


class TestScanConfigModel:
    def test_valid_minimal(self) -> None:
        cfg = ScanConfig(patterns=["**/*.py"])
        assert cfg.patterns == ["**/*.py"]
        assert cfg.root == "."
        assert cfg.mode == "full"
        assert cfg.deletion_policy == "archive"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            ScanConfig.model_validate({"patterns": ["*.py"], "unknown_field": True})

    def test_deletion_policy_values(self) -> None:
        for policy in ("archive", "supersede", "ignore"):
            cfg = ScanConfig(patterns=["*.py"], deletion_policy=policy)  # type: ignore[arg-type]
            assert cfg.deletion_policy == policy


class TestScanSnapshot:
    def test_extra_fields_forbidden(self) -> None:
        scan_cfg = ScanConfig(patterns=["*.py"])
        with pytest.raises(Exception):
            ScanSnapshot.model_validate(
                {
                    "scan_config": scan_cfg.model_dump(),
                    "entries": [],
                    "unexpected": True,
                }
            )

    def test_round_trip_serialization(self) -> None:
        scan_cfg = ScanConfig(patterns=["src/**/*.py"], root="/tmp")
        snap = ScanSnapshot(
            scan_config=scan_cfg,
            entries=[],
        )
        dumped = snap.model_dump()
        restored = ScanSnapshot.model_validate(dumped)
        assert restored.scan_config.patterns == ["src/**/*.py"]


class TestComputeScanDelta:
    def _make_snap(
        self,
        entries: list[dict[str, Any]],
        patterns: list[str] | None = None,
    ) -> ScanSnapshot:
        from workflows_mcp.tools_memory import FileSnapshotEntry

        return ScanSnapshot(
            scan_config=ScanConfig(patterns=patterns or ["*.py"]),
            entries=[FileSnapshotEntry(**e) for e in entries],
        )

    def test_added_files(self) -> None:
        old = self._make_snap([])
        new = self._make_snap(
            [{"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "abc"}]
        )
        added, modified, deleted = _compute_scan_delta(old, new)
        assert added == ["a.py"]
        assert modified == []
        assert deleted == []

    def test_deleted_files(self) -> None:
        old = self._make_snap(
            [{"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "abc"}]
        )
        new = self._make_snap([])
        added, modified, deleted = _compute_scan_delta(old, new)
        assert added == []
        assert modified == []
        assert deleted == ["a.py"]

    def test_modified_files_hash_differs(self) -> None:
        old = self._make_snap(
            [{"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "old"}]
        )
        new = self._make_snap(
            [{"path": "a.py", "size_bytes": 12, "mtime_ns": 2, "content_hash": "new"}]
        )
        added, modified, deleted = _compute_scan_delta(old, new)
        assert modified == ["a.py"]

    def test_no_change_same_hash(self) -> None:
        entry = {"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "same"}
        old = self._make_snap([entry])
        new = self._make_snap([entry])
        added, modified, deleted = _compute_scan_delta(old, new)
        assert added == modified == deleted == []

    def test_mtime_size_same_but_hash_differs_counts_as_modified(self) -> None:
        # Fix 3: same mtime AND size but different hash MUST be detected as modified.
        # Previously a prefilter skipped hash comparison when mtime+size matched; removed.
        entry_old = {"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "old"}
        entry_new = {"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "new"}
        old = self._make_snap([entry_old])
        new = self._make_snap([entry_new])
        added, modified, deleted = _compute_scan_delta(old, new)
        assert "a.py" in modified  # same mtime+size, different hash → must be modified


# ============================================================================
# Security: path containment, sensitive defaults
# ============================================================================


class TestScanPathSafety:
    """Verify scan path containment enforcement with MEM_SCAN_PATH_OUT_OF_ROOT."""

    def test_out_of_override_root_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A scan.root outside WORKFLOWS_SCAN_ROOT override must raise MemoryContractError."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(allowed))

        with pytest.raises(MemoryContractError) as exc_info:
            _validate_scan_path_within_workspace(outside.resolve(), "scan.root")
        assert exc_info.value.code == "MEM_SCAN_PATH_OUT_OF_ROOT"
        assert "scan.root" in exc_info.value.message

    def test_in_root_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A scan.root inside WORKFLOWS_SCAN_ROOT must not raise."""
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(tmp_path))
        _validate_scan_path_within_workspace(tmp_path / "src", "scan.root")

    def test_workspace_root_itself_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The configured root itself must be accepted."""
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(tmp_path))
        _validate_scan_path_within_workspace(tmp_path.resolve(), "scan.root")

    @pytest.mark.asyncio
    async def test_project_onboard_rejects_out_of_override_root_scan_root(
        self, monkeypatch: pytest.MonkeyPatch, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        """project_onboard must return MEM_SCAN_PATH_OUT_OF_ROOT when scan.root is
        outside the WORKFLOWS_SCAN_ROOT override."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(allowed))

        result = await onboard(
            scan={"patterns": ["*.py"], "root": str(outside)},
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_SCAN_PATH_OUT_OF_ROOT"


class TestSensitiveExcludeDefaults:
    """Verify _SENSITIVE_EXCLUDE_PATTERNS contains the required secret-file patterns."""

    def test_env_files_excluded(self) -> None:
        assert "**/.env" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/.env.*" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_pem_key_excluded(self) -> None:
        assert "**/*.pem" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/*.key" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_rsa_key_excluded(self) -> None:
        assert "**/id_rsa" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/id_rsa.*" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_credentials_secrets_excluded(self) -> None:
        assert "**/credentials" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/secrets" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_p12_pfx_excluded(self) -> None:
        assert "**/*.p12" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/*.pfx" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_sqlite_db_excluded(self) -> None:
        assert "**/*.sqlite" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/*.db" in _SENSITIVE_EXCLUDE_PATTERNS

    @pytest.mark.asyncio
    async def test_run_readfiles_scan_excludes_env_file(self, workspace_tmp: Path) -> None:
        """run_readfiles_scan must never return .env files even when they match the glob."""
        (workspace_tmp / ".env").write_text("SECRET=hunter2\n")
        (workspace_tmp / "app.py").write_text("print('hello')\n")

        from workflows_mcp.engine.executors_file import run_readfiles_scan

        files = await run_readfiles_scan(
            patterns=["**/*", "**/.env"],
            base_path=str(workspace_tmp),
            max_files=50,
        )
        paths = {f["path"] for f in files}
        assert ".env" not in paths
        assert "app.py" in paths


# ============================================================================
# Scan onboarding integration tests
# ============================================================================


class TestProjectOnboardWithScan:
    @pytest.mark.asyncio
    async def test_scan_auto_generates_ingest_when_ingest_omitted(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """When scan is provided without ingest, ingest is auto-generated from scanned files."""
        (workspace_tmp / "a.py").write_text("def foo(): pass\n")

        backend_mock = _make_backend_mock()
        captured_records: list[dict[str, Any]] = []

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _execute(request: Any) -> MemoryResult:
                if request.operation == "ingest":
                    assert request.record is not None
                    captured_records.append(request.record)
                    return MemoryResult(
                        operation="ingest",
                        manage=ManageMemoryResult(
                            operation="store", memory_ids=["m-scan-1"], stored_count=1
                        ),
                    )
                raise AssertionError(f"Unexpected operation: {request.operation}")

            mock_service.execute = AsyncMock(side_effect=_execute)

            result = await onboard(
                scope={"palace": "proj"},
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                response={"mode": "compact", "debug": False},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed", payload
        assert payload["completed_operations"] == ["ingest"]
        assert len(captured_records) == 1
        # Auto-generated ingest must contain structured memories
        rec = captured_records[0]
        memories = rec.memories if hasattr(rec, "memories") else rec["memories"]
        assert memories is not None
        assert any("foo" in (m.get("content", "") if isinstance(m, dict) else "") for m in memories)

    @pytest.mark.asyncio
    async def test_scan_snapshot_persisted_in_checkpoint(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """Completed checkpoint must carry scan_snapshot when scan was used."""
        (workspace_tmp / "module.py").write_text("x = 1\n")

        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store", memory_ids=["m-1"], stored_count=1
                    ),
                )
            )

            result = await onboard(
                scope={"palace": "proj"},
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        checkpoint = payload["checkpoint"]
        assert checkpoint["version"] == "oss-r3"
        assert "scan" in checkpoint
        assert "scan_snapshot" in checkpoint
        snap = checkpoint["scan_snapshot"]
        assert checkpoint["scan"]["patterns"] == ["*.py"]
        assert snap["scan_config"]["patterns"] == ["*.py"]
        assert len(snap["entries"]) == 1
        assert snap["entries"][0]["path"] == "module.py"

    @pytest.mark.asyncio
    async def test_explicit_ingest_wins_over_scan_generated(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """Explicit ingest keys must override auto-generated scan ingest keys."""
        (workspace_tmp / "x.py").write_text("pass\n")

        backend_mock = _make_backend_mock()
        captured_records: list[dict[str, Any]] = []

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _capture(request: Any) -> MemoryResult:
                if request.operation == "ingest":
                    captured_records.append(request.record)
                    return MemoryResult(
                        operation="ingest",
                        manage=ManageMemoryResult(
                            operation="store", memory_ids=["m-x"], stored_count=1
                        ),
                    )
                raise AssertionError(f"Unexpected op: {request.operation}")

            mock_service.execute = AsyncMock(side_effect=_capture)

            explicit_memories = [{"content": "explicit content", "path": "override.py"}]
            result = await onboard(
                scope={"palace": "proj"},
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                ingest={"format": "structured", "memories": explicit_memories},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert len(captured_records) == 1
        # The explicit memories list must win
        rec = captured_records[0]
        actual_memories = rec.memories if hasattr(rec, "memories") else rec["memories"]
        assert actual_memories is not None
        assert len(actual_memories) == len(explicit_memories)
        # Check content matches (memories may be dicts or Pydantic objects)
        actual_contents = [
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            for m in actual_memories
        ]
        assert actual_contents == [m["content"] for m in explicit_memories]

    @pytest.mark.asyncio
    async def test_oss_r3_checkpoint_version_enforced_on_onboard(
        self, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        """project_onboard must produce oss-r3 checkpoints, not older versions."""
        (tmp_path / "f.py").write_text("1\n")
        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store", memory_ids=["m-1"], stored_count=1
                    ),
                )
            )
            result = await onboard(
                scope={"palace": "p"},
                ingest={"format": "raw", "content": "c", "memory_tier": "direct"},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["checkpoint"]["version"] == "oss-r3"

    @pytest.mark.asyncio
    async def test_oss_r3_checkpoint_rejected_when_version_is_older(
        self, mock_ctx: MagicMock
    ) -> None:
        """project_onboard must reject checkpoints with version != oss-r3."""
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await onboard(
                checkpoint={
                    "version": "oss-r2",
                    "scope": {"palace": "acme"},
                    "plan": [
                        {"operation": "ingest", "payload": {"content": "x"}},
                    ],
                    "next_index": 0,
                    "completed": [],
                },
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_CHECKPOINT_INVALID"

    @pytest.mark.asyncio
    async def test_scan_no_matching_files_returns_actionable_error_not_internal(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """When scan patterns match no files, project_onboard must return a deterministic
        contract error (MEM_SCAN_NO_FILES_MATCHED), not the opaque MEM_INTERNAL_ERROR."""
        # workspace_tmp is empty — no *.py files exist
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await onboard(
                scope={"palace": "proj"},
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert "error" in payload, f"Expected error envelope, got: {payload}"
        assert payload["error"]["code"] == "MEM_SCAN_NO_FILES_MATCHED", (
            f"Expected MEM_SCAN_NO_FILES_MATCHED, got {payload['error']['code']!r}. "
            f"Full payload: {payload}"
        )
        assert payload["error"]["retryable"] is False
        assert "correlation_id" in payload["error"]

    @pytest.mark.asyncio
    async def test_scan_nonexistent_single_path_returns_actionable_error_not_internal(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """When scan.path points to a non-existent file, project_onboard must return
        MEM_SCAN_NO_FILES_MATCHED, not the opaque MEM_INTERNAL_ERROR."""
        missing = str(workspace_tmp / "does_not_exist.py")
        backend_mock = _make_backend_mock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock):
            result = await onboard(
                scope={"palace": "proj"},
                scan={"path": missing, "root": str(workspace_tmp)},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert "error" in payload, f"Expected error envelope, got: {payload}"
        assert payload["error"]["code"] != "MEM_INTERNAL_ERROR", (
            f"Root cause masked: got opaque MEM_INTERNAL_ERROR instead of actionable code. "
            f"Full payload: {payload}"
        )
        assert payload["error"]["retryable"] is False


# ============================================================================
# Scan sync integration tests
# ============================================================================


class TestProjectSyncWithScan:
    @pytest.mark.asyncio
    async def test_sync_detects_added_file_and_auto_ingests(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """project_sync should detect newly added files and generate ingest automatically."""
        from workflows_mcp.tools_memory import FileSnapshotEntry

        (workspace_tmp / "existing.py").write_text("old = 1\n")
        (workspace_tmp / "new_file.py").write_text("new = 2\n")

        # Build prior snapshot: only existing.py
        prior_content = "old = 1\n"
        prior_snap = ScanSnapshot(
            scan_config=ScanConfig(patterns=["*.py"], root=str(workspace_tmp)),
            entries=[
                FileSnapshotEntry(
                    path="existing.py",
                    size_bytes=len(prior_content),
                    mtime_ns=1,
                    content_hash=_hash_content(prior_content),
                )
            ],
        )

        checkpoint = {
            "version": "oss-r3",
            "scope": {"palace": "proj"},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 0,
            "completed": [],
            "scan_snapshot": prior_snap.model_dump(),
        }

        backend_mock = _make_backend_mock()
        captured_records: list[dict[str, Any]] = []

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _capture(request: Any) -> MemoryResult:
                if request.operation == "ingest":
                    captured_records.append(request.record)
                    return MemoryResult(
                        operation="ingest",
                        manage=ManageMemoryResult(
                            operation="store", memory_ids=["m-new"], stored_count=1
                        ),
                    )
                raise AssertionError(f"Unexpected op: {request.operation}")

            mock_service.execute = AsyncMock(side_effect=_capture)

            result = await sync(
                checkpoint=checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        # Should have run ingest for the new file
        assert payload["status"] == "completed", payload
        assert len(captured_records) == 1
        rec = captured_records[0]
        memories = rec.memories if hasattr(rec, "memories") else rec.get("memories", [])
        paths = set()
        for m in memories or []:
            if isinstance(m, dict):
                meta = m.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("path"):
                    paths.add(meta["path"])
            else:
                meta = getattr(m, "metadata", None) or {}
                if isinstance(meta, dict) and meta.get("path"):
                    paths.add(meta["path"])
        assert "new_file.py" in paths

    @pytest.mark.asyncio
    async def test_sync_refreshes_snapshot_in_checkpoint(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """After sync, the returned checkpoint must have an updated scan_snapshot."""

        (workspace_tmp / "a.py").write_text("a = 1\n")

        prior_snap = ScanSnapshot(
            scan_config=ScanConfig(patterns=["*.py"], root=str(workspace_tmp)),
            entries=[],  # Empty prior → a.py is "added"
        )
        checkpoint = {
            "version": "oss-r3",
            "scope": {"palace": "proj"},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 0,
            "completed": [],
            "scan_snapshot": prior_snap.model_dump(),
        }

        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store", memory_ids=["m-1"], stored_count=1
                    ),
                )
            )

            result = await sync(
                checkpoint=checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        new_snap = payload["checkpoint"].get("scan_snapshot")
        new_scan = payload["checkpoint"].get("scan")
        assert new_snap is not None
        assert new_scan is not None
        assert len(new_snap["entries"]) == 1
        assert new_snap["entries"][0]["path"] == "a.py"

    @pytest.mark.asyncio
    async def test_sync_no_scan_checkpoint_is_backward_compatible(
        self, mock_ctx: MagicMock
    ) -> None:
        """project_sync with non-scan checkpoint must behave exactly as before."""
        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store", memory_ids=["m-1"], stored_count=1
                    ),
                )
            )

            result = await sync(
                checkpoint={
                    "version": "oss-r3",
                    "scope": {"palace": "acme"},
                    "plan": [{"operation": "ingest", "payload": {"content": "x"}}],
                    "next_index": 0,
                    "completed": [],
                },
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert payload["completed_operations"] == ["ingest"]
        # No scan_snapshot in checkpoint for non-scan flows
        assert payload["checkpoint"].get("scan") is None
        assert payload["checkpoint"].get("scan_snapshot") is None


class TestScanMemoryIdPropagation:
    """Verify blocker-2 fix: memory_ids returned from ingest are written back to snapshot."""

    @pytest.mark.asyncio
    async def test_onboard_scan_populates_memory_id_in_snapshot(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """After project_onboard with scan, snapshot entries must carry the returned memory_id."""
        (workspace_tmp / "mod.py").write_text("x = 1\n")

        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store", memory_ids=["mem-abc-1"], stored_count=1
                    ),
                )
            )

            result = await onboard(
                scope={"palace": "proj"},
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed", payload
        snap = payload["checkpoint"]["scan_snapshot"]
        entries = snap["entries"]
        assert len(entries) == 1
        assert entries[0]["path"] == "mod.py"
        # memory_id must be back-populated from the ingest result
        assert entries[0]["memory_id"] == "mem-abc-1"

    @pytest.mark.asyncio
    async def test_sync_scan_populates_memory_id_in_snapshot(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """After project_sync with scan, newly-ingested file entries get their memory_id."""
        (workspace_tmp / "added.py").write_text("y = 2\n")

        prior_snap = ScanSnapshot(
            scan_config=ScanConfig(patterns=["*.py"], root=str(workspace_tmp)),
            entries=[],  # No prior entries → added.py is detected as added
        )
        checkpoint = {
            "version": "oss-r3",
            "scope": {"palace": "proj"},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 0,
            "completed": [],
            "scan_snapshot": prior_snap.model_dump(),
        }

        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store", memory_ids=["mem-sync-99"], stored_count=1
                    ),
                )
            )

            result = await sync(
                checkpoint=checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed", payload
        snap = payload["checkpoint"]["scan_snapshot"]
        entries = snap["entries"]
        # The new snapshot should contain the added file with memory_id populated
        assert len(entries) == 1
        assert entries[0]["path"] == "added.py"
        assert entries[0]["memory_id"] == "mem-sync-99"


# ============================================================================
# Blocker 1: scan→structured-ingest embedding failure produces deterministic error
# ============================================================================


class TestScanStructuredIngestEmbeddingFailure:
    """Verify that embedding failures in the scan→structured-ingest path produce a
    deterministic MEM_EMBEDDING_FAILED error code rather than MEM_INTERNAL_ERROR.

    All existing tests mock MemoryService entirely, meaning the real
    _manage_ingest_structured path (which calls compute_embedding) was never exercised.
    This suite tests through the real MemoryService while only mocking the backend
    and the embedding call.
    """

    @pytest.mark.asyncio
    async def test_embedding_failure_in_memory_tool_returns_mem_embedding_failed(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """Direct memory(operation='ingest', format='structured') with embedding failure
        must return MEM_EMBEDDING_FAILED, not MEM_INTERNAL_ERROR."""
        backend_mock = _make_backend_mock()
        backend_mock.begin_transaction = AsyncMock()
        backend_mock.rollback = AsyncMock()
        backend_mock.commit = AsyncMock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch(
                "workflows_mcp.engine.memory_service.compute_embedding",
                side_effect=ValueError("ExecutionContext not available"),
            ),
        ):
            result = await memory(
                operation="ingest",
                scope={
                    "palace": "acme",
                    "wing": "svc",
                    "room": "component",
                    "compartment": "topic",
                },
                record={
                    "format": "structured",
                    "memories": [{"content": "def foo(): pass"}],
                },
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert "error" in payload, f"Expected error envelope, got: {payload}"
        assert payload["error"]["code"] == "MEM_EMBEDDING_FAILED", (
            f"Expected MEM_EMBEDDING_FAILED but got: {payload['error']['code']!r}. "
            "Embedding failures must surface with a deterministic code, not MEM_INTERNAL_ERROR."
        )
        assert payload["error"]["retryable"] is True, (
            "MEM_EMBEDDING_FAILED must be retryable (transient infrastructure failure)"
        )

    @pytest.mark.asyncio
    async def test_scan_ingest_embedding_failure_returns_mem_embedding_failed(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """project_onboard with scan that triggers structured ingest, when compute_embedding
        fails, must return MEM_EMBEDDING_FAILED in the checkpoint error — not MEM_INTERNAL_ERROR.

        This is Blocker 1: scan→structured-ingest path through real MemoryService."""
        (workspace_tmp / "service.py").write_text("def handle(): pass\n")

        backend_mock = _make_backend_mock()
        backend_mock.begin_transaction = AsyncMock()
        backend_mock.rollback = AsyncMock()
        backend_mock.commit = AsyncMock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch(
                "workflows_mcp.engine.memory_service.compute_embedding",
                side_effect=ValueError("ExecutionContext not available"),
            ),
        ):
            result = await onboard(
                scope={"palace": "proj", "wing": "svc", "room": "comp", "compartment": "t"},
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload.get("status") in {"checkpoint", "failed"}, (
            f"Expected checkpoint/failed status on embedding failure, got: {payload}"
        )
        error = payload.get("error", {})
        assert error.get("code") == "MEM_EMBEDDING_FAILED", (
            f"Expected MEM_EMBEDDING_FAILED but got: {error.get('code')!r}. "
            "Scan→structured-ingest embedding failures must not surface as MEM_INTERNAL_ERROR."
        )

    @pytest.mark.asyncio
    async def test_snapshot_memory_ids_enable_archive_on_delete(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """Snapshot memory_ids from a prior ingest are used for archive when a file is deleted."""
        from workflows_mcp.tools_memory import FileSnapshotEntry

        (workspace_tmp / "kept.py").write_text("kept = 1\n")
        # deleted.py is NOT created on disk — only in the prior snapshot with a memory_id.

        prior_snap = ScanSnapshot(
            scan_config=ScanConfig(
                patterns=["*.py"],
                root=str(workspace_tmp),
                deletion_policy="archive",
            ),
            entries=[
                FileSnapshotEntry(
                    path="kept.py",
                    size_bytes=9,
                    mtime_ns=1,
                    content_hash=_hash_content("kept = 1\n"),
                    memory_id="mem-kept-1",
                ),
                FileSnapshotEntry(
                    path="deleted.py",
                    size_bytes=10,
                    mtime_ns=1,
                    content_hash=_hash_content("gone = 1\n"),
                    memory_id="mem-deleted-42",
                ),
            ],
        )
        checkpoint = {
            "version": "oss-r3",
            "scope": {"palace": "proj"},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 0,
            "completed": [],
            "scan_snapshot": prior_snap.model_dump(),
        }

        backend_mock = _make_backend_mock()
        archive_records: list[Any] = []

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _capture(request: Any) -> MemoryResult:
                if request.operation == "ingest":
                    return MemoryResult(
                        operation="ingest",
                        manage=ManageMemoryResult(
                            operation="store", memory_ids=["mem-kept-new"], stored_count=1
                        ),
                    )
                if request.operation == "archive":
                    archive_records.append(request.record)
                    return MemoryResult(
                        operation="archive",
                        manage=ManageMemoryResult(
                            operation="archive", memory_ids=["mem-deleted-42"], archived_count=1
                        ),
                    )
                raise AssertionError(f"Unexpected op: {request.operation}")

            mock_service.execute = AsyncMock(side_effect=_capture)

            result = await sync(
                checkpoint=checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed", payload
        # Archive must have been called with the deleted file's memory_id
        assert len(archive_records) == 1
        archived_ids = (
            archive_records[0].ids
            if hasattr(archive_records[0], "ids")
            else archive_records[0].get("ids", [])
        )
        assert "mem-deleted-42" in archived_ids


# ============================================================================
# A) Single scan root: WORKFLOWS_SCAN_ROOT
# ============================================================================


class TestScanRoot:
    """Verify WORKFLOWS_SCAN_ROOT env var controls the single allowed scan root."""

    def test_get_scan_root_defaults_to_filesystem_root_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When WORKFLOWS_SCAN_ROOT is unset, _get_scan_root must return Path('/')."""
        monkeypatch.delenv("WORKFLOWS_SCAN_ROOT", raising=False)
        root = _get_scan_root()
        assert root == Path("/")

    def test_get_scan_root_returns_resolved_env_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When WORKFLOWS_SCAN_ROOT is set, _get_scan_root returns that resolved path."""
        custom_root = tmp_path / "myrepo"
        custom_root.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(custom_root))
        root = _get_scan_root()
        assert root == custom_root.resolve()

    def test_get_scan_root_blank_env_falls_back_to_filesystem_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blank WORKFLOWS_SCAN_ROOT must behave as if unset (root = '/')."""
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", "   ")
        root = _get_scan_root()
        assert root == Path("/")

    def test_path_under_override_root_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A path inside the WORKFLOWS_SCAN_ROOT override must not raise."""
        custom_root = tmp_path / "allowed_repo"
        custom_root.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(custom_root))
        path_inside = custom_root / "src"
        path_inside.mkdir()
        # Must not raise
        _validate_scan_path_within_workspace(path_inside, "scan.root")

    def test_path_outside_override_root_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A path outside WORKFLOWS_SCAN_ROOT must raise MEM_SCAN_PATH_OUT_OF_ROOT
        with a message mentioning the env var and the single allowed root."""
        custom_root = tmp_path / "allowed_repo"
        custom_root.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(custom_root))

        outside_path = tmp_path / "not_in_root"
        outside_path.mkdir()

        with pytest.raises(MemoryContractError) as exc_info:
            _validate_scan_path_within_workspace(outside_path, "scan.root")

        err = exc_info.value
        assert err.code == "MEM_SCAN_PATH_OUT_OF_ROOT"
        # Message must mention the env var and the root for actionability
        assert "WORKFLOWS_SCAN_ROOT" in err.message
        assert str(custom_root.resolve()) in err.message

    def test_default_root_accepts_any_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With no WORKFLOWS_SCAN_ROOT set, any absolute path must be accepted (root is '/')."""
        monkeypatch.delenv("WORKFLOWS_SCAN_ROOT", raising=False)
        # tmp_path is always absolute — should be accepted under root='/'
        _validate_scan_path_within_workspace(tmp_path.resolve(), "scan.root")

    @pytest.mark.asyncio
    async def test_override_root_accepted_via_project_onboard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """project_onboard must accept scan.root inside WORKFLOWS_SCAN_ROOT override."""
        custom_root = tmp_path / "cross_repo"
        custom_root.mkdir()
        (custom_root / "main.py").write_text("def main(): pass\n")
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(custom_root))

        mock_ctx_obj = MagicMock()
        app_ctx_obj = MagicMock()
        app_ctx_obj.memory_backend = None
        app_ctx_obj.memory_backend_lock = None
        mock_ctx_obj.request_context.lifespan_context = app_ctx_obj
        exec_context_obj = MagicMock()
        exec_context_obj.user_string_id = None
        app_ctx_obj.create_execution_context.return_value = exec_context_obj
        app_ctx_obj.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")

        backend_mock = _make_backend_mock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                return_value=MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store", memory_ids=["m-cross-1"], stored_count=1
                    ),
                )
            )
            result = await onboard(
                scope={"palace": "cross"},
                scan={"patterns": ["*.py"], "root": str(custom_root)},
                max_operations=5,
                ctx=mock_ctx_obj,
            )

        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", (
            f"Expected completed with override root, got: {payload}"
        )

    @pytest.mark.asyncio
    async def test_out_of_override_root_rejected_via_project_onboard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """project_onboard must reject scan.root outside WORKFLOWS_SCAN_ROOT override."""
        custom_root = tmp_path / "allowed"
        custom_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(custom_root))

        mock_ctx_obj = MagicMock()
        app_ctx_obj = MagicMock()
        app_ctx_obj.memory_backend = None
        app_ctx_obj.memory_backend_lock = None
        mock_ctx_obj.request_context.lifespan_context = app_ctx_obj
        exec_context_obj = MagicMock()
        exec_context_obj.user_string_id = None
        app_ctx_obj.create_execution_context.return_value = exec_context_obj
        app_ctx_obj.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")

        result = await onboard(
            scope={"palace": "cross"},
            scan={"patterns": ["*.py"], "root": str(outside)},
            ctx=mock_ctx_obj,
        )

        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_SCAN_PATH_OUT_OF_ROOT", (
            f"Expected MEM_SCAN_PATH_OUT_OF_ROOT for out-of-root path, got: {payload}"
        )


# ============================================================================
# B) Runtime schema discoverability: memory(operation="schema")
# ============================================================================


class TestMemorySchemaOperation:
    """Verify memory(operation='schema') returns a contract snapshot without DB."""

    @pytest.mark.asyncio
    async def test_schema_operation_returns_dict_without_db(self, mock_ctx: MagicMock) -> None:
        """memory(operation='schema') must return a schema/contract snapshot dict,
        not raise an error, and not require DB connectivity."""
        # Deliberately do NOT patch PostgresBackend — schema must not need it
        result = await memory(
            operation="schema",
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "error" not in payload, f"schema operation returned error: {payload}"

    @pytest.mark.asyncio
    async def test_schema_operation_contains_expected_keys(self, mock_ctx: MagicMock) -> None:
        """memory(operation='schema') response must include schema, version, and operations keys."""
        result = await memory(
            operation="schema",
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert "version" in payload, f"Missing 'version' key in schema response: {payload}"
        assert "operations" in payload, f"Missing 'operations' key in schema response: {payload}"
        assert "scan" in payload, f"Missing 'scan' key in schema response: {payload}"
        assert "checkpoint" in payload, f"Missing 'checkpoint' key in schema response: {payload}"

    @pytest.mark.asyncio
    async def test_schema_operation_checkpoint_version_matches_flow_version(
        self, mock_ctx: MagicMock
    ) -> None:
        """Checkpoint version in schema response must match _PROJECT_FLOW_VERSION."""
        from workflows_mcp.tools_memory import _PROJECT_FLOW_VERSION

        result = await memory(
            operation="schema",
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload.get("checkpoint", {}).get("version") == _PROJECT_FLOW_VERSION

    @pytest.mark.asyncio
    async def test_schema_operation_lists_known_operations(self, mock_ctx: MagicMock) -> None:
        """Schema response operations list must contain the standard memory operations."""
        result = await memory(
            operation="schema",
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        ops = payload.get("operations", [])
        for expected_op in ("query", "ingest", "supersede", "archive", "maintain", "schema"):
            assert expected_op in ops, (
                f"Expected operation {expected_op!r} in schema.operations, got: {ops}"
            )

    @pytest.mark.asyncio
    async def test_schema_operation_scan_includes_deletion_policies(
        self, mock_ctx: MagicMock
    ) -> None:
        """Schema response scan section must document deletion_policy options."""
        result = await memory(
            operation="schema",
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        scan = payload.get("scan", {})
        deletion_policies = scan.get("deletion_policies", [])
        for policy in ("archive", "supersede", "ignore"):
            assert policy in deletion_policies, (
                "Expected deletion policy "
                f"{policy!r} in schema.scan.deletion_policies: {deletion_policies}"
            )

    @pytest.mark.asyncio
    async def test_schema_operation_includes_query_section(self, mock_ctx: MagicMock) -> None:
        """Issue 2 regression: schema payload must include a 'query' section with contract details.

        Agents use the schema response to discover how to call memory(operation='query').
        The schema must document the required shape so agents don't have to guess.
        """
        result = await memory(operation="schema", ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        assert "query" in payload, (
            f"schema payload must contain a 'query' section; got keys: {list(payload)}"
        )
        query_section = payload["query"]
        assert isinstance(query_section, dict), "schema.query must be an object"

        # Must include at least one concrete example showing the minimum valid shape
        assert "examples" in query_section, (
            "schema.query must include 'examples' showing minimum valid call shapes"
        )
        examples = query_section["examples"]
        assert isinstance(examples, list) and len(examples) >= 1, (
            "schema.query.examples must be a non-empty list"
        )
        # At least one example must demonstrate the text-search pattern
        text_example_present = any(
            isinstance(ex, dict) and ex.get("query", {}).get("text") is not None
            for ex in examples
        )
        assert text_example_present, (
            "schema.query.examples must include at least one example with "
            "{'query': {'text': '...'}} to show minimum valid shape"
        )

        # Must document supported query keys so agents know what fields are valid
        assert "supported_keys" in query_section, (
            "schema.query must include 'supported_keys' listing valid query object fields"
        )
        supported_keys = query_section["supported_keys"]
        assert isinstance(supported_keys, list), "schema.query.supported_keys must be a list"
        assert "text" in supported_keys, (
            "'text' must appear in schema.query.supported_keys (primary search field)"
        )


class TestProjectOnboardScanThenSync:
    """Live-symptom regression: project_onboard(scan=...) completed -> project_sync must not replay.

    Root cause investigation: when project_onboard finishes all steps in one call and returns
    status='completed' with a checkpoint that includes scan/scan_snapshot, a subsequent call to
    project_sync(checkpoint=that) was observed to replay the ingest step in MCP trials.

    These tests use the actual tool functions end-to-end (with mocked MemoryService/backend)
    to reproduce the exact call shape seen in MCP: onboard returns the checkpoint dict, sync
    receives it verbatim.
    """

    @pytest.mark.asyncio
    async def test_project_onboard_scan_completed_then_sync_does_not_replay(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """project_onboard(scan=...) returning status='completed' must not cause project_sync
        to replay any operations when called with the returned checkpoint.
        """
        (workspace_tmp / "main.py").write_text("print('hello')\n")
        scan_cfg = {"patterns": ["*.py"], "root": str(workspace_tmp)}

        backend_mock = _make_backend_mock()
        execute_call_count = 0

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _execute(request: Any) -> MemoryResult:
                nonlocal execute_call_count
                execute_call_count += 1
                return MemoryResult(
                    operation="ingest",
                    manage=ManageMemoryResult(
                        operation="store",
                        memory_ids=["m-scan-1"],
                        stored_count=1,
                    ),
                )

            mock_service.execute = AsyncMock(side_effect=_execute)

            # Phase 1: project_onboard with scan — single ingest step should complete in one call
            onboard_result = await onboard(
                scope={"palace": "scanproj"},
                ingest={"format": "raw", "content": "print('hello')"},
                scan=scan_cfg,
                max_operations=5,
                ctx=mock_ctx,
            )

        onboard_payload = json.loads(onboard_result.content[0].text)
        assert onboard_payload.get("status") == "completed", (
            "project_onboard with single ingest step must complete in one call; "
            f"got: {onboard_payload}"
        )
        ingest_count_after_onboard = execute_call_count

        # The returned checkpoint must include scan/scan_snapshot so project_sync can detect delta
        returned_checkpoint = onboard_payload["checkpoint"]
        assert "scan_snapshot" in returned_checkpoint, (
            "Completed checkpoint from project_onboard(scan=...) must include scan_snapshot"
        )

        # Phase 2: project_sync receives the completed checkpoint — must NOT replay
        execute_call_count = 0

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls2,
        ):
            mock_service2 = mock_service_cls2.return_value

            async def _should_not_be_called(*_a: Any, **_kw: Any) -> None:
                nonlocal execute_call_count
                execute_call_count += 1
                raise AssertionError(
                    "project_sync must not execute any memory operation on a "
                    "completed checkpoint returned by project_onboard"
                )

            mock_service2.execute = _should_not_be_called

            sync_result = await sync(
                checkpoint=returned_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        sync_payload = json.loads(sync_result.content[0].text)
        assert sync_payload.get("status") == "completed", (
            f"project_sync must return status='completed' for a completed onboard checkpoint; "
            f"got: {sync_payload}"
        )
        assert execute_call_count == 0, (
            f"project_sync executed {execute_call_count} memory operation(s) on a completed "
            f"checkpoint — this is the live-symptom replay bug"
        )
        sync_ckpt = sync_payload.get("checkpoint", {})
        assert sync_ckpt.get("next_index") == returned_checkpoint.get("next_index"), (
            "next_index must remain stable after project_sync on a completed checkpoint"
        )
        _ = ingest_count_after_onboard  # used for context; ingest ran exactly once in onboard

    @pytest.mark.asyncio
    async def test_project_onboard_scan_checkpoint_then_sync_does_not_replay(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """project_onboard(scan=..., max_operations=1) returning status='checkpoint' followed by
        project_sync resuming to completion must execute each step exactly once (no replay).
        """
        (workspace_tmp / "app.py").write_text("x = 1\n")
        scan_cfg = {"patterns": ["*.py"], "root": str(workspace_tmp)}

        backend_mock = _make_backend_mock()
        execute_log: list[str] = []

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _execute(request: Any) -> MemoryResult:
                execute_log.append(request.operation)
                return MemoryResult(
                    operation=request.operation,
                    manage=ManageMemoryResult(
                        operation="store",
                        memory_ids=[f"m-{request.operation}"],
                        stored_count=1,
                    ),
                )

            mock_service.execute = AsyncMock(side_effect=_execute)

            # Step 1: onboard with max_operations=1 → produces a checkpoint
            onboard_result = await onboard(
                scope={"palace": "scanproj2"},
                ingest={"format": "raw", "content": "x = 1"},
                maintain={"mode": "community_refresh"},
                scan=scan_cfg,
                max_operations=1,
                ctx=mock_ctx,
            )
            onboard_payload = json.loads(onboard_result.content[0].text)
            assert onboard_payload.get("status") == "checkpoint", (
                "Expected status='checkpoint' with max_operations=1 and 2-step plan; "
                f"got: {onboard_payload}"
            )
            mid_checkpoint = onboard_payload["checkpoint"]
            assert mid_checkpoint["next_index"] == 1

            # Step 2: sync resumes and completes the remaining step
            sync_result = await sync(
                checkpoint=mid_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )
            sync_payload = json.loads(sync_result.content[0].text)
            assert sync_payload.get("status") == "completed", (
                f"project_sync must complete the remaining steps; got: {sync_payload}"
            )
            completed_checkpoint = sync_payload["checkpoint"]

        # Each operation must appear exactly once
        assert execute_log.count("ingest") == 1, (
            f"ingest must run exactly once across onboard+sync; got log={execute_log}"
        )
        assert execute_log.count("maintain") == 1, (
            f"maintain must run exactly once across onboard+sync; got log={execute_log}"
        )

        # Step 3: sync again on the completed checkpoint → no replay
        execute_log.clear()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls3,
        ):
            mock_service3 = mock_service_cls3.return_value

            async def _should_not_be_called(*_a: Any, **_kw: Any) -> None:
                raise AssertionError("project_sync must not replay on a completed checkpoint")

            mock_service3.execute = _should_not_be_called

            final_sync = await sync(
                checkpoint=completed_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        final_payload = json.loads(final_sync.content[0].text)
        assert final_payload.get("status") == "completed"
        assert execute_log == [], (
            "No operations must run on completed checkpoint; "
            f"got: {execute_log}"
        )


class TestProjectSyncCompletedCheckpointProgression:
    """Issue 1 regression: completed checkpoints must not replay completed steps."""

    @pytest.mark.asyncio
    async def test_sync_completed_checkpoint_without_scan_is_stable(
        self, mock_ctx: MagicMock
    ) -> None:
        """A completed checkpoint (next_index == len(plan)) passed to project_sync must return
        status='completed' immediately without replaying any operations.
        """
        backend_mock = _make_backend_mock()
        execute_call_count = 0

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _track_and_fail(*_a: Any, **_kw: Any) -> None:
                nonlocal execute_call_count
                execute_call_count += 1
                raise AssertionError(
                    "project_sync must not execute any step on a completed checkpoint"
                )

            mock_service.execute = _track_and_fail

            completed_checkpoint = {
                "version": "oss-r3",
                "scope": {"palace": "myproj"},
                "plan": [{"operation": "ingest", "payload": {"content": "data"}}],
                "next_index": 1,  # == len(plan) → already completed
                "completed": [{"operation": "ingest", "result": {"stored": 1}}],
            }

            result = await sync(
                checkpoint=completed_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", (
            f"Expected status='completed' for completed checkpoint; got: {payload}"
        )
        assert execute_call_count == 0, (
            "project_sync must not execute any memory operation for an already-completed checkpoint"
        )
        returned_ckpt = payload.get("checkpoint", {})
        assert returned_ckpt.get("next_index") == 1, (
            f"next_index must remain stable at len(plan)=1; got {returned_ckpt.get('next_index')}"
        )

    @pytest.mark.asyncio
    async def test_sync_completed_checkpoint_with_scan_is_stable(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """A completed checkpoint that also carries scan/scan_snapshot must not be replayed
        when passed back to project_sync, even when scanned files have changed.

        Root cause: project_sync was rebuilding a new plan from scan delta and resetting
        next_index=0, causing the ingest to re-run.
        """
        (workspace_tmp / "file.py").write_text("x = 1\n")
        scan_cfg = {"patterns": ["*.py"], "root": str(workspace_tmp)}

        # Build a completed checkpoint that includes a scan_snapshot
        snap_entry = {
            "path": "file.py",
            "size_bytes": 6,
            "mtime_ns": 1000,
            "content_hash": "aabbcc",
            "memory_id": "m-existing",
        }
        completed_checkpoint = {
            "version": "oss-r3",
            "scope": {"palace": "myproj"},
            "plan": [{"operation": "ingest", "payload": {"format": "raw", "content": "x = 1"}}],
            "next_index": 1,  # == len(plan) → already completed
            "completed": [{"operation": "ingest", "result": {"stored": 1}}],
            "scan": scan_cfg,
            "scan_snapshot": {
                "scan_config": scan_cfg,
                "entries": [snap_entry],
            },
        }

        backend_mock = _make_backend_mock()
        execute_call_count = 0

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _track_and_fail(*_a: Any, **_kw: Any) -> None:
                nonlocal execute_call_count
                execute_call_count += 1
                raise AssertionError(
                    "project_sync must not execute any step on a completed checkpoint"
                )

            mock_service.execute = _track_and_fail

            result = await sync(
                checkpoint=completed_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", (
            f"Completed checkpoint with scan must remain completed; got: {payload}"
        )
        assert execute_call_count == 0, (
            "project_sync must not re-execute any step for a completed scan checkpoint"
        )
        returned_ckpt = payload.get("checkpoint", {})
        assert returned_ckpt.get("next_index") == 1, (
            "next_index must remain stable at len(plan)=1 for completed scan checkpoint"
        )


class TestProjectSyncFastPathHardening:
    """QA hardening tests for project_sync completed-checkpoint fast-path.

    Covers:
    - from_checkpoint flag in completed response
    - float integral next_index coercion (e.g. 1.0 → 1)
    - non-integral float next_index yields MEM_CHECKPOINT_INVALID
    - no memory operation executed on fast-path in all these cases
    """

    _COMPLETED_CHECKPOINT: dict[str, Any] = {
        "version": "oss-r3",
        "scope": {"palace": "hardenproj"},
        "plan": [{"operation": "ingest", "payload": {"content": "data"}}],
        "next_index": 1,
        "completed": [{"operation": "ingest", "result": {"stored": 1}}],
    }

    @pytest.mark.asyncio
    async def test_completed_checkpoint_response_includes_from_checkpoint_flag(
        self, mock_ctx: MagicMock
    ) -> None:
        """project_sync fast-path response must include from_checkpoint=True."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.execute = AsyncMock(
                side_effect=AssertionError("must not execute on fast-path")
            )

            result = await sync(
                checkpoint=self._COMPLETED_CHECKPOINT,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", f"Expected completed; got: {payload}"
        assert payload.get("from_checkpoint") is True, (
            f"Completed fast-path response must include from_checkpoint=True; got: {payload}"
        )

    @pytest.mark.asyncio
    async def test_float_integral_next_index_takes_fast_path(
        self, mock_ctx: MagicMock
    ) -> None:
        """next_index=1.0 (integral float) must be accepted and take the fast-path."""
        checkpoint = {**self._COMPLETED_CHECKPOINT, "next_index": 1.0}
        backend_mock = _make_backend_mock()
        execute_call_count = 0
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _track(*_a: Any, **_kw: Any) -> None:
                nonlocal execute_call_count
                execute_call_count += 1
                raise AssertionError("must not execute on fast-path")

            mock_service.execute = _track

            result = await sync(
                checkpoint=checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload.get("status") == "completed", (
            f"next_index=1.0 must complete via fast-path; got: {payload}"
        )
        assert execute_call_count == 0, (
            "No memory operation must run on fast-path for float integral next_index"
        )
        assert payload.get("from_checkpoint") is True, (
            "Completed fast-path response must include from_checkpoint=True for float integral"
        )

    @pytest.mark.asyncio
    async def test_non_integral_float_next_index_yields_checkpoint_invalid(
        self, mock_ctx: MagicMock
    ) -> None:
        """next_index=1.5 (non-integral float) must yield MEM_CHECKPOINT_INVALID envelope."""
        checkpoint = {**self._COMPLETED_CHECKPOINT, "next_index": 1.5}
        backend_mock = _make_backend_mock()
        execute_call_count = 0
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value

            async def _track(*_a: Any, **_kw: Any) -> None:
                nonlocal execute_call_count
                execute_call_count += 1
                raise AssertionError("must not execute on non-integral next_index")

            mock_service.execute = _track

            result = await sync(
                checkpoint=checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        error = payload.get("error", {})
        assert error.get("code") == "MEM_CHECKPOINT_INVALID", (
            f"non-integral next_index=1.5 must return MEM_CHECKPOINT_INVALID; got: {payload}"
        )
        assert "status" not in payload, (
            f"Error envelope must not include a status key; got: {payload}"
        )
        assert execute_call_count == 0, (
            "No memory operation must run for non-integral float next_index"
        )


# ---------------------------------------------------------------------------
# Compact-by-default / debug expansion tests
# ---------------------------------------------------------------------------


class TestProjectFlowCompactResponse:
    """Verify that project_onboard/project_sync responses are compact by default.

    Default (no response arg or debug=False):
      - checkpoint.completed[].result blobs are stripped
      - checkpoint.scan_snapshot.entries are stripped (summary count only)
      - checkpoint.plan[].payload values are preserved (needed for resume)
      - outer `results` list is absent (replaced by summary counts)
      - status, last_operation, completed_operations, remaining_operations present

    debug=True:
      - full checkpoint with completed[].result blobs present
      - full scan_snapshot entries present
      - outer `results` list present with full result payloads
    """

    _BASE_PLAN = [
        {"operation": "ingest", "payload": {"content": "hello"}},
        {"operation": "archive", "payload": {"ids": ["m-old"]}},
    ]
    _BASE_SCOPE: dict[str, Any] = {"palace": "compacttest"}

    @staticmethod
    def _make_ingest_result() -> MemoryResult:
        return MemoryResult(
            operation="ingest",
            manage=ManageMemoryResult(
                operation="store",
                memory_ids=["m-new"],
                stored_count=1,
            ),
        )

    @staticmethod
    def _make_archive_result() -> MemoryResult:
        return MemoryResult(
            operation="archive",
            manage=ManageMemoryResult(operation="forget", archived_count=2),
        )

    @pytest.mark.asyncio
    async def test_onboard_checkpoint_response_default_strips_completed_results(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default response: checkpoint.completed[].result must be absent."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=lambda req: (
                    self._make_ingest_result()
                    if req.operation == "ingest"
                    else self._make_archive_result()
                )
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        completed_items = payload["checkpoint"]["completed"]
        for item in completed_items:
            assert "result" not in item, (
                f"Default response must strip result from completed items; got: {item}"
            )

    @pytest.mark.asyncio
    async def test_onboard_checkpoint_response_debug_includes_completed_results(
        self, mock_ctx: MagicMock
    ) -> None:
        """debug=True response: checkpoint.completed[].result must be present."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=lambda req: (
                    self._make_ingest_result()
                    if req.operation == "ingest"
                    else self._make_archive_result()
                )
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        completed_items = payload["checkpoint"]["completed"]
        assert len(completed_items) == 1
        assert "result" in completed_items[0], (
            f"debug=True must include result in completed items; got: {completed_items[0]}"
        )

    @pytest.mark.asyncio
    async def test_onboard_completed_default_omits_results_list(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default response: completed status must not include a `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                return_value=self._make_ingest_result()
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" not in payload, (
            f"Default response must not include `results` list; got keys: {list(payload)}"
        )
        assert "completed_operations" in payload

    @pytest.mark.asyncio
    async def test_onboard_completed_debug_includes_results_list(
        self, mock_ctx: MagicMock
    ) -> None:
        """debug=True: completed status must include full `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                return_value=self._make_ingest_result()
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" in payload, (
            f"debug=True must include `results` list; got keys: {list(payload)}"
        )
        assert isinstance(payload["results"], list)

    @pytest.mark.asyncio
    async def test_sync_completed_default_omits_results_list(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default response: project_sync completed must not include a `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                return_value=self._make_ingest_result()
            )
            result = await sync(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" not in payload, (
            f"Default project_sync must not include `results` list; got keys: {list(payload)}"
        )

    @pytest.mark.asyncio
    async def test_sync_completed_debug_includes_results_list(
        self, mock_ctx: MagicMock
    ) -> None:
        """debug=True: project_sync completed must include full `results` list."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                return_value=self._make_ingest_result()
            )
            result = await sync(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                max_operations=5,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert "results" in payload, (
            f"debug=True project_sync must include `results` list; got keys: {list(payload)}"
        )

    @pytest.mark.asyncio
    async def test_checkpoint_plan_payloads_preserved_for_resume(
        self, mock_ctx: MagicMock
    ) -> None:
        """Compact checkpoint must keep plan[].payload intact so resume works."""
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                return_value=self._make_ingest_result()
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        plan = payload["checkpoint"]["plan"]
        assert len(plan) == 2  # noqa: PLR2004
        for step in plan:
            assert "payload" in step, (
                f"Compact checkpoint must preserve plan[].payload; got: {step}"
            )

    @pytest.mark.asyncio
    async def test_compact_checkpoint_is_resumable(
        self, mock_ctx: MagicMock
    ) -> None:
        """The compact checkpoint returned by default must be accepted for resume."""
        backend_mock = _make_backend_mock()
        execute_results = [self._make_ingest_result(), self._make_archive_result()]
        call_index = 0

        async def _side_effect(_req: Any) -> MemoryResult:
            nonlocal call_index
            r = execute_results[call_index % len(execute_results)]
            call_index += 1
            return r

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(side_effect=_side_effect)

            # Step 1: get compact checkpoint from onboard
            r1 = await onboard(
                scope=self._BASE_SCOPE,
                ingest={"content": "hello"},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )
            p1 = json.loads(r1.content[0].text)
            assert p1["status"] == "checkpoint"

            # Step 2: resume with the compact checkpoint
            r2 = await sync(
                checkpoint=p1["checkpoint"],
                max_operations=5,
                ctx=mock_ctx,
            )
            p2 = json.loads(r2.content[0].text)
            assert p2["status"] == "completed", (
                f"Compact checkpoint must be resumable; got: {p2}"
            )
            assert "archive" in p2["completed_operations"]

    @pytest.mark.asyncio
    async def test_scan_snapshot_always_includes_entries_for_resume(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """scan_snapshot.entries must be preserved in checkpoint even in compact mode.

        Entries are required by project_sync for delta computation; stripping
        them would break resume functionality for scan-based flows.
        """
        py_file = workspace_tmp / "sample.py"
        py_file.write_text("x = 1\n")

        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                return_value=self._make_ingest_result()
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        snap = payload["checkpoint"].get("scan_snapshot")
        assert snap is not None, "scan_snapshot must be present in checkpoint"
        # entries must be present (needed for project_sync delta computation)
        assert "entries" in snap, (
            f"scan_snapshot.entries must be preserved for resume; got keys: {list(snap)}"
        )

    @pytest.mark.asyncio
    async def test_scan_snapshot_debug_includes_full_entries(
        self, mock_ctx: MagicMock, workspace_tmp: Path
    ) -> None:
        """debug=True: scan_snapshot in checkpoint must include full entries list."""
        py_file = workspace_tmp / "sample.py"
        py_file.write_text("x = 1\n")

        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                return_value=self._make_ingest_result()
            )
            result = await onboard(
                scope=self._BASE_SCOPE,
                scan={"patterns": ["*.py"], "root": str(workspace_tmp)},
                max_operations=5,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        snap = payload["checkpoint"].get("scan_snapshot")
        assert snap is not None
        assert "entries" in snap, (
            f"debug=True must include full scan_snapshot.entries; got keys: {list(snap)}"
        )

    @pytest.mark.asyncio
    async def test_sync_fast_path_compact_default_omits_results_list(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default: project_sync fast-path (completed checkpoint) must not include results."""
        completed_checkpoint: dict[str, Any] = {
            "version": "oss-r3",
            "scope": {"palace": "compacttest"},
            "plan": [{"operation": "ingest", "payload": {"content": "data"}}],
            "next_index": 1,
            "completed": [{"operation": "ingest", "result": {"stored": 1}}],
        }
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=AssertionError("must not execute on fast-path")
            )
            result = await sync(
                checkpoint=completed_checkpoint,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert payload.get("from_checkpoint") is True
        assert "results" not in payload, (
            f"Default fast-path must not include results list; got keys: {list(payload)}"
        )

    @pytest.mark.asyncio
    async def test_sync_fast_path_debug_includes_results_list(
        self, mock_ctx: MagicMock
    ) -> None:
        """debug=True: project_sync fast-path must include results list."""
        completed_checkpoint: dict[str, Any] = {
            "version": "oss-r3",
            "scope": {"palace": "compacttest"},
            "plan": [{"operation": "ingest", "payload": {"content": "data"}}],
            "next_index": 1,
            "completed": [{"operation": "ingest", "result": {"stored": 1}}],
        }
        backend_mock = _make_backend_mock()
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(
                side_effect=AssertionError("must not execute on fast-path")
            )
            result = await sync(
                checkpoint=completed_checkpoint,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed"
        assert payload.get("from_checkpoint") is True
        assert "results" in payload, (
            f"debug=True fast-path must include results list; got keys: {list(payload)}"
        )


class TestCheckpointPlanPayloadCompaction:
    """Verify compact checkpoint strips plan[].payload for already-completed steps.

    Requirement (A): default checkpoint should not include full memory content in
    completed plan steps' payloads — only pending (not-yet-executed) step payloads
    need to be preserved for resume.

    Tests:
    1. Default onboard/sync response does not contain full plan payload content
       for completed steps.
    2. debug=true includes full plan payload content for all steps.
    3. Default compact checkpoint from onboard can be passed to sync and resume works.
    4. Backward compatibility: sync accepts legacy full checkpoint shape.
    5. No replay regression: pending steps keep their payloads; resume executes them.
    """

    _SCOPE: dict[str, Any] = {"palace": "payloadtest"}
    _LARGE_CONTENT = "x" * 500  # Simulates a large memory body

    @staticmethod
    def _make_ingest_result() -> MemoryResult:
        return MemoryResult(
            operation="ingest",
            manage=ManageMemoryResult(
                operation="store",
                memory_ids=["m-1"],
                stored_count=1,
            ),
        )

    @staticmethod
    def _make_archive_result() -> MemoryResult:
        return MemoryResult(
            operation="archive",
            manage=ManageMemoryResult(operation="forget", archived_count=1),
        )

    @pytest.mark.asyncio
    async def test_default_compact_checkpoint_strips_completed_plan_payload_content(
        self, mock_ctx: MagicMock
    ) -> None:
        """Default (no debug): plan[i].payload.memories[].content must be absent for
        completed steps (those at index < next_index)."""
        backend_mock = _make_backend_mock()
        large_memories = [{"content": self._LARGE_CONTENT}]
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            # Two steps; max_operations=1 → only ingest runs, archive is pending
            result = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": large_memories},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        next_index = payload["checkpoint"]["next_index"]
        plan = payload["checkpoint"]["plan"]
        assert next_index == 1  # ingest completed

        # The completed step (index 0) must NOT contain heavy memories content
        completed_step_payload = plan[0].get("payload", {})
        memories_in_plan = completed_step_payload.get("memories", [])
        for mem in memories_in_plan:
            assert "content" not in mem, (
                "Compact checkpoint must strip content from completed plan step payloads; "
                f"got memory with content: {mem!r}"
            )

        # The pending step (index 1) must keep its payload intact for resume
        pending_step = plan[1]
        assert pending_step.get("payload") is not None

    @pytest.mark.asyncio
    async def test_debug_true_includes_full_plan_payload_content(
        self, mock_ctx: MagicMock
    ) -> None:
        """debug=True: plan[i].payload must be present and intact for all steps."""
        backend_mock = _make_backend_mock()
        large_memories = [{"content": self._LARGE_CONTENT}]
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_ingest_result())
            result = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": large_memories},
                archive={"ids": ["m-old"]},
                max_operations=1,
                response={"debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "checkpoint"
        plan = payload["checkpoint"]["plan"]

        # In debug mode the completed step payload must still carry memories with content
        completed_step_payload = plan[0].get("payload", {})
        memories_in_plan = completed_step_payload.get("memories", [])
        assert len(memories_in_plan) == 1
        assert memories_in_plan[0].get("content") == self._LARGE_CONTENT, (
            "debug=True must include full content in completed plan step payloads"
        )

    @pytest.mark.asyncio
    async def test_compact_checkpoint_from_onboard_is_resumable_by_sync(
        self, mock_ctx: MagicMock
    ) -> None:
        """The compact checkpoint returned by default onboard must be accepted by
        project_sync and allow the remaining steps to execute."""
        backend_mock = _make_backend_mock()
        call_order: list[str] = []

        async def _side_effect(req: Any) -> MemoryResult:
            call_order.append(req.operation)
            if req.operation == "ingest":
                return self._make_ingest_result()
            return self._make_archive_result()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(side_effect=_side_effect)
            large_memories = [{"content": self._LARGE_CONTENT}]

            # Phase 1: onboard runs ingest, returns compact checkpoint
            r1 = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": large_memories},
                archive={"ids": ["m-old"]},
                max_operations=1,
                ctx=mock_ctx,
            )
            p1 = json.loads(r1.content[0].text)
            assert p1["status"] == "checkpoint"
            compact_checkpoint = p1["checkpoint"]

            # Phase 2: sync resumes from compact checkpoint, executes archive
            r2 = await sync(
                checkpoint=compact_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )
            p2 = json.loads(r2.content[0].text)

        assert p2["status"] == "completed", (
            f"Compact checkpoint must allow sync to complete; got: {p2}"
        )
        assert "archive" in p2["completed_operations"], (
            f"Sync must execute the pending archive step; got: {p2['completed_operations']}"
        )
        # ingest ran once (in onboard), archive once (in sync)
        assert call_order == ["ingest", "archive"]

    @pytest.mark.asyncio
    async def test_backward_compat_sync_accepts_legacy_full_checkpoint(
        self, mock_ctx: MagicMock
    ) -> None:
        """project_sync must accept a legacy checkpoint where plan[].payload carries
        full memories content (old format without compaction)."""
        backend_mock = _make_backend_mock()
        large_memories = [{"content": self._LARGE_CONTENT, "metadata": {"path": "a.py"}}]
        full_checkpoint: dict[str, Any] = {
            "version": "oss-r3",
            "scope": self._SCOPE,
            "plan": [
                {
                    "operation": "ingest",
                    "payload": {"format": "structured", "memories": large_memories},
                },
                {"operation": "archive", "payload": {"ids": ["m-old"]}},
            ],
            "next_index": 1,
            "completed": [
                # Legacy full format with result included
                {"operation": "ingest", "result": {"stored": 1, "id": "m-1"}},
            ],
        }
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            mock_svc.return_value.execute = AsyncMock(return_value=self._make_archive_result())
            result = await sync(
                checkpoint=full_checkpoint,
                max_operations=5,
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "completed", (
            f"Sync must accept legacy full checkpoint; got: {payload}"
        )
        assert "archive" in payload["completed_operations"]

    @pytest.mark.asyncio
    async def test_no_replay_regression_pending_steps_keep_payload(
        self, mock_ctx: MagicMock
    ) -> None:
        """Pending steps (index >= next_index) must preserve their full payload
        so they can be executed on resume — no payload stripping on pending steps."""
        backend_mock = _make_backend_mock()
        archive_ids = ["m-a", "m-b", "m-c"]
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch("workflows_mcp.tools_memory.MemoryService") as mock_svc,
        ):
            executed_payloads: list[Any] = []

            async def _capture(req: Any) -> MemoryResult:
                executed_payloads.append(req.record)
                return self._make_ingest_result()

            mock_svc.return_value.execute = AsyncMock(side_effect=_capture)

            # Two steps: ingest then archive; run only first (ingest)
            r1 = await onboard(
                scope=self._SCOPE,
                ingest={"format": "structured", "memories": [{"content": "data"}]},
                archive={"ids": archive_ids},
                max_operations=1,
                ctx=mock_ctx,
            )
            p1 = json.loads(r1.content[0].text)
            assert p1["status"] == "checkpoint"
            compact_cp = p1["checkpoint"]

            # Verify pending archive step still has its payload
            plan = compact_cp["plan"]
            archive_step = next(s for s in plan if s["operation"] == "archive")
            assert archive_step.get("payload", {}).get("ids") == archive_ids, (
                f"Pending archive step must keep its ids payload; got: {archive_step}"
            )
