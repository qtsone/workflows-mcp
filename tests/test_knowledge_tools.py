"""Tests for unified memory MCP tool."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.memory_service import (
    ManageMemoryResult,
    MemoryContractError,
    MemoryResult,
    QueryMemoryResult,
)
from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_memory import register_memory_tools

register_memory_tools(_mcp_server)


def _get_tool_fn(name: str) -> Any:
    tool_manager = _mcp_server._tool_manager
    tool = tool_manager._tools.get(name)
    if tool is None:
        raise ValueError(f"Tool {name!r} not registered")
    return tool.fn


memory = _get_tool_fn("memory")
project_onboard = _get_tool_fn("project_onboard")
project_sync = _get_tool_fn("project_sync")


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
    return ctx


def _make_backend_mock() -> MagicMock:
    backend = MagicMock()
    backend.connect = AsyncMock()
    backend.disconnect = AsyncMock()
    return backend


class TestMemoryToolRegistration:
    def test_memory_tool_registered(self) -> None:
        tool = _mcp_server._tool_manager._tools.get("memory")
        assert tool is not None
        assert tool.description
        assert "Query and update memory" in tool.description
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False

    def test_project_tools_registered_with_actionable_descriptions(self) -> None:
        onboard_tool = _mcp_server._tool_manager._tools.get("project_onboard")
        assert onboard_tool is not None
        assert onboard_tool.description
        assert "start or continue project memory onboarding" in onboard_tool.description.lower()

        sync_tool = _mcp_server._tool_manager._tools.get("project_sync")
        assert sync_tool is not None
        assert sync_tool.description
        assert "continue project memory synchronization" in sync_tool.description.lower()


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

            onboard_result = await project_onboard(
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

            sync_result = await project_sync(
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
            result = await project_sync(
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
            result = await project_onboard(
                checkpoint={
                    "version": "oss-r2",
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
            result = await project_sync(
                checkpoint={
                    "version": "oss-r2",
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
            result = await project_sync(
                checkpoint={
                    "version": "oss-r2",
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
            result = await project_sync(
                checkpoint={
                    "version": "oss-r2",
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
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )
            await memory(
                operation="query",
                query={"text": "find this too"},
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
                    response={"mode": "compact", "debug": False},
                    ctx=mock_ctx,
                ),
                memory(
                    operation="query",
                    query={"text": "find this too"},
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

            result = await project_onboard(
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

            result = await project_sync(
                checkpoint={
                    "version": "oss-r2",
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

            result = await project_onboard(
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
            assert payload["error"]["message"] == "project_onboard failed"
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

            result = await project_sync(
                checkpoint={
                    "version": "oss-r2",
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
            assert payload["error"]["message"] == "project_sync failed"
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
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload == {"communities_updated": 2}
