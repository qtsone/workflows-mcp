"""Tests for unified memory MCP tool."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.memory_service import ManageMemoryResult, MemoryResult, QueryMemoryResult
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
        assert "Unified memory operations" in tool.description
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False


class TestMemoryTool:
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
                record={
                    "format": "structured",
                    "memories": [{"content": "fails before writes"}],
                },
                response={"mode": "compact", "debug": True},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        assert payload["error"] == "memory failed"
        assert payload.get("details")
        assert "begin" in payload["details"].lower()
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
                scope={"wing": "svc", "room": "component", "hall": "topic"},
                query={"text": "find this", "mode": "search", "radius": 1, "precision": 0.5},
                response={"mode": "compact", "debug": False},
                ctx=mock_ctx,
            )

            payload = json.loads(result.content[0].text)
            assert payload["facts"][0]["content"] == "fact one"

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
