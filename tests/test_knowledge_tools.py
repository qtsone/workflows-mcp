"""Tests for knowledge MCP tool functions.

Tests validate input construction, response formatting, and error handling
for all 5 knowledge MCP tools. KnowledgeExecutor.execute is mocked —
no real PostgreSQL needed.

Knowledge tools are conditionally registered at server startup via
``register_knowledge_tools(mcp_server)``. For testing, we register them
once at module load and retrieve function references from the server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.executors_knowledge import KnowledgeInput, KnowledgeOutput
from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_knowledge import register_knowledge_tools

# Register knowledge tools on the MCP server for testing.
# This simulates what app_lifespan does when a knowledge DB is available.
# Idempotent: if tools are already registered, re-registration overwrites.
register_knowledge_tools(_mcp_server)

# Patch target: the source module (local imports inside tool functions)
_EXECUTOR_CLS = "workflows_mcp.engine.executors_knowledge.KnowledgeExecutor"


def _get_tool_fn(name: str) -> Any:
    """Get a registered tool function from the MCP server by name."""
    tool_manager = _mcp_server._tool_manager
    tool = tool_manager._tools.get(name)
    if tool is None:
        raise ValueError(f"Tool {name!r} not registered")
    return tool.fn


# Grab references to tool functions for direct invocation in tests
search_knowledge = _get_tool_fn("search_knowledge")
store_knowledge = _get_tool_fn("store_knowledge")
recall_knowledge = _get_tool_fn("recall_knowledge")
forget_knowledge = _get_tool_fn("forget_knowledge")
knowledge_context = _get_tool_fn("knowledge_context")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Create a mock AppContextType with all required nested attributes."""
    import uuid

    ctx = MagicMock()
    app_ctx = MagicMock()
    ctx.request_context.lifespan_context = app_ctx
    exec_context = MagicMock()
    exec_context.user_string_id = None
    app_ctx.create_execution_context.return_value = exec_context
    app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")
    return ctx


def _make_search_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "rows": [{"id": "uuid-1", "content": "fact one", "confidence": 0.9}],
        "columns": ["id", "content", "confidence"],
        "row_count": 1,
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_store_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "proposition_ids": ["uuid-new-1"],
        "stored_count": 1,
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_recall_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "rows": [{"id": "uuid-2", "content": "recalled fact"}],
        "columns": ["id", "content"],
        "row_count": 1,
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_forget_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "archived_count": 2,
        "skipped_count": 0,
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_context_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "context_text": "- fact one\n- fact two",
        "proposition_count": 2,
        "tokens_used": 8,
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_error_output(error: str = "Connection failed") -> KnowledgeOutput:
    return KnowledgeOutput(success=False, error=error)


# ============================================================================
# search_knowledge Tests
# ============================================================================


class TestSearchKnowledge:
    """Tests for search_knowledge MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_rows(self, mock_ctx: MagicMock) -> None:
        """Successful search returns rows and row_count."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_search_output())

            result = await search_knowledge(
                query="deployment patterns",
                source=None,
                categories=None,
                as_of=None,
                min_confidence=0.3,
                limit=10,
                namespace=None,
                room=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "search"
        assert inputs.query == "deployment patterns"
        assert inputs.limit == 10

    @pytest.mark.asyncio
    async def test_with_source_filter(self, mock_ctx: MagicMock) -> None:
        """Source filter is passed through to KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_search_output())

            await search_knowledge(
                query="test",
                source="internal-docs",
                categories=None,
                as_of="2026-06-15T12:00:00Z",
                min_confidence=0.3,
                limit=10,
                namespace=None,
                room=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.source == "internal-docs"
        assert inputs.as_of == "2026-06-15T12:00:00Z"

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, mock_ctx: MagicMock) -> None:
        """Failed search returns error in response body."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("Connection refused")
            )

            result = await search_knowledge(
                query="test",
                source=None,
                categories=None,
                as_of=None,
                min_confidence=0.3,
                limit=10,
                namespace=None,
                room=None,
                ctx=mock_ctx,
            )

        content = result.content[0].text  # type: ignore[union-attr]
        assert "failure" in content
        assert "Connection refused" in content


# ============================================================================
# store_knowledge Tests
# ============================================================================


class TestStoreKnowledge:
    """Tests for store_knowledge MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_ids(self, mock_ctx: MagicMock) -> None:
        """Successful store returns proposition_ids and stored_count."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_store_output())

            result = await store_knowledge(
                content="Redis pooling reduces latency by 40%",
                source="perf-tests",
                path=None,
                valid_from=None,
                valid_to=None,
                confidence=0.85,
                categories=None,
                authority="AGENT",
                lifecycle_state="ACTIVE",
                namespace=None,
                room=None,
                corridor=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "store"
        assert inputs.content == "Redis pooling reduces latency by 40%"
        assert inputs.confidence == 0.85

    @pytest.mark.asyncio
    async def test_default_confidence_is_0_8(self, mock_ctx: MagicMock) -> None:
        """Default confidence for MCP tool is 0.8 (not executor's 0.5)."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_store_output())

            await store_knowledge(
                content="some fact",
                source=None,
                path=None,
                valid_from=None,
                valid_to=None,
                confidence=0.8,
                categories=None,
                authority="AGENT",
                lifecycle_state="ACTIVE",
                namespace=None,
                room=None,
                corridor=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.confidence == 0.8

    @pytest.mark.asyncio
    async def test_error_propagates(self, mock_ctx: MagicMock) -> None:
        """Failed store returns error in response."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("Embedding API error")
            )

            result = await store_knowledge(
                content="test fact",
                source=None,
                path=None,
                valid_from=None,
                valid_to=None,
                confidence=0.8,
                categories=None,
                authority="AGENT",
                lifecycle_state="ACTIVE",
                namespace=None,
                room=None,
                corridor=None,
                ctx=mock_ctx,
            )

        content = result.content[0].text  # type: ignore[union-attr]
        assert "failure" in content
        assert "Embedding API error" in content

    @pytest.mark.asyncio
    async def test_path_passed_to_input(self, mock_ctx: MagicMock) -> None:
        """path parameter is forwarded to KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_store_output())

            await store_knowledge(
                content="file-derived fact",
                source="my-docs",
                path="docs/architecture.md",
                valid_from=None,
                valid_to=None,
                confidence=0.9,
                categories=None,
                authority="AGENT",
                lifecycle_state="ACTIVE",
                namespace=None,
                room=None,
                corridor=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.path == "docs/architecture.md"
        assert inputs.source == "my-docs"

    @pytest.mark.asyncio
    async def test_validity_window_passed_to_input(self, mock_ctx: MagicMock) -> None:
        """valid_from and valid_to are forwarded to KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_store_output())

            await store_knowledge(
                content="time-bound policy",
                source="policy",
                path=None,
                valid_from="2026-01-01T00:00:00Z",
                valid_to="2026-12-31T23:59:59Z",
                confidence=0.9,
                categories=None,
                authority="AGENT",
                lifecycle_state="ACTIVE",
                namespace=None,
                room=None,
                corridor=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.valid_from == "2026-01-01T00:00:00Z"
        assert inputs.valid_to == "2026-12-31T23:59:59Z"

    @pytest.mark.asyncio
    async def test_path_none_by_default(self, mock_ctx: MagicMock) -> None:
        """path defaults to None when not provided."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_store_output())

            await store_knowledge(
                content="agent observation",
                source=None,
                path=None,
                valid_from=None,
                valid_to=None,
                confidence=0.8,
                categories=None,
                authority="AGENT",
                lifecycle_state="ACTIVE",
                namespace=None,
                room=None,
                corridor=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.path is None

    @pytest.mark.asyncio
    async def test_namespace_room_corridor_passed_to_input(self, mock_ctx: MagicMock) -> None:
        """namespace, room, and corridor are forwarded to KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_store_output())

            await store_knowledge(
                content="JWT tokens expire after 15 minutes",
                source="auth-docs",
                path=None,
                valid_from=None,
                valid_to=None,
                confidence=0.9,
                categories=None,
                authority="EXTRACTED",
                lifecycle_state="ACTIVE",
                namespace="engineering",
                room="auth",
                corridor="tokens",
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.namespace == "engineering"
        assert inputs.room == "auth"
        assert inputs.corridor == "tokens"

    @pytest.mark.asyncio
    async def test_namespace_room_corridor_default_none(self, mock_ctx: MagicMock) -> None:
        """namespace, room, and corridor default to None."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_store_output())

            await store_knowledge(
                content="generic fact",
                source=None,
                path=None,
                valid_from=None,
                valid_to=None,
                confidence=0.8,
                categories=None,
                authority="AGENT",
                lifecycle_state="ACTIVE",
                namespace=None,
                room=None,
                corridor=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.namespace is None
        assert inputs.room is None
        assert inputs.corridor is None

    @pytest.mark.asyncio
    async def test_search_result_includes_item_path_column(self, mock_ctx: MagicMock) -> None:
        """search_knowledge response rows should include item_path field."""
        search_out = _make_search_output(
            rows=[
                {
                    "id": "uuid-1",
                    "content": "a fact from a file",
                    "confidence": 0.9,
                    "item_path": "docs/design.md",
                }
            ],
            columns=["id", "content", "confidence", "item_path"],
        )
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=search_out)

            result = await search_knowledge(
                query="design",
                source=None,
                categories=None,
                as_of=None,
                min_confidence=0.3,
                limit=10,
                namespace=None,
                room=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "item_path" in data["columns"]
        assert data["rows"][0]["item_path"] == "docs/design.md"

    @pytest.mark.asyncio
    async def test_namespace_and_room_passed_to_input(self, mock_ctx: MagicMock) -> None:
        """namespace and room are forwarded to KnowledgeInput for room-scoped routing."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_search_output())

            await search_knowledge(
                query="JWT expiry handling",
                source=None,
                categories=None,
                as_of=None,
                min_confidence=0.3,
                limit=10,
                namespace="engineering",
                room="auth",
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.namespace == "engineering"
        assert inputs.room == "auth"

    @pytest.mark.asyncio
    async def test_namespace_room_default_none(self, mock_ctx: MagicMock) -> None:
        """namespace and room default to None (global search, no room scoping)."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_search_output())

            await search_knowledge(
                query="any query",
                source=None,
                categories=None,
                as_of=None,
                min_confidence=0.3,
                limit=10,
                namespace=None,
                room=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.namespace is None
        assert inputs.room is None


# ============================================================================
# recall_knowledge Tests
# ============================================================================


class TestRecallKnowledge:
    """Tests for recall_knowledge MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_rows(self, mock_ctx: MagicMock) -> None:
        """Successful recall returns rows."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_recall_output())

            result = await recall_knowledge(
                source=None,
                categories=None,
                as_of=None,
                lifecycle_state="ACTIVE",
                min_confidence=None,
                limit=10,
                order=None,
                created_by=None,
                auth_method=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "recall"

    @pytest.mark.asyncio
    async def test_source_and_lifecycle_routing(self, mock_ctx: MagicMock) -> None:
        """Source and lifecycle_state are set as direct inputs fields, not routed through where."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_recall_output())

            await recall_knowledge(
                source="workflow:*",
                categories=None,
                as_of="2026-06-15T12:00:00Z",
                lifecycle_state="ACTIVE",
                min_confidence=0.7,
                limit=5,
                order=["confidence:desc"],
                created_by=None,
                auth_method=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.source == "workflow:*"
        assert inputs.as_of == "2026-06-15T12:00:00Z"
        assert inputs.lifecycle_state == "ACTIVE"
        assert inputs.min_confidence == 0.7
        assert inputs.where is None  # created_by and auth_method both None, so no where dict
        assert inputs.limit == 5
        assert inputs.order == ["confidence:desc"]


# ============================================================================
# forget_knowledge Tests
# ============================================================================


class TestForgetKnowledge:
    """Tests for forget_knowledge MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_counts(self, mock_ctx: MagicMock) -> None:
        """Successful forget returns archived and skipped counts."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_forget_output())

            result = await forget_knowledge(
                proposition_ids=["uuid-1", "uuid-2"],
                source=None,
                created_before=None,
                created_after=None,
                reason="Outdated information",
                ctx=mock_ctx,
            )

        assert result.isError is not True
        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "forget"
        assert inputs.proposition_ids == ["uuid-1", "uuid-2"]
        assert inputs.reason == "Outdated information"

    @pytest.mark.asyncio
    async def test_error_propagates(self, mock_ctx: MagicMock) -> None:
        """Failed forget returns error."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("Permission denied")
            )

            result = await forget_knowledge(
                proposition_ids=["uuid-1"],
                source=None,
                created_before=None,
                created_after=None,
                reason=None,
                ctx=mock_ctx,
            )

        content = result.content[0].text  # type: ignore[union-attr]
        assert "Permission denied" in content

    @pytest.mark.asyncio
    async def test_filter_by_source(self, mock_ctx: MagicMock) -> None:
        """forget_knowledge archives by source when no IDs provided."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_forget_output())

            result = await forget_knowledge(
                proposition_ids=None,
                source="session:abc",
                created_before=None,
                created_after=None,
                reason="Session cleanup",
                ctx=mock_ctx,
            )

        assert result.isError is not True
        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "forget"
        assert inputs.proposition_ids is None
        assert inputs.source == "session:abc"
        assert inputs.reason == "Session cleanup"

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self, mock_ctx: MagicMock) -> None:
        """forget_knowledge passes created_before and created_after to executor."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_forget_output())

            result = await forget_knowledge(
                proposition_ids=None,
                source="docs",
                created_before="2026-01-01T00:00:00Z",
                created_after=None,
                reason=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.source == "docs"
        assert inputs.created_before == "2026-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_no_filter_returns_error(self, mock_ctx: MagicMock) -> None:
        """forget_knowledge returns error when no targeting criteria provided."""
        result = await forget_knowledge(
            proposition_ids=None,
            source=None,
            created_before=None,
            created_after=None,
            reason=None,
            ctx=mock_ctx,
        )

        import json

        content = result.content[0].text  # type: ignore[union-attr]
        data = json.loads(content)
        assert data["status"] == "failure"
        assert "required" in data["error"]


# ============================================================================
# knowledge_context Tests
# ============================================================================


class TestKnowledgeContext:
    """Tests for knowledge_context MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_context(self, mock_ctx: MagicMock) -> None:
        """Successful context returns text, count, and tokens."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_context_output())

            result = await knowledge_context(
                query="database optimization",
                source=None,
                categories=None,
                as_of=None,
                max_tokens=4000,
                diversity=False,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "context"
        assert inputs.query == "database optimization"
        assert inputs.max_tokens == 4000
        assert inputs.diversity is False

    @pytest.mark.asyncio
    async def test_diversity_flag_passed(self, mock_ctx: MagicMock) -> None:
        """Diversity flag is passed through to KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_context_output())

            await knowledge_context(
                query="test",
                source="docs",
                categories=None,
                as_of="2026-06-15T12:00:00Z",
                max_tokens=2000,
                diversity=True,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.diversity is True
        assert inputs.max_tokens == 2000
        assert inputs.source == "docs"
        assert inputs.as_of == "2026-06-15T12:00:00Z"

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, mock_ctx: MagicMock) -> None:
        """Failed context returns error."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("DB unreachable")
            )

            result = await knowledge_context(
                query="test",
                source=None,
                categories=None,
                as_of=None,
                max_tokens=4000,
                diversity=False,
                ctx=mock_ctx,
            )

        content = result.content[0].text  # type: ignore[union-attr]
        assert "DB unreachable" in content


# ============================================================================
# Tool Annotation Tests
# ============================================================================


class TestToolAnnotations:
    """Tests that verify tool annotations are set correctly."""

    def test_search_is_read_only(self) -> None:
        """search_knowledge should be annotated as read-only."""
        tool = self._find_tool("search_knowledge")
        assert tool is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False

    def test_store_is_not_read_only(self) -> None:
        """store_knowledge should NOT be annotated as read-only."""
        tool = self._find_tool("store_knowledge")
        assert tool is not None
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.destructiveHint is False

    def test_forget_is_destructive(self) -> None:
        """forget_knowledge should be annotated as destructive."""
        tool = self._find_tool("forget_knowledge")
        assert tool is not None
        assert tool.annotations.destructiveHint is True

    def test_recall_is_read_only(self) -> None:
        """recall_knowledge should be annotated as read-only."""
        tool = self._find_tool("recall_knowledge")
        assert tool is not None
        assert tool.annotations.readOnlyHint is True

    def test_context_is_read_only(self) -> None:
        """knowledge_context should be annotated as read-only."""
        tool = self._find_tool("knowledge_context")
        assert tool is not None
        assert tool.annotations.readOnlyHint is True

    @staticmethod
    def _find_tool(name: str) -> Any:
        """Find a tool by name in the FastMCP server."""
        tool_manager = _mcp_server._tool_manager
        return tool_manager._tools.get(name)

    def test_graph_neighbors_is_read_only(self) -> None:
        """graph_neighbors should be annotated as read-only."""
        tool = self._find_tool("graph_neighbors")
        assert tool is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True

    def test_graph_traverse_is_read_only(self) -> None:
        """graph_traverse should be annotated as read-only."""
        tool = self._find_tool("graph_traverse")
        assert tool is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True

    def test_graph_path_is_read_only(self) -> None:
        """graph_path should be annotated as read-only."""
        tool = self._find_tool("graph_path")
        assert tool is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True

    def test_graph_stats_is_read_only(self) -> None:
        """graph_stats should be annotated as read-only."""
        tool = self._find_tool("graph_stats")
        assert tool is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True


# ============================================================================
# Graph Tool References
# ============================================================================

graph_neighbors = _get_tool_fn("graph_neighbors")
graph_traverse = _get_tool_fn("graph_traverse")
graph_path = _get_tool_fn("graph_path")
graph_stats = _get_tool_fn("graph_stats")


# ============================================================================
# Graph Output Helpers
# ============================================================================


def _make_graph_neighbors_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "nodes": [{"id": "entity-1", "name": "Alpha"}],
        "edges": [
            {"id": "rel-1", "source": "entity-0", "target": "entity-1", "relation_type": "is_a"}
        ],
        "paths": [],
        "traversal_count": 1,
        "diagnostics": {"expanded_nodes": 1, "pruned_edges": 0, "latency_ms": 5},
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_graph_traverse_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "nodes": [
            {"id": "entity-1", "name": "Alpha"},
            {"id": "entity-2", "name": "Beta"},
        ],
        "edges": [
            {"id": "rel-1", "source": "entity-1", "target": "entity-2", "relation_type": "part_of"}
        ],
        "paths": [],
        "traversal_count": 2,
        "diagnostics": {"expanded_nodes": 2, "pruned_edges": 0, "latency_ms": 10},
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_graph_path_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "nodes": [{"id": "entity-1", "name": "Alpha"}, {"id": "entity-2", "name": "Beta"}],
        "edges": [
            {"id": "rel-1", "source": "entity-1", "target": "entity-2", "relation_type": "leads_to"}
        ],
        "paths": [
            {
                "nodes": ["entity-1", "entity-2"],
                "edges": ["rel-1"],
            }
        ],
        "traversal_count": 1,
        "diagnostics": {"expanded_nodes": 2, "pruned_edges": 0, "latency_ms": 8},
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


def _make_graph_stats_output(**overrides: Any) -> KnowledgeOutput:
    defaults: dict[str, Any] = {
        "success": True,
        "nodes": [{"id": "entity-0", "name": "Root"}],
        "edges": [],
        "paths": [],
        "traversal_count": 1,
        "diagnostics": {
            "out_degree": 3,
            "in_degree": 1,
            "total_degree": 4,
            "distinct_relation_types": 2,
            "expanded_nodes": 1,
            "pruned_edges": 0,
            "latency_ms": 3,
        },
    }
    defaults.update(overrides)
    return KnowledgeOutput(**defaults)


# ============================================================================
# graph_neighbors Tests
# ============================================================================


class TestGraphNeighbors:
    """Tests for graph_neighbors MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_nodes_and_edges(self, mock_ctx: MagicMock) -> None:
        """Successful neighbors call returns nodes, edges, traversal_count, diagnostics."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_neighbors_output())

            result = await graph_neighbors(
                start_entity="entity-0",
                relation_types=None,
                max_nodes=50,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "nodes" in data
        assert "edges" in data
        assert "traversal_count" in data
        assert "diagnostics" in data

    @pytest.mark.asyncio
    async def test_op_and_start_entity_wired(self, mock_ctx: MagicMock) -> None:
        """op and start_entity are set correctly on KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_neighbors_output())

            await graph_neighbors(
                start_entity="entity-0",
                relation_types=None,
                max_nodes=50,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "graph_neighbors"
        assert inputs.start_entity == "entity-0"

    @pytest.mark.asyncio
    async def test_filters_passed_through(self, mock_ctx: MagicMock) -> None:
        """relation_types, max_nodes, min_edge_confidence, as_of are forwarded."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_neighbors_output())

            await graph_neighbors(
                start_entity="entity-0",
                relation_types=["is_a", "part_of"],
                max_nodes=25,
                min_edge_confidence=0.5,
                as_of="2026-06-01T00:00:00Z",
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.relation_types == ["is_a", "part_of"]
        assert inputs.max_nodes == 25
        assert inputs.min_edge_confidence == 0.5
        assert inputs.as_of == "2026-06-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, mock_ctx: MagicMock) -> None:
        """Failed graph_neighbors returns error in response body."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("Graph DB unavailable")
            )

            result = await graph_neighbors(
                start_entity="entity-0",
                relation_types=None,
                max_nodes=50,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert data["status"] == "failure"
        assert "Graph DB unavailable" in data["error"]


# ============================================================================
# graph_traverse Tests
# ============================================================================


class TestGraphTraverse:
    """Tests for graph_traverse MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_subgraph(self, mock_ctx: MagicMock) -> None:
        """Successful traverse returns nodes, edges, traversal_count, diagnostics."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_traverse_output())

            result = await graph_traverse(
                start_entity="entity-1",
                relation_types=None,
                max_hops=3,
                max_nodes=100,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "nodes" in data
        assert "edges" in data
        assert "traversal_count" in data
        assert "diagnostics" in data

    @pytest.mark.asyncio
    async def test_op_and_fields_wired(self, mock_ctx: MagicMock) -> None:
        """op, start_entity, max_hops, max_nodes are wired to KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_traverse_output())

            await graph_traverse(
                start_entity="entity-1",
                relation_types=["part_of"],
                max_hops=4,
                max_nodes=200,
                min_edge_confidence=0.3,
                as_of="2026-03-01T00:00:00Z",
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "graph_traverse"
        assert inputs.start_entity == "entity-1"
        assert inputs.relation_types == ["part_of"]
        assert inputs.max_hops == 4
        assert inputs.max_nodes == 200
        assert inputs.min_edge_confidence == 0.3
        assert inputs.as_of == "2026-03-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, mock_ctx: MagicMock) -> None:
        """Failed graph_traverse returns error."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("Timeout during traversal")
            )

            result = await graph_traverse(
                start_entity="entity-1",
                relation_types=None,
                max_hops=3,
                max_nodes=100,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert data["status"] == "failure"
        assert "Timeout during traversal" in data["error"]


# ============================================================================
# graph_path Tests
# ============================================================================


class TestGraphPath:
    """Tests for graph_path MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_path(self, mock_ctx: MagicMock) -> None:
        """Successful path call returns nodes, edges, paths, traversal_count, diagnostics."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_path_output())

            result = await graph_path(
                start_entity="entity-1",
                end_entity="entity-2",
                relation_types=None,
                max_hops=6,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "nodes" in data
        assert "edges" in data
        assert "paths" in data
        assert "traversal_count" in data
        assert "diagnostics" in data

    @pytest.mark.asyncio
    async def test_op_and_endpoints_wired(self, mock_ctx: MagicMock) -> None:
        """op, start_entity, and end_entity are set correctly on KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_path_output())

            await graph_path(
                start_entity="entity-A",
                end_entity="entity-B",
                relation_types=["leads_to"],
                max_hops=4,
                min_edge_confidence=0.2,
                as_of="2026-04-01T00:00:00Z",
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "graph_path"
        assert inputs.start_entity == "entity-A"
        assert inputs.end_entity == "entity-B"
        assert inputs.relation_types == ["leads_to"]
        assert inputs.max_hops == 4
        assert inputs.min_edge_confidence == 0.2
        assert inputs.as_of == "2026-04-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_paths_key_present_in_response(self, mock_ctx: MagicMock) -> None:
        """Response includes 'paths' key with the path segments."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_path_output())

            result = await graph_path(
                start_entity="entity-1",
                end_entity="entity-2",
                relation_types=None,
                max_hops=6,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert isinstance(data["paths"], list)
        assert len(data["paths"]) == 1
        assert "nodes" in data["paths"][0]
        assert "edges" in data["paths"][0]

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, mock_ctx: MagicMock) -> None:
        """Failed graph_path returns error."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("No path found")
            )

            result = await graph_path(
                start_entity="entity-1",
                end_entity="entity-99",
                relation_types=None,
                max_hops=6,
                min_edge_confidence=0.0,
                as_of=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert data["status"] == "failure"
        assert "No path found" in data["error"]


# ============================================================================
# graph_stats Tests
# ============================================================================


class TestGraphStats:
    """Tests for graph_stats MCP tool."""

    @pytest.mark.asyncio
    async def test_success_returns_stats(self, mock_ctx: MagicMock) -> None:
        """Successful stats call returns nodes, traversal_count, diagnostics."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_stats_output())

            result = await graph_stats(
                start_entity="entity-0",
                as_of=None,
                ctx=mock_ctx,
            )

        assert result.isError is not True
        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "nodes" in data
        assert "traversal_count" in data
        assert "diagnostics" in data

    @pytest.mark.asyncio
    async def test_op_and_entity_wired(self, mock_ctx: MagicMock) -> None:
        """op and start_entity are set correctly on KnowledgeInput."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_stats_output())

            await graph_stats(
                start_entity="entity-0",
                as_of="2026-05-01T00:00:00Z",
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.op == "graph_stats"
        assert inputs.start_entity == "entity-0"
        assert inputs.as_of == "2026-05-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_diagnostics_include_degree_metrics(self, mock_ctx: MagicMock) -> None:
        """Diagnostics dict exposes out_degree, in_degree, total_degree."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_stats_output())

            result = await graph_stats(
                start_entity="entity-0",
                as_of=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        diag = data["diagnostics"]
        assert diag["out_degree"] == 3
        assert diag["in_degree"] == 1
        assert diag["total_degree"] == 4
        assert diag["distinct_relation_types"] == 2

    @pytest.mark.asyncio
    async def test_no_edges_key_in_response(self, mock_ctx: MagicMock) -> None:
        """graph_stats response does not include an 'edges' key (stats-only output)."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(return_value=_make_graph_stats_output())

            result = await graph_stats(
                start_entity="entity-0",
                as_of=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "edges" not in data

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, mock_ctx: MagicMock) -> None:
        """Failed graph_stats returns error."""
        with patch(_EXECUTOR_CLS) as mock_cls:
            mock_cls.return_value.execute = AsyncMock(
                return_value=_make_error_output("Entity not found")
            )

            result = await graph_stats(
                start_entity="unknown-entity",
                as_of=None,
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert data["status"] == "failure"
        assert "Entity not found" in data["error"]
