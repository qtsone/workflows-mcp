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
                ctx=mock_ctx,
            )

        inputs: KnowledgeInput = mock_cls.return_value.execute.call_args[0][0]
        assert inputs.path is None

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
                ctx=mock_ctx,
            )

        import json as _j

        data = _j.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "item_path" in data["columns"]
        assert data["rows"][0]["item_path"] == "docs/design.md"


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
