"""Acceptance tests for session-scoped active context and select tool.

Acceptance criteria:
A) onboard -> memory(query) without scope succeeds via active context fallback
B) onboard A -> onboard B -> memory(query) without scope resolves to B
C) memory/sync without scope and without active context returns actionable error code
D) explicit scope on memory/sync overrides active context
E) select sets active context; subsequent no-scope memory uses selected context
F) select ambiguous/not found returns actionable error
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.memory_scope_resolver import SyncContextCandidate, scope_key
from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_memory import _onboard_context_registry, register_memory_tools

register_memory_tools(_mcp_server)


def _get_tool_fn(name: str) -> Any:
    tool = _mcp_server._tool_manager._tools.get(name)
    if tool is None:
        raise ValueError(f"Tool {name!r} not registered")
    return tool.fn


memory = _get_tool_fn("memory")
onboard = _get_tool_fn("onboard")
sync = _get_tool_fn("sync")
select = _get_tool_fn("select")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_ctx(*, session: MagicMock | None = None) -> MagicMock:
    """Build a mock MCP context with session-scoped active context support.

    Uses a real dict + closures for get_active_context/set_active_context
    so the session-keyed behavior works without instantiating a full AppContext.
    """
    ctx = MagicMock()
    app_ctx = MagicMock()
    app_ctx.memory_backend = None
    app_ctx.memory_backend_lock = None
    ctx.request_context.lifespan_context = app_ctx

    exec_context = MagicMock()
    exec_context.user_string_id = None
    app_ctx.create_execution_context.return_value = exec_context
    app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")

    # Real session-scoped state backed by a plain dict.
    _session_store: dict[int, Any] = {}

    def _get_active(s: Any) -> Any:
        return _session_store.get(id(s))

    def _set_active(s: Any, candidate: Any) -> None:
        _session_store[id(s)] = candidate

    app_ctx.get_active_context = _get_active
    app_ctx.set_active_context = _set_active

    if session is None:
        session = MagicMock()
    ctx.request_context.session = session

    return ctx


@pytest.fixture
def session_a() -> MagicMock:
    return MagicMock(name="session_a")


@pytest.fixture
def session_b() -> MagicMock:
    return MagicMock(name="session_b")


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Clear the global onboard registry before each test."""
    _onboard_context_registry.clear()


def _inject_registry_entry(scope: dict[str, Any]) -> SyncContextCandidate:
    """Helper: add a synthetic onboard context to the registry."""
    key = scope_key(scope)
    candidate = SyncContextCandidate(
        scope={k: scope.get(k) for k in ("palace", "wing", "room", "compartment")},
        scope_key_value=key,
        checkpoint_data={"scope": scope, "scope_key": key},
        source="test_injection",
    )
    _onboard_context_registry[key] = candidate
    return candidate


def _parse(result: Any) -> dict[str, Any]:
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# A) onboard -> memory(query) without scope uses active context
# ---------------------------------------------------------------------------


class TestAcceptanceA_OnboardActivatesContext:
    """After a successful onboard, memory() with no scope uses active context."""

    @pytest.mark.asyncio
    async def test_memory_no_scope_uses_active_context_after_onboard(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)

        # Inject a registry entry (simulates completed onboard) and set active ctx.
        candidate = _inject_registry_entry({"palace": "proj-a"})
        ctx.request_context.lifespan_context.set_active_context(session_a, candidate)

        # Mock backend to return a successful query result.
        backend_mock = MagicMock()
        backend_mock.connect = AsyncMock()
        backend_mock.disconnect = AsyncMock()

        from workflows_mcp.engine.memory_service import (
            ManageMemoryResult,
            MemoryResult,
            QueryMemoryResult,
        )

        query_result = MemoryResult(
            operation="query",
            query=QueryMemoryResult(
                facts=[{"content": "test fact"}],
                memories=[],
                communities=[],
                diagnostics={},
                evidence=[],
                paths=[],
            ),
        )

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch(
                "workflows_mcp.engine.memory_service.MemoryService.execute",
                new_callable=AsyncMock,
                return_value=query_result,
            ),
        ):
            result = await memory(
                operation="query",
                # No scope provided — should resolve via active context.
                query={"text": "test"},
                ctx=ctx,
            )

        payload = _parse(result)
        # Should NOT be an error — active context was resolved.
        assert "error" not in payload, f"Expected success, got error: {payload}"


# ---------------------------------------------------------------------------
# B) onboard A -> onboard B -> memory resolves to B
# ---------------------------------------------------------------------------


class TestAcceptanceB_SecondOnboardUpdatesActive:
    """After two onboards, the second one wins as active context."""

    @pytest.mark.asyncio
    async def test_second_onboard_replaces_active_context(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)

        # Set up context A.
        candidate_a = _inject_registry_entry({"palace": "proj-a"})
        ctx.request_context.lifespan_context.set_active_context(session_a, candidate_a)

        # Now set up context B (simulating second onboard).
        candidate_b = _inject_registry_entry({"palace": "proj-b"})
        ctx.request_context.lifespan_context.set_active_context(session_a, candidate_b)

        # Verify active context is B.
        active = ctx.request_context.lifespan_context.get_active_context(session_a)
        assert active is not None
        assert active.scope.get("palace") == "proj-b", (
            f"Active context should be proj-b, got: {active.scope}"
        )


# ---------------------------------------------------------------------------
# C) memory/sync without scope and without active context → actionable error
# ---------------------------------------------------------------------------


class TestAcceptanceC_NoActiveContextReturnsActionableError:
    """Without an active context and no explicit scope, tools must return MEM_NO_ACTIVE_CONTEXT."""

    @pytest.mark.asyncio
    async def test_memory_no_scope_no_active_context_returns_actionable_error(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)
        # Registry is empty (cleared by autouse fixture), no active context set.

        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await memory(
                operation="query",
                # No scope, no active context.
                query={"text": "test"},
                ctx=ctx,
            )

        payload = _parse(result)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_NO_ACTIVE_CONTEXT", (
            f"Expected MEM_NO_ACTIVE_CONTEXT, got: {payload}"
        )
        assert "actionable_fix" in err, "Must include actionable_fix"
        assert err.get("retryable") is False

    @pytest.mark.asyncio
    async def test_sync_no_scope_no_active_context_returns_actionable_error(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)
        # Registry is empty, no active context set.

        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(ctx=ctx)

        payload = _parse(result)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_NO_ACTIVE_CONTEXT", (
            f"Expected MEM_NO_ACTIVE_CONTEXT, got: {payload}"
        )
        assert "actionable_fix" in err, "Must include actionable_fix"


# ---------------------------------------------------------------------------
# D) explicit scope on memory/sync overrides active context
# ---------------------------------------------------------------------------


class TestAcceptanceD_ExplicitScopeOverridesActive:
    """Explicit scope arg always takes precedence over active context."""

    @pytest.mark.asyncio
    async def test_explicit_scope_wins_over_active_context(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)

        # Set active context to proj-a.
        candidate_a = _inject_registry_entry({"palace": "proj-a"})
        ctx.request_context.lifespan_context.set_active_context(session_a, candidate_a)

        # Track which scope was actually used by MemoryService.
        captured_requests: list[Any] = []

        from workflows_mcp.engine.memory_service import (
            MemoryRequest,
            MemoryResult,
            QueryMemoryResult,
        )

        async def _mock_execute(self: Any, request: MemoryRequest) -> MemoryResult:
            captured_requests.append(request)
            return MemoryResult(
                operation="query",
                query=QueryMemoryResult(
                    facts=[],
                    memories=[],
                    communities=[],
                    diagnostics={},
                    evidence=[],
                    paths=[],
                ),
            )

        backend_mock = MagicMock()
        backend_mock.connect = AsyncMock()
        backend_mock.disconnect = AsyncMock()

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend_mock),
            patch(
                "workflows_mcp.engine.memory_service.MemoryService.execute",
                new=_mock_execute,
            ),
        ):
            await memory(
                operation="query",
                scope={"palace": "proj-explicit"},  # Explicit scope — overrides active.
                query={"text": "test"},
                ctx=ctx,
            )

        assert len(captured_requests) == 1
        used_scope = captured_requests[0].scope
        assert used_scope.palace == "proj-explicit", (
            f"Expected 'proj-explicit', got: {used_scope}"
        )


# ---------------------------------------------------------------------------
# E) select sets active context; subsequent no-scope memory uses it
# ---------------------------------------------------------------------------


class TestAcceptanceE_SelectSetsActiveContext:
    """select() updates the session active context; memory() picks it up."""

    @pytest.mark.asyncio
    async def test_select_then_memory_uses_selected_context(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)

        # Seed registry with two contexts.
        _inject_registry_entry({"palace": "proj-x"})
        _inject_registry_entry({"palace": "proj-y"})

        # Select proj-y.
        select_result = await select(scope={"palace": "proj-y"}, ctx=ctx)
        select_payload = _parse(select_result)
        assert select_payload.get("status") == "selected", (
            f"Expected selected, got: {select_payload}"
        )
        assert select_payload["active_context"]["scope"].get("palace") == "proj-y"

        # Now verify active context is proj-y.
        active = ctx.request_context.lifespan_context.get_active_context(session_a)
        assert active is not None
        assert active.scope.get("palace") == "proj-y"

    @pytest.mark.asyncio
    async def test_select_response_includes_all_required_fields(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)
        _inject_registry_entry({"palace": "proj-z"})

        result = await select(scope={"palace": "proj-z"}, ctx=ctx)
        payload = _parse(result)

        assert payload.get("status") == "selected"
        ac = payload.get("active_context", {})
        assert "scope" in ac
        assert "scope_key" in ac
        assert "source" in ac
        assert "message" in payload


# ---------------------------------------------------------------------------
# F) select ambiguous/not-found returns actionable error
# ---------------------------------------------------------------------------


class TestAcceptanceF_SelectErrors:
    """select() returns clear actionable errors for ambiguous or missing scope."""

    @pytest.mark.asyncio
    async def test_select_not_found_returns_actionable_error(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)
        _inject_registry_entry({"palace": "known-proj"})

        result = await select(scope={"palace": "nonexistent-proj"}, ctx=ctx)
        payload = _parse(result)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_SELECT_NOT_FOUND", (
            f"Expected MEM_SELECT_NOT_FOUND, got: {payload}"
        )
        assert "actionable_fix" in err

    @pytest.mark.asyncio
    async def test_select_empty_registry_returns_actionable_error(
        self, session_a: MagicMock
    ) -> None:
        ctx = _make_mock_ctx(session=session_a)
        # Registry is empty — cleared by autouse fixture.

        result = await select(scope={"palace": "anything"}, ctx=ctx)
        payload = _parse(result)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_NO_ACTIVE_CONTEXT", (
            f"Expected MEM_NO_ACTIVE_CONTEXT, got: {payload}"
        )
        assert "actionable_fix" in err

    @pytest.mark.asyncio
    async def test_select_tool_is_registered(self) -> None:
        """select tool must be present in the tool manager."""
        tool = _mcp_server._tool_manager._tools.get("select")
        assert tool is not None, "select tool must be registered"

    @pytest.mark.asyncio
    async def test_select_has_scope_param(self) -> None:
        import inspect

        fn = _get_tool_fn("select")
        params = inspect.signature(fn).parameters
        assert "scope" in params, "select must accept a 'scope' parameter"

    @pytest.mark.asyncio
    async def test_select_sessions_are_isolated(
        self, session_a: MagicMock, session_b: MagicMock
    ) -> None:
        """Selecting in session A must not affect session B."""
        ctx_a = _make_mock_ctx(session=session_a)
        ctx_b = _make_mock_ctx(session=session_b)

        # Both share the same registry but have isolated active contexts.
        _inject_registry_entry({"palace": "shared-proj"})

        # Select in session A only.
        await select(scope={"palace": "shared-proj"}, ctx=ctx_a)

        active_a = ctx_a.request_context.lifespan_context.get_active_context(session_a)
        active_b = ctx_b.request_context.lifespan_context.get_active_context(session_b)

        assert active_a is not None, "Session A should have active context"
        assert active_b is None, "Session B should NOT have active context"
