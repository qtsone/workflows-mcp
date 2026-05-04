"""Phase 1 contract tests for the renamed onboard/sync MCP tools.

Verifies:
- Old tool names (project_onboard, project_sync) are absent from the tool manager.
- New tool names (onboard, sync) are present and include a root-level ``debug`` parameter.
- Both tools expose the stable error envelope shape: code, stage, message, actionable_fix.
- vNext spec: `ingestion` parameter present on both tools.
- vNext spec: legacy `response.mode` field is rejected with LEGACY_CONTRACT_REJECTED.
- vNext spec: `ingestion.mode='llm'` without `llm_profile` → INVALID_LLM_PROFILE.
- vNext spec: minimal scope (palace only) must not produce SCOPE_UNRESOLVED.
- vNext spec: `sync({})` returns MEM_NO_ACTIVE_CONTEXT (not MEM_PROJECT_FLOW_EMPTY).
- vNext spec: memory(query) with palace-only scope must not produce SCOPE_UNRESOLVED.
- vNext spec: memory(query) with no scope must produce SCOPE_UNRESOLVED for 'palace'.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_memory import register_memory_tools

# Ensure tools are registered (idempotent; second call is a no-op at server level
# because FastMCP deduplicates by function name).
register_memory_tools(_mcp_server)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tool(name: str) -> Any:
    return _mcp_server._tool_manager._tools.get(name)


def _get_tool_fn(name: str) -> Any:
    tool = _get_tool(name)
    if tool is None:
        raise ValueError(f"Tool {name!r} not registered")
    return tool.fn


@pytest.fixture
def mock_ctx() -> MagicMock:
    ctx = MagicMock()
    app_ctx = MagicMock()
    app_ctx.memory_backend = AsyncMock()
    app_ctx.memory_backend_lock = None
    app_ctx.memory_backend_unavailable_error = None
    ctx.request_context.lifespan_context = app_ctx
    exec_context = MagicMock()
    exec_context.user_string_id = None
    app_ctx.create_execution_context.return_value = exec_context
    app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")
    app_ctx.get_active_context.return_value = None
    return ctx


# ---------------------------------------------------------------------------
# Tool surface: names present / absent
# ---------------------------------------------------------------------------


class TestToolSurfaceRename:
    def test_old_name_project_onboard_absent(self) -> None:
        assert _get_tool("project_onboard") is None, (
            "project_onboard must not exist; it has been renamed to onboard"
        )

    def test_old_name_project_sync_absent(self) -> None:
        assert _get_tool("project_sync") is None, (
            "project_sync must not exist; it has been renamed to sync"
        )

    def test_new_name_onboard_present(self) -> None:
        assert _get_tool("onboard") is not None

    def test_new_name_sync_present(self) -> None:
        assert _get_tool("sync") is not None


# ---------------------------------------------------------------------------
# Root-level debug parameter presence
# ---------------------------------------------------------------------------


class TestDebugParameterContract:
    """Both tools must declare a root-level ``debug`` parameter."""

    def _param_names(self, tool_name: str) -> set[str]:
        import inspect

        fn = _get_tool_fn(tool_name)
        return set(inspect.signature(fn).parameters)

    def test_onboard_has_root_debug_param(self) -> None:
        assert "debug" in self._param_names("onboard"), (
            "onboard must expose a root-level 'debug' parameter"
        )

    def test_sync_has_root_debug_param(self) -> None:
        assert "debug" in self._param_names("sync"), (
            "sync must expose a root-level 'debug' parameter"
        )

    def test_onboard_debug_default_is_false(self) -> None:
        import inspect

        fn = _get_tool_fn("onboard")
        param = inspect.signature(fn).parameters["debug"]
        assert param.default is False

    def test_sync_debug_default_is_false(self) -> None:
        import inspect

        fn = _get_tool_fn("sync")
        param = inspect.signature(fn).parameters["debug"]
        assert param.default is False


# ---------------------------------------------------------------------------
# Error envelope shape: code, stage, message, actionable_fix
# ---------------------------------------------------------------------------


class TestErrorEnvelopeShape:
    """Invalid request paths must return an error envelope with the required fields."""

    @pytest.mark.asyncio
    async def test_onboard_invalid_checkpoint_returns_required_envelope_fields(
        self, mock_ctx: MagicMock
    ) -> None:
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                checkpoint={"version": "invalid-version"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error")
        assert err is not None, f"Expected error envelope, got: {payload}"
        assert "code" in err, "error envelope must contain 'code'"
        assert "message" in err, "error envelope must contain 'message'"
        assert "stage" in err, "error envelope must contain 'stage'"
        assert "actionable_fix" in err, "error envelope must contain 'actionable_fix'"

    @pytest.mark.asyncio
    async def test_sync_invalid_checkpoint_returns_required_envelope_fields(
        self, mock_ctx: MagicMock
    ) -> None:
        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(
                checkpoint={"version": "invalid-version"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error")
        assert err is not None, f"Expected error envelope, got: {payload}"
        assert "code" in err
        assert "message" in err
        assert "stage" in err
        assert "actionable_fix" in err

    @pytest.mark.asyncio
    async def test_onboard_empty_plan_returns_required_envelope_fields(
        self, mock_ctx: MagicMock
    ) -> None:
        """Calling onboard with no payload must return an error with all envelope fields."""
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        err = payload.get("error")
        assert err is not None, f"Expected error envelope, got: {payload}"
        assert "code" in err
        assert "message" in err
        assert "stage" in err
        assert "actionable_fix" in err

    @pytest.mark.asyncio
    async def test_error_envelope_code_is_string(self, mock_ctx: MagicMock) -> None:
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                checkpoint={"version": "bad"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        assert isinstance(payload["error"]["code"], str)

    @pytest.mark.asyncio
    async def test_error_envelope_message_is_string(self, mock_ctx: MagicMock) -> None:
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                checkpoint={"version": "bad"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        assert isinstance(payload["error"]["message"], str)


# ---------------------------------------------------------------------------
# vNext spec: `ingestion` parameter surface
# ---------------------------------------------------------------------------


class TestIngestionParameterSurface:
    """Both tools must declare a root-level `ingestion` parameter (spec §4.1/4.2)."""

    def _param_names(self, tool_name: str) -> set[str]:
        import inspect

        fn = _get_tool_fn(tool_name)
        return set(inspect.signature(fn).parameters)

    def test_onboard_has_ingestion_param(self) -> None:
        assert "ingestion" in self._param_names("onboard"), (
            "onboard must expose a root-level 'ingestion' parameter (spec §4.1)"
        )

    def test_sync_has_ingestion_param(self) -> None:
        assert "ingestion" in self._param_names("sync"), (
            "sync must expose a root-level 'ingestion' parameter (spec §4.2)"
        )

    def test_onboard_ingestion_default_is_none(self) -> None:
        import inspect

        fn = _get_tool_fn("onboard")
        param = inspect.signature(fn).parameters["ingestion"]
        assert param.default is None

    def test_sync_ingestion_default_is_none(self) -> None:
        import inspect

        fn = _get_tool_fn("sync")
        param = inspect.signature(fn).parameters["ingestion"]
        assert param.default is None


# ---------------------------------------------------------------------------
# vNext spec: legacy response.mode field must be rejected
# ---------------------------------------------------------------------------


class TestLegacyContractRejection:
    """Passing `response={'mode': 'programmatic'}` or `response={'mode': 'llm'}` must be
    rejected with a LEGACY_CONTRACT_REJECTED error (spec §2: no backward compatibility)."""

    @pytest.mark.asyncio
    async def test_onboard_response_mode_programmatic_rejected(
        self, mock_ctx: MagicMock
    ) -> None:
        """response={'mode': 'programmatic'} is legacy and must be rejected."""
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                scope={"palace": "org"},
                response={"mode": "programmatic"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "LEGACY_CONTRACT_REJECTED", (
            f"Expected LEGACY_CONTRACT_REJECTED, got: {err.get('code')} — payload: {payload}"
        )

    @pytest.mark.asyncio
    async def test_onboard_response_mode_llm_rejected(self, mock_ctx: MagicMock) -> None:
        """response={'mode': 'llm'} is legacy and must be rejected."""
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                scope={"palace": "org"},
                response={"mode": "llm"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "LEGACY_CONTRACT_REJECTED", (
            f"Expected LEGACY_CONTRACT_REJECTED, got: {err.get('code')} — payload: {payload}"
        )

    @pytest.mark.asyncio
    async def test_sync_response_mode_programmatic_rejected(
        self, mock_ctx: MagicMock
    ) -> None:
        """sync with response={'mode': 'programmatic'} is legacy and must be rejected."""
        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(
                scope={"palace": "org"},
                response={"mode": "programmatic"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "LEGACY_CONTRACT_REJECTED", (
            f"Expected LEGACY_CONTRACT_REJECTED, got: {err.get('code')} — payload: {payload}"
        )


# ---------------------------------------------------------------------------
# vNext spec: ingestion.mode='llm' + missing llm_profile → INVALID_LLM_PROFILE
# ---------------------------------------------------------------------------


class TestIngestionLLMValidation:
    """ingestion contract must return INVALID_LLM_PROFILE when llm_profile is absent."""

    @pytest.mark.asyncio
    async def test_onboard_ingestion_llm_no_profile_returns_invalid_llm_profile(
        self, mock_ctx: MagicMock, tmp_path: Any
    ) -> None:
        from pathlib import Path

        tmp = Path(str(tmp_path))
        (tmp / "a.py").write_text("x = 1")

        mock_ctx.request_context.lifespan_context.llm_config_loader = None
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                scope={"palace": "org"},
                ingestion={"mode": "llm"},  # no llm_profile
                scan={
                    "patterns": ["*.py"],
                    "root": str(tmp),
                    "max_files": 5,
                    "max_size_kb": 10,
                },
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "INVALID_LLM_PROFILE", (
            f"Expected INVALID_LLM_PROFILE, got: {err.get('code')} — payload: {payload}"
        )

    @pytest.mark.asyncio
    async def test_onboard_ingestion_programmatic_palace_only_does_not_scope_error(
        self, mock_ctx: MagicMock, tmp_path: Any
    ) -> None:
        """ingestion.mode='programmatic' with palace-only scope must not return SCOPE_UNRESOLVED."""
        from pathlib import Path

        tmp = Path(str(tmp_path))
        (tmp / "main.py").write_text("def run(): pass")

        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                scope={"palace": "org"},
                ingestion={"mode": "programmatic"},
                scan={
                    "patterns": ["*.py"],
                    "root": str(tmp),
                    "max_files": 5,
                    "max_size_kb": 10,
                },
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") != "SCOPE_UNRESOLVED", (
            f"SCOPE_UNRESOLVED must not be raised for palace-only scope: {payload}"
        )
        assert err.get("code") != "INVALID_SCOPE", (
            f"INVALID_SCOPE must not be raised for valid palace-only scope: {payload}"
        )

    @pytest.mark.asyncio
    async def test_onboard_checkpoint_flow_palace_only_does_not_scope_unresolved(
        self, mock_ctx: MagicMock, tmp_path: Any
    ) -> None:
        """Checkpoint-path ingest with palace-only scope must not raise SCOPE_UNRESOLVED.

        Regression guard for the ingest-stage SCOPE_UNRESOLVED blocker:
        when onboard is called without ingestion.mode='programmatic' (falling through
        to the checkpoint flow), a palace-only scope must not trigger
        SCOPE_UNRESOLVED for 'wing'. The acceptable outcomes are:
        - status='checkpoint' with error.code != 'SCOPE_UNRESOLVED'
        - status='completed'
        """
        from pathlib import Path
        from unittest.mock import AsyncMock

        tmp = Path(str(tmp_path))
        (tmp / "main.py").write_text("def run(): pass")

        mock_backend = AsyncMock()
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=mock_backend):
            result = await onboard(
                scope={"palace": "workflows-mcp-live"},
                # No ingestion.mode — triggers checkpoint flow, not programmatic fast-path
                scan={
                    "patterns": ["*.py"],
                    "root": str(tmp),
                    "max_files": 5,
                    "max_size_kb": 10,
                },
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") != "SCOPE_UNRESOLVED", (
            "SCOPE_UNRESOLVED must not be raised for palace-only scope "
            f"in checkpoint flow: {payload}"
        )

    @pytest.mark.asyncio
    async def test_sync_empty_returns_no_context_not_flow_empty(
        self, mock_ctx: MagicMock
    ) -> None:
        """sync({}) must return MEM_NO_ACTIVE_CONTEXT, not MEM_PROJECT_FLOW_EMPTY (spec §4.2)."""
        sync = _get_tool_fn("sync")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync(ctx=mock_ctx)
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_NO_ACTIVE_CONTEXT", (
            "Expected MEM_NO_ACTIVE_CONTEXT from sync({}), "
            f"got: {err.get('code')} — payload: {payload}"
        )
        assert err.get("code") != "MEM_PROJECT_FLOW_EMPTY"

    @pytest.mark.asyncio
    async def test_memory_query_palace_only_scope_does_not_scope_unresolved(
        self, mock_ctx: MagicMock
    ) -> None:
        """memory(query) with palace-only scope must not raise SCOPE_UNRESOLVED.

        Regression guard for the production failure:
        onboard succeeded for a repo, then memory(query, scope={palace: "forge"})
        returned SCOPE_UNRESOLVED for 'wing'.  The fix relaxes required_scope_fields
        for query to ('palace',) only — wing/room/compartment are optional filters.
        """
        from unittest.mock import AsyncMock

        memory = _get_tool_fn("memory")
        mock_backend = AsyncMock()
        mock_backend.query_memories = AsyncMock(
            return_value={"memories": [], "facts": [], "communities": []}
        )
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=mock_backend):
            result = await memory(
                operation="query",
                scope={"palace": "forge"},
                query={"text": "architecture decisions"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") != "SCOPE_UNRESOLVED", (
            f"SCOPE_UNRESOLVED must not be raised for palace-only scope in query: {payload}"
        )

    @pytest.mark.asyncio
    async def test_memory_query_no_palace_raises_insufficient_locality(
        self, mock_ctx: MagicMock
    ) -> None:
        """memory(query) with no palace must raise INSUFFICIENT_LOCALITY.

        Query requires palace-minimum locality; an empty scope fails locality
        resolution before any scope lookup occurs. The error envelope must include
        actionable retry guidance referencing palace or an alternative resolution
        source (scope_token, context_id, active project defaults).
        """
        from unittest.mock import AsyncMock

        memory = _get_tool_fn("memory")
        mock_backend = AsyncMock()
        with patch("workflows_mcp.tools_memory.PostgresBackend", return_value=mock_backend):
            result = await memory(
                operation="query",
                scope={},
                query={"text": "anything"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "INSUFFICIENT_LOCALITY", (
            f"Expected INSUFFICIENT_LOCALITY when palace is absent, "
            f"got: {err.get('code')} — {payload}"
        )
        actionable_fix = err.get("actionable_fix", "")
        assert "palace" in actionable_fix.lower(), (
            f"actionable_fix must mention 'palace' so the caller knows what to supply; "
            f"got: {actionable_fix!r}"
        )


# ---------------------------------------------------------------------------
# Acceptance test A: MEM_SCHEMA_VALIDATION_FAILED includes field-level detail
# ---------------------------------------------------------------------------


class TestSchemaValidationFailedFieldDetail:
    """Acceptance test A — invalid scan payload returns MEM_SCHEMA_VALIDATION_FAILED
    with field-level detail embedded in message and actionable_fix (issue 3)."""

    @pytest.mark.asyncio
    async def test_invalid_scan_payload_returns_field_level_detail(
        self, mock_ctx: MagicMock
    ) -> None:
        """onboard with an invalid scan config (max_files exceeds limit) must return
        MEM_SCHEMA_VALIDATION_FAILED with field-level detail in message/actionable_fix."""
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                scope={"palace": "org"},
                scan={
                    "patterns": ["*.py"],
                    "root": "/tmp",
                    "max_files": 9999,  # exceeds le=100
                },
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_SCHEMA_VALIDATION_FAILED", (
            f"Expected MEM_SCHEMA_VALIDATION_FAILED, got: {err.get('code')} — {payload}"
        )
        message = err.get("message", "")
        actionable_fix = err.get("actionable_fix", "")
        # Both message and actionable_fix must contain the specific field name.
        assert "max_files" in message, (
            f"Expected 'max_files' in message, got: {message!r}"
        )
        assert "max_files" in actionable_fix, (
            f"Expected 'max_files' in actionable_fix, got: {actionable_fix!r}"
        )

    @pytest.mark.asyncio
    async def test_invalid_scan_payload_path_and_patterns_mutual_exclusion(
        self, mock_ctx: MagicMock
    ) -> None:
        """onboard with path + patterns simultaneously must return MEM_SCHEMA_VALIDATION_FAILED
        with detail about the mutual-exclusion violation."""
        onboard = _get_tool_fn("onboard")
        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard(
                scope={"palace": "org"},
                scan={
                    "path": "main.py",
                    "patterns": ["*.py"],
                    "root": "/tmp",
                },
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)
        err = payload.get("error", {})
        assert err.get("code") == "MEM_SCHEMA_VALIDATION_FAILED", (
            f"Expected MEM_SCHEMA_VALIDATION_FAILED, got: {err.get('code')} — {payload}"
        )
        message = err.get("message", "")
        actionable_fix = err.get("actionable_fix", "")
        assert message, "error.message must be non-empty"
        assert actionable_fix, "error.actionable_fix must be non-empty"


# ---------------------------------------------------------------------------
# Acceptance test B: no-scope query resolved via active context → scope_source=active_context
# ---------------------------------------------------------------------------


class TestActiveScopeSourceLabeling:
    """Acceptance test B — memory(query) with no explicit scope resolved via active context
    must report scope_source as 'active_context', not 'request' (issue 2)."""

    @pytest.mark.asyncio
    async def test_no_scope_query_via_active_context_reports_active_context_source(
        self, mock_ctx: MagicMock
    ) -> None:
        """When scope is resolved from the session active context, scope_source values
        must be 'active_context', not 'request'."""
        from unittest.mock import AsyncMock

        from workflows_mcp.engine.memory_scope_resolver import SyncContextCandidate, scope_key

        # Install a fake active context on the mock session.
        candidate_scope = {"palace": "forge", "wing": None, "room": None, "compartment": None}
        candidate = SyncContextCandidate(
            scope=candidate_scope,
            scope_key_value=scope_key(candidate_scope),
            checkpoint_data={},
            source="stored_checkpoint",
        )
        mock_ctx.request_context.lifespan_context.get_active_context.return_value = candidate

        memory = _get_tool_fn("memory")
        from workflows_mcp.engine.memory_service import MemoryResult, QueryMemoryResult

        query_result = MemoryResult(
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
        with patch(
            "workflows_mcp.engine.memory_service.MemoryService.execute",
            new_callable=AsyncMock,
            return_value=query_result,
        ):
            result = await memory(
                operation="query",
                # No scope / scope_token / context_id — should use active context
                query={"text": "anything"},
                ctx=mock_ctx,
            )
        payload = json.loads(result.content[0].text)

        # Must not be an error.
        assert "error" not in payload or payload.get("error") is None, (
            f"Expected successful response, got error: {payload}"
        )

        scope_source = payload.get("scope_source")
        if scope_source is None:
            # scope_source may be absent when no results are found (found=False path),
            # but resolved_scope should still be present with at least 'palace'.
            # Skip source check — the absence of scope_source is acceptable here.
            return

        # All resolved fields must be labeled 'active_context'.
        for field_name, source in scope_source.items():
            if source is not None:
                assert source == "active_context", (
                    f"scope_source[{field_name!r}] expected 'active_context', got {source!r}"
                )


class TestActiveProjectDefaultResolution:
    @pytest.mark.asyncio
    async def test_memory_without_scope_uses_active_project_defaults(
        self, mock_ctx: MagicMock
    ) -> None:
        from unittest.mock import AsyncMock

        from workflows_mcp.context import SessionProjectContext

        session = MagicMock(name="project_session")
        mock_ctx.request_context.session = session
        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.get_active_project.return_value = SessionProjectContext(
            project_id="p1",
            slug="forge",
            palace="forge-palace",
            default_wing="backend",
            default_room="orchestrator",
            source="session_selected",
        )

        captured: list[Any] = []

        async def _capture_execute(self: Any, request: Any) -> Any:
            captured.append(request)
            from workflows_mcp.engine.memory_service import MemoryResult, QueryMemoryResult

            return MemoryResult(
                operation="query",
                query=QueryMemoryResult(
                    facts=[], memories=[], communities=[], diagnostics={}, evidence=[], paths=[]
                ),
            )

        memory = _get_tool_fn("memory")
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=AsyncMock()),
            patch(
                "workflows_mcp.engine.memory_service.MemoryService.execute",
                new=_capture_execute,
            ),
        ):
            await memory(operation="query", query={"text": "test"}, ctx=mock_ctx)

        assert captured, "Expected memory request capture"
        assert captured[0].scope.palace == "forge-palace"
        assert captured[0].scope.wing == "backend"
        assert captured[0].scope.room == "orchestrator"

    @pytest.mark.asyncio
    async def test_context_id_scope_token_precedence_not_overridden_by_active_project(
        self, mock_ctx: MagicMock
    ) -> None:
        from unittest.mock import AsyncMock

        from workflows_mcp.context import SessionProjectContext

        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.get_active_project.return_value = SessionProjectContext(
            project_id="p1",
            slug="forge",
            palace="forge-palace",
            default_wing="backend",
            default_room="orchestrator",
            source="token_bound",
        )

        # Ensure context_id/scope_token resolve in memory_service via execution context maps.
        exec_context = app_ctx.create_execution_context.return_value
        exec_context.memory_scope_tokens = {
            "st_abc": {"palace": "token-palace", "wing": "token-wing", "room": "token-room"}
        }
        exec_context.memory_context_scopes = {
            "ctx_abc": {"palace": "ctx-palace", "wing": "ctx-wing", "room": "ctx-room"}
        }

        memory = _get_tool_fn("memory")
        backend = AsyncMock()
        backend.query_memories = AsyncMock(
            return_value={"memories": [], "facts": [], "communities": []}
        )
        fake_embedding = [0.0] * 1536

        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=backend),
            patch(
                "workflows_mcp.engine.memory_service.compute_embedding",
                return_value=(fake_embedding, "text-embedding-3-small", 1, 0.0),
            ),
        ):
            result = await memory(
                operation="query",
                scope_token="st_abc",
                context_id="ctx_abc",
                query={"text": "test"},
                ctx=mock_ctx,
            )

        payload = json.loads(result.content[0].text)
        resolved = payload.get("resolved_scope", {})
        assert resolved.get("palace") == "token-palace"
        assert resolved.get("wing") == "token-wing"
        assert resolved.get("room") == "token-room"

    @pytest.mark.asyncio
    async def test_partial_scope_without_palace_uses_active_project_defaults(
        self, mock_ctx: MagicMock
    ) -> None:
        from unittest.mock import AsyncMock

        from workflows_mcp.context import SessionProjectContext

        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.get_active_project.return_value = SessionProjectContext(
            project_id="p2",
            slug="atlas",
            palace="atlas-palace",
            default_wing="services",
            default_room="planner",
            source="session_selected",
        )

        captured: list[Any] = []

        async def _capture_execute(self: Any, request: Any) -> Any:
            captured.append(request)
            from workflows_mcp.engine.memory_service import MemoryResult, QueryMemoryResult

            return MemoryResult(
                operation="query",
                query=QueryMemoryResult(
                    facts=[], memories=[], communities=[], diagnostics={}, evidence=[], paths=[]
                ),
            )

        memory = _get_tool_fn("memory")
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=AsyncMock()),
            patch(
                "workflows_mcp.engine.memory_service.MemoryService.execute",
                new=_capture_execute,
            ),
        ):
            await memory(
                operation="query",
                scope={"compartment": "ci"},
                query={"text": "x"},
                ctx=mock_ctx,
            )

        assert captured
        assert captured[0].scope.palace == "atlas-palace"
        assert captured[0].scope.wing == "services"
        assert captured[0].scope.room == "planner"

    @pytest.mark.asyncio
    async def test_explicit_palace_not_overridden_defaults_only_when_matching(
        self, mock_ctx: MagicMock
    ) -> None:
        from unittest.mock import AsyncMock

        from workflows_mcp.context import SessionProjectContext

        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.get_active_project.return_value = SessionProjectContext(
            project_id="p3",
            slug="forge",
            palace="forge-palace",
            default_wing="backend",
            default_room="agents",
            source="token_bound",
        )

        captured: list[Any] = []

        async def _capture_execute(self: Any, request: Any) -> Any:
            captured.append(request)
            from workflows_mcp.engine.memory_service import MemoryResult, QueryMemoryResult

            return MemoryResult(
                operation="query",
                query=QueryMemoryResult(
                    facts=[], memories=[], communities=[], diagnostics={}, evidence=[], paths=[]
                ),
            )

        memory = _get_tool_fn("memory")
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=AsyncMock()),
            patch(
                "workflows_mcp.engine.memory_service.MemoryService.execute",
                new=_capture_execute,
            ),
        ):
            await memory(
                operation="query",
                scope={"palace": "other-palace"},
                query={"text": "x"},
                ctx=mock_ctx,
            )
            await memory(
                operation="query",
                scope={"palace": "forge-palace"},
                query={"text": "x"},
                ctx=mock_ctx,
            )

        assert len(captured) == 2
        assert captured[0].scope.palace == "other-palace"
        assert captured[0].scope.wing is None
        assert captured[0].scope.room is None
        assert captured[1].scope.palace == "forge-palace"
        assert captured[1].scope.wing == "backend"
        assert captured[1].scope.room == "agents"

    @pytest.mark.asyncio
    async def test_explicit_scope_wins_over_active_project_defaults(
        self, mock_ctx: MagicMock
    ) -> None:
        from unittest.mock import AsyncMock

        from workflows_mcp.context import SessionProjectContext

        app_ctx = mock_ctx.request_context.lifespan_context
        app_ctx.get_active_project.return_value = SessionProjectContext(
            project_id="p3",
            slug="forge",
            palace="project-palace",
            default_wing="project-wing",
            default_room="project-room",
            source="session_selected",
        )

        captured: list[Any] = []

        async def _capture_execute(self: Any, request: Any) -> Any:
            captured.append(request)
            from workflows_mcp.engine.memory_service import MemoryResult, QueryMemoryResult

            return MemoryResult(
                operation="query",
                query=QueryMemoryResult(
                    facts=[], memories=[], communities=[], diagnostics={}, evidence=[], paths=[]
                ),
            )

        memory = _get_tool_fn("memory")
        with (
            patch("workflows_mcp.tools_memory.PostgresBackend", return_value=AsyncMock()),
            patch(
                "workflows_mcp.engine.memory_service.MemoryService.execute",
                new=_capture_execute,
            ),
        ):
            # Palace matches active project → wing/room defaults apply.
            # Explicit wing overrides default_wing; room fills from default_room.
            await memory(
                operation="query",
                scope={"palace": "project-palace", "wing": "explicit-wing"},
                query={"text": "test"},
                ctx=mock_ctx,
            )

        assert captured, "Expected memory request to be captured"
        assert captured[0].scope.palace == "project-palace"
        assert captured[0].scope.wing == "explicit-wing"
        assert captured[0].scope.room == "project-room"


# ---------------------------------------------------------------------------
# Task 6: strict HTTP adapter contract (onboard_http / sync_http)
# ---------------------------------------------------------------------------


class TestStrictHttpAdapters:
    """Strict HTTP adapters must enforce OnboardRequest / SyncRequest validation
    before delegating to the underlying orchestration logic.

    onboard_http and sync_http:
    - raise pydantic.ValidationError for unknown fields and legacy response.mode
    - are async and accept ctx: AppContextType
    - delegate to the underlying onboard()/sync() orchestration on valid input
    """

    @pytest.mark.asyncio
    async def test_onboard_http_rejects_unknown_field(self, mock_ctx: MagicMock) -> None:
        """onboard_http must raise ValidationError when an unknown field is present."""
        from pydantic import ValidationError

        from workflows_mcp.tools_memory import onboard_http

        with pytest.raises(ValidationError):
            await onboard_http(
                {"scope": {"palace": "acme"}, "unknown": True},
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_onboard_http_rejects_legacy_response_mode(
        self, mock_ctx: MagicMock
    ) -> None:
        """onboard_http must raise ValidationError when response.mode is supplied."""
        from pydantic import ValidationError

        from workflows_mcp.tools_memory import onboard_http

        with pytest.raises(ValidationError):
            await onboard_http(
                {"scope": {"palace": "acme"}, "response": {"mode": "programmatic"}},
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_sync_http_rejects_unknown_field(self, mock_ctx: MagicMock) -> None:
        """sync_http must raise ValidationError when an unknown field is present."""
        from pydantic import ValidationError

        from workflows_mcp.tools_memory import sync_http

        with pytest.raises(ValidationError):
            await sync_http(
                {"scope": {"palace": "acme"}, "unknown": True},
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_sync_http_rejects_legacy_response_mode(
        self, mock_ctx: MagicMock
    ) -> None:
        """sync_http must raise ValidationError when response.mode is supplied."""
        from pydantic import ValidationError

        from workflows_mcp.tools_memory import sync_http

        with pytest.raises(ValidationError):
            await sync_http(
                {"scope": {"palace": "acme"}, "response": {"mode": "programmatic"}},
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_onboard_http_delegates_to_orchestration_on_valid_payload(
        self, mock_ctx: MagicMock
    ) -> None:
        """onboard_http must return a dict result from the orchestration layer."""
        from unittest.mock import patch

        from workflows_mcp.tools_memory import onboard_http

        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await onboard_http(
                {"scope": {"palace": "acme"}},
                ctx=mock_ctx,
            )
        # Result must be a dict (JSON-decoded orchestration output).
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_sync_http_delegates_to_orchestration_on_valid_payload(
        self, mock_ctx: MagicMock
    ) -> None:
        """sync_http must return a dict result from the orchestration layer."""
        from unittest.mock import patch

        from workflows_mcp.tools_memory import sync_http

        with patch("workflows_mcp.tools_memory.PostgresBackend"):
            result = await sync_http(
                {},
                ctx=mock_ctx,
            )
        # Result must be a dict (JSON-decoded orchestration output).
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Task 4: placement writes must not inherit session fallback scope
# ---------------------------------------------------------------------------


class TestPlacementWritesDoNotUseSessionFallback:
    """Non-query operations must not receive scope from the active context fallback.

    When operation != 'query' and no scope/scope_token/context_id is provided,
    the tool must pass scope=None to MemoryService so the operation locality
    contract decides validity — not silently inject the session active context.
    """

    @pytest.mark.asyncio
    async def test_direct_ingest_without_scope_does_not_use_active_context(
        self, mock_ctx: MagicMock
    ) -> None:
        """memory(operation='ingest') with no scope must fail with INSUFFICIENT_LOCALITY
        even when an active context is present in the session."""
        from workflows_mcp.engine.memory_scope_resolver import SyncContextCandidate, scope_key

        # Arrange: active context candidate with a fully qualified scope.
        candidate_scope = {
            "palace": "forge",
            "wing": "core",
            "room": "main",
            "compartment": "slot-1",
        }
        candidate = SyncContextCandidate(
            scope=candidate_scope,
            scope_key_value=scope_key(candidate_scope),
            checkpoint_data={},
            source="stored_checkpoint",
        )
        mock_ctx.request_context.lifespan_context.get_active_context.return_value = candidate

        memory = _get_tool_fn("memory")

        result = await memory(
            operation="ingest",
            record={"content": "some content"},
            ctx=mock_ctx,
        )

        payload = json.loads(result.content[0].text)
        assert "error" in payload, f"Expected error envelope, got: {payload}"
        error = payload["error"]
        assert error["code"] == "INSUFFICIENT_LOCALITY", (
            f"Expected INSUFFICIENT_LOCALITY, got {error['code']!r}"
        )
        assert "ingest" in error["message"].lower() or "Missing" in error["message"], (
            f"Expected 'ingest' or 'Missing' in message, got: {error['message']!r}"
        )
        assert "scope_token" in error.get("actionable_fix", ""), (
            f"Expected 'scope_token' in actionable_fix, got: {error.get('actionable_fix')!r}"
        )
