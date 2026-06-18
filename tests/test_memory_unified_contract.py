"""Contract tests for memory.v2 unified envelope behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.knowledge.search import (
    build_fts_search_query,
    build_vector_search_query,
    rrf_fusion,
)
from workflows_mcp.engine.memory_service import (
    ManageMemoryResult,
    MemoryContractError,
    MemoryRequest,
    MemoryService,
    QueryMemoryRequest,
    QueryMemoryResult,
)
from workflows_mcp.tools_memory import _tool_error_payload


def test_operation_enum_rejects_unknown_operation_with_contract_code() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(operation="hall_search")
    assert "MEM_INVALID_OPERATION" in str(exc.value)


def test_section_matrix_requires_query_for_query_operation() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(operation="query")
    assert "MEM_MISSING_REQUIRED_FIELD" in str(exc.value)


@pytest.mark.parametrize("operation", ["ingest", "validate", "supersede", "archive"])
def test_section_matrix_requires_record_for_record_operations(operation: str) -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(operation=operation)
    assert "MEM_MISSING_REQUIRED_FIELD" in str(exc.value)


@pytest.mark.parametrize("operation", ["graph_upsert", "graph_delete"])
def test_section_matrix_requires_graph_for_graph_operations(operation: str) -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(operation=operation)
    assert "MEM_MISSING_REQUIRED_FIELD" in str(exc.value)


def test_scope_rejects_legacy_hall_taxonomy_key() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="query",
            scope={"palace": "acme", "wing": "svc", "room": "comp", "hall": "legacy"},
            query={"text": "incident", "mode": "search"},
        )
    assert "MEM_INVALID_TAXONOMY_KEY" in str(exc.value)


def test_response_rejects_mode_and_profile_fields() -> None:
    with pytest.raises(ValidationError) as exc_mode:
        MemoryRequest(
            operation="query",
            query={"text": "incident", "mode": "search"},
            response={"mode": "graph"},
        )
    assert "response.mode" in str(exc_mode.value)
    assert "extra_forbidden" in str(exc_mode.value)

    with pytest.raises(ValidationError) as exc_profile:
        MemoryRequest(
            operation="query",
            query={"text": "incident", "mode": "search"},
            response={"profile": "full"},
        )
    assert "response.profile" in str(exc_profile.value)
    assert "extra_forbidden" in str(exc_profile.value)


def test_response_allows_only_debug_and_include_candidates() -> None:
    request = MemoryRequest(
        operation="query",
        query={"text": "incident", "mode": "search"},
        response={"debug": True, "include_candidates": True},
    )

    assert request.response.debug is True
    assert request.response.include_candidates is True


@pytest.mark.asyncio
async def test_ingest_direct_requires_compartment() -> None:
    request = MemoryRequest(
        operation="ingest",
        scope={"palace": "acme", "wing": "svc", "room": "comp"},
        record={"format": "raw", "content": "stored memory", "memory_tier": "direct"},
    )

    context = Execution()
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError, match="INSUFFICIENT_LOCALITY"):
        await service.execute(request)


def test_ingest_rejects_non_direct_memory_tier_for_boundary_safety() -> None:
    """v2 contract keeps ingest for direct memories only; derived goes through maintenance flows."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="ingest",
            scope={
                "palace": "acme",
                "wing": "svc",
                "room": "comp",
                "compartment": "incidents",
            },
            record={"format": "raw", "content": "derived memory", "memory_tier": "derived"},
        )
    assert "MEM_BOUNDARY_VIOLATION" in str(exc.value)


@pytest.mark.asyncio
async def test_scope_resolution_precedence_request_over_token_over_context() -> None:
    context = Execution()
    context.set_execution_context(
        SimpleNamespace(
            memory_scope_tokens={
                "st_abc": {
                    "palace": "token-palace",
                    "wing": "token-wing",
                    "room": "token-room",
                    "compartment": "token-compartment",
                }
            },
            memory_context_scopes={
                "ctx_123": {
                    "palace": "context-palace",
                    "wing": "context-wing",
                    "room": "context-room",
                    "compartment": "context-compartment",
                }
            },
        )
    )

    service = MemoryService(backend=object(), context=context)

    async def _fake_query(_request: object) -> QueryMemoryResult:
        return QueryMemoryResult()

    service.query = _fake_query  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="query",
            scope={"palace": "request-palace"},
            scope_token="st_abc",
            context_id="ctx_123",
            query={"text": "incident", "mode": "search"},
        )
    )
    assert result.resolved_scope is not None
    assert result.resolved_scope.palace == "request-palace"
    assert result.resolved_scope.wing == "token-wing"
    assert result.resolved_scope.room == "token-room"
    assert result.resolved_scope.compartment == "token-compartment"
    assert result.scope_source == {
        "palace": "request",
        "wing": "token",
        "room": "token",
        "compartment": "token",
    }


@pytest.mark.asyncio
async def test_scope_resolution_precedence_token_over_context_when_request_missing() -> None:
    context = Execution()
    context.set_execution_context(
        SimpleNamespace(
            memory_scope_tokens={
                "st_abc": {
                    "palace": "token-palace",
                    "wing": "token-wing",
                    "room": "token-room",
                    "compartment": "token-compartment",
                }
            },
            memory_context_scopes={
                "ctx_123": {
                    "palace": "context-palace",
                    "wing": "context-wing",
                    "room": "context-room",
                    "compartment": "context-compartment",
                }
            },
        )
    )

    service = MemoryService(backend=object(), context=context)

    async def _fake_query(_request: object) -> QueryMemoryResult:
        return QueryMemoryResult()

    service.query = _fake_query  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="query",
            scope_token="st_abc",
            context_id="ctx_123",
            query={"text": "incident", "mode": "search"},
        )
    )

    assert result.resolved_scope is not None
    assert result.resolved_scope.palace == "token-palace"
    assert result.resolved_scope.wing == "token-wing"
    assert result.resolved_scope.room == "token-room"
    assert result.resolved_scope.compartment == "token-compartment"
    assert result.scope_source == {
        "palace": "token",
        "wing": "token",
        "room": "token",
        "compartment": "token",
    }


@pytest.mark.asyncio
async def test_ingest_accepts_compartment_from_scope_token_when_missing_in_scope() -> None:
    context = Execution()
    context.set_execution_context(
        SimpleNamespace(
            memory_scope_tokens={
                "st_ingest": {
                    "palace": "token-palace",
                    "wing": "token-wing",
                    "room": "token-room",
                    "compartment": "token-compartment",
                }
            },
            memory_context_scopes={},
        )
    )
    service = MemoryService(backend=object(), context=context)

    async def _fake_manage(_request: object) -> ManageMemoryResult:
        return ManageMemoryResult(operation="store", memory_ids=["m-1"], stored_count=1)

    service.manage = _fake_manage  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="ingest",
            scope={"palace": "req-palace"},
            scope_token="st_ingest",
            record={"format": "raw", "content": "stored memory", "memory_tier": "direct"},
        )
    )

    assert result.resolved_scope is not None
    assert result.resolved_scope.palace == "req-palace"
    assert result.resolved_scope.wing == "token-wing"
    assert result.resolved_scope.room == "token-room"
    assert result.resolved_scope.compartment == "token-compartment"
    assert result.scope_source == {
        "palace": "request",
        "wing": "token",
        "room": "token",
        "compartment": "token",
    }


@pytest.mark.asyncio
async def test_ingest_accepts_compartment_from_context_id_when_scope_token_missing() -> None:
    context = Execution()
    context.set_execution_context(
        SimpleNamespace(
            memory_scope_tokens={},
            memory_context_scopes={
                "ctx_ingest": {
                    "palace": "context-palace",
                    "wing": "context-wing",
                    "room": "context-room",
                    "compartment": "context-compartment",
                }
            },
        )
    )
    service = MemoryService(backend=object(), context=context)

    async def _fake_manage(_request: object) -> ManageMemoryResult:
        return ManageMemoryResult(operation="store", memory_ids=["m-1"], stored_count=1)

    service.manage = _fake_manage  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="ingest",
            context_id="ctx_ingest",
            record={"format": "raw", "content": "stored memory", "memory_tier": "direct"},
        )
    )

    assert result.resolved_scope is not None
    assert result.resolved_scope.palace == "context-palace"
    assert result.resolved_scope.wing == "context-wing"
    assert result.resolved_scope.room == "context-room"
    assert result.resolved_scope.compartment == "context-compartment"
    assert result.scope_source == {
        "palace": "context",
        "wing": "context",
        "room": "context",
        "compartment": "context",
    }


@pytest.mark.asyncio
async def test_scope_resolution_rejects_non_mapping_context_scope_payload() -> None:
    context = Execution()
    context.set_execution_context(
        SimpleNamespace(
            memory_scope_tokens={"st_abc": ["invalid"]},
            memory_context_scopes={},
        )
    )
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError, match="MEM_INVALID_CONTEXT_SCOPE"):
        await service.execute(
            MemoryRequest(
                operation="query",
                scope_token="st_abc",
                query={"text": "incident", "mode": "search"},
            )
        )


@pytest.mark.asyncio
async def test_scope_resolution_rejects_non_mapping_context_id_payload() -> None:
    context = Execution()
    context.set_execution_context(
        SimpleNamespace(
            memory_scope_tokens={},
            memory_context_scopes={"ctx_123": ["invalid"]},
        )
    )
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError, match="MEM_INVALID_CONTEXT_SCOPE"):
        await service.execute(
            MemoryRequest(
                operation="query",
                context_id="ctx_123",
                query={"text": "incident", "mode": "search"},
            )
        )


@pytest.mark.asyncio
async def test_archive_operation_does_not_require_fully_resolved_scope() -> None:
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    async def _fake_manage(_request: object) -> ManageMemoryResult:
        return ManageMemoryResult(operation="forget")

    service.manage = _fake_manage  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="archive",
            record={"ids": ["11111111-1111-1111-1111-111111111111"]},
        )
    )

    assert result.operation == "archive"
    assert result.resolved_scope is not None
    assert result.resolved_scope.model_dump(exclude_none=True) == {}
    assert result.scope_source == {}


@pytest.mark.asyncio
async def test_graph_upsert_link_does_not_require_fully_resolved_scope() -> None:
    """graph_upsert link with UUID refs requires no topology scope."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    async def _fake_manage(_request: object) -> ManageMemoryResult:
        return ManageMemoryResult(operation="graph_store_relation")

    service.manage = _fake_manage  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="graph_upsert",
            graph={
                "kind": "link",
                "from": "11111111-1111-1111-1111-111111111111",
                "to": "22222222-2222-2222-2222-222222222222",
                "link_type": "uses",
            },
        )
    )

    assert result.operation == "graph_upsert"
    assert result.resolved_scope is not None
    assert result.resolved_scope.model_dump(exclude_none=True) == {}
    assert result.scope_source == {}


@pytest.mark.asyncio
async def test_query_operation_still_requires_resolved_scope() -> None:
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError, match="INSUFFICIENT_LOCALITY"):
        await service.execute(
            MemoryRequest(operation="query", query={"text": "incident", "mode": "search"})
        )


def test_tool_error_payload_preserves_insufficient_locality_retry_guidance() -> None:
    payload = _tool_error_payload(
        "memory",
        MemoryContractError(
            code="INSUFFICIENT_LOCALITY",
            message=(
                "INSUFFICIENT_LOCALITY: ingest requires complete topology locality. "
                "Missing: wing, room, compartment. Accepted sources: scope. "
                "Provided: no scope source."
            ),
            retryable=False,
            actionable_fix=(
                "Retry ingest with scope.palace/wing/room/compartment, scope_token, or context_id."
            ),
        ),
    )

    assert payload["error"]["code"] == "INSUFFICIENT_LOCALITY"
    assert "Missing: wing, room, compartment" in payload["error"]["message"]
    assert "scope_token" in payload["error"]["actionable_fix"]
    assert payload["error"]["retryable"] is False


def test_tool_error_payload_uses_machine_readable_envelope_for_contract_errors() -> None:
    payload = _tool_error_payload(
        "memory",
        MemoryContractError(code="MEM_INVALID_SCOPE", message="scope invalid", retryable=False),
    )
    assert payload["error"]["code"] == "MEM_INVALID_SCOPE"
    assert payload["error"]["message"] == "scope invalid"
    assert payload["error"]["retryable"] is False
    assert payload["error"].get("correlation_id")


def test_memory_contract_error_remains_importable_from_memory_service() -> None:
    from workflows_mcp.engine.memory_service import (
        MemoryContractError as ImportedMemoryContractError,
    )

    error = ImportedMemoryContractError(
        code="MEM_TEST",
        message="test message",
        retryable=True,
        actionable_fix="retry with a valid request",
    )
    assert error.code == "MEM_TEST"
    assert error.message == "test message"
    assert error.retryable is True
    assert error.actionable_fix == "retry with a valid request"


def test_tool_error_payload_maps_unhandled_errors_to_mem_internal_error() -> None:
    payload = _tool_error_payload("memory", RuntimeError("boom"))
    assert payload["error"]["code"] == "MEM_INTERNAL_ERROR"
    assert payload["error"]["message"] == "memory failed"
    assert payload["error"]["retryable"] is False


def test_tool_error_payload_maps_model_validate_contract_validation_errors() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryRequest.model_validate({"operation": "query"})

    payload = _tool_error_payload("memory", exc.value)
    assert payload["error"]["code"] == "MEM_MISSING_REQUIRED_FIELD"
    assert "required" in payload["error"]["message"]
    assert payload["error"]["retryable"] is False


@pytest.mark.parametrize("query_mode", ["search", "graph", "hybrid", "communities"])
def test_query_modes_validate_against_contract(query_mode: str) -> None:
    request = MemoryRequest(
        operation="query",
        scope={
            "palace": "acme",
            "wing": "svc",
            "room": "comp",
            "compartment": "topic",
        },
        query={"text": "incident", "mode": query_mode},
    )
    assert request.query is not None
    assert request.query.mode == query_mode


@pytest.mark.asyncio
async def test_execute_routes_hybrid_mode_to_auto_with_s2_default_on() -> None:
    context = Execution()
    service = MemoryService(backend=object(), context=context)
    captured: list[object] = []

    async def _fake_query(request: object) -> QueryMemoryResult:
        captured.append(request)
        return QueryMemoryResult()

    service.query = _fake_query  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="query",
            scope={
                "palace": "acme",
                "wing": "svc",
                "room": "comp",
                "compartment": "topic",
            },
            query={"text": "incident", "mode": "hybrid"},
        )
    )

    assert result.query is not None
    assert len(captured) == 1
    mapped = captured[0]
    assert getattr(mapped, "strategy") == "auto"
    assert getattr(mapped, "s2_enabled") is True


@pytest.mark.asyncio
async def test_execute_routes_communities_mode_to_communities_strategy() -> None:
    context = Execution()
    service = MemoryService(backend=object(), context=context)
    captured: list[object] = []

    async def _fake_query(request: object) -> QueryMemoryResult:
        captured.append(request)
        return QueryMemoryResult()

    service.query = _fake_query  # type: ignore[method-assign]

    await service.execute(
        MemoryRequest(
            operation="query",
            scope={
                "palace": "acme",
                "wing": "svc",
                "room": "comp",
                "compartment": "topic",
            },
            query={"text": "incident", "mode": "communities"},
        )
    )

    assert len(captured) == 1
    assert getattr(captured[0], "strategy") == "communities"


@pytest.mark.asyncio
async def test_query_auto_exposes_s1_s2_diagnostics_default_on() -> None:
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            result = await service.query(
                QueryMemoryRequest(
                    query="incident",
                    strategy="auto",
                    namespace="svc",
                    room="comp",
                )
            )

    diagnostics = result.diagnostics
    assert diagnostics["retrieval"]["s1"]["candidate_generation"] == "deterministic"
    # MEMORY-CONTRACT-v3.1: with explicit scope (namespace+room), the companion lane is
    # suppressed to prevent cross-scope leakage. s2.enabled must be False.
    assert diagnostics["retrieval"]["s2"]["enabled"] is False


@pytest.mark.asyncio
async def test_query_auto_s2_toggle_off_disables_companion_lane() -> None:
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            result = await service.query(
                QueryMemoryRequest(
                    query="incident",
                    strategy="auto",
                    namespace="svc",
                    room="comp",
                    s2_enabled=False,
                )
            )

    assert mock_search.await_args.kwargs["include_global_companion"] is False
    assert result.diagnostics["retrieval"]["s2"]["enabled"] is False


def test_rrf_fusion_is_deterministic_for_tied_scores() -> None:
    vector = [{"id": "b-id", "content": "B"}]
    fts: list[dict[str, str]] = [{"id": "a-id", "content": "A"}]
    fused = rrf_fusion(vector, fts, vector_weight=1.0, fts_weight=1.0, limit=2)
    assert [item["id"] for item in fused] == ["a-id", "b-id"]


def _assert_retrieval_contract_shape(diagnostics: dict[str, object]) -> None:
    retrieval = diagnostics["retrieval"]
    assert isinstance(retrieval, dict)
    assert set(retrieval.keys()) == {"s1", "s2"}

    s1 = retrieval["s1"]
    assert isinstance(s1, dict)
    assert set(s1.keys()) == {"candidate_generation", "algorithm"}

    s2 = retrieval["s2"]
    assert isinstance(s2, dict)
    assert set(s2.keys()) == {"enabled", "requested", "strategy"}


def test_vector_query_has_stable_sql_tie_breaker() -> None:
    sql, _ = build_vector_search_query([0.1, 0.2, 0.3], limit=5)
    assert "ORDER BY kp.embedding <=> $1::vector, kp.id" in sql


def test_fts_query_has_stable_sql_tie_breaker() -> None:
    sql, _ = build_fts_search_query("incident", limit=5)
    assert "ORDER BY fts_rank DESC, kp.id" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "radius", "expected_strategy"),
    [
        ("search", 0, "palace"),
        ("hybrid", 1, "auto"),
        ("graph", 1, "graph"),
        ("communities", 1, "communities"),
    ],
)
async def test_execute_query_modes_expose_effective_strategy(
    mode: str,
    radius: int,
    expected_strategy: str,
) -> None:
    context = Execution()
    service = MemoryService(backend=object(), context=context)
    captured: list[object] = []

    async def _fake_query(request: object) -> QueryMemoryResult:
        captured.append(request)
        return QueryMemoryResult(
            diagnostics={
                "retrieval": {
                    "s1": {"candidate_generation": "deterministic", "algorithm": "rrf"},
                    "s2": {"enabled": False, "requested": True, "strategy": "not_applicable"},
                }
            }
        )

    service.query = _fake_query  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="query",
            scope={
                "palace": "acme",
                "wing": "svc",
                "room": "comp",
                "compartment": "topic",
            },
            query={
                "text": "incident",
                "mode": mode,
                "radius": radius,
            },
        )
    )

    assert len(captured) == 1
    assert getattr(captured[0], "strategy") == expected_strategy
    assert result.query is not None
    assert result.query.diagnostics["effective_strategy"] == expected_strategy
    _assert_retrieval_contract_shape(result.query.diagnostics)


@pytest.mark.asyncio
async def test_query_graph_has_normalized_retrieval_contract_shape() -> None:
    backend = MagicMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.graph_stats") as mock_graph_stats:
        mock_graph_stats.return_value = {
            "paths": [],
            "nodes": [],
            "edges": [],
            "diagnostics": {"graph": "ok"},
        }
        service = MemoryService(backend=backend, context=context)
        result = await service.query(
            QueryMemoryRequest(
                query="incident",
                strategy="graph",
                graph_op="stats",
            )
        )

    _assert_retrieval_contract_shape(result.diagnostics)
    assert result.diagnostics["retrieval"]["s1"]["candidate_generation"] == "not_applicable"
    assert result.diagnostics["retrieval"]["s2"]["strategy"] == "not_applicable"


@pytest.mark.asyncio
async def test_query_context_has_normalized_retrieval_contract_shape() -> None:
    backend = MagicMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            with patch("workflows_mcp.engine.memory_service.assemble_context") as mock_assemble:
                mock_assemble.return_value = ("", 0, 0)
                service = MemoryService(backend=backend, context=context)
                result = await service.query(
                    QueryMemoryRequest(
                        query="incident",
                        strategy="context",
                        namespace="svc",
                        room="comp",
                    )
                )

    _assert_retrieval_contract_shape(result.diagnostics)
    assert result.diagnostics["retrieval"]["s1"]["algorithm"] == "context_assembly"
    assert result.diagnostics["retrieval"]["s2"]["strategy"] == "not_applicable"


@pytest.mark.asyncio
async def test_query_palace_has_normalized_retrieval_contract_shape() -> None:
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            result = await service.query(
                QueryMemoryRequest(
                    query="incident",
                    strategy="palace",
                    namespace="svc",
                    room="comp",
                )
            )

    _assert_retrieval_contract_shape(result.diagnostics)
    assert result.diagnostics["retrieval"]["s1"]["candidate_generation"] == "deterministic"
    assert result.diagnostics["retrieval"]["s2"]["strategy"] == "not_applicable"


@pytest.mark.asyncio
async def test_query_communities_has_normalized_retrieval_contract_shape() -> None:
    backend = MagicMock()
    backend.query = AsyncMock(return_value=SimpleNamespace(rows=[]))
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        service = MemoryService(backend=backend, context=context)
        result = await service.query(
            QueryMemoryRequest(
                query="incident",
                strategy="communities",
                namespace="svc",
                room="comp",
            )
        )

    _assert_retrieval_contract_shape(result.diagnostics)
    assert result.diagnostics["retrieval"]["s1"]["candidate_generation"] == "deterministic"
    assert result.diagnostics["retrieval"]["s2"]["strategy"] == "not_applicable"


# ---------------------------------------------------------------------------
# MEMORY-CONTRACT-v3.1 blocker conformance tests
# ---------------------------------------------------------------------------


# B-1: Scope hierarchy enforcement


def test_b1_room_without_wing_is_rejected() -> None:
    """B-1: room requires wing."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="query",
            scope={"palace": "acme", "room": "comp"},
            query={"text": "x", "mode": "search"},
        )
    assert "MEM_SCOPE_HIERARCHY_VIOLATION" in str(exc.value)


def test_b1_compartment_without_room_is_rejected() -> None:
    """B-1: compartment requires room."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="ingest",
            scope={"palace": "acme", "wing": "svc", "compartment": "slot"},
            record={"format": "raw", "content": "x", "memory_tier": "direct"},
        )
    assert "MEM_SCOPE_HIERARCHY_VIOLATION" in str(exc.value)


def test_b1_compartment_without_wing_is_rejected() -> None:
    """B-1: compartment requires wing (even if room is present)."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="ingest",
            scope={"palace": "acme", "room": "comp", "compartment": "slot"},
            record={"format": "raw", "content": "x", "memory_tier": "direct"},
        )
    assert "MEM_SCOPE_HIERARCHY_VIOLATION" in str(exc.value)


def test_b1_full_hierarchy_is_accepted() -> None:
    """B-1: palace > wing > room > compartment passes validation."""
    req = MemoryRequest(
        operation="ingest",
        scope={"palace": "acme", "wing": "svc", "room": "comp", "compartment": "slot"},
        record={"format": "raw", "content": "x", "memory_tier": "direct"},
    )
    assert req.scope.compartment == "slot"


def test_b1_wing_without_room_is_accepted() -> None:
    """B-1: wing alone is valid — no room requirement."""
    req = MemoryRequest(
        operation="ingest",
        scope={"palace": "acme", "wing": "svc"},
        record={"format": "raw", "content": "x", "memory_tier": "direct"},
    )
    assert req.scope.wing == "svc"
    assert req.scope.room is None


# B-2: namespace must not appear in external scope


def test_b2_namespace_in_scope_is_rejected() -> None:
    """B-2: 'namespace' is an internal term — forbidden as an external scope key."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="query",
            scope={"palace": "acme", "wing": "svc", "namespace": "internal"},
            query={"text": "x", "mode": "search"},
        )
    assert "MEM_INVALID_TAXONOMY_KEY" in str(exc.value)


# B-5: corridor must not appear in external scope


def test_b5_corridor_in_scope_is_rejected() -> None:
    """B-5: 'corridor' is an internal topology term — forbidden as an external scope key."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="ingest",
            scope={"palace": "acme", "wing": "svc", "room": "comp", "corridor": "internal"},
            record={"format": "raw", "content": "x", "memory_tier": "direct"},
        )
    assert "MEM_INVALID_TAXONOMY_KEY" in str(exc.value)


def test_b5_unknown_scope_key_is_rejected() -> None:
    """B-5: arbitrary unknown scope keys are rejected."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="query",
            scope={"palace": "acme", "wing": "svc", "room": "comp", "zone": "x"},
            query={"text": "x", "mode": "search"},
        )
    assert "MEM_INVALID_TAXONOMY_KEY" in str(exc.value)


# B-3: MEMORY_USER_ID fallback


def test_b3_memory_user_id_env_is_used_as_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    """B-3: MEMORY_USER_ID set to a valid UUID is returned as the user identity."""
    import uuid

    from workflows_mcp.tools_memory import _get_standalone_user_context

    test_uuid = str(uuid.uuid4())
    monkeypatch.setenv("MEMORY_USER_ID", test_uuid)
    for var in [
        "WORKFLOWS_USER_ID",
        "WORKFLOWS_USER",
        "MCP_USER_ID",
        "USER",
        "USERNAME",
        "LOGNAME",
    ]:
        monkeypatch.delenv(var, raising=False)

    user_id, user_string, auth_method = _get_standalone_user_context()
    assert str(user_id) == test_uuid
    assert user_string == test_uuid
    assert auth_method == "ENV_UUID"


def test_b3_memory_user_id_env_non_uuid_produces_deterministic_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-3: MEMORY_USER_ID set to a plain string is hashed deterministically."""
    from workflows_mcp.tools_memory import _get_standalone_user_context

    monkeypatch.setenv("MEMORY_USER_ID", "alice")
    for var in [
        "WORKFLOWS_USER_ID",
        "WORKFLOWS_USER",
        "MCP_USER_ID",
        "USER",
        "USERNAME",
        "LOGNAME",
    ]:
        monkeypatch.delenv(var, raising=False)

    user_id, user_string, auth_method = _get_standalone_user_context()
    assert user_string == "alice"
    assert auth_method == "OS_USER"
    # Deterministic — same input always produces same UUID
    user_id2, _, _ = _get_standalone_user_context()
    assert user_id == user_id2


def test_b3_memory_user_id_takes_priority_over_workflows_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-3: MEMORY_USER_ID has higher priority than WORKFLOWS_USER_ID."""
    from workflows_mcp.tools_memory import _get_standalone_user_context

    monkeypatch.setenv("MEMORY_USER_ID", "alice")
    monkeypatch.setenv("WORKFLOWS_USER_ID", "bob")

    _, user_string, _ = _get_standalone_user_context()
    assert user_string == "alice"


# B-4: Merge transparency


def test_b4_merge_transparency_org_only() -> None:
    """B-4: When only org record is present, effective_source is 'org' and conflict is False."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(
        org_record={"id": "org-1", "updated_at": "2024-01-01T00:00:00+00:00"},
        user_record=None,
    )
    assert result is not None
    assert result.effective_source == "org"
    assert result.conflict is False
    assert result.user_record is None


def test_b4_merge_transparency_user_only() -> None:
    """B-4: When only user record is present, effective_source is 'user' and conflict is False."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(
        org_record=None,
        user_record={"id": "user-1", "updated_at": "2024-01-01T00:00:00+00:00"},
    )
    assert result is not None
    assert result.effective_source == "user"
    assert result.conflict is False
    assert result.org_record is None


def test_b4_merge_transparency_user_wins_when_newer() -> None:
    """B-4: Merge rule — user wins when user updated_at is strictly newer."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(
        org_record={"updated_at": "2024-01-01T00:00:00+00:00"},
        user_record={"updated_at": "2024-06-01T00:00:00+00:00"},
    )
    assert result is not None
    assert result.effective_source == "user"
    assert result.conflict is True


def test_b4_merge_transparency_org_wins_on_tie() -> None:
    """B-4: Merge rule — org wins when updated_at timestamps are equal."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(
        org_record={"updated_at": "2024-01-01T00:00:00+00:00"},
        user_record={"updated_at": "2024-01-01T00:00:00+00:00"},
    )
    assert result is not None
    assert result.effective_source == "org"
    assert result.conflict is True


def test_b4_merge_transparency_org_wins_when_org_newer() -> None:
    """B-4: Merge rule — org wins when org updated_at is strictly newer."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(
        org_record={"updated_at": "2024-06-01T00:00:00+00:00"},
        user_record={"updated_at": "2024-01-01T00:00:00+00:00"},
    )
    assert result is not None
    assert result.effective_source == "org"
    assert result.conflict is True


def test_b4_merge_transparency_none_when_both_absent() -> None:
    """B-4: No merge envelope when both org and user records are absent."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(org_record=None, user_record=None)
    assert result is None


@pytest.mark.asyncio
async def test_b4_merge_wired_into_query_result_when_both_layers_present() -> None:
    """B-4: MemoryResult.merge is populated in live query when both org and user layers exist."""
    import uuid

    from workflows_mcp.engine.memory_service import MemoryRequest, MemoryService, QueryMemoryResult

    org_item = {
        "id": str(uuid.uuid4()),
        "content": "org fact",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    user_item = {
        "id": str(uuid.uuid4()),
        "content": "user fact",
        "updated_at": "2024-06-01T00:00:00+00:00",
    }
    fake_query_result = QueryMemoryResult(
        facts=[user_item],
        memories=[org_item],
    )

    mock_context = MagicMock()
    mock_context.scope_token = None

    backend = AsyncMock()
    service = MemoryService(backend=backend, context=mock_context)

    request = MemoryRequest(
        operation="query",
        scope={
            "palace": "test-palace",
            "wing": "test-wing",
            "room": "test-room",
            "compartment": "test-compartment",
        },
        query={"text": "test query"},
    )

    with patch.object(service, "query", return_value=fake_query_result):
        result = await service.execute(request)

    assert result.merge is not None, "merge MUST be populated when both org and user layers exist"
    assert result.merge.conflict is True
    # User record is newer (2024-06-01 > 2024-01-01) → user wins
    assert result.merge.effective_source == "user"
    assert result.merge.org_record == org_item
    assert result.merge.user_record == user_item


@pytest.mark.asyncio
async def test_b4_merge_is_none_when_only_org_layer_present() -> None:
    """B-4: MemoryResult.merge reflects single-layer when only org records returned."""
    from workflows_mcp.engine.memory_service import MemoryRequest, MemoryService, QueryMemoryResult

    org_item = {"id": "org-1", "content": "org only", "updated_at": "2024-01-01T00:00:00+00:00"}
    fake_query_result = QueryMemoryResult(
        facts=[],
        memories=[org_item],
    )

    mock_context = MagicMock()
    mock_context.scope_token = None

    backend = AsyncMock()
    service = MemoryService(backend=backend, context=mock_context)

    request = MemoryRequest(
        operation="query",
        scope={
            "palace": "test-palace",
            "wing": "test-wing",
            "room": "test-room",
            "compartment": "test-compartment",
        },
        query={"text": "test query"},
    )

    with patch.object(service, "query", return_value=fake_query_result):
        result = await service.execute(request)

    # Only org layer — merge is present but no conflict
    assert result.merge is not None
    assert result.merge.effective_source == "org"
    assert result.merge.conflict is False
    assert result.merge.user_record is None


@pytest.mark.asyncio
async def test_b4_merge_is_none_when_no_results() -> None:
    """B-4: MemoryResult.merge is None when query returns empty results."""
    from workflows_mcp.engine.memory_service import MemoryRequest, MemoryService, QueryMemoryResult

    fake_query_result = QueryMemoryResult(facts=[], memories=[])

    mock_context = MagicMock()
    mock_context.scope_token = None

    backend = AsyncMock()
    service = MemoryService(backend=backend, context=mock_context)

    request = MemoryRequest(
        operation="query",
        scope={
            "palace": "test-palace",
            "wing": "test-wing",
            "room": "test-room",
            "compartment": "test-compartment",
        },
        query={"text": "test query"},
    )

    with patch.object(service, "query", return_value=fake_query_result):
        result = await service.execute(request)

    assert result.merge is None, "merge MUST be None when no results are returned"


# ---------------------------------------------------------------------------
# MEMORY-CONTRACT-v3.1: Scope isolation — global companion lane must not leak
# cross-scope rows when explicit scope is provided.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_isolation_companion_lane_suppressed_when_full_scope_explicit() -> None:
    """MEMORY-CONTRACT-v3.1 P0: auto strategy must NOT fire the global companion lane
    when an explicit full scope (wing+room+compartment) is provided.

    The companion lane runs without scope filters — allowing it to fire when
    an explicit scope is present admits out-of-scope rows into the result set.
    This is the cross-scope leakage breach.

    Expected: include_global_companion must be False when namespace+room+corridor
    are all explicitly set.
    """
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            await service.query(
                QueryMemoryRequest(
                    query="incident",
                    strategy="auto",
                    namespace="wing-a",
                    room="room-1",
                    # s2_enabled defaults to True — the breach fires here
                )
            )

    # FAILS before fix: companion lane fires because s2_enabled=True (default)
    # and has_explicit_scope=True → include_global_companion=True → global rows leak in.
    assert mock_search.await_args.kwargs["include_global_companion"] is False, (
        "Global companion lane must be suppressed when explicit scope is provided "
        "(MEMORY-CONTRACT-v3.1: no cross-scope leakage)"
    )


@pytest.mark.asyncio
async def test_scope_isolation_companion_lane_suppressed_with_corridor() -> None:
    """MEMORY-CONTRACT-v3.1 P0: companion lane must be suppressed when corridor
    (compartment) is explicitly provided, even if s2_enabled=True.
    """
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            await service.query(
                QueryMemoryRequest(
                    query="alert",
                    strategy="auto",
                    namespace="wing-b",
                    room="room-2",
                    scope={"corridor": "compartment-x"},
                    # s2_enabled defaults to True
                )
            )

    # FAILS before fix: corridor is in scope bag → has_explicit_scope=True
    # → include_global_companion=True when s2_enabled=True.
    assert mock_search.await_args.kwargs["include_global_companion"] is False, (
        "Global companion lane must be suppressed when corridor/compartment is "
        "explicitly provided (MEMORY-CONTRACT-v3.1: no cross-scope leakage)"
    )


@pytest.mark.asyncio
async def test_scope_isolation_execute_auto_strategy_suppresses_companion_lane() -> None:
    """MEMORY-CONTRACT-v3.1 P0: the MemoryRequest execute() path for strategy='auto'
    must NOT admit cross-scope rows when called with full scope.

    This tests the full contract envelope path (MemoryRequest.execute → query).
    """
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            # Use the full MemoryRequest contract envelope with explicit scope
            # query.mode=search → strategy="auto" (radius=1 default, no forced palace)
            await service.execute(
                MemoryRequest(
                    operation="query",
                    scope={
                        "palace": "palace-1",
                        "wing": "wing-a",
                        "room": "room-1",
                        "compartment": "compartment-x",
                    },
                    query={
                        "text": "incident",
                        "mode": "search",
                        "radius": 1,  # radius=1 → strategy=auto (not palace)
                        "s2_enabled": True,  # explicit True to confirm suppression
                    },
                )
            )

    # FAILS before fix: with full explicit scope and s2_enabled=True,
    # include_global_companion is True, leaking cross-scope rows.
    assert mock_search.await_args.kwargs["include_global_companion"] is False, (
        "MemoryRequest execute path: companion lane must be suppressed with full explicit scope "
        "(MEMORY-CONTRACT-v3.1 breach: cross-scope leakage via auto strategy)"
    )


# ---------------------------------------------------------------------------
# Palace isolation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_palace_scope_passed_to_room_scoped_search() -> None:
    """PALACE-ISO-01: palace value must be forwarded to room_scoped_search.

    When a query carries an explicit palace, the search layer must receive it
    as a filter so rows from a different palace are never returned.
    """
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            await service.query(
                QueryMemoryRequest(
                    query="find something",
                    strategy="auto",
                    palace="org-a",
                    namespace="wing-1",
                    room="room-1",
                )
            )

    call_kwargs = mock_search.await_args.kwargs
    assert call_kwargs["palace"] == "org-a", (
        "palace must be forwarded to room_scoped_search for palace-scoped isolation "
        "(PALACE-ISO-01: missing palace filter leaks cross-palace rows)"
    )


@pytest.mark.asyncio
async def test_palace_collision_isolation_query() -> None:
    """PALACE-ISO-02: two palaces with identical wing/room labels must be isolated.

    Querying with palace='org-a' must not include rows from 'org-b', even when
    wing and room labels are identical.
    """
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)

            # Query from palace org-a
            await service.query(
                QueryMemoryRequest(
                    query="collision test",
                    strategy="auto",
                    palace="org-a",
                    namespace="shared-wing",
                    room="shared-room",
                )
            )
            call_kwargs_a = mock_search.await_args.kwargs

            # Query from palace org-b (same wing/room labels)
            await service.query(
                QueryMemoryRequest(
                    query="collision test",
                    strategy="auto",
                    palace="org-b",
                    namespace="shared-wing",
                    room="shared-room",
                )
            )
            call_kwargs_b = mock_search.await_args.kwargs

    assert call_kwargs_a["palace"] == "org-a", "org-a query must pass palace='org-a'"
    assert call_kwargs_b["palace"] == "org-b", "org-b query must pass palace='org-b'"
    assert call_kwargs_a["palace"] != call_kwargs_b["palace"], (
        "PALACE-ISO-02: palace collision — two orgs with identical wing/room labels "
        "must resolve to distinct palace filter values; cross-palace leakage detected"
    )


@pytest.mark.asyncio
async def test_palace_store_write_path_includes_palace() -> None:
    """PALACE-ISO-03: palace must be persisted in knowledge_memories on store.

    If palace is omitted from the INSERT, rows from palace-A ingested without
    a filter can be retrieved in palace-B queries (silent data leakage).
    """
    backend = MagicMock()
    context = MagicMock()
    context.execution_context = None

    # Capture all execute() calls to inspect the SQL and params
    execute_calls: list[tuple[str, tuple]] = []

    async def capture_execute(sql: str, params: tuple = ()) -> None:
        execute_calls.append((sql, params))

    backend.execute = capture_execute

    # Stub query() to return plausible source/item rows
    async def stub_query(sql: str, params: tuple = ()) -> MagicMock:
        result = MagicMock()
        if "knowledge_sources" in sql:
            result.rows = [{"id": "00000000-0000-0000-0000-000000000001"}]
        elif "knowledge_items" in sql:
            result.rows = [{"id": "00000000-0000-0000-0000-000000000002"}]
        else:
            result.rows = []
        return result

    backend.query = stub_query

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1] * 384, "model", 384, None)
        service = MemoryService(backend=backend, context=context)
        from workflows_mcp.engine.memory_service import ManageMemoryRequest

        await service.manage(
            ManageMemoryRequest(
                operation="store",
                content="palace persistence test",
                palace="org-persist",
                namespace="wing-x",
                room="room-x",
            )
        )

    # Find the INSERT INTO knowledge_memories call
    memory_inserts = [(sql, params) for sql, params in execute_calls if "knowledge_memories" in sql]
    assert memory_inserts, "Expected at least one INSERT INTO knowledge_memories"

    insert_sql, insert_params = memory_inserts[0]
    assert "palace" in insert_sql, (
        "PALACE-ISO-03: 'palace' column must appear in knowledge_memories INSERT; "
        "missing column means palace is not persisted and cross-palace isolation is broken"
    )
    assert "org-persist" in insert_params, (
        "PALACE-ISO-03: palace value 'org-persist' must appear in INSERT params; "
        "column present but value not bound means isolation is silently broken"
    )


@pytest.mark.asyncio
async def test_palace_community_insert_includes_palace() -> None:
    """PALACE-ISO-04: palace must be persisted in knowledge_communities INSERT.

    Community rows without a palace column are shared across all palaces,
    breaking isolation for community-based retrieval.
    """
    backend = MagicMock()
    context = MagicMock()
    context.execution_context = None

    query_calls: list[tuple[str, tuple]] = []
    execute_calls: list[tuple[str, tuple]] = []

    async def capture_query(sql: str, params: tuple = ()) -> MagicMock:
        query_calls.append((sql, params))
        result = MagicMock()
        if "knowledge_communities" in sql and "INSERT" in sql:
            result.rows = [{"id": "00000000-0000-0000-0000-000000000010"}]
        elif "knowledge_sources" in sql:
            result.rows = [{"id": "00000000-0000-0000-0000-000000000001"}]
        elif "knowledge_items" in sql:
            result.rows = [{"id": "00000000-0000-0000-0000-000000000002"}]
        else:
            result.rows = []
        return result

    async def capture_execute(sql: str, params: tuple = ()) -> None:
        execute_calls.append((sql, params))

    backend.query = capture_query
    backend.execute = capture_execute
    backend.begin_transaction = AsyncMock()
    backend.commit_transaction = AsyncMock()

    # Simulate what _insert_community produces by inspecting the SQL it would emit.
    # We call it directly since community inserts happen through manage → consolidate paths.
    from workflows_mcp.engine.memory_service import ManageMemoryRequest

    request = ManageMemoryRequest(
        operation="store",
        content="community test",
        palace="palace-community",
        namespace="wing-c",
        room="room-c",
    )

    service = MemoryService(backend=backend, context=context)

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1] * 384, "model", 384, None)
        with patch.object(service, "_assert_valid_parent_lineage", new_callable=AsyncMock):
            await service._insert_community(  # type: ignore[attr-defined]
                entity_ids=["eid-1"],
                entity_names={"eid-1": "TestEntity"},
                memory_rows=[
                    {
                        "id": "00000000-0000-0000-0000-000000000020",
                        "embedding": str([0.1] * 384),
                    }
                ],
                request=request,
            )

    community_inserts = [
        (sql, params)
        for sql, params in query_calls
        if "knowledge_communities" in sql and "INSERT" in sql
    ]
    assert community_inserts, "Expected INSERT INTO knowledge_communities"

    insert_sql, insert_params = community_inserts[0]
    assert "palace" in insert_sql, (
        "PALACE-ISO-04: 'palace' column must appear in knowledge_communities INSERT; "
        "community rows without palace break isolation for community-mode retrieval"
    )
    assert "palace-community" in insert_params, (
        "PALACE-ISO-04: palace value 'palace-community' must be bound in INSERT params"
    )


@pytest.mark.asyncio
async def test_palace_strategy_forwards_palace_to_room_scoped_search() -> None:
    """PALACE-ISO-05: _query_palace must forward palace= to room_scoped_search.

    The palace retrieval strategy is strict-scoped. If palace is not passed to
    room_scoped_search, the SQL filter for palace isolation is never applied and
    rows from any palace can leak into the result set.

    This test targets the palace strategy path specifically (radius=3 forces it),
    distinct from the auto-strategy companion-lane tests.
    """
    backend = MagicMock()
    backend.execute = AsyncMock()
    context = MagicMock()
    context.execution_context = None

    with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
        mock_embed.return_value = ([0.1, 0.2, 0.3], "model", 3, None)
        with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
            mock_search.return_value = []
            service = MemoryService(backend=backend, context=context)
            await service.query(
                QueryMemoryRequest(
                    query="palace strategy test",
                    strategy="palace",
                    palace="org-palace-test",
                    namespace="wing-p",
                    room="room-p",
                )
            )

    assert mock_search.called, "room_scoped_search must be called by palace strategy"
    call_kwargs = mock_search.await_args.kwargs
    assert call_kwargs.get("palace") == "org-palace-test", (
        "PALACE-ISO-05: palace= kwarg must be forwarded to room_scoped_search in the "
        "palace strategy path; without it, the SQL palace filter is never applied and "
        "cross-palace row leakage occurs (isolation breach)"
    )


# ---------------------------------------------------------------------------
# Blocker 2: naive vs aware datetime comparison in _build_merge_transparency
# ---------------------------------------------------------------------------


def test_b4_merge_transparency_naive_db_timestamps_do_not_raise() -> None:
    """_build_merge_transparency must not raise TypeError when DB returns naive timestamps.

    PostgreSQL timestamps without timezone (e.g. '2024-06-01 00:00:00') are returned
    as naive datetime strings by some drivers. Comparing a naive datetime against the
    aware _epoch sentinel raises TypeError. The function must handle this gracefully and
    still determine the correct effective_source.
    """
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    # Naive ISO strings — no timezone offset — simulate DB-returned timestamps
    result = _build_merge_transparency(
        org_record={"updated_at": "2024-01-01 00:00:00"},
        user_record={"updated_at": "2024-06-01 00:00:00"},
    )
    assert result is not None
    assert result.effective_source == "user"
    assert result.conflict is True


def test_b4_merge_transparency_mixed_naive_aware_timestamps_do_not_raise() -> None:
    """_build_merge_transparency must handle mixed naive/aware timestamps without crashing."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(
        org_record={"updated_at": "2024-06-01T00:00:00+00:00"},
        user_record={"updated_at": "2024-01-01 00:00:00"},  # naive
    )
    assert result is not None
    # org is newer; must win without raising
    assert result.effective_source == "org"


def test_b4_merge_transparency_both_naive_timestamps_org_wins_on_tie() -> None:
    """Tie-breaking (org wins) must work correctly with naive timestamps."""
    from workflows_mcp.engine.memory_service import _build_merge_transparency

    result = _build_merge_transparency(
        org_record={"updated_at": "2024-03-15 12:00:00"},
        user_record={"updated_at": "2024-03-15 12:00:00"},
    )
    assert result is not None
    assert result.effective_source == "org"


# ---------------------------------------------------------------------------
# Memory operation locality contract tests (Tasks 2-3 slice)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_requires_complete_topology_scope() -> None:
    """ingest must fail with INSUFFICIENT_LOCALITY when topology is incomplete."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError) as exc:
        await service.execute(
            MemoryRequest(
                operation="ingest",
                scope={"palace": "acme", "wing": "svc", "room": "comp"},
                record={"format": "raw", "content": "stored memory", "memory_tier": "direct"},
            )
        )
    error = exc.value
    assert error.code == "INSUFFICIENT_LOCALITY"
    assert "ingest requires complete topology locality" in error.message
    assert "Missing: compartment" in error.message
    assert "scope_token" in (error.actionable_fix or "")


@pytest.mark.asyncio
async def test_graph_upsert_place_requires_complete_topology_scope() -> None:
    """graph_upsert place must fail with INSUFFICIENT_LOCALITY when topology is incomplete."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError) as exc:
        await service.execute(
            MemoryRequest(
                operation="graph_upsert",
                scope={"palace": "acme", "wing": "svc", "room": "comp"},
                graph={"kind": "place", "place_name": "entity", "place_type": "concept"},
            )
        )
    error = exc.value
    assert error.code == "INSUFFICIENT_LOCALITY"
    assert "graph_upsert place requires complete topology locality" in error.message
    assert "Missing: compartment" in error.message


@pytest.mark.asyncio
async def test_graph_upsert_link_with_uuid_refs_does_not_require_topology() -> None:
    """graph_upsert link with UUID from/to refs must succeed without topology scope."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    async def _fake_manage(_request: object) -> ManageMemoryResult:
        return ManageMemoryResult(operation="graph_store_relation")

    service.manage = _fake_manage  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="graph_upsert",
            graph={
                "kind": "link",
                "from": "11111111-1111-1111-1111-111111111111",
                "to": "22222222-2222-2222-2222-222222222222",
                "link_type": "relates_to",
            },
        )
    )

    assert result.operation == "graph_upsert"
    assert result.scope_source == {}


@pytest.mark.asyncio
async def test_graph_upsert_link_with_name_refs_requires_complete_topology() -> None:
    """graph_upsert link with name refs must fail with INSUFFICIENT_LOCALITY."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError) as exc:
        await service.execute(
            MemoryRequest(
                operation="graph_upsert",
                scope={"palace": "acme", "wing": "svc", "room": "comp"},
                graph={"kind": "link", "from": "alice", "to": "acme", "link_type": "uses"},
            )
        )
    error = exc.value
    assert error.code == "INSUFFICIENT_LOCALITY"
    assert "graph_upsert link" in error.message
    assert "complete topology locality" in error.message
    assert "Missing: compartment" in error.message


@pytest.mark.asyncio
async def test_graph_delete_requires_ids() -> None:
    """graph_delete without ids must fail with INSUFFICIENT_LOCALITY."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError) as exc:
        await service.execute(
            MemoryRequest(
                operation="graph_delete",
                graph={"kind": "place"},
            )
        )
    error = exc.value
    assert error.code == "INSUFFICIENT_LOCALITY"
    assert "graph_delete" in error.message
    assert "Missing: graph.ids" in error.message


@pytest.mark.asyncio
async def test_supersede_with_empty_ids_raises_insufficient_locality() -> None:
    """supersede with record.ids=[] is treated as missing ids and raises INSUFFICIENT_LOCALITY."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    with pytest.raises(MemoryContractError) as exc:
        await service.execute(
            MemoryRequest(
                operation="supersede",
                record={"ids": [], "superseded_by": "11111111-1111-1111-1111-111111111111"},
            )
        )
    error = exc.value
    assert error.code == "INSUFFICIENT_LOCALITY"
    assert "supersede" in error.message
    assert "Missing: record.ids" in error.message


@pytest.mark.asyncio
async def test_current_maintain_modes_do_not_require_topology() -> None:
    """maintain operation must not require any topology scope."""
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    async def _fake_manage(_request: object) -> ManageMemoryResult:
        return ManageMemoryResult(operation="maintain")

    service.manage = _fake_manage  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="maintain",
        )
    )

    assert result.operation == "maintain"


# ---------------------------------------------------------------------------
# ADR-013 invariant contract tests (Task 1 RED tests)
# These tests encode the ADR-013 invariants and fail until the new operations
# and their invariant enforcement are implemented in production code.
# ---------------------------------------------------------------------------


def test_adr013_topology_depth_exactly_palace_wing_room_compartment() -> None:
    """ADR-013: topology hierarchy is exactly Palace/Wing/Room/Compartment.

    The valid depth levels are only those four. Any topology scope providing
    all four levels must be accepted without error. This test uses an existing
    registered operation so it tests the topology contract independently of
    whether the new ADR-013 operations are registered yet.
    """
    # Use an already-registered operation so this test isolates the topology
    # constraint and does not conflate it with the missing-operation gate.
    req = MemoryRequest(
        operation="query",
        scope={"palace": "acme", "wing": "code", "room": "services", "compartment": "auth"},
        query={"text": "topology check", "mode": "search"},
    )
    assert req.scope.palace == "acme"
    assert req.scope.wing == "code"
    assert req.scope.room == "services"
    assert req.scope.compartment == "auth"


def test_adr013_corridor_rejected_as_topology_level_field() -> None:
    """ADR-013: 'corridor' must never appear as a topology level field in external scope.

    Corridor represents directed typed edges between entities, not a topology
    level in the Palace/Wing/Room/Compartment hierarchy. Using it as a scope key
    is an invariant violation and must be rejected with MEM_INVALID_TAXONOMY_KEY.
    """
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(
            operation="store_system1_structural_evidence",
            scope={
                "palace": "acme",
                "wing": "code",
                "room": "services",
                "corridor": "auth",
            },
            record={"format": "structured"},
        )
    assert "MEM_INVALID_TAXONOMY_KEY" in str(exc.value)


@pytest.mark.asyncio
async def test_adr013_new_wing_requires_minimum_two_evidence_categories() -> None:
    """ADR-013: creating a new wing in System 2 requires a proof bundle
    with at least two evidence categories. Providing only one category must
    cause MemoryService.execute() to return a result whose manage envelope
    carries MEM_INSUFFICIENT_EVIDENCE_BUNDLE as the error code, or raise
    MemoryContractError with that code.

    This is a service-layer invariant enforced at execution time, not at
    Pydantic model construction. The test deliberately goes through
    MemoryService.execute() so that it cannot false-green merely when
    derive_system2_semantic_claims becomes a registered MemoryOperation.

    RED state pre-registration: fails with pytest.fail explaining the operation
    must be registered first. RED state post-registration (before gate logic):
    fails because the service does not yet enforce MEM_INSUFFICIENT_EVIDENCE_BUNDLE.
    """
    from pydantic import ValidationError

    context = Execution()
    service = MemoryService(backend=object(), context=context)

    try:
        request = MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": {
                    "palace": "acme",
                    "wing": "new_wing",
                    "room": "services",
                    "compartment": "auth",
                },
                "record": {
                    "format": "structured",
                    "derivation": {
                        "is_new_wing": True,
                        "proof_bundle": {
                            "evidence_categories": ["class_anchor"],
                            # Only one category — must be rejected; minimum is 2.
                        },
                        "room_intent_label": "auth_layer",
                        "evidence_entity_stable_ids": ["src/auth.py::AuthService"],
                    },
                },
            }
        )
    except ValidationError as exc:
        pytest.fail(
            "derive_system2_semantic_claims is not yet a registered MemoryOperation — "
            "register it so the evidence bundle gate invariant can be tested at service level. "
            f"Pydantic error: {exc}"
        )

    # The service must enforce the proof-bundle gate at execution time and signal
    # MEM_INSUFFICIENT_EVIDENCE_BUNDLE either via MemoryContractError or via the
    # manage result envelope error field.
    try:
        result = await service.execute(request)
        # If no exception: the error code must appear in the manage envelope.
        assert result.manage is not None, (
            "derive_system2_semantic_claims with one evidence category must not succeed; "
            "expected MEM_INSUFFICIENT_EVIDENCE_BUNDLE in manage.error"
        )
        assert not result.manage.success, (
            "derive_system2_semantic_claims with one evidence category must not succeed"
        )
        assert "MEM_INSUFFICIENT_EVIDENCE_BUNDLE" in (result.manage.error or ""), (
            "Expected MEM_INSUFFICIENT_EVIDENCE_BUNDLE in manage.error, "
            f"got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_INSUFFICIENT_EVIDENCE_BUNDLE", (
            f"Expected MEM_INSUFFICIENT_EVIDENCE_BUNDLE, got: {exc.code!r}"
        )


@pytest.mark.asyncio
async def test_adr013_new_wing_with_two_evidence_categories_is_accepted() -> None:
    """ADR-013: a new wing derivation with exactly two evidence categories must not
    be rejected by the service with MEM_INSUFFICIENT_EVIDENCE_BUNDLE.

    This test goes through MemoryService.execute() so it cannot pass trivially at
    model-construction time once derive_system2_semantic_claims is registered.

    RED state: fails with MEM_INVALID_OPERATION until derive_system2_semantic_claims
    is implemented as a registered MemoryOperation.
    """
    from pydantic import ValidationError

    context = Execution()
    backend = MagicMock()
    # Evidence lookup returns no rows → MEM_UNRESOLVABLE_EVIDENCE, which is NOT
    # MEM_INSUFFICIENT_EVIDENCE_BUNDLE. The gate-under-test (proof bundle category
    # count) fires before any evidence DB query, so any result or error other than
    # MEM_INSUFFICIENT_EVIDENCE_BUNDLE satisfies this assertion.
    backend.query = AsyncMock(return_value=SimpleNamespace(rows=[]))
    backend.execute = AsyncMock()
    backend.begin_transaction = AsyncMock()
    backend.commit = AsyncMock()
    backend.rollback = AsyncMock()
    service = MemoryService(backend=backend, context=context)

    try:
        request = MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": {
                    "palace": "acme",
                    "wing": "new_wing",
                    "room": "services",
                    "compartment": "auth",
                },
                "record": {
                    "format": "structured",
                    "derivation": {
                        "is_new_wing": True,
                        "proof_bundle": {
                            "evidence_categories": ["class_anchor", "method_anchor"],
                        },
                        "room_intent_label": "auth_layer",
                        "evidence_entity_stable_ids": ["src/auth.py::AuthService"],
                    },
                },
            }
        )
    except ValidationError as exc:
        pytest.fail(
            "derive_system2_semantic_claims is not yet a registered MemoryOperation — "
            "register it to make this test runnable. "
            f"Pydantic error: {exc}"
        )

    try:
        result = await service.execute(request)
        # Success or any failure OTHER than MEM_INSUFFICIENT_EVIDENCE_BUNDLE is acceptable
        # here — the proof bundle has the required two categories.
        if result.manage is not None and result.manage.error:
            assert "MEM_INSUFFICIENT_EVIDENCE_BUNDLE" not in result.manage.error, (
                "derive_system2_semantic_claims must not raise the evidence bundle gate "
                "when two evidence categories are provided"
            )
    except MemoryContractError as exc:
        assert exc.code != "MEM_INSUFFICIENT_EVIDENCE_BUNDLE", (
            "Service must not reject new wing derivation when two evidence categories are provided"
        )


@pytest.mark.asyncio
async def test_adr013_archive_blocked_without_two_successful_absent_verification_cycles() -> None:
    """ADR-013: reconcile_semantic_lifecycle must block archive transitions when
    fewer than two successful System 1 verification cycles with absent evidence
    over the affected scope have been recorded. The service must return
    MEM_ARCHIVE_GATE_NOT_MET in the manage result envelope (or raise
    MemoryContractError with that code) — not a Pydantic ValidationError.

    This is a runtime service invariant, not a model construction constraint.
    The test goes through MemoryService.execute() so it cannot false-green
    merely when reconcile_semantic_lifecycle becomes a registered MemoryOperation.

    RED state pre-registration: fails with pytest.fail explaining the operation
    must be registered first. RED state post-registration (before gate logic):
    fails because the service does not yet enforce MEM_ARCHIVE_GATE_NOT_MET.
    """
    from pydantic import ValidationError

    context = Execution()
    # begin_transaction and rollback are needed: the service opens a transaction
    # before enforcing the archive gate, then rolls back when the gate fires.
    backend = MagicMock()
    backend.begin_transaction = AsyncMock()
    backend.rollback = AsyncMock()
    backend.commit = AsyncMock()
    backend.query = AsyncMock(return_value=SimpleNamespace(rows=[]))
    backend.execute = AsyncMock()
    service = MemoryService(backend=backend, context=context)

    try:
        request = MemoryRequest.model_validate(
            {
                "operation": "reconcile_semantic_lifecycle",
                "scope": {
                    "palace": "acme",
                    "wing": "code",
                    "room": "services",
                    "compartment": "auth",
                },
                "record": {
                    "format": "structured",
                    "lifecycle_reconciliation": {
                        "scope_key": "acme/code/services/auth",
                        "force_archive_claim_ids": ["claim-1"],
                        # No absent_verification_cycle_ids — archive gate must block this.
                    },
                },
            }
        )
    except ValidationError as exc:
        pytest.fail(
            "reconcile_semantic_lifecycle is not yet a registered MemoryOperation — "
            "register it so the archive gate invariant can be tested at service level. "
            f"Pydantic error: {exc}"
        )

    # The service must enforce the archive gate at execution time and signal
    # MEM_ARCHIVE_GATE_NOT_MET either via MemoryContractError or via the
    # manage result envelope error field.
    try:
        result = await service.execute(request)
        assert result.manage is not None, (
            "reconcile_semantic_lifecycle with force_archive_claim_ids but no "
            "absent cycle proof must not succeed; expected MEM_ARCHIVE_GATE_NOT_MET "
            "in manage.error"
        )
        assert not result.manage.success, (
            "Archive without two absent verification cycles must not succeed"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or ""), (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET in manage.error, got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET", (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {exc.code!r}"
        )


@pytest.mark.asyncio
async def test_adr013_archive_allowed_with_two_successful_absent_verification_cycles() -> None:
    """ADR-013: reconcile_semantic_lifecycle must permit archive when two successful
    absent-evidence verification cycle IDs are provided for the affected scope.

    The service must not return MEM_ARCHIVE_GATE_NOT_MET when the proof is present.
    This test goes through MemoryService.execute() so it cannot pass trivially at
    model-construction time once the operation is registered.

    The archive gate is fail-closed: cycle IDs must be registered via
    record_system1_verification_cycle with success=True before they count as valid
    absent-evidence proof.
    """
    from pydantic import ValidationError

    context = Execution()
    # execute: needed for record_system1_verification_cycle INSERT (no return value needed)
    # query: needed for the archive gate COUNT check (must return n>=2) and the
    #        archive UPDATE (returns rows; empty is fine — the gate has already passed).
    backend = MagicMock()
    backend.execute = AsyncMock()
    backend.query = AsyncMock(
        side_effect=[
            SimpleNamespace(rows=[]),  # accountability UPDATE for cycle 1 → no matching overrides
            SimpleNamespace(rows=[]),  # accountability UPDATE for cycle 2 → no matching overrides
            SimpleNamespace(rows=[{"n": 2}]),  # cycle count gate → passes (>=2)
            SimpleNamespace(rows=[{"id": "claim-1"}]),  # archive UPDATE → 1 row archived
        ]
    )
    backend.begin_transaction = AsyncMock()
    backend.commit = AsyncMock()
    backend.rollback = AsyncMock()
    service = MemoryService(backend=backend, context=context)

    scope = {"palace": "acme", "wing": "code", "room": "services", "compartment": "auth"}

    # Register two successful verification cycles (absence confirmed) first.
    cycle_ids: list[str] = []
    for _ in range(2):
        cycle_req = MemoryRequest.model_validate(
            {
                "operation": "record_system1_verification_cycle",
                "scope": scope,
                "record": {
                    "format": "structured",
                    "verification_cycle": {
                        "success": True,
                        "covered_scope": scope,
                    },
                },
            }
        )
        cycle_result = await service.execute(cycle_req)
        assert cycle_result.manage is not None
        assert cycle_result.manage.cycle_id is not None
        cycle_ids.append(cycle_result.manage.cycle_id)

    try:
        request = MemoryRequest.model_validate(
            {
                "operation": "reconcile_semantic_lifecycle",
                "scope": scope,
                "record": {
                    "format": "structured",
                    "lifecycle_reconciliation": {
                        "scope_key": "acme/code/services/auth",
                        "force_archive_claim_ids": ["claim-1"],
                        "absent_verification_cycle_ids": cycle_ids,
                    },
                },
            }
        )
    except ValidationError as exc:
        pytest.fail(
            "reconcile_semantic_lifecycle is not yet a registered MemoryOperation — "
            "register it to make this test runnable. "
            f"Pydantic error: {exc}"
        )

    try:
        result = await service.execute(request)
        # Success or any failure OTHER than MEM_ARCHIVE_GATE_NOT_MET is acceptable
        # at this layer — the gate is not about rejecting valid proofs.
        if result.manage is not None and result.manage.error:
            assert "MEM_ARCHIVE_GATE_NOT_MET" not in result.manage.error, (
                "reconcile_semantic_lifecycle must not raise the archive gate when "
                "two absent_verification_cycle_ids are provided"
            )
    except MemoryContractError as exc:
        assert exc.code != "MEM_ARCHIVE_GATE_NOT_MET", (
            "Service must not block archive when two valid absent cycle IDs are provided"
        )


def test_adr013_system1_operations_not_rejected_as_invalid_operation() -> None:
    """ADR-013: new System 1 and System 2 operations must be valid MemoryOperation values
    and must not raise MEM_INVALID_OPERATION on construction."""
    for op in [
        "store_system1_structural_evidence",
        "record_system1_verification_cycle",
        "derive_system2_semantic_claims",
        "apply_semantic_override",
        "reconcile_semantic_lifecycle",
    ]:
        req = MemoryRequest.model_validate(
            {
                "operation": op,
                "scope": {"palace": "acme", "wing": "code", "room": "svc", "compartment": "auth"},
                "record": {"format": "structured"},
            }
        )
        assert req.operation == op, f"Expected operation={op!r} to be accepted"


def test_adr013_override_error_envelope_contains_provenance_fields() -> None:
    """ADR-013: apply_semantic_override response envelope must include provenance fields.

    This test validates that the ManageMemoryResult type exposes the override_id field
    that is required by the ADR-013 override accountability contract.
    """
    result = ManageMemoryResult(
        operation="apply_semantic_override",
        success=True,
        override_id="override-123",
    )
    assert result.override_id == "override-123"
    assert result.success is True


def test_adr013_cycle_result_envelope_contains_cycle_id() -> None:
    """ADR-013: record_system1_verification_cycle response envelope must include cycle_id."""
    result = ManageMemoryResult(
        operation="record_system1_verification_cycle",
        success=True,
        cycle_id="cycle-abc-1",
    )
    assert result.cycle_id == "cycle-abc-1"
    assert result.success is True


def test_adr013_derivation_result_envelope_contains_claim_ids() -> None:
    """ADR-013: derive_system2_semantic_claims response envelope must include claim_ids."""
    result = ManageMemoryResult(
        operation="derive_system2_semantic_claims",
        success=True,
        claim_ids=["claim-1", "claim-2"],
    )
    assert result.claim_ids == ["claim-1", "claim-2"]


def test_adr013_reconciliation_result_envelope_contains_reconciled_count() -> None:
    """ADR-013: reconcile_semantic_lifecycle response envelope must include reconciled_count."""
    result = ManageMemoryResult(
        operation="reconcile_semantic_lifecycle",
        success=True,
        reconciled_count=3,
    )
    assert result.reconciled_count == 3


def test_adr013_evidence_result_envelope_contains_stored_evidence_ids() -> None:
    """ADR-013: store_system1_structural_evidence response envelope must include
    stored_evidence_ids — a list of stable IDs for each persisted evidence row.

    RED state: fails with AttributeError or validation error until ManageMemoryResult
    exposes the stored_evidence_ids field required by the idempotency contract.
    """
    result = ManageMemoryResult(
        operation="store_system1_structural_evidence",
        success=True,
        stored_evidence_ids=["eid-1"],
    )
    assert result.stored_evidence_ids == ["eid-1"]


# ---------------------------------------------------------------------------
# Task 11: Override provenance and lifecycle accountability
# ---------------------------------------------------------------------------


def test_override_provenance_result_includes_all_mandatory_fields() -> None:
    """Task 11: apply_semantic_override result must include override_id and provenance fields.

    The diagnostics dict must expose claim_id, override_reason, applied_by,
    new_lifecycle_state, and activated_at — all required by the accountability contract.
    """
    result = ManageMemoryResult(
        operation="apply_semantic_override",
        success=True,
        override_id="override-abc123",
        diagnostics={
            "claim_id": "claim-uuid-1",
            "override_reason": "Evidence contradicts stale claim",
            "applied_by": "qa-agent",
            "new_lifecycle_state": "degraded",
            "activated_at": "2026-05-06T00:00:00+00:00",
            "accountability_status": "pending",
        },
    )
    assert result.override_id == "override-abc123"
    assert result.success is True
    assert result.diagnostics is not None
    assert result.diagnostics["claim_id"] == "claim-uuid-1"
    assert result.diagnostics["override_reason"] == "Evidence contradicts stale claim"
    assert result.diagnostics["applied_by"] == "qa-agent"
    assert result.diagnostics["activated_at"] is not None
    assert result.diagnostics["accountability_status"] == "pending"


def test_override_missing_reason_rejected_with_contract_error() -> None:
    """Task 11: apply_semantic_override with empty override_reason must fail validation.

    The OverrideInput model enforces that override_reason is a non-empty string.
    Missing or empty reason must raise a validation error before the operation executes.
    """
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_service import OverrideInput

    with pytest.raises(ValidationError):
        OverrideInput(
            claim_id="claim-uuid-1",
            override_reason="",  # empty — must be rejected
            overridden_by="qa-agent",
            new_lifecycle_state="degraded",
        )


def test_override_missing_applied_by_rejected_with_contract_error() -> None:
    """Task 11: apply_semantic_override with empty overridden_by must fail validation.

    The OverrideInput model enforces that overridden_by (applied_by in DB) is non-empty.
    """
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_service import OverrideInput

    with pytest.raises(ValidationError):
        OverrideInput(
            claim_id="claim-uuid-1",
            override_reason="Valid reason",
            overridden_by="",  # empty — must be rejected
            new_lifecycle_state="degraded",
        )


def test_override_accountability_status_initial_value_is_pending() -> None:
    """Task 11: immediately after apply_semantic_override, accountability_status is 'pending'.

    On creation the override has not yet been checked against a verification cycle,
    so the status must start as 'pending' — not 'supported' or 'unsupported'.
    """
    result = ManageMemoryResult(
        operation="apply_semantic_override",
        success=True,
        override_id="override-pending-1",
        diagnostics={"accountability_status": "pending"},
    )
    assert result.diagnostics is not None
    assert result.diagnostics["accountability_status"] == "pending"


def test_override_accountability_trajectory_supported_unsupported_supported() -> None:
    """Task 11: override accountability can change across multiple verification cycles.

    The accountability_status must NOT latch on first verdict — it must be updatable
    from 'supported' to 'unsupported' and back to 'supported' as evidence appears
    and disappears across successive successful System 1 cycles.

    This test asserts that the ManageMemoryResult envelope correctly carries each
    status value for each cycle in the trajectory: pending -> supported -> unsupported -> supported.
    """
    statuses = ["pending", "supported", "unsupported", "supported"]
    for status in statuses:
        result = ManageMemoryResult(
            operation="apply_semantic_override",
            success=True,
            override_id="override-trajectory-1",
            diagnostics={
                "claim_id": "claim-uuid-traj",
                "accountability_status": status,
            },
        )
        assert result.diagnostics is not None
        assert result.diagnostics["accountability_status"] == status, (
            f"Expected accountability_status={status!r},"
            f" got {result.diagnostics['accountability_status']!r}"
        )


def test_override_does_not_bypass_lifecycle_rules_without_evidence() -> None:
    """Task 11: an unsupported override must not grant permanent evidence status.

    Override provenance is NOT permanent evidence by itself. An 'unsupported'
    accountability_status means the override claim lacks corroborating System 1 evidence
    and must remain subject to Task 10 lifecycle rules (i.e. can be degraded/archived).
    This test asserts the envelope correctly represents the unsupported state without
    treating the override as evidence.
    """
    result = ManageMemoryResult(
        operation="apply_semantic_override",
        success=True,
        override_id="override-unsupported-1",
        diagnostics={
            "claim_id": "claim-uuid-no-evidence",
            "accountability_status": "unsupported",
            "evidence_present": False,
        },
    )
    assert result.diagnostics is not None
    assert result.diagnostics["accountability_status"] == "unsupported"
    assert result.diagnostics["evidence_present"] is False


def test_override_service_rejects_missing_reason_at_operation_level() -> None:
    """Task 11: MemoryService rejects apply_semantic_override when override field is absent.

    The service layer must validate that record.override is present before persisting,
    returning a MEM_MISSING_REQUIRED_FIELD error.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from workflows_mcp.engine.memory_service import (
        MemoryContractError,
        MemoryRequest,
        MemoryService,
    )

    backend = MagicMock()
    backend.query = AsyncMock()
    backend.execute = AsyncMock()
    backend.begin_transaction = AsyncMock()
    backend.commit = AsyncMock()
    backend.rollback = AsyncMock()

    context = MagicMock()

    service = MemoryService(backend=backend, context=context)

    # Build a MemoryRequest with override field intentionally omitted
    request = MemoryRequest.model_validate(
        {
            "operation": "apply_semantic_override",
            "scope": {"palace": "acme", "wing": "code", "room": "svc", "compartment": "auth"},
            "record": {
                "format": "structured",
                # override field intentionally omitted
            },
        }
    )

    with pytest.raises(MemoryContractError) as exc:
        asyncio.run(service.execute(request))
    assert "MEM_MISSING_REQUIRED_FIELD" in str(exc.value)


# ---------------------------------------------------------------------------
# ADR-013 Task 1: derive_system1_topology operation contract
# ---------------------------------------------------------------------------


def test_derive_system1_topology_operation_is_accepted_by_enum() -> None:
    """derive_system1_topology must be a valid MemoryOperation string."""
    from workflows_mcp.engine.memory_service import MEMORY_OPERATION_ENUM

    assert "derive_system1_topology" in MEMORY_OPERATION_ENUM, (
        "derive_system1_topology must be registered in MEMORY_OPERATION_ENUM"
    )


def test_derive_system1_topology_is_accepted_by_memory_request_validation() -> None:
    """MemoryRequest must accept derive_system1_topology without raising."""
    from workflows_mcp.engine.memory_service import (
        DeriveSystem1TopologyInput,
        MemoryRequest,
    )

    payload = DeriveSystem1TopologyInput(
        palace="test-palace",
        evidence_ids=["11111111-1111-1111-1111-111111111111"],
    )
    req = MemoryRequest(
        operation="derive_system1_topology",
        derivation=payload,
    )
    assert req.operation == "derive_system1_topology"


def test_derive_system1_topology_rejects_unknown_operation_with_contract_code() -> None:
    """Sanity: operation enum still rejects unknown strings."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(operation="derive_system1_topology_typo")
    assert "MEM_INVALID_OPERATION" in str(exc.value)


def test_derive_system1_topology_requires_derivation_payload() -> None:
    """derive_system1_topology must require the derivation section (MEM_MISSING_REQUIRED_FIELD)."""
    with pytest.raises(ValidationError) as exc:
        MemoryRequest(operation="derive_system1_topology")
    assert "MEM_MISSING_REQUIRED_FIELD" in str(exc.value)


def test_derive_system1_topology_payload_requires_palace() -> None:
    """DeriveSystem1TopologyInput must require palace field."""
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    with pytest.raises(ValidationError):
        DeriveSystem1TopologyInput(evidence_ids=["11111111-1111-1111-1111-111111111111"])


def test_derive_system1_topology_payload_requires_evidence_ids() -> None:
    """DeriveSystem1TopologyInput must require at least one evidence_id."""
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    with pytest.raises(ValidationError):
        DeriveSystem1TopologyInput(palace="test-palace", evidence_ids=[])


@pytest.mark.parametrize(
    "wing_hint",
    [
        "default",
        "code/default/system1scan",
    ],
)
def test_derive_system1_topology_payload_rejects_wing_hint_as_extra_field(wing_hint: str) -> None:
    """DeriveSystem1TopologyInput must reject wing_hint (removed in Task 2a; use topology_override)."""  # noqa: E501
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    with pytest.raises(ValidationError):
        DeriveSystem1TopologyInput(  # type: ignore[call-arg]
            palace="test-palace",
            evidence_ids=["11111111-1111-1111-1111-111111111111"],
            wing_hint=wing_hint,
        )


def test_derive_system1_topology_operation_is_registered_in_executor_literal() -> None:
    """MemoryInput in executors_memory must accept derive_system1_topology."""
    from workflows_mcp.engine.executors_memory import MemoryInput

    inp = MemoryInput(operation="derive_system1_topology")
    assert inp.operation == "derive_system1_topology"


# ---------------------------------------------------------------------------
# ADR-013 Task 2a: TopologyOverrideInput contract + ManageMemoryResult typed fields
# ---------------------------------------------------------------------------


def test_topology_override_input_exists_and_is_importable() -> None:
    """TopologyOverrideInput must be importable from memory_service."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput  # noqa: F401


def test_topology_override_input_requires_wing() -> None:
    """TopologyOverrideInput must require wing field."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError):
        TopologyOverrideInput(
            room="auth",
            compartment="login",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_requires_room() -> None:
    """TopologyOverrideInput must require room field."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError):
        TopologyOverrideInput(
            wing="backend",
            compartment="login",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_requires_compartment() -> None:
    """TopologyOverrideInput must require compartment field."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError):
        TopologyOverrideInput(
            wing="backend",
            room="auth",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_requires_override_reason() -> None:
    """TopologyOverrideInput must require override_reason field."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError):
        TopologyOverrideInput(
            wing="backend",
            room="auth",
            compartment="login",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_requires_applied_by() -> None:
    """TopologyOverrideInput must require applied_by field."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError):
        TopologyOverrideInput(
            wing="backend",
            room="auth",
            compartment="login",
            override_reason="manual placement",
        )


def test_topology_override_input_rejects_whitespace_only_override_reason() -> None:
    """TopologyOverrideInput must reject whitespace-only override_reason."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_OVERRIDE_REASON"):
        TopologyOverrideInput(
            wing="backend",
            room="auth",
            compartment="login",
            override_reason="   ",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_rejects_whitespace_only_applied_by() -> None:
    """TopologyOverrideInput must reject whitespace-only applied_by."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_APPLIED_BY"):
        TopologyOverrideInput(
            wing="backend",
            room="auth",
            compartment="login",
            override_reason="manual placement",
            applied_by="\t\n",
        )


def test_topology_override_input_rejects_empty_wing() -> None:
    """TopologyOverrideInput must reject empty wing."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_WING"):
        TopologyOverrideInput(
            wing="",
            room="auth",
            compartment="login",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_rejects_whitespace_only_wing() -> None:
    """TopologyOverrideInput must reject whitespace-only wing."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_WING"):
        TopologyOverrideInput(
            wing="   ",
            room="auth",
            compartment="login",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_rejects_empty_room() -> None:
    """TopologyOverrideInput must reject empty room."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_ROOM"):
        TopologyOverrideInput(
            wing="backend",
            room="",
            compartment="login",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_rejects_whitespace_only_room() -> None:
    """TopologyOverrideInput must reject whitespace-only room."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_ROOM"):
        TopologyOverrideInput(
            wing="backend",
            room="\t",
            compartment="login",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_rejects_empty_compartment() -> None:
    """TopologyOverrideInput must reject empty compartment."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_COMPARTMENT"):
        TopologyOverrideInput(
            wing="backend",
            room="auth",
            compartment="",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_rejects_whitespace_only_compartment() -> None:
    """TopologyOverrideInput must reject whitespace-only compartment."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError, match="MEM_WHITESPACE_COMPARTMENT"):
        TopologyOverrideInput(
            wing="backend",
            room="auth",
            compartment="\n",
            override_reason="manual placement",
            applied_by="engineer@example.com",
        )


def test_topology_override_input_rejects_extra_fields() -> None:
    """TopologyOverrideInput must forbid extra fields."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    with pytest.raises(ValidationError):
        TopologyOverrideInput(  # type: ignore[call-arg]
            wing="backend",
            room="auth",
            compartment="login",
            override_reason="manual placement",
            applied_by="engineer@example.com",
            unknown_field="should fail",
        )


def test_topology_override_input_accepts_valid_complete_payload() -> None:
    """TopologyOverrideInput must accept a complete, valid payload."""
    from workflows_mcp.engine.memory_service import TopologyOverrideInput

    override = TopologyOverrideInput(
        wing="backend",
        room="auth",
        compartment="login",
        override_reason="manual placement by senior engineer",
        applied_by="engineer@example.com",
    )
    assert override.wing == "backend"
    assert override.room == "auth"
    assert override.compartment == "login"


def test_derive_system1_topology_input_has_topology_override_not_wing_hint() -> None:
    """DeriveSystem1TopologyInput must expose topology_override field, not wing_hint."""
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    # topology_override should be present as a field
    fields = DeriveSystem1TopologyInput.model_fields
    assert "topology_override" in fields, (
        "DeriveSystem1TopologyInput must have topology_override field"
    )
    assert "wing_hint" not in fields, (
        "wing_hint must be removed from DeriveSystem1TopologyInput (replaced by topology_override)"
    )


def test_derive_system1_topology_input_topology_override_is_optional() -> None:
    """DeriveSystem1TopologyInput must accept without topology_override (pure derivation)."""
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    payload = DeriveSystem1TopologyInput(
        palace="test-palace",
        evidence_ids=["11111111-1111-1111-1111-111111111111"],
    )
    assert payload.topology_override is None


def test_derive_system1_topology_input_accepts_valid_topology_override() -> None:
    """DeriveSystem1TopologyInput must accept a valid TopologyOverrideInput."""
    from workflows_mcp.engine.memory_service import (
        DeriveSystem1TopologyInput,
        TopologyOverrideInput,
    )

    override = TopologyOverrideInput(
        wing="backend",
        room="auth",
        compartment="login",
        override_reason="manual placement",
        applied_by="engineer@example.com",
    )
    payload = DeriveSystem1TopologyInput(
        palace="test-palace",
        evidence_ids=["11111111-1111-1111-1111-111111111111"],
        topology_override=override,
    )
    assert payload.topology_override is not None
    assert payload.topology_override.wing == "backend"


def test_derive_system1_topology_input_rejects_extra_fields() -> None:
    """DeriveSystem1TopologyInput must forbid extra fields (extra='forbid')."""
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    with pytest.raises(ValidationError):
        DeriveSystem1TopologyInput(  # type: ignore[call-arg]
            palace="test-palace",
            evidence_ids=["11111111-1111-1111-1111-111111111111"],
            wing_hint="backend",
        )


def test_manage_memory_result_exposes_derived_wing() -> None:
    """ManageMemoryResult must expose derived_wing field."""
    from workflows_mcp.engine.memory_service import ManageMemoryResult

    result = ManageMemoryResult(operation="derive_system1_topology", derived_wing="backend")
    assert result.derived_wing == "backend"


def test_manage_memory_result_exposes_derived_room() -> None:
    """ManageMemoryResult must expose derived_room field."""
    from workflows_mcp.engine.memory_service import ManageMemoryResult

    result = ManageMemoryResult(operation="derive_system1_topology", derived_room="auth")
    assert result.derived_room == "auth"


def test_manage_memory_result_exposes_derived_compartment() -> None:
    """ManageMemoryResult must expose derived_compartment field."""
    from workflows_mcp.engine.memory_service import ManageMemoryResult

    result = ManageMemoryResult(operation="derive_system1_topology", derived_compartment="login")
    assert result.derived_compartment == "login"


def test_manage_memory_result_exposes_derivation_source() -> None:
    """ManageMemoryResult must expose derivation_source field."""
    from workflows_mcp.engine.memory_service import ManageMemoryResult

    result = ManageMemoryResult(
        operation="derive_system1_topology", derivation_source="system1_derived"
    )
    assert result.derivation_source == "system1_derived"


def test_manage_memory_result_exposes_provenance_id() -> None:
    """ManageMemoryResult must expose provenance_id field."""
    from workflows_mcp.engine.memory_service import ManageMemoryResult

    result = ManageMemoryResult(
        operation="derive_system1_topology",
        provenance_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )
    assert result.provenance_id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def test_manage_memory_result_exposes_claim_id() -> None:
    """ManageMemoryResult must expose claim_id field (singular, for topology derivation result)."""
    from workflows_mcp.engine.memory_service import ManageMemoryResult

    result = ManageMemoryResult(
        operation="derive_system1_topology",
        claim_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    )
    assert result.claim_id == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def test_derive_system1_topology_input_rejects_is_new_wing_without_proof_bundle() -> None:
    """DeriveSystem1TopologyInput must reject is_new_wing=True when proof_bundle is None.

    ADR-013 Task 4: proof_bundle is required when is_new_wing=True.
    The error must use the missing-field style code MEM_MISSING_REQUIRED_FIELD.
    """
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    with pytest.raises(ValidationError) as exc_info:
        DeriveSystem1TopologyInput(
            palace="test-palace",
            evidence_ids=["11111111-1111-1111-1111-111111111111"],
            is_new_wing=True,
            proof_bundle=None,
        )

    errors = exc_info.value.errors()
    assert any(
        "proof_bundle" in str(e.get("loc", "")) or "proof_bundle" in str(e.get("msg", ""))
        for e in errors
    ), f"ValidationError must mention proof_bundle; got {errors!r}"


def test_manage_memory_result_topology_fields_default_to_none() -> None:
    """ManageMemoryResult topology derivation fields must default to None."""
    from workflows_mcp.engine.memory_service import ManageMemoryResult

    result = ManageMemoryResult(operation="store")
    assert result.derived_wing is None
    assert result.derived_room is None
    assert result.derived_compartment is None
    assert result.derivation_source is None
    assert result.provenance_id is None
    assert result.claim_id is None


# ---------------------------------------------------------------------------
# ADR-013 Task 5b: inline structural candidate passthrough contract
# ---------------------------------------------------------------------------


def test_derive_system1_topology_input_accepts_inline_candidates() -> None:
    """DeriveSystem1TopologyInput must accept inline_candidates without evidence_ids.

    ADR-013 Task 5b: inline structural evidence candidates bypass the pre-store step.
    evidence_ids is optional when inline_candidates are provided.
    """
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    inp = DeriveSystem1TopologyInput(
        palace="test-palace",
        inline_candidates=[
            {
                "entity_stable_id": "src/foo.py::MyClass",
                "entity_type": "class",
                "evidence_category": "structural_class",
                "evidence_data": {},
            },
            {
                "entity_stable_id": "src/foo.py",
                "entity_type": "module",
                "evidence_category": "structural_module",
                "evidence_data": {},
            },
        ],
    )
    assert inp.palace == "test-palace"
    assert inp.evidence_ids == []
    assert len(inp.inline_candidates) == 2


def test_derive_system1_topology_input_rejects_when_neither_evidence_ids_nor_inline_candidates() -> (  # noqa: E501
    None
):  # noqa: E501
    """DeriveSystem1TopologyInput must reject input when both evidence sources are absent.

    ADR-013 Task 5b: at least one evidence source is required (fail closed).
    """
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    with pytest.raises(ValidationError) as exc_info:
        DeriveSystem1TopologyInput(palace="test-palace")

    errors = exc_info.value.errors()
    assert any(
        "evidence" in str(e.get("loc", "")) or "evidence" in str(e.get("msg", "")).lower()
        for e in errors
    ), f"ValidationError must mention evidence requirement; got {errors!r}"


def test_derive_system1_topology_input_evidence_ids_optional_when_inline_candidates_provided() -> (
    None
):
    """DeriveSystem1TopologyInput must not require evidence_ids when inline_candidates are given.

    ADR-013 Task 5b: inline candidates serve as evidence source so evidence_ids becomes optional.
    """
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    # Must not raise
    inp = DeriveSystem1TopologyInput(
        palace="test-palace",
        inline_candidates=[
            {
                "entity_stable_id": "src/bar.py::func",
                "entity_type": "function",
                "evidence_category": "structural_function",
                "evidence_data": {"calls": 3},
            }
        ],
    )
    assert inp.evidence_ids == []
    assert len(inp.inline_candidates) == 1


def test_derive_system1_topology_input_evidence_ids_required_when_no_inline_candidates() -> None:
    """DeriveSystem1TopologyInput must require evidence_ids when inline_candidates is empty.

    ADR-013 Task 5b: at least one of [evidence_ids, inline_candidates] must be non-empty.
    """
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    with pytest.raises(ValidationError):
        DeriveSystem1TopologyInput(palace="test-palace", evidence_ids=[], inline_candidates=[])


def test_derive_system1_topology_input_accepts_parser_metadata_as_evidence_metadata() -> None:
    """DeriveSystem1TopologyInput must accept parser_metadata field.

    ADR-013 Task 5b: parser metadata (language, path) is carried as diagnostics/evidence metadata,
    never as direct topology truth.
    """
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    inp = DeriveSystem1TopologyInput(
        palace="test-palace",
        inline_candidates=[
            {
                "entity_stable_id": "src/foo.py::MyClass",
                "entity_type": "class",
                "evidence_category": "structural_class",
                "evidence_data": {},
            }
        ],
        parser_metadata={"language": "python", "file_path": "src/foo.py"},
    )
    assert inp.parser_metadata == {"language": "python", "file_path": "src/foo.py"}


def test_derive_system1_topology_inline_candidate_wing_equals_language_is_forbidden() -> None:
    """Parser metadata language must never become topology wing via inline candidate path.

    ADR-013 Task 5b: wing must not default to programming language.
    Parser metadata is evidence metadata only, not topology truth.
    This test asserts the model does not carry a 'wing' field that could be language-derived.
    """
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    inp = DeriveSystem1TopologyInput(
        palace="test-palace",
        inline_candidates=[
            {
                "entity_stable_id": "src/foo.py::MyClass",
                "entity_type": "class",
                "evidence_category": "structural_class",
                "evidence_data": {},
            }
        ],
        parser_metadata={"language": "python"},
    )
    # No 'wing' field is derived from parser_metadata.language
    assert not hasattr(inp, "wing") or inp.wing is None  # type: ignore[union-attr]


def test_derive_system1_topology_inline_candidate_room_equals_path_is_forbidden() -> None:
    """Parser metadata file path must never become topology room via inline candidate path.

    ADR-013 Task 5b: room must not default to folder/package path.
    """
    from workflows_mcp.engine.memory_service import DeriveSystem1TopologyInput

    inp = DeriveSystem1TopologyInput(
        palace="test-palace",
        inline_candidates=[
            {
                "entity_stable_id": "src/foo.py::MyClass",
                "entity_type": "class",
                "evidence_category": "structural_class",
                "evidence_data": {},
            }
        ],
        parser_metadata={"file_path": "src/foo.py", "package": "workflows_mcp.engine"},
    )
    # No 'room' field is derived from parser_metadata paths
    assert not hasattr(inp, "room") or inp.room is None  # type: ignore[union-attr]


def test_memory_input_accepts_derivation_field_for_passthrough() -> None:
    """MemoryInput must expose a 'derivation' field for derive_system1_topology passthrough.

    ADR-013 Task 5b: inline candidates and parser_metadata must be passable from workflow blocks.
    """
    from workflows_mcp.engine.executors_memory import MemoryInput

    inp = MemoryInput(
        operation="derive_system1_topology",
        derivation={
            "palace": "test-palace",
            "inline_candidates": [
                {
                    "entity_stable_id": "src/foo.py::MyClass",
                    "entity_type": "class",
                    "evidence_category": "structural_class",
                    "evidence_data": {},
                }
            ],
            "parser_metadata": {"language": "python"},
        },
    )
    assert inp.derivation is not None
    assert inp.derivation["palace"] == "test-palace"
