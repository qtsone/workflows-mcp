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
                "Retry ingest with scope.palace/wing/room/compartment, "
                "scope_token, or context_id."
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
    for var in ["WORKFLOWS_USER_ID", "WORKFLOWS_USER", "MCP_USER_ID", "USER", "USERNAME", "LOGNAME"
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
    for var in ["WORKFLOWS_USER_ID", "WORKFLOWS_USER", "MCP_USER_ID", "USER", "USERNAME", "LOGNAME"
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
    memory_inserts = [
        (sql, params) for sql, params in execute_calls if "knowledge_memories" in sql
    ]
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
