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

    with pytest.raises(MemoryContractError, match="COMPARTMENT_REQUIRED"):
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
    context = Execution()
    service = MemoryService(backend=object(), context=context)

    async def _fake_manage(_request: object) -> ManageMemoryResult:
        return ManageMemoryResult(operation="graph_store_relation")

    service.manage = _fake_manage  # type: ignore[method-assign]

    result = await service.execute(
        MemoryRequest(
            operation="graph_upsert",
            graph={"kind": "link", "from": "alice", "to": "acme", "link_type": "uses"},
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

    with pytest.raises(MemoryContractError, match="SCOPE_UNRESOLVED"):
        await service.execute(
            MemoryRequest(operation="query", query={"text": "incident", "mode": "search"})
        )


def test_tool_error_payload_uses_machine_readable_envelope_for_contract_errors() -> None:
    payload = _tool_error_payload(
        "memory",
        MemoryContractError(code="MEM_INVALID_SCOPE", message="scope invalid", retryable=False),
    )
    assert payload["error"]["code"] == "MEM_INVALID_SCOPE"
    assert payload["error"]["message"] == "scope invalid"
    assert payload["error"]["retryable"] is False
    assert payload["error"].get("correlation_id")


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
    assert diagnostics["retrieval"]["s2"]["enabled"] is True


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
