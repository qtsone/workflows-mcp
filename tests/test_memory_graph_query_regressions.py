"""Regression tests for graph query behavior in memory service/tooling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.knowledge.graph import graph_stats
from workflows_mcp.engine.memory_service import (
    ManageMemoryRequest,
    ManageMemoryResult,
    MemoryGraphInput,
    MemoryRequest,
    MemoryResponseInput,
    MemoryResult,
    MemoryService,
    QueryMemoryRequest,
    QueryMemoryResult,
)
from workflows_mcp.tools_memory import _shape_memory_response


def _make_service() -> MemoryService:
    backend = MagicMock()
    context = MagicMock()
    context.execution_context = None
    return MemoryService(backend=backend, context=context)


@pytest.mark.asyncio
async def test_query_graph_path_includes_evidence_when_path_exists() -> None:
    service = _make_service()

    path_result = {
        "nodes": [
            {"id": "a-id", "entity_type": "SERVICE", "name": "a", "confidence": 1.0},
            {"id": "b-id", "entity_type": "SERVICE", "name": "b", "confidence": 1.0},
        ],
        "edges": [
            {
                "id": "e-1",
                "source_entity_id": "a-id",
                "target_entity_id": "b-id",
                "relation_type": "depends_on",
                "confidence": 1.0,
                "valid_from": None,
                "valid_to": None,
            }
        ],
        "paths": [
            {
                "nodes": [
                    {"id": "a-id", "entity_type": "SERVICE", "name": "a", "confidence": 1.0},
                    {"id": "b-id", "entity_type": "SERVICE", "name": "b", "confidence": 1.0},
                ],
                "edges": [
                    {
                        "id": "e-1",
                        "source_entity_id": "a-id",
                        "target_entity_id": "b-id",
                        "relation_type": "depends_on",
                        "confidence": 1.0,
                        "valid_from": None,
                        "valid_to": None,
                    }
                ],
                "hop_count": 1,
            }
        ],
        "traversal_count": 2,
        "diagnostics": {"expanded_nodes": 1, "pruned_edges": 0, "latency_ms": 1.0},
    }

    with patch(
        "workflows_mcp.engine.memory_service.graph_path", new=AsyncMock(return_value=path_result)
    ):
        result = await service._query_graph(
            QueryMemoryRequest(
                query="dependency path",
                strategy="graph",
                graph_op="path",
                start_entity="a",
                end_entity="b",
            )
        )

    assert result.paths
    assert result.evidence[0]["nodes"] == path_result["nodes"]
    assert result.evidence[0]["edges"][0]["id"] == "e-1"
    assert result.evidence[0]["edges"][0]["supporting_memories"] == []
    assert result.diagnostics["fusion_version"] == "graph-only.v1"
    assert result.diagnostics["algorithm_versions"]["graph"] == "bfs-shortest-path.v1"


@pytest.mark.asyncio
async def test_query_graph_traverse_includes_evidence_when_neighbors_exist() -> None:
    service = _make_service()

    traverse_result = {
        "nodes": [
            {"id": "a-id", "entity_type": "SERVICE", "name": "a", "confidence": 1.0},
            {"id": "b-id", "entity_type": "SERVICE", "name": "b", "confidence": 1.0},
        ],
        "edges": [
            {
                "id": "e-1",
                "source_entity_id": "a-id",
                "target_entity_id": "b-id",
                "relation_type": "depends_on",
                "confidence": 1.0,
                "valid_from": None,
                "valid_to": None,
            }
        ],
        "paths": [],
        "traversal_count": 2,
        "diagnostics": {"expanded_nodes": 1, "pruned_edges": 0, "latency_ms": 1.0},
    }

    with patch(
        "workflows_mcp.engine.memory_service.graph_traverse",
        new=AsyncMock(return_value=traverse_result),
    ):
        result = await service._query_graph(
            QueryMemoryRequest(
                query="neighbors",
                strategy="graph",
                graph_op="traverse",
                start_entity="a",
            )
        )

    assert result.evidence[0]["nodes"] == traverse_result["nodes"]
    assert result.evidence[0]["edges"][0]["id"] == "e-1"
    assert result.evidence[0]["edges"][0]["supporting_memories"] == []
    assert result.diagnostics["fusion_version"] == "graph-only.v1"
    assert result.diagnostics["algorithm_versions"]["graph"] == "bfs-traverse.v1"


@pytest.mark.asyncio
async def test_query_graph_stats_without_start_entity_does_not_error() -> None:
    service = _make_service()

    stats_result = {
        "nodes": [],
        "edges": [],
        "paths": [],
        "traversal_count": 0,
        "diagnostics": {
            "expanded_nodes": 0,
            "pruned_edges": 0,
            "latency_ms": 1.0,
            "entity_count": 2,
            "relation_count": 1,
        },
    }

    with patch(
        "workflows_mcp.engine.memory_service.graph_stats", new=AsyncMock(return_value=stats_result)
    ) as mock_stats:
        result = await service._query_graph(
            QueryMemoryRequest(
                query="graph stats",
                strategy="graph",
                graph_op="stats",
                start_entity=None,
            )
        )

    assert "error" not in result.diagnostics
    assert result.diagnostics["fusion_version"] == "graph-only.v1"
    assert result.diagnostics["algorithm_versions"]["graph"] == "degree-stats.v1"
    mock_stats.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_graph_rejects_interval_temporal_filters() -> None:
    service = _make_service()

    result = await service._query_graph(
        QueryMemoryRequest(
            query="graph stats",
            strategy="graph",
            graph_op="stats",
            from_="2026-04-01T00:00:00Z",
            to="2026-04-30T23:59:59Z",
        )
    )

    assert (
        result.diagnostics["error"]
        == "graph strategy supports 'as_of' only; 'from'/'to' is not supported"
    )


@pytest.mark.asyncio
async def test_query_graph_applies_scope_for_start_entity_resolution() -> None:
    service = _make_service()
    service._backend.query = AsyncMock(return_value=MagicMock(rows=[]))

    result = await service._query_graph(
        QueryMemoryRequest(
            query="neighbors",
            strategy="graph",
            graph_op="neighbors",
            start_entity="service-a",
            namespace="svc",
            room="component",
            scope={"corridor": "topic"},
        )
    )

    assert result.diagnostics["scope_mode"] == "graph_scoped"
    assert result.diagnostics["scope_applied"] is True
    assert result.diagnostics["error"] == "start_entity not found in scoped graph"


@pytest.mark.asyncio
async def test_query_graph_stats_uses_scoped_aggregate_when_scope_present() -> None:
    service = _make_service()
    service._backend.query = AsyncMock(
        return_value=MagicMock(
            rows=[{"entity_count": 4, "relation_count": 3, "distinct_relation_types": 2}]
        )
    )

    result = await service._query_graph(
        QueryMemoryRequest(
            query="scoped stats",
            strategy="graph",
            graph_op="stats",
            namespace="svc",
            room="component",
            scope={"corridor": "topic"},
        )
    )

    assert result.diagnostics["scope_mode"] == "graph_scoped"
    assert result.diagnostics["scope_applied"] is True
    assert result.diagnostics["entity_count"] == 4
    assert result.diagnostics["relation_count"] == 3
    assert result.diagnostics["distinct_relation_types"] == 2
    assert result.diagnostics["scope_status"] == "applied_with_results"


@pytest.mark.asyncio
async def test_graph_stats_supports_global_aggregate_when_entity_ref_is_none() -> None:
    backend = MagicMock()
    backend.query = AsyncMock(
        return_value=MagicMock(
            rows=[{"entity_count": 3, "relation_count": 2, "distinct_relation_types": 1}]
        )
    )

    result = await graph_stats(None, backend)

    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["paths"] == []
    assert result["diagnostics"]["entity_count"] == 3
    assert result["diagnostics"]["relation_count"] == 2


def test_graph_response_shape_includes_diagnostics_for_all_graph_ops() -> None:
    result = MemoryResult(
        operation="query",
        query=QueryMemoryResult(
            paths=[{"hop_count": 1}],
            evidence=[{"nodes": [{"id": "a"}], "edges": [{"id": "e"}]}],
            diagnostics={"expanded_nodes": 1, "pruned_edges": 0, "latency_ms": 1.0},
        ),
    )

    payload = _shape_memory_response(result, MemoryResponseInput(mode="graph"))

    assert payload["paths"] == [{"hop_count": 1}]
    assert payload["nodes"] == [{"id": "a"}]
    assert payload["edges"] == [{"id": "e"}]
    assert payload["diagnostics"] == {"expanded_nodes": 1, "pruned_edges": 0, "latency_ms": 1.0}


@pytest.mark.asyncio
async def test_graph_store_relation_rejects_non_curated_without_evidence() -> None:
    service = _make_service()

    result = await service._manage_graph_store_relation(
        ManageMemoryRequest(
            operation="graph_store_relation",
            source_entity="service-a",
            target_entity="service-b",
            relation_type="CORRIDOR",
            curated=False,
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("MEM_GRAPH_EVIDENCE_REQUIRED")


@pytest.mark.asyncio
async def test_graph_store_relation_allows_curated_without_evidence() -> None:
    service = _make_service()
    service._backend.query = AsyncMock(
        side_effect=[
            MagicMock(rows=[{"id": "11111111-1111-1111-1111-111111111111"}]),
            MagicMock(rows=[{"id": "22222222-2222-2222-2222-222222222222"}]),
            MagicMock(rows=[]),
            MagicMock(rows=[{"id": "33333333-3333-3333-3333-333333333333"}]),
        ]
    )

    result = await service._manage_graph_store_relation(
        ManageMemoryRequest(
            operation="graph_store_relation",
            source_entity="service-a",
            target_entity="service-b",
            relation_type="depends_on",
            curated=True,
        )
    )

    assert result.success is True
    assert result.relation_id == "33333333-3333-3333-3333-333333333333"


@pytest.mark.asyncio
async def test_graph_store_relation_preserves_curated_true_on_update_when_request_false() -> None:
    service = _make_service()
    service._backend.query = AsyncMock(
        side_effect=[
            MagicMock(rows=[{"id": "11111111-1111-1111-1111-111111111111"}]),
            MagicMock(rows=[{"id": "22222222-2222-2222-2222-222222222222"}]),
            MagicMock(rows=[{"id": "33333333-3333-3333-3333-333333333333"}]),
            MagicMock(rows=[{"id": "33333333-3333-3333-3333-333333333333"}]),
        ]
    )

    result = await service._manage_graph_store_relation(
        ManageMemoryRequest(
            operation="graph_store_relation",
            source_entity="service-a",
            target_entity="service-b",
            relation_type="depends_on",
            curated=False,
        )
    )

    assert result.success is True
    update_sql, update_params = service._backend.query.await_args_list[3].args
    assert "WHEN curated AND NOT $5 THEN TRUE" in update_sql
    assert update_params[4] is False


@pytest.mark.asyncio
async def test_graph_store_relation_rejects_case_variant_corridor_without_evidence() -> None:
    service = _make_service()

    result = await service._manage_graph_store_relation(
        ManageMemoryRequest(
            operation="graph_store_relation",
            source_entity="service-a",
            target_entity="service-b",
            relation_type="corridor",
            curated=False,
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("MEM_GRAPH_EVIDENCE_REQUIRED")


@pytest.mark.asyncio
async def test_graph_store_relation_accepts_corridor_with_only_legacy_single_evidence_field() -> (
    None
):
    service = _make_service()
    service._backend.query = AsyncMock(
        side_effect=[
            MagicMock(rows=[{"id": "11111111-1111-1111-1111-111111111111"}]),
            MagicMock(rows=[{"id": "22222222-2222-2222-2222-222222222222"}]),
            MagicMock(rows=[{"cnt": 1}]),
            MagicMock(rows=[]),
            MagicMock(rows=[{"id": "33333333-3333-3333-3333-333333333333"}]),
        ]
    )

    result = await service._manage_graph_store_relation(
        ManageMemoryRequest(
            operation="graph_store_relation",
            source_entity="service-a",
            target_entity="service-b",
            relation_type="CORRIDOR",
            curated=False,
            evidence_memory_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        )
    )

    assert result.success is True
    assert result.relation_id == "33333333-3333-3333-3333-333333333333"


@pytest.mark.asyncio
async def test_graph_upsert_corridor_uses_effective_evidence_list_for_validation() -> None:
    service = _make_service()
    service.manage = AsyncMock(
        return_value=ManageMemoryResult(
            operation="graph_store_relation",
            success=True,
            relation_id="33333333-3333-3333-3333-333333333333",
        )
    )

    result = await service.execute(
        MemoryRequest(
            operation="graph_upsert",
            graph=MemoryGraphInput(
                kind="link",
                from_ref="service-a",
                to_ref="service-b",
                link_type="CORRIDOR",
                curated=False,
                evidence_memory_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            ),
        )
    )

    assert result.manage is not None
    assert result.manage.success is True
    manage_request = service.manage.await_args.args[0]
    assert manage_request.evidence_memory_id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert manage_request.evidence_memory_ids == ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]


@pytest.mark.asyncio
async def test_query_graph_hydrates_supporting_memories_for_evidence_links() -> None:
    service = _make_service()
    service._backend.query = AsyncMock(
        return_value=MagicMock(
            rows=[
                {
                    "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "content": "Memory evidence",
                    "confidence": 0.9,
                    "authority": "USER_VALIDATED",
                    "source_name": "src",
                    "item_path": "path/to/file.py",
                    "namespace": "payments",
                    "room": "orders",
                    "corridor": "events",
                }
            ]
        )
    )

    traverse_result = {
        "nodes": [
            {"id": "a-id", "entity_type": "SERVICE", "name": "a", "confidence": 1.0},
            {"id": "b-id", "entity_type": "SERVICE", "name": "b", "confidence": 1.0},
        ],
        "edges": [
            {
                "id": "e-1",
                "source_entity_id": "a-id",
                "target_entity_id": "b-id",
                "relation_type": "depends_on",
                "confidence": 1.0,
                "evidence_memory_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "evidence_memory_ids": ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
                "curated": False,
                "valid_from": None,
                "valid_to": None,
            }
        ],
        "paths": [],
        "traversal_count": 2,
        "diagnostics": {"expanded_nodes": 1, "pruned_edges": 0, "latency_ms": 1.0},
    }

    with patch(
        "workflows_mcp.engine.memory_service.graph_traverse",
        new=AsyncMock(return_value=traverse_result),
    ):
        result = await service._query_graph(
            QueryMemoryRequest(
                query="neighbors",
                strategy="graph",
                graph_op="traverse",
                start_entity="a",
            )
        )

    assert len(result.memories) == 1
    assert result.memories[0]["id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert result.evidence[0]["edges"][0]["supporting_memories"][0]["id"] == (
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    )
