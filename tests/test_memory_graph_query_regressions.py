"""Regression tests for graph query behavior in memory service/tooling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.knowledge.graph import graph_stats
from workflows_mcp.engine.memory_service import (
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

    with patch("workflows_mcp.engine.memory_service.graph_path", new=AsyncMock(return_value=path_result)):
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
    assert result.evidence == [{"nodes": path_result["nodes"], "edges": path_result["edges"]}]


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

    assert result.evidence == [{"nodes": traverse_result["nodes"], "edges": traverse_result["edges"]}]


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

    with patch("workflows_mcp.engine.memory_service.graph_stats", new=AsyncMock(return_value=stats_result)) as mock_stats:
        result = await service._query_graph(
            QueryMemoryRequest(
                query="graph stats",
                strategy="graph",
                graph_op="stats",
                start_entity=None,
            )
        )

    assert "error" not in result.diagnostics
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

    assert result.diagnostics["error"] == "graph strategy supports 'as_of' only; 'from'/'to' is not supported"


@pytest.mark.asyncio
async def test_graph_stats_supports_global_aggregate_when_entity_ref_is_none() -> None:
    backend = MagicMock()
    backend.query = AsyncMock(
        return_value=MagicMock(rows=[{"entity_count": 3, "relation_count": 2, "distinct_relation_types": 1}])
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
