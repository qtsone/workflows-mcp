"""Behavior tests asserting palace isolation across all graph SQL paths.

Each test seeds two palaces with overlapping entity names and a relation
inside each. After the fix, queries scoped to palace A must never observe
palace B rows. Before the fix, they do — these tests fail loudly.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio


def _make_config() -> ConnectionConfig:
    return ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=os.environ.get("MEMORY_DB_HOST", "localhost"),
        port=int(os.environ.get("MEMORY_DB_PORT", "5432")),
        database=os.environ.get("MEMORY_DB_NAME", "workflows"),
        username=os.environ.get("MEMORY_DB_USER", "workflows"),
        password=os.environ.get("MEMORY_DB_PASSWORD", "supersecret"),
    )


@pytest_asyncio.fixture
async def knowledge_backend() -> AsyncIterator[PostgresBackend]:
    """Provide a connected PostgresBackend with the knowledge schema applied."""
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


@pytest_asyncio.fixture
async def memory_service(knowledge_backend: PostgresBackend) -> Any:
    """Provide a MemoryService wired to the live PostgresBackend."""
    from unittest.mock import MagicMock

    from workflows_mcp.engine.executor_base import Execution
    from workflows_mcp.engine.memory_service import MemoryService

    context = MagicMock(spec=Execution)
    context.execution_context = MagicMock()
    context.execution_context.get = MagicMock(return_value=None)
    context.execution_context.user_id = None
    context.execution_context.user_string_id = None
    context.execution_context.auth_method = None
    return MemoryService(backend=knowledge_backend, context=context)


async def _seed_palace(
    backend: Any,
    *,
    palace: str,
    namespace: str = "code",
    room: str = "default",
    corridor: str = "iso",
) -> dict[str, str]:
    """Insert two entities ('alpha', 'beta') and an alpha→beta relation in one palace.

    Returns a dict with the inserted entity ids by name.
    """
    alpha = await backend.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name)
        VALUES ($1, $2, $3, $4, 'Function', 'alpha')
        RETURNING id
        """,
        (palace, namespace, room, corridor),
    )
    beta = await backend.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name)
        VALUES ($1, $2, $3, $4, 'Function', 'beta')
        RETURNING id
        """,
        (palace, namespace, room, corridor),
    )
    alpha_id = str(alpha.rows[0]["id"])
    beta_id = str(beta.rows[0]["id"])
    await backend.execute(
        """
        INSERT INTO knowledge_relations
            (source_entity_id, target_entity_id, relation_type)
        VALUES ($1::uuid, $2::uuid, 'CALLS')
        """,
        (alpha_id, beta_id),
    )
    return {"alpha": alpha_id, "beta": beta_id}


@pytest_asyncio.fixture
async def two_palaces(knowledge_backend: PostgresBackend) -> AsyncIterator[dict[str, dict[str, str]]]:
    """Seed palace_a and palace_b with the same shape; return per-palace ids."""
    a_ids = await _seed_palace(knowledge_backend, palace="palace_a")
    b_ids = await _seed_palace(knowledge_backend, palace="palace_b")
    yield {"a": a_ids, "b": b_ids}
    # Cleanup after test
    await knowledge_backend.execute(
        "DELETE FROM knowledge_relations "
        "WHERE source_entity_id IN (SELECT id FROM knowledge_entities WHERE palace IN ('palace_a', 'palace_b'))",
        (),
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_entity_memories "
        "WHERE memory_id IN (SELECT id FROM knowledge_memories WHERE palace IN ('palace_a', 'palace_b'))",
        (),
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_entity_embeddings "
        "WHERE entity_id IN (SELECT id FROM knowledge_entities WHERE palace IN ('palace_a', 'palace_b'))",
        (),
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_memories WHERE palace IN ('palace_a', 'palace_b')", ()
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_entities WHERE palace IN ('palace_a', 'palace_b')", ()
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_communities WHERE palace IN ('palace_a', 'palace_b')", ()
    )


async def test_two_palaces_seed_helper(knowledge_backend: PostgresBackend, two_palaces: dict[str, dict[str, str]]) -> None:
    """Sanity: both palaces have an alpha and beta entity."""
    counts = await knowledge_backend.query(
        "SELECT palace, COUNT(*)::int AS n FROM knowledge_entities "
        "WHERE name IN ('alpha','beta') AND palace IN ('palace_a','palace_b') "
        "GROUP BY palace ORDER BY palace",
        (),
    )
    assert {row["palace"]: row["n"] for row in counts.rows} == {
        "palace_a": 2,
        "palace_b": 2,
    }


async def test_graph_query_without_palace_is_rejected(memory_service: Any) -> None:
    """External graph requests must not fall back to deployment-global scope."""
    from workflows_mcp.engine.memory_service import QueryMemoryRequest

    request = QueryMemoryRequest.model_validate(
        {
            "query": "",
            "strategy": "graph",
            "graph_op": "neighbors",
            "start_entity": "alpha",
            "namespace": "code",
            "room": "default",
            "scope": {"compartment": "iso"},
        }
    )
    result = await memory_service.query(request)

    assert result.diagnostics.get("error_code") == "MEM_PALACE_REQUIRED"
    assert "MEM_PALACE_REQUIRED" in result.diagnostics.get("error", "")


async def test_graph_stats_without_entity_and_without_palace_is_rejected(
    memory_service: Any,
) -> None:
    """Stats without start entity still requires request-layer palace."""
    from workflows_mcp.engine.memory_service import QueryMemoryRequest

    request = QueryMemoryRequest.model_validate(
        {
            "query": "",
            "strategy": "graph",
            "graph_op": "stats",
            "namespace": "code",
            "room": "default",
            "scope": {"compartment": "iso"},
        }
    )
    result = await memory_service.query(request)

    assert result.diagnostics.get("error_code") == "MEM_PALACE_REQUIRED"
    assert "MEM_PALACE_REQUIRED" in result.diagnostics.get("error", "")


async def test_resolve_scoped_graph_entity_rejects_cross_palace_uuid(
    memory_service: Any, two_palaces: dict[str, dict[str, str]]
) -> None:
    """Querying palace_a with palace_b's UUID must not resolve."""
    from workflows_mcp.engine.memory_service import (
        QueryMemoryRequest,
    )

    palace_b_alpha_uuid = two_palaces["b"]["alpha"]

    # Build a graph_op='neighbors' request scoped to palace_a but with palace_b's UUID
    request = QueryMemoryRequest.model_validate(
        {
            "query": "",
            "strategy": "graph",
            "graph_op": "neighbors",
            "start_entity": palace_b_alpha_uuid,
            "palace": "palace_a",
            "namespace": "code",
            "room": "default",
            "scope": {"compartment": "iso"},
        }
    )
    result = await memory_service.query(request)

    # The fix: palace_a cannot resolve palace_b's UUID; expect the not-found error
    assert result.diagnostics.get("error") == "start_entity not found in scoped graph"


async def test_resolve_entity_id_manage_cross_palace_uuid_returns_not_found(
    knowledge_backend: PostgresBackend, two_palaces: dict[str, dict[str, str]]
) -> None:
    """The shared manage resolver must not leak UUID existence across palaces."""
    from workflows_mcp.engine.memory_service import _resolve_entity_id_manage

    resolved = await _resolve_entity_id_manage(
        two_palaces["b"]["alpha"],
        knowledge_backend,
        palace="palace_a",
        namespace="code",
        room="default",
        corridor="iso",
    )

    assert resolved is None


async def test_filter_graph_result_drops_cross_palace_nodes(
    memory_service: Any, knowledge_backend: PostgresBackend, two_palaces: dict[str, dict[str, str]]
) -> None:
    """A traversal in palace_a must drop any node whose row is in palace_b.

    We force a leak by inserting a cross-palace relation row with palace_a's
    alpha as source and palace_b's beta as target. The post-filter must drop
    palace_b's beta from the result.
    """
    palace_a_alpha = two_palaces["a"]["alpha"]
    palace_b_beta = two_palaces["b"]["beta"]

    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_relations
            (source_entity_id, target_entity_id, relation_type)
        VALUES ($1::uuid, $2::uuid, 'CALLS')
        """,
        (palace_a_alpha, palace_b_beta),
    )

    from workflows_mcp.engine.memory_service import QueryMemoryRequest

    request = QueryMemoryRequest.model_validate(
        {
            "query": "",
            "strategy": "graph",
            "graph_op": "traverse",
            "start_entity": palace_a_alpha,
            "palace": "palace_a",
            "namespace": "code",
            "room": "default",
            "scope": {"compartment": "iso"},
            "max_hops": 3,
            "max_nodes": 20,
        }
    )
    result = await memory_service.query(request)

    node_ids = {
        str(node["id"])
        for evidence in (result.evidence or [])
        for node in evidence.get("nodes", [])
        if node.get("id")
    }
    assert palace_b_beta not in node_ids, (
        "cross-palace beta leaked through the post-filter"
    )


async def test_graph_traverse_refuses_to_cross_palace_via_relation(
    memory_service: Any, knowledge_backend: PostgresBackend, two_palaces: dict[str, dict[str, str]]
) -> None:
    """Even with a cross-palace relation, traverse must not yield foreign nodes."""
    palace_a_alpha = two_palaces["a"]["alpha"]
    palace_b_beta = two_palaces["b"]["beta"]

    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_relations
            (source_entity_id, target_entity_id, relation_type)
        VALUES ($1::uuid, $2::uuid, 'CALLS')
        """,
        (palace_a_alpha, palace_b_beta),
    )

    from workflows_mcp.engine.knowledge.graph import graph_traverse

    result = await graph_traverse(
        palace_a_alpha,
        knowledge_backend,
        relation_types=["CALLS"],
        max_hops=3,
        max_nodes=20,
        as_of=None,
        palace="palace_a",
    )
    node_ids = {str(node["id"]) for node in result["nodes"]}
    assert palace_b_beta not in node_ids


async def test_scoped_graph_stats_counts_only_own_palace(
    memory_service: Any, two_palaces: dict[str, dict[str, str]]
) -> None:
    """A stats query in palace_a must not include palace_b entity/relation counts."""
    from workflows_mcp.engine.memory_service import QueryMemoryRequest

    request = QueryMemoryRequest.model_validate(
        {
            "query": "",
            "strategy": "graph",
            "graph_op": "stats",
            "palace": "palace_a",
            "namespace": "code",
            "room": "default",
            "scope": {"compartment": "iso"},
        }
    )
    result = await memory_service.query(request)
    diags = result.diagnostics
    assert diags.get("entity_count") == 2  # palace_a's alpha + beta only
    assert diags.get("relation_count") == 1  # palace_a's alpha→beta only


async def test_community_refresh_does_not_touch_other_palace(
    memory_service: Any, knowledge_backend: PostgresBackend, two_palaces: dict[str, dict[str, str]]
) -> None:
    """Running community_refresh against palace_a must leave palace_b's
    entities and communities untouched."""
    # Pre-write a community row for palace_b so we can detect deletion
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_communities
            (palace, namespace, room, corridor, name)
        VALUES ('palace_b', 'code', 'default', 'iso', 'pre_existing_b')
        """,
        (),
    )

    from workflows_mcp.engine.memory_service import (
        MemoryRequest,
        MemoryMaintenanceInput,
    )

    request = MemoryRequest.model_validate(
        {
            "operation": "maintain",
            "scope": {
                "palace": "palace_a",
                "wing": "code",
                "room": "default",
                "compartment": "iso",
            },
            "maintenance": {"mode": "community_refresh"},
        }
    )
    await memory_service.execute(request)

    # palace_b's community must still exist
    surviving = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_communities "
        "WHERE palace = 'palace_b' AND name = 'pre_existing_b'",
        (),
    )
    assert surviving.rows[0]["n"] == 1, "community_refresh leaked across palace"


async def test_community_refresh_without_palace_is_rejected(memory_service: Any) -> None:
    """External community_refresh requires palace even if lower scope exists."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    request = MemoryRequest.model_validate(
        {
            "operation": "maintain",
            "scope": {"wing": "code", "room": "default", "compartment": "iso"},
            "maintenance": {"mode": "community_refresh"},
        }
    )
    result = await memory_service.execute(request)
    assert not result.success or "MEM_PALACE_REQUIRED" in (result.error or "")
