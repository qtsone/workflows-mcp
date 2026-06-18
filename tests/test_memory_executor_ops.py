"""Behavior tests for the eight low-level memory executor ops.

Each op is exercised through MemoryService.execute() so the public
envelope (MemoryRequest) is what's contracted, not the internal
ManageMemoryRequest.
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

PALACE = "palace_ops_test"
WING = "default"
CODE_WING = "code"
ROOM = "default"
COMPARTMENT = "ops"


def _scope() -> dict[str, str]:
    return {"palace": PALACE, "wing": WING, "room": ROOM, "compartment": COMPARTMENT}


def _scope_key(
    palace: str = PALACE,
    wing: str = WING,
    room: str = ROOM,
    compartment: str = COMPARTMENT,
) -> str:
    from workflows_mcp.engine.memory_scope_resolver import scope_key as _sk

    return _sk({"palace": palace, "wing": wing, "room": room, "compartment": compartment})


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


@pytest_asyncio.fixture
async def clean_palace(knowledge_backend: PostgresBackend) -> AsyncIterator[None]:
    """Wipe rows belonging to the test palace before and after each test."""

    async def _wipe() -> None:
        # Wipe all test palaces (primary and any cross-palace variants created by tests).
        palace_pattern = f"{PALACE}%"
        await knowledge_backend.execute(
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_memories "
            "WHERE memory_id IN (SELECT id FROM knowledge_memories WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_embeddings "
            "WHERE entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_memories WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entities WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_items WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_sources WHERE palace LIKE $1", (palace_pattern,)
        )
        # Delete topology provenance evidence links before structural evidence (RESTRICT FK).
        await knowledge_backend.execute(
            """
            DELETE FROM knowledge_topology_provenance_evidence
             WHERE provenance_id IN (
                 SELECT id FROM knowledge_topology_provenance WHERE palace LIKE $1
             )
            """,
            (palace_pattern,),
        )
        # Delete topology provenance rows before semantic claims and overrides.
        await knowledge_backend.execute(
            "DELETE FROM knowledge_topology_provenance WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_structural_evidence WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_verification_cycles WHERE palace LIKE $1", (palace_pattern,)
        )
        # Delete corridor edge rows before claims — from_claim_id/to_claim_id FKs use CASCADE
        # but we delete explicitly to keep teardown order safe across schema versions.
        await knowledge_backend.execute(
            """
            DELETE FROM knowledge_semantic_corridors
             WHERE from_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
               OR to_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
            """,
            (palace_pattern,),
        )
        # Delete override rows referencing claims before deleting claims (FK RESTRICT).
        await knowledge_backend.execute(
            """
            DELETE FROM knowledge_semantic_overrides
             WHERE claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
            """,
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_semantic_claims WHERE palace LIKE $1", (palace_pattern,)
        )

    await _wipe()
    yield
    await _wipe()


async def test_clean_palace_starts_empty(
    knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities WHERE palace = $1",
        (PALACE,),
    )
    assert rows.rows[0]["n"] == 0


def test_memory_item_input_schema_exists() -> None:
    """MemoryItemInput must accept item identity + file metadata fields."""
    from workflows_mcp.engine.memory_service import MemoryItemInput

    item = MemoryItemInput.model_validate(
        {
            "id": str(uuid.uuid4()),
            "content_hash": "abc123",
            "size_bytes": 4096,
            "mtime_ns": 1714780800_000_000_000,
            "language": "python",
            "error_metadata": {"reason": "test"},
        }
    )
    assert item.content_hash == "abc123"
    assert item.size_bytes == 4096


def test_memory_record_input_accepts_item_and_embeddings() -> None:
    from workflows_mcp.engine.memory_service import MemoryRecordInput

    rec = MemoryRecordInput.model_validate(
        {
            "format": "structured",
            "item": {"content_hash": "h", "size_bytes": 1, "mtime_ns": 1, "language": "py"},
            "entity_embeddings": [
                {
                    "entity_id": str(uuid.uuid4()),
                    "profile": "embedding",
                    "model": "text-embedding-3-small",
                    "dimension": 1536,
                    "embedding": [0.1] * 1536,
                }
            ],
        }
    )
    assert rec.item is not None
    assert rec.entity_embeddings is not None
    assert len(rec.entity_embeddings[0].embedding) == 1536
    assert rec.entity_embeddings[0].dimension == 1536


@pytest.mark.parametrize(
    "op",
    [
        "ensure_source",
        "ensure_item",
        "store_entities",
        "store_relations",
        "store_memories",
        "store_entity_embeddings",
        "archive_memories",
        "mark_item_dirty",
    ],
)
def test_memory_request_accepts_new_operations(op: str) -> None:
    """MemoryRequest must accept all eight new operation names without raising."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    payload: dict[str, Any] = {
        "operation": op,
        "scope": _scope(),
        "record": {"format": "raw"},
    }
    req = MemoryRequest.model_validate(payload)
    assert req.operation == op


# ---------------------------------------------------------------------------
# Task 4: ensure_source
# ---------------------------------------------------------------------------


async def test_ensure_source_upsert_returns_id(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "ensure_source",
            "scope": _scope(),
            "record": {"format": "raw", "source": "ops-test-source-a"},
        }
    )
    result = await memory_service.execute(req)
    manage = result.manage
    assert manage is not None
    assert manage.success
    first_id = manage.source_id
    assert first_id is not None

    # Second call returns the same id.
    result2 = await memory_service.execute(req)
    assert result2.manage.source_id == first_id

    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_sources WHERE palace = $1 AND name = $2",
        (PALACE, "ops-test-source-a"),
    )
    assert rows.rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# Task 5: ensure_item
# ---------------------------------------------------------------------------


async def test_ensure_item_upsert_updates_metadata(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    base = {
        "operation": "ensure_item",
        "scope": _scope(),
        "record": {
            "format": "raw",
            "source": "ops-test-source-b",
            "path": "src/lib/foo.py",
            "item": {
                "content_hash": "h1",
                "size_bytes": 100,
                "mtime_ns": 1,
                "language": "python",
            },
        },
    }
    r1 = await memory_service.execute(MemoryRequest.model_validate(base))
    assert r1.manage.success
    item_id = r1.manage.item_id
    assert item_id is not None

    # Update with a new hash + size; same (palace, source, path) keeps the same id.
    import copy

    payload2 = copy.deepcopy(base)
    payload2["record"]["item"]["content_hash"] = "h2"
    payload2["record"]["item"]["size_bytes"] = 200
    r2 = await memory_service.execute(MemoryRequest.model_validate(payload2))
    assert r2.manage.item_id == item_id

    rows = await knowledge_backend.query(
        "SELECT palace, content_hash, size_bytes FROM knowledge_items WHERE id = $1::uuid",
        (item_id,),
    )
    assert rows.rows[0]["palace"] == PALACE
    assert rows.rows[0]["content_hash"] == "h2"
    assert rows.rows[0]["size_bytes"] == 200


async def test_ensure_item_trigger_rejects_cross_palace_source(
    knowledge_backend, clean_palace
) -> None:
    """The DB trigger rejects an item whose palace differs from its source palace."""
    other_palace = f"{PALACE}_other"
    source = await knowledge_backend.query(
        """
        INSERT INTO knowledge_sources (palace, name, source_type)
        VALUES ($1, 'ops-test-cross-palace-source', 'FILE')
        RETURNING id
        """,
        (other_palace,),
    )
    source_id = str(source.rows[0]["id"])

    with pytest.raises(Exception, match="KNOWLEDGE_ITEM_SOURCE_PALACE_MISMATCH"):
        await knowledge_backend.execute(
            """
            INSERT INTO knowledge_items
                (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ($1, $2::uuid, 'src/mismatch.py', 'mismatch.py', 'h', 1, 1)
            """,
            (PALACE, source_id),
        )

    await knowledge_backend.execute(
        "DELETE FROM knowledge_sources WHERE palace = $1", (other_palace,)
    )


async def test_ensure_item_rejects_missing_not_null_fields(memory_service, clean_palace) -> None:
    """ensure_item must return MEM_FIELD_REQUIRED when any NOT NULL item field is absent."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "ensure_item",
                "scope": _scope(),
                "record": {
                    "format": "raw",
                    "source": "ops-test-source-b2",
                    "path": "src/lib/missing.py",
                    "item": {"language": "python"},
                },
            }
        )
    )
    assert not result.manage.success
    assert "MEM_FIELD_REQUIRED" in (result.manage.error or "")
    assert "content_hash" in (result.manage.error or "")


# ---------------------------------------------------------------------------
# Task 6: store_entities
# ---------------------------------------------------------------------------


async def test_store_entities_bulk_upsert_idempotent(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "ensure_item",
                "scope": _scope(),
                "record": {
                    "format": "raw",
                    "source": "ops-test-source-c",
                    "path": "src/mod.py",
                    "item": {
                        "content_hash": "h",
                        "size_bytes": 1,
                        "mtime_ns": 1,
                        "language": "python",
                    },
                },
            }
        )
    )
    item_row = await knowledge_backend.query(
        "SELECT ki.id FROM knowledge_items ki "
        "JOIN knowledge_sources ks ON ks.id = ki.source_id "
        "WHERE ki.palace = $1 AND ks.palace = $1 "
        "AND ks.name = 'ops-test-source-c' AND ki.path = 'src/mod.py'",
        (PALACE,),
    )
    item_id = str(item_row.rows[0]["id"])

    payload = {
        "operation": "store_entities",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "source": "STRUCTURAL",
            "item": {"id": item_id},
            "entities": [
                {
                    "entity_type": "Function",
                    "name": "alpha",
                    "stable_id": "src/mod.py::alpha",
                    "metadata": {"start_line": 10, "end_line": 20},
                },
                {
                    "entity_type": "Function",
                    "name": "beta",
                    "stable_id": "src/mod.py::beta",
                    "metadata": {"start_line": 30, "end_line": 40},
                },
            ],
        },
    }
    r1 = await memory_service.execute(MemoryRequest.model_validate(payload))
    assert r1.manage.success
    assert len(r1.manage.entity_ids) == 2

    # Idempotent: second call returns the same ids.
    r2 = await memory_service.execute(MemoryRequest.model_validate(payload))
    assert sorted(r2.manage.entity_ids) == sorted(r1.manage.entity_ids)

    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities WHERE palace = $1 AND source = $2",
        (PALACE, "STRUCTURAL"),
    )
    assert rows.rows[0]["n"] == 2


# ---------------------------------------------------------------------------
# Task 7: store_relations
# ---------------------------------------------------------------------------


async def test_store_relations_bulk_insert(memory_service, knowledge_backend, clean_palace) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    seed_payload = {
        "operation": "store_entities",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "source": "STRUCTURAL",
            "entities": [
                {"entity_type": "Function", "name": "a", "stable_id": "f::a"},
                {"entity_type": "Function", "name": "b", "stable_id": "f::b"},
            ],
        },
    }
    seeded = await memory_service.execute(MemoryRequest.model_validate(seed_payload))
    a_id, b_id = seeded.manage.entity_ids

    rel_payload = {
        "operation": "store_relations",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "relations": [
                {"source_entity_id": a_id, "target_entity_id": b_id, "relation_type": "CALLS"}
            ],
        },
    }
    result = await memory_service.execute(MemoryRequest.model_validate(rel_payload))
    assert result.manage.success
    assert len(result.manage.relation_ids) == 1

    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations "
        "WHERE source_entity_id = $1::uuid AND target_entity_id = $2::uuid",
        (a_id, b_id),
    )
    assert rows.rows[0]["n"] == 1


async def test_store_relations_rejects_cross_palace_endpoint(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    own = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [{"entity_type": "Function", "name": "own", "stable_id": "own::f"}],
                },
            }
        )
    )
    foreign = await knowledge_backend.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES ($1, 'code', 'default', 'ops', 'Function', 'foreign', 'STRUCTURAL', 'foreign::f')
        RETURNING id
        """,
        (f"{PALACE}_foreign",),
    )

    from workflows_mcp.engine.memory_errors import MemoryContractError

    with pytest.raises(MemoryContractError, match="MEM_PALACE_MISMATCH"):
        await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "store_relations",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "relations": [
                            {
                                "source_entity_id": own.manage.entity_ids[0],
                                "target_entity_id": str(foreign.rows[0]["id"]),
                                "relation_type": "CALLS",
                            }
                        ],
                    },
                }
            )
        )


# ---------------------------------------------------------------------------
# Task 8: store_memories
# ---------------------------------------------------------------------------


async def test_store_memories_anchored_to_entity(
    memory_service, knowledge_backend, clean_palace, monkeypatch
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    seeded = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [{"entity_type": "Function", "name": "f", "stable_id": "x::f"}],
                },
            }
        )
    )
    entity_id = seeded.manage.entity_ids[0]

    async def fake_compute_embedding(  # noqa: E501
        *args: Any, **kwargs: Any
    ) -> tuple[list[float], str, int, dict[str, Any]]:
        return [0.0] * 1536, "test-embedding-model", 1536, {}

    monkeypatch.setattr(
        "workflows_mcp.engine.memory_service.compute_embedding",
        fake_compute_embedding,
    )

    payload = {
        "operation": "store_memories",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "memories": [
                {
                    "content": "f returns 1 unconditionally",
                    "anchor_entity_id": entity_id,
                    "anchor_kind": "symbol",
                    "start_line": 10,
                    "end_line": 12,
                    "start_col": 0,
                    "end_col": 4,
                }
            ],
        },
    }
    result = await memory_service.execute(MemoryRequest.model_validate(payload))
    assert result.manage.success
    assert len(result.manage.memory_ids) == 1
    memory_id = result.manage.memory_ids[0]

    link = await knowledge_backend.query(
        """
        SELECT entity_id, start_line, end_line, start_col, end_col, anchor_kind
          FROM knowledge_entity_memories
         WHERE memory_id = $1::uuid
        """,
        (memory_id,),
    )
    assert len(link.rows) == 1
    row = link.rows[0]
    assert str(row["entity_id"]) == entity_id
    assert row["start_line"] == 10
    assert row["end_line"] == 12
    assert row["anchor_kind"] == "symbol"


async def test_store_memories_rejects_code_wing(memory_service, clean_palace, monkeypatch) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    async def fake_compute_embedding(  # noqa: E501
        *args: Any, **kwargs: Any
    ) -> tuple[list[float], str, int, dict[str, Any]]:
        return [0.0] * 1536, "test-embedding-model", 1536, {}

    monkeypatch.setattr(  # noqa: E501
        "workflows_mcp.engine.memory_service.compute_embedding", fake_compute_embedding
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_memories",
                "scope": {
                    "palace": PALACE,
                    "wing": CODE_WING,
                    "room": ROOM,
                    "compartment": COMPARTMENT,
                },
                "record": {
                    "format": "structured",
                    "memories": [{"content": "do not write to code wing"}],
                },
            }
        )
    )
    assert not result.manage.success
    assert "MEM_RESERVED_WING" in (result.manage.error or "")


async def test_store_memories_rejects_cross_palace_anchor(
    memory_service, knowledge_backend, clean_palace, monkeypatch
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    async def fake_compute_embedding(  # noqa: E501
        *args: Any, **kwargs: Any
    ) -> tuple[list[float], str, int, dict[str, Any]]:
        return [0.0] * 1536, "test-embedding-model", 1536, {}

    monkeypatch.setattr(  # noqa: E501
        "workflows_mcp.engine.memory_service.compute_embedding", fake_compute_embedding
    )

    foreign = await knowledge_backend.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES ($1, 'code', 'default', 'ops', 'Function', 'foreign_anchor',
                'STRUCTURAL', 'foreign::anchor')
        RETURNING id
        """,
        (f"{PALACE}_foreign",),
    )

    from workflows_mcp.engine.memory_errors import MemoryContractError

    with pytest.raises(MemoryContractError, match="MEM_PALACE_MISMATCH"):
        await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "store_memories",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "memories": [
                            {
                                "content": "bad anchor",
                                "anchor_entity_id": str(foreign.rows[0]["id"]),
                                "anchor_kind": "symbol",
                            }
                        ],
                    },
                }
            )
        )


# ---------------------------------------------------------------------------
# Task 9: store_entity_embeddings
# ---------------------------------------------------------------------------


async def test_store_entity_embeddings_upsert(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    seeded = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [{"entity_type": "Function", "name": "g", "stable_id": "x::g"}],
                },
            }
        )
    )
    entity_id = seeded.manage.entity_ids[0]

    vec_v1 = [0.1] * 1536
    vec_v2 = [0.2] * 1536

    payload = {
        "operation": "store_entity_embeddings",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "entity_embeddings": [
                {
                    "entity_id": entity_id,
                    "profile": "embedding",
                    "model": "text-embedding-3-small",
                    "dimension": 1536,
                    "embedding": vec_v1,
                }
            ],
        },
    }
    r1 = await memory_service.execute(MemoryRequest.model_validate(payload))
    assert r1.manage.success

    import copy

    payload2 = copy.deepcopy(payload)
    payload2["record"]["entity_embeddings"][0]["embedding"] = vec_v2
    r2 = await memory_service.execute(MemoryRequest.model_validate(payload2))
    assert r2.manage.success

    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entity_embeddings "
        "WHERE entity_id = $1::uuid AND profile = 'embedding'",
        (entity_id,),
    )
    assert rows.rows[0]["n"] == 1


async def test_store_entity_embeddings_rejects_dimension_mismatch(
    memory_service, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    seeded = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [{"entity_type": "Function", "name": "dim", "stable_id": "dim::f"}],
                },
            }
        )
    )
    from workflows_mcp.engine.memory_errors import MemoryContractError

    with pytest.raises(MemoryContractError, match="MEM_EMBEDDING_DIMENSION_MISMATCH"):
        await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "store_entity_embeddings",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "entity_embeddings": [
                            {
                                "entity_id": seeded.manage.entity_ids[0],
                                "profile": "embedding",
                                "model": "text-embedding-3-small",
                                "dimension": 1536,
                                "embedding": [0.1] * 3,
                            }
                        ],
                    },
                }
            )
        )


async def test_store_entity_embeddings_rejects_cross_palace_entity(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    foreign = await knowledge_backend.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES ($1, 'code', 'default', 'ops', 'Function', 'foreign_embedding',
                'STRUCTURAL', 'foreign::embedding')
        RETURNING id
        """,
        (f"{PALACE}_foreign",),
    )
    from workflows_mcp.engine.memory_errors import MemoryContractError

    with pytest.raises(MemoryContractError, match="MEM_PALACE_MISMATCH"):
        await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "store_entity_embeddings",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "entity_embeddings": [
                            {
                                "entity_id": str(foreign.rows[0]["id"]),
                                "profile": "embedding",
                                "model": "text-embedding-3-small",
                                "dimension": 3,
                                "embedding": [0.1] * 3,
                            }
                        ],
                    },
                }
            )
        )


# ---------------------------------------------------------------------------
# Task 10: archive_memories
# ---------------------------------------------------------------------------


async def test_archive_memories_by_item(memory_service, knowledge_backend, clean_palace) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "ensure_item",
                "scope": _scope(),
                "record": {
                    "format": "raw",
                    "source": "ops-test-source-g",
                    "path": "to/archive.py",
                    "item": {
                        "content_hash": "h",
                        "size_bytes": 1,
                        "mtime_ns": 1,
                        "language": "python",
                    },
                },
            }
        )
    )
    item_row = await knowledge_backend.query(
        "SELECT ki.id FROM knowledge_items ki "
        "JOIN knowledge_sources ks ON ks.id = ki.source_id "
        "WHERE ki.palace = $1 AND ks.palace = $1 "
        "AND ks.name = 'ops-test-source-g' AND ki.path = 'to/archive.py'",
        (PALACE,),
    )
    item_id = str(item_row.rows[0]["id"])

    memory_id = str(uuid.uuid4())
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_memories
            (id, item_id, content, embedding,
             authority, lifecycle_state, confidence, embedding_model,
             metadata, created_by, auth_method,
             palace, namespace, room, corridor,
             memory_tier, derived_kind, parent_memory_ids)
        VALUES
            ($1::uuid, $2::uuid, 'doomed', ($3)::vector,
             'AGENT', 'ACTIVE', 0.9, 'noop',
             '{}'::jsonb, $4::uuid, 'TEST',
             $5, $6, $7, $8,
             'direct', NULL, '{}'::uuid[])
        """,
        (
            memory_id,
            item_id,
            "[" + ",".join(["0.0"] * 1536) + "]",
            str(uuid.uuid4()),
            PALACE,
            WING,
            ROOM,
            COMPARTMENT,
        ),
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "archive_memories",
                "scope": _scope(),
                "record": {"format": "raw", "item": {"id": item_id}},
            }
        )
    )
    assert result.manage.success
    assert result.manage.stored_count == 1

    state = await knowledge_backend.query(
        "SELECT lifecycle_state FROM knowledge_memories WHERE id = $1::uuid",
        (memory_id,),
    )
    assert state.rows[0]["lifecycle_state"] == "ARCHIVED"

    # Idempotent: second call reports zero affected.
    again = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "archive_memories",
                "scope": _scope(),
                "record": {"format": "raw", "item": {"id": item_id}},
            }
        )
    )
    assert again.manage.success
    assert again.manage.stored_count == 0


async def test_archive_memories_rejects_item_in_different_palace(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    foreign_palace = f"{PALACE}_foreign"
    src = await knowledge_backend.query(
        "INSERT INTO knowledge_sources "
        "(palace, name, source_type) VALUES ($1, 'archive-foreign', 'FILE') RETURNING id",
        (foreign_palace,),
    )
    item = await knowledge_backend.query(
        """
        INSERT INTO knowledge_items
            (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
        VALUES ($1, $2::uuid, 'foreign.py', 'foreign.py', 'h', 1, 1)
        RETURNING id
        """,
        (foreign_palace, str(src.rows[0]["id"])),
    )

    from workflows_mcp.engine.memory_errors import MemoryContractError

    with pytest.raises(MemoryContractError, match="MEM_PALACE_MISMATCH"):
        await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "archive_memories",
                    "scope": _scope(),
                    "record": {"format": "raw", "item": {"id": str(item.rows[0]["id"])}},
                }
            )
        )


# ---------------------------------------------------------------------------
# Task 11: mark_item_dirty
# ---------------------------------------------------------------------------


async def test_mark_item_dirty_sets_lifecycle_and_error(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "ensure_item",
                "scope": _scope(),
                "record": {
                    "format": "raw",
                    "source": "ops-test-source-h",
                    "path": "to/dirty.py",
                    "item": {
                        "content_hash": "h",
                        "size_bytes": 1,
                        "mtime_ns": 1,
                        "language": "python",
                    },
                },
            }
        )
    )
    item_row = await knowledge_backend.query(
        "SELECT ki.id FROM knowledge_items ki "
        "JOIN knowledge_sources ks ON ks.id = ki.source_id "
        "WHERE ki.palace = $1 AND ks.palace = $1 "
        "AND ks.name = 'ops-test-source-h' AND ki.path = 'to/dirty.py'",
        (PALACE,),
    )
    item_id = str(item_row.rows[0]["id"])

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "mark_item_dirty",
                "scope": _scope(),
                "record": {
                    "format": "raw",
                    "item": {
                        "id": item_id,
                        "error_metadata": {"reason": "embedding too long", "phase": "system2"},
                    },
                },
            }
        )
    )
    assert result.manage.success

    state = await knowledge_backend.query(
        "SELECT lifecycle_state, error_metadata FROM knowledge_items WHERE id = $1::uuid",
        (item_id,),
    )
    row = state.rows[0]
    assert row["lifecycle_state"] == "DIRTY"
    err = row["error_metadata"]
    if isinstance(err, str):
        import json as _json

        err = _json.loads(err)
    assert err["reason"] == "embedding too long"
    assert err["phase"] == "system2"


async def test_mark_item_dirty_rejects_cross_palace_item(
    memory_service, knowledge_backend, clean_palace
) -> None:
    from workflows_mcp.engine.memory_service import MemoryRequest

    foreign_palace = f"{PALACE}_foreign_dirty"
    src = await knowledge_backend.query(
        "INSERT INTO knowledge_sources "
        "(palace, name, source_type) VALUES ($1, 'dirty-foreign', 'FILE') RETURNING id",
        (foreign_palace,),
    )
    item = await knowledge_backend.query(
        """
        INSERT INTO knowledge_items
            (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
        VALUES ($1, $2::uuid, 'foreign_dirty.py', 'foreign_dirty.py', 'h', 1, 1)
        RETURNING id
        """,
        (foreign_palace, str(src.rows[0]["id"])),
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "mark_item_dirty",
                "scope": _scope(),
                "record": {
                    "format": "raw",
                    "item": {
                        "id": str(item.rows[0]["id"]),
                        "error_metadata": {"reason": "cross palace test"},
                    },
                },
            }
        )
    )
    # Should fail — item belongs to a different palace
    assert not result.manage.success
    assert "MEM_PALACE_MISMATCH" in (result.manage.error or "")


# ---------------------------------------------------------------------------
# Track 4 prereqs (v14): qualified_name, parent_class_id, relations.metadata
# ---------------------------------------------------------------------------


async def test_store_entities_persists_qualified_name_and_parent_class_id(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """store_entities must persist qualified_name and parent_class_id columns."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Seed parent Class entity first.
    parent_resp = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [
                        {
                            "entity_type": "Class",
                            "name": "MyClass",
                            "stable_id": "src/m.py::MyClass",
                            "qualified_name": "src.m.MyClass",
                        }
                    ],
                },
            }
        )
    )
    assert parent_resp.manage.success
    parent_id = parent_resp.manage.entity_ids[0]

    # Insert Method that points at parent_class_id.
    method_resp = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [
                        {
                            "entity_type": "Method",
                            "name": "do_thing",
                            "stable_id": "src/m.py::MyClass.do_thing",
                            "qualified_name": "src.m.MyClass.do_thing",
                            "parent_class_id": parent_id,
                        }
                    ],
                },
            }
        )
    )
    assert method_resp.manage.success
    method_id = method_resp.manage.entity_ids[0]

    rows = await knowledge_backend.query(
        "SELECT id, qualified_name, parent_class_id FROM knowledge_entities WHERE id = $1::uuid",
        (method_id,),
    )
    assert rows.rows[0]["qualified_name"] == "src.m.MyClass.do_thing"
    assert str(rows.rows[0]["parent_class_id"]) == parent_id


async def test_store_entities_upsert_updates_qualified_name_and_parent(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """Upsert path (ON CONFLICT) must refresh qualified_name and parent_class_id."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Seed two Class entities to swap parents between.
    classes = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [
                        {
                            "entity_type": "Class",
                            "name": "A",
                            "stable_id": "f::A",
                            "qualified_name": "f.A",
                        },
                        {
                            "entity_type": "Class",
                            "name": "B",
                            "stable_id": "f::B",
                            "qualified_name": "f.B",
                        },
                    ],
                },
            }
        )
    )
    a_id, b_id = classes.manage.entity_ids

    # First insert with parent A.
    payload = {
        "operation": "store_entities",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "source": "STRUCTURAL",
            "entities": [
                {
                    "entity_type": "Method",
                    "name": "m",
                    "stable_id": "f::A.m",
                    "qualified_name": "f.A.m",
                    "parent_class_id": a_id,
                }
            ],
        },
    }
    first = await memory_service.execute(MemoryRequest.model_validate(payload))
    method_id = first.manage.entity_ids[0]

    # Re-upsert with parent B and a renamed qualified_name.
    payload["record"]["entities"][0]["parent_class_id"] = b_id
    payload["record"]["entities"][0]["qualified_name"] = "f.B.m"
    second = await memory_service.execute(MemoryRequest.model_validate(payload))
    assert second.manage.entity_ids[0] == method_id  # same row

    rows = await knowledge_backend.query(
        "SELECT qualified_name, parent_class_id FROM knowledge_entities WHERE id = $1::uuid",
        (method_id,),
    )
    assert rows.rows[0]["qualified_name"] == "f.B.m"
    assert str(rows.rows[0]["parent_class_id"]) == b_id


async def test_store_relations_persists_metadata(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """store_relations must persist per-edge metadata JSONB."""
    import json as _json

    from workflows_mcp.engine.memory_service import MemoryRequest

    seeded = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [
                        {"entity_type": "Function", "name": "a", "stable_id": "f::a"},
                        {"entity_type": "Function", "name": "b", "stable_id": "f::b"},
                    ],
                },
            }
        )
    )
    a_id, b_id = seeded.manage.entity_ids

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "relations": [
                        {
                            "source_entity_id": a_id,
                            "target_entity_id": b_id,
                            "relation_type": "CALLS",
                            "metadata": {"resolution": "unresolved", "call_site_line": 42},
                        }
                    ],
                },
            }
        )
    )
    assert result.manage.success
    rel_id = result.manage.relation_ids[0]

    rows = await knowledge_backend.query(
        "SELECT metadata FROM knowledge_relations WHERE id = $1::uuid",
        (rel_id,),
    )
    raw = rows.rows[0]["metadata"]
    meta = raw if isinstance(raw, dict) else _json.loads(raw)
    assert meta == {"resolution": "unresolved", "call_site_line": 42}


async def test_store_relations_metadata_defaults_to_empty_object(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """Omitting metadata must default to empty JSONB object (NOT NULL column)."""
    import json as _json

    from workflows_mcp.engine.memory_service import MemoryRequest

    seeded = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [
                        {"entity_type": "Function", "name": "x", "stable_id": "f::x"},
                        {"entity_type": "Function", "name": "y", "stable_id": "f::y"},
                    ],
                },
            }
        )
    )
    x_id, y_id = seeded.manage.entity_ids

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "relations": [
                        {
                            "source_entity_id": x_id,
                            "target_entity_id": y_id,
                            "relation_type": "CALLS",
                        }
                    ],
                },
            }
        )
    )
    assert result.manage.success
    rel_id = result.manage.relation_ids[0]

    rows = await knowledge_backend.query(
        "SELECT metadata FROM knowledge_relations WHERE id = $1::uuid",
        (rel_id,),
    )
    raw = rows.rows[0]["metadata"]
    meta = raw if isinstance(raw, dict) else _json.loads(raw)
    assert meta == {}


# ---------------------------------------------------------------------------
# ADR-013 System 1 / System 2 operations (Task 1 RED tests)
# These tests fail until the new operations are added to MemoryRequest and
# ManageMemoryRequest and the corresponding service methods are implemented.
# ---------------------------------------------------------------------------


def test_system1_store_structural_evidence_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'store_system1_structural_evidence' without raising."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_evidence",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "store_system1_structural_evidence"


def test_system1_record_verification_cycle_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'record_system1_verification_cycle' without raising."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "record_system1_verification_cycle",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "record_system1_verification_cycle"


def test_system2_derive_semantic_claims_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'derive_system2_semantic_claims' without raising."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "derive_system2_semantic_claims",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "derive_system2_semantic_claims"


def test_system2_apply_semantic_override_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'apply_semantic_override' without raising."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "apply_semantic_override",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "apply_semantic_override"


def test_system2_reconcile_semantic_lifecycle_operation_accepted_by_request() -> None:
    """MemoryRequest must accept 'reconcile_semantic_lifecycle' without raising."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "reconcile_semantic_lifecycle",
            "scope": _scope(),
            "record": {"format": "structured"},
        }
    )
    assert req.operation == "reconcile_semantic_lifecycle"


async def test_system1_store_structural_evidence_persists_evidence_rows(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """store_system1_structural_evidence must write structural evidence rows keyed by
    scope/entity and return success with stored evidence IDs."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/mod.py::MyClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/mod.py", "line": 1},
                        },
                        {
                            "entity_stable_id": "src/mod.py::MyClass.method",
                            "entity_type": "function",
                            "evidence_category": "structural_function",
                            "evidence_data": {"file": "src/mod.py", "line": 10},
                        },
                    ],
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.stored_count >= 2
    assert result.manage.stored_evidence_ids is not None
    assert len(result.manage.stored_evidence_ids) == 2, (
        "Expected exactly 2 stored evidence IDs, one per submitted item"
    )


@pytest.mark.asyncio
async def test_system1_store_structural_evidence_row_exists_in_db(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """store_system1_structural_evidence must write a real row to the DB.
    Querying knowledge_structural_evidence after the call must return exactly
    one row matching the submitted scope/entity/category key.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/db_test.py::DbCheck",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/db_test.py", "line": 5},
                        },
                    ],
                },
            }
        )
    )

    rows = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1
           AND wing = $2
           AND room = $3
           AND compartment = $4
           AND entity_stable_id = $5
           AND evidence_category = $6
        """,
        (PALACE, WING, ROOM, COMPARTMENT, "src/db_test.py::DbCheck", "structural_class"),
    )
    assert len(rows.rows) == 1, (
        f"Expected exactly 1 DB row in knowledge_structural_evidence,"
        f" got {len(rows.rows)}: {rows.rows!r}"
    )


async def test_system1_record_verification_cycle_persists_cycle_metadata(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """record_system1_verification_cycle must persist a verification cycle row
    with scope_key identity and success/failure status."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "record_system1_verification_cycle",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "verification_cycle": {
                        "success": True,
                        "covered_scope": {
                            "palace": PALACE,
                            "wing": WING,
                            "room": ROOM,
                            "compartment": COMPARTMENT,
                        },
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.cycle_id is not None


async def test_system2_derive_semantic_claims_returns_claim_ids(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must derive claims backed by System 1 evidence
    and return claim IDs in the response."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Pre-store structural evidence so the derivation can resolve stable IDs.
    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/mod.py::MyClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/mod.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "data_access_layer",
                        "evidence_entity_stable_ids": ["src/mod.py::MyClass"],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1


async def test_system2_apply_semantic_override_activates_immediately_with_provenance(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """apply_semantic_override must activate the override immediately and persist
    provenance fields (override_reason, overridden_by, activated_at)."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    # First, store structural evidence so a real claim can be derived.
    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/override_test.py::OverrideClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/override_test.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )

    # Derive a claim to obtain a real claim_id (FK constraint on overrides table).
    derive_result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "data_access_layer",
                        "evidence_entity_stable_ids": ["src/override_test.py::OverrideClass"],
                    },
                },
            }
        )
    )
    assert derive_result.manage is not None
    assert derive_result.manage.claim_ids and len(derive_result.manage.claim_ids) >= 1
    real_claim_id = derive_result.manage.claim_ids[0]

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "apply_semantic_override",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "override": {
                        "claim_id": real_claim_id,
                        "override_reason": "manual correction by architect",
                        "overridden_by": "alice",
                        "new_lifecycle_state": "active_evidenced",
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.override_id is not None


async def test_system2_reconcile_semantic_lifecycle_returns_transition_summary(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """reconcile_semantic_lifecycle must evaluate lifecycle transitions and return
    a summary of affected claims and their new states."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "reconcile_semantic_lifecycle",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "lifecycle_reconciliation": {
                        "scope_key": _scope_key(),
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success
    assert result.manage.reconciled_count is not None


@pytest.mark.asyncio
async def test_system1_failed_verification_cycle_does_not_count_for_archive_gate(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """ADR-013: only *successful* absent-evidence verification cycles count toward
    the archive eligibility gate. Recording two cycles with success=False and then
    attempting a force-archive must still be rejected with MEM_ARCHIVE_GATE_NOT_MET.

    This is a runtime service invariant — failed cycles must not satisfy the gate
    even when two cycle IDs are technically present.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    scope_key = _scope_key()

    # Record two verification cycles, both with success=False (i.e., evidence was
    # found — the absence condition was NOT met).
    cycle_ids: list[str] = []
    for _ in range(2):
        cycle_result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "record_system1_verification_cycle",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "verification_cycle": {
                            "success": False,  # Evidence still present — absence NOT confirmed.
                            "covered_scope": {
                                "palace": PALACE,
                                "wing": WING,
                                "room": ROOM,
                                "compartment": COMPARTMENT,
                            },
                        },
                    },
                }
            )
        )
        # The operation itself succeeds (cycle is recorded), but the cycle is
        # marked as failed (absence not confirmed). Collect the cycle IDs.
        assert cycle_result.manage is not None
        assert cycle_result.manage.cycle_id is not None
        cycle_ids.append(cycle_result.manage.cycle_id)

    # Now attempt to archive using those two failed-cycle IDs as proof.
    # The service must reject this with MEM_ARCHIVE_GATE_NOT_MET because the
    # cycles did not confirm absence of evidence.
    try:
        result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "reconcile_semantic_lifecycle",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "lifecycle_reconciliation": {
                            "scope_key": scope_key,
                            "force_archive_claim_ids": ["claim-from-test"],
                            "absent_verification_cycle_ids": cycle_ids,
                        },
                    },
                }
            )
        )
        assert result.manage is not None, (
            "Archive using failed verification cycles must not succeed silently; "
            "expected MEM_ARCHIVE_GATE_NOT_MET"
        )
        assert not result.manage.success, (
            "Archive gate must reject force-archive backed only by failed verification cycles"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or ""), (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET", (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {exc.code!r}"
        )


@pytest.mark.asyncio
async def test_system1_store_structural_evidence_is_idempotent_for_same_scope_entity_category(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """ADR-013 Task 4: storing the same structural evidence (same scope +
    entity_stable_id + evidence_category) twice must be idempotent.

    Semantics locked in by this test:
    - stored_count: number of submitted items processed (insert or upsert).
      It is NOT a count of newly inserted rows.  Callers use it to confirm
      every submitted item reached the DB, whether new or refreshed.
    - stored_evidence_ids: the UUIDs of the persisted rows, stable across
      calls for the same key — the second call returns the same row UUID
      as the first, proving ON CONFLICT DO UPDATE hit the existing row.
    - DB row count: exactly 1 row must exist in knowledge_structural_evidence
      for the conflict key after two store calls, proving no duplicate was
      created.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    evidence_payload = {
        "operation": "store_system1_structural_evidence",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "structural_evidence": [
                {
                    "entity_stable_id": "src/idempotent.py::MyClass",
                    "entity_type": "class",
                    "evidence_category": "structural_class",
                    "evidence_data": {"file": "src/idempotent.py", "line": 1},
                },
            ],
        },
    }

    first = await memory_service.execute(MemoryRequest.model_validate(evidence_payload))
    assert first.manage is not None
    assert first.manage.success
    # stored_count == number of submitted items (1), not number of new rows.
    assert first.manage.stored_count == 1
    first_ids = first.manage.stored_evidence_ids
    assert first_ids is not None and len(first_ids) == 1

    second = await memory_service.execute(MemoryRequest.model_validate(evidence_payload))
    assert second.manage is not None
    assert second.manage.success
    # Second call: still 1 item submitted, still reports stored_count == 1.
    assert second.manage.stored_count == 1
    second_ids = second.manage.stored_evidence_ids
    assert second_ids is not None and len(second_ids) == 1

    # Row UUID must be identical — ON CONFLICT returned the existing row id.
    assert first_ids == second_ids, (
        "Idempotency violation: re-storing the same structural evidence"
        f" returned different IDs: first={first_ids!r}, second={second_ids!r}"
    )

    # Direct DB assertion: exactly 1 row for the conflict key, no duplicate.
    rows = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1
           AND wing = $2
           AND room = $3
           AND compartment = $4
           AND entity_stable_id = $5
           AND evidence_category = $6
        """,
        (
            PALACE,
            WING,
            ROOM,
            COMPARTMENT,
            "src/idempotent.py::MyClass",
            "structural_class",
        ),
    )
    assert len(rows.rows) == 1, (
        f"Expected exactly 1 DB row after two idempotent store calls,"
        f" got {len(rows.rows)}: {rows.rows!r}"
    )
    # The DB row UUID must match the returned evidence ID.
    assert str(rows.rows[0]["id"]) == first_ids[0], (
        "DB row UUID does not match the ID returned by store operation: "
        f"db={rows.rows[0]['id']!r}, returned={first_ids[0]!r}"
    )


@pytest.mark.asyncio
async def test_archive_gate_rejects_unregistered_fabricated_cycle_ids(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """ADR-013 archive gate must be fail-closed: cycle IDs that were never registered
    via record_system1_verification_cycle must NOT count as successful absent-evidence
    cycles.  Passing two fabricated/unregistered IDs must be rejected with
    MEM_ARCHIVE_GATE_NOT_MET — unknown IDs are not countable.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    scope_key = _scope_key()
    # Fabricated UUIDs — never registered with the service (not in DB).
    fake_cycle_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    try:
        result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "reconcile_semantic_lifecycle",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "lifecycle_reconciliation": {
                            "scope_key": scope_key,
                            "force_archive_claim_ids": ["claim-fabricated"],
                            "absent_verification_cycle_ids": fake_cycle_ids,
                        },
                    },
                }
            )
        )
        assert result.manage is not None, (
            "Archive with fabricated cycle IDs must not succeed silently; "
            "expected MEM_ARCHIVE_GATE_NOT_MET"
        )
        assert not result.manage.success, (
            "Archive gate must reject force-archive backed by unregistered cycle IDs"
        )
        assert "MEM_ARCHIVE_GATE_NOT_MET" in (result.manage.error or ""), (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {result.manage.error!r}"
        )
    except MemoryContractError as exc:
        assert exc.code == "MEM_ARCHIVE_GATE_NOT_MET", (
            f"Expected MEM_ARCHIVE_GATE_NOT_MET, got: {exc.code!r}"
        )


# ---------------------------------------------------------------------------
# Task 7: System 2 claim derivation — room intent, reasoning unit, corridors
# ---------------------------------------------------------------------------


async def _store_evidence(memory_service: Any, stable_id: str) -> None:
    """Helper: store one structural evidence row in the test palace/scope."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": stable_id,
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/t7.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )
    assert result.manage is not None and result.manage.success, (
        f"Pre-condition failed: could not store evidence for {stable_id!r}"
    )


@pytest.mark.asyncio
async def test_system2_room_intent_label_persists_claim_and_evidence_link(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must persist a room_intent claim backed by
    at least one evidence link.  The returned claim ID must map to a real DB row
    in knowledge_semantic_claims with claim_type='room_intent' and
    lifecycle_state='active_evidenced'.  A matching knowledge_claim_evidence_links
    row must also exist."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    stable_id = "src/t7_room.py::RoomClass"
    await _store_evidence(memory_service, stable_id)

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "data_access_layer",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1
    claim_id = result.manage.claim_ids[0]

    # Must be a real UUID (not a stub hash).
    try:
        uuid.UUID(claim_id)
    except ValueError:
        pytest.fail(f"claim_id is not a UUID: {claim_id!r}")

    # DB row must exist with correct type and lifecycle.
    claim_row = await knowledge_backend.query(
        "SELECT claim_type, lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert len(claim_row.rows) == 1, f"No DB row for claim_id={claim_id!r}"
    assert claim_row.rows[0]["claim_type"] == "room_intent"
    assert claim_row.rows[0]["lifecycle_state"] == "active_evidenced"

    # Evidence link must exist.
    link_row = await knowledge_backend.query(
        "SELECT count(*) AS n FROM knowledge_claim_evidence_links WHERE claim_id = $1::uuid",
        (claim_id,),
    )
    assert int(link_row.rows[0]["n"]) >= 1, (
        f"Expected at least 1 evidence link for claim {claim_id!r}, got 0"
    )


@pytest.mark.asyncio
async def test_system2_compartment_reasoning_unit_persists_claim_and_evidence_link(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must persist a compartment_reasoning_unit claim
    backed by at least one evidence link when compartment_reasoning_unit is supplied."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    stable_id = "src/t7_comp.py::CompClass"
    await _store_evidence(memory_service, stable_id)

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "compartment_reasoning_unit": "query_builder",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1
    claim_id = result.manage.claim_ids[0]

    uuid.UUID(claim_id)  # must be a real UUID

    claim_row = await knowledge_backend.query(
        "SELECT claim_type, lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert len(claim_row.rows) == 1, f"No DB row for claim_id={claim_id!r}"
    assert claim_row.rows[0]["claim_type"] == "compartment_reasoning_unit"
    assert claim_row.rows[0]["lifecycle_state"] == "active_evidenced"

    link_row = await knowledge_backend.query(
        "SELECT count(*) AS n FROM knowledge_claim_evidence_links WHERE claim_id = $1::uuid",
        (claim_id,),
    )
    assert int(link_row.rows[0]["n"]) >= 1, (
        f"Expected at least 1 evidence link for claim {claim_id!r}, got 0"
    )


@pytest.mark.asyncio
async def test_system2_semantic_corridor_persists_claim_edge_and_canonical_type(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims with corridor fields must:
    1. Persist a semantic_corridor claim row in knowledge_semantic_claims.
    2. Persist a directed edge row in knowledge_semantic_corridors with
       canonicalized corridor_type and original raw type in corridor_type_raw.
    3. Return the corridor claim ID in claim_ids."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    stable_id = "src/t7_corr.py::CorrClass"
    await _store_evidence(memory_service, stable_id)

    # Derive two endpoint claims first.
    r_from = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "from_room",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_from.manage is not None and r_from.manage.success
    from_claim_id = r_from.manage.claim_ids[0]

    r_to = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "to_room",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_to.manage is not None and r_to.manage.success
    to_claim_id = r_to.manage.claim_ids[0]

    # Derive the corridor claim.
    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "corridor_from_claim_id": from_claim_id,
                        "corridor_to_claim_id": to_claim_id,
                        "corridor_type": "calls-into",  # raw: should canonicalize to CALLS_INTO
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert result.manage is not None
    assert result.manage.success, f"Expected success, got error: {result.manage.error!r}"
    assert result.manage.claim_ids is not None
    assert len(result.manage.claim_ids) >= 1
    corridor_claim_id = result.manage.claim_ids[0]
    uuid.UUID(corridor_claim_id)

    # Claim row must be semantic_corridor.
    claim_row = await knowledge_backend.query(
        "SELECT claim_type, lifecycle_state FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (corridor_claim_id,),
    )
    assert len(claim_row.rows) == 1, f"No claim row for {corridor_claim_id!r}"
    assert claim_row.rows[0]["claim_type"] == "semantic_corridor"
    assert claim_row.rows[0]["lifecycle_state"] == "active_evidenced"

    # Edge row must exist with canonical type and raw type preserved.
    edge_row = await knowledge_backend.query(
        """
        SELECT from_claim_id, to_claim_id, corridor_type, corridor_type_raw
          FROM knowledge_semantic_corridors
         WHERE claim_id = $1::uuid
        """,
        (corridor_claim_id,),
    )
    assert len(edge_row.rows) == 1, f"Expected 1 edge row for {corridor_claim_id!r}"
    row = edge_row.rows[0]
    assert str(row["from_claim_id"]) == from_claim_id
    assert str(row["to_claim_id"]) == to_claim_id
    assert row["corridor_type"] == "CALLS_INTO", (
        f"Expected canonical type 'CALLS_INTO', got {row['corridor_type']!r}"
    )
    assert row["corridor_type_raw"] == "calls-into", (
        f"Expected raw type 'calls-into', got {row['corridor_type_raw']!r}"
    )


@pytest.mark.asyncio
async def test_system2_partial_corridor_fields_rejected(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must reject a request that supplies only
    some corridor fields (only from_claim_id, no to_claim_id/corridor_type).
    The model validator must raise ValidationError with a message that names
    all three required fields."""
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_service import MemoryRequest

    stable_id = "src/t7_partial.py::PartialClass"
    await _store_evidence(memory_service, stable_id)

    with pytest.raises(ValidationError) as exc_info:
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "corridor_from_claim_id": str(uuid.uuid4()),
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    error_text = str(exc_info.value)
    assert "corridor_from_claim_id" in error_text, (
        f"Expected 'corridor_from_claim_id' in error; got: {error_text!r}"
    )
    assert "corridor_to_claim_id" in error_text, (
        f"Expected 'corridor_to_claim_id' in error; got: {error_text!r}"
    )
    assert "corridor_type" in error_text, f"Expected 'corridor_type' in error; got: {error_text!r}"


@pytest.mark.asyncio
async def test_system2_corridor_self_loop_rejected(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must reject corridor edges where
    from_claim_id == to_claim_id.  The model validator must raise ValidationError
    with a message that mentions self-loop."""
    from pydantic import ValidationError

    from workflows_mcp.engine.memory_service import MemoryRequest

    stable_id = "src/t7_loop.py::LoopClass"
    await _store_evidence(memory_service, stable_id)

    same_id = str(uuid.uuid4())
    with pytest.raises(ValidationError) as exc_info:
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "corridor_from_claim_id": same_id,
                        "corridor_to_claim_id": same_id,
                        "corridor_type": "depends_on",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    error_text = str(exc_info.value)
    assert "self-loop" in error_text.lower(), (
        f"Expected 'self-loop' in error message; got: {error_text!r}"
    )


@pytest.mark.asyncio
async def test_system2_derive_without_resolvable_evidence_fails(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """derive_system2_semantic_claims must fail when no matching structural evidence
    rows exist in the scope — claims must not be born without resolvable evidence."""
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    # Do NOT store evidence — stable ID will not resolve.
    try:
        result = await memory_service.execute(
            MemoryRequest.model_validate(
                {
                    "operation": "derive_system2_semantic_claims",
                    "scope": _scope(),
                    "record": {
                        "format": "structured",
                        "derivation": {
                            "room_intent_label": "orphan_room",
                            "evidence_entity_stable_ids": ["nonexistent::StableId"],
                        },
                    },
                }
            )
        )
        assert result.manage is not None
        assert not result.manage.success, "Derivation with unresolvable evidence must not succeed"
        assert result.manage.error is not None
    except MemoryContractError:
        pass  # contract error is also acceptable


# ---------------------------------------------------------------------------
# Task 7 atomicity: duplicate directed edge rolls back — no orphaned claims
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_corridor_edge_rolls_back_and_leaves_no_orphaned_claim(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """Atomicity invariant: when the semantic_corridor edge insert fails due to a
    duplicate directed-edge unique constraint (from_claim_id, to_claim_id,
    corridor_type), the entire transaction must roll back.

    After the failed second attempt:
    - Exactly one claim row exists (the first successful one).
    - Exactly one corridor edge row exists.
    - No orphaned claim row (half-inserted state) was left in
      knowledge_semantic_claims from the second call.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    stable_id = "src/t7_atomicity.py::AtomicityClass"
    await _store_evidence(memory_service, stable_id)

    # Derive two endpoint claims to act as corridor endpoints.
    r_from = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "atomicity_from",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_from.manage is not None and r_from.manage.success
    from_claim_id = r_from.manage.claim_ids[0]

    r_to = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "atomicity_to",
                        "evidence_entity_stable_ids": [stable_id],
                    },
                },
            }
        )
    )
    assert r_to.manage is not None and r_to.manage.success
    to_claim_id = r_to.manage.claim_ids[0]

    corridor_payload = {
        "operation": "derive_system2_semantic_claims",
        "scope": _scope(),
        "record": {
            "format": "structured",
            "derivation": {
                "corridor_from_claim_id": from_claim_id,
                "corridor_to_claim_id": to_claim_id,
                "corridor_type": "calls-into",
                "evidence_entity_stable_ids": [stable_id],
            },
        },
    }

    # First derivation: must succeed.
    first = await memory_service.execute(MemoryRequest.model_validate(corridor_payload))
    first_error = first.manage.error if first.manage else "no manage result"
    assert first.manage is not None and first.manage.success, (
        f"First corridor derivation must succeed; error: {first_error!r}"
    )
    first_claim_id = first.manage.claim_ids[0]

    # Count claims and edges before second attempt.
    claims_before = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_semantic_claims WHERE palace = $1",
        (PALACE,),
    )
    count_before = claims_before.rows[0]["n"]

    # Second derivation: same directed+typed edge — unique constraint must fire,
    # transaction must roll back, operation must return failure.
    second = await memory_service.execute(MemoryRequest.model_validate(corridor_payload))
    assert second.manage is not None, "Second derivation must return a manage result"
    assert not second.manage.success, (
        "Duplicate corridor edge must not succeed; expected rollback and failure"
    )
    assert second.manage.error is not None
    assert "MEM_DB_ERROR" in second.manage.error, (
        f"Expected MEM_DB_ERROR in error; got: {second.manage.error!r}"
    )

    # Claim count must be unchanged — no orphaned row from the rolled-back attempt.
    claims_after = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_semantic_claims WHERE palace = $1",
        (PALACE,),
    )
    count_after = claims_after.rows[0]["n"]
    assert count_after == count_before, (
        f"Orphaned claim detected: claim count changed from {count_before} to {count_after}"
        f" after a rolled-back corridor insert"
    )

    # Exactly one corridor edge must exist for this directed+typed pair.
    edge_count = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
          FROM knowledge_semantic_corridors
         WHERE from_claim_id = $1::uuid
           AND to_claim_id   = $2::uuid
           AND corridor_type = 'CALLS_INTO'
        """,
        (from_claim_id, to_claim_id),
    )
    assert edge_count.rows[0]["n"] == 1, (
        f"Expected exactly 1 corridor edge, got {edge_count.rows[0]['n']}"
    )

    # The first claim must still be intact (not rolled back by the second attempt).
    first_claim_row = await knowledge_backend.query(
        "SELECT id FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (first_claim_id,),
    )
    assert len(first_claim_row.rows) == 1, (
        f"First corridor claim {first_claim_id!r} must survive the second attempt"
    )


# ---------------------------------------------------------------------------
# Override accountability wired into verification-cycle dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_cycle_updates_override_accountability(
    memory_service, knowledge_backend, clean_palace
) -> None:
    """record_system1_verification_cycle must automatically update override
    accountability_status for active/pending overrides in the covered scope,
    based on live evidence links — not on the cycle.success boolean.

    Acceptance criteria (evidence-based, ADR-013 v1 deterministic semantics):
    - Before any cycle: override accountability_status == 'pending'.
    - After a successful cycle (success=True) when live evidence links exist in the
      covered scope: the override transitions to 'supported'.
    - After a successful cycle (success=True) when no live evidence links exist:
      the override transitions to 'unsupported'.
    - After a failed cycle (success=False): accountability_status does NOT change.
    - Trajectory via cycle-operation path: pending → supported → unsupported → supported.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Step 1: Store structural evidence so we can derive a real claim.
    await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "structural_evidence": [
                        {
                            "entity_stable_id": "src/accountability_test.py::AccountabilityClass",
                            "entity_type": "class",
                            "evidence_category": "structural_class",
                            "evidence_data": {"file": "src/accountability_test.py", "line": 1},
                        },
                    ],
                },
            }
        )
    )

    # Step 2: Derive a semantic claim with the evidence entity — this creates
    # knowledge_claim_evidence_links rows linking claim_id → evidence_id.
    derive_result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system2_semantic_claims",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "derivation": {
                        "room_intent_label": "accountability_test_layer",
                        "evidence_entity_stable_ids": [
                            "src/accountability_test.py::AccountabilityClass"
                        ],
                    },
                },
            }
        )
    )
    assert derive_result.manage is not None and derive_result.manage.success
    assert derive_result.manage.claim_ids
    real_claim_id = derive_result.manage.claim_ids[0]

    # Step 3: Apply an override — starts as 'pending'.
    override_result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "apply_semantic_override",
                "scope": _scope(),
                "record": {
                    "format": "structured",
                    "override": {
                        "claim_id": real_claim_id,
                        "override_reason": "architect manual correction for accountability test",
                        "overridden_by": "test-architect",
                        "new_lifecycle_state": "active_evidenced",
                    },
                },
            }
        )
    )
    assert override_result.manage is not None and override_result.manage.success
    override_id = override_result.manage.override_id
    assert override_id is not None

    # Verify initial accountability_status is 'pending'.
    before_row = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert before_row.rows, "Override row must exist in DB"
    assert before_row.rows[0]["accountability_status"] == "pending", (
        f"Expected 'pending' before cycle; got {before_row.rows[0]['accountability_status']!r}"
    )

    def _covered_cycle(success: bool) -> dict:  # type: ignore[type-arg]
        return {
            "operation": "record_system1_verification_cycle",
            "scope": _scope(),
            "record": {
                "format": "structured",
                "verification_cycle": {
                    "success": success,
                    "covered_scope": {
                        "palace": PALACE,
                        "wing": WING,
                        "room": ROOM,
                        "compartment": COMPARTMENT,
                    },
                },
            },
        }

    # Step 4: Successful cycle with live evidence links in scope → 'supported'.
    # Evidence links were created in Step 2 (derive_system2_semantic_claims inserts them).
    cycle_result = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=True))
    )
    assert cycle_result.manage is not None and cycle_result.manage.success

    after_row = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row.rows[0]["accountability_status"] == "supported", (
        "After success=True cycle with live evidence links, override must be 'supported'; "
        f"got {after_row.rows[0]['accountability_status']!r}"
    )

    # Step 5: Remove all evidence links for the claim to simulate absence of evidence.
    await knowledge_backend.execute(
        "DELETE FROM knowledge_claim_evidence_links WHERE claim_id = $1::uuid",
        (real_claim_id,),
    )

    # Successful cycle with NO live evidence links in scope → 'unsupported'.
    cycle_result2 = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=True))
    )
    assert cycle_result2.manage is not None and cycle_result2.manage.success

    after_row2 = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row2.rows[0]["accountability_status"] == "unsupported", (
        "After success=True cycle with NO evidence links, override must be 'unsupported'; "
        f"got {after_row2.rows[0]['accountability_status']!r}"
    )

    # Step 6: Failed cycle (success=False) — accountability must NOT change (stays 'unsupported').
    cycle_result3 = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=False))
    )
    assert cycle_result3.manage is not None and cycle_result3.manage.success

    after_row3 = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row3.rows[0]["accountability_status"] == "unsupported", (
        "After success=False (failed) cycle, accountability_status must NOT change; "
        f"got {after_row3.rows[0]['accountability_status']!r}"
    )

    # Step 7: Re-add evidence link and run another successful cycle → back to 'supported'.
    # Re-query the evidence_id we stored in Step 1.
    evidence_row = await knowledge_backend.query(
        """
        SELECT id, evidence_category
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = $2
        """,
        (PALACE, "src/accountability_test.py::AccountabilityClass"),
    )
    assert evidence_row.rows, "Structural evidence row must still exist"
    evidence_id = evidence_row.rows[0]["id"]
    evidence_category = evidence_row.rows[0]["evidence_category"]

    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_claim_evidence_links
            (claim_id, evidence_id, evidence_category, linked_at)
        VALUES ($1::uuid, $2::uuid, $3, NOW())
        ON CONFLICT DO NOTHING
        """,
        (real_claim_id, evidence_id, evidence_category),
    )

    # Final successful cycle with re-linked evidence → 'supported' again (trajectory complete).
    cycle_result4 = await memory_service.execute(
        MemoryRequest.model_validate(_covered_cycle(success=True))
    )
    assert cycle_result4.manage is not None and cycle_result4.manage.success

    after_row4 = await knowledge_backend.query(
        "SELECT accountability_status FROM knowledge_semantic_overrides WHERE id = $1::uuid",
        (override_id,),
    )
    assert after_row4.rows[0]["accountability_status"] == "supported", (
        "After success=True cycle with re-added evidence links, override must be 'supported'; "
        f"got {after_row4.rows[0]['accountability_status']!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 1: derive_system1_topology fail-closed behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_fails_closed_when_evidence_rows_are_indeterminate(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Evidence IDs that reference no persisted rows → MEM_INSUFFICIENT_EVIDENCE.

    ADR-013 constraint: no fallback to resolved_scope.*, no literal 'default'.
    When the provided evidence_ids do not match any persisted structural evidence
    rows in this palace, the heuristic has nothing to work from and must fail closed.
    This is the authoritative indeterminate case: zero evidence rows loaded.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    # Use a fabricated UUID that does not exist in knowledge_structural_evidence.
    nonexistent_evidence_id = "00000000-dead-beef-0000-000000000099"

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [nonexistent_evidence_id],
            },
        }
    )

    with pytest.raises(MemoryContractError) as exc:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    error_code = exc.value.code
    assert "MEM_INSUFFICIENT_EVIDENCE" in error_code, (
        f"Expected MEM_INSUFFICIENT_EVIDENCE (or stricter descendant), got: {error_code!r}. "
        "Topology derivation must fail closed when evidence is insufficient — "
        "no fallback to resolved_scope.* or literals like 'default'."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_does_not_fallback_to_default_literal(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """derive_system1_topology must never produce 'default' as a wing/topology output.

    Even if evidence rows exist, the result must not silently assign a default
    topology value. The operation must either produce a proven topology or fail closed.
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    # Attempt derivation with a fabricated evidence_id (not actually stored).
    # The operation should fail closed, not silently return 'default'.
    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": ["00000000-dead-beef-0000-000000000001"],
            },
        }
    )

    with pytest.raises(MemoryContractError) as exc:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    error_code = exc.value.code
    assert "MEM_INSUFFICIENT_EVIDENCE" in error_code, (
        f"Expected fail-closed MEM_INSUFFICIENT_EVIDENCE, got: {error_code!r}. "
        "Operation must never silently produce 'default' topology — no hidden fallback permitted."
    )
    # Extra guard: confirm no 'default' leaks into the error message either
    assert "default" not in exc.value.message.lower(), (
        "Error message must not contain 'default' — that would suggest a fallback was attempted."
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 2b: derive_system1_topology explicit override path
# ---------------------------------------------------------------------------


async def _store_structural_evidence_and_get_id(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    *,
    entity_stable_id: str,
    wing: str = "owl-wing",
    room: str = "owl-room",
    compartment: str = "owl-compartment",
) -> str:
    """Helper: store one structural evidence row and return its UUID string."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_evidence",
            "scope": {
                "palace": PALACE,
                "wing": wing,
                "room": room,
                "compartment": compartment,
            },
            "record": {
                "format": "structured",
                "structural_evidence": [
                    {
                        "entity_stable_id": entity_stable_id,
                        "entity_type": "module",
                        "evidence_category": "structural_module",
                        "evidence_data": {"path": f"src/{entity_stable_id}.py"},
                    }
                ],
            },
        }
    )
    result = await memory_service.execute(req)  # type: ignore[union-attr]
    assert result.manage is not None and result.manage.success

    row = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1 AND entity_stable_id = $2
        """,
        (PALACE, entity_stable_id),
    )
    assert row.rows, f"Evidence row for {entity_stable_id!r} must exist after store"
    return str(row.rows[0]["id"])


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_success(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Explicit complete override succeeds, persists all rows, returns typed result fields.

    ADR-013 Task 2b: when topology_override is fully populated and evidence IDs
    reference persisted rows, the operation must:
    - Write/upsert a semantic claim row.
    - Write a semantic override accountability row (override_id).
    - Append a knowledge_topology_provenance row.
    - Append knowledge_topology_provenance_evidence rows for each evidence ID.
    - Return derivation_source='explicit_override', derived_wing/room/compartment,
      provenance_id, claim_id.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/owl.py::OwlClass"
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": "Authoritative placement confirmed by architect review.",
                    "applied_by": "architect-agent-v1",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]

    assert result.manage is not None, "manage section must be present"
    manage = result.manage
    assert manage.success, f"Operation must succeed; error={manage.error!r}"

    # Typed result fields
    assert manage.derived_wing == "owl-wing", (
        f"derived_wing must equal override wing; got {manage.derived_wing!r}"
    )
    assert manage.derived_room == "owl-room", (
        f"derived_room must equal override room; got {manage.derived_room!r}"
    )
    assert manage.derived_compartment == "owl-compartment", (
        f"derived_compartment must equal override compartment; got {manage.derived_compartment!r}"
    )
    assert manage.derivation_source == "explicit_override", (
        f"derivation_source must be 'explicit_override'; got {manage.derivation_source!r}"
    )
    assert manage.provenance_id is not None, "provenance_id must be set to a persisted row ID"
    assert manage.claim_id is not None, "claim_id must reference persisted semantic claim"

    # Durable DB inspectability: provenance row must exist
    prov_row = await knowledge_backend.query(
        """
        SELECT id, derivation_source, wing, room, compartment, override_reason, applied_by
          FROM knowledge_topology_provenance
         WHERE id = $1::uuid
        """,
        (manage.provenance_id,),
    )
    assert prov_row.rows, "knowledge_topology_provenance row must be persisted"
    prov = prov_row.rows[0]
    assert prov["derivation_source"] == "explicit_override"
    assert prov["wing"] == "owl-wing"
    assert prov["room"] == "owl-room"
    assert prov["compartment"] == "owl-compartment"
    assert prov["override_reason"] == "Authoritative placement confirmed by architect review."
    assert prov["applied_by"] == "architect-agent-v1"

    # Evidence link row must exist
    link_row = await knowledge_backend.query(
        """
        SELECT provenance_id, evidence_id
          FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid AND evidence_id = $2::uuid
        """,
        (manage.provenance_id, evidence_id),
    )
    assert link_row.rows, "knowledge_topology_provenance_evidence link row must be persisted"

    # Semantic claim must exist
    claim_row = await knowledge_backend.query(
        """
        SELECT id, claim_type, palace, wing, room, compartment
          FROM knowledge_semantic_claims
         WHERE id = $1::uuid
        """,
        (manage.claim_id,),
    )
    assert claim_row.rows, "knowledge_semantic_claims row must exist"
    claim = claim_row.rows[0]
    assert claim["palace"] == PALACE
    assert claim["wing"] == "owl-wing"

    # Semantic override accountability row must exist (links override to claim)
    override_row = await knowledge_backend.query(
        """
        SELECT id, claim_id, override_reason, applied_by
          FROM knowledge_semantic_overrides
         WHERE claim_id = $1::uuid
        """,
        (manage.claim_id,),
    )
    assert override_row.rows, "knowledge_semantic_overrides accountability row must exist"
    ov = override_row.rows[0]
    assert ov["override_reason"] == "Authoritative placement confirmed by architect review."
    assert ov["applied_by"] == "architect-agent-v1"


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_provenance_is_append_only(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Repeated accepted overrides each append a new provenance row (not update).

    ADR-013 append-only requirement: every accepted override call must produce
    a distinct knowledge_topology_provenance row, preserving full history.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/append.py::AppendClass"
    )

    def _derive_req(reason: str) -> dict:  # type: ignore[type-arg]
        return {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": reason,
                    "applied_by": "architect-agent-v1",
                },
            },
        }

    result1 = await memory_service.execute(  # type: ignore[union-attr]
        MemoryRequest.model_validate(_derive_req("First override"))
    )
    result2 = await memory_service.execute(  # type: ignore[union-attr]
        MemoryRequest.model_validate(_derive_req("Second override"))
    )

    assert result1.manage is not None and result1.manage.success
    assert result2.manage is not None and result2.manage.success

    prov_id_1 = result1.manage.provenance_id
    prov_id_2 = result2.manage.provenance_id
    assert prov_id_1 != prov_id_2, (
        "Each accepted override must produce a distinct provenance_id (append-only)."
    )

    # Both rows must be present in DB
    count_row = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND derivation_source = 'explicit_override'
        """,
        (PALACE,),
    )
    count = count_row.rows[0]["cnt"]
    assert count >= 2, f"Expected at least 2 provenance rows for append-only; found {count}"

    # Idempotent claim: repeated overrides for the same scope must NOT create
    # duplicate knowledge_semantic_claims rows — exactly 1 claim row expected.
    # Re-query using scope_key derived from the placement
    from workflows_mcp.engine.memory_service import _scope_key_fn  # type: ignore[import]

    scope_key = _scope_key_fn(
        {"palace": PALACE, "wing": "owl-wing", "room": "owl-room", "compartment": "owl-compartment"}
    )
    claim_count_row2 = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1
           AND scope_key = $2
           AND claim_type = 'room_intent'
        """,
        (PALACE, scope_key),
    )
    claim_count = claim_count_row2.rows[0]["cnt"]
    assert claim_count == 1, (
        f"Repeated overrides for same scope must produce exactly 1 claim row "
        f"(idempotent upsert); found {claim_count}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_multiple_evidence_ids(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """All provided evidence IDs are linked in knowledge_topology_provenance_evidence.

    When multiple evidence IDs are supplied, each must have a corresponding
    link row — no partial linking permitted.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    ev1 = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/multi1.py::ClassOne"
    )
    ev2 = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/multi2.py::ClassTwo"
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev1, ev2],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": "Multi-evidence authoritative placement.",
                    "applied_by": "architect-agent-v1",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    assert result.manage is not None and result.manage.success

    prov_id = result.manage.provenance_id
    link_rows = await knowledge_backend.query(
        """
        SELECT evidence_id::text
          FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid
        ORDER BY evidence_id
        """,
        (prov_id,),
    )
    linked_ids = {r["evidence_id"] for r in link_rows.rows}
    assert ev1 in linked_ids, f"Evidence ID {ev1!r} not linked in provenance_evidence"
    assert ev2 in linked_ids, f"Evidence ID {ev2!r} not linked in provenance_evidence"


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_nonexistent_evidence_id_fails(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Nonexistent evidence ID causes fail-closed error and full rollback.

    ADR-013 Task 2b: evidence IDs must reference persisted structural evidence rows.
    A nonexistent ID must produce an error and no orphaned rows (atomic rollback).
    """
    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    fake_id = "00000000-dead-beef-0000-000000000099"

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [fake_id],
                "topology_override": {
                    "wing": "owl-wing",
                    "room": "owl-room",
                    "compartment": "owl-compartment",
                    "override_reason": "Override with fake evidence.",
                    "applied_by": "bad-agent",
                },
            },
        }
    )

    with pytest.raises(MemoryContractError) as exc:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    assert "MEM_" in exc.value.code, (
        f"Expected a MEM_ error code for nonexistent evidence ID; got {exc.value.code!r}"
    )

    # Rollback: all four write tables must have zero committed rows for this attempt.
    prov_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND applied_by = 'bad-agent'
        """,
        (PALACE,),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "No knowledge_topology_provenance row must be committed when evidence ID is nonexistent."
    )

    claim_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1
        """,
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "No knowledge_semantic_claims row must be committed when evidence ID is nonexistent."
    )

    override_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_overrides so
          JOIN knowledge_semantic_claims sc ON sc.id = so.claim_id
         WHERE sc.palace = $1
        """,
        (PALACE,),
    )
    assert override_count.rows[0]["cnt"] == 0, (
        "No knowledge_semantic_overrides row must be committed when evidence ID is nonexistent."
    )

    prov_ev_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance_evidence kpe
          JOIN knowledge_topology_provenance ktp ON ktp.id = kpe.provenance_id
         WHERE ktp.palace = $1 AND ktp.applied_by = 'bad-agent'
        """,
        (PALACE,),
    )
    assert prov_ev_count.rows[0]["cnt"] == 0, (
        "No knowledge_topology_provenance_evidence row must be committed "
        "when evidence ID is nonexistent."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_without_override_succeeds_via_structural_heuristic(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """No topology_override supplied → structural heuristic derives topology (Task 3).

    Task 3 implements the structural derivation path. Without topology_override,
    the operation must now succeed when sufficient structural evidence is present,
    returning derivation_source='system1_derived'.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service, knowledge_backend, entity_stable_id="src/no_override.py::NoOverrideClass"
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                # No topology_override — structural heuristic path
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, (
        f"Non-override path must succeed via structural heuristic (Task 3); "
        f"got error={manage.error!r}"
    )
    assert manage.derivation_source == "system1_derived", (
        f"derivation_source must be 'system1_derived'; got {manage.derivation_source!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 3: derive_system1_topology structural heuristic (system1_derived)
# ---------------------------------------------------------------------------


async def _store_structural_evidence_typed(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    *,
    entity_stable_id: str,
    entity_type: str,
    evidence_category: str,
    wing: str,
    room: str,
    compartment: str,
) -> str:
    """Helper: store a structural evidence row with explicit type/category and return its UUID."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    req = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_evidence",
            "scope": {
                "palace": PALACE,
                "wing": wing,
                "room": room,
                "compartment": compartment,
            },
            "record": {
                "format": "structured",
                "structural_evidence": [
                    {
                        "entity_stable_id": entity_stable_id,
                        "entity_type": entity_type,
                        "evidence_category": evidence_category,
                        "evidence_data": {"path": f"src/{entity_stable_id}"},
                    }
                ],
            },
        }
    )
    result = await memory_service.execute(req)  # type: ignore[union-attr]
    assert result.manage is not None and result.manage.success

    row = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_structural_evidence
         WHERE palace = $1 AND entity_stable_id = $2
        """,
        (PALACE, entity_stable_id),
    )
    assert row.rows, f"Evidence row for {entity_stable_id!r} must exist after store"
    return str(row.rows[0]["id"])


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_success_returns_system1_derived(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Sufficient structural evidence → successful derivation with system1_derived source.

    ADR-013 Task 3: when evidence rows carry deterministic structural signals that allow
    wing/room/compartment to be resolved, the operation must:
    - Return success=True.
    - Return derivation_source='system1_derived'.
    - Return derivation_algorithm_version='system1.v1'.
    - Return non-empty derived_wing, derived_room, derived_compartment.
    - Return non-None provenance_id and claim_id.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Store a class entity (highest anchor priority) in a known wing/room/compartment.
    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_success/core.py::CoreEngine",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-wing",
        room="t3-room",
        compartment="t3-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, f"Expected success=True; got error={manage.error!r}"

    assert manage.derivation_source == "system1_derived", (
        f"derivation_source must be 'system1_derived'; got {manage.derivation_source!r}"
    )
    assert manage.derived_wing, f"derived_wing must be non-empty; got {manage.derived_wing!r}"
    assert manage.derived_room, f"derived_room must be non-empty; got {manage.derived_room!r}"
    assert manage.derived_compartment, (
        f"derived_compartment must be non-empty; got {manage.derived_compartment!r}"
    )
    assert manage.provenance_id is not None, "provenance_id must be returned"
    assert manage.claim_id is not None, "claim_id must be returned"


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_algorithm_version_is_system1_v1(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Derived topology provenance row carries derivation_algorithm_version='system1.v1'.

    ADR-013 Task 3: algorithm version must be persisted for inspectability and future
    upgrade tracking.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_version/mod.py::VersionMod",
        entity_type="module",
        evidence_category="structural_module",
        wing="t3-ver-wing",
        room="t3-ver-room",
        compartment="t3-ver-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success

    prov_row = await knowledge_backend.query(
        """
        SELECT derivation_algorithm_version, derivation_source
          FROM knowledge_topology_provenance
         WHERE id = $1::uuid
        """,
        (manage.provenance_id,),
    )
    assert prov_row.rows, "knowledge_topology_provenance row must be persisted"
    prov = prov_row.rows[0]
    assert prov["derivation_algorithm_version"] == "system1.v1", (
        f"algorithm version must be 'system1.v1'; got {prov['derivation_algorithm_version']!r}"
    )
    assert prov["derivation_source"] == "system1_derived", (
        f"derivation_source must be 'system1_derived'; got {prov['derivation_source']!r}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_provenance_is_append_only(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Two identical structural derivation calls produce two distinct provenance rows.

    ADR-013 Task 3: provenance is history, not mutable state. Each accepted derivation
    must produce a fresh knowledge_topology_provenance row.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_append/util.py::UtilClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-append-wing",
        room="t3-append-room",
        compartment="t3-append-comp",
    )

    async def _derive() -> str:
        req = MemoryRequest.model_validate(
            {
                "operation": "derive_system1_topology",
                "derivation": {
                    "palace": PALACE,
                    "evidence_ids": [ev_id],
                },
            }
        )
        res = await memory_service.execute(req)  # type: ignore[union-attr]
        assert res.manage is not None and res.manage.success
        return str(res.manage.provenance_id)

    prov_id_1 = await _derive()
    prov_id_2 = await _derive()

    assert prov_id_1 != prov_id_2, (
        "Each structural derivation must produce a distinct provenance_id (append-only); "
        f"got same id={prov_id_1!r} for both calls"
    )

    count_row = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND derivation_source = 'system1_derived'
           AND id IN ($2::uuid, $3::uuid)
        """,
        (PALACE, prov_id_1, prov_id_2),
    )
    assert count_row.rows[0]["cnt"] == 2, (
        "Both provenance rows must be persisted independently (append-only)"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_evidence_links_persisted(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Structural derivation links evidence IDs in knowledge_topology_provenance_evidence.

    ADR-013 Task 3: evidence IDs used for structural derivation must be linked in the
    provenance evidence table for accountability.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_evlink/svc.py::SvcClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-evlink-wing",
        room="t3-evlink-room",
        compartment="t3-evlink-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success

    link_row = await knowledge_backend.query(
        """
        SELECT provenance_id, evidence_id
          FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid AND evidence_id = $2::uuid
        """,
        (manage.provenance_id, ev_id),
    )
    assert link_row.rows, (
        "knowledge_topology_provenance_evidence row must link the evidence ID used in derivation"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_class_anchor_priority(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Class entity is preferred as anchor over function when both are present.

    ADR-013 Task 3 spec: anchor selection priority is class/module first, then
    function/doc unit. The derived topology must reflect the class entity's placement
    rather than the function entity's placement.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Store a class entity in one wing/room/compartment.
    cls_ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_anchor/engine.py::Engine",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-anchor-wing",
        room="t3-anchor-room",
        compartment="t3-anchor-class-comp",
    )
    # Store a function entity in a different compartment.
    fn_ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_anchor/utils.py::helper_fn",
        entity_type="function",
        evidence_category="structural_function",
        wing="t3-anchor-wing",
        room="t3-anchor-room",
        compartment="t3-anchor-fn-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [cls_ev_id, fn_ev_id],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True

    # Anchor is the class entity → compartment must reflect class entity identity.
    assert manage.derived_wing == "t3-anchor-wing", (
        f"derived_wing must be 't3-anchor-wing'; got {manage.derived_wing!r}"
    )
    assert manage.derived_room == "t3-anchor-room", (
        f"derived_room must be 't3-anchor-room'; got {manage.derived_room!r}"
    )
    # Compartment is derived from the class anchor's compartment column.
    assert manage.derived_compartment == "t3-anchor-class-comp", (
        f"derived_compartment should reflect class anchor placement; "
        f"got {manage.derived_compartment!r}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_no_semantic_labels_influence(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Structural derivation result must not depend on semantic intent labels.

    ADR-013 Task 3 spec: 'No semantic intent labels in System 1.' Two derivation calls
    with identical structural evidence but different derivation contexts (one with a
    semantic-sounding entity_stable_id prefix, one without) must produce the same
    wing/room/compartment topology from structural signals only.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Evidence rows in identical structural positions but with names that could
    # be mistaken for semantic categories.  The heuristic must use only structural
    # columns (wing/room/compartment, entity_type), not parse entity_stable_id
    # for intent/semantic signals.
    ev_structural = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_semantic/plain.py::PlainClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-semantic-wing",
        room="t3-semantic-room",
        compartment="t3-semantic-comp",
    )
    ev_intent_named = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_semantic/intent_domain_model.py::IntentDomainClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t3-semantic-wing",
        room="t3-semantic-room",
        compartment="t3-semantic-comp",
    )

    async def _derive(ev_id: str) -> tuple[str, str, str]:
        req = MemoryRequest.model_validate(
            {
                "operation": "derive_system1_topology",
                "derivation": {
                    "palace": PALACE,
                    "evidence_ids": [ev_id],
                },
            }
        )
        res = await memory_service.execute(req)  # type: ignore[union-attr]
        assert res.manage is not None and res.manage.success
        m = res.manage
        return (m.derived_wing or "", m.derived_room or "", m.derived_compartment or "")

    topo_plain = await _derive(ev_structural)
    topo_intent = await _derive(ev_intent_named)

    assert topo_plain == topo_intent, (
        "Structural derivation must produce identical topology for evidence rows in the same "
        f"structural position regardless of entity name semantics; "
        f"plain={topo_plain!r} intent={topo_intent!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 3 (fix): modal wing/room tie-break is lexicographic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_modal_wing_room_tie_break_is_lexicographic(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Equal-count wing and room values must resolve via lexicographic tie-break.

    This test constructs two structural evidence rows where:
    - wing "z-wing" and wing "a-wing" each appear exactly once (equal count).
    - room "z-room" and room "a-room" each appear exactly once (equal count).

    The rows are inserted in order that would cause Counter.most_common(1) to
    non-deterministically (or insertion-order-biased) select the non-lexicographic
    winner.  The correct implementation must always select "a-room" and "a-wing".

    To force the non-lexicographic value to appear first in insertion/DB order,
    the "z-*" row is stored first so that naive Counter iteration or DB row order
    would return it before "a-*".
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Store "z-wing"/"z-room" with a stable_id that sorts BEFORE "a-wing" entity's stable_id.
    # The DB query returns rows ORDER BY entity_stable_id ASC, so "aaa_..." sorts first.
    # This means "z-wing" is encountered first by Counter — naive most_common(1) in CPython
    # will return "z-wing" (first-seen wins ties), causing the non-lexicographic value to win.
    ev_z = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_tiebreak/aaa_first.py::FirstEntityZWing",
        entity_type="class",
        evidence_category="structural_class",
        wing="z-wing",
        room="z-room",
        compartment="tiebreak-comp",
    )
    # Store "a-wing"/"a-room" with a stable_id that sorts AFTER — it is seen second by Counter.
    ev_a = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t3_tiebreak/zzz_second.py::SecondEntityAWing",
        entity_type="class",
        evidence_category="structural_class",
        wing="a-wing",
        room="a-room",
        compartment="tiebreak-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_z, ev_a],
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, f"Expected success=True; got error={manage.error!r}"

    # Lexicographically first wins the tie: "a-wing" < "z-wing", "a-room" < "z-room".
    assert manage.derived_wing == "a-wing", (
        f"Equal-count wing tie must select lexicographically first value 'a-wing'; "
        f"got {manage.derived_wing!r}"
    )
    assert manage.derived_room == "a-room", (
        f"Equal-count room tie must select lexicographically first value 'a-room'; "
        f"got {manage.derived_room!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 4: proof-bundle new-wing gate against persisted evidence rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_new_wing_insufficient_bundle_one_row_two_declared_categories(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Single persisted evidence row cannot satisfy two declared categories.

    ADR-013 Task 4: is_new_wing=True with proof_bundle listing two categories but
    only one persisted evidence artifact backing them must fail with
    MEM_INSUFFICIENT_EVIDENCE_BUNDLE.

    The gate must count distinct persisted rows per declared category — one
    artifact can satisfy at most one category. Two category labels with one row
    must not pass.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Store exactly ONE evidence row under one category.
    ev_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_bundle/single.py::SingleClass",
        entity_type="class",
        evidence_category="structural_class",
        wing="t4-wing",
        room="t4-room",
        compartment="t4-comp",
    )

    # Declare two categories in proof_bundle but only supply the one evidence ID.
    # 'structural_class' is backed by the one row; 'structural_module' has no row.
    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev_id],
                "is_new_wing": True,
                "proof_bundle": {
                    "evidence_categories": ["structural_class", "structural_module"],
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is False, (
        "New-wing derivation with one persisted row must fail; "
        f"got success=True with derivation_source={manage.derivation_source!r}"
    )
    error_code = manage.error or ""
    assert "MEM_INSUFFICIENT_EVIDENCE_BUNDLE" in error_code, (
        f"Expected MEM_INSUFFICIENT_EVIDENCE_BUNDLE in error; got {error_code!r}. "
        "Gate must reject when distinct persisted evidence rows per category < 2."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_new_wing_sufficient_bundle_two_rows_two_categories(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Two distinct persisted evidence rows across two categories passes the gate.

    ADR-013 Task 4: is_new_wing=True with proof_bundle listing two categories,
    each backed by one distinct persisted evidence artifact, must succeed.

    Verifies: one artifact per category, >=2 categories, >=2 distinct rows.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    ev1 = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_bundle/class_a.py::ClassA",
        entity_type="class",
        evidence_category="structural_class",
        wing="t4b-wing",
        room="t4b-room",
        compartment="t4b-comp",
    )
    ev2 = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_bundle/mod_a.py::ModA",
        entity_type="module",
        evidence_category="structural_module",
        wing="t4b-wing",
        room="t4b-room",
        compartment="t4b-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev1, ev2],
                "is_new_wing": True,
                "proof_bundle": {
                    "evidence_categories": ["structural_class", "structural_module"],
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, (
        f"New-wing derivation with two distinct evidence rows across two categories "
        f"must succeed; got error={manage.error!r}"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_bypasses_proof_bundle_gate(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """topology_override bypasses the new-wing proof-bundle gate even when is_new_wing=True.

    ADR-013 Task 4: when topology_override is supplied (explicit override path),
    the proof-bundle gate must NOT be enforced regardless of is_new_wing value.
    An override is authoritative by definition; requiring a proof bundle for an
    override would break the explicit-override contract from Task 2b.

    Verifies: result succeeds with is_new_wing=True, no proof_bundle, and a valid
    topology_override — the gate must be bypassed entirely.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    ev = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t4_override_bypass/class_a.py::ClassA",
        entity_type="class",
        evidence_category="structural_class",
        wing="t4ob-wing",
        room="t4ob-room",
        compartment="t4ob-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [ev],
                "is_new_wing": True,
                # No proof_bundle — override must bypass the gate
                "topology_override": {
                    "wing": "t4ob-wing",
                    "room": "t4ob-room",
                    "compartment": "t4ob-comp",
                    "override_reason": "test override bypass",
                    "applied_by": "test-agent",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None
    assert manage.success is True, (
        "topology_override with is_new_wing=True and no proof_bundle must succeed "
        f"(override bypasses proof-bundle gate); got error={manage.error!r}"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 5: Atomic persistence — simulated failure and success-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_structural_atomic_rollback_on_db_failure(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Simulated DB failure during structural path leaves zero committed rows.

    ADR-013 Task 5: atomicity must be implementation-real, not documented-only.
    We inject a failure after the transaction opens (by replacing the backend's
    execute/query method to raise on the provenance INSERT) and assert that no
    claim, provenance, or provenance_evidence rows survive in the database.
    """
    from unittest.mock import patch

    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    evidence_id = await _store_structural_evidence_and_get_id(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t5_structural_atomic.py::AtomicClass",
        wing="t5s-wing",
        room="t5s-room",
        compartment="t5s-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
            },
        }
    )

    # Track call count so we can fail on the provenance INSERT (2nd write query
    # after the claim SELECT/INSERT — makes the failure happen mid-transaction).
    original_query = knowledge_backend.query
    call_count = 0

    async def failing_query(sql: str, params: object = None) -> object:
        nonlocal call_count
        # Let the claim SELECT through, then fail on the provenance INSERT.
        if "knowledge_topology_provenance" in sql and "INSERT" in sql:
            raise RuntimeError("simulated DB failure during provenance insert")
        return await original_query(sql, params)

    # NOTE: knowledge_backend.query is patched rather than knowledge_backend.execute because
    # the current provenance INSERT (and provenance_evidence INSERT) are issued via
    # _backend.query (RETURNING clause required).  If production moves DML to
    # _backend.execute, the injection target below must change accordingly.
    with patch.object(knowledge_backend, "query", side_effect=failing_query):
        with pytest.raises((MemoryContractError, Exception)) as exc_info:
            await memory_service.execute(derive_req)  # type: ignore[union-attr]

    # Confirm the monkeypatch actually fired — the raised exception must carry the
    # injected text.  A pass without this assertion could mean the test succeeded
    # because of an unrelated validation or setup error, not the intended failure.
    exc_str = str(exc_info.value)
    assert "simulated DB failure" in exc_str, (
        f"Expected 'simulated DB failure' in exception message, got: {exc_str!r}. "
        "The monkeypatch may not have fired — check the injection target."
    )

    # Assert rollback: no committed rows for this evidence set.
    prov_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance ktp
          JOIN knowledge_topology_provenance_evidence kpe ON kpe.provenance_id = ktp.id
         WHERE ktp.palace = $1
           AND kpe.evidence_id = $2::uuid
        """,
        (PALACE, evidence_id),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance_evidence must have zero rows after structural "
        "path transaction rollback on simulated DB failure."
    )

    claim_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1
           AND wing = 't5s-wing'
           AND room = 't5s-room'
        """,
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_claims must have zero rows after structural "
        "path transaction rollback on simulated DB failure."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_atomic_rollback_on_db_failure(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Simulated DB failure during override path leaves zero committed rows.

    ADR-013 Task 5: all five writes (claim, override, provenance, evidence links)
    must be committed atomically. A failure after the override INSERT but before
    commit must leave no claim, override, provenance, or provenance_evidence rows.
    """
    from unittest.mock import patch

    from workflows_mcp.engine.memory_service import MemoryContractError, MemoryRequest

    evidence_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t5_override_atomic/class_a.py::ClassA",
        entity_type="class",
        evidence_category="structural_class",
        wing="t5o-wing",
        room="t5o-room",
        compartment="t5o-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "t5o-wing",
                    "room": "t5o-room",
                    "compartment": "t5o-comp",
                    "override_reason": "atomic-test override",
                    "applied_by": "t5-atomic-agent",
                },
            },
        }
    )

    original_query = knowledge_backend.query

    async def failing_query(sql: str, params: object = None) -> object:
        # Fail on the provenance_evidence INSERT (last write before commit).
        if "knowledge_topology_provenance_evidence" in sql and "INSERT" in sql:
            raise RuntimeError("simulated DB failure during provenance_evidence insert")
        return await original_query(sql, params)

    # NOTE: knowledge_backend.query is patched rather than knowledge_backend.execute because
    # the current provenance_evidence INSERT uses RETURNING (issued via _backend.query).
    # If production moves DML to _backend.execute, the injection target below must change.
    with patch.object(knowledge_backend, "query", side_effect=failing_query):
        with pytest.raises((MemoryContractError, Exception)) as exc_info:
            await memory_service.execute(derive_req)  # type: ignore[union-attr]

    # Confirm the monkeypatch actually fired — the raised exception must carry the
    # injected text.  A pass without this assertion could mean the test succeeded
    # because of an unrelated validation or setup error, not the intended failure.
    exc_str = str(exc_info.value)
    assert "simulated DB failure" in exc_str, (
        f"Expected 'simulated DB failure' in exception message, got: {exc_str!r}. "
        "The monkeypatch may not have fired — check the injection target."
    )

    # Assert rollback: no surviving rows for this override attempt.
    prov_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance
         WHERE palace = $1 AND applied_by = 't5-atomic-agent'
        """,
        (PALACE,),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance must have zero rows after override "
        "path transaction rollback on simulated DB failure."
    )

    override_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_overrides so
          JOIN knowledge_semantic_claims sc ON sc.id = so.claim_id
         WHERE sc.palace = $1 AND so.applied_by = 't5-atomic-agent'
        """,
        (PALACE,),
    )
    assert override_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_overrides must have zero rows after override "
        "path transaction rollback on simulated DB failure."
    )

    claim_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_semantic_claims
         WHERE palace = $1 AND wing = 't5o-wing' AND room = 't5o-room'
        """,
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_claims must have zero rows after override "
        "path transaction rollback on simulated DB failure."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_explicit_override_all_four_tables_commit_atomically(
    memory_service: object,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
) -> None:
    """Success path: claim, override, provenance, and evidence link all commit together.

    ADR-013 Task 5: on a clean success, all four write tables must contain
    exactly one new row traceable back to the same derive_system1_topology call.
    This is the positive atomicity assertion — all-or-nothing in the success direction.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    evidence_id = await _store_structural_evidence_typed(
        memory_service,
        knowledge_backend,
        entity_stable_id="src/t5_success_atomic/class_b.py::ClassB",
        entity_type="class",
        evidence_category="structural_class",
        wing="t5a-wing",
        room="t5a-room",
        compartment="t5a-comp",
    )

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "evidence_ids": [evidence_id],
                "topology_override": {
                    "wing": "t5a-wing",
                    "room": "t5a-room",
                    "compartment": "t5a-comp",
                    "override_reason": "all-four-tables test",
                    "applied_by": "t5-success-agent",
                },
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True, (
        f"derive_system1_topology explicit override must succeed; error={manage.error!r}"
    )

    provenance_id = manage.provenance_id
    claim_id = manage.claim_id
    assert provenance_id, "provenance_id must be returned on success"
    assert claim_id, "claim_id must be returned on success"

    # 1. Semantic claim exists.
    claim_row = await knowledge_backend.query(
        "SELECT id FROM knowledge_semantic_claims WHERE id = $1::uuid",
        (claim_id,),
    )
    assert claim_row.rows, "knowledge_semantic_claims row must exist after successful commit"

    # 2. Semantic override exists linked to the claim.
    override_row = await knowledge_backend.query(
        """
        SELECT id FROM knowledge_semantic_overrides
         WHERE claim_id = $1::uuid AND applied_by = 't5-success-agent'
        """,
        (claim_id,),
    )
    assert override_row.rows, "knowledge_semantic_overrides row must exist after successful commit"

    # 3. Topology provenance exists.
    prov_row = await knowledge_backend.query(
        "SELECT id FROM knowledge_topology_provenance WHERE id = $1::uuid",
        (provenance_id,),
    )
    assert prov_row.rows, "knowledge_topology_provenance row must exist after successful commit"

    # 4. Provenance evidence link exists.
    prov_ev_row = await knowledge_backend.query(
        """
        SELECT evidence_id FROM knowledge_topology_provenance_evidence
         WHERE provenance_id = $1::uuid AND evidence_id = $2::uuid
        """,
        (provenance_id, evidence_id),
    )
    assert prov_ev_row.rows, (
        "knowledge_topology_provenance_evidence row must exist after successful commit"
    )


# ---------------------------------------------------------------------------
# ADR-013 Task 5b: inline candidate passthrough + atomic store+derive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_system1_topology_inline_candidates_stored_and_derived_atomically(
    memory_service: Any,
    knowledge_backend: Any,
    clean_palace: None,
) -> None:
    """Inline candidates must be stored as knowledge_structural_evidence and derived atomically.

    ADR-013 Task 5b: one operation accepts inline candidates, stores them, and derives topology
    in a single transaction. On success, structural evidence rows must exist in the database.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "inline_candidates": [
                    {
                        "entity_stable_id": "src/t5b_inline/engine.py::EngineClass",
                        "entity_type": "class",
                        "evidence_category": "structural_class",
                        "evidence_data": {"source": "treesitter"},
                    },
                    {
                        "entity_stable_id": "src/t5b_inline/engine.py",
                        "entity_type": "module",
                        "evidence_category": "structural_module",
                        "evidence_data": {"source": "treesitter"},
                    },
                ],
                "topology_override": {
                    "wing": "t5b-wing",
                    "room": "t5b-room",
                    "compartment": "t5b-comp",
                    "override_reason": "inline candidate atomic test",
                    "applied_by": "t5b-agent",
                },
                "parser_metadata": {"language": "python", "file_path": "src/t5b_inline/engine.py"},
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True, (
        f"derive_system1_topology with inline_candidates must succeed; error={manage.error!r}"
    )

    provenance_id = manage.provenance_id
    claim_id = manage.claim_id
    assert provenance_id, "provenance_id must be returned on success"
    assert claim_id, "claim_id must be returned on success"

    # Inline candidates must have been stored as knowledge_structural_evidence rows.
    ev_rows = await knowledge_backend.query(
        "SELECT id FROM knowledge_structural_evidence"
        " WHERE palace = $1 AND entity_stable_id LIKE $2",
        (PALACE, "src/t5b_inline/%"),
    )
    assert len(ev_rows.rows) >= 2, (
        f"inline_candidates must be stored as knowledge_structural_evidence rows; "
        f"found {len(ev_rows.rows)}"
    )

    # Provenance evidence links must reference the stored inline candidate rows.
    prov_ev_rows = await knowledge_backend.query(
        "SELECT evidence_id FROM knowledge_topology_provenance_evidence"
        " WHERE provenance_id = $1::uuid",
        (provenance_id,),
    )
    assert prov_ev_rows.rows, (
        "knowledge_topology_provenance_evidence must link to inline candidate evidence rows"
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_inline_candidates_insufficient_no_partial_write(
    memory_service: Any,
    knowledge_backend: Any,
    clean_palace: None,
) -> None:
    """Insufficient inline candidates without topology_override must fail closed with no write.

    ADR-013 Task 5b: when inline candidates cannot establish complete topology and no explicit
    override is provided, the operation must fail closed. No evidence rows, claim rows, or
    provenance rows must be written (atomic rollback).
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    # Single inline candidate without override — insufficient for complete topology.
    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "inline_candidates": [
                    {
                        "entity_stable_id": "src/t5b_insuff/lone.py::LoneFunc",
                        "entity_type": "function",
                        "evidence_category": "structural_function",
                        "evidence_data": {},
                    }
                ],
                # No topology_override — heuristic must fail closed for single-file evidence.
            },
        }
    )

    from workflows_mcp.engine.memory_service import MemoryContractError

    with pytest.raises(MemoryContractError) as exc_info:
        await memory_service.execute(derive_req)  # type: ignore[union-attr]

    error_code = exc_info.value.code
    assert "MEM_INSUFFICIENT" in error_code, (
        f"error must indicate insufficient evidence; got {error_code!r}"
    )

    # Rollback assertions: no rows must survive in the DB for this scope.
    # The inline candidate used entity_stable_id "src/t5b_insuff/lone.py::LoneFunc";
    # all four tables must be empty for this palace after the failed operation.
    evidence_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id LIKE 'src/t5b_insuff/%'
        """,
        (PALACE,),
    )
    assert evidence_count.rows[0]["cnt"] == 0, (
        "knowledge_structural_evidence must have zero rows for 't5b_insuff' scope "
        "after MEM_INSUFFICIENT failure — inline candidate evidence must not survive rollback."
    )

    prov_count = await knowledge_backend.query(
        "SELECT COUNT(*) AS cnt FROM knowledge_topology_provenance WHERE palace = $1",
        (PALACE,),
    )
    assert prov_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance must have zero rows after MEM_INSUFFICIENT failure — "
        "no provenance row must survive rollback."
    )

    prov_evidence_count = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS cnt
          FROM knowledge_topology_provenance_evidence kpe
          JOIN knowledge_topology_provenance ktp ON ktp.id = kpe.provenance_id
         WHERE ktp.palace = $1
        """,
        (PALACE,),
    )
    assert prov_evidence_count.rows[0]["cnt"] == 0, (
        "knowledge_topology_provenance_evidence must have zero rows after MEM_INSUFFICIENT "
        "failure — no evidence link must survive rollback."
    )

    claim_count = await knowledge_backend.query(
        "SELECT COUNT(*) AS cnt FROM knowledge_semantic_claims WHERE palace = $1",
        (PALACE,),
    )
    assert claim_count.rows[0]["cnt"] == 0, (
        "knowledge_semantic_claims must have zero rows after MEM_INSUFFICIENT failure — "
        "no claim row must survive rollback."
    )


@pytest.mark.asyncio
async def test_derive_system1_topology_inline_candidates_wing_not_from_language(
    memory_service: Any,
    knowledge_backend: Any,
    clean_palace: None,
) -> None:
    """Derived wing must not equal parser_metadata language value.

    ADR-013 Task 5b: wing must not default to programming language.
    Parser metadata is evidence metadata only; language value must never appear as wing.
    """
    from workflows_mcp.engine.memory_service import MemoryRequest

    derive_req = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": PALACE,
                "inline_candidates": [
                    {
                        "entity_stable_id": "src/t5b_lang/module.py::LangClass",
                        "entity_type": "class",
                        "evidence_category": "structural_class",
                        "evidence_data": {},
                    },
                    {
                        "entity_stable_id": "src/t5b_lang/module.py",
                        "entity_type": "module",
                        "evidence_category": "structural_module",
                        "evidence_data": {},
                    },
                ],
                "topology_override": {
                    "wing": "t5b-lang-wing",
                    "room": "t5b-lang-room",
                    "compartment": "t5b-lang-comp",
                    "override_reason": "language-not-wing test",
                    "applied_by": "t5b-agent",
                },
                "parser_metadata": {"language": "python"},
            },
        }
    )

    result = await memory_service.execute(derive_req)  # type: ignore[union-attr]
    manage = result.manage
    assert manage is not None and manage.success is True, (
        f"derive_system1_topology must succeed with override; error={manage.error!r}"
    )
    # Derived wing must come from override, not from parser_metadata.language.
    assert manage.derived_wing != "python", (
        "derived_wing must not equal parser_metadata.language ('python')"
    )
    assert manage.derived_wing == "t5b-lang-wing", (
        f"derived_wing must equal the override value; got {manage.derived_wing!r}"
    )
