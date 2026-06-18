"""Behavior tests for the eight low-level memory executor ops.

Each op is exercised through MemoryService.execute() so the public envelope
(MemoryRequest) is what's contracted, not the internal ManageMemoryRequest.

Covers ensure_source, ensure_item, store_entities, store_relations,
store_memories, store_entity_embeddings, archive_memories, mark_item_dirty,
plus the Track 4 qualified-name / parent-class / relation-metadata prereqs.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from _ops_helpers import CODE_WING, COMPARTMENT, PALACE, ROOM, WING, _scope

from workflows_mcp.engine.sql.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio


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
    from workflows_mcp.memory.memory_schema import MemoryItemInput

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
    from workflows_mcp.memory.memory_schema import MemoryRecordInput

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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

    from workflows_mcp.memory.memory_errors import MemoryContractError

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
        "workflows_mcp.memory.memory_service.compute_embedding",
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
    from workflows_mcp.memory.memory_schema import MemoryRequest

    async def fake_compute_embedding(  # noqa: E501
        *args: Any, **kwargs: Any
    ) -> tuple[list[float], str, int, dict[str, Any]]:
        return [0.0] * 1536, "test-embedding-model", 1536, {}

    monkeypatch.setattr(  # noqa: E501
        "workflows_mcp.memory.memory_service.compute_embedding", fake_compute_embedding
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
    from workflows_mcp.memory.memory_schema import MemoryRequest

    async def fake_compute_embedding(  # noqa: E501
        *args: Any, **kwargs: Any
    ) -> tuple[list[float], str, int, dict[str, Any]]:
        return [0.0] * 1536, "test-embedding-model", 1536, {}

    monkeypatch.setattr(  # noqa: E501
        "workflows_mcp.memory.memory_service.compute_embedding", fake_compute_embedding
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

    from workflows_mcp.memory.memory_errors import MemoryContractError

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_errors import MemoryContractError

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_errors import MemoryContractError

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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

    from workflows_mcp.memory.memory_errors import MemoryContractError

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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
    from workflows_mcp.memory.memory_schema import MemoryRequest

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

    from workflows_mcp.memory.memory_schema import MemoryRequest

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

    from workflows_mcp.memory.memory_schema import MemoryRequest

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
