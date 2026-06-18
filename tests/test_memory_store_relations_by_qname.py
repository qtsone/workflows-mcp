"""TDD contract tests for Memory operation store_relations_by_qname (ADR-012).

All tests run through MemoryService.execute() — same seam as test_memory_executor_ops.py.
Tests are arranged RED-first: they must fail before production code exists.

Operation contract (ADR-012):
  Request: operation="store_relations_by_qname", scope (palace), graph.relations[],
           graph.external_fallback ("module"|"reject", default "module").
  Response (in manage): operation, success, relation_ids[], created_count, existing_count,
                        external_entities_created[], unresolved_or_ambiguous[].

STRUCTURAL relation types: CONTAINS, INHERITS_FROM, IMPORTS, CALLS.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from workflows_mcp.engine.executor_base import Execution
from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.memory_service import MemoryRequest, MemoryService
from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio

PALACE = "palace_qname_rel_test"
WING = "code"
ROOM = "default"
COMPARTMENT = "qname_rel"

STRUCTURAL_TYPES = ("CONTAINS", "INHERITS_FROM", "IMPORTS", "CALLS")


def _scope(palace: str = PALACE) -> dict[str, str]:
    return {"palace": palace, "wing": WING, "room": ROOM, "compartment": COMPARTMENT}


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
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


@pytest_asyncio.fixture
async def memory_service(knowledge_backend: PostgresBackend) -> MemoryService:
    context = MagicMock(spec=Execution)
    context.execution_context = MagicMock()
    context.execution_context.get = MagicMock(return_value=None)
    context.execution_context.user_id = None
    context.execution_context.user_string_id = None
    context.execution_context.auth_method = None
    return MemoryService(backend=knowledge_backend, context=context)


@pytest_asyncio.fixture
async def clean_palace(knowledge_backend: PostgresBackend) -> AsyncIterator[None]:
    """Wipe rows belonging to any test palace variant before and after each test."""

    async def _wipe() -> None:
        pattern = f"{PALACE}%"
        await knowledge_backend.execute(
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_memories "
            "WHERE memory_id IN (SELECT id FROM knowledge_memories WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_embeddings "
            "WHERE entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_memories WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entities WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_items WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_sources WHERE palace LIKE $1", (pattern,)
        )

    await _wipe()
    yield
    await _wipe()


# ---------------------------------------------------------------------------
# Helper: seed a STRUCTURAL entity directly via store_entities
# ---------------------------------------------------------------------------


async def _seed_entity(
    memory_service: MemoryService,
    *,
    palace: str = PALACE,
    entity_type: str,
    name: str,
    stable_id: str,
    qualified_name: str,
    source_item_id: str | None = None,
) -> str:
    """Insert one STRUCTURAL entity and return its UUID."""
    entity: dict[str, Any] = {
        "entity_type": entity_type,
        "name": name,
        "stable_id": stable_id,
        "qualified_name": qualified_name,
    }
    if source_item_id is not None:
        entity["source_item_id"] = source_item_id

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_entities",
                "scope": _scope(palace),
                "record": {
                    "format": "structured",
                    "source": "STRUCTURAL",
                    "entities": [entity],
                },
            }
        )
    )
    assert result.manage.success, f"seed_entity failed: {result.manage.error}"
    return result.manage.entity_ids[0]


def _stable_id_for_external(palace: str, qname: str) -> str:
    """Compute the EXTERNAL Module stable_id as the server does."""
    raw = f"{palace}:__external__:{qname}:Module"
    return hashlib.sha256(raw.encode()).hexdigest()


async def _seed_item(
    memory_service: MemoryService,
    *,
    palace: str = PALACE,
    source: str,
    path: str,
) -> str:
    """Create a knowledge_items row via ensure_item and return its UUID."""
    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "ensure_item",
                "scope": _scope(palace),
                "record": {
                    "format": "raw",
                    "source": source,
                    "path": path,
                    "item": {
                        "content_hash": "test_hash",
                        "size_bytes": 1,
                        "mtime_ns": 0,
                    },
                },
            }
        )
    )
    assert result.manage.success, f"seed_item failed: {result.manage.error}"
    assert result.manage.item_id is not None
    return result.manage.item_id


# ===========================================================================
# Test 1 — Happy path: CALLS relation inserted by qname, returns valid UUID
# ===========================================================================


async def test_happy_path_calls_relation_by_qname(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Insert one CALLS relation via qname; verify UUID returned and row exists."""
    src_id = await _seed_entity(
        memory_service,
        entity_type="Function",
        name="caller",
        stable_id="happy::caller",
        qualified_name="pkg.mod.caller",
    )
    tgt_id = await _seed_entity(
        memory_service,
        entity_type="Function",
        name="callee",
        stable_id="happy::callee",
        qualified_name="pkg.mod.callee",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "pkg.mod.caller",
                            "source_entity_type": "Function",
                            "target_qname": "pkg.mod.callee",
                            "target_entity_type": "Function",
                            "relation_type": "CALLS",
                            "confidence": 0.9,
                            "metadata": {"call_site": {"line": 10, "col": 4}},
                        }
                    ],
                    "external_fallback": "module",
                },
            }
        )
    )

    m = result.manage
    assert m.success, f"expected success, got error: {m.error}"
    assert m.operation == "store_relations_by_qname"
    assert isinstance(m.relation_ids, list)
    assert len(m.relation_ids) == 1
    rel_id = m.relation_ids[0]
    assert rel_id is not None
    # Must be a valid UUID string
    uuid.UUID(rel_id)
    assert m.created_count == 1
    assert m.existing_count == 0

    # Verify DB row
    rows = await knowledge_backend.query(
        "SELECT source_entity_id, target_entity_id, relation_type "
        "FROM knowledge_relations WHERE id = $1::uuid",
        (rel_id,),
    )
    assert len(rows.rows) == 1
    assert str(rows.rows[0]["source_entity_id"]) == src_id
    assert str(rows.rows[0]["target_entity_id"]) == tgt_id
    assert rows.rows[0]["relation_type"] == "CALLS"


# ===========================================================================
# Test 2 — Idempotency: second identical call inserts zero new rows
# ===========================================================================


async def test_idempotency_second_call_zero_inserts(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Second identical call: created_count=0, existing_count=1, same relation id."""
    await _seed_entity(
        memory_service,
        entity_type="Function",
        name="f_src",
        stable_id="idem::src",
        qualified_name="pkg.idem.src",
    )
    await _seed_entity(
        memory_service,
        entity_type="Function",
        name="f_tgt",
        stable_id="idem::tgt",
        qualified_name="pkg.idem.tgt",
    )

    payload = {
        "operation": "store_relations_by_qname",
        "scope": _scope(),
        "graph": {
            "relations": [
                {
                    "source_qname": "pkg.idem.src",
                    "source_entity_type": "Function",
                    "target_qname": "pkg.idem.tgt",
                    "target_entity_type": "Function",
                    "relation_type": "CALLS",
                    "confidence": 0.8,
                    "metadata": {"call_site": {"line": 5}},
                }
            ]
        },
    }

    r1 = await memory_service.execute(MemoryRequest.model_validate(payload))
    assert r1.manage.success
    assert r1.manage.created_count == 1
    assert r1.manage.existing_count == 0
    rel_id_first = r1.manage.relation_ids[0]

    r2 = await memory_service.execute(MemoryRequest.model_validate(payload))
    assert r2.manage.success
    assert r2.manage.created_count == 0
    assert r2.manage.existing_count == 1
    assert r2.manage.relation_ids[0] == rel_id_first

    # Confirm only one row in DB
    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations WHERE id = $1::uuid",
        (rel_id_first,),
    )
    assert rows.rows[0]["n"] == 1


# ===========================================================================
# Test 3 — Palace isolation: target in palace B invisible to palace A
# ===========================================================================


async def test_palace_isolation_target_in_other_palace(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Source in palace A; target qname exists only in palace B — must be unresolved."""
    palace_a = PALACE
    palace_b = f"{PALACE}_b"

    # Source in palace A
    await _seed_entity(
        memory_service,
        palace=palace_a,
        entity_type="Function",
        name="src",
        stable_id="iso::src",
        qualified_name="iso.pkg.src",
    )

    # Target in palace B — palace A should never see it
    await _seed_entity(
        memory_service,
        palace=palace_b,
        entity_type="Function",
        name="tgt_only_in_b",
        stable_id="iso::tgt",
        qualified_name="iso.pkg.tgt",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(palace_a),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "iso.pkg.src",
                            "source_entity_type": "Function",
                            "target_qname": "iso.pkg.tgt",
                            "target_entity_type": "Function",
                            "relation_type": "CALLS",
                            "confidence": 0.7,
                            "metadata": {},
                            "external_fallback": "reject",
                        }
                    ],
                    "external_fallback": "reject",
                },
            }
        )
    )

    m = result.manage
    assert m.success
    # Target was invisible → must appear in unresolved_or_ambiguous
    assert m.relation_ids[0] is None
    assert len(m.unresolved_or_ambiguous) == 1
    entry = m.unresolved_or_ambiguous[0]
    assert entry["index"] == 0
    assert "unresolved" in entry["reason"]

    # No cross-palace relation inserted (check within palace A only)
    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations r "
        "JOIN knowledge_entities e ON r.source_entity_id = e.id "
        "WHERE e.palace = $1",
        (palace_a,),
    )
    assert rows.rows[0]["n"] == 0


# ===========================================================================
# Test 4 — Qname collision across different entity_type: target_entity_type disambiguates
# ===========================================================================


async def test_qname_collision_different_entity_type_disambiguates(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Same qname in same palace but different entity_type; target_entity_type selects correct."""
    await _seed_entity(
        memory_service,
        entity_type="Function",
        name="multi",
        stable_id="col::multi_fn",
        qualified_name="pkg.col.multi",
    )
    tgt_class_id = await _seed_entity(
        memory_service,
        entity_type="Class",
        name="multi",
        stable_id="col::multi_cls",
        qualified_name="pkg.col.multi",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "pkg.col.multi",
                            "source_entity_type": "Function",
                            "target_qname": "pkg.col.multi",
                            "target_entity_type": "Class",  # narrows to the Class entity
                            "relation_type": "CALLS",
                            "confidence": 0.7,
                            "metadata": {},
                        }
                    ]
                },
            }
        )
    )

    m = result.manage
    assert m.success
    # Must not be ambiguous — entity_type resolved it
    assert len(m.unresolved_or_ambiguous) == 0
    rel_id = m.relation_ids[0]
    assert rel_id is not None

    # Verify the target UUID is the Class entity
    rows = await knowledge_backend.query(
        "SELECT target_entity_id FROM knowledge_relations WHERE id = $1::uuid",
        (rel_id,),
    )
    assert str(rows.rows[0]["target_entity_id"]) == tgt_class_id


# ===========================================================================
# Test 5 — Ambiguity: two Function entities same qname/entity_type, different source_item_id
# ===========================================================================


async def test_ambiguity_same_qname_entity_type_different_items(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Two Function entities share qname+entity_type but exist in different rooms
    → ambiguous, no insert."""
    # Insert two entities with the same (palace, entity_type, qualified_name) but placed in
    # different rooms to avoid the unique (palace, namespace, room, corridor, entity_type, name)
    # constraint. The resolver matches on qname+entity_type across all rooms within the palace.
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source,
             stable_id, qualified_name)
        VALUES ($1, $2, 'room_a', 'qname_rel', 'Function', 'helper', 'STRUCTURAL',
                'amb::a', 'mod.helper')
        """,
        (PALACE, WING),
    )
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source,
             stable_id, qualified_name)
        VALUES ($1, $2, 'room_b', 'qname_rel', 'Function', 'helper', 'STRUCTURAL',
                'amb::b', 'mod.helper')
        """,
        (PALACE, WING),
    )

    # Seed a source entity (clean name)
    await _seed_entity(
        memory_service,
        entity_type="Function",
        name="caller",
        stable_id="amb::caller",
        qualified_name="mod.caller",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "mod.caller",
                            "source_entity_type": "Function",
                            "target_qname": "mod.helper",
                            "target_entity_type": "Function",
                            # No target_item_id — must surface ambiguity
                            "relation_type": "CALLS",
                            "confidence": 0.7,
                            "metadata": {},
                        }
                    ]
                },
            }
        )
    )

    m = result.manage
    assert m.success
    assert m.relation_ids[0] is None  # no insert on ambiguity
    assert len(m.unresolved_or_ambiguous) == 1
    entry = m.unresolved_or_ambiguous[0]
    assert entry["reason"] == "ambiguous"
    assert "candidates" in entry
    assert len(entry["candidates"]) == 2

    # No relation should have been inserted in this palace
    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations r "
        "JOIN knowledge_entities e ON r.source_entity_id = e.id "
        "WHERE e.palace = $1",
        (PALACE,),
    )
    assert rows.rows[0]["n"] == 0


# ===========================================================================
# Test 5b — source_item_id narrows duplicate source candidates to one, inserts
# ===========================================================================


async def test_source_item_id_narrows_ambiguous_source_to_one(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Supplying source_item_id selects the correct source entity among duplicates.

    Two Function entities share qualified_name + entity_type but belong to different
    source items (created via ensure_item) and different rooms (to bypass the unique
    name-per-room constraint).  Without source_item_id the resolver returns ambiguous;
    with it the correct entity is pinned and the relation is inserted.
    """
    item_a_id = await _seed_item(memory_service, source="narrow-source-a", path="narrow/a.py")
    item_b_id = await _seed_item(memory_service, source="narrow-source-b", path="narrow/b.py")

    # Two source candidates: same qname + entity_type, different rooms + source_item_ids.
    # Direct SQL required because store_entities enforces unique (palace, ns, room, corridor,
    # entity_type, name) — two entries in the same room would violate it.
    src_a_id = str(uuid.uuid4())
    src_b_id = str(uuid.uuid4())
    for eid, iid, room, stable in [
        (src_a_id, item_a_id, "narrow_room_a", "narrow::src_a"),
        (src_b_id, item_b_id, "narrow_room_b", "narrow::src_b"),
    ]:
        await knowledge_backend.execute(
            """
            INSERT INTO knowledge_entities
                (id, palace, namespace, room, corridor, entity_type, name, source,
                 stable_id, source_item_id, qualified_name)
            VALUES ($1::uuid, $2, 'code', $3, 'qname_rel',
                    'Function', 'shared_fn', 'STRUCTURAL',
                    $4, $5::uuid, 'mod.shared_fn')
            """,
            (eid, PALACE, room, stable, iid),
        )

    tgt_id = await _seed_entity(
        memory_service,
        entity_type="Function",
        name="target_fn",
        stable_id="narrow::target",
        qualified_name="mod.target_fn",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "mod.shared_fn",
                            "source_entity_type": "Function",
                            "source_item_id": item_a_id,  # pins to src_a only
                            "target_qname": "mod.target_fn",
                            "target_entity_type": "Function",
                            "relation_type": "CALLS",
                            "confidence": 0.9,
                            "metadata": {},
                        }
                    ]
                },
            }
        )
    )

    m = result.manage
    assert m.success
    assert len(m.relation_ids) == 1
    assert m.relation_ids[0] is not None, "Expected UUID; source narrowing should succeed"
    assert m.created_count == 1
    assert m.existing_count == 0
    assert m.unresolved_or_ambiguous == []

    # Verify the inserted relation links the correct source entity.
    row = await knowledge_backend.query(
        "SELECT source_entity_id, target_entity_id FROM knowledge_relations WHERE id = $1::uuid",
        (m.relation_ids[0],),
    )
    assert len(row.rows) == 1
    assert str(row.rows[0]["source_entity_id"]) == src_a_id
    assert str(row.rows[0]["target_entity_id"]) == tgt_id


# ===========================================================================
# Test 5c — mixed batch: one success + one unresolved; counts and order correct
# ===========================================================================


async def test_mixed_batch_success_and_unresolved_in_order(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Mixed batch: first relation succeeds, second is unresolved.

    Verifies:
    - relation_ids preserves input order (UUID at index 0, None at index 1).
    - created_count == 1, existing_count == 0.
    - unresolved_or_ambiguous has exactly one entry at index 1.
    - The successful relation is committed; the unresolved one is not inserted.
    """
    src_id = await _seed_entity(
        memory_service,
        entity_type="Function",
        name="batch_src",
        stable_id="mixed::src",
        qualified_name="mod.batch_src",
    )
    tgt_id = await _seed_entity(
        memory_service,
        entity_type="Function",
        name="batch_tgt",
        stable_id="mixed::tgt",
        qualified_name="mod.batch_tgt",
    )
    # "mod.ghost" is intentionally absent — will be unresolved.

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "external_fallback": "reject",  # ensure no ghost entity is auto-created
                    "relations": [
                        {
                            # index 0 — will succeed
                            "source_qname": "mod.batch_src",
                            "source_entity_type": "Function",
                            "target_qname": "mod.batch_tgt",
                            "target_entity_type": "Function",
                            "relation_type": "CALLS",
                            "confidence": 0.85,
                            "metadata": {},
                        },
                        {
                            # index 1 — target does not exist → unresolved (reject mode)
                            "source_qname": "mod.batch_src",
                            "source_entity_type": "Function",
                            "target_qname": "mod.ghost",
                            "target_entity_type": "Function",
                            "relation_type": "CALLS",
                            "confidence": 0.85,
                            "metadata": {},
                        },
                    ],
                },
            }
        )
    )

    m = result.manage
    assert m.success
    assert len(m.relation_ids) == 2
    assert m.relation_ids[0] is not None, "Index 0 should be a UUID (successful insert)"
    assert m.relation_ids[1] is None, "Index 1 should be None (unresolved target)"
    assert m.created_count == 1
    assert m.existing_count == 0
    assert len(m.unresolved_or_ambiguous) == 1
    entry = m.unresolved_or_ambiguous[0]
    assert entry["index"] == 1
    assert entry["reason"] in ("unresolved_target", "reject")

    # Confirm exactly one relation was committed.
    committed = await knowledge_backend.query(
        "SELECT source_entity_id, target_entity_id FROM knowledge_relations r "
        "JOIN knowledge_entities e ON r.source_entity_id = e.id "
        "WHERE e.palace = $1",
        (PALACE,),
    )
    assert len(committed.rows) == 1
    assert str(committed.rows[0]["source_entity_id"]) == src_id
    assert str(committed.rows[0]["target_entity_id"]) == tgt_id


# ===========================================================================
# Test 6 — EXTERNAL fallback: unresolved target creates Module; second call reuses it
# ===========================================================================


async def test_external_fallback_creates_and_reuses_module(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Unresolved import target → new EXTERNAL Module; second call reuses it."""
    await _seed_entity(
        memory_service,
        entity_type="Module",
        name="my_module",
        stable_id="ext::my_module",
        qualified_name="my_module",
    )

    payload = {
        "operation": "store_relations_by_qname",
        "scope": _scope(),
        "graph": {
            "relations": [
                {
                    "source_qname": "my_module",
                    "source_entity_type": "Module",
                    "target_qname": "numpy",
                    "target_entity_type": "Module",
                    "relation_type": "IMPORTS",
                    "confidence": 0.9,
                    "metadata": {},
                }
            ],
            "external_fallback": "module",
        },
    }

    r1 = await memory_service.execute(MemoryRequest.model_validate(payload))
    m1 = r1.manage
    assert m1.success
    rel_id = m1.relation_ids[0]
    assert rel_id is not None
    assert m1.created_count == 1
    assert m1.existing_count == 0
    assert len(m1.external_entities_created) == 1
    ext_entry = m1.external_entities_created[0]
    assert ext_entry["qname"] == "numpy"
    ext_entity_id = ext_entry["entity_id"]

    # Verify EXTERNAL entity has correct attributes
    rows = await knowledge_backend.query(
        "SELECT entity_type, qualified_name, source, confidence, metadata "
        "FROM knowledge_entities WHERE id = $1::uuid",
        (ext_entity_id,),
    )
    assert len(rows.rows) == 1
    row = rows.rows[0]
    assert row["entity_type"] == "Module"
    assert row["qualified_name"] == "numpy"
    assert row["source"] == "STRUCTURAL"
    assert row["confidence"] <= 0.35  # low confidence (~0.3)
    meta = row["metadata"]
    if isinstance(meta, str):
        import json as _json

        meta = _json.loads(meta)
    assert meta.get("external") is True

    # Second call: reuses external entity, no new external entity created
    r2 = await memory_service.execute(MemoryRequest.model_validate(payload))
    m2 = r2.manage
    assert m2.success
    assert m2.created_count == 0
    assert m2.existing_count == 1
    assert m2.relation_ids[0] == rel_id
    # external_entities_created should be empty on reuse (already existed)
    assert len(m2.external_entities_created) == 0

    # Confirm only one EXTERNAL Module entity for "numpy"
    count = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities "
        "WHERE palace = $1 AND qualified_name = 'numpy' AND source = 'STRUCTURAL'",
        (PALACE,),
    )
    assert count.rows[0]["n"] == 1


# ===========================================================================
# Test 7 — Reject mode: unresolved target with external_fallback=reject → null, unresolved entry
# ===========================================================================


async def test_reject_mode_unresolved_target(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """external_fallback=reject with unresolved target: no entity/relation created."""
    await _seed_entity(
        memory_service,
        entity_type="Module",
        name="my_module",
        stable_id="rej::my_module",
        qualified_name="rej.my_module",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "rej.my_module",
                            "source_entity_type": "Module",
                            "target_qname": "unknown.external.pkg",
                            "target_entity_type": "Module",
                            "relation_type": "IMPORTS",
                            "confidence": 0.8,
                            "metadata": {},
                        }
                    ],
                    "external_fallback": "reject",
                },
            }
        )
    )

    m = result.manage
    assert m.success
    assert m.relation_ids[0] is None
    assert len(m.unresolved_or_ambiguous) == 1
    entry = m.unresolved_or_ambiguous[0]
    assert entry["index"] == 0
    assert "unresolved" in entry["reason"]

    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations r "
        "JOIN knowledge_entities e ON r.source_entity_id = e.id "
        "WHERE e.palace = $1",
        (PALACE,),
    )
    assert rows.rows[0]["n"] == 0

    # No EXTERNAL entity created
    ext_rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities "
        "WHERE palace = $1 AND qualified_name = 'unknown.external.pkg'",
        (PALACE,),
    )
    assert ext_rows.rows[0]["n"] == 0


# ===========================================================================
# Test 8 — Invalid relation_type raises MEM_INVALID_RELATION_TYPE
# ===========================================================================


async def test_invalid_relation_type_rejected(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """relation_type=MENTIONS → MEM_INVALID_RELATION_TYPE (not a STRUCTURAL type)."""
    await _seed_entity(
        memory_service,
        entity_type="Function",
        name="fn",
        stable_id="inv::fn",
        qualified_name="inv.pkg.fn",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "inv.pkg.fn",
                            "source_entity_type": "Function",
                            "target_qname": "inv.pkg.fn",
                            "target_entity_type": "Function",
                            "relation_type": "MENTIONS",
                            "confidence": 0.5,
                            "metadata": {},
                        }
                    ]
                },
            }
        )
    )

    m = result.manage
    assert not m.success
    assert m.error is not None
    assert "MEM_INVALID_RELATION_TYPE" in m.error


# ===========================================================================
# Test 9 — Empty relations raises MEM_FIELD_REQUIRED
# ===========================================================================


async def test_empty_relations_raises_field_required(
    memory_service: MemoryService, clean_palace: None
) -> None:
    """Empty graph.relations list → MEM_FIELD_REQUIRED."""
    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [],
                },
            }
        )
    )

    m = result.manage
    assert not m.success
    assert m.error is not None
    assert "MEM_FIELD_REQUIRED" in m.error


# ===========================================================================
# Test 9b — Source ambiguity surfaces reason="ambiguous" (ADR-012 unified contract)
# ===========================================================================


async def test_source_ambiguity_reason_is_ambiguous(
    memory_service: MemoryService, knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Two source entities share qname+entity_type in different rooms → reason='ambiguous'.

    ADR-012 mandates a unified 'ambiguous' reason for multiple-match scenarios regardless
    of whether the ambiguous end is source or target.  This test seeds two Function entities
    with the same qualified_name in different rooms (bypassing the unique name constraint),
    supplies one unambiguous target, and calls store_relations_by_qname without
    source_item_id.  The result must carry reason='ambiguous' and list both candidates.
    """
    # Two source candidates — same palace/entity_type/qname, different rooms.
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source,
             stable_id, qualified_name)
        VALUES ($1, $2, 'src_room_a', 'qname_rel', 'Function', 'ambig_src', 'STRUCTURAL',
                'src_amb::a', 'mod.ambig_src')
        """,
        (PALACE, WING),
    )
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source,
             stable_id, qualified_name)
        VALUES ($1, $2, 'src_room_b', 'qname_rel', 'Function', 'ambig_src', 'STRUCTURAL',
                'src_amb::b', 'mod.ambig_src')
        """,
        (PALACE, WING),
    )

    # One unambiguous target.
    await _seed_entity(
        memory_service,
        entity_type="Function",
        name="clear_target",
        stable_id="src_amb::target",
        qualified_name="mod.clear_target",
    )

    result = await memory_service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "store_relations_by_qname",
                "scope": _scope(),
                "graph": {
                    "relations": [
                        {
                            "source_qname": "mod.ambig_src",
                            "source_entity_type": "Function",
                            "target_qname": "mod.clear_target",
                            "target_entity_type": "Function",
                            # No source_item_id — must surface source ambiguity.
                            "relation_type": "CALLS",
                            "confidence": 0.9,
                            "metadata": {},
                        }
                    ]
                },
            }
        )
    )

    m = result.manage
    assert m.success
    assert m.relation_ids[0] is None  # No insert when source is ambiguous.
    assert len(m.unresolved_or_ambiguous) == 1
    entry = m.unresolved_or_ambiguous[0]
    assert entry["reason"] == "ambiguous", (
        f"Expected reason='ambiguous' (ADR-012 unified contract), got {entry['reason']!r}"
    )
    assert "candidates" in entry
    assert len(entry["candidates"]) == 2

    # Confirm no relation was inserted.
    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations r "
        "JOIN knowledge_entities e ON r.source_entity_id = e.id "
        "WHERE e.palace = $1",
        (PALACE,),
    )
    assert rows.rows[0]["n"] == 0


# ===========================================================================
# Test 10 — Resolution latency benchmark on 10k STRUCTURAL entity fixture
# ===========================================================================
# Opt-in: run with `uv run pytest -m slow tests/test_memory_store_relations_by_qname.py -v -s`
# Seeds 10k entities in the existing DB fixture, measures resolution query latency,
# and asserts median per-relation execution time < 5ms (ADR-012 requirement).


@pytest.mark.slow
async def test_explain_analyze_benchmark(
    knowledge_backend: PostgresBackend, clean_palace: None
) -> None:
    """Resolution latency < 5ms median on 10k STRUCTURAL entities (ADR-012).

    Seeds 10k STRUCTURAL Module entities via a single bulk INSERT, then samples
    100 resolution queries and measures wall-clock execution time per query.

    To run:
        uv run pytest -m slow \
            tests/test_memory_store_relations_by_qname.py::test_explain_analyze_benchmark \
            -v -s

    Index being exercised:
        SELECT id, source_item_id FROM knowledge_entities
        WHERE palace = $1 AND source = 'STRUCTURAL'
          AND entity_type = $2 AND qualified_name = $3

    Threshold breach -> re-evaluate; do NOT silently degrade.
    """
    import time

    bench_palace = f"{PALACE}_bench"

    # Seed 10k STRUCTURAL entities via generate_series (single round-trip).
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source,
             stable_id, qualified_name)
        SELECT
            $1,
            'code',
            'bench_room',
            'bench_corridor',
            'Module',
            'bench_mod_' || i,
            'STRUCTURAL',
            'bench_stable_' || i,
            'bench.module.' || i
        FROM generate_series(1, 10000) AS i
        ON CONFLICT DO NOTHING
        """,
        (bench_palace,),
    )

    # Sample 100 qualified_names to probe.
    sample = await knowledge_backend.query(
        "SELECT qualified_name, entity_type FROM knowledge_entities "
        "WHERE palace = $1 AND source = 'STRUCTURAL' AND qualified_name IS NOT NULL "
        "ORDER BY random() LIMIT 100",
        (bench_palace,),
    )
    assert len(sample.rows) == 100, (
        f"Expected 100 sample rows, got {len(sample.rows)}; seeding may have failed."
    )

    # Measure wall-clock time for each resolution query (the exact query
    # used by _resolve_entity_by_qname).
    latencies: list[float] = []
    for row in sample.rows:
        t0 = time.perf_counter()
        await knowledge_backend.query(
            "SELECT id, source_item_id FROM knowledge_entities "
            "WHERE palace = $1 AND source = 'STRUCTURAL' "
            "AND entity_type = $2 AND qualified_name = $3",
            (bench_palace, row["entity_type"], row["qualified_name"]),
        )
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)
    median = latencies[n // 2]
    p95 = latencies[int(n * 0.95)]

    print(  # noqa: T201
        f"\nBenchmark (10k entities, {n} queries): median={median:.3f}ms  p95={p95:.3f}ms"
    )

    assert median < 5.0, (
        f"Median resolution latency {median:.2f}ms exceeds 5ms threshold on 10k entities. "
        "Consider adding a (palace, source, entity_type, qualified_name) index or "
        "verifying existing partial index coverage. Do NOT silently degrade."
    )
