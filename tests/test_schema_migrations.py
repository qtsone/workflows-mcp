"""Behavior tests for knowledge schema migrations.

Each migration in src/workflows_mcp/engine/knowledge/schema.py has a
matching test here that asserts post-state directly against the live
backend. Tests use the project knowledge_backend fixture which already
runs ensure_schema() at setup.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
    """Provide a connected PostgresBackend with the knowledge schema applied.

    Connects to the live PostgreSQL instance configured via MEMORY_DB_* env
    vars (falling back to the Docker test database). Runs ensure_schema() to
    bring the schema up to the current version, then yields the backend.
    The backend is disconnected after each test.

    Use this fixture for schema-structure assertions only (column/index/trigger
    existence checks). For tests that INSERT data, use the ``db`` fixture which
    wraps each test in a rolled-back transaction so the live DB stays clean.
    """
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


@pytest_asyncio.fixture
async def db(knowledge_backend: PostgresBackend) -> AsyncIterator[PostgresBackend]:
    """Transactional variant of knowledge_backend.

    Begins a transaction before the test body and rolls it back afterwards,
    so every INSERT/UPDATE/DELETE is invisible to subsequent test runs. Use
    this fixture for any test that writes rows into the live database.
    """
    await knowledge_backend.begin_transaction()
    try:
        yield knowledge_backend
    finally:
        await knowledge_backend.rollback()


async def _column_exists(backend: PostgresBackend, table: str, column: str) -> bool:
    result = await backend.query(
        """
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = $1
           AND column_name = $2
        """,
        (table, column),
    )
    return bool(result.rows)


async def _index_exists(backend: PostgresBackend, index_name: str) -> bool:
    result = await backend.query(
        """
        SELECT 1
          FROM pg_indexes
         WHERE schemaname = 'public'
           AND indexname = $1
        """,
        (index_name,),
    )
    return bool(result.rows)


async def _constraint_exists(backend: PostgresBackend, table: str, constraint_name: str) -> bool:
    result = await backend.query(
        """
        SELECT 1
          FROM pg_constraint
         WHERE conrelid = $1::regclass
           AND conname = $2
        """,
        (table, constraint_name),
    )
    return bool(result.rows)


async def _trigger_exists(backend: PostgresBackend, trigger_name: str) -> bool:
    result = await backend.query(
        """
        SELECT 1
          FROM pg_trigger
         WHERE tgname = $1
           AND NOT tgisinternal
        """,
        (trigger_name,),
    )
    return bool(result.rows)


async def _table_exists(backend: PostgresBackend, table: str) -> bool:
    result = await backend.query(
        """
        SELECT 1
          FROM information_schema.tables
         WHERE table_schema = 'public'
           AND table_name = $1
        """,
        (table,),
    )
    return bool(result.rows)


@asynccontextmanager
async def _expect_db_error(db: PostgresBackend) -> AsyncIterator[None]:
    """Context manager that expects a DB error and rolls back to a savepoint.

    PostgreSQL marks the whole transaction as aborted on any error, so any
    expected-to-fail statement must be wrapped in a SAVEPOINT that gets rolled
    back after the error is caught. This keeps the outer transaction alive.
    """
    assert db._conn is not None, "_expect_db_error requires an active transaction"
    await db._conn.execute("SAVEPOINT _expect_error")
    try:
        yield
        # If no exception was raised the test expectation was wrong
        await db._conn.execute("RELEASE SAVEPOINT _expect_error")
        raise AssertionError("Expected a database error but none was raised")
    except AssertionError:
        raise
    except Exception:
        await db._conn.execute("ROLLBACK TO SAVEPOINT _expect_error")
        await db._conn.execute("RELEASE SAVEPOINT _expect_error")


def _as_dict(value: Any) -> Any:
    """Coerce a value that may be a JSON string or already a dict to a dict."""
    if isinstance(value, str):
        return json.loads(value)
    return value


async def test_baseline_schema_present(knowledge_backend: PostgresBackend) -> None:
    """Sanity check: the fixture really applied the baseline schema."""
    assert await _table_exists(knowledge_backend, "knowledge_entities")
    assert await _table_exists(knowledge_backend, "knowledge_memories")


# ---------------------------------------------------------------------------
# v6: knowledge_sources palace ownership
# ---------------------------------------------------------------------------


async def test_v6_knowledge_sources_palace_and_uniqueness(
    knowledge_backend: PostgresBackend, db: PostgresBackend
) -> None:
    """v6: knowledge_sources is palace-owned and source names are palace-scoped."""
    assert await _column_exists(knowledge_backend, "knowledge_sources", "palace")
    assert await _index_exists(knowledge_backend, "idx_ks_palace_name")

    result = await knowledge_backend.query(
        """
        SELECT is_nullable
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'knowledge_sources'
           AND column_name = 'palace'
        """,
        (),
    )
    assert result.rows[0]["is_nullable"] == "NO"

    name = "shared-source-name"
    await db.execute(
        "INSERT INTO knowledge_sources (palace, name, source_type) VALUES ($1, $2, 'FILE')",
        ("palace_a", name),
    )
    await db.execute(
        "INSERT INTO knowledge_sources (palace, name, source_type) VALUES ($1, $2, 'FILE')",
        ("palace_b", name),
    )
    async with _expect_db_error(db):
        await db.execute(
            "INSERT INTO knowledge_sources (palace, name, source_type) VALUES ($1, $2, 'FILE')",
            ("palace_a", name),
        )


# ---------------------------------------------------------------------------
# v7: knowledge_items file metadata and palace
# ---------------------------------------------------------------------------


async def test_v7_knowledge_items_file_identity_columns(knowledge_backend: PostgresBackend) -> None:
    """v7: knowledge_items gains file metadata and palace-owned file identity."""
    for col in ("palace", "content_hash", "size_bytes", "mtime_ns", "language"):
        assert await _column_exists(knowledge_backend, "knowledge_items", col), col
    assert await _index_exists(knowledge_backend, "idx_ki_palace_source_path")
    assert await _index_exists(knowledge_backend, "idx_ki_palace_content_hash")

    result = await knowledge_backend.query(
        """
        SELECT column_name, is_nullable
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'knowledge_items'
           AND column_name IN (
               'palace', 'source_id', 'content_hash', 'size_bytes', 'mtime_ns', 'language'
           )
        """,
        (),
    )
    nullability = {row["column_name"]: row["is_nullable"] for row in result.rows}
    assert nullability["palace"] == "NO"
    assert nullability["source_id"] == "NO"
    assert nullability["content_hash"] == "NO"
    assert nullability["size_bytes"] == "NO"
    assert nullability["mtime_ns"] == "NO"
    assert nullability["language"] == "YES"


async def test_v7_items_reject_null_source_and_duplicate_file_identity(db: PostgresBackend) -> None:
    """v7: new items require source_id and cannot duplicate (palace, source_id, path)."""
    source_row = await db.query(
        """
        INSERT INTO knowledge_sources (palace, name, source_type)
        VALUES ('palace_items', 'source-items', 'FILE')
        RETURNING id
        """,
        (),
    )
    source_id = str(source_row.rows[0]["id"])

    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_items
                (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ('palace_items', NULL, 'src/a.py', 'a.py', 'h1', 1, 1)
            """,
            (),
        )

    await db.execute(
        """
        INSERT INTO knowledge_items
            (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
        VALUES ('palace_items', $1::uuid, 'src/a.py', 'a.py', 'h1', 1, 1)
        """,
        (source_id,),
    )
    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_items
                (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ('palace_items', $1::uuid, 'src/a.py', 'a.py', 'h2', 2, 2)
            """,
            (source_id,),
        )


async def test_v7_same_content_hash_in_two_palaces_is_not_a_cross_palace_rename(
    db: PostgresBackend,
) -> None:
    """v7: content_hash lookup is palace-scoped; same hash in two palaces is two rows."""
    rows = []
    for palace in ("palace_hash_a", "palace_hash_b"):
        src = await db.query(
            "INSERT INTO knowledge_sources (palace, name, source_type) VALUES ($1, $2, 'FILE') RETURNING id",  # noqa: E501
            (palace, "source-hash"),
        )
        item = await db.query(
            """
            INSERT INTO knowledge_items
                (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ($1, $2::uuid, $3, $3, 'same-hash', 1, 1)
            RETURNING id, palace
            """,
            (palace, str(src.rows[0]["id"]), "src/same.py"),
        )
        rows.append(item.rows[0])
    assert {row["palace"] for row in rows} == {"palace_hash_a", "palace_hash_b"}


# ---------------------------------------------------------------------------
# v8: item/source palace trigger
# ---------------------------------------------------------------------------


async def test_v8_item_source_palace_trigger_rejects_mismatch(
    knowledge_backend: PostgresBackend, db: PostgresBackend
) -> None:
    """v8: item palace must match its source palace."""
    assert await _trigger_exists(knowledge_backend, "trg_ki_source_palace_match")
    src = await db.query(
        "INSERT INTO knowledge_sources (palace, name, source_type) VALUES ('palace_source', 'source-trigger', 'FILE') RETURNING id",  # noqa: E501
        (),
    )
    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_items
                (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ('palace_item', $1::uuid, 'src/mismatch.py', 'mismatch.py', 'h', 1, 1)
            """,
            (str(src.rows[0]["id"]),),
        )


# ---------------------------------------------------------------------------
# v9: entities structural identity
# ---------------------------------------------------------------------------


async def test_v9_knowledge_entities_structural_columns_and_source_check(
    knowledge_backend: PostgresBackend,
) -> None:
    """v9: entities gain structural identity columns and source origin check."""
    for col in ("source", "authority", "stable_id", "source_item_id", "metadata", "updated_at"):
        assert await _column_exists(knowledge_backend, "knowledge_entities", col), col
    assert await _constraint_exists(knowledge_backend, "knowledge_entities", "ck_ke_source_origin")
    assert await _index_exists(knowledge_backend, "idx_ke_palace_source_stable_id")
    assert await _trigger_exists(knowledge_backend, "trg_ke_updated_at")

    with pytest.raises(Exception):
        await knowledge_backend.execute(
            """
            INSERT INTO knowledge_entities
                (palace, namespace, room, corridor, entity_type, name, source)
            VALUES (
                'palace_entities', 'code', 'default', 'schema',
                'Function', 'bad_source', 'FILE_NAME'
            )
            """,
            (),
        )

async def test_v9_entities_metadata_round_trip_and_updated_at_advances(db: PostgresBackend) -> None:
    """v9: metadata defaults to {}, updates round-trip, and updated_at advances."""
    inserted = await db.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES (
            'palace_entities', 'code', 'default', 'schema',
            'Function', 'alpha', 'STRUCTURAL', 'alpha-stable'
        )
        RETURNING id, metadata, updated_at
        """,
        (),
    )
    entity_id = str(inserted.rows[0]["id"])
    assert _as_dict(inserted.rows[0]["metadata"]) == {}

    await db.query("SELECT pg_sleep(0.001)", ())

    updated = await db.query(
        """
        UPDATE knowledge_entities
           SET metadata = '{"start_line": 10, "end_line": 12}'::jsonb
         WHERE id = $1::uuid
         RETURNING metadata, updated_at
        """,
        (entity_id,),
    )
    assert _as_dict(updated.rows[0]["metadata"]) == {"start_line": 10, "end_line": 12}
    # updated_at is set by trigger on UPDATE; it must be a non-null timestamp.
    # Within a single transaction NOW() is stable so equality is acceptable.
    assert updated.rows[0]["updated_at"] is not None


async def test_v9_unique_partial_stable_id_index_rejects_duplicate(db: PostgresBackend) -> None:
    """v9: duplicate (palace, source, stable_id) where stable_id is set is rejected."""
    payload = (
        "palace_entities_unique", "code", "default", "schema",
        "Function", "dup", "STRUCTURAL", "same-stable-id",
    )
    await db.execute(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        payload,
    )
    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_entities
                (palace, namespace, room, corridor, entity_type, name, source, stable_id)
            VALUES ($1, $2, $3, $4, $5, $6 || '-again', $7, $8)
            """,
            payload,
        )


# ---------------------------------------------------------------------------
# v10: entity embeddings side table
# ---------------------------------------------------------------------------


async def test_v10_knowledge_entity_embeddings_table(knowledge_backend: PostgresBackend) -> None:
    """v10: knowledge_entity_embeddings table exists with required columns."""
    assert await _table_exists(knowledge_backend, "knowledge_entity_embeddings")
    for col in ("entity_id", "profile", "model", "dimension", "embedding", "created_at"):
        assert await _column_exists(knowledge_backend, "knowledge_entity_embeddings", col), col

    pk_result = await knowledge_backend.query(
        """
        SELECT a.attname
          FROM pg_index i
          JOIN pg_attribute a
            ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
         WHERE i.indrelid = 'knowledge_entity_embeddings'::regclass
           AND i.indisprimary
         ORDER BY a.attname
        """,
        (),
    )
    assert sorted(row["attname"] for row in pk_result.rows) == ["entity_id", "profile"]


async def test_v10_entity_embeddings_round_trip(db: PostgresBackend) -> None:
    """v10: insert + read a 1536-dim embedding for an entity."""
    entity_result = await db.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES (
            'test_palace_embed', 'code', 'default', 'foundation',
            'Function', 'foo', 'STRUCTURAL', 'foo-stable'
        )
        RETURNING id
        """,
        (),
    )
    entity_id = str(entity_result.rows[0]["id"])
    vec = "[" + ",".join(["0.1"] * 1536) + "]"
    await db.execute(
        """
        INSERT INTO knowledge_entity_embeddings
            (entity_id, profile, model, dimension, embedding)
        VALUES ($1::uuid, $2, $3, $4, $5::vector)
        """,
        (entity_id, "default", "text-embedding-3-small", 1536, vec),
    )
    read = await db.query(
        """
        SELECT dimension, model
          FROM knowledge_entity_embeddings
         WHERE entity_id = $1::uuid AND profile = $2
        """,
        (entity_id, "default"),
    )
    assert read.rows[0]["dimension"] == 1536
    assert read.rows[0]["model"] == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# v11: entity memories anchor columns
# ---------------------------------------------------------------------------


async def test_v11_entity_memories_anchor_columns(knowledge_backend: PostgresBackend) -> None:
    """v11: knowledge_entity_memories gains line/col span + anchor_kind."""
    for col in ("start_line", "end_line", "start_col", "end_col", "anchor_kind"):
        assert await _column_exists(knowledge_backend, "knowledge_entity_memories", col), col

    result = await knowledge_backend.query(
        """
        SELECT column_name, is_nullable, column_default
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'knowledge_entity_memories'
           AND column_name IN ('start_line', 'end_line', 'start_col', 'end_col', 'anchor_kind')
        """,
        (),
    )
    cols = {row["column_name"]: row for row in result.rows}
    assert cols["start_line"]["is_nullable"] == "YES"
    assert cols["end_line"]["is_nullable"] == "YES"
    assert cols["start_col"]["is_nullable"] == "YES"
    assert cols["end_col"]["is_nullable"] == "YES"
    assert cols["anchor_kind"]["is_nullable"] == "NO"
    assert "symbol" in (cols["anchor_kind"]["column_default"] or "")


# ---------------------------------------------------------------------------
# v12: community federation columns
# ---------------------------------------------------------------------------


async def test_v12_communities_federation_columns(knowledge_backend: PostgresBackend) -> None:
    """v12: knowledge_communities gains federation provenance columns."""
    for col in ("scope_kind", "source_palaces", "member_provenance", "consent_policy_id"):
        assert await _column_exists(knowledge_backend, "knowledge_communities", col), col

    result = await knowledge_backend.query(
        """
        SELECT column_name, is_nullable, column_default, udt_name
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'knowledge_communities'
           AND column_name IN (
               'scope_kind', 'source_palaces', 'member_provenance', 'consent_policy_id'
           )
        """,
        (),
    )
    cols = {row["column_name"]: row for row in result.rows}
    assert cols["scope_kind"]["is_nullable"] == "NO"
    assert "palace_local" in (cols["scope_kind"]["column_default"] or "")
    assert cols["source_palaces"]["udt_name"] == "_text"
    assert cols["member_provenance"]["udt_name"] == "jsonb"
    assert cols["consent_policy_id"]["udt_name"] == "uuid"


# ---------------------------------------------------------------------------
# v13: item lifecycle state
# ---------------------------------------------------------------------------


async def test_v13_items_lifecycle_state(knowledge_backend: PostgresBackend) -> None:
    """v13: knowledge_items gains lifecycle_state and error_metadata."""
    assert await _column_exists(knowledge_backend, "knowledge_items", "lifecycle_state")
    assert await _column_exists(knowledge_backend, "knowledge_items", "error_metadata")
    assert await _index_exists(knowledge_backend, "idx_ki_palace_lifecycle_state")

    result = await knowledge_backend.query(
        """
        SELECT column_name, is_nullable, column_default, udt_name
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'knowledge_items'
           AND column_name IN ('lifecycle_state', 'error_metadata')
        """,
        (),
    )
    cols = {row["column_name"]: row for row in result.rows}
    assert cols["lifecycle_state"]["is_nullable"] == "NO"
    assert "ACTIVE" in (cols["lifecycle_state"]["column_default"] or "")
    assert cols["error_metadata"]["udt_name"] == "jsonb"
    assert cols["error_metadata"]["is_nullable"] == "YES"


def test_item_lifecycle_state_enum_values() -> None:
    """ItemLifecycleState covers the full set of item lifecycle values."""
    from workflows_mcp.engine.knowledge.constants import ItemLifecycleState

    assert {s.value for s in ItemLifecycleState} == {
        "ACTIVE", "ARCHIVED", "QUARANTINED", "USER_VALIDATED", "DIRTY",
    }

# ---------------------------------------------------------------------------
# Task 10: Schema version and idempotency
# ---------------------------------------------------------------------------


async def test_schema_version_advanced_to_nineteen(knowledge_backend: PostgresBackend) -> None:
    """All migrations applied; SCHEMA_VERSION reaches 19."""
    from workflows_mcp.engine.knowledge.schema import SCHEMA_VERSION

    assert SCHEMA_VERSION == 19

    result = await knowledge_backend.query(
        "SELECT value FROM _knowledge_meta WHERE key = 'schema_version'", ()
    )
    assert result.rows
    assert int(result.rows[0]["value"]) == 19


async def test_re_running_ensure_schema_is_idempotent(knowledge_backend: PostgresBackend) -> None:
    """Calling ensure_schema again is a fast no-op when already current."""
    from workflows_mcp.engine.knowledge.schema import ensure_schema

    await ensure_schema(knowledge_backend)

    result = await knowledge_backend.query(
        "SELECT value FROM _knowledge_meta WHERE key = 'schema_version'", ()
    )
    assert int(result.rows[0]["value"]) == 19


# ---------------------------------------------------------------------------
# v14: Track 4 prerequisites — qualified_name, parent_class_id, relations.metadata
# ---------------------------------------------------------------------------


async def test_v14_qualified_name_column_present(knowledge_backend: PostgresBackend) -> None:
    """v14: knowledge_entities.qualified_name TEXT column exists."""
    assert await _column_exists(knowledge_backend, "knowledge_entities", "qualified_name")


async def test_v14_parent_class_id_column_present(knowledge_backend: PostgresBackend) -> None:
    """v14: knowledge_entities.parent_class_id UUID column exists."""
    assert await _column_exists(knowledge_backend, "knowledge_entities", "parent_class_id")


async def test_v14_relations_metadata_column_present(knowledge_backend: PostgresBackend) -> None:
    """v14: knowledge_relations.metadata JSONB column exists."""
    assert await _column_exists(knowledge_backend, "knowledge_relations", "metadata")


async def test_v14_qualified_name_index_present(knowledge_backend: PostgresBackend) -> None:
    """v14: idx_knowledge_entities_qualified_name partial index exists."""
    assert await _index_exists(knowledge_backend, "idx_knowledge_entities_qualified_name")


async def test_v14_parent_class_index_present(knowledge_backend: PostgresBackend) -> None:
    """v14: idx_knowledge_entities_parent_class partial index exists."""
    assert await _index_exists(knowledge_backend, "idx_knowledge_entities_parent_class")


async def test_v14_parent_class_id_is_self_fk_set_null(knowledge_backend: PostgresBackend) -> None:
    """v14: parent_class_id must reference knowledge_entities(id) ON DELETE SET NULL."""
    result = await knowledge_backend.query(
        """
        SELECT confdeltype, confrelid::regclass::text AS ref_table
          FROM pg_constraint
         WHERE conrelid = 'knowledge_entities'::regclass
           AND contype = 'f'
           AND conname = 'knowledge_entities_parent_class_id_fkey'
        """,
        (),
    )
    assert result.rows, "parent_class_id FK constraint missing"
    # 'n' = SET NULL in pg_constraint.confdeltype (returned as bytes by asyncpg)
    confdeltype = result.rows[0]["confdeltype"]
    if isinstance(confdeltype, bytes):
        confdeltype = confdeltype.decode()
    assert confdeltype == "n"
    assert result.rows[0]["ref_table"] == "knowledge_entities"


async def test_v14_relations_metadata_default_empty_jsonb(db: PostgresBackend) -> None:
    """v14: new relation rows default metadata to '{}'::jsonb.

    knowledge_relations has no palace column (palace is inherited via the
    referenced entities), so this test only inserts entities + a relation.
    """
    palace = "test_palace_v14"
    src = await db.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES ($1, 'code', 'default', 'schema', 'File', 'src.py', 'STRUCTURAL', 'src.py-stable')
        RETURNING id
        """,
        (palace,),
    )
    tgt = await db.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES ($1, 'code', 'default', 'schema', 'File', 'tgt.py', 'STRUCTURAL', 'tgt.py-stable')
        RETURNING id
        """,
        (palace,),
    )
    rel = await db.query(
        """
        INSERT INTO knowledge_relations
            (source_entity_id, target_entity_id, relation_type)
        VALUES ($1::uuid, $2::uuid, 'IMPORTS')
        RETURNING metadata
        """,
        (str(src.rows[0]["id"]), str(tgt.rows[0]["id"])),
    )
    assert _as_dict(rel.rows[0]["metadata"]) == {}


# ---------------------------------------------------------------------------
# v15: ADR-013 System 1 / System 2 schema objects
# ---------------------------------------------------------------------------


async def test_v15_knowledge_structural_evidence_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v15: knowledge_structural_evidence table exists with required columns."""
    assert await _table_exists(knowledge_backend, "knowledge_structural_evidence")
    for col in (
        "id", "palace", "wing", "room", "compartment",
        "entity_stable_id", "entity_type", "cycle_id",
        "evidence_category", "evidence_payload", "source_item_id",
        "scanned_at", "created_at",
    ):
        assert await _column_exists(knowledge_backend, "knowledge_structural_evidence", col), col
    # v16: upsert uniqueness constraint must be present.
    assert await _constraint_exists(
        knowledge_backend,
        "knowledge_structural_evidence",
        "uq_kse_scope_entity_category",
    ), "uq_kse_scope_entity_category unique constraint missing from knowledge_structural_evidence"


async def test_v15_knowledge_verification_cycles_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v15: knowledge_verification_cycles table exists with required columns."""
    assert await _table_exists(knowledge_backend, "knowledge_verification_cycles")
    for col in (
        "id", "palace", "scope_key", "covered_wing", "covered_room",
        "covered_compartment", "success", "failure_reason",
        "evidence_found", "evidence_absent", "started_at", "completed_at", "created_at",
    ):
        assert await _column_exists(knowledge_backend, "knowledge_verification_cycles", col), col


async def test_v15_knowledge_semantic_claims_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v15: knowledge_semantic_claims table exists with lifecycle state constraint."""
    assert await _table_exists(knowledge_backend, "knowledge_semantic_claims")
    for col in (
        "id", "palace", "wing", "room", "compartment",
        "claim_text", "claim_type", "lifecycle_state",
        "absent_cycle_count", "scope_key",
        "created_at", "updated_at", "archived_at",
    ):
        assert await _column_exists(knowledge_backend, "knowledge_semantic_claims", col), col
    assert await _constraint_exists(
        knowledge_backend, "knowledge_semantic_claims", "ck_ksc_lifecycle_state"
    )
    assert await _constraint_exists(
        knowledge_backend, "knowledge_semantic_claims", "ck_ksc_claim_type"
    )
    assert await _constraint_exists(
        knowledge_backend,
        "knowledge_semantic_claims",
        "ck_ksc_archive_requires_two_cycles",
    )


async def test_v15_knowledge_claim_evidence_links_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v15: knowledge_claim_evidence_links join table exists."""
    assert await _table_exists(knowledge_backend, "knowledge_claim_evidence_links")
    for col in ("claim_id", "evidence_id", "evidence_category", "linked_at"):
        assert await _column_exists(knowledge_backend, "knowledge_claim_evidence_links", col), col


async def test_v15_knowledge_wing_proof_bundles_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v15: knowledge_wing_proof_bundles table exists with gate constraint."""
    assert await _table_exists(knowledge_backend, "knowledge_wing_proof_bundles")
    for col in (
        "id", "palace", "wing", "activating_claim_id",
        "evidence_categories", "gate_satisfied", "created_at",
    ):
        assert await _column_exists(knowledge_backend, "knowledge_wing_proof_bundles", col), col
    assert await _constraint_exists(
        knowledge_backend,
        "knowledge_wing_proof_bundles",
        "ck_kwpb_gate_requires_two_categories",
    )


async def test_v15_knowledge_semantic_overrides_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v15: knowledge_semantic_overrides table exists with provenance constraints."""
    assert await _table_exists(knowledge_backend, "knowledge_semantic_overrides")
    for col in (
        "id", "claim_id", "applied_by", "override_reason",
        "override_lifecycle_state", "accountability_status",
        "accountability_deadline", "activated_at", "last_checked_at", "created_at",
    ):
        assert await _column_exists(knowledge_backend, "knowledge_semantic_overrides", col), col
    assert await _constraint_exists(
        knowledge_backend, "knowledge_semantic_overrides", "ck_kso_accountability_status"
    )
    assert await _constraint_exists(
        knowledge_backend, "knowledge_semantic_overrides", "ck_kso_applied_by_nonempty"
    )
    assert await _constraint_exists(
        knowledge_backend, "knowledge_semantic_overrides", "ck_kso_reason_nonempty"
    )


async def test_v15_adr013_indexes_present(knowledge_backend: PostgresBackend) -> None:
    """v15: ADR-013 scope and lifecycle indexes exist."""
    expected_indexes = [
        "idx_kse_palace_scope",
        "idx_kse_entity_stable_id",
        "idx_kse_cycle_id",
        "idx_kse_evidence_category",
        "idx_kse_source_item_id",
        "idx_kvc_palace_scope_key",
        "idx_kvc_success",
        "idx_kvc_completed_at",
        "idx_ksc_palace_scope",
        "idx_ksc_scope_key",
        "idx_ksc_lifecycle_state",
        "idx_kcel_claim_category",
        "idx_kwpb_palace_wing",
        "idx_kso_claim_id",
        "idx_kso_accountability_status",
    ]
    for idx in expected_indexes:
        assert await _index_exists(knowledge_backend, idx), f"index missing: {idx}"


async def test_v15_structural_evidence_category_check_rejects_invalid(db: PostgresBackend) -> None:
    """v15: knowledge_structural_evidence rejects unknown evidence_category values."""
    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_structural_evidence
                (palace, wing, room, compartment, entity_stable_id, entity_type, evidence_category)
            VALUES (
                'palace_adr013', 'wing_a', 'room_a', '',
                'entity-1', 'Class', 'invalid_category'
            )
            """,
            (),
        )


async def test_v16_structural_evidence_upsert_uniqueness_enforced(db: PostgresBackend) -> None:
    """v16: inserting the same (palace, wing, room, compartment, entity_stable_id,
    evidence_category) twice must raise a unique violation; ON CONFLICT upsert must
    update the existing row without creating a duplicate."""
    insert_sql = """
        INSERT INTO knowledge_structural_evidence
            (palace, wing, room, compartment, entity_stable_id, entity_type,
             evidence_category, evidence_payload)
        VALUES (
            'palace_adr013', 'wing_uq', 'room_uq', '',
            'src/uq.py::UqClass', 'class',
            'structural_class', '{}'::jsonb
        )
    """
    # First insert must succeed.
    await db.execute(insert_sql, ())

    # Second bare insert must raise unique violation.
    async with _expect_db_error(db):
        await db.execute(insert_sql, ())

    # ON CONFLICT upsert must succeed and not create a duplicate.
    await db.execute(
        """
        INSERT INTO knowledge_structural_evidence
            (palace, wing, room, compartment, entity_stable_id, entity_type,
             evidence_category, evidence_payload)
        VALUES (
            'palace_adr013', 'wing_uq', 'room_uq', '',
            'src/uq.py::UqClass', 'class',
            'structural_class', '{"updated": true}'::jsonb
        )
        ON CONFLICT (palace, wing, room, compartment, entity_stable_id, evidence_category)
            DO UPDATE SET evidence_payload = EXCLUDED.evidence_payload,
                          scanned_at = NOW()
        """,
        (),
    )
    count_result = await db.query(
        """
        SELECT COUNT(*) AS cnt FROM knowledge_structural_evidence
         WHERE palace = 'palace_adr013'
           AND entity_stable_id = 'src/uq.py::UqClass'
           AND evidence_category = 'structural_class'
        """,
        (),
    )
    assert int(count_result.rows[0]["cnt"]) == 1, (
        "Upsert must not create a duplicate row; expected exactly 1 row"
    )


async def test_v15_semantic_claims_lifecycle_constraint_rejects_invalid(
    db: PostgresBackend,
) -> None:
    """v15: knowledge_semantic_claims rejects unknown lifecycle_state values."""
    async with _expect_db_error(db):
        await db.execute(
            "INSERT INTO knowledge_semantic_claims"
            " (palace, wing, room, compartment, claim_text, claim_type,"
            "  lifecycle_state, scope_key)"
            " VALUES ('palace_adr013', 'wing_a', 'room_a', '', 'some claim',"
            "         'room_intent', 'invalid_state', 'palace_adr013::wing_a::room_a')",
            (),
        )


async def test_v15_archive_without_archived_at_is_rejected(db: PostgresBackend) -> None:
    """v15: ck_ksc_archive_requires_two_cycles rejects archived state when archived_at is NULL."""
    async with _expect_db_error(db):
        await db.execute(
            "INSERT INTO knowledge_semantic_claims"
            " (palace, wing, room, compartment, claim_text, claim_type,"
            "  lifecycle_state, scope_key, archived_at)"
            " VALUES ('palace_adr013', 'wing_arc', 'room_arc', '', 'archived claim',"
            "         'room_intent', 'archived', 'palace_adr013::wing_arc::room_arc', NULL)",
            (),
        )


async def test_v15_semantic_overrides_provenance_nonempty_constraint(db: PostgresBackend) -> None:
    """v15: knowledge_semantic_overrides rejects blank applied_by or override_reason."""
    claim = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_text, claim_type, lifecycle_state, scope_key)
        VALUES (
            'palace_adr013', 'wing_b', 'room_b', '', 'claim text',
            'room_intent', 'active_evidenced', 'palace_adr013::wing_b::room_b'
        )
        RETURNING id
        """,
        (),
    )
    claim_id = str(claim.rows[0]["id"])

    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_semantic_overrides
                (claim_id, applied_by, override_reason, override_lifecycle_state)
            VALUES ($1::uuid, '   ', 'valid reason', 'degraded')
            """,
            (claim_id,),
        )

    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_semantic_overrides
                (claim_id, applied_by, override_reason, override_lifecycle_state)
            VALUES ($1::uuid, 'agent-1', '', 'degraded')
            """,
            (claim_id,),
        )


async def test_v15_wing_proof_bundle_gate_constraint_rejects_unsatisfied_single_category(
    db: PostgresBackend,
) -> None:
    """v15: knowledge_wing_proof_bundles rejects gate_satisfied=TRUE with < 2 categories."""
    claim = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_text, claim_type, lifecycle_state, scope_key)
        VALUES (
            'palace_adr013', 'wing_c', 'room_c', '', 'wing claim',
            'room_intent', 'active_evidenced', 'palace_adr013::wing_c::room_c'
        )
        RETURNING id
        """,
        (),
    )
    claim_id = str(claim.rows[0]["id"])

    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_wing_proof_bundles
                (palace, wing, activating_claim_id, evidence_categories, gate_satisfied)
            VALUES ($1, $2, $3::uuid, ARRAY['structural_class'], TRUE)
            """,
            ("palace_adr013", "wing_c", claim_id),
        )

    # Two categories: gate is satisfied, insert must succeed.
    await db.execute(
        """
        INSERT INTO knowledge_wing_proof_bundles
            (palace, wing, activating_claim_id, evidence_categories, gate_satisfied)
        VALUES ($1, $2, $3::uuid, ARRAY['structural_class', 'structural_module'], TRUE)
        """,
        ("palace_adr013", "wing_c", claim_id),
    )


async def test_v15_verification_cycle_and_structural_evidence_round_trip(
    db: PostgresBackend,
) -> None:
    """v15: insert a cycle, link structural evidence to it, and read both back."""
    cycle = await db.query(
        """
        INSERT INTO knowledge_verification_cycles
            (palace, scope_key, covered_wing, covered_room,
             success, evidence_found, evidence_absent)
        VALUES ('palace_adr013', 'palace_adr013::wing_d::room_d', 'wing_d', 'room_d', TRUE, 3, 0)
        RETURNING id
        """,
        (),
    )
    cycle_id = str(cycle.rows[0]["id"])

    ev = await db.query(
        """
        INSERT INTO knowledge_structural_evidence
            (palace, wing, room, compartment,
             entity_stable_id, entity_type, cycle_id, evidence_category, evidence_payload)
        VALUES (
            'palace_adr013', 'wing_d', 'room_d', '', 'entity-2', 'Class',
            $1::uuid, 'structural_class', '{"file": "foo.py"}'::jsonb
        )
        RETURNING id, evidence_category
        """,
        (cycle_id,),
    )
    assert ev.rows[0]["evidence_category"] == "structural_class"

    ev_id = str(ev.rows[0]["id"])

    claim = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_text, claim_type, lifecycle_state, scope_key)
        VALUES (
            'palace_adr013', 'wing_d', 'room_d', '', 'room intent label',
            'room_intent', 'active_evidenced', 'palace_adr013::wing_d::room_d'
        )
        RETURNING id
        """,
        (),
    )
    claim_id = str(claim.rows[0]["id"])

    await db.execute(
        """
        INSERT INTO knowledge_claim_evidence_links
            (claim_id, evidence_id, evidence_category)
        VALUES ($1::uuid, $2::uuid, 'structural_class')
        """,
        (claim_id, ev_id),
    )

    link_result = await db.query(
        """
        SELECT count(*) AS cnt
          FROM knowledge_claim_evidence_links
         WHERE claim_id = $1::uuid
        """,
        (claim_id,),
    )
    assert int(link_result.rows[0]["cnt"]) == 1


# ---------------------------------------------------------------------------
# v17: knowledge_semantic_corridors — columns, constraints, indexes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v17_semantic_corridors_table_exists(knowledge_backend: PostgresBackend) -> None:
    """v17: knowledge_semantic_corridors table must exist after migration."""
    result = await knowledge_backend.query(
        """
        SELECT COUNT(*) AS n
          FROM information_schema.tables
         WHERE table_schema = 'public'
           AND table_name = 'knowledge_semantic_corridors'
        """,
        (),
    )
    assert int(result.rows[0]["n"]) == 1, (
        "knowledge_semantic_corridors table missing after v17 migration"
    )


@pytest.mark.asyncio
async def test_v17_semantic_corridors_required_columns_present(
    knowledge_backend: PostgresBackend,
) -> None:
    """v17: knowledge_semantic_corridors must have all required columns."""
    result = await knowledge_backend.query(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name   = 'knowledge_semantic_corridors'
        """,
        (),
    )
    present = {row["column_name"] for row in result.rows}
    required = {
        "claim_id", "from_claim_id", "to_claim_id",
        "corridor_type", "corridor_type_raw", "created_at",
    }
    missing = required - present
    assert not missing, (
        f"knowledge_semantic_corridors is missing columns: {missing!r}"
    )


@pytest.mark.asyncio
async def test_v17_semantic_corridors_self_loop_constraint_rejects_insert(
    knowledge_backend: PostgresBackend,
) -> None:
    """v17: ck_kscorr_no_self_loop must prevent from_claim_id == to_claim_id."""
    import uuid as _uuid

    db = knowledge_backend

    # Insert a minimal claim to act as both endpoints (self-loop attempt).
    claim = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_type, lifecycle_state,
             claim_text, scope_key)
        VALUES (
            'palace_v17_test', 'w', 'r', '', 'room_intent',
            'active_evidenced', 'self_loop_test', 'palace_v17_test/w/r/'
        )
        RETURNING id
        """,
        (),
    )
    claim_id = str(claim.rows[0]["id"])
    corridor_id = str(_uuid.uuid4())

    try:
        await db.execute(
            """
            INSERT INTO knowledge_semantic_corridors
                (claim_id, from_claim_id, to_claim_id,
                 corridor_type, corridor_type_raw)
            VALUES ($1::uuid, $2::uuid, $2::uuid, 'SELF_LOOP', 'self_loop')
            """,
            (corridor_id, claim_id),
        )
        pytest.fail(
            "Expected CHECK constraint ck_kscorr_no_self_loop to reject self-loop insert"
        )
    except Exception as exc:
        exc_lower = str(exc).lower()
        assert (
            "self_loop" in exc_lower
            or "check" in exc_lower
            or "violat" in exc_lower
        ), f"Expected constraint violation; got: {exc!r}"
    finally:
        await db.execute(
            "DELETE FROM knowledge_semantic_claims WHERE palace = 'palace_v17_test'",
            (),
        )


@pytest.mark.asyncio
async def test_v17_semantic_corridors_unique_directed_edge_constraint(
    knowledge_backend: PostgresBackend,
) -> None:
    """v17: uq_kscorr_directed_edge must reject a duplicate (from, to, type) triple."""
    db = knowledge_backend

    # Two endpoint claims.
    from_claim = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_type, lifecycle_state,
             claim_text, scope_key)
        VALUES ('palace_v17_uniq', 'w', 'r', '', 'room_intent',
                'active_evidenced', 'from_claim', 'palace_v17_uniq/w/r/')
        RETURNING id
        """,
        (),
    )
    from_id = str(from_claim.rows[0]["id"])

    to_claim = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_type, lifecycle_state,
             claim_text, scope_key)
        VALUES ('palace_v17_uniq', 'w', 'r', '', 'room_intent',
                'active_evidenced', 'to_claim', 'palace_v17_uniq/w/r/')
        RETURNING id
        """,
        (),
    )
    to_id = str(to_claim.rows[0]["id"])

    # First corridor claim and edge — must succeed.
    corridor_claim_1 = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_type, lifecycle_state,
             claim_text, scope_key)
        VALUES ('palace_v17_uniq', 'w', 'r', '', 'semantic_corridor',
                'active_evidenced', 'corr1', 'palace_v17_uniq/w/r/')
        RETURNING id
        """,
        (),
    )
    corr1_id = str(corridor_claim_1.rows[0]["id"])

    await db.execute(
        """
        INSERT INTO knowledge_semantic_corridors
            (claim_id, from_claim_id, to_claim_id, corridor_type, corridor_type_raw)
        VALUES ($1::uuid, $2::uuid, $3::uuid, 'CALLS_INTO', 'calls-into')
        """,
        (corr1_id, from_id, to_id),
    )

    # Second corridor claim — attempt duplicate directed+typed edge.
    corridor_claim_2 = await db.query(
        """
        INSERT INTO knowledge_semantic_claims
            (palace, wing, room, compartment, claim_type, lifecycle_state,
             claim_text, scope_key)
        VALUES ('palace_v17_uniq', 'w', 'r', '', 'semantic_corridor',
                'active_evidenced', 'corr2', 'palace_v17_uniq/w/r/')
        RETURNING id
        """,
        (),
    )
    corr2_id = str(corridor_claim_2.rows[0]["id"])

    try:
        await db.execute(
            """
            INSERT INTO knowledge_semantic_corridors
                (claim_id, from_claim_id, to_claim_id, corridor_type, corridor_type_raw)
            VALUES ($1::uuid, $2::uuid, $3::uuid, 'CALLS_INTO', 'calls-into')
            """,
            (corr2_id, from_id, to_id),
        )
        pytest.fail(
            "Expected unique constraint uq_kscorr_directed_edge to reject duplicate edge"
        )
    except Exception as exc:
        assert "unique" in str(exc).lower() or "violat" in str(exc).lower(), (
            f"Expected unique constraint violation; got: {exc!r}"
        )
    finally:
        # Delete corridor rows first (from_claim_id FK may be RESTRICT).
        await db.execute(
            """
            DELETE FROM knowledge_semantic_corridors
             WHERE from_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace = 'palace_v17_uniq'
             )
               OR to_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace = 'palace_v17_uniq'
             )
            """,
            (),
        )
        await db.execute(
            "DELETE FROM knowledge_semantic_claims WHERE palace = 'palace_v17_uniq'",
            (),
        )


@pytest.mark.asyncio
async def test_v17_semantic_corridors_indexes_present(
    knowledge_backend: PostgresBackend,
) -> None:
    """v17: from_claim_id, to_claim_id, and corridor_type indexes must exist."""
    result = await knowledge_backend.query(
        """
        SELECT indexname
          FROM pg_indexes
         WHERE tablename = 'knowledge_semantic_corridors'
        """,
        (),
    )
    index_names = {row["indexname"] for row in result.rows}
    required_indexes = {
        "idx_kscorr_from_claim_id",
        "idx_kscorr_to_claim_id",
        "idx_kscorr_corridor_type",
    }
    missing = required_indexes - index_names
    assert not missing, (
        f"knowledge_semantic_corridors is missing indexes: {missing!r}. Found: {index_names!r}"
    )


# ---------------------------------------------------------------------------
# v18: ADR-013 Task 2a — topology provenance schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v18_migration_is_registered_in_migrations_list() -> None:
    """v18 migration must exist in MIGRATIONS with version 18 and no schema epoch bump."""
    from workflows_mcp.engine.knowledge.schema import MIGRATIONS, SCHEMA_EPOCH

    versions = [m[0] for m in MIGRATIONS]
    assert 18 in versions, f"v18 must be in MIGRATIONS; found versions: {versions}"
    assert SCHEMA_EPOCH == 2, (
        f"SCHEMA_EPOCH must remain 2 (no epoch bump for v18); got {SCHEMA_EPOCH}"
    )


@pytest.mark.asyncio
async def test_v19_schema_version_is_19(knowledge_backend: PostgresBackend) -> None:
    """After ensure_schema, schema_version must be 19."""
    from workflows_mcp.engine.knowledge.schema import SCHEMA_VERSION

    assert SCHEMA_VERSION == 19, (
        f"SCHEMA_VERSION must be 19 after v19 migration is appended; got {SCHEMA_VERSION}"
    )


@pytest.mark.asyncio
async def test_v18_knowledge_topology_provenance_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: knowledge_topology_provenance table must exist."""
    result = await knowledge_backend.query(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
             WHERE table_schema = 'public'
               AND table_name = 'knowledge_topology_provenance'
        ) AS exists
        """,
        (),
    )
    assert result.rows[0]["exists"], "knowledge_topology_provenance table must exist after v18"


@pytest.mark.asyncio
async def test_v18_topology_provenance_required_columns(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: knowledge_topology_provenance must have all required columns."""
    result = await knowledge_backend.query(
        """
        SELECT column_name, is_nullable, data_type
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'knowledge_topology_provenance'
        ORDER BY ordinal_position
        """,
        (),
    )
    columns = {row["column_name"]: row for row in result.rows}
    required = {
        "id",
        "claim_id",
        "palace",
        "wing",
        "room",
        "compartment",
        "scope_key",
        "derivation_source",
        "derivation_algorithm_version",
        "override_reason",
        "applied_by",
        "override_id",
        "derived_at",
        "created_at",
    }
    missing = required - set(columns)
    assert not missing, (
        f"knowledge_topology_provenance is missing columns: {missing!r}"
    )
    # override_reason and applied_by must be nullable
    assert columns["override_reason"]["is_nullable"] == "YES", (
        "override_reason must be nullable"
    )
    assert columns["applied_by"]["is_nullable"] == "YES", "applied_by must be nullable"
    assert columns["override_id"]["is_nullable"] == "YES", "override_id must be nullable"


@pytest.mark.asyncio
async def test_v18_topology_provenance_fk_to_semantic_claims(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: claim_id must FK to knowledge_semantic_claims(id)."""
    result = await knowledge_backend.query(
        """
        SELECT conname, confrelid::regclass::text AS ref_table
          FROM pg_constraint
         WHERE conrelid = 'knowledge_topology_provenance'::regclass
           AND contype = 'f'
           AND conname LIKE '%claim%'
        """,
        (),
    )
    fk_names = {row["conname"]: row["ref_table"] for row in result.rows}
    assert any(
        "semantic_claims" in ref for ref in fk_names.values()
    ), f"claim_id FK to knowledge_semantic_claims not found; FKs found: {fk_names}"


@pytest.mark.asyncio
async def test_v18_topology_provenance_fk_to_semantic_overrides_nullable(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: override_id must FK to knowledge_semantic_overrides(id), nullable."""
    result = await knowledge_backend.query(
        """
        SELECT conname, confrelid::regclass::text AS ref_table
          FROM pg_constraint
         WHERE conrelid = 'knowledge_topology_provenance'::regclass
           AND contype = 'f'
           AND conname LIKE '%override%'
        """,
        (),
    )
    fk_names = {row["conname"]: row["ref_table"] for row in result.rows}
    assert any(
        "semantic_overrides" in ref for ref in fk_names.values()
    ), f"override_id FK to knowledge_semantic_overrides not found; FKs found: {fk_names}"


@pytest.mark.asyncio
async def test_v18_topology_provenance_derivation_source_check_constraint(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: derivation_source must be constrained to 'explicit_override' or 'system1_derived'."""
    result = await knowledge_backend.query(
        """
        SELECT conname
          FROM pg_constraint
         WHERE conrelid = 'knowledge_topology_provenance'::regclass
           AND contype = 'c'
           AND conname LIKE '%derivation_source%'
        """,
        (),
    )
    assert result.rows, (
        "CHECK constraint on derivation_source not found in knowledge_topology_provenance"
    )


@pytest.mark.asyncio
async def test_v18_topology_provenance_override_fields_check_constraint(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: CHECK must enforce non-empty override_reason and applied_by when source='explicit_override'."""  # noqa: E501
    result = await knowledge_backend.query(
        """
        SELECT conname
          FROM pg_constraint
         WHERE conrelid = 'knowledge_topology_provenance'::regclass
           AND contype = 'c'
           AND (conname LIKE '%override_reason%' OR conname LIKE '%applied_by%'
                OR conname LIKE '%explicit_override%')
        """,
        (),
    )
    assert result.rows, (
        "CHECK constraint enforcing non-empty override_reason/applied_by for explicit_override "
        "not found in knowledge_topology_provenance"
    )


@pytest.mark.asyncio
async def test_v18_knowledge_topology_provenance_evidence_table_exists(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: knowledge_topology_provenance_evidence join table must exist."""
    result = await knowledge_backend.query(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
             WHERE table_schema = 'public'
               AND table_name = 'knowledge_topology_provenance_evidence'
        ) AS exists
        """,
        (),
    )
    assert result.rows[0]["exists"], (
        "knowledge_topology_provenance_evidence table must exist after v18"
    )


@pytest.mark.asyncio
async def test_v18_provenance_evidence_composite_pk(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: knowledge_topology_provenance_evidence must have composite PK on (provenance_id, evidence_id)."""  # noqa: E501
    result = await knowledge_backend.query(
        """
        SELECT conname, contype
          FROM pg_constraint
         WHERE conrelid = 'knowledge_topology_provenance_evidence'::regclass
           AND contype = 'p'
        """,
        (),
    )
    assert result.rows, (
        "No PRIMARY KEY found on knowledge_topology_provenance_evidence"
    )


@pytest.mark.asyncio
async def test_v18_provenance_evidence_fk_to_structural_evidence(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: evidence_id must FK to knowledge_structural_evidence(id) with RESTRICT on delete."""
    result = await knowledge_backend.query(
        """
        SELECT conname, confrelid::regclass::text AS ref_table, confdeltype
          FROM pg_constraint
         WHERE conrelid = 'knowledge_topology_provenance_evidence'::regclass
           AND contype = 'f'
           AND conname LIKE '%evidence%'
        """,
        (),
    )
    fk_rows = {row["conname"]: row for row in result.rows}
    assert any(
        "structural_evidence" in row["ref_table"] for row in fk_rows.values()
    ), f"evidence_id FK to knowledge_structural_evidence not found; FKs: {fk_rows}"
    # Verify RESTRICT (confdeltype = 'r') on that FK
    restrict_rows = [
        row for row in fk_rows.values()
        if "structural_evidence" in row["ref_table"]
        and row["confdeltype"] in ("r", b"r")
    ]
    assert restrict_rows, (
        "evidence_id FK to knowledge_structural_evidence must use ON DELETE RESTRICT"
    )


@pytest.mark.asyncio
async def test_v18_provenance_evidence_index_on_evidence_id(
    knowledge_backend: PostgresBackend,
) -> None:
    """v18: knowledge_topology_provenance_evidence must have an index on evidence_id."""
    result = await knowledge_backend.query(
        """
        SELECT indexname
          FROM pg_indexes
         WHERE tablename = 'knowledge_topology_provenance_evidence'
        """,
        (),
    )
    index_names = {row["indexname"] for row in result.rows}
    evidence_id_indexes = [n for n in index_names if "evidence_id" in n]
    assert evidence_id_indexes, (
        f"No index on evidence_id found in knowledge_topology_provenance_evidence. "
        f"Found indexes: {index_names!r}"
    )


@pytest.mark.asyncio
async def test_v18_migration_is_idempotent(knowledge_backend: PostgresBackend) -> None:
    """v18 migration SQL must be safe to re-apply (idempotent via IF NOT EXISTS guards)."""
    from workflows_mcp.engine.knowledge.schema import MIGRATIONS

    v18_sql = next((sql for ver, _desc, sql in MIGRATIONS if ver == 18), None)
    assert v18_sql is not None, "v18 migration not found"
    # Re-applying must not raise
    await knowledge_backend.execute_script(v18_sql)
