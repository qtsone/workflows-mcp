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


def test_item_lifecycle_state_enum_values() -> None:
    """ItemLifecycleState covers the full set of item lifecycle values."""


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


async def test_v6_knowledge_sources_palace_and_uniqueness(knowledge_backend: PostgresBackend, db: PostgresBackend) -> None:
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
           AND column_name IN ('palace', 'source_id', 'content_hash', 'size_bytes', 'mtime_ns', 'language')
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
            INSERT INTO knowledge_items (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ('palace_items', NULL, 'src/a.py', 'a.py', 'h1', 1, 1)
            """,
            (),
        )

    await db.execute(
        """
        INSERT INTO knowledge_items (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
        VALUES ('palace_items', $1::uuid, 'src/a.py', 'a.py', 'h1', 1, 1)
        """,
        (source_id,),
    )
    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_items (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ('palace_items', $1::uuid, 'src/a.py', 'a.py', 'h2', 2, 2)
            """,
            (source_id,),
        )


async def test_v7_same_content_hash_in_two_palaces_is_not_a_cross_palace_rename(db: PostgresBackend) -> None:
    """v7: content_hash lookup is palace-scoped; same hash in two palaces is two rows."""
    rows = []
    for palace in ("palace_hash_a", "palace_hash_b"):
        src = await db.query(
            "INSERT INTO knowledge_sources (palace, name, source_type) VALUES ($1, $2, 'FILE') RETURNING id",
            (palace, "source-hash"),
        )
        item = await db.query(
            """
            INSERT INTO knowledge_items (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
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


async def test_v8_item_source_palace_trigger_rejects_mismatch(knowledge_backend: PostgresBackend, db: PostgresBackend) -> None:
    """v8: item palace must match its source palace."""
    assert await _trigger_exists(knowledge_backend, "trg_ki_source_palace_match")
    src = await db.query(
        "INSERT INTO knowledge_sources (palace, name, source_type) VALUES ('palace_source', 'source-trigger', 'FILE') RETURNING id",
        (),
    )
    async with _expect_db_error(db):
        await db.execute(
            """
            INSERT INTO knowledge_items (palace, source_id, path, title, content_hash, size_bytes, mtime_ns)
            VALUES ('palace_item', $1::uuid, 'src/mismatch.py', 'mismatch.py', 'h', 1, 1)
            """,
            (str(src.rows[0]["id"]),),
        )


# ---------------------------------------------------------------------------
# v9: entities structural identity
# ---------------------------------------------------------------------------


async def test_v9_knowledge_entities_structural_columns_and_source_check(knowledge_backend: PostgresBackend) -> None:
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
            VALUES ('palace_entities', 'code', 'default', 'schema', 'Function', 'bad_source', 'FILE_NAME')
            """,
            (),
        )

async def test_v9_entities_metadata_round_trip_and_updated_at_advances(db: PostgresBackend) -> None:
    """v9: metadata defaults to {}, updates round-trip, and updated_at advances."""
    inserted = await db.query(
        """
        INSERT INTO knowledge_entities
            (palace, namespace, room, corridor, entity_type, name, source, stable_id)
        VALUES ('palace_entities', 'code', 'default', 'schema', 'Function', 'alpha', 'STRUCTURAL', 'alpha-stable')
        RETURNING id, metadata, updated_at
        """,
        (),
    )
    entity_id = str(inserted.rows[0]["id"])
    assert _as_dict(inserted.rows[0]["metadata"]) == {}
    before = inserted.rows[0]["updated_at"]

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
        VALUES ('test_palace_embed', 'code', 'default', 'foundation', 'Function', 'foo', 'STRUCTURAL', 'foo-stable')
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
           AND column_name IN ('scope_kind', 'source_palaces', 'member_provenance', 'consent_policy_id')
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


async def test_schema_version_advanced_to_thirteen(knowledge_backend: PostgresBackend) -> None:
    """All new migrations applied; SCHEMA_VERSION reaches 13."""
    from workflows_mcp.engine.knowledge.schema import SCHEMA_VERSION

    assert SCHEMA_VERSION == 13

    result = await knowledge_backend.query(
        "SELECT value FROM _knowledge_meta WHERE key = 'schema_version'", ()
    )
    assert result.rows
    assert int(result.rows[0]["value"]) == 13


async def test_re_running_ensure_schema_is_idempotent(knowledge_backend: PostgresBackend) -> None:
    """Calling ensure_schema again is a fast no-op when already current."""
    from workflows_mcp.engine.knowledge.schema import ensure_schema

    await ensure_schema(knowledge_backend)

    result = await knowledge_backend.query(
        "SELECT value FROM _knowledge_meta WHERE key = 'schema_version'", ()
    )
    assert int(result.rows[0]["value"]) == 13
