"""Idempotent DDL and versioned migrations for Knowledge tables.

Creates pgvector extension and knowledge tables on first use,
then applies any pending schema migrations automatically.

The migration system uses a ``_knowledge_meta`` table to track
the current schema version.  At server startup, ``ensure_schema()``
is called once to:

1. Create the meta table (if missing).
2. Create base tables (idempotent ``IF NOT EXISTS``).
3. Read the current schema version (default 0).
4. Apply any ``MIGRATIONS`` with version > current.
5. Update the stored version.

Adding a new migration
----------------------
Append a tuple to ``MIGRATIONS`` with ``(version, description, sql)``.
The SQL **must** be idempotent (use IF EXISTS / IF NOT EXISTS guards)
so it is safe on both fresh and existing databases.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base DDL — creates tables on first use (version 1)
# ---------------------------------------------------------------------------

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

_CREATE_KNOWLEDGE_SOURCES = """
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    source_type VARCHAR(50) NOT NULL DEFAULT 'DOCUMENT_UPLOAD',
    config JSONB DEFAULT '{}'::jsonb,
    category_ids UUID[] DEFAULT '{}',
    allowed_team_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_ITEMS = """
CREATE TABLE IF NOT EXISTS knowledge_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    path VARCHAR(1000),
    title VARCHAR(500),
    content_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_PROPOSITIONS = """
CREATE TABLE IF NOT EXISTS knowledge_propositions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID REFERENCES knowledge_items(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector,
    search_vector tsvector,
    authority VARCHAR(50) NOT NULL DEFAULT 'EXTRACTED',
    lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    relevance_score FLOAT DEFAULT 0.5,
    confidence FLOAT DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    embedding_model VARCHAR(100),
    embedding_dimensions INTEGER,
    source_section VARCHAR(500),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_ENTITIES = """
CREATE TABLE IF NOT EXISTS knowledge_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    name VARCHAR(500) NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_kp_lifecycle ON knowledge_propositions(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_kp_item_id ON knowledge_propositions(item_id);
CREATE INDEX IF NOT EXISTS idx_kp_search_vector ON knowledge_propositions USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_ks_category_ids ON knowledge_sources USING gin(category_ids);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ks_name ON knowledge_sources(name);
CREATE INDEX IF NOT EXISTS idx_ki_source_id ON knowledge_items(source_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ki_source_path ON knowledge_items(source_id, path);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_type_name ON knowledge_entities(entity_type, name);
"""

# ---------------------------------------------------------------------------
# Schema migrations — numbered, idempotent, applied automatically
# ---------------------------------------------------------------------------
# Each entry: (version, description, sql)
# SQL MUST be idempotent — safe on both fresh and existing databases.

MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "Remove org_id columns (single-user engine, no multi-tenancy)",
        """
        DO $$
        BEGIN
            -- Drop org_id indexes
            DROP INDEX IF EXISTS idx_knowledge_sources_org_id;
            DROP INDEX IF EXISTS idx_knowledge_items_org_id;
            DROP INDEX IF EXISTS idx_knowledge_propositions_org_id;
            DROP INDEX IF EXISTS idx_knowledge_entities_org_id;
            DROP INDEX IF EXISTS idx_ke_org_type_name;

            -- Rebuild unique index if it includes org_id (3+ columns → 2)
            IF EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = 'idx_ke_type_name'
                  AND indexdef LIKE '%org_id%'
            ) THEN
                DROP INDEX idx_ke_type_name;
                CREATE UNIQUE INDEX idx_ke_type_name
                    ON knowledge_entities(entity_type, name);
            END IF;

            -- Drop org_id columns
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_sources' AND column_name = 'org_id'
            ) THEN
                ALTER TABLE knowledge_sources DROP COLUMN org_id;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_items' AND column_name = 'org_id'
            ) THEN
                ALTER TABLE knowledge_items DROP COLUMN org_id;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'org_id'
            ) THEN
                ALTER TABLE knowledge_propositions DROP COLUMN org_id;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_entities' AND column_name = 'org_id'
            ) THEN
                ALTER TABLE knowledge_entities DROP COLUMN org_id;
            END IF;
        END $$;
        """,
    ),
    (
        2,
        "Rename metadata_ to metadata in knowledge_propositions",
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'metadata_'
            ) THEN
                ALTER TABLE knowledge_propositions RENAME COLUMN metadata_ TO metadata;
            END IF;
        END $$;
        """,
    ),
]

SCHEMA_VERSION = MIGRATIONS[-1][0] if MIGRATIONS else 0


async def ensure_schema(backend: Any) -> None:
    """Ensure knowledge tables exist and are up to date.

    This is the primary entry point for schema management.
    Safe to call on every operation — fast no-op when already current.

    Args:
        backend: A connected ``DatabaseBackendBase`` instance with
            ``execute_script()``, ``query()``, and ``execute()`` methods.
    """
    # 1. Create meta table for version tracking
    await backend.execute_script(
        "CREATE TABLE IF NOT EXISTS _knowledge_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    )

    # 2. Create base tables (idempotent)
    base_ddl = "\n".join(
        [
            _CREATE_EXTENSION,
            _CREATE_KNOWLEDGE_SOURCES,
            _CREATE_KNOWLEDGE_ITEMS,
            _CREATE_KNOWLEDGE_PROPOSITIONS,
            _CREATE_KNOWLEDGE_ENTITIES,
            _CREATE_INDEXES,
        ]
    )
    await backend.execute_script(base_ddl)

    # 3. Read current version
    result = await backend.query("SELECT value FROM _knowledge_meta WHERE key = 'schema_version'")
    current_version = int(result.rows[0]["value"]) if result.rows else 0

    if current_version >= SCHEMA_VERSION:
        return

    # 4. Apply pending migrations
    for version, description, sql in MIGRATIONS:
        if version <= current_version:
            continue
        logger.info("Applying knowledge schema migration v%d: %s", version, description)
        await backend.execute_script(sql)

    # 5. Store new version
    await backend.execute_script(
        f"INSERT INTO _knowledge_meta (key, value) VALUES ('schema_version', '{SCHEMA_VERSION}') "
        f"ON CONFLICT (key) DO UPDATE SET value = '{SCHEMA_VERSION}';"
    )
    logger.info("Knowledge schema is at version %d", SCHEMA_VERSION)
