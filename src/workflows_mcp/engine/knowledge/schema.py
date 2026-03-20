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
    (
        3,
        "Add audit trail columns and knowledge_proposition_audits table",
        """
        DO $$
        BEGIN
            -- Add created_by column to knowledge_propositions
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'created_by'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN created_by UUID;
            END IF;

            -- Add archived_by column to knowledge_propositions
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'archived_by'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN archived_by UUID;
            END IF;

            -- Add auth_method column to knowledge_propositions
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'auth_method'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN auth_method VARCHAR(50);
            END IF;

            -- Add archive_reason column to knowledge_propositions
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'archive_reason'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN archive_reason VARCHAR(100);
            END IF;
        END $$;

        -- Create knowledge_proposition_audits table
        CREATE TABLE IF NOT EXISTS knowledge_proposition_audits (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            proposition_id UUID NOT NULL REFERENCES knowledge_propositions(id) ON DELETE CASCADE,
            action VARCHAR(50) NOT NULL,
            performed_by UUID NOT NULL,
            auth_method VARCHAR(50),
            performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        );

        -- Create indexes for audit columns
        CREATE INDEX IF NOT EXISTS idx_kp_created_by ON knowledge_propositions(created_by);
        CREATE INDEX IF NOT EXISTS idx_kp_archived_by ON knowledge_propositions(archived_by);
        CREATE INDEX IF NOT EXISTS idx_knowledge_proposition_audits_proposition_id ON knowledge_proposition_audits(proposition_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_proposition_audits_performed_by ON knowledge_proposition_audits(performed_by);
        CREATE INDEX IF NOT EXISTS idx_knowledge_proposition_audits_performed_at ON knowledge_proposition_audits(performed_at);
        """,
    ),
    (
        4,
        "Add source columns to knowledge_propositions for query optimization",
        """
        DO $$
        BEGIN
            -- Add source_name column for direct source filtering
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'source_name'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN source_name VARCHAR(500);
            END IF;

            -- Add source_type column for efficient categorization
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'source_type'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN source_type VARCHAR(50) DEFAULT NULL;
            END IF;
        END $$;

        -- Create indexes for query optimization
        CREATE INDEX IF NOT EXISTS idx_kp_source_name ON knowledge_propositions(source_name);
        CREATE INDEX IF NOT EXISTS idx_kp_source_type ON knowledge_propositions(source_type);
        CREATE INDEX IF NOT EXISTS idx_kp_source_name_lifecycle ON knowledge_propositions(source_name, lifecycle_state);
        """,
    ),
    (
        5,
        "Add HNSW index on knowledge_propositions.embedding for ANN vector search",
        """
        CREATE INDEX IF NOT EXISTS idx_kp_embedding
            ON knowledge_propositions
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """,
    ),
    (
        6,
        "Add updated_at trigger to all knowledge tables that carry the column",
        """
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'trg_ks_updated_at'
            ) THEN
                CREATE TRIGGER trg_ks_updated_at
                    BEFORE UPDATE ON knowledge_sources
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'trg_ki_updated_at'
            ) THEN
                CREATE TRIGGER trg_ki_updated_at
                    BEFORE UPDATE ON knowledge_items
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgname = 'trg_kp_updated_at'
            ) THEN
                CREATE TRIGGER trg_kp_updated_at
                    BEFORE UPDATE ON knowledge_propositions
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
            END IF;
        END $$;
        """,
    ),
    (
        7,
        "Clarify knowledge_propositions.source_type semantics: drop misleading default, null-out stale values",
        """
        DO $$
        BEGIN
            -- Drop the misleading DEFAULT 'DOCUMENT' — source_type must be set explicitly by callers.
            -- NULL means "origin unknown" (rows inserted before this was tracked correctly).
            ALTER TABLE knowledge_propositions
                ALTER COLUMN source_type DROP DEFAULT;

            -- Null out the old heuristic values (DOCUMENT/TOOL were derived from source/path
            -- presence, not from actual ingestion origin — they are semantically meaningless).
            UPDATE knowledge_propositions SET source_type = NULL
                WHERE source_type IN ('DOCUMENT', 'TOOL');
        END $$;
        """,
    ),
    (
        8,
        "Add knowledge_proposition_categories junction table for per-proposition categories",
        """
        CREATE TABLE IF NOT EXISTS knowledge_proposition_categories (
            proposition_id UUID NOT NULL
                REFERENCES knowledge_propositions(id) ON DELETE CASCADE,
            category_id    UUID NOT NULL
                REFERENCES knowledge_entities(id) ON DELETE CASCADE,
            assigned_by    VARCHAR(20) NOT NULL DEFAULT 'EXPLICIT',
            assigned_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (proposition_id, category_id)
        );

        CREATE INDEX IF NOT EXISTS idx_kpc_category_id
            ON knowledge_proposition_categories(category_id);
        CREATE INDEX IF NOT EXISTS idx_kpc_proposition_id
            ON knowledge_proposition_categories(proposition_id);

        -- Backfill: inherit source-level categories onto existing document-derived propositions.
        -- Agent observations (item_id IS NULL) are untouched — they had no categories before
        -- and correctly get none unless explicitly set at store time.
        INSERT INTO knowledge_proposition_categories (proposition_id, category_id, assigned_by)
        SELECT kp.id, unnest(ks.category_ids), 'INHERITED'
        FROM knowledge_propositions kp
        JOIN knowledge_items ki ON kp.item_id = ki.id
        JOIN knowledge_sources ks ON ki.source_id = ks.id
        WHERE ks.category_ids IS NOT NULL
          AND ks.category_ids != '{}'
        ON CONFLICT (proposition_id, category_id) DO NOTHING;
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
