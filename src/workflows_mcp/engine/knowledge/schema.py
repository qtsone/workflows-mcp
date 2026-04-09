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

Schema ownership
----------------
This file defines and owns the **base schema** — only columns that this
engine reads or writes.  Any application embedding this engine may extend
the tables with additional columns by running its own ``ADD COLUMN IF NOT
EXISTS`` statements after ``ensure_schema()`` completes.  Neither layer
should touch the other's columns.
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
    category_ids UUID[] DEFAULT '{}',
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_PROPOSITIONS = """
CREATE TABLE IF NOT EXISTS knowledge_propositions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID REFERENCES knowledge_items(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    search_vector tsvector,
    authority VARCHAR(50) NOT NULL DEFAULT 'EXTRACTED',
    lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    confidence FLOAT DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    embedding_model VARCHAR(100),
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
        DO $$
        BEGIN
            -- HNSW indexes require the column to have fixed dimensions.
            -- Newer pgvector (≥0.6) rejects dimensionless vector columns.
            -- Wrap in exception so a dimensionless column degrades to sequential
            -- scan rather than blocking server startup entirely.
            CREATE INDEX IF NOT EXISTS idx_kp_embedding
                ON knowledge_propositions
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        EXCEPTION WHEN others THEN
            RAISE WARNING 'HNSW index on embedding skipped (column may lack fixed dimensions): %', SQLERRM;
        END $$;
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
    (
        9,
        "Remove unused columns from OSS schema",
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_sources' AND column_name = 'allowed_team_ids'
            ) THEN
                ALTER TABLE knowledge_sources DROP COLUMN allowed_team_ids;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_sources' AND column_name = 'config'
            ) THEN
                ALTER TABLE knowledge_sources DROP COLUMN config;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_items' AND column_name = 'content_hash'
            ) THEN
                ALTER TABLE knowledge_items DROP COLUMN content_hash;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'source_section'
            ) THEN
                ALTER TABLE knowledge_propositions DROP COLUMN source_section;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'relevance_score'
            ) THEN
                ALTER TABLE knowledge_propositions DROP COLUMN relevance_score;
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_entities' AND column_name = 'properties'
            ) THEN
                ALTER TABLE knowledge_entities DROP COLUMN properties;
            END IF;
        END $$;
        """,
    ),
    (
        10,
        "Introduce knowledge_categories table; re-point knowledge_proposition_categories FK",
        """
        -- 1. Create dedicated category registry
        CREATE TABLE IF NOT EXISTS knowledge_categories (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name       VARCHAR(500) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_knowledge_categories_name UNIQUE (name)
        );

        -- 2. Migrate existing category entities (preserve UUIDs for FK integrity)
        INSERT INTO knowledge_categories (id, name, created_at)
        SELECT id, name, created_at
        FROM knowledge_entities
        WHERE entity_type = 'category'
        ON CONFLICT (id) DO NOTHING;

        DO $$
        DECLARE
            fk_name TEXT;
        BEGIN
            -- 3. Drop old FK: knowledge_proposition_categories.category_id → knowledge_entities
            SELECT conname INTO fk_name
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
            WHERE t.relname = 'knowledge_proposition_categories'
              AND c.contype = 'f'
              AND a.attname = 'category_id';

            IF fk_name IS NOT NULL THEN
                EXECUTE 'ALTER TABLE knowledge_proposition_categories DROP CONSTRAINT ' || fk_name;
            END IF;

            -- 4. Add new FK → knowledge_categories
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'kpc_category_id_fkey'
            ) THEN
                ALTER TABLE knowledge_proposition_categories
                    ADD CONSTRAINT kpc_category_id_fkey
                    FOREIGN KEY (category_id) REFERENCES knowledge_categories(id) ON DELETE CASCADE;
            END IF;
        END $$;

        -- 5. Clean up migrated category rows from knowledge_entities
        DELETE FROM knowledge_entities WHERE entity_type = 'category';
        """,
    ),
    (
        11,
        "Add temporal validity window columns to knowledge_propositions",
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'valid_from'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN valid_from TIMESTAMPTZ NULL;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'valid_to'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN valid_to TIMESTAMPTZ NULL;
            END IF;
        END $$;
        """,
    ),
    (
        12,
        "Add entity/relation backbone: knowledge_relations and knowledge_proposition_entities",
        """
        -- Relation graph: directed edges between knowledge entities
        CREATE TABLE IF NOT EXISTS knowledge_relations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_entity_id UUID NOT NULL
                REFERENCES knowledge_entities(id) ON DELETE CASCADE,
            target_entity_id UUID NOT NULL
                REFERENCES knowledge_entities(id) ON DELETE CASCADE,
            relation_type VARCHAR(100) NOT NULL,
            confidence FLOAT DEFAULT 1.0,
            provenance_proposition_id UUID
                REFERENCES knowledge_propositions(id) ON DELETE SET NULL,
            valid_from TIMESTAMPTZ NULL,
            valid_to TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Junction: which entities are mentioned/subject/object in a proposition
        CREATE TABLE IF NOT EXISTS knowledge_proposition_entities (
            proposition_id UUID NOT NULL
                REFERENCES knowledge_propositions(id) ON DELETE CASCADE,
            entity_id UUID NOT NULL
                REFERENCES knowledge_entities(id) ON DELETE CASCADE,
            role VARCHAR(100) NOT NULL DEFAULT 'mentioned',
            confidence FLOAT DEFAULT 1.0,
            PRIMARY KEY (proposition_id, entity_id)
        );

        -- Indexes for graph traversal performance
        CREATE INDEX IF NOT EXISTS idx_kr_relation_type
            ON knowledge_relations(relation_type);
        CREATE INDEX IF NOT EXISTS idx_kr_source_target
            ON knowledge_relations(source_entity_id, target_entity_id);
        CREATE INDEX IF NOT EXISTS idx_kr_temporal
            ON knowledge_relations(valid_from, valid_to);
        CREATE INDEX IF NOT EXISTS idx_kpe_entity_id
            ON knowledge_proposition_entities(entity_id);
        CREATE INDEX IF NOT EXISTS idx_kpe_proposition_id
            ON knowledge_proposition_entities(proposition_id);
        """,
    ),
    (
        13,
        "Align junction table and relation column names with platform conventions",
        """
        -- Rename junction table: knowledge_proposition_entities → knowledge_entity_propositions
        -- Platform convention: entity_id, proposition_id (entity-first naming, matches knowledge_admin.py)
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'knowledge_proposition_entities'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'knowledge_entity_propositions'
            ) THEN
                ALTER TABLE knowledge_proposition_entities
                    RENAME TO knowledge_entity_propositions;
            END IF;
        END $$;

        -- Rename indexes to match new table name
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 'idx_kpe_entity_id'
            ) THEN
                ALTER INDEX idx_kpe_entity_id RENAME TO idx_kep_entity_id;
            END IF;

            IF EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 'idx_kpe_proposition_id'
            ) THEN
                ALTER INDEX idx_kpe_proposition_id RENAME TO idx_kep_proposition_id;
            END IF;
        END $$;

        -- Rename knowledge_relations.provenance_proposition_id → evidence_proposition_id
        -- Platform convention: evidence_proposition_id (matches knowledge_admin.py usage)
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_relations'
                  AND column_name = 'provenance_proposition_id'
            ) THEN
                ALTER TABLE knowledge_relations
                    RENAME COLUMN provenance_proposition_id TO evidence_proposition_id;
            END IF;
        END $$;
        """,
    ),
    (
        14,
        "Add room/topology columns to knowledge_propositions for retrieval routing",
        """
        DO $$
        BEGIN
            -- namespace: major domain or tenant compartment (e.g. 'engineering', 'finance')
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'namespace'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN namespace VARCHAR(200) NULL;
            END IF;

            -- room: task or topic compartment within the namespace (e.g. 'api-design', 'auth')
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'room'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN room VARCHAR(200) NULL;
            END IF;

            -- corridor: optional process lane within a room (e.g. 'sprint-42', 'incident-007')
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_propositions' AND column_name = 'corridor'
            ) THEN
                ALTER TABLE knowledge_propositions ADD COLUMN corridor VARCHAR(200) NULL;
            END IF;
        END $$;

        -- Indexes for room-scoped retrieval routing
        CREATE INDEX IF NOT EXISTS idx_kp_namespace ON knowledge_propositions(namespace)
            WHERE namespace IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_kp_room ON knowledge_propositions(room)
            WHERE room IS NOT NULL;
         CREATE INDEX IF NOT EXISTS idx_kp_namespace_room ON knowledge_propositions(namespace, room)
             WHERE namespace IS NOT NULL AND room IS NOT NULL;
         """,
    ),
    (
        15,
        "Enforce 1536-dim embeddings: vector(1536) column type",
        """
        -- Convert embedding column from dimensionless vector to vector(1536).
        -- This is the authoritative enforcement: Postgres rejects any other dimension
        -- at the type level, with no application-layer workaround needed.
        -- Only runs if the column is still dimensionless (idempotent on vector(1536) DBs).
        -- The sync trigger and HNSW index must be dropped first due to column dependencies;
        -- the HNSW index is recreated below (the sync trigger is dropped permanently in v16).
        DO $$
        BEGIN
            IF (
                SELECT pg_catalog.format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = 'knowledge_propositions'::regclass
                  AND attname = 'embedding'
            ) = 'vector' THEN
                DROP TRIGGER IF EXISTS trg_kp_sync_embedding_dimensions ON knowledge_propositions;
                DROP INDEX IF EXISTS idx_kp_embedding;
                ALTER TABLE knowledge_propositions
                    ALTER COLUMN embedding TYPE vector(1536);
            END IF;
        END $$;

        -- Drop the now-redundant CHECK constraint (superseded by vector(1536)).
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'chk_kp_embedding_dimensions_1536'
                  AND conrelid = 'knowledge_propositions'::regclass
            ) THEN
                ALTER TABLE knowledge_propositions
                    DROP CONSTRAINT chk_kp_embedding_dimensions_1536;
            END IF;
        END $$;

        -- Recreate HNSW index on the now-typed vector(1536) column.
        DO $$
        BEGIN
            CREATE INDEX IF NOT EXISTS idx_kp_embedding
                ON knowledge_propositions
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        EXCEPTION WHEN others THEN
            RAISE WARNING 'HNSW index on embedding skipped: %', SQLERRM;
        END $$;
        """,
    ),
    (
        16,
        "Drop embedding_dimensions: redundant after vector(1536) enforcement",
        """
        -- Drop the sync trigger and its backing function.
        -- With vector(1536) enforced at the column type level, there is nothing to sync.
        DROP TRIGGER IF EXISTS trg_kp_sync_embedding_dimensions ON knowledge_propositions;
        DROP FUNCTION IF EXISTS sync_embedding_dimensions();

        -- Drop the composite index that used embedding_dimensions as a filter column.
        -- With a single valid dimension, that column has no selectivity.
        DROP INDEX IF EXISTS idx_kp_namespace_room_embedding_dim;

        -- Drop the column itself.
        ALTER TABLE knowledge_propositions DROP COLUMN IF EXISTS embedding_dimensions;
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
