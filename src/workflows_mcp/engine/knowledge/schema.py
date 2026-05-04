"""DDL and versioned migrations for Knowledge tables.

**Migration model (post clean-slate)**
The base DDL below defines the canonical fresh-install schema. Incremental
migrations in ``MIGRATIONS`` evolve compatible versions forward. Incompatible
schema epochs fail closed at startup with migration guidance.

The migration system uses a ``_knowledge_meta`` table to track schema epoch
and version. At server startup ``ensure_schema()`` is called once to:

1. Create the meta table (if missing).
2. Acquire a schema advisory lock.
3. Validate schema compatibility for this epoch.
4. Fail fast on incompatible schema epoch with migration guidance.
5. Apply pending migrations.
6. Update stored schema epoch/version.

Adding a new migration
----------------------
Append a tuple to ``MIGRATIONS`` with ``(version, description, sql)``.
The SQL **must** be idempotent (use IF EXISTS / IF NOT EXISTS guards)
so it is safe on both fresh databases and any future upgrades.

Schema ownership
----------------
This file owns the **complete base schema** for every table the engine
reads or writes.  Do not extend tables externally without a migration.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_LOCK_SQL = "SELECT pg_advisory_lock(hashtext('workflows_mcp_knowledge_schema_lock'));"
_SCHEMA_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtext('workflows_mcp_knowledge_schema_lock'));"
_META_SCHEMA_VERSION_KEY = "schema_version"
_META_SCHEMA_EPOCH_KEY = "schema_epoch"

# ---------------------------------------------------------------------------
# Base DDL — complete canonical schema for fresh installs (v1 clean slate)
# ---------------------------------------------------------------------------

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

_CREATE_KNOWLEDGE_SOURCES = """
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         VARCHAR(500) NOT NULL,
    source_type  VARCHAR(50)  NOT NULL DEFAULT 'DOCUMENT_UPLOAD',
    category_ids UUID[] DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_ITEMS = """
CREATE TABLE IF NOT EXISTS knowledge_items (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id          UUID REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    path               VARCHAR(1000),
    title              VARCHAR(500),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_updated_at TIMESTAMPTZ
);
"""

_CREATE_KNOWLEDGE_COMMUNITIES = """
CREATE TABLE IF NOT EXISTS knowledge_communities (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    embedding   vector(1536),
    member_count INTEGER NOT NULL DEFAULT 0,
    memory_count INTEGER NOT NULL DEFAULT 0,
    palace      VARCHAR(200),
    namespace   VARCHAR(200),
    room        VARCHAR(200),
    corridor    VARCHAR(200),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_MEMORIES = """
CREATE TABLE IF NOT EXISTS knowledge_memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id         UUID REFERENCES knowledge_items(id) ON DELETE CASCADE,
    community_id    UUID REFERENCES knowledge_communities(id) ON DELETE SET NULL,
    content         TEXT NOT NULL,
    embedding       vector(1536),
    search_vector   tsvector,
    -- Scoring & decay
    authority       VARCHAR(50)  NOT NULL DEFAULT 'EXTRACTED',
    lifecycle_state VARCHAR(50)  NOT NULL DEFAULT 'ACTIVE',
    confidence      FLOAT DEFAULT 0.5,
    base_score      FLOAT DEFAULT 0.5,
    relevance_score FLOAT DEFAULT NULL,
    retrieval_count INTEGER DEFAULT 0,
    -- Retrieval & decay timestamps
    last_retrieved_at TIMESTAMPTZ,
    -- Lifecycle timestamps
    archived_at    TIMESTAMPTZ,
    quarantined_at TIMESTAMPTZ,
    flagged_at     TIMESTAMPTZ,
    -- Embedding metadata
    embedding_model VARCHAR(100),
    -- Generic metadata bag
    metadata JSONB DEFAULT '{}'::jsonb,
    -- Audit trail
    created_by     UUID,
    archived_by    UUID,
    auth_method    VARCHAR(50),
    archive_reason VARCHAR(100),
    -- Source provenance (denormalised for query speed)
    source_name VARCHAR(500),
    source_type VARCHAR(50),
    -- Temporal validity window
    valid_from TIMESTAMPTZ,
    valid_to   TIMESTAMPTZ,
    -- Memory provenance model
    memory_tier VARCHAR(20) NOT NULL DEFAULT 'direct',
    derived_kind VARCHAR(20),
    parent_memory_ids UUID[] NOT NULL DEFAULT '{}',
    superseded_by_memory_id UUID REFERENCES knowledge_memories(id) ON DELETE RESTRICT,
    -- Topology / room system
    palace    VARCHAR(200),
    namespace VARCHAR(200),
    room      VARCHAR(200),
    corridor  VARCHAR(200),
    CONSTRAINT ck_km_memory_tier
        CHECK (memory_tier IN ('direct', 'derived')),
    CONSTRAINT ck_km_derived_kind
        CHECK (derived_kind IS NULL OR derived_kind IN ('community', 'summary')),
    CONSTRAINT ck_km_lineage_integrity
        CHECK (
            (
                memory_tier = 'direct'
                AND derived_kind IS NULL
                AND cardinality(parent_memory_ids) = 0
            )
            OR (
                memory_tier = 'derived'
                AND derived_kind IS NOT NULL
                AND cardinality(parent_memory_ids) > 0
            )
        ),
    CONSTRAINT ck_km_supersede_state
        CHECK (superseded_by_memory_id IS NULL OR lifecycle_state = 'SUPERSEDED'),
    -- Housekeeping
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_ENTITIES = """
CREATE TABLE IF NOT EXISTS knowledge_entities (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_id UUID REFERENCES knowledge_communities(id) ON DELETE SET NULL,
    palace      VARCHAR(200) NOT NULL DEFAULT '',
    namespace   VARCHAR(200) NOT NULL DEFAULT '',
    room        VARCHAR(200) NOT NULL DEFAULT '',
    corridor    VARCHAR(200) NOT NULL DEFAULT '',
    entity_type VARCHAR(50)  NOT NULL,
    name        VARCHAR(500) NOT NULL,
    confidence  FLOAT DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_KNOWLEDGE_MEMORY_AUDITS = """
CREATE TABLE IF NOT EXISTS knowledge_memory_audits (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id      UUID NOT NULL REFERENCES knowledge_memories(id) ON DELETE CASCADE,
    action         VARCHAR(50) NOT NULL,
    performed_by   UUID NOT NULL,
    auth_method    VARCHAR(50),
    performed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata       JSONB DEFAULT '{}'::jsonb
);
"""

_CREATE_KNOWLEDGE_CATEGORIES = """
CREATE TABLE IF NOT EXISTS knowledge_categories (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(500) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_knowledge_categories_name UNIQUE (name)
);
"""

_CREATE_KNOWLEDGE_MEMORY_CATEGORIES = """
CREATE TABLE IF NOT EXISTS knowledge_memory_categories (
    memory_id   UUID NOT NULL REFERENCES knowledge_memories(id) ON DELETE CASCADE,
    category_id UUID NOT NULL REFERENCES knowledge_categories(id) ON DELETE CASCADE,
    assigned_by    VARCHAR(20) NOT NULL DEFAULT 'EXPLICIT',
    assigned_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (memory_id, category_id)
);
"""

_CREATE_KNOWLEDGE_RELATIONS = """
CREATE TABLE IF NOT EXISTS knowledge_relations (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id        UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    target_entity_id        UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    relation_type           VARCHAR(100) NOT NULL,
    confidence              FLOAT DEFAULT 1.0,
    evidence_memory_id UUID REFERENCES knowledge_memories(id) ON DELETE SET NULL,
    evidence_memory_ids UUID[] NOT NULL DEFAULT '{}',
    curated BOOLEAN NOT NULL DEFAULT FALSE,
    valid_from              TIMESTAMPTZ,
    valid_to                TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_kr_corridor_evidence_required
        CHECK (
            UPPER(BTRIM(relation_type)) <> 'CORRIDOR'
            OR curated
            OR cardinality(evidence_memory_ids) > 0
        )
);
"""

_CREATE_KNOWLEDGE_ENTITY_MEMORIES = """
CREATE TABLE IF NOT EXISTS knowledge_entity_memories (
    memory_id UUID NOT NULL REFERENCES knowledge_memories(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    role           VARCHAR(100) NOT NULL DEFAULT 'mentioned',
    confidence     FLOAT DEFAULT 1.0,
    PRIMARY KEY (memory_id, entity_id)
);
"""

_CREATE_KNOWLEDGE_CONFLICTS = """
CREATE TABLE IF NOT EXISTS knowledge_conflicts (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    new_memory_id UUID REFERENCES knowledge_memories(id) ON DELETE CASCADE,
    old_memory_id UUID REFERENCES knowledge_memories(id) ON DELETE SET NULL,
    resolution         VARCHAR(100),
    resolved_at        TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_TRIGGERS = """
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_ks_updated_at') THEN
        CREATE TRIGGER trg_ks_updated_at
            BEFORE UPDATE ON knowledge_sources
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_ki_updated_at') THEN
        CREATE TRIGGER trg_ki_updated_at
            BEFORE UPDATE ON knowledge_items
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_kc_updated_at') THEN
        CREATE TRIGGER trg_kc_updated_at
            BEFORE UPDATE ON knowledge_communities
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_km_updated_at') THEN
        CREATE TRIGGER trg_km_updated_at
            BEFORE UPDATE ON knowledge_memories
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END $$;
"""

_CREATE_INDEXES = """
-- knowledge_memories
CREATE INDEX IF NOT EXISTS idx_km_lifecycle              ON knowledge_memories(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_km_community_id
    ON knowledge_memories(community_id)
    WHERE community_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_item_id                ON knowledge_memories(item_id);
CREATE INDEX IF NOT EXISTS idx_km_search_vector
    ON knowledge_memories USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_km_created_by             ON knowledge_memories(created_by);
CREATE INDEX IF NOT EXISTS idx_km_archived_by            ON knowledge_memories(archived_by);
CREATE INDEX IF NOT EXISTS idx_km_source_name            ON knowledge_memories(source_name);
CREATE INDEX IF NOT EXISTS idx_km_source_type            ON knowledge_memories(source_type);
CREATE INDEX IF NOT EXISTS idx_km_source_name_lifecycle
    ON knowledge_memories(source_name, lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_km_namespace
    ON knowledge_memories(namespace)
    WHERE namespace IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_room
    ON knowledge_memories(room)
    WHERE room IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_namespace_room
    ON knowledge_memories(namespace, room)
    WHERE namespace IS NOT NULL AND room IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_palace
    ON knowledge_memories(palace)
    WHERE palace IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_palace_namespace_room
    ON knowledge_memories(palace, namespace, room)
    WHERE palace IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_superseded_by
    ON knowledge_memories(superseded_by_memory_id)
    WHERE superseded_by_memory_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_relevance_score
    ON knowledge_memories(relevance_score ASC)
    WHERE relevance_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_quarantined_at
    ON knowledge_memories(quarantined_at)
    WHERE quarantined_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_flagged_at
    ON knowledge_memories(flagged_at)
    WHERE flagged_at IS NOT NULL;
-- knowledge_sources
CREATE INDEX  IF NOT EXISTS idx_ks_category_ids
    ON knowledge_sources USING gin(category_ids);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ks_name            ON knowledge_sources(name);
-- knowledge_items
CREATE INDEX  IF NOT EXISTS idx_ki_source_id             ON knowledge_items(source_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ki_source_path     ON knowledge_items(source_id, path);
-- knowledge_communities
CREATE INDEX IF NOT EXISTS idx_kc_palace
    ON knowledge_communities(palace)
    WHERE palace IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kc_namespace
    ON knowledge_communities(namespace)
    WHERE namespace IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kc_room
    ON knowledge_communities(room)
    WHERE room IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kc_namespace_room
    ON knowledge_communities(namespace, room)
    WHERE namespace IS NOT NULL AND room IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kc_palace_namespace_room
    ON knowledge_communities(palace, namespace, room)
    WHERE palace IS NOT NULL;
-- knowledge_entities
CREATE INDEX IF NOT EXISTS idx_ke_community_id
    ON knowledge_entities(community_id)
    WHERE community_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_type_name
    ON knowledge_entities(palace, namespace, room, corridor, entity_type, name);
-- knowledge_memory_audits
CREATE INDEX IF NOT EXISTS idx_kma_memory_id             ON knowledge_memory_audits(memory_id);
CREATE INDEX IF NOT EXISTS idx_kma_performed_by          ON knowledge_memory_audits(performed_by);
CREATE INDEX IF NOT EXISTS idx_kma_performed_at          ON knowledge_memory_audits(performed_at);
-- knowledge_memory_categories
CREATE INDEX IF NOT EXISTS idx_kmc_category_id
    ON knowledge_memory_categories(category_id);
CREATE INDEX IF NOT EXISTS idx_kmc_memory_id             ON knowledge_memory_categories(memory_id);
-- knowledge_relations
CREATE INDEX IF NOT EXISTS idx_kr_relation_type          ON knowledge_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_kr_source_target
    ON knowledge_relations(source_entity_id, target_entity_id);
CREATE INDEX IF NOT EXISTS idx_kr_temporal
    ON knowledge_relations(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_kr_evidence_memory_ids
    ON knowledge_relations USING gin(evidence_memory_ids);
-- knowledge_entity_memories
CREATE INDEX IF NOT EXISTS idx_kem_entity_id             ON knowledge_entity_memories(entity_id);
CREATE INDEX IF NOT EXISTS idx_kem_memory_id             ON knowledge_entity_memories(memory_id);
-- knowledge_conflicts
CREATE INDEX IF NOT EXISTS idx_kc_new_memory_id          ON knowledge_conflicts(new_memory_id);
"""

# HNSW index requires pgvector ≥ 0.5 — wrapped in exception block for safety
_CREATE_HNSW_INDEX = """
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_km_embedding
        ON knowledge_memories
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    CREATE INDEX IF NOT EXISTS idx_kc_embedding
        ON knowledge_communities
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
EXCEPTION WHEN others THEN
    RAISE WARNING 'HNSW index on knowledge embeddings skipped: %', SQLERRM;
END $$;
"""

# Ordered list of all DDL statements executed on a fresh install
_BASE_DDL: list[str] = [
    _CREATE_EXTENSION,
    _CREATE_KNOWLEDGE_SOURCES,
    _CREATE_KNOWLEDGE_ITEMS,
    _CREATE_KNOWLEDGE_COMMUNITIES,
    _CREATE_KNOWLEDGE_MEMORIES,
    _CREATE_KNOWLEDGE_ENTITIES,
    _CREATE_KNOWLEDGE_MEMORY_AUDITS,
    _CREATE_KNOWLEDGE_CATEGORIES,
    _CREATE_KNOWLEDGE_MEMORY_CATEGORIES,
    _CREATE_KNOWLEDGE_RELATIONS,
    _CREATE_KNOWLEDGE_ENTITY_MEMORIES,
    _CREATE_KNOWLEDGE_CONFLICTS,
    _CREATE_TRIGGERS,
    _CREATE_INDEXES,
    _CREATE_HNSW_INDEX,
]

_V1_BASELINE_SQL = "\n".join(_BASE_DDL)

_V2_SCOPED_ENTITY_UNIQUENESS_SQL = """
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS namespace VARCHAR(200) NOT NULL DEFAULT '';
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS room VARCHAR(200) NOT NULL DEFAULT '';
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS corridor VARCHAR(200) NOT NULL DEFAULT '';

DROP INDEX IF EXISTS idx_ke_type_name;
CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_type_name
    ON knowledge_entities(namespace, room, corridor, entity_type, name);
"""

_V3_MEMORY_LINEAGE_INTEGRITY_SQL = """
ALTER TABLE knowledge_memories
    ADD COLUMN IF NOT EXISTS memory_tier VARCHAR(20) NOT NULL DEFAULT 'direct';
ALTER TABLE knowledge_memories
    ADD COLUMN IF NOT EXISTS derived_kind VARCHAR(20);
ALTER TABLE knowledge_memories
    ADD COLUMN IF NOT EXISTS parent_memory_ids UUID[] NOT NULL DEFAULT '{}';
ALTER TABLE knowledge_memories
    ADD COLUMN IF NOT EXISTS superseded_by_memory_id UUID
        REFERENCES knowledge_memories(id)
        ON DELETE RESTRICT;

-- Deterministic pre-normalization for dirty pre-v3 rows before enabling constraints.
-- Safety policy: preserve data, collapse ambiguous lineage to direct, and only keep
-- derived rows when they already carry explicit valid lineage hints.
UPDATE knowledge_memories
SET memory_tier = CASE
    WHEN memory_tier IN ('direct', 'derived') THEN memory_tier
    WHEN derived_kind IN ('community', 'summary') THEN 'derived'
    WHEN COALESCE(array_length(parent_memory_ids, 1), 0) > 0 THEN 'derived'
    ELSE 'direct'
END;

UPDATE knowledge_memories
SET derived_kind = CASE
    WHEN memory_tier = 'derived' AND derived_kind IN ('community', 'summary') THEN derived_kind
    WHEN memory_tier = 'derived' THEN 'summary'
    ELSE NULL
END;

UPDATE knowledge_memories
SET parent_memory_ids = '{}'
WHERE parent_memory_ids IS NULL;

UPDATE knowledge_memories
SET parent_memory_ids = '{}'
WHERE memory_tier = 'direct';

UPDATE knowledge_memories
SET memory_tier = 'direct',
    derived_kind = NULL,
    parent_memory_ids = '{}'
WHERE memory_tier = 'derived'
  AND COALESCE(array_length(parent_memory_ids, 1), 0) = 0;

UPDATE knowledge_memories
SET lifecycle_state = 'SUPERSEDED'
WHERE superseded_by_memory_id IS NOT NULL
  AND lifecycle_state <> 'SUPERSEDED';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_km_memory_tier'
    ) THEN
        ALTER TABLE knowledge_memories
            ADD CONSTRAINT ck_km_memory_tier
            CHECK (memory_tier IN ('direct', 'derived'));
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_km_derived_kind'
    ) THEN
        ALTER TABLE knowledge_memories
            ADD CONSTRAINT ck_km_derived_kind
            CHECK (derived_kind IS NULL OR derived_kind IN ('community', 'summary'));
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_km_lineage_integrity'
    ) THEN
        ALTER TABLE knowledge_memories
            ADD CONSTRAINT ck_km_lineage_integrity
            CHECK (
                (
                    memory_tier = 'direct'
                    AND derived_kind IS NULL
                    AND cardinality(parent_memory_ids) = 0
                )
                OR (
                    memory_tier = 'derived'
                    AND derived_kind IS NOT NULL
                    AND cardinality(parent_memory_ids) > 0
                )
            );
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_km_supersede_state'
    ) THEN
        ALTER TABLE knowledge_memories
            ADD CONSTRAINT ck_km_supersede_state
            CHECK (superseded_by_memory_id IS NULL OR lifecycle_state = 'SUPERSEDED');
    END IF;
END $$;

CREATE OR REPLACE FUNCTION enforce_km_supersede_append_only()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.lifecycle_state = 'SUPERSEDED' AND NEW.lifecycle_state <> 'SUPERSEDED' THEN
        RAISE EXCEPTION 'MEM_SUPERSEDE_APPEND_ONLY: superseded lifecycle state cannot be reverted';
    END IF;

    IF OLD.superseded_by_memory_id IS NOT NULL
       AND NEW.superseded_by_memory_id IS DISTINCT FROM OLD.superseded_by_memory_id THEN
        RAISE EXCEPTION 'MEM_SUPERSEDE_APPEND_ONLY: superseded_by_memory_id is immutable once set';
    END IF;

    IF NEW.superseded_by_memory_id IS NOT NULL
       AND NEW.lifecycle_state <> 'SUPERSEDED' THEN
        RAISE EXCEPTION
            'MEM_SUPERSEDE_STATE_REQUIRED: superseded_by_memory_id requires '
            'SUPERSEDED lifecycle_state';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_km_supersede_append_only ON knowledge_memories;
CREATE TRIGGER trg_km_supersede_append_only
    BEFORE UPDATE ON knowledge_memories
    FOR EACH ROW EXECUTE FUNCTION enforce_km_supersede_append_only();

CREATE INDEX IF NOT EXISTS idx_km_superseded_by
    ON knowledge_memories(superseded_by_memory_id)
    WHERE superseded_by_memory_id IS NOT NULL;
"""

_V4_RELATION_EVIDENCE_LINKAGE_SQL = """
ALTER TABLE knowledge_relations
    ADD COLUMN IF NOT EXISTS evidence_memory_ids UUID[] NOT NULL DEFAULT '{}';

ALTER TABLE knowledge_relations
    ADD COLUMN IF NOT EXISTS curated BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE knowledge_relations
SET evidence_memory_ids = ARRAY[evidence_memory_id]::uuid[]
WHERE evidence_memory_id IS NOT NULL
  AND cardinality(evidence_memory_ids) = 0;

UPDATE knowledge_relations
SET curated = CASE
    WHEN cardinality(evidence_memory_ids) > 0 THEN FALSE
    ELSE COALESCE(curated, FALSE)
END;

-- Startup-safe normalization: legacy corridor edges that have no evidence linkage
-- are promoted to curated=true before enabling corridor evidence constraints.
UPDATE knowledge_relations
SET curated = TRUE
WHERE UPPER(BTRIM(relation_type)) = 'CORRIDOR'
  AND NOT COALESCE(curated, FALSE)
  AND cardinality(evidence_memory_ids) = 0;

ALTER TABLE knowledge_relations
    DROP CONSTRAINT IF EXISTS ck_kr_corridor_evidence_required;

DO $$
BEGIN
    ALTER TABLE knowledge_relations
        ADD CONSTRAINT ck_kr_corridor_evidence_required
        CHECK (
            UPPER(BTRIM(relation_type)) <> 'CORRIDOR'
            OR curated
            OR cardinality(evidence_memory_ids) > 0
        );
END $$;

CREATE INDEX IF NOT EXISTS idx_kr_evidence_memory_ids
    ON knowledge_relations USING gin(evidence_memory_ids);
"""

_V5_PALACE_SQL = """
-- Add palace column to topology tables.
-- palace is the org-level discriminator above namespace/wing.
-- Default '' (empty string) for NOT NULL tables; NULL for nullable tables.

ALTER TABLE knowledge_memories
    ADD COLUMN IF NOT EXISTS palace VARCHAR(200);

ALTER TABLE knowledge_communities
    ADD COLUMN IF NOT EXISTS palace VARCHAR(200);

ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS palace VARCHAR(200) NOT NULL DEFAULT '';

-- Indexes for palace-scoped retrieval on knowledge_memories
CREATE INDEX IF NOT EXISTS idx_km_palace
    ON knowledge_memories(palace)
    WHERE palace IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_km_palace_namespace_room
    ON knowledge_memories(palace, namespace, room)
    WHERE palace IS NOT NULL;

-- Indexes for palace-scoped retrieval on knowledge_communities
CREATE INDEX IF NOT EXISTS idx_kc_palace
    ON knowledge_communities(palace)
    WHERE palace IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kc_palace_namespace_room
    ON knowledge_communities(palace, namespace, room)
    WHERE palace IS NOT NULL;

-- Rebuild entity uniqueness index to include palace so that identical
-- entity names in colliding namespace/room/corridor under different
-- palaces are treated as distinct entities.
DROP INDEX IF EXISTS idx_ke_type_name;
CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_type_name
    ON knowledge_entities(palace, namespace, room, corridor, entity_type, name);
"""

_V6_SOURCE_PALACE_SQL = """
-- B3 hybrid palace ownership: sources are project-owned.
-- Existing rows cannot be backfilled safely because source.name was global.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'knowledge_sources'
          AND column_name = 'palace'
    ) AND EXISTS (SELECT 1 FROM knowledge_sources) THEN
        RAISE EXCEPTION
            'KS_PALACE_BACKFILL_REQUIRED: assign palace per row before v6';
    END IF;
END $$;

ALTER TABLE knowledge_sources
    ADD COLUMN IF NOT EXISTS palace TEXT;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM knowledge_sources WHERE palace IS NULL OR palace = '') THEN
        RAISE EXCEPTION
            'KS_PALACE_BACKFILL_REQUIRED: palace must be set before NOT NULL';
    END IF;
END $$;

ALTER TABLE knowledge_sources
    ALTER COLUMN palace SET NOT NULL;

DROP INDEX IF EXISTS idx_ks_name;
CREATE UNIQUE INDEX IF NOT EXISTS idx_ks_palace_name
    ON knowledge_sources(palace, name);
"""

_V7_ITEM_FILE_IDENTITY_SQL = """
-- §3.1 Knowledge ingestion architecture + B3 hybrid palace ownership.
-- File-backed items are always tied to a source and palace.

ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS size_bytes BIGINT NOT NULL DEFAULT 0;
ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS mtime_ns BIGINT NOT NULL DEFAULT 0;
ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS language TEXT;
ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS palace TEXT;

UPDATE knowledge_items ki
SET palace = ks.palace
FROM knowledge_sources ks
WHERE ki.source_id = ks.id
  AND ki.palace IS NULL;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM knowledge_items WHERE source_id IS NULL) THEN
        RAISE EXCEPTION
            'KI_SOURCE_ID_REQUIRED: attach items to a source before v7';
    END IF;
    IF EXISTS (SELECT 1 FROM knowledge_items WHERE palace IS NULL OR palace = '') THEN
        RAISE EXCEPTION
            'KI_PALACE_BACKFILL_REQUIRED: palace not derivable from sources';
    END IF;
END $$;

ALTER TABLE knowledge_items
    ALTER COLUMN source_id SET NOT NULL;
ALTER TABLE knowledge_items
    ALTER COLUMN palace SET NOT NULL;

DROP INDEX IF EXISTS idx_ki_source_path;
CREATE UNIQUE INDEX IF NOT EXISTS idx_ki_palace_source_path
    ON knowledge_items(palace, source_id, path);

CREATE INDEX IF NOT EXISTS idx_ki_palace_content_hash
    ON knowledge_items(palace, content_hash)
    WHERE content_hash <> '';
"""

_V8_ITEM_SOURCE_PALACE_TRIGGER_SQL = """
-- Cross-table invariant: knowledge_items.palace must equal knowledge_sources.palace.

CREATE OR REPLACE FUNCTION enforce_ki_source_palace_match()
RETURNS TRIGGER AS $$
DECLARE
    source_palace TEXT;
BEGIN
    SELECT palace INTO source_palace
      FROM knowledge_sources
     WHERE id = NEW.source_id;

    IF source_palace IS NULL THEN
        RAISE EXCEPTION 'KI_SOURCE_NOT_FOUND: source_id % missing', NEW.source_id;
    END IF;

    IF NEW.palace IS DISTINCT FROM source_palace THEN
        RAISE EXCEPTION
            'KNOWLEDGE_ITEM_SOURCE_PALACE_MISMATCH: item palace % does not match source palace %',
            NEW.palace, source_palace;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ki_source_palace_match ON knowledge_items;
CREATE TRIGGER trg_ki_source_palace_match
    BEFORE INSERT OR UPDATE OF palace, source_id ON knowledge_items
    FOR EACH ROW EXECUTE FUNCTION enforce_ki_source_palace_match();
"""

_V9_ENTITY_STRUCTURAL_SQL = """
-- §3.2 Knowledge ingestion architecture: structural identity columns.
-- knowledge_entities.source is an origin enum, NOT a source/file name.
-- File/source identity flows through source_item_id -> knowledge_items -> knowledge_sources.

ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'EXTRACTED';
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS authority TEXT NOT NULL DEFAULT 'EXTRACTED';
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS stable_id TEXT;
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS source_item_id UUID REFERENCES knowledge_items(id) ON DELETE SET NULL;
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_ke_source_origin'
    ) THEN
        ALTER TABLE knowledge_entities
            ADD CONSTRAINT ck_ke_source_origin
            CHECK (source IN ('STRUCTURAL', 'EXTRACTED', 'USER'));
    END IF;
END $$;

DROP INDEX IF EXISTS idx_ke_palace_source_stable_id;
CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_palace_source_stable_id
    ON knowledge_entities(palace, source, stable_id)
    WHERE stable_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ke_source_item_id
    ON knowledge_entities(source_item_id)
    WHERE source_item_id IS NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_ke_updated_at') THEN
        CREATE TRIGGER trg_ke_updated_at
            BEFORE UPDATE ON knowledge_entities
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END $$;
"""

_V10_ENTITY_EMBEDDINGS_SQL = """
-- §3.3 Knowledge ingestion architecture: side table for entity embeddings.

CREATE TABLE IF NOT EXISTS knowledge_entity_embeddings (
    entity_id  UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    profile    TEXT NOT NULL,
    model      TEXT NOT NULL,
    dimension  INTEGER NOT NULL,
    embedding  vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_id, profile)
);

CREATE INDEX IF NOT EXISTS idx_kee_profile
    ON knowledge_entity_embeddings(profile);

-- HNSW per profile is built lazily by the embedding workflow because
-- different profiles can have different dimensions.
"""

_V11_ENTITY_MEMORY_ANCHOR_SQL = """
-- §3.4 Knowledge ingestion architecture: span anchoring for memory→entity links.

ALTER TABLE knowledge_entity_memories
    ADD COLUMN IF NOT EXISTS start_line INTEGER;
ALTER TABLE knowledge_entity_memories
    ADD COLUMN IF NOT EXISTS end_line INTEGER;
ALTER TABLE knowledge_entity_memories
    ADD COLUMN IF NOT EXISTS start_col INTEGER;
ALTER TABLE knowledge_entity_memories
    ADD COLUMN IF NOT EXISTS end_col INTEGER;
ALTER TABLE knowledge_entity_memories
    ADD COLUMN IF NOT EXISTS anchor_kind TEXT NOT NULL DEFAULT 'symbol';
"""

_V12_COMMUNITY_FEDERATION_SQL = """
-- §3.5 Knowledge ingestion architecture: federation provenance on communities.

ALTER TABLE knowledge_communities
    ADD COLUMN IF NOT EXISTS scope_kind TEXT NOT NULL DEFAULT 'palace_local';
ALTER TABLE knowledge_communities
    ADD COLUMN IF NOT EXISTS source_palaces TEXT[];
ALTER TABLE knowledge_communities
    ADD COLUMN IF NOT EXISTS member_provenance JSONB;
ALTER TABLE knowledge_communities
    ADD COLUMN IF NOT EXISTS consent_policy_id UUID;

CREATE INDEX IF NOT EXISTS idx_kc_scope_kind
    ON knowledge_communities(scope_kind);
"""

_V13_ITEM_LIFECYCLE_SQL = """
-- §3.6 Knowledge ingestion architecture: item lifecycle states.

ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'ACTIVE';
ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS error_metadata JSONB;

CREATE INDEX IF NOT EXISTS idx_ki_palace_lifecycle_state
    ON knowledge_items(palace, lifecycle_state);
"""

_V14_TRACK4_PREREQS_SQL = """
-- v14: Track 4 prerequisites — structural entity columns + relation metadata.
-- See docs/architecture/knowledge-ingestion.md §4.3, §4.4.

-- knowledge_entities: promote qualified_name to first-class for symbol lookup,
-- stable_id derivation join, and CALLS-edge resolution joins.
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS qualified_name TEXT;

-- knowledge_entities: parent_class_id as self-FK for Class->Method containment.
-- ON DELETE SET NULL: deleting a class must not cascade-delete its methods,
-- which would lose memory anchoring. Methods become detectable orphans.
ALTER TABLE knowledge_entities
    ADD COLUMN IF NOT EXISTS parent_class_id UUID
        REFERENCES knowledge_entities(id) ON DELETE SET NULL;

-- Lookup index for symbol resolution (used by CALLS-edge construction in System 1).
-- Includes palace + source_item_id to keep cardinality low and lookups palace-isolated.
CREATE INDEX IF NOT EXISTS idx_knowledge_entities_qualified_name
    ON knowledge_entities(palace, source_item_id, qualified_name)
    WHERE qualified_name IS NOT NULL;

-- Reverse-index for "find methods of class X".
CREATE INDEX IF NOT EXISTS idx_knowledge_entities_parent_class
    ON knowledge_entities(parent_class_id)
    WHERE parent_class_id IS NOT NULL;

-- knowledge_relations: metadata JSONB for resolution flags, call-site spans,
-- and any future per-edge provenance. NOT NULL with empty default keeps
-- existing rows valid and avoids null-handling at every read site.
ALTER TABLE knowledge_relations
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
"""

_HAS_KNOWLEDGE_TABLES_SQL = """
SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
            AND table_name IN (
                'knowledge_sources',
                'knowledge_items',
                'knowledge_communities',
                'knowledge_memories',
                'knowledge_entities',
                'knowledge_memory_audits',
                'knowledge_categories',
                'knowledge_memory_categories',
                'knowledge_relations',
                'knowledge_entity_memories',
                'knowledge_conflicts',
                'knowledge_entity_embeddings'
            )
) AS has_tables;
"""

# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------
# Each entry: (version, description, sql)
# SQL MUST be idempotent — safe on both fresh and existing databases.
#
# BREAKING CHANGE (April 2026): migrations v1–v19 were consolidated into the
# base DDL above.  Existing databases must be dropped and re-created.
# This single v1 entry is the executable baseline for fresh installs.

MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "Initial schema (consolidated clean-slate release)",
        _V1_BASELINE_SQL,
    ),
    (
        2,
        "Scope knowledge entity uniqueness by topology keys",
        _V2_SCOPED_ENTITY_UNIQUENESS_SQL,
    ),
    (
        3,
        "Enforce memory lineage integrity and append-only supersede semantics",
        _V3_MEMORY_LINEAGE_INTEGRITY_SQL,
    ),
    (
        4,
        "Add relation evidence linkage for non-curated corridor edges",
        _V4_RELATION_EVIDENCE_LINKAGE_SQL,
    ),
    (
        5,
        "Add palace column to topology tables for org-level isolation",
        _V5_PALACE_SQL,
    ),
    (
        6,
        "Add palace ownership and per-palace name uniqueness to knowledge_sources",
        _V6_SOURCE_PALACE_SQL,
    ),
    (
        7,
        "Add file metadata and palace-scoped file identity to knowledge_items",
        _V7_ITEM_FILE_IDENTITY_SQL,
    ),
    (
        8,
        "Enforce knowledge_items palace/source palace invariant",
        _V8_ITEM_SOURCE_PALACE_TRIGGER_SQL,
    ),
    (
        9,
        "Add structural identity and metadata columns to knowledge_entities",
        _V9_ENTITY_STRUCTURAL_SQL,
    ),
    (
        10,
        "Add knowledge_entity_embeddings side table",
        _V10_ENTITY_EMBEDDINGS_SQL,
    ),
    (
        11,
        "Add anchor span columns to knowledge_entity_memories",
        _V11_ENTITY_MEMORY_ANCHOR_SQL,
    ),
    (
        12,
        "Add federation provenance columns to knowledge_communities",
        _V12_COMMUNITY_FEDERATION_SQL,
    ),
    (
        13,
        "Add lifecycle_state and error_metadata to knowledge_items",
        _V13_ITEM_LIFECYCLE_SQL,
    ),
    (
        14,
        "Add qualified_name + parent_class_id to entities, metadata to relations (Track 4 prereqs)",
        _V14_TRACK4_PREREQS_SQL,
    ),
]

SCHEMA_VERSION = MIGRATIONS[-1][0] if MIGRATIONS else 0
SCHEMA_EPOCH = 2


def _has_executable_sql(sql: str) -> bool:
    """Return True when a migration script contains executable SQL.

    Comment-only/no-op scripts are valid migration markers and should be skipped.
    """
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return True
    return False


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


async def ensure_schema(backend: Any) -> None:
    """Ensure knowledge tables exist and are up to date.

    This is the sole entry point for schema management.  Safe to call on
    every server startup — it is a fast no-op when the schema is current.

    Args:
        backend: A connected ``DatabaseBackendBase`` instance exposing
            ``execute_script(sql)``, ``query(sql)``, and ``execute(sql)``
            coroutines.
    """
    # 1. Create version-tracking meta table
    await backend.execute_script(
        "CREATE TABLE IF NOT EXISTS _knowledge_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    )

    # 2. Acquire schema lock so only one startup mutates schema at a time.
    await backend.query(_SCHEMA_LOCK_SQL)
    try:
        # 3. Read current schema metadata.
        version_result = await backend.query(
            f"SELECT value FROM _knowledge_meta WHERE key = '{_META_SCHEMA_VERSION_KEY}'"
        )
        current_version = int(version_result.rows[0]["value"]) if version_result.rows else 0

        epoch_result = await backend.query(
            f"SELECT value FROM _knowledge_meta WHERE key = '{_META_SCHEMA_EPOCH_KEY}'"
        )
        current_epoch = int(epoch_result.rows[0]["value"]) if epoch_result.rows else None

        table_result = await backend.query(_HAS_KNOWLEDGE_TABLES_SQL)
        has_knowledge_tables = bool(table_result.rows and table_result.rows[0]["has_tables"])

        is_incompatible = False
        if has_knowledge_tables and current_epoch != SCHEMA_EPOCH:
            is_incompatible = True
        elif current_version > SCHEMA_VERSION:
            is_incompatible = True

        # 4. Fail closed for incompatible schemas; runtime reset is never destructive.
        if is_incompatible:
            raise RuntimeError(
                "Incompatible knowledge schema detected. "
                "Run the documented migration path for this release; "
                "destructive runtime reset is disabled. "
                f"(current_epoch={current_epoch}, current_version={current_version}, "
                f"expected_epoch={SCHEMA_EPOCH}, expected_version={SCHEMA_VERSION})"
            )

        if current_version >= SCHEMA_VERSION and current_epoch == SCHEMA_EPOCH:
            return

        # 5. Apply pending migrations in version order.
        for version, description, sql in MIGRATIONS:
            if version <= current_version:
                continue
            logger.info("Applying knowledge schema migration v%d: %s", version, description)
            if not _has_executable_sql(sql):
                logger.info(
                    "Skipping no-op knowledge schema migration v%d: %s",
                    version,
                    description,
                )
                continue
            await backend.execute_script(sql)

        # 6. Persist current epoch/version markers.
        await backend.execute_script(
            f"INSERT INTO _knowledge_meta (key, value) VALUES "
            f"('{_META_SCHEMA_VERSION_KEY}', '{SCHEMA_VERSION}') "
            f"ON CONFLICT (key) DO UPDATE SET value = '{SCHEMA_VERSION}';"
        )
        await backend.execute_script(
            f"INSERT INTO _knowledge_meta (key, value) VALUES "
            f"('{_META_SCHEMA_EPOCH_KEY}', '{SCHEMA_EPOCH}') "
            f"ON CONFLICT (key) DO UPDATE SET value = '{SCHEMA_EPOCH}';"
        )
        logger.info("Knowledge schema is at epoch %d version %d", SCHEMA_EPOCH, SCHEMA_VERSION)
    finally:
        await backend.query(_SCHEMA_UNLOCK_SQL)
