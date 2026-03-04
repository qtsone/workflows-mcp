"""Idempotent DDL for Knowledge tables.

Creates pgvector extension and knowledge tables on first use.
"""

# The full DDL is split into statements for readability but assembled
# into a single script string for PostgresBackend.execute_script().

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

_CREATE_KNOWLEDGE_SOURCES = """
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
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
    org_id UUID NOT NULL,
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
    org_id UUID NOT NULL,
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
    metadata_ JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_kp_org_id ON knowledge_propositions(org_id);
CREATE INDEX IF NOT EXISTS idx_kp_lifecycle ON knowledge_propositions(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_kp_item_id ON knowledge_propositions(item_id);
CREATE INDEX IF NOT EXISTS idx_kp_search_vector ON knowledge_propositions USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_ks_org_id ON knowledge_sources(org_id);
CREATE INDEX IF NOT EXISTS idx_ks_category_ids ON knowledge_sources USING gin(category_ids);
CREATE INDEX IF NOT EXISTS idx_ki_org_id ON knowledge_items(org_id);
CREATE INDEX IF NOT EXISTS idx_ki_source_id ON knowledge_items(source_id);
"""


def get_init_schema_sql() -> str:
    """Return the full idempotent DDL script for knowledge tables.

    Safe to run multiple times — uses IF NOT EXISTS throughout.
    Designed for PostgresBackend.execute_script() (multi-statement).
    """
    return "\n".join(
        [
            _CREATE_EXTENSION,
            _CREATE_KNOWLEDGE_SOURCES,
            _CREATE_KNOWLEDGE_ITEMS,
            _CREATE_KNOWLEDGE_PROPOSITIONS,
            _CREATE_INDEXES,
        ]
    )
