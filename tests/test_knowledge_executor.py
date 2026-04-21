"""Tests for the Knowledge executor and supporting modules.

Tests focus on pure logic (RRF fusion, context assembly, schema DDL,
migration correctness) and Memory executor registration — no real database needed.
PostgreSQL-specific tests require optional deps.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from workflows_mcp.engine.executors_memory import (
    MemoryExecutor,
    MemoryInput,
    MemoryOutput,
)
from workflows_mcp.engine.knowledge.constants import (
    Authority,
    LifecycleState,
)
from workflows_mcp.engine.knowledge.context import (
    _cosine_similarity,
    _mmr_rerank,
    assemble_context,
    estimate_tokens,
)
from workflows_mcp.engine.knowledge.schema import (
    _CREATE_EXTENSION,
    _CREATE_INDEXES,
    _CREATE_KNOWLEDGE_COMMUNITIES,
    _CREATE_KNOWLEDGE_CONFLICTS,
    _CREATE_KNOWLEDGE_ENTITIES,
    _CREATE_KNOWLEDGE_ITEMS,
    _CREATE_KNOWLEDGE_MEMORIES,
    _CREATE_KNOWLEDGE_SOURCES,
    _V1_BASELINE_SQL,
    _V3_MEMORY_LINEAGE_INTEGRITY_SQL,
    MIGRATIONS,
    SCHEMA_EPOCH,
    SCHEMA_VERSION,
    ensure_schema,
)
from workflows_mcp.engine.knowledge.search import (
    build_fts_search_query,
    build_vector_search_query,
    rrf_fusion,
)


def _temporal_truth_matches(
    as_of: str,
    valid_from: str | None,
    valid_to: str | None,
) -> bool:
    """Mirror RFC temporal semantics used in SQL predicates for test expectations."""
    return (valid_from is None or valid_from <= as_of) and (valid_to is None or valid_to >= as_of)


# All DDL combined — used by test_ddl_is_idempotent to check IF NOT EXISTS guards.
_ALL_DDL = "\n".join(
    [
        _CREATE_EXTENSION,
        _CREATE_KNOWLEDGE_SOURCES,
        _CREATE_KNOWLEDGE_ITEMS,
        _CREATE_KNOWLEDGE_COMMUNITIES,
        _CREATE_KNOWLEDGE_MEMORIES,
        _CREATE_KNOWLEDGE_ENTITIES,
        _CREATE_INDEXES,
    ]
)


class TestConstants:
    """Tests for knowledge constants."""

    def test_lifecycle_states(self) -> None:
        """All lifecycle states should be string enums."""
        assert LifecycleState.ACTIVE == "ACTIVE"
        assert LifecycleState.QUARANTINED == "QUARANTINED"
        assert LifecycleState.FLAGGED == "FLAGGED"
        assert LifecycleState.ARCHIVED == "ARCHIVED"

    def test_authority_levels(self) -> None:
        """All authority levels should be string enums."""
        assert Authority.EXTRACTED == "EXTRACTED"
        assert Authority.COMMUNITY_SUMMARY == "COMMUNITY_SUMMARY"
        assert Authority.USER_VALIDATED == "USER_VALIDATED"
        assert Authority.AGENT == "AGENT"


class TestSchemaDDL:
    """Tests for idempotent schema DDL constants."""

    def test_schema_uses_memory_table_names(self) -> None:
        """Canonical baseline DDL should use memory table names only."""
        assert "CREATE TABLE IF NOT EXISTS knowledge_memories" in _V1_BASELINE_SQL
        assert "CREATE TABLE IF NOT EXISTS knowledge_entity_memories" in _V1_BASELINE_SQL
        assert "knowledge_propositions" not in _V1_BASELINE_SQL

    def test_ddl_contains_extension(self) -> None:
        """DDL should create pgvector extension."""
        assert "CREATE EXTENSION IF NOT EXISTS vector" in _CREATE_EXTENSION

    def test_ddl_contains_all_tables(self) -> None:
        """DDL should create all knowledge tables."""
        assert "CREATE TABLE IF NOT EXISTS knowledge_sources" in _CREATE_KNOWLEDGE_SOURCES
        assert "CREATE TABLE IF NOT EXISTS knowledge_items" in _CREATE_KNOWLEDGE_ITEMS
        assert "CREATE TABLE IF NOT EXISTS knowledge_communities" in _CREATE_KNOWLEDGE_COMMUNITIES
        assert "CREATE TABLE IF NOT EXISTS knowledge_memories" in _CREATE_KNOWLEDGE_MEMORIES
        assert "CREATE TABLE IF NOT EXISTS knowledge_entities" in _CREATE_KNOWLEDGE_ENTITIES
        assert "CREATE TABLE IF NOT EXISTS knowledge_conflicts" in _CREATE_KNOWLEDGE_CONFLICTS

    def test_ddl_contains_community_id_columns(self) -> None:
        """Entity and memory tables should persist community assignments directly."""
        assert "community_id" in _CREATE_KNOWLEDGE_MEMORIES
        assert "REFERENCES knowledge_communities(id)" in _CREATE_KNOWLEDGE_MEMORIES
        assert "community_id" in _CREATE_KNOWLEDGE_ENTITIES
        assert "REFERENCES knowledge_communities(id)" in _CREATE_KNOWLEDGE_ENTITIES

    def test_ddl_contains_indexes(self) -> None:
        """DDL should create necessary indexes."""
        assert "CREATE INDEX IF NOT EXISTS idx_km_lifecycle" in _CREATE_INDEXES
        assert "CREATE INDEX IF NOT EXISTS idx_km_search_vector" in _CREATE_INDEXES

    def test_ddl_contains_unique_source_index(self) -> None:
        """DDL should create unique index on knowledge_sources(name)."""
        assert "idx_ks_name" in _CREATE_INDEXES
        assert "knowledge_sources(name)" in _CREATE_INDEXES

    def test_ddl_contains_unique_entity_index(self) -> None:
        """DDL should scope entity uniqueness by topology + type + name."""
        assert "idx_ke_type_name" in _CREATE_INDEXES
        assert "knowledge_entities(namespace, room, corridor, entity_type, name)" in _CREATE_INDEXES

    def test_ddl_contains_unique_source_path_index(self) -> None:
        """DDL should create unique index on knowledge_items(source_id, path)."""
        assert "idx_ki_source_path" in _CREATE_INDEXES
        assert "knowledge_items(source_id, path)" in _CREATE_INDEXES

    def test_ddl_propositions_uses_metadata_not_metadata_underscore(self) -> None:
        """knowledge_memories DDL must use 'metadata' column, not 'metadata_'."""
        assert "metadata JSONB" in _CREATE_KNOWLEDGE_MEMORIES
        assert "metadata_" not in _CREATE_KNOWLEDGE_MEMORIES

    def test_ddl_propositions_uses_vector_1536(self) -> None:
        """knowledge_memories DDL must declare embedding as vector(1536), not dimensionless."""
        assert "embedding" in _CREATE_KNOWLEDGE_MEMORIES
        assert "vector(1536)" in _CREATE_KNOWLEDGE_MEMORIES
        assert "embedding vector," not in _CREATE_KNOWLEDGE_MEMORIES

    def test_ddl_propositions_has_no_embedding_dimensions_column(self) -> None:
        """knowledge_memories DDL must not include the redundant embedding_dimensions column."""
        assert "embedding_dimensions" not in _CREATE_KNOWLEDGE_MEMORIES

    def test_ddl_includes_content_updated_at(self) -> None:
        """Base DDL for knowledge_items must include content_updated_at."""
        assert "content_updated_at" in _CREATE_KNOWLEDGE_ITEMS

    def test_ddl_includes_decay_and_lifecycle_columns(self) -> None:
        """Base DDL must include decay and lifecycle support columns."""
        assert "base_score" in _CREATE_KNOWLEDGE_MEMORIES
        assert "relevance_score" in _CREATE_KNOWLEDGE_MEMORIES
        assert "last_retrieved_at" in _CREATE_KNOWLEDGE_MEMORIES
        assert "quarantined_at" in _CREATE_KNOWLEDGE_MEMORIES
        assert "flagged_at" in _CREATE_KNOWLEDGE_MEMORIES

    def test_migration_v3_normalizes_dirty_rows_before_constraints(self) -> None:
        """v3 migration normalizes legacy rows before lineage constraints are added."""
        assert "UPDATE knowledge_memories" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "SET memory_tier = CASE" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "SET derived_kind = CASE" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "WHERE parent_memory_ids IS NULL" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "SET parent_memory_ids = '{}'" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "SET lifecycle_state = 'SUPERSEDED'" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "ADD CONSTRAINT ck_km_lineage_integrity" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "ADD CONSTRAINT ck_km_supersede_state" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL

    def test_migration_v3_defines_append_only_supersede_trigger_guards(self) -> None:
        """v3 migration must enforce append-only supersede semantics via trigger guards."""
        assert (
            "CREATE OR REPLACE FUNCTION enforce_km_supersede_append_only()"
            in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        )
        assert "OLD.lifecycle_state = 'SUPERSEDED'" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "NEW.lifecycle_state <> 'SUPERSEDED'" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert "OLD.superseded_by_memory_id IS NOT NULL" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert (
            "NEW.superseded_by_memory_id IS DISTINCT FROM OLD.superseded_by_memory_id"
            in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        )
        assert "MEM_SUPERSEDE_APPEND_ONLY" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        assert (
            "DROP TRIGGER IF EXISTS trg_km_supersede_append_only"
            in _V3_MEMORY_LINEAGE_INTEGRITY_SQL
        )
        assert "CREATE TRIGGER trg_km_supersede_append_only" in _V3_MEMORY_LINEAGE_INTEGRITY_SQL


class TestMigrationV1:
    """Tests for the consolidated clean-slate baseline migration."""

    def test_migration_v1_present(self) -> None:
        """MIGRATIONS must include v1 consolidated clean-slate baseline."""
        assert MIGRATIONS[0][0] == 1
        assert MIGRATIONS[0][1] == "Initial schema (consolidated clean-slate release)"
        assert "CREATE TABLE IF NOT EXISTS knowledge_sources" in MIGRATIONS[0][2]

    def test_schema_version_is_latest_declared_migration(self) -> None:
        """SCHEMA_VERSION must match the latest migration marker."""
        assert SCHEMA_VERSION >= 1
        assert SCHEMA_VERSION == MIGRATIONS[-1][0]

    def test_ddl_is_idempotent(self) -> None:
        """All DDL statements should use IF NOT EXISTS guards."""
        for line in _ALL_DDL.split("\n"):
            line = line.strip()
            if line.startswith("CREATE TABLE"):
                assert "IF NOT EXISTS" in line, f"Missing IF NOT EXISTS: {line}"
            if line.startswith(("CREATE INDEX", "CREATE UNIQUE INDEX")):
                assert "IF NOT EXISTS" in line, f"Missing IF NOT EXISTS: {line}"
            if line.startswith("CREATE EXTENSION"):
                assert "IF NOT EXISTS" in line, f"Missing IF NOT EXISTS: {line}"


class TestSearchQueryBuilder:
    """Tests for SQL query builders using asyncpg $N params."""

    def test_vector_search_basic(self) -> None:
        """Basic vector search should produce valid SQL and params."""
        embedding = [0.1, 0.2, 0.3]
        sql, params = build_vector_search_query(
            query_embedding=embedding,
        )

        assert "$1" in sql  # embedding param
        assert "<=>" in sql  # cosine distance
        assert "knowledge_memories" in sql
        assert len(params) == 4  # embedding, state, confidence, candidate_limit

    def test_vector_search_with_source_filter(self) -> None:
        """Source filter uses denormalized source_name column — no JOIN required."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            source="internal-docs",
        )

        assert "kp.source_name =" in sql
        assert "internal-docs" in params
        assert "knowledge_sources" not in sql

    def test_vector_search_with_source_prefix(self) -> None:
        """Source prefix with * should use LIKE."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            source="workflow:*",
        )

        assert "LIKE" in sql
        assert "workflow:%" in params

    def test_fts_search_basic(self) -> None:
        """Basic FTS search should produce valid SQL."""
        sql, params = build_fts_search_query(
            query_text="deployment patterns",
        )

        assert "plainto_tsquery" in sql
        assert "ts_rank" in sql
        assert "deployment patterns" in params

    def test_fts_search_with_categories(self) -> None:
        """Category filter uses EXISTS subquery on junction table — no JOIN to knowledge_sources."""
        sql, params = build_fts_search_query(
            query_text="test",
            categories=["cat-uuid-1"],
        )

        assert "EXISTS" in sql
        assert "knowledge_memory_categories" in sql
        assert "knowledge_sources" not in sql

    def test_vector_search_with_community_ids_filter(self) -> None:
        """Vector search should support community_id restriction with uuid[] binding."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            community_ids=["550e8400-e29b-41d4-a716-446655440000"],
        )

        assert "kp.community_id = ANY(" in sql
        assert "::uuid[]" in sql
        assert ["550e8400-e29b-41d4-a716-446655440000"] in params

    def test_fts_search_with_community_ids_filter(self) -> None:
        """FTS search should support community_id restriction with uuid[] binding."""
        sql, params = build_fts_search_query(
            query_text="test",
            community_ids=["550e8400-e29b-41d4-a716-446655440000"],
        )

        assert "kp.community_id = ANY(" in sql
        assert "::uuid[]" in sql
        assert ["550e8400-e29b-41d4-a716-446655440000"] in params

    def test_vector_search_with_corridor_filter(self) -> None:
        """Vector search should support corridor restriction."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            corridor="cluster-a",
        )

        assert "kp.corridor =" in sql
        assert "cluster-a" in params

    def test_fts_search_with_corridor_filter(self) -> None:
        """FTS search should support corridor restriction."""
        sql, params = build_fts_search_query(
            query_text="test",
            corridor="cluster-a",
        )

        assert "kp.corridor =" in sql
        assert "cluster-a" in params

    def test_vector_search_category_uses_exists_subquery(self) -> None:
        """build_vector_search_query with categories uses EXISTS, not JOIN."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            categories=["550e8400-e29b-41d4-a716-446655440000"],
        )

        assert "EXISTS" in sql
        assert "knowledge_memory_categories" in sql
        assert "kpc.memory_id = kp.id" in sql
        assert "knowledge_sources" not in sql
        assert "JOIN knowledge_items ki " not in sql

    def test_fts_search_category_uses_exists_subquery(self) -> None:
        """build_fts_search_query with categories uses EXISTS, not JOIN."""
        sql, params = build_fts_search_query(
            query_text="test",
            categories=["550e8400-e29b-41d4-a716-446655440000"],
        )

        assert "EXISTS" in sql
        assert "knowledge_memory_categories" in sql
        assert "kpc.memory_id = kp.id" in sql
        assert "knowledge_sources" not in sql
        assert "JOIN knowledge_items ki " not in sql

    def test_positional_params_are_sequential(self) -> None:
        """All $N params should be sequential in the SQL."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1],
            source="test",
            categories=["cat-1"],
        )

        for i, _p in enumerate(params, start=1):
            assert f"${i}" in sql, f"Missing param ${i} in SQL"

    def test_vector_search_with_as_of_applies_temporal_predicate(self) -> None:
        """Vector query should include temporal validity predicate when as_of is set."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            as_of=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        )
        assert "WITH lifecycle_filtered AS" in sql
        assert "WHERE kp.lifecycle_state =" in sql
        assert "kp.valid_from IS NULL OR kp.valid_from <=" in sql
        assert "kp.valid_to IS NULL OR kp.valid_to >=" in sql
        assert datetime(2026, 4, 7, 12, 0, tzinfo=UTC) in params

    def test_fts_search_with_as_of_applies_temporal_predicate(self) -> None:
        """FTS query should include temporal validity predicate when as_of is set."""
        sql, params = build_fts_search_query(
            query_text="temporal test",
            as_of=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        )
        assert "WITH lifecycle_filtered AS" in sql
        assert "WHERE kp.lifecycle_state =" in sql
        assert "kp.valid_from IS NULL OR kp.valid_from <=" in sql
        assert "kp.valid_to IS NULL OR kp.valid_to >=" in sql
        assert datetime(2026, 4, 7, 12, 0, tzinfo=UTC) in params

    def test_vector_search_with_interval_applies_overlap_predicate(self) -> None:
        """Vector query should support temporal interval overlap filtering."""
        range_from = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
        range_to = datetime(2026, 4, 30, 23, 59, tzinfo=UTC)
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            from_dt=range_from,
            to_dt=range_to,
        )

        assert "(kp.valid_to IS NULL OR kp.valid_to >=" in sql
        assert "(kp.valid_from IS NULL OR kp.valid_from <=" in sql
        assert range_from in params
        assert range_to in params

    def test_fts_search_with_interval_applies_overlap_predicate(self) -> None:
        """FTS query should support temporal interval overlap filtering."""
        range_from = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
        range_to = datetime(2026, 4, 30, 23, 59, tzinfo=UTC)
        sql, params = build_fts_search_query(
            query_text="temporal test",
            from_dt=range_from,
            to_dt=range_to,
        )

        assert "(kp.valid_to IS NULL OR kp.valid_to >=" in sql
        assert "(kp.valid_from IS NULL OR kp.valid_from <=" in sql
        assert range_from in params
        assert range_to in params

    def test_search_queries_without_as_of_do_not_include_temporal_predicate(self) -> None:
        """Temporal predicate should be omitted when as_of is not provided."""
        vector_sql, _ = build_vector_search_query(query_embedding=[0.1, 0.2])
        fts_sql, _ = build_fts_search_query(query_text="no temporal")
        assert "kp.valid_from" not in vector_sql
        assert "kp.valid_to" not in vector_sql
        assert "kp.valid_from" not in fts_sql
        assert "kp.valid_to" not in fts_sql


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""

    def _make_row(self, id_: str, content: str, **kwargs: Any) -> dict[str, Any]:
        return {"id": id_, "content": content, **kwargs}

    def test_empty_results(self) -> None:
        """Empty inputs should return empty output."""
        result = rrf_fusion([], [])
        assert result == []

    def test_vector_only(self) -> None:
        """Vector-only results should produce ordered output."""
        vector = [
            self._make_row("a", "fact a"),
            self._make_row("b", "fact b"),
        ]
        result = rrf_fusion(vector, [])

        assert len(result) == 2
        assert result[0]["id"] == "a"  # Higher rank
        assert all("rrf_score" in r for r in result)

    def test_fts_only(self) -> None:
        """FTS-only results should produce ordered output."""
        fts = [
            self._make_row("x", "fact x"),
            self._make_row("y", "fact y"),
        ]
        result = rrf_fusion([], fts)

        assert len(result) == 2

    def test_overlapping_results_boost(self) -> None:
        """Items appearing in both lists should get boosted scores."""
        vector = [self._make_row("a", "fact a"), self._make_row("b", "fact b")]
        fts = [self._make_row("a", "fact a"), self._make_row("c", "fact c")]

        result = rrf_fusion(vector, fts)

        # 'a' appears in both, should have highest score
        assert result[0]["id"] == "a"
        assert len(result) == 3  # a, b, c

    def test_limit_applied(self) -> None:
        """Limit should cap the number of results."""
        vector = [self._make_row(f"v{i}", f"fact {i}") for i in range(20)]
        result = rrf_fusion(vector, [], limit=5)

        assert len(result) == 5

    def test_rrf_score_values(self) -> None:
        """RRF scores should be positive floats."""
        vector = [self._make_row("a", "fact")]
        result = rrf_fusion(vector, [])

        assert result[0]["rrf_score"] > 0
        assert isinstance(result[0]["rrf_score"], float)


class TestContextAssembly:
    """Tests for context assembly with token budgeting."""

    def _make_memory(self, content: str, **kwargs: Any) -> dict[str, Any]:
        return {"content": content, **kwargs}

    def test_empty_propositions(self) -> None:
        """Empty input should return empty context."""
        text, count, tokens = assemble_context([])
        assert text == ""
        assert count == 0
        assert tokens == 0

    def test_basic_assembly(self) -> None:
        """Propositions should be joined as markdown bullets."""
        props = [
            self._make_memory("fact one"),
            self._make_memory("fact two"),
        ]
        text, count, tokens = assemble_context(props)

        assert count == 2
        assert "- fact one" in text
        assert "- fact two" in text
        assert tokens > 0

    def test_token_budget_enforcement(self) -> None:
        """Assembly should stop when token budget is reached."""
        # Each line "- short" ≈ 2 tokens (7 chars / 4)
        # Generate enough props to exceed a small budget
        props = [self._make_memory(f"fact number {i}") for i in range(100)]

        text, count, tokens = assemble_context(props, max_tokens=10)

        assert count < 100
        assert tokens <= 10

    def test_skips_empty_content(self) -> None:
        """Properties with empty content should be skipped."""
        props = [
            self._make_memory("real fact"),
            self._make_memory(""),
            self._make_memory("  "),
            self._make_memory("another fact"),
        ]
        text, count, tokens = assemble_context(props)

        assert count == 2
        assert "real fact" in text
        assert "another fact" in text

    def test_no_metadata_in_output(self) -> None:
        """Output should contain only content, never metadata."""
        props = [
            self._make_memory(
                "clean fact",
                id="uuid-123",
                confidence=0.9,
                authority="EXTRACTED",
                source_section="section1",
            ),
        ]
        text, count, tokens = assemble_context(props)

        assert "clean fact" in text
        assert "uuid-123" not in text
        assert "0.9" not in text
        assert "EXTRACTED" not in text
        assert "section1" not in text

    def test_estimate_tokens(self) -> None:
        """Token estimation should be reasonable."""
        assert estimate_tokens("") == 1  # min 1
        assert estimate_tokens("hello world here") == 4  # 16 chars / 4

    def test_cosine_similarity_identical(self) -> None:
        """Identical vectors should have similarity 1.0."""
        sim = _cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        sim = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_cosine_similarity_empty(self) -> None:
        """Empty vectors should return 0.0."""
        sim = _cosine_similarity([], [])
        assert sim == 0.0


class TestMMRRerank:
    """Tests for _mmr_rerank and the diversity path in assemble_context.

    Verifies TASK-340: MMR must actually reorder results when embeddings and
    query_embedding are provided — it was silently falling back before.
    """

    def _make_prop(self, content: str, embedding: list[float]) -> dict[str, Any]:
        return {"content": content, "embedding": embedding}

    def test_mmr_fallback_when_no_embeddings(self) -> None:
        """Should return original order when propositions lack embeddings."""
        props = [{"content": "a"}, {"content": "b"}]
        result = _mmr_rerank(props, query_embedding=[1.0, 0.0])
        assert result == props

    def test_mmr_fallback_when_query_embedding_none(self) -> None:
        """Should return original order when query_embedding is None."""
        props = [self._make_prop("a", [1.0, 0.0]), self._make_prop("b", [0.0, 1.0])]
        result = _mmr_rerank(props, query_embedding=None)
        assert result == props

    def test_mmr_reorders_for_diversity(self) -> None:
        """MMR must reorder results to maximise diversity.

        Setup: query points along [1, 0].
        - p1 is highly relevant (embedding == query), selected first.
        - p2 is identical to p1 → after p1 selected, max_sim=1.0 → heavy penalty.
        - p3 is orthogonal to p1 (max_sim=0) and irrelevant to query.

        With mmr_lambda=0.3 (diversity-biased):
            MMR(p2) = 0.3*1.0 - 0.7*1.0 = -0.4
            MMR(p3) = 0.3*0.0 - 0.7*0.0 =  0.0   ← wins
        """
        query = [1.0, 0.0]
        p1 = self._make_prop("relevant A", [1.0, 0.0])  # most relevant, selected first
        p2 = self._make_prop("relevant B", [1.0, 0.0])  # identical to p1 → heavy penalty
        p3 = self._make_prop("diverse C", [0.0, 1.0])  # orthogonal — no similarity penalty

        result = _mmr_rerank([p1, p2, p3], query_embedding=query, mmr_lambda=0.3)

        assert result[0]["content"] == "relevant A"
        # p3 should be ranked above p2 because it is diverse
        contents = [r["content"] for r in result]
        assert contents.index("diverse C") < contents.index("relevant B")

    def test_assemble_context_diversity_true_requires_embeddings(self) -> None:
        """assemble_context with diversity=True but no embeddings falls back gracefully."""
        props = [{"content": "fact A"}, {"content": "fact B"}]
        # No embeddings, no query_embedding — should still assemble without error
        text, count, tokens = assemble_context(props, diversity=True, query_embedding=None)
        assert count == 2
        assert "fact A" in text

    def test_assemble_context_diversity_reorders_results(self) -> None:
        """assemble_context with diversity=True and embeddings must change ordering.

        Uses mmr_lambda=0.3 so the diversity penalty dominates (see test_mmr_reorders_for_diversity
        for the MMR score derivation).
        """
        query = [1.0, 0.0]
        # p1 and p2 are near-duplicates; p3 is orthogonal (diverse)
        props = [
            {"content": "near-dup A", "embedding": [1.0, 0.0]},
            {"content": "near-dup B", "embedding": [1.0, 0.0]},
            {"content": "diverse C", "embedding": [0.0, 1.0]},
        ]
        text_diverse, count, _ = assemble_context(
            props, diversity=True, query_embedding=query, mmr_lambda=0.3
        )
        text_plain, _, _ = assemble_context(props, diversity=False)

        # Diverse ordering should differ from plain ordering
        assert text_diverse != text_plain
        # "diverse C" should appear before "near-dup B" in the diverse output
        assert text_diverse.index("diverse C") < text_diverse.index("near-dup B")

    def test_build_vector_search_includes_embedding_column(self) -> None:
        """build_vector_search_query with include_embeddings=True must SELECT kp.embedding."""
        sql, _ = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            include_embeddings=True,
        )
        assert ", kp.embedding" in sql

    def test_build_vector_search_excludes_embedding_column_by_default(self) -> None:
        """build_vector_search_query must NOT explicitly SELECT kp.embedding by default."""
        sql, _ = build_vector_search_query(query_embedding=[0.1, 0.2])
        # The explicit ", kp.embedding" column alias must not appear
        assert ", kp.embedding" not in sql


class TestTemporalCorrectness:
    """Tests for temporal validity semantics (before/within/after/null bounds)."""

    def test_temporal_before_within_after_and_null_bounds(self) -> None:
        """RFC predicate should exclude before/after window and include null-bound rows."""
        as_of = "2026-04-07T12:00:00Z"

        assert (
            _temporal_truth_matches(
                as_of=as_of,
                valid_from="2026-04-08T00:00:00Z",
                valid_to=None,
            )
            is False
        )
        assert (
            _temporal_truth_matches(
                as_of=as_of,
                valid_from="2026-04-01T00:00:00Z",
                valid_to="2026-04-30T23:59:59Z",
            )
            is True
        )
        assert (
            _temporal_truth_matches(
                as_of=as_of,
                valid_from=None,
                valid_to="2026-04-01T00:00:00Z",
            )
            is False
        )
        assert (
            _temporal_truth_matches(
                as_of=as_of,
                valid_from=None,
                valid_to=None,
            )
            is True
        )


# ============================================================================
# Executor Registration Tests
# ============================================================================


class TestExecutorRegistration:
    """Tests for MemoryExecutor registration."""

    def test_memory_executor_not_in_default_registry(self) -> None:
        """MemoryExecutor should NOT be in the default registry (conditional loading)."""
        from workflows_mcp.engine.executor_base import create_default_registry

        registry = create_default_registry()
        assert not registry.has("Memory")

    def test_memory_executor_can_be_registered(self) -> None:
        """MemoryExecutor can be registered manually (as done at server startup)."""
        from workflows_mcp.engine.executor_base import ExecutorRegistry

        registry = ExecutorRegistry()
        registry.register(MemoryExecutor())
        executor = registry.get("Memory")
        assert isinstance(executor, MemoryExecutor)

    def test_executor_type_name(self) -> None:
        """Type name should be 'Memory'."""
        executor = MemoryExecutor()
        assert executor.type_name == "Memory"


# ============================================================================
# MemoryInput Validation Tests
# ============================================================================


class TestMemoryInputValidation:
    """Tests for MemoryInput Pydantic model validation."""

    def test_requires_operation(self) -> None:
        """Missing operation field raises validation error."""
        with pytest.raises(ValidationError):
            MemoryInput()  # type: ignore[call-arg]

    def test_query_operation_valid(self) -> None:
        """operation=query with query envelope is valid."""
        inp = MemoryInput(operation="query", query={"text": "test query"})
        assert inp.operation == "query"
        assert inp.query == {"text": "test query"}

    def test_ingest_operation_valid(self) -> None:
        """operation=ingest with record envelope is valid."""
        inp = MemoryInput(operation="ingest", record={"format": "raw", "content": "some fact"})
        assert inp.operation == "ingest"
        assert inp.record == {"format": "raw", "content": "some fact"}

    def test_ingest_structured_record_envelope_valid(self) -> None:
        """operation=ingest accepts structured content via record envelope."""
        inp = MemoryInput(
            operation="ingest",
            record={
                "format": "structured",
                "memories": [{"content": "Alice works at Acme."}],
                "entities": [
                    {
                        "name": "Alice",
                        "entity_type": "PERSON",
                        "memory_indices": [0],
                    }
                ],
                "relations": [],
            },
        )
        assert inp.operation == "ingest"
        assert inp.record is not None
        assert inp.record["format"] == "structured"
        assert inp.record["memories"][0]["content"] == "Alice works at Acme."

    def test_db_defaults_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MEMORY_DB_HOST and MEMORY_DB_NAME env vars set defaults."""
        monkeypatch.setenv("MEMORY_DB_HOST", "db.example.com")
        monkeypatch.setenv("MEMORY_DB_NAME", "my_memory")

        inp = MemoryInput(operation="query", query={"text": "test"})
        assert inp.host == "db.example.com"
        assert inp.database == "my_memory"

    def test_defaults_without_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without env vars, defaults to localhost/memory_db."""
        monkeypatch.delenv("MEMORY_DB_HOST", raising=False)
        monkeypatch.delenv("MEMORY_DB_NAME", raising=False)

        inp = MemoryInput(operation="query", query={"text": "test"})
        assert inp.host == "localhost"
        assert inp.database == "memory_db"


# ============================================================================
# MemoryOutput Tests
# ============================================================================


class TestMemoryOutput:
    """Tests for MemoryOutput Pydantic model."""

    def test_defaults(self) -> None:
        """All default values are correct."""
        out = MemoryOutput()
        assert out.success is False
        assert out.error is None
        assert out.operation == ""
        assert out.result == {}

    def test_success_flag(self) -> None:
        """success=True exposes payload through result envelope."""
        out = MemoryOutput(
            success=True,
            operation="ingest",
            result={"manage": {"stored_count": 1, "memory_ids": ["uuid-1"]}},
        )
        assert out.success is True
        assert out.operation == "ingest"
        assert out.result["manage"]["stored_count"] == 1
        assert out.result["manage"]["memory_ids"] == ["uuid-1"]


class TestSchemaBootstrapReset:
    """Tests for fail-fast schema compatibility behavior in ensure_schema."""

    @pytest.mark.asyncio
    async def test_incompatible_schema_fails_fast_with_migration_guidance(self) -> None:
        """Any existing incompatible schema must fail fast with migration guidance."""
        backend = MagicMock()
        backend.execute_script = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[]),
                MagicMock(rows=[{"value": "1"}]),
                MagicMock(rows=[]),
                MagicMock(rows=[{"has_tables": True}]),
                MagicMock(rows=[]),
            ]
        )

        with pytest.raises(
            RuntimeError, match="Incompatible knowledge schema detected"
        ) as exc_info:
            await ensure_schema(backend)

        error_message = str(exc_info.value)
        assert "migration" in error_message.lower()
        assert "memory_schema_reset_mode" not in error_message.lower()
        executed = [call.args[0] for call in backend.execute_script.await_args_list]
        assert not any("DROP TABLE" in sql for sql in executed)
        assert not any("DELETE FROM _knowledge_meta" in sql for sql in executed)

    @pytest.mark.asyncio
    async def test_new_db_applies_v1_and_stamps_version_and_epoch(self) -> None:
        """Fresh databases should apply v1 migration and stamp both meta keys."""
        backend = MagicMock()
        backend.execute_script = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[]),
                MagicMock(rows=[{"value": "0"}]),
                MagicMock(rows=[]),
                MagicMock(rows=[{"has_tables": False}]),
                MagicMock(rows=[]),
            ]
        )

        await ensure_schema(backend)

        executed = [call.args[0] for call in backend.execute_script.await_args_list]
        assert any("CREATE TABLE IF NOT EXISTS knowledge_sources" in sql for sql in executed)
        assert any(f"VALUES ('schema_version', '{SCHEMA_VERSION}')" in sql for sql in executed)
        assert any(f"VALUES ('schema_epoch', '{SCHEMA_EPOCH}')" in sql for sql in executed)


class TestRoomScopedSearch:
    """Tests for room_scoped_search routing logic."""

    @pytest.mark.asyncio
    async def test_no_room_runs_global_only(self) -> None:
        """When namespace and room are both None, only the global lane runs (2 backend calls)."""
        from workflows_mcp.engine.knowledge.search import room_scoped_search

        backend = MagicMock()
        empty_result = MagicMock()
        empty_result.rows = []
        backend.query = AsyncMock(return_value=empty_result)

        await room_scoped_search(
            query_embedding=[0.1, 0.2, 0.3],
            query_text="test query",
            backend=backend,
            namespace=None,
            room=None,
        )

        # Only global lane: 2 queries (vector + FTS)
        assert backend.query.call_count == 2

    @pytest.mark.asyncio
    async def test_with_room_runs_four_queries(self) -> None:
        """When namespace+room provided, room and global lanes each run vector+FTS (4 total)."""
        from workflows_mcp.engine.knowledge.search import room_scoped_search

        backend = MagicMock()
        empty_result = MagicMock()
        empty_result.rows = []
        backend.query = AsyncMock(return_value=empty_result)

        await room_scoped_search(
            query_embedding=[0.1, 0.2, 0.3],
            query_text="test query",
            backend=backend,
            namespace="engineering",
            room="auth",
        )

        # Room lane (2) + global companion lane (2) = 4 total
        assert backend.query.call_count == 4

    @pytest.mark.asyncio
    async def test_with_scope_and_strict_mode_runs_room_lane_only(self) -> None:
        """Strict mode should disable companion global lane and keep scoped hybrid search."""
        from workflows_mcp.engine.knowledge.search import room_scoped_search

        backend = MagicMock()
        empty_result = MagicMock()
        empty_result.rows = []
        backend.query = AsyncMock(return_value=empty_result)

        await room_scoped_search(
            query_embedding=[0.1, 0.2, 0.3],
            query_text="test query",
            backend=backend,
            namespace="engineering",
            room="auth",
            corridor="cluster-a",
            include_global_companion=False,
        )

        assert backend.query.call_count == 2
        called_sql = [call.args[0] for call in backend.query.await_args_list]
        assert all("kp.namespace =" in sql for sql in called_sql)
        assert all("kp.room =" in sql for sql in called_sql)
        assert all("kp.corridor =" in sql for sql in called_sql)

    @pytest.mark.asyncio
    async def test_room_lane_includes_room_filter_in_sql(self) -> None:
        """The room-scoped lane must pass namespace and room as WHERE conditions."""
        from workflows_mcp.engine.knowledge.search import build_vector_search_query

        sql, params = build_vector_search_query(
            [0.1, 0.2],
            namespace="engineering",
            room="auth",
        )
        assert "kp.namespace" in sql
        assert "kp.room" in sql
        assert "engineering" in params
        assert "auth" in params

    @pytest.mark.asyncio
    async def test_global_lane_no_room_filter(self) -> None:
        """The global companion lane must NOT include namespace/room WHERE conditions."""
        from workflows_mcp.engine.knowledge.search import build_vector_search_query

        sql, params = build_vector_search_query([0.1, 0.2])
        # kp.namespace appears in the SELECT list but must not appear in a WHERE predicate
        assert "kp.namespace =" not in sql
        assert "kp.room =" not in sql

    @pytest.mark.asyncio
    async def test_room_scoped_returns_fused_results(self) -> None:
        """Results from both lanes are fused and deduplicated by RRF."""
        from workflows_mcp.engine.knowledge.search import room_scoped_search

        row_a = {
            "id": "aaaa-0001",
            "content": "from room",
            "confidence": 0.9,
            "authority": "AGENT",
            "retrieval_count": 0,
            "similarity": 0.9,
            "item_path": None,
        }
        row_b = {
            "id": "bbbb-0002",
            "content": "from global",
            "confidence": 0.8,
            "authority": "AGENT",
            "retrieval_count": 0,
            "similarity": 0.8,
            "item_path": None,
        }
        row_c = {
            "id": "aaaa-0001",
            "content": "from room",
            "confidence": 0.9,
            "authority": "AGENT",
            "retrieval_count": 0,
            "fts_rank": 0.5,
            "item_path": None,
        }

        room_vec_result = MagicMock(rows=[row_a])
        room_fts_result = MagicMock(rows=[row_c])
        global_vec_result = MagicMock(rows=[row_b])
        global_fts_result = MagicMock(rows=[])

        backend = MagicMock()
        backend.query = AsyncMock(
            side_effect=[
                room_vec_result,
                room_fts_result,
                global_vec_result,
                global_fts_result,
            ]
        )

        results = await room_scoped_search(
            query_embedding=[0.1, 0.2, 0.3],
            query_text="test",
            backend=backend,
            namespace="engineering",
            room="auth",
            limit=10,
        )

        # Both ids should appear in results; row_a appears in both room lanes so scores higher
        result_ids = [r["id"] for r in results]
        assert "aaaa-0001" in result_ids
        assert "bbbb-0002" in result_ids
        # row_a (room hit) should rank above row_b (global-only)
        assert result_ids.index("aaaa-0001") < result_ids.index("bbbb-0002")


class TestMemoryService:
    """Tests that MemoryService orchestrates query and manage paths."""

    def test_memory_service_importable(self) -> None:
        """MemoryService must be importable from memory_service module."""
        from workflows_mcp.engine.memory_service import (
            ManageMemoryRequest,
            MemoryService,
            QueryMemoryRequest,
        )

        assert MemoryService is not None
        assert QueryMemoryRequest is not None
        assert ManageMemoryRequest is not None

    @pytest.mark.asyncio
    async def test_memory_service_handles_query_path(self) -> None:
        """MemoryService.query returns structured facts/memories without summary."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from workflows_mcp.engine.memory_service import MemoryService, QueryMemoryRequest

        backend = MagicMock()
        backend.query = AsyncMock(
            return_value=MagicMock(
                rows=[
                    {
                        "id": "abc",
                        "content": "test fact",
                        "confidence": 0.9,
                        "authority": "AGENT",
                        "rrf_score": 0.5,
                        "item_path": None,
                    }
                ]
            )
        )
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None

        with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
            mock_embed.return_value = ([0.1, 0.2, 0.3], "text-embedding-3-small", 3, None)
            with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
                mock_search.return_value = [
                    {
                        "id": "abc",
                        "content": "test fact",
                        "confidence": 0.9,
                        "authority": "AGENT",
                        "rrf_score": 0.5,
                        "item_path": None,
                    }
                ]
                service = MemoryService(backend=backend, context=context)
                result = await service.query(QueryMemoryRequest(query="test query"))

        assert hasattr(result, "memories")
        assert hasattr(result, "facts")
        assert len(result.memories) >= 0

    @pytest.mark.asyncio
    async def test_memory_service_handles_manage_path(self) -> None:
        """MemoryService.manage returns a result with operation field."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from workflows_mcp.engine.memory_service import ManageMemoryRequest, MemoryService

        backend = MagicMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None

        with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
            mock_embed.return_value = ([0.1, 0.2, 0.3], "text-embedding-3-small", 3, None)
            service = MemoryService(backend=backend, context=context)
            result = await service.manage(
                ManageMemoryRequest(
                    operation="store",
                    content="A fact to store",
                )
            )

        assert result.operation == "store"
        assert hasattr(result, "memory_ids")

    @pytest.mark.asyncio
    async def test_execute_ingest_raw_forwards_validity_window_to_store(self) -> None:
        """Unified ingest must forward valid_from/valid_to from record to store request."""
        from workflows_mcp.engine.memory_service import (
            ManageMemoryResult,
            MemoryRequest,
            MemoryService,
        )

        backend = MagicMock()
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)
        service.manage = AsyncMock(  # type: ignore[method-assign]
            return_value=ManageMemoryResult(operation="store", memory_ids=["m-1"], stored_count=1)
        )

        await service.execute(
            MemoryRequest(
                operation="ingest",
                scope={
                    "palace": "acme",
                    "wing": "engineering",
                    "room": "memory",
                    "compartment": "temporal",
                },
                record={
                    "format": "raw",
                    "content": "temporal fact",
                    "valid_from": "2026-04-01T00:00:00Z",
                    "valid_to": "2026-04-30T23:59:59Z",
                },
            )
        )

        called_request = service.manage.await_args.args[0]  # type: ignore[attr-defined]
        assert called_request.operation == "store"
        assert called_request.valid_from == "2026-04-01T00:00:00Z"
        assert called_request.valid_to == "2026-04-30T23:59:59Z"

    @pytest.mark.asyncio
    async def test_query_with_as_of_passes_temporal_filter_to_search_layer(self) -> None:
        """Query as_of should be converted and forwarded to search SQL builder layer."""
        from workflows_mcp.engine.memory_service import MemoryService, QueryMemoryRequest

        backend = MagicMock()
        backend.execute = AsyncMock()
        context = MagicMock()
        context.execution_context = None

        with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
            mock_embed.return_value = ([0.1, 0.2, 0.3], "text-embedding-3-small", 3, None)
            with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
                mock_search.return_value = []
                service = MemoryService(backend=backend, context=context)
                await service.query(
                    QueryMemoryRequest(
                        query="temporal query",
                        strategy="auto",
                        as_of="2026-04-07T12:00:00Z",
                    )
                )

        assert mock_search.await_count == 1
        assert mock_search.await_args.kwargs["as_of"] == datetime(2026, 4, 7, 12, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_query_with_interval_passes_temporal_range_to_search_layer(self) -> None:
        """Query from/to should be converted and forwarded to search SQL builder layer."""
        from workflows_mcp.engine.memory_service import MemoryService, QueryMemoryRequest

        backend = MagicMock()
        backend.execute = AsyncMock()
        context = MagicMock()
        context.execution_context = None

        with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
            mock_embed.return_value = ([0.1, 0.2, 0.3], "text-embedding-3-small", 3, None)
            with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
                mock_search.return_value = []
                service = MemoryService(backend=backend, context=context)
                await service.query(
                    QueryMemoryRequest(
                        query="temporal query",
                        strategy="auto",
                        from_="2026-04-01T00:00:00Z",
                        to="2026-04-30T23:59:59Z",
                    )
                )

        assert mock_search.await_count == 1
        assert mock_search.await_args.kwargs["from_dt"] == datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
        assert mock_search.await_args.kwargs["to_dt"] == datetime(
            2026, 4, 30, 23, 59, 59, tzinfo=UTC
        )

    @pytest.mark.asyncio
    async def test_manage_context_preserves_filter_fields(self) -> None:
        """Context management must forward all scoping/filter fields to QueryMemoryRequest."""
        from workflows_mcp.engine.memory_service import (
            ManageMemoryRequest,
            MemoryService,
            QueryMemoryResult,
        )

        backend = MagicMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None

        service = MemoryService(backend=backend, context=context)
        service._query_context = AsyncMock(  # type: ignore[method-assign]
            return_value=QueryMemoryResult(
                evidence=[
                    {
                        "context_text": "ctx",
                        "memory_count": 1,
                        "tokens_used": 10,
                    }
                ]
            )
        )

        await service.manage(
            ManageMemoryRequest(
                operation="context",
                query="investigate auth bug",
                namespace="engineering",
                room="auth",
                corridor="critical-path",
                source="runbook:*",
                categories=["incident", "auth"],
                as_of="2026-04-19T12:30:00Z",
                min_confidence=0.77,
                lifecycle_state="ACTIVE",
                max_items=7,
                max_tokens=512,
                embedding_profile="embedding",
            )
        )

        called_request = service._query_context.await_args.args[0]  # type: ignore[attr-defined]
        assert called_request.query == "investigate auth bug"
        assert called_request.strategy == "context"
        assert called_request.namespace == "engineering"
        assert called_request.room == "auth"
        assert called_request.scope == {"corridor": "critical-path"}
        assert called_request.source == "runbook:*"
        assert called_request.categories == ["incident", "auth"]
        assert called_request.as_of == "2026-04-19T12:30:00Z"
        assert called_request.min_confidence == 0.77
        assert called_request.lifecycle_state == "ACTIVE"
        assert called_request.max_items == 7
        assert called_request.max_tokens == 512

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", ["auto", "communities", "palace"])
    async def test_query_retrieval_updates_last_retrieved_timestamp(self, strategy: str) -> None:
        """Retrieval update SQL must increment count and stamp last_retrieved_at."""
        from workflows_mcp.engine.memory_service import MemoryService, QueryMemoryRequest

        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))

        context = MagicMock()
        context.execution_context = None

        with patch("workflows_mcp.engine.memory_service.compute_embedding") as mock_embed:
            mock_embed.return_value = ([0.1, 0.2, 0.3], "text-embedding-3-small", 3, None)
            with patch("workflows_mcp.engine.memory_service.room_scoped_search") as mock_search:
                mock_search.return_value = [
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "content": "match",
                        "confidence": 0.9,
                        "authority": "AGENT",
                        "rrf_score": 0.8,
                        "item_path": None,
                        "source_name": "docs",
                        "namespace": "engineering",
                    }
                ]

                if strategy == "communities":
                    backend.query = AsyncMock(
                        return_value=MagicMock(
                            rows=[
                                {
                                    "id": "22222222-2222-2222-2222-222222222222",
                                    "content": "community",
                                    "member_count": 1,
                                    "memory_count": 1,
                                    "namespace": "engineering",
                                    "room": "auth",
                                    "corridor": None,
                                    "similarity": 0.9,
                                }
                            ]
                        )
                    )

                service = MemoryService(backend=backend, context=context)
                await service.query(
                    QueryMemoryRequest(
                        query="auth incident",
                        strategy=strategy,
                        namespace="engineering",
                        room="auth",
                    )
                )

        assert backend.execute.await_count >= 1
        update_sql = backend.execute.await_args_list[-1].args[0]
        assert "retrieval_count = retrieval_count + 1" in update_sql
        assert "last_retrieved_at = NOW()" in update_sql


class TestMemoryExecutorManageWiring:
    """Tests that MemoryExecutor forwards manage payloads correctly."""

    @pytest.mark.asyncio
    async def test_memory_executor_forwards_ingest_structured_request(self) -> None:
        """MemoryExecutor passes unified ingest envelope to MemoryService.execute."""
        backend = MagicMock()
        backend.connect = AsyncMock()
        backend.disconnect = AsyncMock()

        captured_requests: list[Any] = []

        async def capture_execute(request: Any) -> Any:
            captured_requests.append(request)
            from workflows_mcp.engine.memory_service import ManageMemoryResult, MemoryResult

            return MemoryResult(
                operation="ingest",
                manage=ManageMemoryResult(
                    operation="ingest_structured",
                    success=True,
                    stored_count=1,
                    memory_ids=["memory-0-id"],
                    entity_ids=["entity-alice-id"],
                    relation_ids=["relation-0-id"],
                    entities_stored_count=1,
                    relations_stored_count=1,
                ),
            )

        context = MagicMock()
        context.execution_context = None

        with (
            patch.object(MemoryExecutor, "_create_backend", return_value=backend),
            patch("workflows_mcp.engine.executors_memory.MemoryService") as mock_service_cls,
        ):
            mock_service_cls.return_value.execute = capture_execute

            executor = MemoryExecutor()
            result = await executor.execute(
                MemoryInput(
                    operation="ingest",
                    scope={"wing": "engineering", "room": "memory", "compartment": "ingest"},
                    record={
                        "format": "structured",
                        "source": "spec-tests",
                        "path": "docs/spec.md",
                        "memories": [{"content": "Alice works at Acme."}],
                        "entities": [
                            {
                                "name": "Alice",
                                "entity_type": "PERSON",
                                "memory_indices": [0],
                            }
                        ],
                        "relations": [],
                    },
                ),
                context,
            )

        assert result.success is True
        assert result.operation == "ingest"
        assert result.result["manage"]["stored_count"] == 1
        assert result.result["manage"]["entity_ids"] == ["entity-alice-id"]
        assert result.result["manage"]["relation_ids"] == ["relation-0-id"]
        assert result.result["manage"]["entities_stored_count"] == 1
        assert result.result["manage"]["relations_stored_count"] == 1
        assert len(captured_requests) == 1
        assert captured_requests[0].operation == "ingest"
        assert captured_requests[0].scope.wing == "engineering"
        assert captured_requests[0].scope.room == "memory"
        assert captured_requests[0].scope.compartment == "ingest"
        assert captured_requests[0].record is not None
        assert captured_requests[0].record.format == "structured"
        assert captured_requests[0].record.memories is not None
        assert captured_requests[0].record.memories[0]["content"] == "Alice works at Acme."
        assert captured_requests[0].record.entities is not None
        assert captured_requests[0].record.entities[0]["memory_indices"] == [0]
        assert captured_requests[0].record.relations == []
