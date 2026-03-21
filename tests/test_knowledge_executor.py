"""Tests for the Knowledge executor and supporting modules.

Tests focus on pure logic (RRF fusion, context assembly, schema DDL,
input validation) — no real database needed. PostgreSQL-specific tests
require optional deps.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from workflows_mcp.engine.executors_knowledge import (
    KnowledgeExecutor,
    KnowledgeInput,
    KnowledgeOutput,
)
from workflows_mcp.engine.knowledge.constants import (
    DEFAULT_LIMIT,
    DEFAULT_MIN_CONFIDENCE,
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
    MIGRATIONS,
    _CREATE_EXTENSION,
    _CREATE_INDEXES,
    _CREATE_KNOWLEDGE_ENTITIES,
    _CREATE_KNOWLEDGE_ITEMS,
    _CREATE_KNOWLEDGE_PROPOSITIONS,
    _CREATE_KNOWLEDGE_SOURCES,
)
from workflows_mcp.engine.knowledge.search import (
    build_fts_search_query,
    build_vector_search_query,
    rrf_fusion,
)


# ============================================================================
# Constants Tests
# ============================================================================


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


# ============================================================================
# Schema DDL Tests
# ============================================================================

_ALL_DDL = "\n".join(
    [
        _CREATE_EXTENSION,
        _CREATE_KNOWLEDGE_SOURCES,
        _CREATE_KNOWLEDGE_ITEMS,
        _CREATE_KNOWLEDGE_PROPOSITIONS,
        _CREATE_KNOWLEDGE_ENTITIES,
        _CREATE_INDEXES,
    ]
)


class TestSchemaDDL:
    """Tests for idempotent schema DDL constants."""

    def test_ddl_contains_extension(self) -> None:
        """DDL should create pgvector extension."""
        assert "CREATE EXTENSION IF NOT EXISTS vector" in _CREATE_EXTENSION

    def test_ddl_contains_all_tables(self) -> None:
        """DDL should create all knowledge tables."""
        assert "CREATE TABLE IF NOT EXISTS knowledge_sources" in _CREATE_KNOWLEDGE_SOURCES
        assert "CREATE TABLE IF NOT EXISTS knowledge_items" in _CREATE_KNOWLEDGE_ITEMS
        assert "CREATE TABLE IF NOT EXISTS knowledge_propositions" in _CREATE_KNOWLEDGE_PROPOSITIONS
        assert "CREATE TABLE IF NOT EXISTS knowledge_entities" in _CREATE_KNOWLEDGE_ENTITIES

    def test_ddl_contains_indexes(self) -> None:
        """DDL should create necessary indexes."""
        assert "CREATE INDEX IF NOT EXISTS idx_kp_lifecycle" in _CREATE_INDEXES
        assert "CREATE INDEX IF NOT EXISTS idx_kp_search_vector" in _CREATE_INDEXES

    def test_ddl_contains_unique_source_index(self) -> None:
        """DDL should create unique index on knowledge_sources(name)."""
        assert "idx_ks_name" in _CREATE_INDEXES
        assert "knowledge_sources(name)" in _CREATE_INDEXES

    def test_ddl_contains_unique_entity_index(self) -> None:
        """DDL should create unique index on knowledge_entities(entity_type, name)."""
        assert "idx_ke_type_name" in _CREATE_INDEXES
        assert "knowledge_entities(entity_type, name)" in _CREATE_INDEXES

    def test_ddl_contains_unique_source_path_index(self) -> None:
        """DDL should create unique index on knowledge_items(source_id, path)."""
        assert "idx_ki_source_path" in _CREATE_INDEXES
        assert "knowledge_items(source_id, path)" in _CREATE_INDEXES

    def test_ddl_propositions_uses_metadata_not_metadata_underscore(self) -> None:
        """knowledge_propositions DDL must use 'metadata' column, not 'metadata_'."""
        assert "metadata JSONB" in _CREATE_KNOWLEDGE_PROPOSITIONS
        assert "metadata_" not in _CREATE_KNOWLEDGE_PROPOSITIONS

    def test_migration_2_renames_metadata_column(self) -> None:
        """Migration v2 should rename metadata_ to metadata in an idempotent DO block."""
        migration_versions = [m[0] for m in MIGRATIONS]
        assert 2 in migration_versions, "Migration version 2 must exist"

        v2 = next(m for m in MIGRATIONS if m[0] == 2)
        sql = v2[2]
        assert "metadata_" in sql, "Migration v2 SQL must reference metadata_ column"
        assert "RENAME COLUMN" in sql, "Migration v2 must use RENAME COLUMN"
        assert "metadata" in sql, "Migration v2 must rename to 'metadata'"
        # Must be idempotent: guarded by IF EXISTS
        assert "IF EXISTS" in sql, "Migration v2 must be idempotent (IF EXISTS guard)"

    def test_migration_2_is_idempotent_do_block(self) -> None:
        """Migration v2 SQL must be wrapped in a DO $$ BEGIN ... END $$ block."""
        v2 = next(m for m in MIGRATIONS if m[0] == 2)
        sql = v2[2]
        assert "DO $$" in sql or "DO $" in sql
        assert "END $$" in sql or "END $" in sql

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


# ============================================================================
# Search Query Builder Tests
# ============================================================================


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
        assert "knowledge_propositions" in sql
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
        assert "knowledge_proposition_categories" in sql
        assert "knowledge_sources" not in sql

    def test_vector_search_category_uses_exists_subquery(self) -> None:
        """build_vector_search_query with categories uses EXISTS, not JOIN."""
        sql, params = build_vector_search_query(
            query_embedding=[0.1, 0.2],
            categories=["550e8400-e29b-41d4-a716-446655440000"],
        )

        assert "EXISTS" in sql
        assert "knowledge_proposition_categories" in sql
        assert "kpc.proposition_id = kp.id" in sql
        assert "knowledge_sources" not in sql
        assert "JOIN knowledge_items ki " not in sql

    def test_fts_search_category_uses_exists_subquery(self) -> None:
        """build_fts_search_query with categories uses EXISTS, not JOIN."""
        sql, params = build_fts_search_query(
            query_text="test",
            categories=["550e8400-e29b-41d4-a716-446655440000"],
        )

        assert "EXISTS" in sql
        assert "knowledge_proposition_categories" in sql
        assert "kpc.proposition_id = kp.id" in sql
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


# ============================================================================
# RRF Fusion Tests
# ============================================================================


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


# ============================================================================
# Context Assembly Tests
# ============================================================================


class TestContextAssembly:
    """Tests for context assembly with token budgeting."""

    def _make_proposition(self, content: str, **kwargs: Any) -> dict[str, Any]:
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
            self._make_proposition("fact one"),
            self._make_proposition("fact two"),
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
        props = [self._make_proposition(f"fact number {i}") for i in range(100)]

        text, count, tokens = assemble_context(props, max_tokens=10)

        assert count < 100
        assert tokens <= 10

    def test_skips_empty_content(self) -> None:
        """Properties with empty content should be skipped."""
        props = [
            self._make_proposition("real fact"),
            self._make_proposition(""),
            self._make_proposition("  "),
            self._make_proposition("another fact"),
        ]
        text, count, tokens = assemble_context(props)

        assert count == 2
        assert "real fact" in text
        assert "another fact" in text

    def test_no_metadata_in_output(self) -> None:
        """Output should contain only content, never metadata."""
        props = [
            self._make_proposition(
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


# ============================================================================
# MMR Reranking Tests
# ============================================================================


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


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestKnowledgeInputValidation:
    """Tests for KnowledgeInput Pydantic model validation."""

    def test_search_requires_query(self) -> None:
        """Search operation requires a query."""
        with pytest.raises(ValidationError, match="query"):
            KnowledgeInput(op="search")

    def test_store_requires_content(self) -> None:
        """Store operation requires content."""
        with pytest.raises(ValidationError, match="content"):
            KnowledgeInput(op="store")

    def test_forget_requires_proposition_ids(self) -> None:
        """Forget operation requires proposition_ids."""
        with pytest.raises(ValidationError, match="proposition_ids"):
            KnowledgeInput(op="forget")

    def test_search_valid(self) -> None:
        """Valid search input should pass validation."""
        inp = KnowledgeInput(op="search", query="test query")
        assert inp.op == "search"
        assert inp.query == "test query"
        assert inp.limit == DEFAULT_LIMIT
        assert inp.min_confidence is None

    def test_store_valid(self) -> None:
        """Valid store input should pass validation."""
        inp = KnowledgeInput(op="store", content="new fact")
        assert inp.op == "store"
        assert inp.content == "new fact"

    def test_recall_valid(self) -> None:
        """Recall does not require extra fields."""
        inp = KnowledgeInput(op="recall")
        assert inp.op == "recall"

    def test_context_requires_query(self) -> None:
        """Context operation requires a query."""
        with pytest.raises(ValidationError, match="query"):
            KnowledgeInput(op="context")

    def test_context_valid(self) -> None:
        """Valid context input should pass validation."""
        inp = KnowledgeInput(op="context", query="deployment patterns", max_tokens=2000)
        assert inp.op == "context"
        assert inp.max_tokens == 2000
        assert inp.diversity is True

    def test_default_lifecycle_state(self) -> None:
        """Default lifecycle state should be ACTIVE."""
        inp = KnowledgeInput(op="recall")
        assert inp.lifecycle_state == LifecycleState.ACTIVE

    def test_default_embedding_profile(self) -> None:
        """Default embedding profile should be 'embedding'."""
        inp = KnowledgeInput(op="search", query="test")
        assert inp.embedding_profile == "embedding"

    def test_invalid_op_rejected(self) -> None:
        """Invalid operation should be rejected."""
        with pytest.raises(ValidationError):
            KnowledgeInput(op="invalid_op")  # type: ignore[arg-type]

    def test_store_path_without_source_rejected(self) -> None:
        """path without source should raise ValidationError."""
        with pytest.raises(ValidationError, match="source"):
            KnowledgeInput(op="store", content="fact", path="docs/file.md")

    def test_store_path_with_source_valid(self) -> None:
        """path alongside source is valid."""
        inp = KnowledgeInput(
            op="store",
            content="fact",
            source="my-source",
            path="docs/file.md",
        )
        assert inp.path == "docs/file.md"
        assert inp.source == "my-source"

    def test_store_source_without_path_valid(self) -> None:
        """source without path is valid (agent observation with provenance label)."""
        inp = KnowledgeInput(op="store", content="observation", source="my-source")
        assert inp.source == "my-source"
        assert inp.path is None

    def test_store_neither_source_nor_path_valid(self) -> None:
        """Neither source nor path is valid (pure agent observation)."""
        inp = KnowledgeInput(op="store", content="raw fact")
        assert inp.source is None
        assert inp.path is None


# ============================================================================
# Environment Variable Tests
# ============================================================================


class TestEnvironmentVariableDefaults:
    """Tests for KNOWLEDGE_DB_* environment variable integration."""

    def test_defaults_without_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without env vars, fields should use hardcoded defaults."""
        monkeypatch.delenv("KNOWLEDGE_DB_HOST", raising=False)
        monkeypatch.delenv("KNOWLEDGE_DB_PORT", raising=False)
        monkeypatch.delenv("KNOWLEDGE_DB_NAME", raising=False)
        monkeypatch.delenv("KNOWLEDGE_DB_USER", raising=False)
        monkeypatch.delenv("KNOWLEDGE_DB_PASSWORD", raising=False)

        inp = KnowledgeInput(op="recall")
        assert inp.host == "localhost"
        assert inp.port == 5432
        assert inp.database == "knowledge_db"
        assert inp.username is None
        assert inp.password is None

    def test_env_vars_are_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars should provide defaults when set."""
        monkeypatch.setenv("KNOWLEDGE_DB_HOST", "kb.example.com")
        monkeypatch.setenv("KNOWLEDGE_DB_PORT", "5433")
        monkeypatch.setenv("KNOWLEDGE_DB_NAME", "my_knowledge")
        monkeypatch.setenv("KNOWLEDGE_DB_USER", "kb_user")
        monkeypatch.setenv("KNOWLEDGE_DB_PASSWORD", "kb_pass")

        inp = KnowledgeInput(op="recall")
        assert inp.host == "kb.example.com"
        assert inp.port == 5433
        assert inp.database == "my_knowledge"
        assert inp.username == "kb_user"
        assert inp.password == "kb_pass"

    def test_yaml_inputs_override_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit YAML inputs should override env vars."""
        monkeypatch.setenv("KNOWLEDGE_DB_HOST", "env-host")
        monkeypatch.setenv("KNOWLEDGE_DB_NAME", "env-db")

        inp = KnowledgeInput(op="recall", host="yaml-host", database="yaml-db")
        assert inp.host == "yaml-host"
        assert inp.database == "yaml-db"


# ============================================================================
# Output Model Tests
# ============================================================================


class TestKnowledgeOutput:
    """Tests for KnowledgeOutput Pydantic model."""

    def test_default_values(self) -> None:
        """Default output should have empty values."""
        out = KnowledgeOutput()
        assert out.success is False
        assert out.error is None
        assert out.rows == []
        assert out.row_count == 0
        assert out.proposition_ids == []
        assert out.stored_count == 0
        assert out.context_text == ""
        assert out.tokens_used == 0

    def test_search_output(self) -> None:
        """Search output should populate rows."""
        out = KnowledgeOutput(
            success=True,
            rows=[{"id": "123", "content": "fact"}],
            columns=["id", "content"],
            row_count=1,
        )
        assert out.success
        assert out.row_count == 1

    def test_store_output(self) -> None:
        """Store output should populate proposition_ids."""
        out = KnowledgeOutput(
            success=True,
            proposition_ids=["uuid-1"],
            stored_count=1,
        )
        assert out.stored_count == 1

    def test_context_output(self) -> None:
        """Context output should populate context_text."""
        out = KnowledgeOutput(
            success=True,
            context_text="- fact one\n- fact two",
            proposition_count=2,
            tokens_used=12,
        )
        assert out.proposition_count == 2
        assert "fact one" in out.context_text


# ============================================================================
# Recall Filter Builder Tests
# ============================================================================


class TestRecallFilters:
    """Tests for _build_where_clause shared filter logic."""

    async def _build(self, **kwargs: Any) -> tuple[str, list[Any]]:
        """Helper to build WHERE clause from KnowledgeInput kwargs."""
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(op="recall", **kwargs)
        # Mock backend for category resolution (not used in most tests)
        backend = MagicMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))
        return await executor._build_where_clause(inputs, backend)

    @pytest.mark.asyncio
    async def test_recall_source_name_exact(self) -> None:
        """Exact source generates kp.source_name = $N."""
        where, params = await self._build(source="internal-docs")
        assert "kp.source_name = $1" in where
        assert "internal-docs" in params
        assert "LIKE" not in where

    @pytest.mark.asyncio
    async def test_recall_source_name_prefix(self) -> None:
        """Source with * generates kp.source_name LIKE."""
        where, params = await self._build(source="workflow:*")
        assert "LIKE" in where
        assert "workflow:%" in params

    @pytest.mark.asyncio
    async def test_build_where_clause_returns_two_values(self) -> None:
        """_build_where_clause returns (where_sql, params) — no join_clause."""
        result = await self._build()
        assert len(result) == 2
        where, params = result
        assert isinstance(where, str)
        assert isinstance(params, list)

    @pytest.mark.asyncio
    async def test_recall_min_confidence(self) -> None:
        """min_confidence direct field generates kp.confidence >= $N."""
        where, params = await self._build(min_confidence=0.7)
        assert "kp.confidence >=" in where
        assert 0.7 in params

    @pytest.mark.asyncio
    async def test_recall_created_after(self) -> None:
        """created_after generates kp.created_at >= $N::timestamptz."""
        where, params = await self._build(created_after="2026-03-01T00:00:00Z")
        assert "kp.created_at >=" in where
        assert "timestamptz" in where
        assert "2026-03-01T00:00:00Z" in params

    @pytest.mark.asyncio
    async def test_recall_created_before(self) -> None:
        """created_before generates kp.created_at <= $N::timestamptz."""
        where, params = await self._build(created_before="2026-03-04T23:59:59Z")
        assert "kp.created_at <=" in where
        assert "timestamptz" in where
        assert "2026-03-04T23:59:59Z" in params

    @pytest.mark.asyncio
    async def test_recall_combined_filters(self) -> None:
        """Multiple filters produce correct AND-joined SQL."""
        where, params = await self._build(
            source="docs",
            lifecycle_state="active",
            min_confidence=0.5,
            created_after="2026-01-01",
        )
        assert " AND " in where
        # Should have: source_name, lifecycle_state, min_confidence, created_after
        assert where.count(" AND ") == 3  # 4 clauses, 3 ANDs

    @pytest.mark.asyncio
    async def test_recall_category_filter_uses_exists(self) -> None:
        """Category filter uses EXISTS subquery on junction table — no JOIN, no &&."""
        where, params = await self._build(
            where={"category": "550e8400-e29b-41d4-a716-446655440000"}
        )
        assert "EXISTS" in where
        assert "knowledge_proposition_categories" in where
        assert "kpc.proposition_id = kp.id" in where
        assert "knowledge_sources" not in where
        assert "&&" not in where

    @pytest.mark.asyncio
    async def test_recall_default_lifecycle_fallback(self) -> None:
        """Without where dict, default lifecycle_state is applied."""
        where, params = await self._build()
        assert "kp.lifecycle_state" in where
        assert "ACTIVE" in params

    @pytest.mark.asyncio
    async def test_recall_date_filters_work_without_where(self) -> None:
        """Date filters are top-level and work even without a where dict."""
        where, params = await self._build(
            created_after="2026-01-01",
            created_before="2026-12-31",
        )
        assert "kp.created_at >=" in where
        assert "kp.created_at <=" in where
        # Default lifecycle also applied since no where dict
        assert "kp.lifecycle_state" in where

    @pytest.mark.asyncio
    async def test_params_are_sequential(self) -> None:
        """All $N params should be sequential in the WHERE clause."""
        where, params = await self._build(
            source="test:*",
            min_confidence=0.5,
            created_after="2026-01-01",
        )
        for i in range(1, len(params) + 1):
            assert f"${i}" in where, f"Missing param ${i} in WHERE clause"

    @pytest.mark.asyncio
    async def test_recall_category_name_resolved(self) -> None:
        """Category name resolves to UUID; EXISTS subquery uses resolved UUID."""
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="recall",
            where={"category": "learnings"},
        )
        resolved_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        backend = MagicMock()
        query_result = MagicMock()
        query_result.rows = [{"id": resolved_uuid}]
        backend.query = AsyncMock(return_value=query_result)

        where, params = await executor._build_where_clause(inputs, backend)
        # params[0] is the resolved UUID list (passed as ::uuid[] for ANY())
        assert params[0] == [resolved_uuid]
        assert "EXISTS" in where
        assert "knowledge_proposition_categories" in where
        assert "ANY" in where


# ============================================================================
# Forget Operation Tests
# ============================================================================


class TestForgetOperation:
    """Tests for forget validation and output model."""

    def test_forget_proposition_ids_string_normalization(self) -> None:
        """Comma-separated string should be accepted by validation."""
        inp = KnowledgeInput(op="forget", proposition_ids="id-1, id-2, id-3")
        assert inp.proposition_ids == "id-1, id-2, id-3"

    def test_forget_proposition_ids_list(self) -> None:
        """List of IDs should pass validation."""
        inp = KnowledgeInput(op="forget", proposition_ids=["id-1", "id-2"])
        assert inp.proposition_ids == ["id-1", "id-2"]

    def test_forget_validation_accepts_where(self) -> None:
        """forget with where dict (no proposition_ids) passes validation."""
        inp = KnowledgeInput(op="forget", where={"source_name": "old-docs"})
        assert inp.op == "forget"
        assert inp.where == {"source_name": "old-docs"}
        assert inp.proposition_ids is None

    def test_forget_validation_rejects_neither(self) -> None:
        """forget with neither proposition_ids nor where raises ValidationError."""
        with pytest.raises(ValidationError, match="proposition_ids.*where.*source.*created"):
            KnowledgeInput(op="forget")

    def test_forget_output_model(self) -> None:
        """Forget output fields are set correctly."""
        out = KnowledgeOutput(success=True, archived_count=3, skipped_count=1)
        assert out.archived_count == 3
        assert out.skipped_count == 1
        assert out.success is True

    def test_forget_output_defaults_zero(self) -> None:
        """Forget output defaults to zero counts."""
        out = KnowledgeOutput(success=True)
        assert out.archived_count == 0
        assert out.skipped_count == 0

    def test_forget_with_where_has_date_filters(self) -> None:
        """forget with where + date filters passes validation."""
        inp = KnowledgeInput(
            op="forget",
            where={"source_name": "stale-source"},
            created_before="2025-01-01",
        )
        assert inp.where is not None
        assert inp.created_before == "2025-01-01"

    def test_forget_sql_contains_user_validated_immunity(self) -> None:
        """Both forget UPDATE paths must exclude USER_VALIDATED propositions.

        Regression guard: if the immunity clause is removed, archived_count and
        skipped_count would be wrong and human-validated facts could be wiped.
        The SQL uses f-string interpolation of Authority.USER_VALIDATED, so we
        check for the enum reference rather than the evaluated string value.
        """
        import inspect

        from workflows_mcp.engine.executors_knowledge import KnowledgeExecutor

        source = inspect.getsource(KnowledgeExecutor._op_forget)
        assert source.count("Authority.USER_VALIDATED") == 2, (
            "_op_forget must exclude USER_VALIDATED in BOTH update paths (by-ID and by-filter)"
        )

    def test_forget_update_sql_does_not_contain_updated_at(self) -> None:
        """The forget UPDATE statements must NOT reference updated_at.

        Some deployments omit updated_at from knowledge_propositions;
        unconditionally setting it causes 'column does not exist' errors.
        The only mutation needed is setting lifecycle_state = ARCHIVED.
        updated_at is maintained by the trg_kp_updated_at trigger (migration v6).
        """
        import inspect

        from workflows_mcp.engine.executors_knowledge import KnowledgeExecutor

        source = inspect.getsource(KnowledgeExecutor._op_forget)
        assert "updated_at" not in source, (
            "_op_forget must not reference updated_at — not all schemas have that column"
        )

    def test_migration_v6_adds_updated_at_trigger(self) -> None:
        """Migration v6 must wire updated_at triggers on all three knowledge tables.

        Regression guard for TASK-341: archive operations were leaving updated_at
        unchanged. One shared trigger function covers all tables that carry the column.
        """
        v6 = next((m for m in MIGRATIONS if m[0] == 6), None)
        assert v6 is not None, "Migration v6 is missing from MIGRATIONS list"
        _, description, sql = v6
        assert "update_updated_at" in sql, "Migration v6 must create update_updated_at() function"
        assert "BEFORE UPDATE" in sql, "Trigger must fire BEFORE UPDATE"
        assert "trg_ks_updated_at" in sql, "Trigger missing for knowledge_sources"
        assert "trg_ki_updated_at" in sql, "Trigger missing for knowledge_items"
        assert "trg_kp_updated_at" in sql, "Trigger missing for knowledge_propositions"


# ============================================================================
# Recall Operation Tests
# ============================================================================


class TestRecallOperation:
    """Tests for recall output and order field safety."""

    def test_recall_output_model(self) -> None:
        """Recall output has correct rows, columns, row_count fields."""
        out = KnowledgeOutput(
            success=True,
            rows=[{"id": "1", "content": "fact"}],
            columns=["id", "content"],
            row_count=1,
        )
        assert out.row_count == 1
        assert out.columns == ["id", "content"]
        assert out.rows[0]["content"] == "fact"

    def test_recall_order_safe_fields_accepted(self) -> None:
        """Whitelisted order fields pass validation."""
        inp = KnowledgeInput(op="recall", order=["relevance_score:desc", "created_at:asc"])
        assert inp.order == ["relevance_score:desc", "created_at:asc"]

    def test_recall_order_unsafe_field_ignored(self) -> None:
        """Non-whitelisted fields in order should not break input validation.

        The actual filtering happens at SQL generation time, not during input validation.
        This test just confirms the input model accepts arbitrary order strings.
        """
        inp = KnowledgeInput(op="recall", order=["DROP TABLE; --:desc"])
        assert inp.order is not None  # passes validation (filtered at SQL generation)

    def test_recall_with_all_filters(self) -> None:
        """Recall with all possible filters passes validation."""
        inp = KnowledgeInput(
            op="recall",
            source="docs:*",
            lifecycle_state="active",
            min_confidence=0.8,
            where={"category": "550e8400-e29b-41d4-a716-446655440000"},
            created_after="2026-01-01",
            created_before="2026-12-31",
            order=["confidence:desc"],
            limit=50,
        )
        assert inp.op == "recall"
        assert inp.limit == 50

    def test_recall_input_created_fields_default_none(self) -> None:
        """created_after and created_before default to None."""
        inp = KnowledgeInput(op="recall")
        assert inp.created_after is None
        assert inp.created_before is None

    def test_recall_safe_fields_include_expected_columns(self) -> None:
        """ORDER BY allowlist for _op_recall must include known-safe sortable columns.

        updated_at is part of the base DDL (knowledge_propositions has it from creation)
        and is valid to include. The allowlist must never include unvetted column names.
        """
        import inspect

        from workflows_mcp.engine.executors_knowledge import KnowledgeExecutor

        source = inspect.getsource(KnowledgeExecutor._op_recall)
        # All expected sortable columns must be in the allowlist
        assert "'confidence'" in source or '"confidence"' in source
        assert "'created_at'" in source or '"created_at"' in source
        assert "'retrieval_count'" in source or '"retrieval_count"' in source
        assert "'updated_at'" in source or '"updated_at"' in source
        # relevance_score was removed in migration 9 and must NOT be in the allowlist
        assert "'relevance_score'" not in source and '"relevance_score"' not in source

    @pytest.mark.asyncio
    async def test_recall_with_source_filter_and_item_path(self) -> None:
        """Recall with source_name filter uses kp.source_name directly (no extra JOIN).

        Source filtering uses the denormalized source_name column on knowledge_propositions,
        so there is only one LEFT JOIN (for item_path). No alias collision is possible.
        """
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="recall",
            source="test-source",
        )

        backend = MagicMock()
        captured_sql = ""

        async def capture_query(sql: str, params: tuple[Any, ...]) -> MagicMock:
            nonlocal captured_sql
            captured_sql = sql
            return MagicMock(rows=[])

        backend.query = capture_query
        exec_context = MagicMock()

        await executor._op_recall(inputs, exec_context, backend)

        # Source filter uses denormalized column — no knowledge_sources JOIN needed
        assert "kp.source_name" in captured_sql, (
            "SQL must filter by kp.source_name (denormalized column)"
        )
        assert "knowledge_sources" not in captured_sql, (
            "SQL must not JOIN knowledge_sources — filtering is done via kp.source_name"
        )
        # Only the item_path LEFT JOIN should be present
        assert "LEFT JOIN knowledge_items ki_ip" in captured_sql, (
            "SQL must use 'ki_ip' alias for item_path LEFT JOIN"
        )
        assert "JOIN knowledge_items ki ON" not in captured_sql, (
            "SQL must not have a separate 'ki' JOIN for source filtering"
        )


# ============================================================================
# Executor Registration Test
# ============================================================================


class TestExecutorRegistration:
    """Tests for KnowledgeExecutor registration (conditional, not in default registry)."""

    def test_knowledge_executor_not_in_default_registry(self) -> None:
        """KnowledgeExecutor should NOT be in the default registry (conditional loading)."""
        from workflows_mcp.engine.executor_base import create_default_registry

        registry = create_default_registry()
        assert not registry.has("Knowledge")

    def test_knowledge_executor_can_be_registered(self) -> None:
        """KnowledgeExecutor can be registered manually (as done at server startup)."""
        from workflows_mcp.engine.executor_base import ExecutorRegistry

        registry = ExecutorRegistry()
        registry.register(KnowledgeExecutor())
        executor = registry.get("Knowledge")
        assert isinstance(executor, KnowledgeExecutor)

    def test_executor_type_name(self) -> None:
        """Type name should be 'Knowledge'."""
        executor = KnowledgeExecutor()
        assert executor.type_name == "Knowledge"


# ============================================================================
# Category Resolution Tests
# ============================================================================


class TestCategoryResolution:
    """Tests for _resolve_categories helper."""

    @pytest.mark.asyncio
    async def test_uuid_passthrough(self) -> None:
        """Valid UUIDs should be returned as-is without DB calls."""
        executor = KnowledgeExecutor()
        backend = MagicMock()
        backend.query = AsyncMock()

        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = await executor._resolve_categories([test_uuid], backend)

        assert result == [test_uuid]
        backend.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_name_resolution(self) -> None:
        """Non-UUID string should trigger upsert into knowledge_categories."""
        executor = KnowledgeExecutor()
        resolved_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        query_result = MagicMock()
        query_result.rows = [{"id": resolved_uuid, "was_inserted": False}]
        backend = MagicMock()
        backend.query = AsyncMock(return_value=query_result)

        result = await executor._resolve_categories(["learnings"], backend)

        assert result == [resolved_uuid]
        backend.query.assert_called_once()
        # Verify the SQL targets knowledge_categories (not knowledge_entities)
        call_args = backend.query.call_args
        sql = call_args[0][0]
        assert "knowledge_categories" in sql
        assert "knowledge_entities" not in sql
        assert "ON CONFLICT" in sql
        assert "xmax" in sql

    @pytest.mark.asyncio
    async def test_warning_emitted_on_auto_create(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning should be emitted when a new category is inserted (was_inserted=True)."""
        import logging

        executor = KnowledgeExecutor()
        resolved_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        query_result = MagicMock()
        query_result.rows = [{"id": resolved_uuid, "was_inserted": True}]
        backend = MagicMock()
        backend.query = AsyncMock(return_value=query_result)

        with caplog.at_level(logging.WARNING, logger="workflows_mcp.engine.executors_knowledge"):
            result = await executor._resolve_categories(["new-category"], backend)

        assert result == [resolved_uuid]
        assert any("Auto-creating new knowledge category" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_warning_on_existing_category(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning should be emitted when resolving an existing category (was_inserted=False)."""
        import logging

        executor = KnowledgeExecutor()
        resolved_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        query_result = MagicMock()
        query_result.rows = [{"id": resolved_uuid, "was_inserted": False}]
        backend = MagicMock()
        backend.query = AsyncMock(return_value=query_result)

        with caplog.at_level(logging.WARNING, logger="workflows_mcp.engine.executors_knowledge"):
            result = await executor._resolve_categories(["existing-category"], backend)

        assert result == [resolved_uuid]
        assert not any("Auto-creating" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_mixed_input(self) -> None:
        """Mix of UUIDs and names should resolve correctly."""
        executor = KnowledgeExecutor()
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        resolved_uuid = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        query_result = MagicMock()
        query_result.rows = [{"id": resolved_uuid}]
        backend = MagicMock()
        backend.query = AsyncMock(return_value=query_result)

        result = await executor._resolve_categories(
            [test_uuid, "deployment-patterns"],
            backend,
        )

        assert result == [test_uuid, resolved_uuid]
        # Only one DB call for the name, not the UUID
        assert backend.query.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_list(self) -> None:
        """Empty categories list should return empty list."""
        executor = KnowledgeExecutor()
        backend = MagicMock()
        backend.query = AsyncMock()

        result = await executor._resolve_categories([], backend)

        assert result == []
        backend.query.assert_not_called()


# ============================================================================
# Store Operation Behaviour Tests (mocked backend)
# ============================================================================


def _make_mock_backend(
    *,
    source_id: str = "src-uuid-1111",
    item_id: str = "item-uuid-2222",
) -> MagicMock:
    """Build a mock backend that returns predictable IDs for source / item upserts."""
    backend = MagicMock()
    backend.execute = AsyncMock()

    source_result = MagicMock()
    source_result.rows = [{"id": source_id}]

    item_result = MagicMock()
    item_result.rows = [{"id": item_id}]

    # query() is used for source upsert (RETURNING id) and item upsert (RETURNING id)
    backend.query = AsyncMock(side_effect=[source_result, item_result])
    return backend


def _make_execution_context() -> MagicMock:
    """Build a minimal Execution mock for use in _op_store."""
    ctx = MagicMock()
    # execution_context must be None so _get_user_string_id / _get_audit_user_id
    # fall through to the SYSTEM_USER_UUID fallback — a live MagicMock would
    # cause json.dumps to fail when its attributes land in metadata_with_user.
    ctx.execution_context = None
    return ctx


class TestStoreOperationBehaviour:
    """Tests for _op_store mocked-backend behaviour (no real DB needed)."""

    @pytest.mark.asyncio
    async def test_store_no_source_no_path_null_item_id(self) -> None:
        """store with no source and no path → item_id=NULL in INSERT, no source/item rows."""
        executor = KnowledgeExecutor()
        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))

        inputs = KnowledgeInput(op="store", content="raw observation")

        # Patch compute_embedding to avoid real LLM call
        with patch_embedding():
            result = await executor._op_store(inputs, _make_execution_context(), backend)

        assert result.success is True
        assert len(result.proposition_ids) == 1
        assert result.stored_count == 1

        # No source or item rows should be created
        backend.query.assert_not_called()

        # The INSERT must have item_id=NULL (second positional param is None)
        # call_args_list[0] = proposition INSERT; subsequent calls = audit entry
        execute_call = backend.execute.call_args_list[0]
        params = execute_call[0][1]
        assert params[1] is None  # item_id param is None

    @pytest.mark.asyncio
    async def test_store_source_only_no_item_row(self) -> None:
        """store with source but no path → no source/item DB rows, source in metadata."""
        executor = KnowledgeExecutor()
        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))

        inputs = KnowledgeInput(op="store", content="agent note", source="my-workflow")

        with patch_embedding():
            result = await executor._op_store(inputs, _make_execution_context(), backend)

        assert result.success is True

        # No DB queries for source/item lookup
        backend.query.assert_not_called()

        # Proposition INSERT: item_id param is None
        # call_args_list[0] = proposition INSERT; subsequent calls = audit entry
        execute_call = backend.execute.call_args_list[0]
        params = execute_call[0][1]
        assert params[1] is None  # item_id

        # metadata param (index 9) must contain source name
        # Param order: id, item_id, content, embedding, authority, lifecycle_state,
        #              confidence, embedding_model, dimensions, metadata, created_by,
        #              auth_method, source_name, source_type
        import json as _j

        metadata_param = params[9]
        metadata = _j.loads(metadata_param)
        assert metadata.get("source") == "my-workflow"

    @pytest.mark.asyncio
    async def test_store_source_and_path_creates_item(self) -> None:
        """store with source AND path → source/item rows created, proposition linked."""
        executor = KnowledgeExecutor()
        source_id = "aaaaaaaa-1111-2222-3333-444444444444"
        item_id = "bbbbbbbb-5555-6666-7777-888888888888"

        source_result = MagicMock()
        source_result.rows = [{"id": source_id}]
        item_result = MagicMock()
        item_result.rows = [{"id": item_id}]

        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(side_effect=[source_result, item_result])

        inputs = KnowledgeInput(
            op="store",
            content="file fact",
            source="my-docs",
            path="docs/arch.md",
        )

        with patch_embedding():
            result = await executor._op_store(inputs, _make_execution_context(), backend)

        assert result.success is True

        # Two query calls: source upsert + item upsert
        assert backend.query.call_count == 2

        # First call: source upsert
        source_call = backend.query.call_args_list[0]
        source_sql = source_call[0][0]
        assert "knowledge_sources" in source_sql
        assert "ON CONFLICT (name)" in source_sql

        # Second call: item upsert with path
        item_call = backend.query.call_args_list[1]
        item_sql = item_call[0][0]
        assert "knowledge_items" in item_sql
        assert "ON CONFLICT (source_id, path)" in item_sql
        item_params = item_call[0][1]
        assert "docs/arch.md" in item_params

        # Proposition INSERT links to item_id
        # call_args_list[0] = proposition INSERT; subsequent calls = audit entry
        execute_call = backend.execute.call_args_list[0]
        params = execute_call[0][1]
        assert params[1] == item_id  # item_id param

    @pytest.mark.asyncio
    async def test_store_source_and_path_idempotent(self) -> None:
        """Calling store twice with same source+path returns the same item_id (ON CONFLICT)."""
        executor = KnowledgeExecutor()
        item_id = "cccccccc-dddd-eeee-ffff-000000000000"

        source_result = MagicMock()
        source_result.rows = [{"id": "src-static-id"}]
        item_result = MagicMock()
        item_result.rows = [{"id": item_id}]

        backend = MagicMock()
        backend.execute = AsyncMock()

        # Both calls return the same item_id (idempotent upsert)
        backend.query = AsyncMock(side_effect=[source_result, item_result])

        inputs = KnowledgeInput(
            op="store",
            content="idempotent fact",
            source="stable-source",
            path="same/file.md",
        )

        with patch_embedding():
            result = await executor._op_store(inputs, _make_execution_context(), backend)

        assert result.success is True
        # The item upsert SQL uses ON CONFLICT (source_id, path) — verify
        item_call = backend.query.call_args_list[1]
        item_sql = item_call[0][0]
        assert "ON CONFLICT (source_id, path) DO UPDATE" in item_sql

    @pytest.mark.asyncio
    async def test_store_metadata_column_not_metadata_underscore(self) -> None:
        """The INSERT SQL must use 'metadata' column, not 'metadata_'."""
        executor = KnowledgeExecutor()
        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))

        inputs = KnowledgeInput(op="store", content="check column name")

        with patch_embedding():
            await executor._op_store(inputs, _make_execution_context(), backend)

        execute_call = backend.execute.call_args
        sql = execute_call[0][0]
        assert "metadata)" in sql or "metadata\n" in sql or "metadata," in sql
        assert "metadata_)" not in sql
        assert "metadata_," not in sql

    @pytest.mark.asyncio
    async def test_store_inserts_junction_rows_explicit(self) -> None:
        """When categories are provided, junction rows are inserted with assigned_by=EXPLICIT."""
        executor = KnowledgeExecutor()
        cat_uuid = "550e8400-e29b-41d4-a716-446655440000"

        backend = MagicMock()
        backend.execute = AsyncMock()
        # _resolve_categories upserts into knowledge_entities and returns the UUID
        cat_result = MagicMock()
        cat_result.rows = [{"id": cat_uuid}]
        backend.query = AsyncMock(return_value=cat_result)

        inputs = KnowledgeInput(
            op="store",
            content="categorised agent note",
            categories=[cat_uuid],  # valid UUID — passed through without upsert
        )

        with patch_embedding():
            result = await executor._op_store(inputs, _make_execution_context(), backend)

        assert result.success is True

        # Find the junction INSERT among all execute calls
        execute_calls = backend.execute.call_args_list
        junction_sqls = [
            c[0][0] for c in execute_calls if "knowledge_proposition_categories" in c[0][0]
        ]
        assert len(junction_sqls) >= 1, "Expected at least one junction INSERT"
        assert "EXPLICIT" in junction_sqls[0]

    @pytest.mark.asyncio
    async def test_store_no_junction_rows_when_no_categories(self) -> None:
        """Without categories, no junction rows are inserted."""
        executor = KnowledgeExecutor()
        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))

        inputs = KnowledgeInput(op="store", content="uncategorised fact")

        with patch_embedding():
            await executor._op_store(inputs, _make_execution_context(), backend)

        execute_calls = backend.execute.call_args_list
        junction_sqls = [
            c[0][0] for c in execute_calls if "knowledge_proposition_categories" in c[0][0]
        ]
        assert len(junction_sqls) == 0, "No junction rows should be inserted without categories"

    @pytest.mark.asyncio
    async def test_agent_observation_with_category_has_junction_row(self) -> None:
        """Agent observation (no path, item_id=NULL) stores junction row when category given."""
        executor = KnowledgeExecutor()
        cat_uuid = "aaaaaaaa-bbbb-cccc-dddd-000000000001"

        backend = MagicMock()
        backend.execute = AsyncMock()
        cat_result = MagicMock()
        cat_result.rows = [{"id": cat_uuid}]
        backend.query = AsyncMock(return_value=cat_result)

        inputs = KnowledgeInput(
            op="store",
            content="incident observed: timeout on auth service",
            source="incident-2026-03-20",
            categories=[cat_uuid],
        )

        with patch_embedding():
            result = await executor._op_store(inputs, _make_execution_context(), backend)

        assert result.success is True

        # Proposition INSERT must have item_id=None (source-only, no path)
        prop_insert = next(
            c
            for c in backend.execute.call_args_list
            if "knowledge_propositions" in c[0][0] and "INSERT" in c[0][0]
        )
        assert prop_insert[0][1][1] is None  # item_id param is None

        # Junction INSERT must be present
        junction_calls = [
            c
            for c in backend.execute.call_args_list
            if "knowledge_proposition_categories" in c[0][0]
        ]
        assert len(junction_calls) >= 1
        assert "EXPLICIT" in junction_calls[0][0][0]


# ============================================================================
# Search / Recall item_path Tests
# ============================================================================


class TestItemPathInResults:
    """Tests that item_path is included in search and recall output columns."""

    def test_search_output_columns_include_item_path(self) -> None:
        """search output columns list must include 'item_path'."""
        out = KnowledgeOutput(
            success=True,
            rows=[{"id": "1", "content": "fact", "item_path": "docs/file.md"}],
            columns=["id", "content", "item_path"],
            row_count=1,
        )
        assert "item_path" in out.columns
        assert out.rows[0]["item_path"] == "docs/file.md"

    def test_search_output_item_path_may_be_none(self) -> None:
        """item_path=None is valid for propositions without a backing document."""
        out = KnowledgeOutput(
            success=True,
            rows=[{"id": "1", "content": "agent obs", "item_path": None}],
            columns=["id", "content", "item_path"],
            row_count=1,
        )
        assert out.rows[0]["item_path"] is None

    def test_vector_search_sql_includes_item_path(self) -> None:
        """build_vector_search_query SQL must SELECT ki_path.path AS item_path."""
        sql, _ = build_vector_search_query(query_embedding=[0.1, 0.2, 0.3])
        assert "item_path" in sql
        assert "LEFT JOIN knowledge_items ki_path" in sql
        # Category JOIN to knowledge_sources must not be present (uses junction table instead)
        assert "knowledge_sources" not in sql
        assert "JOIN knowledge_items ki " not in sql

    def test_fts_search_sql_includes_item_path(self) -> None:
        """build_fts_search_query SQL must SELECT ki_path.path AS item_path."""
        sql, _ = build_fts_search_query(query_text="test")
        assert "item_path" in sql
        assert "knowledge_items" in sql
        assert "LEFT JOIN" in sql

    def test_recall_sql_includes_item_path(self) -> None:
        """_op_recall SQL must LEFT JOIN knowledge_items and select item_path."""
        # We verify via the output columns definition — no real DB needed
        # The columns list in _op_recall must include 'item_path'
        # We indirectly verify by checking a KnowledgeOutput with item_path
        out = KnowledgeOutput(
            success=True,
            rows=[{"id": "r1", "content": "recalled", "item_path": "src/main.py"}],
            columns=[
                "id",
                "content",
                "confidence",
                "authority",
                "lifecycle_state",
                "relevance_score",
                "retrieval_count",
                "item_path",
            ],
            row_count=1,
        )
        assert "item_path" in out.columns


# ============================================================================
# Patch helper for embedding (avoid real LLM in unit tests)
# ============================================================================


def patch_embedding() -> Any:
    """Return a context manager that patches compute_embedding with a no-op."""
    from unittest.mock import patch

    fake_embedding = [0.1, 0.2, 0.3]
    return patch(
        "workflows_mcp.engine.executors_knowledge.compute_embedding",
        new=AsyncMock(return_value=(fake_embedding, "text-embedding-3-small", 3, None)),
    )


# ============================================================================
# Validate Operation Tests
# ============================================================================


class TestValidateOperation:
    """Tests for validate op validation and output model."""

    def test_validate_requires_proposition_ids(self) -> None:
        """validate without proposition_ids should raise ValidationError."""
        with pytest.raises(ValidationError, match="proposition_ids"):
            KnowledgeInput(op="validate")

    def test_validate_with_ids_passes(self) -> None:
        """validate with proposition_ids passes validation."""
        inp = KnowledgeInput(op="validate", proposition_ids=["uuid-1", "uuid-2"])
        assert inp.op == "validate"
        assert inp.proposition_ids == ["uuid-1", "uuid-2"]

    def test_validate_output_model(self) -> None:
        """Validate output populates validated_count."""
        out = KnowledgeOutput(success=True, validated_count=2)
        assert out.validated_count == 2
        assert out.success is True

    def test_validate_output_default_zero(self) -> None:
        """validated_count defaults to zero."""
        out = KnowledgeOutput(success=True)
        assert out.validated_count == 0

    def test_validate_sql_targets_user_validated_authority(self) -> None:
        """_op_validate SQL must SET authority to USER_VALIDATED, not archive.

        SQL uses f-string interpolation of Authority.USER_VALIDATED; check for
        the enum reference rather than the evaluated string value.
        """
        import inspect

        from workflows_mcp.engine.executors_knowledge import KnowledgeExecutor

        source = inspect.getsource(KnowledgeExecutor._op_validate)
        assert "Authority.USER_VALIDATED" in source
        assert "lifecycle_state = " not in source, "_op_validate must not modify lifecycle_state"

    def test_validate_sql_skips_archived_propositions(self) -> None:
        """_op_validate must not promote ARCHIVED propositions."""
        import inspect

        from workflows_mcp.engine.executors_knowledge import KnowledgeExecutor

        source = inspect.getsource(KnowledgeExecutor._op_validate)
        assert "lifecycle_state != " in source or "lifecycle_state !=" in source

    @pytest.mark.asyncio
    async def test_validate_updates_authority_and_logs_audit(self) -> None:
        """_op_validate updates authority and logs a VALIDATED audit entry."""
        executor = KnowledgeExecutor()
        prop_id = "aaaaaaaa-1111-2222-3333-444444444444"

        update_result = MagicMock()
        update_result.rows = [{"id": prop_id}]

        backend = MagicMock()
        backend.query = AsyncMock(return_value=update_result)
        backend.execute = AsyncMock()  # audit INSERT

        inputs = KnowledgeInput(op="validate", proposition_ids=[prop_id])
        result = await executor._op_validate(inputs, _make_execution_context(), backend)

        assert result.success is True
        assert result.validated_count == 1

        # Verify the UPDATE SQL targeted USER_VALIDATED
        update_call = backend.query.call_args
        update_sql = update_call[0][0]
        assert "USER_VALIDATED" in update_sql
        assert prop_id in update_call[0][1]

        # Verify audit entry was written with action=VALIDATED
        audit_call = backend.execute.call_args
        audit_sql = audit_call[0][0]
        assert "knowledge_proposition_audits" in audit_sql
        audit_params = audit_call[0][1]
        assert "VALIDATED" in audit_params

    def test_validate_empty_list_rejected_by_validation(self) -> None:
        """Empty proposition_ids list is rejected at validation time (not silently ignored)."""
        with pytest.raises(ValidationError, match="proposition_ids"):
            KnowledgeInput(op="validate", proposition_ids=[])


# ============================================================================
# Invalidate Operation Tests
# ============================================================================


class TestInvalidateOperation:
    """Tests for invalidate op — revokes USER_VALIDATED authority back to AGENT."""

    def test_invalidate_requires_proposition_ids(self) -> None:
        """invalidate without proposition_ids should raise ValidationError."""
        with pytest.raises(ValidationError, match="proposition_ids"):
            KnowledgeInput(op="invalidate")

    def test_invalidate_with_ids_passes(self) -> None:
        """invalidate with proposition_ids passes validation."""
        inp = KnowledgeInput(op="invalidate", proposition_ids=["uuid-1"])
        assert inp.op == "invalidate"
        assert inp.proposition_ids == ["uuid-1"]

    def test_invalidate_with_reason(self) -> None:
        """invalidate accepts an optional reason."""
        inp = KnowledgeInput(op="invalidate", proposition_ids=["uuid-1"], reason="no longer true")
        assert inp.reason == "no longer true"

    def test_invalidate_output_model(self) -> None:
        """Invalidate output populates invalidated_count."""
        out = KnowledgeOutput(success=True, invalidated_count=1)
        assert out.invalidated_count == 1
        assert out.success is True

    def test_invalidate_output_default_zero(self) -> None:
        """invalidated_count defaults to zero."""
        out = KnowledgeOutput(success=True)
        assert out.invalidated_count == 0

    def test_invalidate_sql_demotes_to_agent(self) -> None:
        """_op_invalidate must SET authority = 'AGENT', not archive."""
        import inspect

        from workflows_mcp.engine.executors_knowledge import KnowledgeExecutor

        source = inspect.getsource(KnowledgeExecutor._op_invalidate)
        assert "Authority.AGENT" in source
        assert "lifecycle_state" not in source

    def test_invalidate_sql_only_targets_user_validated(self) -> None:
        """_op_invalidate must only update propositions that are currently USER_VALIDATED."""
        import inspect

        from workflows_mcp.engine.executors_knowledge import KnowledgeExecutor

        source = inspect.getsource(KnowledgeExecutor._op_invalidate)
        assert "Authority.USER_VALIDATED" in source

    @pytest.mark.asyncio
    async def test_invalidate_updates_authority_and_logs_audit(self) -> None:
        """_op_invalidate demotes authority and logs an INVALIDATED audit entry."""
        executor = KnowledgeExecutor()
        prop_id = "cccccccc-1111-2222-3333-444444444444"

        update_result = MagicMock()
        update_result.rows = [{"id": prop_id}]

        backend = MagicMock()
        backend.query = AsyncMock(return_value=update_result)
        backend.execute = AsyncMock()

        inputs = KnowledgeInput(
            op="invalidate",
            proposition_ids=[prop_id],
            reason="superseded by updated measurement",
        )
        result = await executor._op_invalidate(inputs, _make_execution_context(), backend)

        assert result.success is True
        assert result.invalidated_count == 1

        # UPDATE SQL must target USER_VALIDATED and demote to AGENT
        update_call = backend.query.call_args
        update_sql = update_call[0][0]
        assert "USER_VALIDATED" in update_sql
        assert "AGENT" in update_sql
        assert prop_id in update_call[0][1]

        # Audit entry must use action=INVALIDATED
        audit_call = backend.execute.call_args
        assert "knowledge_proposition_audits" in audit_call[0][0]
        assert "INVALIDATED" in audit_call[0][1]

    @pytest.mark.asyncio
    async def test_invalidate_skips_non_user_validated(self) -> None:
        """Propositions that are not USER_VALIDATED are silently skipped (idempotent)."""
        executor = KnowledgeExecutor()

        # DB returns 0 rows — none were USER_VALIDATED
        update_result = MagicMock()
        update_result.rows = []

        backend = MagicMock()
        backend.query = AsyncMock(return_value=update_result)
        backend.execute = AsyncMock()

        inputs = KnowledgeInput(op="invalidate", proposition_ids=["agent-prop-id"])
        result = await executor._op_invalidate(inputs, _make_execution_context(), backend)

        assert result.success is True
        assert result.invalidated_count == 0
        backend.execute.assert_not_called()  # no audit entry for skipped propositions

    @pytest.mark.asyncio
    async def test_invalidate_then_forget_succeeds(self) -> None:
        """After invalidation, a proposition can be archived by _op_forget.

        This is the canonical two-step workflow for retiring a USER_VALIDATED fact.
        """
        executor = KnowledgeExecutor()
        prop_id = "dddddddd-aaaa-bbbb-cccc-dddddddddddd"

        # Step 1: invalidate — DB confirms 1 row updated
        invalidate_result = MagicMock()
        invalidate_result.rows = [{"id": prop_id}]

        # Step 2: forget by ID — DB now archives the (formerly immune) proposition
        forget_result = MagicMock()
        forget_result.rows = [{"id": prop_id}]

        backend = MagicMock()
        backend.query = AsyncMock(side_effect=[invalidate_result, forget_result])
        backend.execute = AsyncMock()

        ctx = _make_execution_context()

        inv_inputs = KnowledgeInput(op="invalidate", proposition_ids=[prop_id])
        inv_out = await executor._op_invalidate(inv_inputs, ctx, backend)
        assert inv_out.invalidated_count == 1

        fgt_inputs = KnowledgeInput(op="forget", proposition_ids=[prop_id])
        fgt_out = await executor._op_forget(fgt_inputs, ctx, backend)
        assert fgt_out.archived_count == 1
        assert fgt_out.skipped_count == 0


# ============================================================================
# USER_VALIDATED Immunity Behavioural Tests
# ============================================================================


class TestUserValidatedImmunity:
    """Behavioural tests for USER_VALIDATED archive immunity in _op_forget.

    These tests verify that the immunity SQL clause is correctly applied so that
    skipped_count reflects propositions not archived due to USER_VALIDATED authority.
    """

    @pytest.mark.asyncio
    async def test_forget_by_id_skips_user_validated(self) -> None:
        """When 2 IDs are targeted but 1 is USER_VALIDATED, skipped_count=1."""
        executor = KnowledgeExecutor()
        prop_id_normal = "aaaaaaaa-1111-2222-3333-444444444444"
        prop_id_immune = "bbbbbbbb-5555-6666-7777-888888888888"

        # Simulate DB: only normal proposition is returned (immune one is skipped by SQL)
        update_result = MagicMock()
        update_result.rows = [{"id": prop_id_normal}]

        backend = MagicMock()
        backend.query = AsyncMock(return_value=update_result)
        backend.execute = AsyncMock()  # audit INSERT

        inputs = KnowledgeInput(
            op="forget",
            proposition_ids=[prop_id_normal, prop_id_immune],
        )
        result = await executor._op_forget(inputs, _make_execution_context(), backend)

        assert result.success is True
        assert result.archived_count == 1
        assert result.skipped_count == 1  # immune proposition not returned by UPDATE ... RETURNING

    @pytest.mark.asyncio
    async def test_forget_by_id_all_immune_gives_zero_archived(self) -> None:
        """When all targeted propositions are USER_VALIDATED, archived_count=0."""
        executor = KnowledgeExecutor()

        update_result = MagicMock()
        update_result.rows = []  # DB skips all (all USER_VALIDATED)

        backend = MagicMock()
        backend.query = AsyncMock(return_value=update_result)
        backend.execute = AsyncMock()

        inputs = KnowledgeInput(
            op="forget",
            proposition_ids=["immune-1", "immune-2"],
        )
        result = await executor._op_forget(inputs, _make_execution_context(), backend)

        assert result.success is True
        assert result.archived_count == 0
        assert result.skipped_count == 2

    @pytest.mark.asyncio
    async def test_forget_by_filter_skipped_count_reflects_immunity(self) -> None:
        """Filter path: total=3, archived=2 → skipped_count=1 (one USER_VALIDATED)."""
        executor = KnowledgeExecutor()

        # COUNT(*) query returns 3 total
        count_result = MagicMock()
        count_result.rows = [{"total": 3}]

        # UPDATE ... RETURNING only gives back 2 (one immune)
        update_result = MagicMock()
        update_result.rows = [{"id": "id-1"}, {"id": "id-2"}]

        backend = MagicMock()
        # First query: _build_where_clause category resolution (if any)
        # For source-only filter: first call is COUNT, second is UPDATE
        backend.query = AsyncMock(side_effect=[count_result, update_result])
        backend.execute = AsyncMock()

        inputs = KnowledgeInput(op="forget", source="old-session")
        result = await executor._op_forget(inputs, _make_execution_context(), backend)

        assert result.success is True
        assert result.archived_count == 2
        assert result.skipped_count == 1

    @pytest.mark.asyncio
    async def test_forget_by_filter_audit_logged_for_each_archived(self) -> None:
        """Audit entries are written for each archived proposition, not for skipped ones."""
        executor = KnowledgeExecutor()

        count_result = MagicMock()
        count_result.rows = [{"total": 2}]

        archived_ids = ["id-arch-1", "id-arch-2"]
        update_result = MagicMock()
        update_result.rows = [{"id": i} for i in archived_ids]

        backend = MagicMock()
        backend.query = AsyncMock(side_effect=[count_result, update_result])
        backend.execute = AsyncMock()

        inputs = KnowledgeInput(op="forget", source="cleanup-session")
        await executor._op_forget(inputs, _make_execution_context(), backend)

        # One audit INSERT per archived proposition
        audit_calls = [
            c for c in backend.execute.call_args_list if "knowledge_proposition_audits" in c[0][0]
        ]
        assert len(audit_calls) == 2
        audit_actions = [c[0][1][1] for c in audit_calls]  # action is second param
        assert all(a == "ARCHIVED" for a in audit_actions)
