"""Tests for the Knowledge executor and supporting modules.

Tests focus on pure logic (RRF fusion, context assembly, schema DDL,
input validation) — no real database needed. PostgreSQL-specific tests
require optional deps.
"""

from __future__ import annotations

from typing import Any

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
    assemble_context,
    estimate_tokens,
)
from workflows_mcp.engine.knowledge.schema import get_init_schema_sql
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


class TestSchemaDDL:
    """Tests for idempotent schema DDL."""

    def test_get_init_schema_sql_returns_string(self) -> None:
        """DDL should be a non-empty string."""
        sql = get_init_schema_sql()
        assert isinstance(sql, str)
        assert len(sql) > 100

    def test_ddl_contains_extension(self) -> None:
        """DDL should create pgvector extension."""
        sql = get_init_schema_sql()
        assert "CREATE EXTENSION IF NOT EXISTS vector" in sql

    def test_ddl_contains_all_tables(self) -> None:
        """DDL should create all knowledge tables."""
        sql = get_init_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS knowledge_sources" in sql
        assert "CREATE TABLE IF NOT EXISTS knowledge_items" in sql
        assert "CREATE TABLE IF NOT EXISTS knowledge_propositions" in sql

    def test_ddl_contains_indexes(self) -> None:
        """DDL should create necessary indexes."""
        sql = get_init_schema_sql()
        assert "CREATE INDEX IF NOT EXISTS idx_kp_org_id" in sql
        assert "CREATE INDEX IF NOT EXISTS idx_kp_search_vector" in sql

    def test_ddl_is_idempotent(self) -> None:
        """DDL should use IF NOT EXISTS throughout."""
        sql = get_init_schema_sql()
        # Every CREATE should have IF NOT EXISTS
        for line in sql.split("\n"):
            line = line.strip()
            if line.startswith("CREATE TABLE"):
                assert "IF NOT EXISTS" in line, f"Missing IF NOT EXISTS: {line}"
            if line.startswith("CREATE INDEX"):
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
            org_id="550e8400-e29b-41d4-a716-446655440000",
            query_embedding=embedding,
        )

        assert "$1" in sql  # embedding param
        assert "$2" in sql  # org_id param
        assert "<=>" in sql  # cosine distance
        assert "knowledge_propositions" in sql
        assert len(params) == 5  # embedding, org_id, state, confidence, candidate_limit

    def test_vector_search_with_source_filter(self) -> None:
        """Source filter should add JOIN and WHERE clause."""
        sql, params = build_vector_search_query(
            org_id="550e8400-e29b-41d4-a716-446655440000",
            query_embedding=[0.1, 0.2],
            source="internal-docs",
        )

        assert "knowledge_sources" in sql
        assert "ks.name =" in sql
        assert "internal-docs" in params

    def test_vector_search_with_source_prefix(self) -> None:
        """Source prefix with * should use LIKE."""
        sql, params = build_vector_search_query(
            org_id="550e8400-e29b-41d4-a716-446655440000",
            query_embedding=[0.1, 0.2],
            source="workflow:*",
        )

        assert "LIKE" in sql
        assert "workflow:%" in params

    def test_fts_search_basic(self) -> None:
        """Basic FTS search should produce valid SQL."""
        sql, params = build_fts_search_query(
            org_id="550e8400-e29b-41d4-a716-446655440000",
            query_text="deployment patterns",
        )

        assert "plainto_tsquery" in sql
        assert "ts_rank" in sql
        assert "deployment patterns" in params

    def test_fts_search_with_categories(self) -> None:
        """Category filter should add category overlap clause."""
        sql, params = build_fts_search_query(
            org_id="550e8400-e29b-41d4-a716-446655440000",
            query_text="test",
            categories=["cat-uuid-1"],
        )

        assert "&&" in sql  # Array overlap operator
        assert "knowledge_sources" in sql

    def test_positional_params_are_sequential(self) -> None:
        """All $N params should be sequential in the SQL."""
        sql, params = build_vector_search_query(
            org_id="550e8400-e29b-41d4-a716-446655440000",
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
        assert inp.min_confidence == DEFAULT_MIN_CONFIDENCE

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
        assert inp.diversity is False

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
        monkeypatch.delenv("KNOWLEDGE_ORG_ID", raising=False)

        inp = KnowledgeInput(op="recall")
        assert inp.host == "localhost"
        assert inp.port == 5432
        assert inp.database == "knowledge_db"
        assert inp.username is None
        assert inp.password is None
        assert inp.org_id is None

    def test_env_vars_are_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars should provide defaults when set."""
        monkeypatch.setenv("KNOWLEDGE_DB_HOST", "kb.example.com")
        monkeypatch.setenv("KNOWLEDGE_DB_PORT", "5433")
        monkeypatch.setenv("KNOWLEDGE_DB_NAME", "my_knowledge")
        monkeypatch.setenv("KNOWLEDGE_DB_USER", "kb_user")
        monkeypatch.setenv("KNOWLEDGE_DB_PASSWORD", "kb_pass")
        monkeypatch.setenv("KNOWLEDGE_ORG_ID", "org-uuid-123")

        inp = KnowledgeInput(op="recall")
        assert inp.host == "kb.example.com"
        assert inp.port == 5433
        assert inp.database == "my_knowledge"
        assert inp.username == "kb_user"
        assert inp.password == "kb_pass"
        assert inp.org_id == "org-uuid-123"

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
# Executor Registration Test
# ============================================================================


class TestExecutorRegistration:
    """Tests for KnowledgeExecutor in the default registry."""

    def test_knowledge_executor_in_registry(self) -> None:
        """KnowledgeExecutor should be registered in the default registry."""
        from workflows_mcp.engine.executor_base import create_default_registry

        registry = create_default_registry()
        executor = registry.get("Knowledge")
        assert executor is not None
        assert isinstance(executor, KnowledgeExecutor)

    def test_executor_type_name(self) -> None:
        """Type name should be 'Knowledge'."""
        executor = KnowledgeExecutor()
        assert executor.type_name == "Knowledge"
