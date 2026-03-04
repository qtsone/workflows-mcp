"""Knowledge executor for workflow-native knowledge access.

Provides search, store, recall, forget, and context operations against
PostgreSQL knowledge tables (pgvector + tsvector). Enables clean YAML
interface for knowledge operations in workflows.

Pattern reference: executors_sql.py (SqlExecutor)
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)
from .executors_llm import compute_embedding
from .knowledge.constants import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_CONFIDENCE,
    LifecycleState,
)
from .knowledge.context import assemble_context
from .knowledge.schema import get_init_schema_sql
from .knowledge.search import (
    build_fts_search_query,
    build_vector_search_query,
    rrf_fusion,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Input / Output Models
# ===========================================================================


class KnowledgeInput(BlockInput):
    """Input schema for Knowledge block.

    Supports five operations: search, store, recall, forget, context.

    Connection config resolves with priority: YAML inputs > env vars > defaults.
    Set KNOWLEDGE_DB_* environment variables once in your MCP server config
    to avoid repeating connection details in every workflow.

    Environment variables:
        KNOWLEDGE_DB_HOST: PostgreSQL host (default: localhost)
        KNOWLEDGE_DB_PORT: PostgreSQL port (default: 5432)
        KNOWLEDGE_DB_NAME: Database name (default: knowledge_db)
        KNOWLEDGE_DB_USER: Username
        KNOWLEDGE_DB_PASSWORD: Password
        KNOWLEDGE_ORG_ID: Organization ID for multi-tenant scoping

    Examples:
        ```yaml
        - id: find_patterns
          type: Knowledge
          inputs:
            op: search
            query: "deployment patterns for web services"
            source: "internal-docs"
            limit: 10
        ```
    """

    op: Literal["search", "store", "recall", "forget", "context"] = Field(
        description="Operation to perform",
    )

    # --- Database connection (env vars > defaults, YAML inputs override both) ---
    host: str = Field(
        default_factory=lambda: os.environ.get("KNOWLEDGE_DB_HOST", "localhost"),
        description="PostgreSQL host (env: KNOWLEDGE_DB_HOST)",
    )
    port: int | str = Field(
        default_factory=lambda: int(os.environ.get("KNOWLEDGE_DB_PORT", "5432")),
        description="PostgreSQL port (env: KNOWLEDGE_DB_PORT)",
    )
    database: str = Field(
        default_factory=lambda: os.environ.get("KNOWLEDGE_DB_NAME", "knowledge_db"),
        description="PostgreSQL database name (env: KNOWLEDGE_DB_NAME)",
    )
    username: str | None = Field(
        default_factory=lambda: os.environ.get("KNOWLEDGE_DB_USER"),
        description="PostgreSQL username (env: KNOWLEDGE_DB_USER)",
    )
    password: str | None = Field(
        default_factory=lambda: os.environ.get("KNOWLEDGE_DB_PASSWORD"),
        description="PostgreSQL password (env: KNOWLEDGE_DB_PASSWORD)",
    )
    org_id: str | None = Field(
        default_factory=lambda: os.environ.get("KNOWLEDGE_ORG_ID"),
        description="Organization ID for scoping queries (env: KNOWLEDGE_ORG_ID)",
    )

    # --- Search / Context fields ---
    query: str | None = Field(
        default=None,
        description="Search query text (required for search/context)",
    )
    source: str | None = Field(
        default=None,
        description="Filter by source name (exact or prefix with *)",
    )
    categories: list[str] | None = Field(
        default=None,
        description="Filter by category UUIDs",
    )
    min_confidence: float = Field(
        default=DEFAULT_MIN_CONFIDENCE,
        description="Minimum confidence threshold",
    )
    lifecycle_state: str = Field(
        default=LifecycleState.ACTIVE,
        description="Filter by lifecycle state (default: ACTIVE)",
    )
    limit: int | str = Field(
        default=DEFAULT_LIMIT,
        description="Maximum number of results",
    )

    # --- Context-specific fields ---
    max_tokens: int | str = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Token budget for context assembly",
    )
    diversity: bool = Field(
        default=False,
        description="Use MMR for diversity in context results",
    )

    # --- Store fields ---
    content: str | None = Field(
        default=None,
        description="Proposition content to store",
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence score for stored proposition",
    )

    # --- Recall fields ---
    where: dict[str, Any] | None = Field(
        default=None,
        description="Filter conditions for recall (key-value pairs)",
    )
    order: list[str] | None = Field(
        default=None,
        description="Order by fields (e.g., ['relevance_score:desc'])",
    )

    # --- Forget fields ---
    proposition_ids: list[str] | str | None = Field(
        default=None,
        description="UUIDs of propositions to archive",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for archiving (stored in metadata)",
    )

    # --- Embedding overrides ---
    embedding_profile: str = Field(
        default="embedding",
        description="LLM config profile for embeddings",
    )

    @model_validator(mode="after")
    def validate_op_fields(self) -> KnowledgeInput:
        """Validate that required fields are present for each operation."""
        if self.op in ("search", "context") and not self.query:
            raise ValueError(f"'query' is required for op='{self.op}'")
        if self.op == "store" and not self.content:
            raise ValueError("'content' is required for op='store'")
        if self.op == "forget" and not self.proposition_ids:
            raise ValueError("'proposition_ids' is required for op='forget'")
        return self


class KnowledgeOutput(BlockOutput):
    """Output schema for Knowledge block.

    Fields are populated based on the operation:
    - search/recall: rows, columns, row_count
    - store: proposition_ids, stored_count
    - forget: archived_count, skipped_count
    - context: context_text, proposition_count, tokens_used
    """

    success: bool = Field(default=False, description="Whether the operation succeeded")
    error: str | None = Field(default=None, description="Error message if failed")

    # Search / Recall output
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Result rows (search/recall)",
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Column names in the result rows",
    )
    row_count: int = Field(default=0, description="Number of result rows")

    # Store output
    proposition_ids: list[str] = Field(
        default_factory=list,
        description="UUIDs of stored propositions",
    )
    stored_count: int = Field(default=0, description="Number of propositions stored")

    # Forget output
    archived_count: int = Field(default=0, description="Number archived")
    skipped_count: int = Field(default=0, description="Number skipped (immune)")

    # Context output
    context_text: str = Field(default="", description="Clean content assembled text")
    proposition_count: int = Field(default=0, description="Propositions included in context")
    tokens_used: int = Field(default=0, description="Estimated tokens used")


# ===========================================================================
# Executor
# ===========================================================================


class KnowledgeExecutor(BlockExecutor):
    """Executor for knowledge operations against PostgreSQL.

    Provides search (vector + FTS + RRF), store (with auto-embedding),
    recall (filtered SELECT), forget (archive), and context (token-budgeted
    clean-content assembly) operations.

    Uses PostgresBackend for async connection pooling.
    Auto-creates pgvector extension and knowledge tables on first use.

    Configuration via environment variables (recommended) or YAML inputs:

        Environment variables (set once in MCP server config):
            KNOWLEDGE_DB_HOST, KNOWLEDGE_DB_PORT, KNOWLEDGE_DB_NAME,
            KNOWLEDGE_DB_USER, KNOWLEDGE_DB_PASSWORD, KNOWLEDGE_ORG_ID

        YAML inputs (override env vars per-block):
            ```yaml
            - id: search_kb
              type: Knowledge
              inputs:
                op: search
                query: "deployment patterns"
            ```
    """

    type_name: ClassVar[str] = "Knowledge"
    input_type: ClassVar[type[BlockInput]] = KnowledgeInput
    output_type: ClassVar[type[BlockOutput]] = KnowledgeOutput
    examples: ClassVar[str] = """```yaml
- id: search_kb
  type: Knowledge
  inputs:
    op: search
    query: "deployment patterns"
    limit: 10
```"""

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_network=True,
        can_read_fs=False,
        can_write_fs=False,
    )

    async def execute(  # type: ignore[override]
        self, inputs: KnowledgeInput, context: Execution
    ) -> KnowledgeOutput:
        """Execute a knowledge operation.

        Follows SqlExecutor pattern: create backend → connect → execute → disconnect.
        """
        try:
            backend = self._create_backend()
            config = self._create_config(inputs)
        except Exception as e:
            return KnowledgeOutput(success=False, error=str(e))

        try:
            await backend.connect(config)

            # Auto-create schema on first use
            await backend.execute_script(get_init_schema_sql())

            # Route to operation handler
            op_handlers = {
                "search": self._op_search,
                "store": self._op_store,
                "recall": self._op_recall,
                "forget": self._op_forget,
                "context": self._op_context,
            }
            handler = op_handlers[inputs.op]
            return await handler(inputs, context, backend)

        except Exception as e:
            logger.error(f"Knowledge operation '{inputs.op}' failed: {e}")
            return KnowledgeOutput(success=False, error=str(e))
        finally:
            await backend.disconnect()

    def _create_backend(self) -> Any:
        """Create PostgresBackend instance."""
        try:
            from .sql.postgres_backend import PostgresBackend
        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires 'asyncpg'. "
                "Install with: pip install workflows-mcp[postgresql]"
            ) from e
        return PostgresBackend()

    def _create_config(self, inputs: KnowledgeInput) -> Any:
        """Create ConnectionConfig from inputs."""
        from .sql import ConnectionConfig, DatabaseEngine

        port = int(inputs.port) if isinstance(inputs.port, str) else inputs.port

        return ConnectionConfig(
            dialect=DatabaseEngine.POSTGRESQL,
            host=inputs.host,
            port=port,
            database=inputs.database,
            username=inputs.username,
            password=inputs.password,
        )

    # ------------------------------------------------------------------
    # Operation Handlers
    # ------------------------------------------------------------------

    async def _op_search(
        self,
        inputs: KnowledgeInput,
        context: Execution,
        backend: Any,
    ) -> KnowledgeOutput:
        """Hybrid search: pgvector cosine + tsvector FTS + RRF fusion."""
        assert inputs.query is not None

        limit = int(inputs.limit) if isinstance(inputs.limit, str) else inputs.limit
        org_id = inputs.org_id or str(uuid.UUID(int=0))

        # Compute query embedding
        embedding, _, _, _ = await compute_embedding(
            text=inputs.query,
            context=context,
            profile=inputs.embedding_profile,
        )

        # Vector search
        vector_sql, vector_params = build_vector_search_query(
            org_id=org_id,
            query_embedding=embedding,
            source=inputs.source,
            categories=inputs.categories,
            min_confidence=inputs.min_confidence,
            lifecycle_state=inputs.lifecycle_state,
            limit=limit,
        )
        vector_result = await backend.query(vector_sql, tuple(vector_params))
        vector_rows = [dict(row) for row in vector_result.rows]

        # FTS search
        fts_sql, fts_params = build_fts_search_query(
            org_id=org_id,
            query_text=inputs.query,
            source=inputs.source,
            categories=inputs.categories,
            min_confidence=inputs.min_confidence,
            lifecycle_state=inputs.lifecycle_state,
            limit=limit,
        )
        fts_result = await backend.query(fts_sql, tuple(fts_params))
        fts_rows = [dict(row) for row in fts_result.rows]

        # RRF fusion
        fused = rrf_fusion(vector_rows, fts_rows, limit=limit)

        # Update retrieval counts for returned propositions
        if fused:
            ids = [row["id"] for row in fused]
            placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
            await backend.execute(
                f"UPDATE knowledge_propositions SET retrieval_count = retrieval_count + 1 "
                f"WHERE id IN ({placeholders})",
                tuple(str(id_) for id_ in ids),
            )

        # Clean output rows (remove embeddings, keep content fields)
        output_rows = []
        for row in fused:
            output_rows.append(
                {
                    "id": str(row.get("id", "")),
                    "content": row.get("content", ""),
                    "confidence": row.get("confidence"),
                    "authority": row.get("authority"),
                    "relevance_score": row.get("relevance_score"),
                    "rrf_score": row.get("rrf_score"),
                }
            )

        columns = ["id", "content", "confidence", "authority", "relevance_score", "rrf_score"]
        return KnowledgeOutput(
            success=True,
            rows=output_rows,
            columns=columns,
            row_count=len(output_rows),
        )

    async def _op_store(
        self,
        inputs: KnowledgeInput,
        context: Execution,
        backend: Any,
    ) -> KnowledgeOutput:
        """Store a proposition with auto-computed embedding."""
        assert inputs.content is not None

        org_id = inputs.org_id or str(uuid.UUID(int=0))
        prop_id = str(uuid.uuid4())

        # Compute embedding for the content
        embedding, model_name, dimensions, _ = await compute_embedding(
            text=inputs.content,
            context=context,
            profile=inputs.embedding_profile,
        )

        # Find or create source if specified
        item_id: str | None = None
        if inputs.source:
            # Upsert source
            source_id = str(uuid.uuid4())
            await backend.execute(
                """
                INSERT INTO knowledge_sources (id, org_id, name, source_type)
                VALUES ($1::uuid, $2::uuid, $3, 'WORKFLOW')
                ON CONFLICT DO NOTHING
                """,
                (source_id, org_id, inputs.source),
            )
            # Get the actual source id (may already exist)
            source_result = await backend.query(
                "SELECT id FROM knowledge_sources WHERE org_id = $1::uuid AND name = $2 LIMIT 1",
                (org_id, inputs.source),
            )
            if source_result.rows:
                actual_source_id = str(source_result.rows[0]["id"])
                # Create an item for this store operation
                item_id = str(uuid.uuid4())
                await backend.execute(
                    """
                    INSERT INTO knowledge_items (id, org_id, source_id, title)
                    VALUES ($1::uuid, $2::uuid, $3::uuid, $4)
                    """,
                    (item_id, org_id, actual_source_id, f"workflow-store-{prop_id[:8]}"),
                )

        # Insert proposition with server-side tsvector computation
        await backend.execute(
            """
            INSERT INTO knowledge_propositions
                (id, org_id, item_id, content, embedding, search_vector,
                 authority, lifecycle_state, confidence,
                 embedding_model, embedding_dimensions, metadata_)
            VALUES
                ($1::uuid, $2::uuid, $3::uuid, $4, $5::vector, to_tsvector('english', $4),
                 $6, $7, $8,
                 $9, $10, $11::jsonb)
            """,
            (
                prop_id,
                org_id,
                item_id,
                inputs.content,
                str(embedding),
                "AGENT",
                LifecycleState.ACTIVE,
                inputs.confidence,
                model_name,
                dimensions,
                "{}",
            ),
        )

        return KnowledgeOutput(
            success=True,
            proposition_ids=[prop_id],
            stored_count=1,
        )

    async def _op_recall(
        self,
        inputs: KnowledgeInput,
        context: Execution,
        backend: Any,
    ) -> KnowledgeOutput:
        """Recall propositions by filter conditions."""
        org_id = inputs.org_id or str(uuid.UUID(int=0))
        limit = int(inputs.limit) if isinstance(inputs.limit, str) else inputs.limit

        params: list[Any] = []
        param_idx = 0

        def next_param(value: Any) -> str:
            nonlocal param_idx
            param_idx += 1
            params.append(value)
            return f"${param_idx}"

        where_clauses = [f"kp.org_id = {next_param(org_id)}::uuid"]

        # Apply where filters
        if inputs.where:
            if "source_name" in inputs.where:
                where_clauses.append(f"ks.name = {next_param(inputs.where['source_name'])}")
            if "lifecycle_state" in inputs.where:
                state = inputs.where["lifecycle_state"].upper()
                where_clauses.append(f"kp.lifecycle_state = {next_param(state)}")
            if "category" in inputs.where:
                where_clauses.append(
                    f"ks.category_ids && ARRAY[{next_param(inputs.where['category'])}]::uuid[]"
                )
        else:
            where_clauses.append(f"kp.lifecycle_state = {next_param(inputs.lifecycle_state)}")

        # Build JOIN if source/category filtering is needed
        needs_join = inputs.where and ("source_name" in inputs.where or "category" in inputs.where)
        join_clause = (
            "JOIN knowledge_items ki ON kp.item_id = ki.id "
            "JOIN knowledge_sources ks ON ki.source_id = ks.id"
            if needs_join
            else ""
        )

        # Order clause
        order_clause = "ORDER BY kp.created_at DESC"
        if inputs.order:
            order_parts = []
            for o in inputs.order:
                if ":" in o:
                    field, direction = o.split(":", 1)
                    direction = "DESC" if direction.lower() == "desc" else "ASC"
                else:
                    field = o
                    direction = "ASC"
                # Only allow safe column names
                safe_fields = {
                    "relevance_score",
                    "confidence",
                    "retrieval_count",
                    "created_at",
                    "updated_at",
                }
                if field in safe_fields:
                    order_parts.append(f"kp.{field} {direction}")
            if order_parts:
                order_clause = "ORDER BY " + ", ".join(order_parts)

        limit_param = next_param(limit)
        where_clause = " AND ".join(where_clauses)

        sql = f"""
            SELECT kp.id, kp.content, kp.confidence, kp.authority,
                   kp.lifecycle_state, kp.relevance_score, kp.retrieval_count
            FROM knowledge_propositions kp
            {join_clause}
            WHERE {where_clause}
            {order_clause}
            LIMIT {limit_param}
        """

        result = await backend.query(sql, tuple(params))
        rows = [
            {
                "id": str(row.get("id", "")),
                "content": row.get("content", ""),
                "confidence": row.get("confidence"),
                "authority": row.get("authority"),
                "lifecycle_state": row.get("lifecycle_state"),
                "relevance_score": row.get("relevance_score"),
                "retrieval_count": row.get("retrieval_count"),
            }
            for row in result.rows
        ]

        columns = [
            "id",
            "content",
            "confidence",
            "authority",
            "lifecycle_state",
            "relevance_score",
            "retrieval_count",
        ]
        return KnowledgeOutput(
            success=True,
            rows=rows,
            columns=columns,
            row_count=len(rows),
        )

    async def _op_forget(
        self,
        inputs: KnowledgeInput,
        context: Execution,
        backend: Any,
    ) -> KnowledgeOutput:
        """Transition propositions to ARCHIVED state."""
        # Normalize proposition_ids
        ids: list[str] = []
        if isinstance(inputs.proposition_ids, str):
            ids = [s.strip() for s in inputs.proposition_ids.split(",") if s.strip()]
        elif isinstance(inputs.proposition_ids, list):
            ids = inputs.proposition_ids
        else:
            return KnowledgeOutput(success=False, error="proposition_ids is required")

        if not ids:
            return KnowledgeOutput(success=True, archived_count=0, skipped_count=0)

        # Archive, but skip USER_VALIDATED (immune to archival)
        placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
        result = await backend.execute(
            f"UPDATE knowledge_propositions "
            f"SET lifecycle_state = '{LifecycleState.ARCHIVED}', "
            f"    updated_at = NOW() "
            f"WHERE id IN ({placeholders}) "
            f"  AND authority != 'USER_VALIDATED'",
            tuple(ids),
        )

        archived = result.affected_rows if result else 0
        skipped = len(ids) - archived

        return KnowledgeOutput(
            success=True,
            archived_count=archived,
            skipped_count=skipped,
        )

    async def _op_context(
        self,
        inputs: KnowledgeInput,
        context: Execution,
        backend: Any,
    ) -> KnowledgeOutput:
        """Token-budgeted, clean-content context assembly."""
        # First, run a search to get candidate propositions
        search_result = await self._op_search(inputs, context, backend)
        if not search_result.success:
            return search_result

        max_tokens = (
            int(inputs.max_tokens) if isinstance(inputs.max_tokens, str) else inputs.max_tokens
        )

        # Assemble context from search results
        context_text, prop_count, tokens_used = assemble_context(
            search_result.rows,
            max_tokens=max_tokens,
            diversity=inputs.diversity,
        )

        return KnowledgeOutput(
            success=True,
            context_text=context_text,
            proposition_count=prop_count,
            tokens_used=tokens_used,
        )
