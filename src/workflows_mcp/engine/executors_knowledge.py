"""Knowledge executor for workflow-native knowledge access.

Provides search, store, recall, forget, and context operations against
PostgreSQL knowledge tables (pgvector + tsvector). Enables clean YAML
interface for knowledge operations in workflows.

Pattern reference: executors_sql.py (SqlExecutor)
"""

from __future__ import annotations

import json
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
from .knowledge.search import (
    build_fts_search_query,
    build_vector_search_query,
    rrf_fusion,
)

logger = logging.getLogger(__name__)

# SECURITY: System user UUID for audit trail when no user context is available
# (e.g., localhost bypass operations, SYSTEM auth)
SYSTEM_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")

# SECURITY: Audit fail-closed configuration
# When true, audit logging failures cause operations to fail (compliance mode)
# When false (default for backward compatibility), audit failures are logged but operations continue
AUDIT_FAIL_CLOSED = os.getenv("AUDIT_FAIL_CLOSED", "false").lower() == "true"


# ===========================================================================
# Audit Trail Helpers
# ===========================================================================


def _get_audit_user_id(context: Execution) -> uuid.UUID:
    """Extract user_id from execution context for audit trail."""
    exec_ctx = context.execution_context
    if exec_ctx and exec_ctx.user_id:
        return exec_ctx.user_id
    return SYSTEM_USER_UUID


def _get_user_string_id(context: Execution) -> str | None:
    """Extract human-readable user identifier for audit metadata."""
    exec_ctx = context.execution_context
    if exec_ctx:
        # Prefer user_string_id for OS users, fallback to UUID string
        if exec_ctx.user_string_id:
            return exec_ctx.user_string_id
        if exec_ctx.user_id:
            return str(exec_ctx.user_id)
    return None


def _get_auth_method(context: Execution) -> str:
    """Extract auth_method from execution context."""
    exec_ctx = context.execution_context
    if exec_ctx and exec_ctx.auth_method:
        return exec_ctx.auth_method
    return "SYSTEM"


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
    path: str | None = Field(
        default=None,
        description=(
            "File path or identifier within the source "
            "(e.g. 'docs/architecture.md'). When provided alongside 'source', "
            "creates a provenance link to the source document. "
            "Required for document-derived propositions; omit for agent observations."
        ),
    )

    # --- Recall fields ---
    where: dict[str, Any] | None = Field(
        default=None,
        description="Filter conditions for recall/forget (key-value pairs)",
    )
    order: list[str] | None = Field(
        default=None,
        description="Order by fields (e.g., ['relevance_score:desc'])",
    )
    created_after: str | None = Field(
        default=None,
        description="Filter: propositions created after this ISO date (recall/forget)",
    )
    created_before: str | None = Field(
        default=None,
        description="Filter: propositions created before this ISO date (recall/forget)",
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
        if self.op == "forget" and not self.proposition_ids and not self.where:
            raise ValueError("'proposition_ids' or 'where' is required for op='forget'")
        if self.path is not None and not self.source:
            raise ValueError("'source' is required when 'path' is provided")
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
            KNOWLEDGE_DB_USER, KNOWLEDGE_DB_PASSWORD

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
    # Helpers
    # ------------------------------------------------------------------

    async def _resolve_categories(
        self,
        categories: list[str],
        backend: Any,
    ) -> list[str]:
        """Resolve category names or UUIDs to a list of UUID strings.

        For each entry in categories:
        - If it's a valid UUID, use as-is.
        - Otherwise, look up by (entity_type='category', name) in
          knowledge_entities. Auto-create the entity if missing.
        """
        resolved: list[str] = []
        for entry in categories:
            # Try parsing as UUID first
            try:
                uuid.UUID(entry)
                resolved.append(entry)
                continue
            except ValueError:
                pass

            # Name-based resolution: upsert into knowledge_entities
            result = await backend.query(
                """
                INSERT INTO knowledge_entities (id, entity_type, name)
                VALUES ($1::uuid, 'category', $2)
                ON CONFLICT (entity_type, name) DO UPDATE
                    SET name = EXCLUDED.name
                RETURNING id
                """,
                (str(uuid.uuid4()), entry),
            )
            if result.rows:
                resolved.append(str(result.rows[0]["id"]))
            else:
                logger.warning("Category resolution returned no rows for %r", entry)

        return resolved

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

        # Resolve category names to UUIDs if provided
        resolved_categories = None
        if inputs.categories:
            resolved_categories = await self._resolve_categories(inputs.categories, backend)

        # Compute query embedding
        embedding, _, _, _ = await compute_embedding(
            text=inputs.query,
            context=context,
            profile=inputs.embedding_profile,
        )

        # Vector search
        vector_sql, vector_params = build_vector_search_query(
            query_embedding=embedding,
            source=inputs.source,
            categories=resolved_categories,
            min_confidence=inputs.min_confidence,
            lifecycle_state=inputs.lifecycle_state,
            limit=limit,
        )
        vector_result = await backend.query(vector_sql, tuple(vector_params))
        vector_rows = [dict(row) for row in vector_result.rows]

        # FTS search
        fts_sql, fts_params = build_fts_search_query(
            query_text=inputs.query,
            source=inputs.source,
            categories=resolved_categories,
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
                    "item_path": row.get("item_path"),
                }
            )

        columns = [
            "id",
            "content",
            "confidence",
            "authority",
            "relevance_score",
            "rrf_score",
            "item_path",
        ]
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
        """Store a proposition with auto-computed embedding.

        Three modes based on (source, path) combination:

        - Neither source nor path: store proposition with item_id=NULL, empty metadata.
        - Source only (no path):   store proposition with item_id=NULL, source recorded
          in metadata for provenance. No knowledge_sources or knowledge_items rows are
          created because there is no backing document.
        - Source AND path:         upsert knowledge_sources + knowledge_items rows, then
          store proposition linked to that item for full provenance tracking.

        SECURITY: Records created_by and auth_method for audit trail.
        """
        assert inputs.content is not None

        prop_id = str(uuid.uuid4())

        # Resolve category names to UUIDs if provided
        category_ids: list[str] = []
        if inputs.categories:
            category_ids = await self._resolve_categories(inputs.categories, backend)

        # Compute embedding for the content
        embedding, model_name, dimensions, _ = await compute_embedding(
            text=inputs.content,
            context=context,
            profile=inputs.embedding_profile,
        )

        item_id: str | None = None
        prop_metadata: dict[str, str] = {}
        source_name: str | None = None
        source_type: str = "DOCUMENT"

        if inputs.source and inputs.path:
            # Full provenance: upsert source + item, link proposition
            source_name = inputs.source
            source_type = "DOCUMENT"
            source_result = await backend.query(
                """
                INSERT INTO knowledge_sources
                    (id, name, source_type, category_ids)
                VALUES ($1::uuid, $2, 'WORKFLOW', $3::uuid[])
                ON CONFLICT (name) DO UPDATE SET
                    category_ids = CASE
                        WHEN EXCLUDED.category_ids != '{}'
                            THEN EXCLUDED.category_ids
                        ELSE knowledge_sources.category_ids
                    END,
                    updated_at = NOW()
                RETURNING id
                """,
                (str(uuid.uuid4()), inputs.source, category_ids),
            )
            if source_result.rows:
                actual_source_id = str(source_result.rows[0]["id"])
                item_title = os.path.basename(inputs.path) or inputs.path
                item_result = await backend.query(
                    """
                    INSERT INTO knowledge_items (id, source_id, path, title)
                    VALUES ($1::uuid, $2::uuid, $3, $4)
                    ON CONFLICT (source_id, path) DO UPDATE SET
                        title = EXCLUDED.title,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (str(uuid.uuid4()), actual_source_id, inputs.path, item_title),
                )
                if item_result.rows:
                    item_id = str(item_result.rows[0]["id"])

        elif inputs.source:
            # Source without path: record source name for provenance only;
            # do NOT create knowledge_sources or knowledge_items rows.
            source_name = inputs.source
            source_type = "TOOL"
            prop_metadata = {"source": inputs.source}

        # SECURITY: Get user attribution for audit trail
        # Always populate created_by/auth_method - fallback to SYSTEM when no user context available
        created_by = _get_audit_user_id(context)
        auth_method = _get_auth_method(context)
        user_string = _get_user_string_id(context)

        # Build metadata JSON string with auth info
        if auth_method and isinstance(auth_method, str):
            prop_metadata["auth_method"] = auth_method
        if user_string:
            prop_metadata["user_identifier"] = user_string
        metadata_json = json.dumps(prop_metadata)

        # Insert proposition with server-side tsvector computation and audit trail
        await backend.execute(
            """
            INSERT INTO knowledge_propositions
                (id, item_id, content, embedding, search_vector,
                 authority, lifecycle_state, confidence,
                 embedding_model, embedding_dimensions, metadata,
                 created_by, auth_method, source_name, source_type)
            VALUES
                ($1::uuid, $2::uuid, $3, $4::vector,
                 to_tsvector('english', $3),
                 $5, $6, $7,
                 $8, $9, $10::jsonb,
                 $11::uuid, $12, $13, $14)
            """,
            (
                prop_id,
                item_id,
                inputs.content,
                str(embedding),
                "AGENT",
                LifecycleState.ACTIVE.value,
                inputs.confidence,
                model_name,
                dimensions,
                metadata_json,
                str(created_by),
                auth_method,
                source_name,
                source_type,
            ),
        )

        # SECURITY: Log to audit table
        await self._log_audit_entry(
            backend=backend,
            proposition_id=prop_id,
            action="CREATED",
            performed_by=created_by,
            auth_method=auth_method,
            user_string=user_string,
            metadata={"source": inputs.source, "path": inputs.path},
        )

        return KnowledgeOutput(
            success=True,
            proposition_ids=[prop_id],
            stored_count=1,
        )

    async def _log_audit_entry(
        self,
        backend: Any,
        proposition_id: str,
        action: str,
        performed_by: uuid.UUID,
        auth_method: str,
        user_string: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit entry for a proposition lifecycle event.

        SECURITY: Records all significant changes for compliance auditing.
        """
        # Include user_string in metadata for human-readable audit trail
        metadata_with_user = metadata or {}
        if user_string:
            metadata_with_user["user_identifier"] = user_string

        try:
            await backend.execute(
                """
                INSERT INTO knowledge_proposition_audits
                    (proposition_id, action, performed_by, auth_method, metadata)
                VALUES
                    ($1::uuid, $2, $3::uuid, $4, $5::jsonb)
                """,
                (
                    proposition_id,
                    action,
                    str(performed_by),
                    auth_method,
                    json.dumps(metadata_with_user),
                ),
            )
        except Exception as e:
            # SECURITY: Audit logging failure handling
            # When AUDIT_FAIL_CLOSED=true, operation fails (compliance mode)
            # When AUDIT_FAIL_CLOSED=false (default), log error but continue
            logger.error(
                f"CRITICAL: Failed to log audit entry for proposition {proposition_id}: {e}",
                extra={
                    "event": "knowledge.audit.failure",
                    "proposition_id": proposition_id,
                    "action": action,
                },
            )
            if AUDIT_FAIL_CLOSED:
                raise RuntimeError(f"Audit logging failed: {e}") from e

    async def _build_where_clause(
        self, inputs: KnowledgeInput, backend: Any
    ) -> tuple[str, str, list[Any]]:
        """Build WHERE and JOIN clauses from recall/forget filters.

        Resolves category names to UUIDs if needed.
        Returns (where_clause, join_clause, params).

        Source filtering uses the denormalized source_name column for consistent,
        performant queries across both document-derived and agent observation
        propositions.
        """
        params: list[Any] = []
        param_idx = 0

        def next_param(value: Any) -> str:
            nonlocal param_idx
            param_idx += 1
            params.append(value)
            return f"${param_idx}"

        where_clauses: list[str] = []
        needs_join = False

        # Handle source filter (top-level inputs.source or where.source_name)
        source_filter = None
        if inputs.source:
            source_filter = inputs.source
        elif inputs.where and "source_name" in inputs.where:
            source_filter = inputs.where["source_name"]

        if source_filter:
            if isinstance(source_filter, str) and source_filter.endswith("*"):
                prefix_param = next_param(source_filter[:-1] + "%")
                # Use source_name column directly (denormalized for performance)
                where_clauses.append(f"kp.source_name LIKE {prefix_param}")
            else:
                source_param = next_param(source_filter)
                # Use source_name column directly (denormalized for performance)
                where_clauses.append(f"kp.source_name = {source_param}")

        # Handle category filter (requires JOIN to knowledge_sources)
        if inputs.where and "category" in inputs.where:
            needs_join = True
            cat_value = inputs.where["category"]
            # Resolve name to UUID if needed
            resolved = await self._resolve_categories([cat_value], backend)
            cat_uuid = resolved[0] if resolved else cat_value
            where_clauses.append(f"ks.category_ids && ARRAY[{next_param(cat_uuid)}]::uuid[]")

        # Handle lifecycle_state filter
        if inputs.where and "lifecycle_state" in inputs.where:
            state = inputs.where["lifecycle_state"].upper()
            where_clauses.append(f"kp.lifecycle_state = {next_param(state)}")
        else:
            # Always apply default lifecycle_state when not explicitly set in where
            where_clauses.append(f"kp.lifecycle_state = {next_param(inputs.lifecycle_state)}")

        # Handle min_confidence filter
        if inputs.where and "min_confidence" in inputs.where:
            where_clauses.append(
                f"kp.confidence >= {next_param(float(inputs.where['min_confidence']))}"
            )

        # Handle created_by filter
        if inputs.where and "created_by" in inputs.where:
            where_clauses.append(f"kp.created_by = {next_param(inputs.where['created_by'])}::uuid")

        # Handle auth_method filter
        if inputs.where and "auth_method" in inputs.where:
            where_clauses.append(f"kp.auth_method = {next_param(inputs.where['auth_method'])}")

        # Date range filters (top-level fields, usable with or without where)
        if inputs.created_after:
            where_clauses.append(
                f"kp.created_at >= {next_param(inputs.created_after)}::timestamptz"
            )
        if inputs.created_before:
            where_clauses.append(
                f"kp.created_at <= {next_param(inputs.created_before)}::timestamptz"
            )

        join_clause = (
            "JOIN knowledge_items ki ON kp.item_id = ki.id "
            "JOIN knowledge_sources ks ON ki.source_id = ks.id"
            if needs_join
            else ""
        )

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
        return where_sql, join_clause, params

    async def _op_recall(
        self,
        inputs: KnowledgeInput,
        context: Execution,
        backend: Any,
    ) -> KnowledgeOutput:
        """Recall propositions by filter conditions.

        SECURITY: Supports filtering by created_by and returns user attribution.
        """
        limit = int(inputs.limit) if isinstance(inputs.limit, str) else inputs.limit

        where_sql, join_clause, params = await self._build_where_clause(inputs, backend)

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

        param_idx = len(params)
        param_idx += 1
        params.append(limit)
        limit_param = f"${param_idx}"

        # SECURITY: Include created_by and auth_method in SELECT
        sql = f"""
            SELECT kp.id, kp.content, kp.confidence, kp.authority,
                   kp.lifecycle_state, kp.relevance_score, kp.retrieval_count,
                   ki_ip.path AS item_path,
                   kp.created_by, kp.auth_method
            FROM knowledge_propositions kp
            LEFT JOIN knowledge_items ki_ip ON kp.item_id = ki_ip.id
            {join_clause}
            WHERE {where_sql}
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
                "item_path": row.get("item_path"),
                "created_by": str(row.get("created_by")) if row.get("created_by") else None,
                "auth_method": row.get("auth_method"),
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
            "item_path",
            "created_by",
            "auth_method",
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
        """Transition propositions to ARCHIVED state by IDs or filter.

        SECURITY: Records archived_by and logs to audit table.
        """
        # Get user attribution for audit trail
        archived_by = _get_audit_user_id(context)
        auth_method = _get_auth_method(context)
        user_string = _get_user_string_id(context)

        # SECURITY: Archive reason - keep clean, archived_by stored in separate column
        archive_reason = inputs.reason or "user_action"

        # Path 1: Archive by explicit IDs
        if inputs.proposition_ids:
            ids: list[str] = []
            if isinstance(inputs.proposition_ids, str):
                ids = [s.strip() for s in inputs.proposition_ids.split(",") if s.strip()]
            elif isinstance(inputs.proposition_ids, list):
                ids = inputs.proposition_ids

            if not ids:
                return KnowledgeOutput(success=True, archived_count=0, skipped_count=0)

            placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
            # SECURITY: Update with archived_by and archive_reason
            update_sql = f"""
                UPDATE knowledge_propositions
                SET lifecycle_state = '{LifecycleState.ARCHIVED.value}',
                    archived_by = ${len(ids) + 1}::uuid,
                    archive_reason = ${len(ids) + 2}
                WHERE id IN ({placeholders})
                  AND authority != 'USER_VALIDATED'
                RETURNING id
            """
            result = await backend.query(
                update_sql,
                tuple(ids) + (str(archived_by) if archived_by else None, archive_reason),
            )

            archived = len(result.rows) if result and result.rows else 0
            skipped = len(ids) - archived

            # SECURITY: Log to audit table for each archived proposition
            if archived > 0:
                archived_ids = [row["id"] for row in result.rows]
                for prop_id in archived_ids:
                    await self._log_audit_entry(
                        backend=backend,
                        proposition_id=prop_id,
                        action="ARCHIVED",
                        performed_by=archived_by,
                        auth_method=auth_method,
                        user_string=user_string,
                        metadata={"reason": inputs.reason, "method": "explicit_ids"},
                    )

            return KnowledgeOutput(
                success=True,
                archived_count=archived,
                skipped_count=skipped,
            )

        # Path 2: Archive by filter
        where_sql, join_clause, params = await self._build_where_clause(inputs, backend)

        # Count total matching (for skipped_count calculation)
        count_sql = f"""
            SELECT COUNT(*) AS total FROM knowledge_propositions kp
            {join_clause}
            WHERE {where_sql}
        """
        count_result = await backend.query(count_sql, tuple(params))
        total = count_result.rows[0]["total"] if count_result.rows else 0

        # Archive with USER_VALIDATED immunity
        # Use subquery to handle JOIN-based filters in UPDATE
        if join_clause:
            param_offset = len(params)
            update_sql = f"""
                UPDATE knowledge_propositions
                SET lifecycle_state = '{LifecycleState.ARCHIVED.value}',
                    archived_by = ${param_offset + 1}::uuid,
                    archive_reason = ${param_offset + 2}
                WHERE id IN (
                    SELECT kp.id FROM knowledge_propositions kp
                    {join_clause}
                    WHERE {where_sql}
                )
                AND authority != 'USER_VALIDATED'
                RETURNING id
            """
            params.extend([str(archived_by) if archived_by else None, archive_reason])
        else:
            param_offset = len(params)
            update_sql = f"""
                UPDATE knowledge_propositions kp
                SET lifecycle_state = '{LifecycleState.ARCHIVED.value}',
                    archived_by = ${param_offset + 1}::uuid,
                    archive_reason = ${param_offset + 2}
                WHERE {where_sql}
                  AND authority != 'USER_VALIDATED'
                RETURNING id
            """
            params.extend([str(archived_by) if archived_by else None, archive_reason])

        result = await backend.query(update_sql, tuple(params))
        archived = len(result.rows) if result and result.rows else 0
        skipped = total - archived

        # SECURITY: Log to audit table for each archived proposition
        if archived > 0:
            archived_ids = [row["id"] for row in result.rows]
            for prop_id in archived_ids:
                await self._log_audit_entry(
                    backend=backend,
                    proposition_id=prop_id,
                    action="ARCHIVED",
                    performed_by=archived_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={"reason": inputs.reason, "method": "filter"},
                )

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
