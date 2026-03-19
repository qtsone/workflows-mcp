"""Knowledge MCP tools — conditionally registered at startup.

These tools are only available when a knowledge database is configured
(KNOWLEDGE_DB_HOST environment variable) and reachable at server boot.

Registration happens via ``register_knowledge_tools(mcp_server)`` called
from the server lifespan, NOT via top-level ``@mcp.tool()`` decorators.
"""

import json
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from .context import AppContextType


def _json_response(data: dict[str, Any]) -> CallToolResult:
    """Build a CallToolResult with compact text and structured content."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, separators=(",", ":")))],
        structuredContent=data,
    )


def _create_knowledge_execution(ctx: AppContextType) -> Any:
    """Create a minimal Execution with ExecutionContext for knowledge operations."""
    from .engine.execution import Execution

    app_ctx = ctx.request_context.lifespan_context
    exec_context = app_ctx.create_execution_context()
    execution = Execution()
    execution.set_execution_context(exec_context)
    return execution


def register_knowledge_tools(mcp_server: FastMCP) -> None:
    """Register knowledge MCP tools on the server instance.

    Called from ``app_lifespan()`` only when the knowledge DB is
    configured and the schema has been successfully initialized.
    """

    @mcp_server.tool(
        annotations=ToolAnnotations(
            title="Search Knowledge",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def search_knowledge(
        query: Annotated[
            str,
            Field(description="Search query text for semantic + full-text hybrid search"),
        ],
        source: Annotated[
            str | None,
            Field(
                description="Filter by source name (exact or prefix with * suffix, e.g. 'docs:*')",
                default=None,
            ),
        ],
        categories: Annotated[
            list[str] | None,
            Field(
                description="Filter by category UUIDs",
                default=None,
            ),
        ],
        min_confidence: Annotated[
            float,
            Field(
                description="Minimum confidence threshold for results",
                default=0.3,
                ge=0.0,
                le=1.0,
            ),
        ],
        limit: Annotated[
            int,
            Field(
                description="Maximum number of results to return",
                default=10,
                ge=1,
                le=100,
            ),
        ],
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """
        Search the knowledge base using hybrid retrieval (vector + full-text + RRF fusion).

        WHEN TO USE: To find relevant facts, propositions, or documentation from the
        organization's knowledge base. Returns ranked results combining semantic similarity
        and keyword matching.

        PARAMETERS:
        - query: What to search for (natural language)
        - source: Optional source filter (exact name or prefix like 'docs:*')
        - categories: Optional category UUID filter
        - min_confidence: Minimum confidence score (0.0-1.0, default 0.3)
        - limit: Max results (default 10)

        RETURNS: {rows: [{id, content, confidence, authority, ...}], row_count: N}

        SEE ALSO: knowledge_context (token-budgeted assembly), recall_knowledge (filter-based)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="search",
            query=query,
            source=source,
            categories=categories,
            min_confidence=min_confidence,
            limit=limit,
        )
        result = await executor.execute(inputs, context=execution)

        if not result.success:
            return _json_response({"status": "failure", "error": result.error})

        return _json_response(
            {
                "rows": result.rows,
                "columns": result.columns,
                "row_count": result.row_count,
            }
        )

    @mcp_server.tool(
        annotations=ToolAnnotations(
            title="Store Knowledge",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    async def store_knowledge(
        content: Annotated[
            str,
            Field(
                description=("The fact or proposition to store (atomic, self-contained statement)")
            ),
        ],
        source: Annotated[
            str | None,
            Field(
                description=(
                    "Source name for provenance tracking "
                    "(e.g. 'deployment-workflow', 'incident-analysis')"
                ),
                default=None,
            ),
        ],
        path: Annotated[
            str | None,
            Field(
                description=(
                    "File path or identifier within the source "
                    "(e.g. 'docs/architecture.md'). Provide alongside 'source' to "
                    "link this fact to its source document for provenance tracking."
                ),
                default=None,
            ),
        ],
        confidence: Annotated[
            float,
            Field(
                description="Confidence score for the stored fact (0.0-1.0)",
                default=0.8,
                ge=0.0,
                le=1.0,
            ),
        ],
        categories: Annotated[
            list[str] | None,
            Field(
                description="Category UUIDs to associate with the stored fact",
                default=None,
            ),
        ],
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """
        Store a new fact in the knowledge base with auto-computed embedding.

        WHEN TO USE: To persist a new finding, insight, or learned fact that should be
        retrievable later. The embedding is computed automatically for semantic search.
        Use after discovering important information during analysis or workflow execution.

        PARAMETERS:
        - content: The fact to store (should be atomic and self-contained)
        - source: Provenance label (e.g. 'my-analysis', 'deploy-workflow')
        - path: Optional file path within the source (e.g. 'docs/architecture.md').
          Provide alongside 'source' to link the fact to its source document.
          Omit for agent observations not tied to a specific file.
        - confidence: How confident you are in this fact (default 0.8)
        - categories: Optional category UUIDs

        RETURNS: {proposition_ids: [uuid], stored_count: 1}

        SEE ALSO: search_knowledge (retrieve stored facts), forget_knowledge (archive facts)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="store",
            content=content,
            source=source,
            path=path,
            confidence=confidence,
            categories=categories,
        )
        result = await executor.execute(inputs, context=execution)

        if not result.success:
            return _json_response({"status": "failure", "error": result.error})

        return _json_response(
            {
                "proposition_ids": result.proposition_ids,
                "stored_count": result.stored_count,
            }
        )

    @mcp_server.tool(
        annotations=ToolAnnotations(
            title="Recall Knowledge",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def recall_knowledge(
        source: Annotated[
            str | None,
            Field(
                description="Filter by source name (exact or prefix with *, e.g. 'workflow:*')",
                default=None,
            ),
        ],
        categories: Annotated[
            list[str] | None,
            Field(
                description="Filter by category UUIDs",
                default=None,
            ),
        ],
        lifecycle_state: Annotated[
            str,
            Field(
                description=(
                    "Filter by lifecycle state: ACTIVE, QUARANTINED, FLAGGED, or ARCHIVED"
                ),
                default="ACTIVE",
            ),
        ],
        min_confidence: Annotated[
            float | None,
            Field(
                description="Minimum confidence threshold",
                default=None,
                ge=0.0,
                le=1.0,
            ),
        ],
        limit: Annotated[
            int,
            Field(
                description="Maximum number of results",
                default=10,
                ge=1,
                le=100,
            ),
        ],
        order: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Order by fields, e.g. ['relevance_score:desc', 'created_at:asc']. "
                    "Allowed: relevance_score, confidence, retrieval_count, "
                    "created_at, updated_at"
                ),
                default=None,
            ),
        ],
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """
        Retrieve knowledge propositions by filter conditions (no semantic search).

        WHEN TO USE: To list or browse facts using metadata filters rather than
        semantic search. Useful for inventory, audit, or filter-based retrieval
        (e.g. 'all facts from source X', 'all flagged propositions').

        PARAMETERS:
        - source: Source name filter (exact or prefix with *)
        - categories: Category UUID filter
        - lifecycle_state: ACTIVE (default), QUARANTINED, FLAGGED, or ARCHIVED
        - min_confidence: Minimum confidence threshold
        - limit: Max results (default 10)
        - order: Sort fields (e.g. ['confidence:desc'])

        RETURNS: {rows: [{id, content, confidence, authority, lifecycle_state, ...}], row_count: N}

        SEE ALSO: search_knowledge (semantic search), forget_knowledge (archive facts)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()

        where: dict[str, Any] = {}
        if source:
            where["source_name"] = source
        if lifecycle_state and lifecycle_state != "ACTIVE":
            where["lifecycle_state"] = lifecycle_state
        if min_confidence is not None:
            where["min_confidence"] = min_confidence

        inputs = KnowledgeInput(
            op="recall",
            where=where if where else None,
            categories=categories,
            lifecycle_state=lifecycle_state,
            limit=limit,
            order=order,
        )
        result = await executor.execute(inputs, context=execution)

        if not result.success:
            return _json_response({"status": "failure", "error": result.error})

        return _json_response(
            {
                "rows": result.rows,
                "columns": result.columns,
                "row_count": result.row_count,
            }
        )

    @mcp_server.tool(
        annotations=ToolAnnotations(
            title="Forget Knowledge",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def forget_knowledge(
        proposition_ids: Annotated[
            list[str],
            Field(description="UUIDs of propositions to archive"),
        ],
        reason: Annotated[
            str | None,
            Field(
                description="Reason for archiving (for audit trail)",
                default=None,
            ),
        ],
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """
        Archive propositions by transitioning them to ARCHIVED lifecycle state.

        WHEN TO USE: To remove outdated, incorrect, or superseded facts from active
        search results. Archived propositions are not deleted — they remain in the
        database but are excluded from search. USER_VALIDATED propositions are immune
        and will be skipped.

        PARAMETERS:
        - proposition_ids: List of proposition UUIDs to archive
        - reason: Optional reason for the archival (audit trail)

        RETURNS: {archived_count: N, skipped_count: M}

        SEE ALSO: recall_knowledge (find propositions to archive), search_knowledge (verify removal)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="forget",
            proposition_ids=proposition_ids,
            reason=reason,
        )
        result = await executor.execute(inputs, context=execution)

        if not result.success:
            return _json_response({"status": "failure", "error": result.error})

        return _json_response(
            {
                "archived_count": result.archived_count,
                "skipped_count": result.skipped_count,
            }
        )

    @mcp_server.tool(
        annotations=ToolAnnotations(
            title="Knowledge Context",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def knowledge_context(
        query: Annotated[
            str,
            Field(description="Query to find relevant knowledge for context assembly"),
        ],
        source: Annotated[
            str | None,
            Field(
                description="Filter by source name (exact or prefix with *)",
                default=None,
            ),
        ],
        categories: Annotated[
            list[str] | None,
            Field(
                description="Filter by category UUIDs",
                default=None,
            ),
        ],
        max_tokens: Annotated[
            int,
            Field(
                description="Maximum token budget for the assembled context",
                default=4000,
                ge=100,
                le=128000,
            ),
        ],
        diversity: Annotated[
            bool,
            Field(
                description="Use MMR to spread results across different entities/sources",
                default=False,
            ),
        ],
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """
        Assemble token-budgeted context from knowledge base for LLM prompts.

        WHEN TO USE: Before making a complex decision or generating content that would
        benefit from organizational knowledge. Returns clean content text (no metadata)
        within a token budget, ready for direct injection into system/user prompts.

        PARAMETERS:
        - query: What topic to gather context about
        - source: Optional source filter
        - categories: Optional category filter
        - max_tokens: Token budget (default 4000, ensures fit within LLM context window)
        - diversity: If true, uses MMR to spread across different topics

        RETURNS: {context_text: "...", proposition_count: N, tokens_used: M}

        SEE ALSO: search_knowledge (raw search results), store_knowledge (persist new findings)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="context",
            query=query,
            source=source,
            categories=categories,
            max_tokens=max_tokens,
            diversity=diversity,
        )
        result = await executor.execute(inputs, context=execution)

        if not result.success:
            return _json_response({"status": "failure", "error": result.error})

        return _json_response(
            {
                "context_text": result.context_text,
                "proposition_count": result.proposition_count,
                "tokens_used": result.tokens_used,
            }
        )
