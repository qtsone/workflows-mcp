"""Knowledge MCP tools — conditionally registered at startup.

These tools are only available when a knowledge database is configured
(KNOWLEDGE_DB_HOST environment variable) and reachable at server boot.

Registration happens via ``register_knowledge_tools(mcp_server)`` called
from the server lifespan, NOT via top-level ``@mcp.tool()`` decorators.
"""

import json
import uuid
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


def _get_standalone_user_context() -> tuple[uuid.UUID | None, str | None, str]:
    """OS/env user detection for standalone (non-platform) mode.

    Priority:
    1. WORKFLOWS_USER_ID env var (UUID or string)
    2. WORKFLOWS_USER, MCP_USER_ID, USER, USERNAME, LOGNAME env vars
    3. getpass.getuser()
    4. SYSTEM fallback
    """
    import getpass
    import os

    from .engine.executors_knowledge import SYSTEM_USER_UUID

    for env_var in [
        "WORKFLOWS_USER_ID",
        "WORKFLOWS_USER",
        "MCP_USER_ID",
        "USER",
        "USERNAME",
        "LOGNAME",
    ]:  # noqa: E501
        if os_user := os.environ.get(env_var):
            os_user = os_user.strip()
            try:
                return (uuid.UUID(os_user), os_user, "ENV_UUID")
            except ValueError:
                det_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"workflows-user:{os_user}")
                return (det_uuid, os_user, "OS_USER")

    try:
        os_user = getpass.getuser()
        det_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"workflows-user:{os_user}")
        return (det_uuid, os_user, "OS_USER")
    except Exception:
        pass

    return (SYSTEM_USER_UUID, "system", "SYSTEM")


def _create_knowledge_execution(ctx: AppContextType) -> Any:
    """Create a minimal Execution with ExecutionContext for knowledge operations.

    Includes user attribution from the current request context for audit trail support.
    Supports both platform mode (via AppContext.get_user_context callback) and
    open-source mode (OS/env-based user detection).
    """
    from .engine.execution import Execution

    app_ctx = ctx.request_context.lifespan_context

    # Use platform callback if available, otherwise fall back to OS/env detection
    if app_ctx.get_user_context:
        user_uuid, user_string, auth_method = app_ctx.get_user_context()
    else:
        user_uuid, user_string, auth_method = _get_standalone_user_context()

    exec_context = app_ctx.create_execution_context(
        user_id=user_uuid,
        auth_method=auth_method,
    )

    # Store user_string in execution context for audit metadata
    if user_string:
        exec_context.user_string_id = user_string

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
        as_of: Annotated[
            str | None,
            Field(
                description=(
                    "Point-in-time filter (ISO datetime). Returns only propositions valid at this"
                    " timestamp based on valid_from/valid_to windows"
                ),
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
        - as_of: Optional ISO datetime for temporal validity filtering
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
            as_of=as_of,
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
        valid_from: Annotated[
            str | None,
            Field(
                description=(
                    "World-truth start datetime in ISO 8601 format. "
                    "If omitted, validity has an open start."
                ),
                default=None,
            ),
        ],
        valid_to: Annotated[
            str | None,
            Field(
                description=(
                    "World-truth end datetime in ISO 8601 format. "
                    "If omitted, validity has an open end."
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
        authority: Annotated[
            str,
            Field(
                description=(
                    "Authority level for this fact. "
                    "EXTRACTED: derived from a document or source file. "
                    "AGENT: inferred by an AI agent (default). "
                    "COMMUNITY_SUMMARY: aggregated or summarized insight. "
                    "USER_VALIDATED: human-reviewed, immune to archiving."
                ),
                default="AGENT",
            ),
        ],
        lifecycle_state: Annotated[
            str,
            Field(
                description=(
                    "Initial lifecycle state. ACTIVE (default): immediately searchable. "
                    "QUARANTINED: stored but excluded from search, pending review."
                ),
                default="ACTIVE",
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
        - valid_from: Optional ISO datetime when proposition becomes true
        - valid_to: Optional ISO datetime when proposition stops being true
        - confidence: How confident you are in this fact (default 0.8)
        - categories: Optional category UUIDs
        - authority: Who vouches for this fact (default AGENT). Use EXTRACTED for
          document-derived facts, USER_VALIDATED to grant archive immunity.
        - lifecycle_state: Initial state (default ACTIVE). Use QUARANTINED for
          uncertain facts pending human review.

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
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            categories=categories,
            source_type="TOOL",
            authority=authority,
            store_lifecycle_state=lifecycle_state,
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
        as_of: Annotated[
            str | None,
            Field(
                description=(
                    "Point-in-time filter (ISO datetime). Returns only propositions valid at this"
                    " timestamp based on valid_from/valid_to windows"
                ),
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
                    "Order by fields, e.g. ['confidence:desc', 'created_at:asc']. "
                    "Allowed: confidence, retrieval_count, created_at, updated_at"
                ),
                default=None,
            ),
        ],
        created_by: Annotated[
            str | None,
            Field(
                description="Filter by user ID who created the proposition (UUID)",
                default=None,
            ),
        ],
        auth_method: Annotated[
            str | None,
            Field(
                description="Filter by authentication method used (PAT, SSO, SYSTEM)",
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
        - as_of: Optional ISO datetime for temporal validity filtering
        - lifecycle_state: ACTIVE (default), QUARANTINED, FLAGGED, or ARCHIVED
        - min_confidence: Minimum confidence threshold
        - limit: Max results (default 10)
        - order: Sort fields (e.g. ['confidence:desc'])
        - created_by: Filter by user ID who created the proposition (UUID)
        - auth_method: Filter by authentication method (PAT, SSO, SYSTEM)

        RETURNS: {rows: [{id, content, confidence, authority, lifecycle_state,
        created_by, auth_method, ...}], row_count: N}

        SEE ALSO: search_knowledge (semantic search), forget_knowledge (archive facts)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()

        where: dict[str, Any] = {}
        if created_by:
            where["created_by"] = created_by
        if auth_method:
            where["auth_method"] = auth_method

        inputs = KnowledgeInput(
            op="recall",
            source=source,
            min_confidence=min_confidence,
            where=where if where else None,
            categories=categories,
            as_of=as_of,
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
            list[str] | None,
            Field(description="UUIDs of propositions to archive", default=None),
        ],
        source: Annotated[
            str | None,
            Field(
                description=(
                    "Archive all propositions from this source "
                    "(exact name or prefix with *, e.g. 'session:*'). "
                    "At least one of proposition_ids or source must be provided."
                ),
                default=None,
            ),
        ],
        created_before: Annotated[
            str | None,
            Field(
                description="Archive propositions created before this ISO 8601 datetime",
                default=None,
            ),
        ],
        created_after: Annotated[
            str | None,
            Field(
                description="Archive propositions created after this ISO 8601 datetime",
                default=None,
            ),
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

        Supports two archiving modes:
        1. By IDs — provide proposition_ids to archive specific propositions.
        2. By filter — provide source, created_before, and/or created_after to archive
           in bulk. Useful for clearing all facts from a session or before a given date.

        PARAMETERS:
        - proposition_ids: List of proposition UUIDs to archive (mode 1)
        - source: Archive all propositions from this source, e.g. 'session:abc' or 'docs:*' (mode 2)
        - created_before: Archive propositions created before this ISO datetime (mode 2)
        - created_after: Archive propositions created after this ISO datetime (mode 2)
        - reason: Optional reason for the archival (audit trail)

        At least one of proposition_ids or source must be provided.

        RETURNS: {archived_count: N, skipped_count: M}

        SEE ALSO: recall_knowledge (find propositions to archive), search_knowledge (verify removal)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        if not proposition_ids and not source and not created_before and not created_after:
            return _json_response(
                {
                    "status": "failure",
                    "error": (
                        "At least one of proposition_ids, source, created_before, "
                        "or created_after is required"
                    ),
                }
            )

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="forget",
            proposition_ids=proposition_ids,
            source=source,
            created_before=created_before,
            created_after=created_after,
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
            title="Validate Knowledge",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def validate_knowledge(
        proposition_ids: Annotated[
            list[str],
            Field(description="UUIDs of propositions to promote to USER_VALIDATED authority"),
        ],
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """
        Promote propositions to USER_VALIDATED authority, granting archive immunity.

        WHEN TO USE: After a human has reviewed and confirmed a fact as permanently
        trustworthy. USER_VALIDATED propositions are immune to forget_knowledge — they
        cannot be archived by automated cleanup or bulk operations.

        This is an in-place authority update, NOT a forget+store cycle. The original
        proposition UUID, created_by, created_at, and category associations are all
        preserved. Only the authority field is changed.

        PARAMETERS:
        - proposition_ids: List of proposition UUIDs to validate

        RETURNS: {validated_count: N}

        SEE ALSO: recall_knowledge (find propositions to validate), forget_knowledge (skips
        USER_VALIDATED propositions automatically)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="validate",
            proposition_ids=proposition_ids,
        )
        result = await executor.execute(inputs, context=execution)

        if not result.success:
            return _json_response({"status": "failure", "error": result.error})

        return _json_response({"validated_count": result.validated_count})

    @mcp_server.tool(
        annotations=ToolAnnotations(
            title="Invalidate Knowledge",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def invalidate_knowledge(
        proposition_ids: Annotated[
            list[str],
            Field(description="UUIDs of USER_VALIDATED propositions to revoke"),
        ],
        reason: Annotated[
            str | None,
            Field(
                description="Reason for revoking validation (for audit trail)",
                default=None,
            ),
        ],
        *,
        ctx: AppContextType,
    ) -> CallToolResult:
        """
        Revoke USER_VALIDATED authority, demoting propositions back to AGENT trust level.

        WHEN TO USE: When a previously human-validated fact is no longer true or has
        been superseded. After invalidation the proposition loses archive immunity and
        can be removed with forget_knowledge. The content and full audit trail are
        preserved — the audit log records that this was once USER_VALIDATED and when
        and why that was revoked.

        Only propositions currently holding USER_VALIDATED authority are affected.
        Propositions with any other authority are silently skipped (idempotent).

        PARAMETERS:
        - proposition_ids: List of proposition UUIDs to invalidate
        - reason: Optional reason for revocation (stored in audit trail)

        RETURNS: {invalidated_count: N}

        SEE ALSO: validate_knowledge (grant immunity), forget_knowledge (archive after invalidation)
        """
        from .engine.executors_knowledge import KnowledgeExecutor, KnowledgeInput

        execution = _create_knowledge_execution(ctx)
        executor = KnowledgeExecutor()
        inputs = KnowledgeInput(
            op="invalidate",
            proposition_ids=proposition_ids,
            reason=reason,
        )
        result = await executor.execute(inputs, context=execution)

        if not result.success:
            return _json_response({"status": "failure", "error": result.error})

        return _json_response({"invalidated_count": result.invalidated_count})

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
        as_of: Annotated[
            str | None,
            Field(
                description=(
                    "Point-in-time filter (ISO datetime). Uses only propositions valid at this"
                    " timestamp before token-budget assembly"
                ),
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
                default=True,
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
        - as_of: Optional ISO datetime for temporal validity filtering
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
            as_of=as_of,
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
