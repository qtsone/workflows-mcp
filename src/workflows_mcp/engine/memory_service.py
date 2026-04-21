"""Shared memory service — single orchestration layer for query and manage operations.

Both MCP tools (query_memory, manage_memory) and the Memory workflow block
delegate through this service. All lifecycle invariants are enforced here
regardless of the caller.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution import Execution
from .executors_llm import compute_embedding
from .knowledge.constants import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_CONFIDENCE,
    Authority,
    LifecycleState,
)
from .knowledge.context import assemble_context
from .knowledge.graph import (
    graph_neighbors,
    graph_path,
    graph_stats,
    graph_traverse,
)
from .knowledge.search import room_scoped_search

logger = logging.getLogger(__name__)

# SECURITY: System user UUID for audit trail when no user context is available
SYSTEM_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")

# SECURITY: Audit fail-closed configuration
AUDIT_FAIL_CLOSED = os.getenv("AUDIT_FAIL_CLOSED", "false").lower() == "true"


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


def _coerce_iso_datetime(value: str | None, field_name: str) -> datetime | None:
    """Parse ISO datetime strings (including trailing Z) to datetime for DB bindings."""
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as e:
        raise ValueError(f"'{field_name}' must be a valid ISO datetime") from e


def _validate_temporal_window(
    valid_from: str | None,
    valid_to: str | None,
) -> tuple[datetime | None, datetime | None]:
    """Parse and validate optional validity window bounds."""
    parsed_valid_from = _coerce_iso_datetime(valid_from, "valid_from")
    parsed_valid_to = _coerce_iso_datetime(valid_to, "valid_to")
    if parsed_valid_from and parsed_valid_to and parsed_valid_from > parsed_valid_to:
        raise ValueError("valid_from must be less than or equal to valid_to")
    return parsed_valid_from, parsed_valid_to


def _validate_query_temporal_inputs(
    as_of: str | None,
    from_value: str | None,
    to_value: str | None,
) -> tuple[datetime | None, datetime | None, datetime | None]:
    """Validate query temporal inputs for point-in-time vs interval semantics."""
    parsed_as_of = _coerce_iso_datetime(as_of, "as_of")
    parsed_from = _coerce_iso_datetime(from_value, "from")
    parsed_to = _coerce_iso_datetime(to_value, "to")

    if parsed_as_of is not None and (parsed_from is not None or parsed_to is not None):
        raise ValueError("as_of cannot be combined with 'from'/'to'")
    if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
        raise ValueError("'from' must be less than or equal to 'to'")

    return parsed_as_of, parsed_from, parsed_to


def _normalize_scope_value(value: str | None) -> str:
    """Normalize optional topology scope value for scoped uniqueness keys."""
    return (value or "").strip()


def _normalize_category_name(value: str) -> str:
    """Canonicalize category names to prevent typo-variant proliferation."""
    normalized = " ".join(value.split()).lower()
    if not normalized:
        raise ValueError("Category names must not be empty")
    return normalized


async def _resolve_entity_id_manage(
    entity_ref: str,
    backend: Any,
    *,
    namespace: str | None,
    room: str | None,
    corridor: str | None,
) -> str | None:
    """Resolve entity UUID or name to UUID string for manage operations."""
    try:
        uuid.UUID(entity_ref)
        result = await backend.query(
            "SELECT id FROM knowledge_entities WHERE id = $1::uuid", (entity_ref,)
        )
        return str(result.rows[0]["id"]) if result.rows else None
    except (ValueError, AttributeError):
        pass

    normalized_namespace = _normalize_scope_value(namespace)
    normalized_room = _normalize_scope_value(room)
    normalized_corridor = _normalize_scope_value(corridor)

    result = await backend.query(
        "SELECT id FROM knowledge_entities "
        "WHERE name = $1 AND namespace = $2 AND room = $3 AND corridor = $4 "
        "ORDER BY id LIMIT 2",
        (entity_ref, normalized_namespace, normalized_room, normalized_corridor),
    )
    if len(result.rows) > 1:
        raise ValueError(
            "Entity name is ambiguous in this scope. Use an entity UUID instead: "
            f"{entity_ref!r}"
        )
    return str(result.rows[0]["id"]) if result.rows else None


def _get_corridor(request: QueryMemoryRequest | ManageMemoryRequest) -> str | None:
    """Resolve corridor from explicit request field or optional scope bag."""
    corridor = getattr(request, "corridor", None)
    if corridor:
        return str(corridor)
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        raw_corridor = scope.get("corridor")
        if isinstance(raw_corridor, str) and raw_corridor:
            return raw_corridor
    return None


def _has_explicit_scope(request: QueryMemoryRequest) -> bool:
    """Return True when any explicit topology scope is provided."""
    return bool(request.namespace or request.room or _get_corridor(request))


def _build_scope_diagnostics(
    *,
    scope_mode: str,
    scope_applied: bool,
    has_results: bool,
    missing_scope: bool = False,
) -> dict[str, Any]:
    """Build a stable scope diagnostics envelope for query responses."""
    if missing_scope:
        scope_status = "missing_scope"
    elif not scope_applied:
        scope_status = "unscoped"
    elif has_results:
        scope_status = "applied_with_results"
    else:
        scope_status = "no_data_in_scope"

    return {
        "scope_applied": scope_applied,
        "scope_mode": scope_mode,
        "scope_status": scope_status,
    }


_AUTHORITY_ROUTING_FACTS_USER_VALIDATED = "facts_user_validated"


def _build_memory_scope_filters(
    namespace: str | None,
    room: str | None,
    corridor: str | None,
    *,
    alias: str,
    start_index: int = 1,
) -> tuple[list[str], list[Any], int]:
    """Build SQL filter fragments for scoped memory queries."""
    clauses: list[str] = []
    params: list[Any] = []
    next_index = start_index

    if namespace is not None:
        clauses.append(f"{alias}.namespace = ${next_index}")
        params.append(namespace)
        next_index += 1
    if room is not None:
        clauses.append(f"{alias}.room = ${next_index}")
        params.append(room)
        next_index += 1
    if corridor is not None:
        clauses.append(f"{alias}.corridor = ${next_index}")
        params.append(corridor)
        next_index += 1

    return clauses, params, next_index


def _parse_embedding(value: Any) -> list[float] | None:
    """Normalize stored vector values into a Python float list."""
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, tuple):
        return [float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    return None


def _average_embeddings(vectors: list[list[float]]) -> list[float] | None:
    """Compute the centroid of same-sized embedding vectors."""
    if not vectors:
        return None

    dimensions = len(vectors[0])
    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        return None

    totals = [0.0] * dimensions
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return [total / len(vectors) for total in totals]


def _connected_components(
    entity_ids: list[str],
    relation_rows: list[dict[str, Any]],
) -> list[list[str]]:
    """Build deterministic connected components from entity relation edges."""
    adjacency: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}
    known_ids = set(entity_ids)

    for row in relation_rows:
        source_id = str(row["source_entity_id"])
        target_id = str(row["target_entity_id"])
        if source_id not in known_ids or target_id not in known_ids:
            continue
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)

    visited: set[str] = set()
    components: list[list[str]] = []
    for entity_id in sorted(entity_ids):
        if entity_id in visited:
            continue
        stack = [entity_id]
        component: list[str] = []
        visited.add(entity_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))

    components.sort(key=lambda component: tuple(component))
    return components


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class QueryMemoryRequest(BaseModel):
    """Request for unified memory retrieval."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str = Field(description="What to search for")
    goal: Literal["answer", "investigate", "plan", "resume_task", "validate", "debug"] = Field(
        default="answer",
        description="Retrieval intent — shapes strategy and output shape",
    )
    strategy: Literal["auto", "communities", "graph", "palace", "context"] = Field(
        default="auto",
        description=(
            "Retrieval strategy. "
            "auto: scoped lane + global companion lane. "
            "communities: scoped existence filtering + communities/members path. "
            "palace: requires explicit scope and uses strict scoped retrieval."
        ),
    )
    depth: Literal["shallow", "balanced", "deep"] = Field(
        default="balanced",
        description="Search depth — controls candidate counts and fusion passes",
    )
    output_mode: Literal["compact", "evidence", "graph", "mixed"] = Field(
        default="compact",
        description="Response shape",
    )
    scope: dict[str, Any] | None = Field(
        default=None,
        description="Optional scope: palace, wing, room, corridor, time window",
    )
    as_of: str | None = Field(
        default=None,
        description="Point-in-time filter (ISO datetime)",
    )
    from_: str | None = Field(default=None, alias="from", description="Interval start (ISO datetime)")
    to: str | None = Field(default=None, description="Interval end (ISO datetime)")
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Token budget for context output_mode",
    )
    max_items: int = Field(
        default=DEFAULT_LIMIT,
        description="Maximum memories to return",
    )
    # Topology routing
    namespace: str | None = Field(
        default=None,
        description="Namespace/wing for scoped retrieval",
    )
    room: str | None = Field(
        default=None,
        description="Room for scoped retrieval (use with namespace)",
    )
    # Filtering
    source: str | None = Field(default=None)
    categories: list[str] | None = Field(default=None)
    min_confidence: float = Field(default=DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0)
    lifecycle_state: str = Field(default=LifecycleState.ACTIVE)
    embedding_profile: str = Field(default="embedding")
    # Graph routing
    start_entity: str | None = Field(default=None, description="Start entity for graph strategy")
    end_entity: str | None = Field(default=None, description="End entity for graph_path")
    graph_op: Literal["traverse", "neighbors", "path", "stats"] = Field(
        default="traverse",
        description=(
            "Graph sub-operation when strategy='graph'. "
            "traverse: BFS subgraph (default). "
            "neighbors: 1-hop direct neighbors. "
            "path: shortest path between start_entity and end_entity. "
            "stats: degree/connectivity statistics for start_entity or global graph."
        ),
    )
    relation_types: list[str] | None = Field(default=None)
    max_hops: int = Field(default=3)
    max_nodes: int = Field(default=100)

    @model_validator(mode="after")
    def validate_query_temporal_window(self) -> QueryMemoryRequest:
        _validate_query_temporal_inputs(self.as_of, self.from_, self.to)
        return self


class QueryMemoryResult(BaseModel):
    """Recollection-first unified memory retrieval result."""
    facts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top atomic memories at VALIDATED trust state",
    )
    memories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Supporting memories when facts alone are insufficient",
    )
    communities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant higher-level inferred memories",
    )
    paths: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Graph routes (graph strategy or graph output_mode)",
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="References and supporting memory identifiers",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieval metadata (only when requested)",
    )


# ---------------------------------------------------------------------------
# Manage Models
# ---------------------------------------------------------------------------


class ManageMemoryRequest(BaseModel):
    """Request for unified memory write and maintenance."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    operation: Literal[
        "ingest_structured",
        "store",
        "validate",
        "supersede",
        "forget",
        "consolidate",
        "maintain",
        "context",
        "graph_store_entity",
        "graph_store_relation",
        "graph_forget_entity",
        "graph_forget_relation",
    ] = Field(description="Operation family to execute")

    # Structured ingest
    memories: list[StructuredMemoryRecord] | None = Field(
        default=None,
        description="Structured memory records for ingest_structured",
    )
    entities: list[StructuredEntityRecord] | None = Field(
        default=None,
        description="Structured entity records for ingest_structured",
    )
    relations: list[StructuredRelationRecord] | None = Field(
        default=None,
        description="Structured relation records for ingest_structured",
    )

    # Common
    memory_ids: list[str] | None = Field(
        default=None,
        description="Memory IDs to act on (validate, supersede, forget)",
    )
    reason: str | None = Field(default=None, description="Reason for the operation")

    # Store
    content: str | None = Field(
        default=None,
        description="Memory content to store (required for store)",
    )
    source: str | None = Field(default=None)
    path: str | None = Field(default=None)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    authority: str = Field(default=Authority.AGENT)
    lifecycle_state: str = Field(default=LifecycleState.ACTIVE)
    valid_from: str | None = Field(default=None)
    valid_to: str | None = Field(default=None)
    namespace: str | None = Field(default=None)
    room: str | None = Field(default=None)
    corridor: str | None = Field(default=None)
    source_type: str = Field(default="TOOL")
    categories: list[str] | None = Field(default=None)
    allow_create_categories: bool = Field(
        default=False,
        description="Explicit opt-in to create missing categories during write operations",
    )

    # Supersede
    superseded_by: str | None = Field(
        default=None,
        description="New memory ID that replaces the superseded ones",
    )

    # Context
    query: str | None = Field(default=None, description="Query for context assembly")
    as_of: str | None = Field(default=None, description="Point-in-time filter (ISO datetime)")
    from_: str | None = Field(default=None, alias="from", description="Interval start (ISO datetime)")
    to: str | None = Field(default=None, description="Interval end (ISO datetime)")
    max_items: int = Field(default=DEFAULT_LIMIT, description="Maximum memories to return")
    min_confidence: float = Field(default=DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS)
    diversity: bool = Field(default=True)

    # Consolidate / Maintain
    mode: str | None = Field(
        default=None,
        description="Sub-mode: e.g. community_refresh, decay_scan, prune_candidates",
    )
    embedding_profile: str = Field(default="embedding")

    # Maintain sub-operation fields
    decay_rate_per_day: float = Field(
        default=0.01,
        description="Decay rate per day for relevance score (decay_scan mode)",
    )
    grace_period_days: int = Field(
        default=30,
        description="Grace period in days for new memories (decay_scan mode)",
    )
    auto_archive_threshold: float = Field(
        default=0.1,
        description="Score below which to auto-archive (prune_candidates mode)",
    )
    review_threshold: float = Field(
        default=0.3,
        description="Score below which to include for review (prune_candidates mode)",
    )
    grace_days: int = Field(
        default=90,
        description="Days before quarantined/flagged items expire",
    )
    # Graph entity/relation fields
    entity_name: str | None = Field(default=None, description="Entity name for graph_store_entity")
    entity_type: str | None = Field(default=None, description="Entity type for graph_store_entity")
    source_entity: str | None = Field(
        default=None, description="Source entity UUID or name for graph_store_relation"
    )
    target_entity: str | None = Field(
        default=None, description="Target entity UUID or name for graph_store_relation"
    )
    relation_type: str | None = Field(
        default=None, description="Relation type string for graph_store_relation"
    )
    evidence_memory_id: str | None = Field(
        default=None, description="Optional memory UUID as evidence for the relation"
    )
    entity_ids: list[str] | None = Field(
        default=None, description="Entity UUIDs for graph_forget_entity"
    )
    relation_ids: list[str] | None = Field(
        default=None, description="Relation UUIDs for graph_forget_relation"
    )


class ManageMemoryResult(BaseModel):
    """Result for unified memory write and maintenance."""

    operation: str = Field(description="The operation that was executed")
    success: bool = Field(default=True)
    error: str | None = Field(default=None)

    # Store
    memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of stored or affected memories",
    )
    stored_count: int = Field(default=0)
    entity_ids: list[str] = Field(
        default_factory=list,
        description="IDs of stored or affected entities",
    )
    relation_ids: list[str] = Field(
        default_factory=list,
        description="IDs of stored or affected relations",
    )
    entities_stored_count: int = Field(
        default=0,
        description="Count of entities affected by ingest upserts (created or updated)",
    )
    relations_stored_count: int = Field(
        default=0,
        description="Count of relation rows inserted by ingest_structured",
    )

    # Forget
    archived_count: int = Field(default=0)
    skipped_count: int = Field(default=0)

    # Validate
    validated_count: int = Field(default=0)

    # Supersede
    superseded_ids: list[str] = Field(default_factory=list)

    # Context
    context_text: str = Field(default="")
    memory_count: int = Field(default=0)
    tokens_used: int = Field(default=0)

    # Consolidate / Maintain
    communities_updated: int = Field(
        default=0,
        description=(
            "Number of communities materialized by consolidate mode='community_refresh'. "
            "When diagnostics.community_count is present for that run, values are identical."
        ),
    )
    prune_candidates: list[dict[str, Any]] = Field(default_factory=list)
    assessed_count: int = Field(default=0)
    below_threshold_count: int = Field(default=0)
    auto_archive_ids: list[str] = Field(default_factory=list)
    needs_review: list[dict[str, Any]] = Field(default_factory=list)
    expired_count: int = Field(default=0)
    resolved_count: int = Field(default=0)

    diagnostics: dict[str, Any] = Field(default_factory=dict)

    # Graph entity/relation outputs
    entity_id: str | None = Field(default=None, description="Stored entity UUID")
    relation_id: str | None = Field(default=None, description="Stored relation UUID")
    deleted_entity_count: int = Field(default=0)
    deleted_relation_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# Unified Memory Contract Models
# ---------------------------------------------------------------------------


class MemoryScope(BaseModel):
    """Topology scope aligned with MemPalace terminology."""

    model_config = ConfigDict(extra="forbid")

    wing: str | None = Field(default=None, description="Service/project scope")
    room: str | None = Field(default=None, description="Component scope")
    hall: str | None = Field(default=None, description="Topic lane scope")


class MemoryLimits(BaseModel):
    """Query result and traversal limits."""

    model_config = ConfigDict(extra="forbid")

    items: int = Field(default=DEFAULT_LIMIT, ge=1)
    tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    hops: int = Field(default=3, ge=1)
    nodes: int = Field(default=100, ge=1)


class MemoryQueryGraph(BaseModel):
    """Graph query controls for query.mode=graph."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["traverse", "neighbors", "path", "stats"] = Field(default="traverse")
    start: str | None = Field(default=None)
    end: str | None = Field(default=None)
    relation_types: list[str] | None = Field(default=None)


class MemoryQueryInput(BaseModel):
    """Read-only query input controls."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    text: str = Field(description="Search text")
    mode: Literal["search", "context", "graph"] = Field(default="search")
    radius: int = Field(default=1, ge=0, description="Topology expansion distance")
    precision: float = Field(default=0.5, ge=0.0, le=1.0, description="Semantic strictness")
    as_of: str | None = Field(default=None)
    from_: str | None = Field(default=None, alias="from")
    to: str | None = Field(default=None)
    source: str | None = Field(default=None)
    categories: list[str] | None = Field(default=None)
    limits: MemoryLimits = Field(default_factory=MemoryLimits)
    graph: MemoryQueryGraph = Field(default_factory=MemoryQueryGraph)

    @model_validator(mode="after")
    def validate_query_temporal_window(self) -> MemoryQueryInput:
        _validate_query_temporal_inputs(self.as_of, self.from_, self.to)
        return self


class MemoryRecordInput(BaseModel):
    """Write/lifecycle payload."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["raw", "structured"] = Field(default="raw")
    content: str | None = Field(default=None)
    memories: list[dict[str, Any]] | None = Field(default=None)
    entities: list[dict[str, Any]] | None = Field(default=None)
    relations: list[dict[str, Any]] | None = Field(default=None)
    ids: list[str] | None = Field(default=None)
    superseded_by: str | None = Field(default=None)
    reason: str | None = Field(default=None)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    authority: str = Field(default=Authority.AGENT)
    lifecycle_state: str = Field(default=LifecycleState.ACTIVE)
    source: str | None = Field(default=None)
    path: str | None = Field(default=None)
    valid_from: str | None = Field(default=None)
    valid_to: str | None = Field(default=None)
    categories: list[str] | None = Field(default=None)
    allow_create_categories: bool = Field(
        default=False,
        description="Explicit opt-in to create missing categories during ingest",
    )

    @model_validator(mode="after")
    def validate_validity_window(self) -> MemoryRecordInput:
        _validate_temporal_window(self.valid_from, self.valid_to)
        return self


class MemoryGraphInput(BaseModel):
    """Graph mutation payload."""

    kind: Literal["place", "link"] = Field(default="place")
    place_name: str | None = Field(default=None)
    place_type: str | None = Field(default=None)
    from_ref: str | None = Field(default=None, alias="from")
    to_ref: str | None = Field(default=None, alias="to")
    link_type: str | None = Field(default=None)
    evidence_memory_id: str | None = Field(default=None)
    ids: list[str] | None = Field(default=None)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MemoryMaintenanceInput(BaseModel):
    """Maintenance controls."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "community_refresh",
        "decay_scan",
        "prune_candidates",
        "expire_quarantine",
        "expire_flags",
    ] = Field(default="community_refresh")
    decay_rate_per_day: float = Field(default=0.01)
    grace_period_days: int = Field(default=30)
    auto_archive_threshold: float = Field(default=0.1)
    review_threshold: float = Field(default=0.3)
    grace_days: int = Field(default=90)


class MemoryResponseInput(BaseModel):
    """Response shaping controls."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["compact", "evidence", "graph"] = Field(default="compact")
    debug: bool = Field(default=False)
    include_candidates: bool = Field(default=False)


class MemoryRequest(BaseModel):
    """Single canonical request envelope for memory tool and block."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "query",
        "ingest",
        "validate",
        "supersede",
        "archive",
        "maintain",
        "graph_upsert",
        "graph_delete",
    ]
    scope: MemoryScope = Field(default_factory=MemoryScope)
    query: MemoryQueryInput | None = Field(default=None)
    record: MemoryRecordInput | None = Field(default=None)
    graph: MemoryGraphInput | None = Field(default=None)
    maintenance: MemoryMaintenanceInput | None = Field(default=None)
    response: MemoryResponseInput = Field(default_factory=MemoryResponseInput)


class MemoryResult(BaseModel):
    """Canonical result envelope."""

    operation: str
    query: QueryMemoryResult | None = Field(default=None)
    manage: ManageMemoryResult | None = Field(default=None)


class StructuredMemoryRecord(BaseModel):
    """Structured memory payload for ingest_structured."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="Memory content")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    authority: str | None = Field(default=None)
    lifecycle_state: str | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


class StructuredEntityRecord(BaseModel):
    """Structured entity payload for ingest_structured."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Entity display name")
    entity_type: str = Field(description="Entity type")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    memory_indices: list[int] | None = Field(default=None)


class StructuredRelationRecord(BaseModel):
    """Structured relation payload for ingest_structured."""

    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(description="Source entity name")
    source_type: str = Field(description="Source entity type")
    target_name: str = Field(description="Target entity name")
    target_type: str = Field(description="Target entity type")
    relation_type: str = Field(description="Relation type")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_memory_index: int | None = Field(default=None)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MemoryService:
    """Shared orchestration layer for all memory operations.

    Both MCP tools (query_memory, manage_memory) and the Memory workflow block
    delegate here. All invariants — validation, audit, lifecycle — are enforced
    in this layer regardless of caller.

    Args:
        backend: Connected DatabaseBackend instance.
        context: Execution context carrying user identity for audit.
    """

    def __init__(self, backend: Any, context: Execution) -> None:
        self._backend = backend
        self._context = context

    async def execute(self, request: MemoryRequest) -> MemoryResult:
        """Execute unified memory operation envelope."""
        op = request.operation

        if op == "query":
            if request.query is None:
                raise ValueError("'query' payload is required for operation='query'")

            query_mode = request.query.mode
            strategy: Literal["auto", "communities", "graph", "palace", "context"]
            if query_mode == "context":
                strategy = "context"
            elif query_mode == "graph":
                strategy = "graph"
            elif request.query.radius == 0 and (
                request.scope.wing or request.scope.room or request.scope.hall
            ):
                strategy = "palace"
            else:
                strategy = "auto"

            query_result = await self.query(
                QueryMemoryRequest(
                    query=request.query.text,
                    strategy=strategy,
                    as_of=request.query.as_of,
                    from_=request.query.from_,
                    to=request.query.to,
                    max_tokens=request.query.limits.tokens,
                    max_items=request.query.limits.items,
                    namespace=request.scope.wing,
                    room=request.scope.room,
                    source=request.query.source,
                    categories=request.query.categories,
                    min_confidence=request.query.precision,
                    start_entity=request.query.graph.start,
                    end_entity=request.query.graph.end,
                    graph_op=request.query.graph.op,
                    relation_types=request.query.graph.relation_types,
                    max_hops=request.query.limits.hops,
                    max_nodes=request.query.limits.nodes,
                    scope={"corridor": request.scope.hall},
                )
            )
            return MemoryResult(operation=op, query=query_result)

        if op == "ingest":
            if request.record is None:
                raise ValueError("'record' payload is required for operation='ingest'")
            if request.record.format == "structured":
                manage_request = ManageMemoryRequest(
                    operation="ingest_structured",
                    source=request.record.source,
                    path=request.record.path,
                    namespace=request.scope.wing,
                    room=request.scope.room,
                    corridor=request.scope.hall,
                    memories=request.record.memories,
                    entities=request.record.entities,
                    relations=request.record.relations,
                    confidence=request.record.confidence,
                    authority=request.record.authority,
                    lifecycle_state=request.record.lifecycle_state,
                    categories=request.record.categories,
                    allow_create_categories=request.record.allow_create_categories,
                )
            else:
                manage_request = ManageMemoryRequest(
                    operation="store",
                    content=request.record.content,
                    source=request.record.source,
                    path=request.record.path,
                    valid_from=request.record.valid_from,
                    valid_to=request.record.valid_to,
                    namespace=request.scope.wing,
                    room=request.scope.room,
                    corridor=request.scope.hall,
                    confidence=request.record.confidence,
                    authority=request.record.authority,
                    lifecycle_state=request.record.lifecycle_state,
                    categories=request.record.categories,
                    allow_create_categories=request.record.allow_create_categories,
                )
            manage_result = await self.manage(manage_request)
            return MemoryResult(operation=op, manage=manage_result)

        if op in {"validate", "supersede", "archive"}:
            if request.record is None:
                raise ValueError(f"'record' payload is required for operation='{op}'")
            mapped_op = "forget" if op == "archive" else op
            manage_result = await self.manage(
                ManageMemoryRequest(
                    operation=cast(Any, mapped_op),
                    memory_ids=request.record.ids,
                    superseded_by=request.record.superseded_by,
                    reason=request.record.reason,
                    valid_to=request.record.valid_to,
                )
            )
            return MemoryResult(operation=op, manage=manage_result)

        if op == "maintain":
            maintenance = request.maintenance or MemoryMaintenanceInput()
            manage_result = await self.manage(
                ManageMemoryRequest(
                    operation="maintain",
                    mode=maintenance.mode,
                    decay_rate_per_day=maintenance.decay_rate_per_day,
                    grace_period_days=maintenance.grace_period_days,
                    auto_archive_threshold=maintenance.auto_archive_threshold,
                    review_threshold=maintenance.review_threshold,
                    grace_days=maintenance.grace_days,
                )
            )
            return MemoryResult(operation=op, manage=manage_result)

        if op == "graph_upsert":
            if request.graph is None:
                raise ValueError("'graph' payload is required for operation='graph_upsert'")
            if request.graph.kind == "place":
                manage_request = ManageMemoryRequest(
                    operation="graph_store_entity",
                    entity_name=request.graph.place_name,
                    entity_type=request.graph.place_type,
                    namespace=request.scope.wing,
                    room=request.scope.room,
                    corridor=request.scope.hall,
                )
            else:
                manage_request = ManageMemoryRequest(
                    operation="graph_store_relation",
                    source_entity=request.graph.from_ref,
                    target_entity=request.graph.to_ref,
                    relation_type=request.graph.link_type,
                    evidence_memory_id=request.graph.evidence_memory_id,
                    namespace=request.scope.wing,
                    room=request.scope.room,
                    corridor=request.scope.hall,
                )
            manage_result = await self.manage(manage_request)
            return MemoryResult(operation=op, manage=manage_result)

        if op == "graph_delete":
            if request.graph is None:
                raise ValueError("'graph' payload is required for operation='graph_delete'")
            if request.graph.kind == "place":
                manage_request = ManageMemoryRequest(
                    operation="graph_forget_entity",
                    entity_ids=request.graph.ids,
                )
            else:
                manage_request = ManageMemoryRequest(
                    operation="graph_forget_relation",
                    relation_ids=request.graph.ids,
                )
            manage_result = await self.manage(manage_request)
            return MemoryResult(operation=op, manage=manage_result)

        raise ValueError(f"Unsupported operation: {op}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Unified memory retrieval.

        Routes based on strategy:
        - auto: hybrid vector+FTS search over memories/facts
        - communities: persisted community retrieval
        - context: token-budgeted context assembly
        - graph: graph traversal from start_entity
        """
        strategy = request.strategy

        if strategy == "auto":
            return await self._query_memories(request)
        if strategy == "communities":
            return await self._query_communities(request)
        if strategy == "context":
            return await self._query_context(request)
        if strategy == "graph":
            return await self._query_graph(request)
        if strategy == "palace":
            return await self._query_palace(request)

        return QueryMemoryResult(
            diagnostics={"error": f"Unknown strategy: {strategy!r}"},
        )

    async def _query_memories(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Run hybrid search and fuse into recollection-first result."""
        limit = request.max_items
        corridor = _get_corridor(request)

        resolved_categories: list[str] | None = None
        if request.categories:
            resolved_categories = await self._resolve_categories(request.categories)

        as_of, from_dt, to_dt = _validate_query_temporal_inputs(
            request.as_of, request.from_, request.to
        )

        embedding, _, _, _ = await compute_embedding(
            text=request.query,
            context=self._context,
            profile=request.embedding_profile,
        )

        rows = await room_scoped_search(
            embedding,
            request.query,
            self._backend,
            namespace=request.namespace,
            room=request.room,
            corridor=corridor,
            source=request.source,
            categories=resolved_categories,
            as_of=as_of,
            from_dt=from_dt,
            to_dt=to_dt,
            min_confidence=request.min_confidence,
            lifecycle_state=request.lifecycle_state,
            limit=limit,
        )

        if rows:
            ids = [row["id"] for row in rows]
            placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
            await self._backend.execute(
                "UPDATE knowledge_memories "
                "SET retrieval_count = retrieval_count + 1, last_retrieved_at = NOW() "
                f"WHERE id IN ({placeholders})",
                tuple(str(id_) for id_ in ids),
            )

        facts: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        for row in rows:
            # Build lean dict — always include content; include optional fields only when present
            cleaned: dict[str, Any] = {
                "id": str(row.get("id", "")),
                "content": row.get("content", ""),
                "confidence": row.get("confidence"),
                "authority": row.get("authority"),
                "rrf_score": row.get("rrf_score"),
                # provenance fields — consumers can locate the source file/project
                "path": row.get("item_path") or None,
                "source": row.get("source_name") or None,
                "namespace": row.get("namespace") or None,
            }
            # Lane contract: USER_VALIDATED is promoted to facts; other authorities stay in memories.
            if row.get("authority") == Authority.USER_VALIDATED:
                facts.append(cleaned)
            else:
                memories.append(cleaned)

        scope_applied = _has_explicit_scope(request)
        scope_diagnostics = _build_scope_diagnostics(
            scope_mode="dual_lane_with_companion",
            scope_applied=scope_applied,
            has_results=bool(facts or memories),
        )

        return QueryMemoryResult(
            facts=facts,
            memories=memories,
            diagnostics={
                **scope_diagnostics,
                "authority_routing": _AUTHORITY_ROUTING_FACTS_USER_VALIDATED,
            },
        )

    async def _query_communities(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Retrieve persisted community summaries using centroid embeddings."""
        resolved_categories: list[str] | None = None
        if request.categories:
            resolved_categories = await self._resolve_categories(request.categories)

        as_of, from_dt, to_dt = _validate_query_temporal_inputs(
            request.as_of, request.from_, request.to
        )
        corridor = _get_corridor(request)

        embedding, _, _, _ = await compute_embedding(
            text=request.query,
            context=self._context,
            profile=request.embedding_profile,
        )

        params: list[Any] = [str(embedding), request.lifecycle_state, request.min_confidence]
        clauses = [
            "km.lifecycle_state = $2",
            "km.confidence >= $3",
        ]

        scope_clauses, scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="km",
            start_index=4,
        )
        clauses.extend(scope_clauses)
        params.extend(scope_params)

        if request.source:
            source_index = len(params) + 1
            if request.source.endswith("*"):
                clauses.append(f"km.source_name LIKE ${source_index}")
                params.append(request.source[:-1] + "%")
            else:
                clauses.append(f"km.source_name = ${source_index}")
                params.append(request.source)

        if resolved_categories:
            category_index = len(params) + 1
            clauses.append(
                "EXISTS ("
                "SELECT 1 FROM knowledge_memory_categories kmc "
                f"WHERE kmc.memory_id = km.id AND kmc.category_id = ANY(${category_index}::uuid[])"
                ")"
            )
            params.append(resolved_categories)

        if as_of is not None:
            as_of_index = len(params) + 1
            clauses.append(f"(km.valid_from IS NULL OR km.valid_from <= ${as_of_index}::timestamptz)")
            clauses.append(f"(km.valid_to IS NULL OR km.valid_to >= ${as_of_index}::timestamptz)")
            params.append(as_of)
        else:
            if from_dt is not None:
                from_index = len(params) + 1
                clauses.append(f"(km.valid_to IS NULL OR km.valid_to >= ${from_index}::timestamptz)")
                params.append(from_dt)
            if to_dt is not None:
                to_index = len(params) + 1
                clauses.append(f"(km.valid_from IS NULL OR km.valid_from <= ${to_index}::timestamptz)")
                params.append(to_dt)

        limit_index = len(params) + 1
        params.append(request.max_items)
        where_clause = " AND ".join(clauses) if clauses else "TRUE"

        result = await self._backend.query(
            f"""
            SELECT
                kc.id,
                kc.content,
                kc.member_count,
                kc.memory_count,
                kc.namespace,
                kc.room,
                kc.corridor,
                1 - (kc.embedding <=> $1::vector) AS similarity
            FROM knowledge_communities kc
            WHERE kc.embedding IS NOT NULL
              AND EXISTS (
                SELECT 1
                FROM knowledge_memories km
                WHERE km.community_id = kc.id
                  AND {where_clause}
              )
            ORDER BY kc.embedding <=> $1::vector
            LIMIT ${limit_index}
            """,
            tuple(params),
        )

        communities: list[dict[str, Any]] = []
        for row in result.rows:
            community = {
                "id": str(row["id"]),
                "content": row.get("content", ""),
                "member_count": row.get("member_count", 0),
                "memory_count": row.get("memory_count", 0),
                "similarity": row.get("similarity"),
            }
            if row.get("namespace") is not None:
                community["namespace"] = row["namespace"]
            if row.get("room") is not None:
                community["room"] = row["room"]
            if row.get("corridor") is not None:
                community["corridor"] = row["corridor"]
            communities.append(community)

        facts: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        community_ids = [str(row["id"]) for row in result.rows if row.get("id") is not None]
        if community_ids:
            rows = await room_scoped_search(
                embedding,
                request.query,
                self._backend,
                namespace=request.namespace,
                room=request.room,
                corridor=corridor,
                source=request.source,
                categories=resolved_categories,
                as_of=as_of,
                from_dt=from_dt,
                to_dt=to_dt,
                min_confidence=request.min_confidence,
                lifecycle_state=request.lifecycle_state,
                limit=request.max_items,
                community_ids=community_ids,
                include_global_companion=False,
            )

            if rows:
                ids = [row["id"] for row in rows]
                placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
                await self._backend.execute(
                    "UPDATE knowledge_memories "
                    "SET retrieval_count = retrieval_count + 1, last_retrieved_at = NOW() "
                    f"WHERE id IN ({placeholders})",
                    tuple(str(id_) for id_ in ids),
                )

            for row in rows:
                cleaned: dict[str, Any] = {
                    "id": str(row.get("id", "")),
                    "content": row.get("content", ""),
                    "confidence": row.get("confidence"),
                    "authority": row.get("authority"),
                    "rrf_score": row.get("rrf_score"),
                    "path": row.get("item_path") or None,
                    "source": row.get("source_name") or None,
                    "namespace": row.get("namespace") or None,
                }
                # Lane contract: USER_VALIDATED is promoted to facts; other authorities stay in memories.
                if row.get("authority") == Authority.USER_VALIDATED:
                    facts.append(cleaned)
                else:
                    memories.append(cleaned)

        scope_applied = _has_explicit_scope(request)
        scope_diagnostics = _build_scope_diagnostics(
            scope_mode="community_exists_filter",
            scope_applied=scope_applied,
            has_results=bool(facts or memories or communities),
        )

        return QueryMemoryResult(
            facts=facts,
            memories=memories,
            communities=communities,
            diagnostics={
                "strategy": "communities",
                **scope_diagnostics,
                "authority_routing": _AUTHORITY_ROUTING_FACTS_USER_VALIDATED,
            },
        )

    async def _query_palace(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Palace-scoped retrieval with explicit namespace/room scoping.

        Palace retrieval requires explicit topology scoping (namespace and/or room).
        """
        if not request.namespace and not request.room:
            return QueryMemoryResult(
                diagnostics={
                    **_build_scope_diagnostics(
                        scope_mode="strict_scoped",
                        scope_applied=False,
                        has_results=False,
                        missing_scope=True,
                    ),
                    "error": (
                        "palace strategy requires namespace or room for scoped retrieval"
                    )
                }
            )

        as_of, from_dt, to_dt = _validate_query_temporal_inputs(
            request.as_of, request.from_, request.to
        )
        corridor = _get_corridor(request)

        embedding, _, _, _ = await compute_embedding(
            text=request.query,
            context=self._context,
            profile=request.embedding_profile,
        )

        # Palace strategy is strict-scoped: scoped lane only (no global companion).
        rows = await room_scoped_search(
            embedding,
            request.query,
            self._backend,
            namespace=request.namespace,
            room=request.room,
            corridor=corridor,
            source=request.source,
            categories=await self._resolve_categories(request.categories)
            if request.categories
            else None,
            as_of=as_of,
            from_dt=from_dt,
            to_dt=to_dt,
            min_confidence=request.min_confidence,
            lifecycle_state=request.lifecycle_state,
            limit=request.max_items,
            include_global_companion=False,
        )

        if rows:
            ids = [row["id"] for row in rows]
            placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
            await self._backend.execute(
                "UPDATE knowledge_memories "
                "SET retrieval_count = retrieval_count + 1, last_retrieved_at = NOW() "
                f"WHERE id IN ({placeholders})",
                tuple(str(id_) for id_ in ids),
            )

        facts: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        for row in rows:
            cleaned: dict[str, Any] = {
                "id": str(row.get("id", "")),
                "content": row.get("content", ""),
                "confidence": row.get("confidence"),
                "authority": row.get("authority"),
                "rrf_score": row.get("rrf_score"),
                "path": row.get("item_path") or None,
                "source": row.get("source_name") or None,
                "namespace": row.get("namespace") or None,
            }
            # Lane contract: USER_VALIDATED is promoted to facts; other authorities stay in memories.
            if row.get("authority") == Authority.USER_VALIDATED:
                facts.append(cleaned)
            else:
                memories.append(cleaned)

        scope_diagnostics = _build_scope_diagnostics(
            scope_mode="strict_scoped",
            scope_applied=True,
            has_results=bool(facts or memories),
        )

        return QueryMemoryResult(
            facts=facts,
            memories=memories,
            diagnostics={
                **scope_diagnostics,
                "authority_routing": _AUTHORITY_ROUTING_FACTS_USER_VALIDATED,
            },
        )

    async def _query_context(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Token-budgeted context assembly."""
        as_of, from_dt, to_dt = _validate_query_temporal_inputs(
            request.as_of, request.from_, request.to
        )
        corridor = _get_corridor(request)

        resolved_categories: list[str] | None = None
        if request.categories:
            resolved_categories = await self._resolve_categories(request.categories)

        embedding, _, _, _ = await compute_embedding(
            text=request.query,
            context=self._context,
            profile=request.embedding_profile,
        )

        rows = await room_scoped_search(
            embedding,
            request.query,
            self._backend,
            namespace=request.namespace,
            room=request.room,
            corridor=corridor,
            source=request.source,
            categories=resolved_categories,
            as_of=as_of,
            from_dt=from_dt,
            to_dt=to_dt,
            min_confidence=request.min_confidence,
            lifecycle_state=request.lifecycle_state,
            limit=request.max_items * 3,
        )

        context_text, included_count, tokens_used = assemble_context(
            rows,
            max_tokens=request.max_tokens,
            diversity=True,
            query_embedding=embedding,
        )

        return QueryMemoryResult(
            evidence=[
                {
                    "context_text": context_text,
                    "memory_count": included_count,
                    "tokens_used": tokens_used,
                }
            ],
        )

    async def _query_graph(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Graph retrieval dispatcher. Routes to sub-operation via request.graph_op."""
        op = request.graph_op
        start_entity = request.start_entity
        end_entity = request.end_entity

        if op in ("traverse", "neighbors") and not start_entity:
            return QueryMemoryResult(
                diagnostics={"error": "start_entity required for graph strategy"},
            )
        if op == "path" and (not start_entity or not end_entity):
            return QueryMemoryResult(
                diagnostics={
                    "error": (
                        "start_entity and end_entity required for "
                        "graph_op='path'"
                    )
                },
            )

        as_of, from_dt, to_dt = _validate_query_temporal_inputs(
            request.as_of, request.from_, request.to
        )
        if from_dt is not None or to_dt is not None:
            return QueryMemoryResult(
                diagnostics={
                    "error": "graph strategy supports 'as_of' only; 'from'/'to' is not supported"
                },
            )

        if op == "traverse":
            assert start_entity is not None
            result = await graph_traverse(
                start_entity,
                self._backend,
                relation_types=request.relation_types,
                max_hops=request.max_hops,
                max_nodes=request.max_nodes,
                as_of=as_of,
            )
            paths: list[dict[str, Any]] = [dict(p) for p in result["paths"]]
            return QueryMemoryResult(
                paths=paths,
                evidence=[{"nodes": result["nodes"], "edges": result["edges"]}],
                diagnostics=dict(result["diagnostics"]),
            )

        if op == "neighbors":
            assert start_entity is not None
            result = await graph_neighbors(
                start_entity,
                self._backend,
                relation_types=request.relation_types,
                max_nodes=request.max_nodes,
                as_of=as_of,
            )
            return QueryMemoryResult(
                paths=[],
                evidence=[{"nodes": result["nodes"], "edges": result["edges"]}],
                diagnostics=dict(result["diagnostics"]),
            )

        if op == "path":
            assert start_entity is not None
            assert end_entity is not None
            result = await graph_path(
                start_entity,
                end_entity,
                self._backend,
                relation_types=request.relation_types,
                max_hops=request.max_hops,
                max_nodes=request.max_nodes,
                as_of=as_of,
            )
            paths = [dict(p) for p in result["paths"]]
            return QueryMemoryResult(
                paths=paths,
                evidence=[{"nodes": result["nodes"], "edges": result["edges"]}],
                diagnostics=dict(result["diagnostics"]),
            )

        if op == "stats":
            result = await graph_stats(
                start_entity,
                self._backend,
                as_of=as_of,
            )
            return QueryMemoryResult(
                paths=[dict(p) for p in result["paths"]],
                evidence=[{"nodes": result["nodes"], "edges": result["edges"]}],
                diagnostics=dict(result["diagnostics"]),
            )

        return QueryMemoryResult(diagnostics={"error": f"Unknown graph_op: {op!r}"})

    # ------------------------------------------------------------------
    # Manage
    # ------------------------------------------------------------------

    async def manage(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Unified memory write and maintenance."""
        op = request.operation
        handlers = {
            "ingest_structured": self._manage_ingest_structured,
            "store": self._manage_store,
            "validate": self._manage_validate,
            "supersede": self._manage_supersede,
            "forget": self._manage_forget,
            "consolidate": self._manage_consolidate,
            "maintain": self._manage_maintain,
            "context": self._manage_context,
            "graph_store_entity": self._manage_graph_store_entity,
            "graph_store_relation": self._manage_graph_store_relation,
            "graph_forget_entity": self._manage_graph_forget_entity,
            "graph_forget_relation": self._manage_graph_forget_relation,
        }
        handler = handlers.get(op)
        if handler is None:
            return ManageMemoryResult(
                operation=op, success=False, error=f"Unknown operation: {op}"
            )
        return await handler(request)

    async def _manage_ingest_structured(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Store structured memories, entities, relations, and links atomically."""
        if not request.memories:
            return ManageMemoryResult(
                operation="ingest_structured",
                success=False,
                error="'memories' is required for ingest_structured operation",
            )

        entities = request.entities or []
        relations = request.relations or []

        transaction_started = False
        try:
            await self._backend.begin_transaction()
            transaction_started = True

            item_id: str | None = None
            source_name: str | None = None
            if request.source and request.path:
                source_result = await self._backend.query(
                    """
                    INSERT INTO knowledge_sources
                        (id, name, source_type, category_ids)
                    VALUES ($1::uuid, $2, $3, '{}'::uuid[])
                    ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
                    RETURNING id
                    """,
                    (str(uuid.uuid4()), request.source, request.source_type),
                )
                if source_result.rows:
                    actual_source_id = str(source_result.rows[0]["id"])
                    item_title = os.path.basename(request.path) or request.path
                    item_result = await self._backend.query(
                        """
                        INSERT INTO knowledge_items (id, source_id, path, title)
                        VALUES ($1::uuid, $2::uuid, $3, $4)
                        ON CONFLICT (source_id, path) DO UPDATE SET
                            title = EXCLUDED.title, updated_at = NOW()
                        RETURNING id
                        """,
                        (str(uuid.uuid4()), actual_source_id, request.path, item_title),
                    )
                    if item_result.rows:
                        item_id = str(item_result.rows[0]["id"])
                source_name = request.source
            elif request.source:
                source_name = request.source

            created_by = _get_audit_user_id(self._context)
            auth_method = _get_auth_method(self._context)
            user_string = _get_user_string_id(self._context)
            memory_ids: list[str] = []
            entity_ids_in_order: list[str] = []
            entity_ids_seen: set[str] = set()
            relation_ids: list[str] = []
            for memory in request.memories:
                memory_id = str(uuid.uuid4())
                memory_ids.append(memory_id)
                embedding, model_name, _, _ = await compute_embedding(
                    text=memory.content,
                    context=self._context,
                    profile=request.embedding_profile,
                )
                await self._backend.execute(
                    """
                    INSERT INTO knowledge_memories
                        (id, item_id, content, embedding, search_vector,
                         authority, lifecycle_state, confidence, embedding_model,
                         metadata, created_by, auth_method,
                         source_name, source_type,
                         namespace, room, corridor)
                    VALUES
                        ($1::uuid, $2::uuid, $3, $4::vector,
                         to_tsvector('english', $3),
                         $5, $6, $7, $8,
                         $9::jsonb, $10::uuid, $11,
                         $12, $13,
                         $14, $15, $16)
                    """,
                    (
                        memory_id,
                        item_id,
                        memory.content,
                        str(embedding),
                        memory.authority or request.authority,
                        memory.lifecycle_state or request.lifecycle_state,
                        memory.confidence if memory.confidence is not None else request.confidence,
                        model_name,
                        json.dumps(memory.metadata or {}),
                        str(created_by),
                        auth_method,
                        source_name,
                        request.source_type,
                        request.namespace,
                        request.room,
                        request.corridor,
                    ),
                )
                await self._log_audit_entry(
                    memory_id=memory_id,
                    action="CREATED",
                    performed_by=created_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={"source": request.source, "path": request.path},
                )

            entity_ids_by_key: dict[tuple[str, str], str] = {}
            entity_scope_namespace = _normalize_scope_value(request.namespace)
            entity_scope_room = _normalize_scope_value(request.room)
            entity_scope_corridor = _normalize_scope_value(_get_corridor(request))
            for entity in entities:
                entity_id = await self._upsert_structured_entity(
                    entity_name=entity.name,
                    entity_type=entity.entity_type,
                    confidence=entity.confidence if entity.confidence is not None else request.confidence,
                    namespace=entity_scope_namespace,
                    room=entity_scope_room,
                    corridor=entity_scope_corridor,
                )
                entity_ids_by_key[(entity.entity_type, entity.name)] = entity_id
                if entity_id not in entity_ids_seen:
                    entity_ids_seen.add(entity_id)
                    entity_ids_in_order.append(entity_id)

                for memory_index in entity.memory_indices or []:
                    memory_id = self._memory_id_for_index(memory_ids, memory_index)
                    await self._link_memory_entity(
                        memory_id=memory_id,
                        entity_id=entity_id,
                        confidence=entity.confidence if entity.confidence is not None else request.confidence,
                    )

            for relation in relations:
                source_entity_id = await self._resolve_or_upsert_structured_entity(
                    entity_ids_by_key,
                    relation.source_type,
                    relation.source_name,
                    request.confidence,
                    namespace=entity_scope_namespace,
                    room=entity_scope_room,
                    corridor=entity_scope_corridor,
                )
                if source_entity_id not in entity_ids_seen:
                    entity_ids_seen.add(source_entity_id)
                    entity_ids_in_order.append(source_entity_id)
                target_entity_id = await self._resolve_or_upsert_structured_entity(
                    entity_ids_by_key,
                    relation.target_type,
                    relation.target_name,
                    request.confidence,
                    namespace=entity_scope_namespace,
                    room=entity_scope_room,
                    corridor=entity_scope_corridor,
                )
                if target_entity_id not in entity_ids_seen:
                    entity_ids_seen.add(target_entity_id)
                    entity_ids_in_order.append(target_entity_id)
                evidence_memory_id = None
                if relation.evidence_memory_index is not None:
                    evidence_memory_id = self._memory_id_for_index(
                        memory_ids,
                        relation.evidence_memory_index,
                    )
                    await self._link_memory_entity(
                        memory_id=evidence_memory_id,
                        entity_id=source_entity_id,
                        confidence=relation.confidence if relation.confidence is not None else request.confidence,
                    )
                    await self._link_memory_entity(
                        memory_id=evidence_memory_id,
                        entity_id=target_entity_id,
                        confidence=relation.confidence if relation.confidence is not None else request.confidence,
                    )
                relation_result = await self._backend.query(
                    """
                    INSERT INTO knowledge_relations
                        (id, source_entity_id, target_entity_id, relation_type,
                         confidence, evidence_memory_id)
                    VALUES
                        ($1::uuid, $2::uuid, $3::uuid, $4,
                         $5, $6::uuid)
                    RETURNING id
                    """,
                    (
                        str(uuid.uuid4()),
                        source_entity_id,
                        target_entity_id,
                        relation.relation_type,
                        relation.confidence if relation.confidence is not None else request.confidence,
                        evidence_memory_id,
                    ),
                )
                if relation_result.rows:
                    relation_ids.append(str(relation_result.rows[0]["id"]))

            await self._backend.commit()
            return ManageMemoryResult(
                operation="ingest_structured",
                success=True,
                stored_count=len(memory_ids),
                memory_ids=memory_ids,
                entity_ids=entity_ids_in_order,
                relation_ids=relation_ids,
                entities_stored_count=len(entity_ids_in_order),
                relations_stored_count=len(relation_ids),
            )
        except Exception as exc:
            try:
                await self._backend.rollback()
            except Exception:
                logger.warning(
                    "Failed to rollback ingest_structured transaction",
                    exc_info=True,
                )

            if not transaction_started:
                detail = str(exc).strip() or exc.__class__.__name__
                raise RuntimeError(
                    f"Failed to begin transaction for ingest_structured: {detail}"
                ) from exc

            raise

    async def _manage_store(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Store a new memory."""
        if not request.content:
            return ManageMemoryResult(
                operation="store",
                success=False,
                error="'content' is required for store operation",
            )

        memory_id = str(uuid.uuid4())

        category_ids: list[str] = []
        if request.categories:
            category_ids = await self._resolve_categories(
                request.categories,
                allow_create=request.allow_create_categories,
            )

        embedding, model_name, _, _ = await compute_embedding(
            text=request.content,
            context=self._context,
            profile=request.embedding_profile,
        )

        item_id: str | None = None
        prop_metadata: dict[str, str] = {}
        source_name: str | None = None

        if request.source and request.path:
            source_result = await self._backend.query(
                """
                INSERT INTO knowledge_sources
                    (id, name, source_type, category_ids)
                VALUES ($1::uuid, $2, 'TOOL', '{}'::uuid[])
                ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
                RETURNING id
                """,
                (str(uuid.uuid4()), request.source),
            )
            if source_result.rows:
                actual_source_id = str(source_result.rows[0]["id"])
                item_title = os.path.basename(request.path) or request.path
                item_result = await self._backend.query(
                    """
                    INSERT INTO knowledge_items (id, source_id, path, title)
                    VALUES ($1::uuid, $2::uuid, $3, $4)
                    ON CONFLICT (source_id, path) DO UPDATE SET
                        title = EXCLUDED.title, updated_at = NOW()
                    RETURNING id
                    """,
                    (str(uuid.uuid4()), actual_source_id, request.path, item_title),
                )
                if item_result.rows:
                    item_id = str(item_result.rows[0]["id"])
            source_name = request.source
        elif request.source:
            source_name = request.source
            prop_metadata = {"source": request.source}

        created_by = _get_audit_user_id(self._context)
        auth_method = _get_auth_method(self._context)
        user_string = _get_user_string_id(self._context)
        valid_from, valid_to = _validate_temporal_window(request.valid_from, request.valid_to)

        if auth_method:
            prop_metadata["auth_method"] = auth_method
        if user_string:
            prop_metadata["user_identifier"] = user_string
        metadata_json = json.dumps(prop_metadata)

        await self._backend.execute(
            """
            INSERT INTO knowledge_memories
                (id, item_id, content, embedding, search_vector,
                 authority, lifecycle_state, confidence,
                 embedding_model, metadata,
                 valid_from, valid_to,
                 created_by, auth_method, source_name, source_type,
                 namespace, room, corridor)
            VALUES
                ($1::uuid, $2::uuid, $3, $4::vector,
                 to_tsvector('english', $3),
                 $5, $6, $7,
                 $8, $9::jsonb,
                 $10::timestamptz, $11::timestamptz,
                 $12::uuid, $13, $14, $15,
                 $16, $17, $18)
            """,
            (
                memory_id,
                item_id,
                request.content,
                str(embedding),
                request.authority,
                request.lifecycle_state,
                request.confidence,
                model_name,
                metadata_json,
                valid_from,
                valid_to,
                str(created_by),
                auth_method,
                source_name,
                request.source_type,
                request.namespace,
                request.room,
                request.corridor,
            ),
        )

        for cat_id in category_ids:
            await self._backend.execute(
                """
                INSERT INTO knowledge_memory_categories
                    (memory_id, category_id, assigned_by)
                VALUES ($1::uuid, $2::uuid, 'EXPLICIT')
                ON CONFLICT (memory_id, category_id) DO NOTHING
                """,
                (memory_id, cat_id),
            )

        await self._log_audit_entry(
            memory_id=memory_id,
            action="CREATED",
            performed_by=created_by,
            auth_method=auth_method,
            user_string=user_string,
            metadata={"source": request.source, "path": request.path},
        )

        return ManageMemoryResult(
            operation="store",
            memory_ids=[memory_id],
            stored_count=1,
        )

    async def _manage_validate(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Promote memories to USER_VALIDATED trust state."""
        if not request.memory_ids:
            return ManageMemoryResult(
                operation="validate",
                success=False,
                error="'memory_ids' is required for validate operation",
            )
        ids = [str(i) for i in request.memory_ids]
        placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
        result = await self._backend.query(
            f"""
            UPDATE knowledge_memories
            SET authority = 'USER_VALIDATED', updated_at = NOW()
            WHERE id IN ({placeholders})
              AND authority != 'USER_VALIDATED'
            RETURNING id
            """,
            tuple(ids),
        )
        validated_ids = [str(r["id"]) for r in (result.rows or [])]
        if validated_ids:
            performed_by = _get_audit_user_id(self._context)
            auth_method = _get_auth_method(self._context)
            user_string = _get_user_string_id(self._context)
            for memory_id in validated_ids:
                await self._log_audit_entry(
                    memory_id=memory_id,
                    action="VALIDATED",
                    performed_by=performed_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={"reason": request.reason},
                )
        return ManageMemoryResult(
            operation="validate",
            validated_count=len(validated_ids),
            memory_ids=validated_ids,
        )

    async def _manage_supersede(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Mark memories as SUPERSEDED."""
        if not request.memory_ids:
            return ManageMemoryResult(
                operation="supersede",
                success=False,
                error="'memory_ids' is required for supersede operation",
            )
        if not request.superseded_by:
            return ManageMemoryResult(
                operation="supersede",
                success=False,
                error="'superseded_by' is required for supersede operation",
            )
        ids = [str(i) for i in request.memory_ids]
        placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
        metadata = json.dumps(
            {"reason": request.reason or "superseded", "superseded_by": request.superseded_by}
        )
        explicit_valid_to = _coerce_iso_datetime(request.valid_to, "valid_to")
        result = await self._backend.query(
            f"""
            UPDATE knowledge_memories
            SET lifecycle_state = 'SUPERSEDED',
                metadata = metadata || ${len(ids) + 1}::jsonb,
                valid_to = COALESCE(${len(ids) + 2}::timestamptz, NOW()),
                updated_at = NOW()
            WHERE id IN ({placeholders})
              AND lifecycle_state NOT IN ('SUPERSEDED', 'ARCHIVED')
            RETURNING id
            """,
            tuple(ids) + (metadata, explicit_valid_to),
        )
        superseded = [str(r["id"]) for r in (result.rows or [])]
        if superseded:
            performed_by = _get_audit_user_id(self._context)
            auth_method = _get_auth_method(self._context)
            user_string = _get_user_string_id(self._context)
            for memory_id in superseded:
                await self._log_audit_entry(
                    memory_id=memory_id,
                    action="SUPERSEDED",
                    performed_by=performed_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={
                        "reason": request.reason,
                        "superseded_by": request.superseded_by,
                    },
                )
        return ManageMemoryResult(
            operation="supersede",
            superseded_ids=superseded,
            memory_ids=superseded,
            archived_count=len(superseded),
        )

    async def _manage_forget(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Archive memories (soft-delete). USER_VALIDATED memories are immune."""
        if not request.memory_ids:
            return ManageMemoryResult(
                operation="forget",
                success=False,
                error="'memory_ids' is required for forget operation",
            )
        ids = [str(i) for i in request.memory_ids]
        reason = request.reason or "manual"
        explicit_valid_to = _coerce_iso_datetime(request.valid_to, "valid_to")
        placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))

        skipped_result = await self._backend.query(
            f"""
            SELECT COUNT(*) AS cnt FROM knowledge_memories
            WHERE id IN ({placeholders})
              AND (authority = 'USER_VALIDATED' OR lifecycle_state = 'ARCHIVED')
            """,
            tuple(ids),
        )
        skipped = skipped_result.rows[0]["cnt"] if skipped_result.rows else 0

        archive_result = await self._backend.query(
            f"""
            UPDATE knowledge_memories
            SET lifecycle_state = 'ARCHIVED',
                archived_at = NOW(),
                archive_reason = ${len(ids) + 1},
                valid_to = COALESCE(${len(ids) + 2}::timestamptz, NOW()),
                updated_at = NOW()
            WHERE id IN ({placeholders})
              AND authority != 'USER_VALIDATED'
              AND lifecycle_state != 'ARCHIVED'
            RETURNING id
            """,
            tuple(ids) + (reason, explicit_valid_to),
        )
        archived_ids = [str(r["id"]) for r in (archive_result.rows or [])]
        archived = len(archived_ids)
        if archived_ids:
            performed_by = _get_audit_user_id(self._context)
            auth_method = _get_auth_method(self._context)
            user_string = _get_user_string_id(self._context)
            for memory_id in archived_ids:
                await self._log_audit_entry(
                    memory_id=memory_id,
                    action="ARCHIVED",
                    performed_by=performed_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={"reason": reason},
                )
        return ManageMemoryResult(
            operation="forget",
            archived_count=archived,
            skipped_count=skipped,
        )

    async def _manage_consolidate(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Community and graph consolidation operations.

        Supported modes:
                - community_refresh: persist graph-derived communities and assignments.
                    communities_updated is the number of community rows refreshed and
                    matches diagnostics.community_count for successful runs.
        """
        if request.mode != "community_refresh":
            return ManageMemoryResult(
                operation="consolidate",
                success=False,
                error=f"Unknown consolidate mode: {request.mode!r}",
            )

        await self._backend.begin_transaction()
        try:
            entity_rows = await self._load_scoped_entities(request)
            entity_ids = [str(row["id"]) for row in entity_rows]

            relation_rows: list[dict[str, Any]] = []
            if entity_ids:
                relation_rows = await self._load_scoped_relations(entity_ids, request)

            await self._clear_communities(request, entity_ids)

            if not entity_rows:
                await self._backend.commit()
                return ManageMemoryResult(
                    operation="consolidate",
                    communities_updated=0,
                    diagnostics={
                        "mode": request.mode,
                        "status": "ok",
                        "entity_count": 0,
                        "community_count": 0,
                    },
                )

            entity_names = {str(row["id"]): row.get("name", "") for row in entity_rows}
            components = _connected_components(entity_ids, relation_rows)

            community_count = 0
            for component in components:
                memory_rows = await self._load_component_memories(component, request)
                community_id = await self._insert_community(component, entity_names, memory_rows, request)
                await self._backend.execute(
                    "UPDATE knowledge_entities SET community_id = $1::uuid WHERE id = ANY($2::uuid[])",
                    (community_id, component),
                )
                community_count += 1

            await self._propagate_memory_communities(request)
            await self._backend.commit()
            return ManageMemoryResult(
                operation="consolidate",
                communities_updated=community_count,
                diagnostics={
                    "mode": request.mode,
                    "status": "ok",
                    "entity_count": len(entity_rows),
                    "community_count": community_count,
                },
            )
        except Exception:
            await self._backend.rollback()
            raise

    async def _manage_maintain(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Decay, prune, and maintenance operations.

        Supported modes:
        - community_refresh: persist graph-derived communities and assignments
        - decay_scan: recompute relevance scores using exponential decay
        - prune_candidates: identify propositions eligible for pruning
        - expire_quarantine: archive quarantined propositions past grace period
        - expire_flags: unflag expired flagged propositions and resolve conflicts
        """
        mode = request.mode

        if mode == "community_refresh":
            consolidate_result = await self._manage_consolidate(request)
            return consolidate_result.model_copy(update={"operation": "maintain"})

        if mode == "decay_scan":
            return await self._maintain_decay_scan(request)
        if mode == "prune_candidates":
            return await self._maintain_prune_candidates(request)
        if mode == "expire_quarantine":
            return await self._maintain_expire_quarantine(request)
        if mode == "expire_flags":
            return await self._maintain_expire_flags(request)

        return ManageMemoryResult(
            operation="maintain",
            success=False,
            error=f"Unknown maintain mode: {mode!r}. "
            "Expected: community_refresh, decay_scan, prune_candidates, "
            "expire_quarantine, expire_flags",
        )

    async def _maintain_decay_scan(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Batch recompute relevance_score using exponential decay formula.

        Formula: relevance_score = base_score × auth_factor × recency_factor
                                   + retrieval_boost - decay_penalty
        Floors: USER_VALIDATED never drops below 0.5; new props keep base × 0.8.
        """
        decay_rate = request.decay_rate_per_day if request.decay_rate_per_day > 0 else 0.01
        grace_period = request.grace_period_days if request.grace_period_days > 0 else 30
        auto_archive_threshold = 0.1

        authority_factors: dict[str, float] = {
            "USER_VALIDATED": 1.0,
            "CURRENT_SOURCE": 0.8,
            "HISTORICAL_SOURCE": 0.5,
            "EXTRACTED": 0.3,
            "COMMUNITY_SUMMARY": 0.4,
        }

        rows_result = await self._backend.query(
            "SELECT p.id, p.base_score, p.authority, p.created_at, "
            "  p.retrieval_count, p.last_retrieved_at, "
            "  COALESCE(ki.content_updated_at, ki.created_at) AS item_updated_at "
            "FROM knowledge_memories p "
            "LEFT JOIN knowledge_items ki ON p.item_id = ki.id "
            "WHERE p.lifecycle_state IN ('ACTIVE', 'QUARANTINED', 'FLAGGED')",
            (),
        )

        now = datetime.now(UTC)
        below_threshold = 0
        update_batch: list[tuple[float, str]] = []

        for row in rows_result.rows:
            base = row.get("base_score") or 0.5
            authority = row.get("authority", "EXTRACTED")
            auth_factor = authority_factors.get(authority, 0.3)

            prop_age_days = 0.0
            created_at = row.get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                prop_age_days = max((now - created_at).total_seconds() / 86400, 0)

            in_grace_period = prop_age_days < grace_period

            content_age_days = 0.0
            item_updated = row.get("item_updated_at")
            if item_updated:
                if isinstance(item_updated, str):
                    item_updated = datetime.fromisoformat(item_updated)
                if item_updated.tzinfo is None:
                    item_updated = item_updated.replace(tzinfo=UTC)
                content_age_days = max((now - item_updated).total_seconds() / 86400, 0)

            recency_factor = math.exp(-0.001 * content_age_days)
            retrieval_boost = min((row.get("retrieval_count") or 0) * 0.1, 1.0)

            decay_penalty = 0.0
            if not in_grace_period:
                days_since_retrieval = 0.0
                last_retrieved = row.get("last_retrieved_at")
                if last_retrieved:
                    if isinstance(last_retrieved, str):
                        last_retrieved = datetime.fromisoformat(last_retrieved)
                    if last_retrieved.tzinfo is None:
                        last_retrieved = last_retrieved.replace(tzinfo=UTC)
                    days_since_retrieval = max(
                        (now - last_retrieved).total_seconds() / 86400, 0
                    )
                elif created_at:
                    days_since_retrieval = max(prop_age_days - grace_period, 0)
                decay_penalty = decay_rate * days_since_retrieval

            score = base * auth_factor * recency_factor + retrieval_boost - decay_penalty
            if in_grace_period:
                score = max(score, base * 0.8)
            if authority == "USER_VALIDATED":
                score = max(score, 0.5)
            score = max(0.0, min(1.0, score))

            if score < auto_archive_threshold:
                below_threshold += 1
            update_batch.append((score, str(row["id"])))

        for i in range(0, len(update_batch), 1000):
            chunk = update_batch[i : i + 1000]
            values_parts: list[str] = []
            params: list[Any] = []
            for j, (score, pid) in enumerate(chunk):
                base_idx = j * 2 + 1
                values_parts.append(f"(${base_idx}::uuid, ${base_idx + 1}::float8)")
                params.extend([pid, score])
            if values_parts:
                await self._backend.execute(
                    "UPDATE knowledge_memories AS p "
                    "SET relevance_score = v.score "
                    f"FROM (VALUES {', '.join(values_parts)}) AS v(id, score) "
                    "WHERE p.id = v.id",
                    tuple(params),
                )

        return ManageMemoryResult(
            operation="maintain",
            assessed_count=len(update_batch),
            below_threshold_count=below_threshold,
        )

    async def _maintain_prune_candidates(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Identify propositions eligible for pruning.

        Two-tier split:
        - auto_archive_ids: score < auto_archive_threshold AND no entity links
        - needs_review: score < review_threshold WITH entity links or between thresholds
        USER_VALIDATED propositions are excluded.
        """
        auto_threshold = request.auto_archive_threshold
        review_thr = request.review_threshold

        rows_result = await self._backend.query(
            "SELECT p.id, p.content, p.relevance_score, p.authority, "
            "  p.lifecycle_state, p.created_at, p.last_retrieved_at, "
            "  COALESCE(lc.link_count, 0) AS entity_link_count "
            "FROM knowledge_memories p "
            "LEFT JOIN ("
            "  SELECT evidence_memory_id, COUNT(*) AS link_count "
            "  FROM knowledge_relations "
            "  WHERE evidence_memory_id IS NOT NULL "
            "  GROUP BY evidence_memory_id"
            ") lc ON p.id = lc.evidence_memory_id "
            "WHERE p.relevance_score < $1 "
            "  AND p.lifecycle_state IN ('ACTIVE', 'QUARANTINED', 'FLAGGED') "
            "  AND p.authority != 'USER_VALIDATED' "
            "ORDER BY p.relevance_score ASC",
            (review_thr,),
        )

        now = datetime.now(UTC)
        auto_archive_ids: list[str] = []
        needs_review: list[dict[str, Any]] = []

        for row in rows_result.rows:
            link_count = row.get("entity_link_count", 0)
            score = row.get("relevance_score", 0.0)
            prop_id = str(row["id"])

            days = 0
            last_ret = row.get("last_retrieved_at")
            created = row.get("created_at")
            ref_time = last_ret or created
            if ref_time:
                if isinstance(ref_time, str):
                    ref_time = datetime.fromisoformat(ref_time)
                if ref_time.tzinfo is None:
                    ref_time = ref_time.replace(tzinfo=UTC)
                days = int((now - ref_time).total_seconds() / 86400)

            if score < auto_threshold and link_count == 0:
                auto_archive_ids.append(prop_id)
            else:
                needs_review.append(
                    {
                        "memory_id": prop_id,
                        "content": (row.get("content") or "")[:500],
                        "relevance_score": score,
                        "authority": row.get("authority", ""),
                        "lifecycle_state": row.get("lifecycle_state", ""),
                        "entity_link_count": link_count,
                        "days_since_retrieval": days,
                    }
                )

        return ManageMemoryResult(
            operation="maintain",
            auto_archive_ids=auto_archive_ids,
            needs_review=needs_review,
            prune_candidates=needs_review,
        )

    async def _maintain_expire_quarantine(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Archive quarantined propositions that exceeded the grace period."""
        result = await self._backend.query(
            "UPDATE knowledge_memories "
            "SET lifecycle_state = 'ARCHIVED', "
            "    archived_at = NOW(), "
            "    archive_reason = 'quarantine_expiry' "
            "WHERE lifecycle_state = 'QUARANTINED' "
            "  AND quarantined_at < (NOW() - ($1 || ' days')::interval) "
            "RETURNING id",
            (str(request.grace_days),),
        )
        archived_ids = [str(r["id"]) for r in (result.rows or [])]
        archived = len(archived_ids)
        if archived_ids:
            performed_by = _get_audit_user_id(self._context)
            auth_method = _get_auth_method(self._context)
            user_string = _get_user_string_id(self._context)
            for memory_id in archived_ids:
                await self._log_audit_entry(
                    memory_id=memory_id,
                    action="ARCHIVED",
                    performed_by=performed_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={"reason": "quarantine_expiry", "grace_days": request.grace_days},
                )
        return ManageMemoryResult(
            operation="maintain",
            archived_count=archived,
        )

    async def _maintain_expire_flags(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Unflag expired flagged propositions and auto-resolve their conflicts."""
        expired_rows = await self._backend.query(
            "UPDATE knowledge_memories "
            "SET lifecycle_state = 'ACTIVE', flagged_at = NULL "
            "WHERE lifecycle_state = 'FLAGGED' "
            "  AND flagged_at < (NOW() - ($1 || ' days')::interval) "
            "RETURNING id",
            (str(request.grace_days),),
        )
        expired_ids = [str(row["id"]) for row in expired_rows.rows]
        expired_count = len(expired_ids)
        if expired_ids:
            performed_by = _get_audit_user_id(self._context)
            auth_method = _get_auth_method(self._context)
            user_string = _get_user_string_id(self._context)
            for memory_id in expired_ids:
                await self._log_audit_entry(
                    memory_id=memory_id,
                    action="UNFLAGGED",
                    performed_by=performed_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={"reason": "flag_expiry", "grace_days": request.grace_days},
                )

        resolved_count = 0
        if expired_ids:
            id_placeholders = ", ".join(
                f"${i + 1}::uuid" for i in range(len(expired_ids))
            )
            conflict_rows = await self._backend.query(
                "SELECT id FROM knowledge_conflicts "
                f"WHERE new_memory_id IN ({id_placeholders}) "
                "AND resolved_at IS NULL",
                (*expired_ids,),
            )
            if conflict_rows.rows:
                conflict_ids = [str(row["id"]) for row in conflict_rows.rows]
                c_placeholders = ", ".join(
                    f"${i + 1}::uuid" for i in range(len(conflict_ids))
                )
                resolve_result = await self._backend.execute(
                    "UPDATE knowledge_conflicts "
                    "SET resolved_at = NOW(), resolution = 'auto_expired' "
                    f"WHERE id IN ({c_placeholders})",
                    (*conflict_ids,),
                )
                resolved_count = resolve_result.affected_rows if resolve_result else 0

        return ManageMemoryResult(
            operation="maintain",
            expired_count=expired_count,
            resolved_count=resolved_count,
        )

    async def _manage_context(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Assemble token-budgeted context."""
        if not request.query:
            return ManageMemoryResult(
                operation="context",
                success=False,
                error="'query' is required for context operation",
            )
        result = await self._query_context(
            QueryMemoryRequest(
                query=request.query,
                strategy="context",
                scope={"corridor": request.corridor} if request.corridor else None,
                as_of=request.as_of,
                from_=request.from_,
                to=request.to,
                max_items=request.max_items,
                max_tokens=request.max_tokens,
                namespace=request.namespace,
                room=request.room,
                source=request.source,
                categories=request.categories,
                min_confidence=request.min_confidence,
                lifecycle_state=request.lifecycle_state,
                embedding_profile=request.embedding_profile,
            )
        )
        ctx_text = ""
        mem_count = 0
        tokens = 0
        if result.evidence:
            ev = result.evidence[0]
            ctx_text = ev.get("context_text", "")
            mem_count = ev.get("memory_count", 0)
            tokens = ev.get("tokens_used", 0)

        return ManageMemoryResult(
            operation="context",
            context_text=ctx_text,
            memory_count=mem_count,
            tokens_used=tokens,
        )

    async def _manage_graph_store_entity(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Upsert a named entity into knowledge_entities."""
        if not request.entity_name:
            return ManageMemoryResult(
                operation="graph_store_entity", success=False, error="'entity_name' is required"
            )
        if not request.entity_type:
            return ManageMemoryResult(
                operation="graph_store_entity", success=False, error="'entity_type' is required"
            )

        entity_scope_namespace = _normalize_scope_value(request.namespace)
        entity_scope_room = _normalize_scope_value(request.room)
        entity_scope_corridor = _normalize_scope_value(_get_corridor(request))

        result = await self._backend.query(
            """
            INSERT INTO knowledge_entities (id, entity_type, name, namespace, room, corridor, confidence)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (namespace, room, corridor, entity_type, name) DO UPDATE
                SET confidence = GREATEST(knowledge_entities.confidence, EXCLUDED.confidence)
            RETURNING id
            """,
            (
                str(uuid.uuid4()),
                request.entity_type,
                request.entity_name,
                entity_scope_namespace,
                entity_scope_room,
                entity_scope_corridor,
                request.confidence,
            ),
        )
        eid = str(result.rows[0]["id"]) if result.rows else None
        return ManageMemoryResult(operation="graph_store_entity", success=True, entity_id=eid)

    async def _manage_graph_store_relation(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Store a directed relation between two entities (resolved by UUID or name)."""
        if not request.source_entity:
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error="'source_entity' is required",
            )
        if not request.target_entity:
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error="'target_entity' is required",
            )
        if not request.relation_type:
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error="'relation_type' is required",
            )

        try:
            source_id = await _resolve_entity_id_manage(
                request.source_entity,
                self._backend,
                namespace=request.namespace,
                room=request.room,
                corridor=_get_corridor(request),
            )
        except ValueError as exc:
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error=str(exc),
            )
        if source_id is None:
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error=f"Source entity not found: {request.source_entity!r}",
            )
        try:
            target_id = await _resolve_entity_id_manage(
                request.target_entity,
                self._backend,
                namespace=request.namespace,
                room=request.room,
                corridor=_get_corridor(request),
            )
        except ValueError as exc:
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error=str(exc),
            )
        if target_id is None:
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error=f"Target entity not found: {request.target_entity!r}",
            )

        valid_from, valid_to = _validate_temporal_window(request.valid_from, request.valid_to)
        evidence_id = request.evidence_memory_id or None

        existing_result = await self._backend.query(
            """
            SELECT id
            FROM knowledge_relations
            WHERE source_entity_id = $1::uuid
              AND target_entity_id = $2::uuid
              AND relation_type = $3
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (
                source_id,
                target_id,
                request.relation_type,
            ),
        )

        if existing_result.rows:
            result = await self._backend.query(
                """
                UPDATE knowledge_relations
                SET confidence = GREATEST(confidence, $2),
                    evidence_memory_id = COALESCE($3::uuid, evidence_memory_id),
                    valid_from = COALESCE($4::timestamptz, valid_from),
                    valid_to = COALESCE($5::timestamptz, valid_to)
                WHERE id = $1::uuid
                RETURNING id
                """,
                (
                    str(existing_result.rows[0]["id"]),
                    request.confidence,
                    evidence_id,
                    valid_from,
                    valid_to,
                ),
            )
        else:
            result = await self._backend.query(
                """
                INSERT INTO knowledge_relations
                    (id, source_entity_id, target_entity_id, relation_type,
                     confidence, evidence_memory_id, valid_from, valid_to)
                VALUES
                    ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6::uuid,
                     $7::timestamptz, $8::timestamptz)
                RETURNING id
                """,
                (
                    str(uuid.uuid4()),
                    source_id,
                    target_id,
                    request.relation_type,
                    request.confidence,
                    evidence_id,
                    valid_from,
                    valid_to,
                ),
            )
        rid = str(result.rows[0]["id"]) if result.rows else None
        return ManageMemoryResult(operation="graph_store_relation", success=True, relation_id=rid)

    async def _manage_graph_forget_entity(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Delete entities by UUID list (cascades to relations)."""
        ids = request.entity_ids or []
        if not ids:
            return ManageMemoryResult(
                operation="graph_forget_entity",
                success=False,
                error="'entity_ids' is required",
            )

        placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
        relation_placeholders = ", ".join(f"${len(ids) + i + 1}::uuid" for i in range(len(ids)))
        relation_count_result = await self._backend.query(
            f"""
            SELECT COUNT(*) AS cnt
            FROM knowledge_relations
            WHERE source_entity_id IN ({placeholders})
               OR target_entity_id IN ({relation_placeholders})
            """,
            tuple(ids) + tuple(ids),
        )
        deleted_relations = int(relation_count_result.rows[0]["cnt"]) if relation_count_result.rows else 0
        result = await self._backend.query(
            f"DELETE FROM knowledge_entities WHERE id IN ({placeholders}) RETURNING id",
            tuple(ids),
        )
        return ManageMemoryResult(
            operation="graph_forget_entity",
            success=True,
            deleted_entity_count=len(result.rows),
            deleted_relation_count=deleted_relations,
        )

    async def _manage_graph_forget_relation(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Delete relations by UUID list."""
        ids = request.relation_ids or []
        if not ids:
            return ManageMemoryResult(
                operation="graph_forget_relation",
                success=False,
                error="'relation_ids' is required",
            )

        placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ids)))
        result = await self._backend.query(
            f"DELETE FROM knowledge_relations WHERE id IN ({placeholders}) RETURNING id",
            tuple(ids),
        )
        return ManageMemoryResult(
            operation="graph_forget_relation",
            success=True,
            deleted_relation_count=len(result.rows),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _resolve_categories(
        self,
        categories: list[str],
        *,
        allow_create: bool = False,
    ) -> list[str]:
        """Resolve category names or UUIDs to a list of UUID strings.

        For each entry: if it's a valid UUID, use as-is. Otherwise, normalize and
        resolve by normalized-name match. Missing names fail by default and are
        created only when explicitly opted-in.
        """
        resolved: list[str] = []
        missing_names: list[str] = []
        for entry in categories:
            try:
                uuid.UUID(entry)
                resolved.append(entry)
                continue
            except ValueError:
                pass

            normalized_name = _normalize_category_name(entry)
            lookup = await self._backend.query(
                """
                SELECT id
                FROM knowledge_categories
                WHERE lower(regexp_replace(name, '\\s+', ' ', 'g')) = $1
                LIMIT 1
                """,
                (normalized_name,),
            )
            if lookup.rows:
                resolved.append(str(lookup.rows[0]["id"]))
                continue

            if not allow_create:
                missing_names.append(entry)
                continue

            result = await self._backend.query(
                """
                INSERT INTO knowledge_categories (id, name)
                VALUES ($1::uuid, $2)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id, (xmax::text = '0') AS was_inserted
                """,
                (str(uuid.uuid4()), normalized_name),
            )
            if result.rows:
                if result.rows[0].get("was_inserted"):
                    logger.warning("Creating new knowledge category via explicit opt-in: %r", normalized_name)
                resolved.append(str(result.rows[0]["id"]))
            else:
                logger.warning("Category resolution returned no rows for %r", normalized_name)

        if missing_names:
            missing = ", ".join(repr(name) for name in missing_names)
            raise ValueError(
                "Unknown categories: "
                f"{missing}. "
                "Set allow_create_categories=true to explicitly create missing categories."
            )
        return resolved

    async def _log_audit_entry(
        self,
        memory_id: str,
        action: str,
        performed_by: uuid.UUID,
        auth_method: str,
        user_string: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit entry for a memory lifecycle event."""
        metadata_with_user = dict(metadata or {})
        if user_string:
            metadata_with_user["user_identifier"] = user_string
        try:
            await self._backend.execute(
                """
                INSERT INTO knowledge_memory_audits
                    (memory_id, action, performed_by, auth_method, metadata)
                VALUES ($1::uuid, $2, $3::uuid, $4, $5::jsonb)
                """,
                (
                    memory_id,
                    action,
                    str(performed_by),
                    auth_method,
                    json.dumps(metadata_with_user),
                ),
            )
        except Exception as e:
            logger.error(
                "CRITICAL: Failed to log audit entry for memory %s: %s",
                memory_id,
                e,
                extra={"event": "memory.audit.failure", "memory_id": memory_id, "action": action},
            )
            if AUDIT_FAIL_CLOSED:
                raise RuntimeError(f"Audit logging failed: {e}") from e

    async def _load_scoped_entities(
        self,
        request: ManageMemoryRequest,
    ) -> list[dict[str, Any]]:
        """Load the entity set participating in the requested memory scope."""
        corridor = _get_corridor(request)
        clauses, params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="km",
        )
        where_clause = " AND ".join(clauses) if clauses else "TRUE"
        result = await self._backend.query(
            f"""
            SELECT DISTINCT e.id, e.entity_type, e.name
            FROM knowledge_entities e
            JOIN knowledge_entity_memories kem ON kem.entity_id = e.id
            JOIN knowledge_memories km ON km.id = kem.memory_id
            WHERE {where_clause}
            ORDER BY e.name, e.id
            """,
            tuple(params),
        )
        return [dict(row) for row in result.rows]

    async def _load_scoped_relations(
        self,
        entity_ids: list[str],
        request: ManageMemoryRequest,
    ) -> list[dict[str, Any]]:
        """Load graph edges for the entities inside the requested memory scope."""
        params: list[Any] = [entity_ids]
        clauses = [
            "kr.source_entity_id = ANY($1::uuid[])",
            "kr.target_entity_id = ANY($1::uuid[])",
        ]

        corridor = _get_corridor(request)
        scope_clauses, scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="km",
            start_index=2,
        )
        if scope_clauses:
            clauses.append("kr.evidence_memory_id IS NOT NULL")
            clauses.append(
                "EXISTS ("
                "SELECT 1 FROM knowledge_memories km "
                "WHERE km.id = kr.evidence_memory_id AND "
                + " AND ".join(scope_clauses)
                + ")"
            )
            params.extend(scope_params)

        result = await self._backend.query(
            "SELECT DISTINCT kr.source_entity_id, kr.target_entity_id "
            "FROM knowledge_relations kr "
            f"WHERE {' AND '.join(clauses)}",
            tuple(params),
        )
        return [dict(row) for row in result.rows]

    async def _load_component_memories(
        self,
        entity_ids: list[str],
        request: ManageMemoryRequest,
    ) -> list[dict[str, Any]]:
        """Load distinct memories linked to a connected entity component."""
        params: list[Any] = [entity_ids]
        corridor = _get_corridor(request)
        scope_clauses, scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="km",
            start_index=2,
        )
        where_clause = ["kem.entity_id = ANY($1::uuid[])", *scope_clauses]
        params.extend(scope_params)
        result = await self._backend.query(
            "SELECT DISTINCT km.id, km.content, km.embedding "
            "FROM knowledge_memories km "
            "JOIN knowledge_entity_memories kem ON kem.memory_id = km.id "
            f"WHERE {' AND '.join(where_clause)} "
            "ORDER BY km.id",
            tuple(params),
        )
        return [dict(row) for row in result.rows]

    async def _clear_communities(
        self,
        request: ManageMemoryRequest,
        entity_ids: list[str],
    ) -> None:
        """Clear stale community rows and assignments for the targeted scope."""
        if entity_ids:
            await self._backend.execute(
                "UPDATE knowledge_entities SET community_id = NULL WHERE id = ANY($1::uuid[])",
                (entity_ids,),
            )

        corridor = _get_corridor(request)
        scope_clauses, scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="knowledge_memories",
        )
        if scope_clauses:
            await self._backend.execute(
                "UPDATE knowledge_memories SET community_id = NULL WHERE " + " AND ".join(scope_clauses),
                tuple(scope_params),
            )
        else:
            await self._backend.execute("UPDATE knowledge_memories SET community_id = NULL", ())

        community_scope_clauses, community_scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="knowledge_communities",
        )
        if community_scope_clauses:
            await self._backend.execute(
                "DELETE FROM knowledge_communities WHERE " + " AND ".join(community_scope_clauses),
                tuple(community_scope_params),
            )
        else:
            await self._backend.execute("DELETE FROM knowledge_communities", ())

    async def _insert_community(
        self,
        entity_ids: list[str],
        entity_names: dict[str, str],
        memory_rows: list[dict[str, Any]],
        request: ManageMemoryRequest,
    ) -> str:
        """Persist one connected entity component as a concrete community row."""
        unique_vectors: dict[str, list[float]] = {}
        for row in memory_rows:
            parsed = _parse_embedding(row.get("embedding"))
            if parsed is None:
                continue
            unique_vectors[str(row["id"])] = parsed
        centroid = _average_embeddings(list(unique_vectors.values()))

        ordered_names = sorted(entity_names[entity_id] for entity_id in entity_ids if entity_names.get(entity_id))
        summary_names = ordered_names[:5]
        content = "Community: " + ", ".join(summary_names) if summary_names else "Community"

        result = await self._backend.query(
            """
            INSERT INTO knowledge_communities
                (id, content, embedding, member_count, memory_count, namespace, room, corridor)
            VALUES
                ($1::uuid, $2, $3::vector, $4, $5, $6, $7, $8)
            RETURNING id
            """,
            (
                str(uuid.uuid4()),
                content,
                str(centroid) if centroid is not None else None,
                len(entity_ids),
                len({str(row["id"]) for row in memory_rows}),
                request.namespace,
                request.room,
                _get_corridor(request),
            ),
        )
        if not result.rows:
            raise RuntimeError("community insert returned no id")
        return str(result.rows[0]["id"])

    async def _propagate_memory_communities(self, request: ManageMemoryRequest) -> None:
        """Propagate one dominant community assignment per memory from linked entities."""
        corridor = _get_corridor(request)
        clauses = ["ke.community_id IS NOT NULL"]
        params: list[Any] = []
        scope_clauses, scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="km",
        )
        clauses.extend(scope_clauses)
        params.extend(scope_params)

        result = await self._backend.query(
            "SELECT km.id AS memory_id, ke.community_id, kem.confidence AS link_confidence "
            "FROM knowledge_memories km "
            "JOIN knowledge_entity_memories kem ON kem.memory_id = km.id "
            "JOIN knowledge_entities ke ON ke.id = kem.entity_id "
            f"WHERE {' AND '.join(clauses)}",
            tuple(params),
        )

        aggregated: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in result.rows:
            memory_id = str(row["memory_id"])
            community_id = str(row["community_id"])
            aggregated[memory_id][community_id] += float(row.get("link_confidence") or 0.0)

        for memory_id, scores in aggregated.items():
            winning_community = min(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            await self._backend.execute(
                "UPDATE knowledge_memories SET community_id = $1::uuid WHERE id = $2::uuid",
                (winning_community, memory_id),
            )

    async def _upsert_structured_entity(
        self,
        entity_name: str,
        entity_type: str,
        confidence: float | None,
        *,
        namespace: str,
        room: str,
        corridor: str,
    ) -> str:
        """Upsert a structured entity and return its UUID string."""
        result = await self._backend.query(
            """
            INSERT INTO knowledge_entities (id, entity_type, name, namespace, room, corridor, confidence)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (namespace, room, corridor, entity_type, name) DO UPDATE
                SET confidence = GREATEST(knowledge_entities.confidence, EXCLUDED.confidence)
            RETURNING id
            """,
            (
                str(uuid.uuid4()),
                entity_type,
                entity_name,
                namespace,
                room,
                corridor,
                confidence if confidence is not None else 1.0,
            ),
        )
        if not result.rows:
            raise RuntimeError(
                f"Failed to upsert entity {entity_type!r}:{entity_name!r} during ingest_structured"
            )
        return str(result.rows[0]["id"])

    async def _resolve_or_upsert_structured_entity(
        self,
        entity_ids_by_key: dict[tuple[str, str], str],
        entity_type: str,
        entity_name: str,
        confidence: float,
        *,
        namespace: str,
        room: str,
        corridor: str,
    ) -> str:
        """Resolve an entity from this payload, creating it if needed."""
        entity_key = (entity_type, entity_name)
        entity_id = entity_ids_by_key.get(entity_key)
        if entity_id is not None:
            return entity_id
        entity_id = await self._upsert_structured_entity(
            entity_name=entity_name,
            entity_type=entity_type,
            confidence=confidence,
            namespace=namespace,
            room=room,
            corridor=corridor,
        )
        entity_ids_by_key[entity_key] = entity_id
        return entity_id

    async def _link_memory_entity(
        self,
        memory_id: str,
        entity_id: str,
        confidence: float,
    ) -> None:
        """Upsert a direct memory-entity link for structured ingest flows."""
        await self._backend.execute(
            """
            INSERT INTO knowledge_entity_memories
                (memory_id, entity_id, role, confidence)
            VALUES ($1::uuid, $2::uuid, $3, $4)
            ON CONFLICT (memory_id, entity_id) DO UPDATE SET
                confidence = GREATEST(
                    knowledge_entity_memories.confidence,
                    EXCLUDED.confidence
                )
            """,
            (
                memory_id,
                entity_id,
                "mentioned",
                confidence,
            ),
        )

    def _memory_id_for_index(self, memory_ids: list[str], memory_index: int) -> str:
        """Resolve a structured memory index to the stored UUID string."""
        if memory_index < 0 or memory_index >= len(memory_ids):
            raise ValueError(f"memory index out of range: {memory_index}")
        return memory_ids[memory_index]
