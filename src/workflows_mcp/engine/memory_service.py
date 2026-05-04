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
    GraphResult,
    graph_neighbors,
    graph_path,
    graph_stats,
    graph_traverse,
)
from .knowledge.search import room_scoped_search
from .memory_errors import MemoryContractError as MemoryContractError
from .memory_errors import _raise_contract_error
from .memory_locality import CONTRACT_SCOPE_FIELDS, LocalityRequest, resolve_memory_locality

logger = logging.getLogger(__name__)

# SECURITY: System user UUID for audit trail when no user context is available
SYSTEM_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")

# SECURITY: Audit fail-closed configuration
AUDIT_FAIL_CLOSED = os.getenv("AUDIT_FAIL_CLOSED", "false").lower() == "true"


MemoryOperation = Literal[
    "query",
    "ingest",
    "validate",
    "supersede",
    "archive",
    "maintain",
    "graph_upsert",
    "graph_delete",
    "ensure_source",
    "ensure_item",
    "store_entities",
    "store_relations",
    "store_memories",
    "store_entity_embeddings",
    "archive_memories",
    "mark_item_dirty",
]


MEMORY_OPERATION_ENUM: tuple[MemoryOperation, ...] = (
    "query",
    "ingest",
    "validate",
    "supersede",
    "archive",
    "maintain",
    "graph_upsert",
    "graph_delete",
    "ensure_source",
    "ensure_item",
    "store_entities",
    "store_relations",
    "store_memories",
    "store_entity_embeddings",
    "archive_memories",
    "mark_item_dirty",
)

MEMORY_SECTION_REQUIRED_BY_OPERATION: dict[MemoryOperation, str] = {
    "query": "query",
    "ingest": "record",
    "validate": "record",
    "supersede": "record",
    "archive": "record",
    "graph_upsert": "graph",
    "graph_delete": "graph",
}


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


def _is_corridor_relation(relation_type: str | None) -> bool:
    """Return True when relation_type denotes a corridor edge."""
    return (relation_type or "").strip().upper() == "CORRIDOR"


async def _resolve_entity_id_manage(
    entity_ref: str,
    backend: Any,
    *,
    namespace: str | None,
    room: str | None,
    corridor: str | None,
    palace: str | None = None,
) -> str | None:
    """Resolve entity UUID or name to UUID string for manage operations."""
    try:
        uuid.UUID(entity_ref)
        if palace is not None:
            result = await backend.query(
                "SELECT id FROM knowledge_entities WHERE id = $1::uuid AND palace = $2",
                (entity_ref, palace),
            )
        else:
            result = await backend.query(
                "SELECT id FROM knowledge_entities WHERE id = $1::uuid", (entity_ref,)
            )
        return str(result.rows[0]["id"]) if result.rows else None
    except (ValueError, AttributeError):
        pass

    normalized_namespace = _normalize_scope_value(namespace)
    normalized_room = _normalize_scope_value(room)
    normalized_corridor = _normalize_scope_value(corridor)

    clauses = [
        "name = $1",
        "namespace = $2",
        "room = $3",
        "corridor = $4",
    ]
    params: list[Any] = [entity_ref, normalized_namespace, normalized_room, normalized_corridor]
    if palace is not None:
        clauses.append(f"palace = ${len(params) + 1}")
        params.append(palace)

    result = await backend.query(
        "SELECT id FROM knowledge_entities WHERE "
        + " AND ".join(clauses)
        + " ORDER BY id LIMIT 2",
        tuple(params),
    )
    if len(result.rows) > 1:
        raise ValueError(
            f"Entity name is ambiguous in this scope. Use an entity UUID instead: {entity_ref!r}"
        )
    return str(result.rows[0]["id"]) if result.rows else None


def _get_corridor(request: QueryMemoryRequest | ManageMemoryRequest) -> str | None:
    """Resolve corridor from explicit request field or optional scope bag.

    The external API surface uses ``compartment`` (MemoryScope vocabulary);
    ``corridor`` is the internal DB column name.  Both are accepted here so
    that callers using either form are handled uniformly.
    """
    corridor = getattr(request, "corridor", None)
    if corridor:
        return str(corridor)
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        # Accept the external "compartment" alias as well as the internal "corridor" key.
        raw_corridor = scope.get("corridor") or scope.get("compartment")
        if isinstance(raw_corridor, str) and raw_corridor:
            return raw_corridor
    elif scope is not None:
        # MemoryScope model: compartment is the public name for the DB corridor column.
        raw_compartment = getattr(scope, "compartment", None)
        if isinstance(raw_compartment, str) and raw_compartment:
            return raw_compartment
    return None


def _get_palace(request: QueryMemoryRequest | ManageMemoryRequest) -> str | None:
    """Resolve palace from explicit request field or optional scope bag."""
    palace = getattr(request, "palace", None)
    if palace:
        return str(palace)
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        raw_palace = scope.get("palace")
        if isinstance(raw_palace, str) and raw_palace:
            return raw_palace
    return None


def _has_explicit_scope(request: QueryMemoryRequest) -> bool:
    """Return True when any explicit topology scope is provided."""
    return bool(request.namespace or request.room or _get_corridor(request) or _get_palace(request))


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


def _build_retrieval_diagnostics(
    *,
    candidate_generation: str,
    algorithm: str,
    s2_enabled: bool,
    s2_requested: bool,
    s2_strategy: str,
) -> dict[str, Any]:
    """Build a stable retrieval diagnostics envelope across all query strategies."""
    return {
        "s1": {
            "candidate_generation": candidate_generation,
            "algorithm": algorithm,
        },
        "s2": {
            "enabled": s2_enabled,
            "requested": s2_requested,
            "strategy": s2_strategy,
        },
    }


def _base_query_diagnostics(
    *,
    scope_mode: str,
    scope_applied: bool,
    has_results: bool,
    missing_scope: bool = False,
    strategy: str | None = None,
    retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized diagnostics contract for query-mode responses."""
    diagnostics: dict[str, Any] = {
        **_build_scope_diagnostics(
            scope_mode=scope_mode,
            scope_applied=scope_applied,
            has_results=has_results,
            missing_scope=missing_scope,
        ),
        "retrieval": retrieval
        or _build_retrieval_diagnostics(
            candidate_generation="not_applicable",
            algorithm="not_applicable",
            s2_enabled=False,
            s2_requested=False,
            s2_strategy="not_applicable",
        ),
    }
    if strategy is not None:
        diagnostics["strategy"] = strategy
    return diagnostics


_AUTHORITY_ROUTING_FACTS_USER_VALIDATED = "facts_user_validated"

_MEMORY_TIER_DIRECT = "direct"
_MEMORY_TIER_DERIVED = "derived"
_DERIVED_KIND_COMMUNITY = "community"


def _build_memory_scope_filters(
    namespace: str | None,
    room: str | None,
    corridor: str | None,
    *,
    alias: str,
    start_index: int = 1,
    palace: str | None = None,
) -> tuple[list[str], list[Any], int]:
    """Build SQL filter fragments for scoped memory queries."""
    clauses: list[str] = []
    params: list[Any] = []
    next_index = start_index

    if palace is not None:
        clauses.append(f"{alias}.palace = ${next_index}")
        params.append(palace)
        next_index += 1
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
    s2_enabled: bool = Field(
        default=True,
        description="Enable S2 companion-lane retrieval for scoped hybrid searches",
    )
    scope: dict[str, Any] | None = Field(
        default=None,
        description="Optional scope: palace, wing, room, corridor, time window",
    )
    as_of: str | None = Field(
        default=None,
        description="Point-in-time filter (ISO datetime)",
    )
    from_: str | None = Field(
        default=None, alias="from", description="Interval start (ISO datetime)"
    )
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
    palace: str | None = Field(
        default=None,
        description="Palace (org-level scope) for strict palace-scoped retrieval",
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
        "ensure_source",
        "ensure_item",
        "store_entities",
        "store_relations",
        "store_memories",
        "store_entity_embeddings",
        "archive_memories",
        "mark_item_dirty",
    ] = Field(description="Operation family to execute")

    # Structured ingest (strict typed records with extra="forbid")
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

    # Raw dict lists for new low-level ops (store_entities, store_relations, store_memories)
    raw_entities: list[dict[str, Any]] | None = Field(
        default=None,
        description="Raw entity dicts for store_entities",
    )
    raw_relations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Raw relation dicts for store_relations",
    )
    raw_memories: list[dict[str, Any]] | None = Field(
        default=None,
        description="Raw memory dicts for store_memories",
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
    palace: str | None = Field(default=None, description="Palace (org-level scope)")
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
    from_: str | None = Field(
        default=None, alias="from", description="Interval start (ISO datetime)"
    )
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
    evidence_memory_ids: list[str] | None = Field(
        default=None,
        description="Optional memory UUIDs as supporting evidence for the relation",
    )
    curated: bool = Field(
        default=False,
        description=(
            "When false, relation writes must provide evidence linkage. "
            "When true, relation is explicitly curated."
        ),
    )
    entity_ids: list[str] | None = Field(
        default=None, description="Entity UUIDs for graph_forget_entity"
    )
    relation_ids: list[str] | None = Field(
        default=None, description="Relation UUIDs for graph_forget_relation"
    )

    # New executor ops (Tasks 4-11)
    item_id: str | None = Field(default=None, description="knowledge_items.id for item-scoped ops")
    content_hash: str | None = Field(default=None, description="Stable content hash")
    size_bytes: int | None = Field(default=None, ge=0, description="File size in bytes")
    mtime_ns: int | None = Field(default=None, ge=0, description="File mtime in nanoseconds")
    language: str | None = Field(default=None, description="Programming language tag")
    error_metadata: dict[str, Any] | None = Field(
        default=None, description="Error metadata for mark_item_dirty"
    )
    entity_embeddings: list[MemoryEntityEmbeddingInput] | None = Field(
        default=None, description="Embeddings for store_entity_embeddings"
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

    # New executor op outputs (Tasks 4-11)
    source_id: str | None = Field(default=None, description="Upserted knowledge_sources.id")
    item_id: str | None = Field(default=None, description="Upserted knowledge_items.id")


# ---------------------------------------------------------------------------
# Unified Memory Contract Models
# ---------------------------------------------------------------------------


class MemoryScope(BaseModel):
    """Topology scope aligned with MemPalace terminology."""

    model_config = ConfigDict(extra="forbid")

    palace: str | None = Field(default=None, description="Organization-level scope")
    wing: str | None = Field(default=None, description="Service/project scope")
    room: str | None = Field(default=None, description="Component scope")
    compartment: str | None = Field(default=None, description="Compartment scope")


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
    mode: Literal["search", "context", "graph", "hybrid", "communities"] = Field(default="search")
    radius: int = Field(default=1, ge=0, description="Topology expansion distance")
    precision: float = Field(default=0.5, ge=0.0, le=1.0, description="Semantic strictness")
    s2_enabled: bool = Field(
        default=True,
        description="Enable S2 companion-lane retrieval for scoped hybrid searches",
    )
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


class MemoryItemInput(BaseModel):
    """File / item identity payload for ensure_item, mark_item_dirty, archive_memories."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(
        default=None, description="Existing knowledge_items.id (uuid). None means lookup-or-create."
    )
    content_hash: str | None = Field(
        default=None, description="Stable content hash for delta detection (§3.1)."
    )
    size_bytes: int | None = Field(default=None, ge=0)
    mtime_ns: int | None = Field(default=None, ge=0)
    language: str | None = Field(
        default=None,
        description="Language tag (e.g. 'python'); informs structural parser selection.",
    )
    error_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Failure reason for mark_item_dirty (free-form jsonb).",
    )


class MemoryEntityEmbeddingInput(BaseModel):
    """One row for store_entity_embeddings."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="knowledge_entities.id (uuid)")
    profile: str = Field(default="embedding", description="Embedding profile name")
    model: str = Field(description="Model identifier as returned by compute_embedding")
    dimension: int = Field(ge=1, description="Vector dimension; must equal len(embedding)")
    embedding: list[float] = Field(description="Vector; length must match model output")


class MemoryRecordInput(BaseModel):
    """Write/lifecycle payload."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["raw", "structured"] = Field(default="raw")
    memory_tier: Literal["direct", "derived"] = Field(default="direct")
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
    item: MemoryItemInput | None = Field(
        default=None,
        description="File/item identity for ensure_item/mark_item_dirty/archive_memories.",
    )
    entity_embeddings: list[MemoryEntityEmbeddingInput] | None = Field(
        default=None,
        description="Entity embeddings for store_entity_embeddings.",
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
    evidence_memory_ids: list[str] | None = Field(default=None)
    curated: bool = Field(default=False)
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

    operation: MemoryOperation
    scope: MemoryScope = Field(default_factory=MemoryScope)
    scope_token: str | None = Field(default=None)
    context_id: str | None = Field(default=None)
    query: MemoryQueryInput | None = Field(default=None)
    record: MemoryRecordInput | None = Field(default=None)
    graph: MemoryGraphInput | None = Field(default=None)
    maintenance: MemoryMaintenanceInput | None = Field(default=None)
    response: MemoryResponseInput = Field(default_factory=MemoryResponseInput)

    @model_validator(mode="before")
    @classmethod
    def validate_operation(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        operation = data.get("operation")
        if not isinstance(operation, str) or operation not in MEMORY_OPERATION_ENUM:
            _raise_contract_error(
                code="MEM_INVALID_OPERATION",
                message=(f"operation must be one of: {', '.join(MEMORY_OPERATION_ENUM)}"),
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_scope_taxonomy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw_scope = data.get("scope")
        if not isinstance(raw_scope, dict):
            return data

        if "hall" in raw_scope:
            _raise_contract_error(
                code="MEM_INVALID_TAXONOMY_KEY",
                message="'hall' is invalid in memory.v2 scope; use 'compartment'",
            )

        # B-5 / B-2: corridor is an internal DB column, not an external scope key
        if "corridor" in raw_scope:
            _raise_contract_error(
                code="MEM_INVALID_TAXONOMY_KEY",
                message=(
                    "'corridor' is an internal topology term and is forbidden in external scope; "
                    "use 'compartment' instead"
                ),
            )

        # B-1: Forbid any key outside the four canonical external scope fields
        invalid_keys = sorted(key for key in raw_scope if key not in CONTRACT_SCOPE_FIELDS)
        if invalid_keys:
            _raise_contract_error(
                code="MEM_INVALID_TAXONOMY_KEY",
                message=f"Unsupported scope keys: {', '.join(invalid_keys)}",
            )

        # B-1: Enforce hierarchy — room requires wing; compartment requires room + wing
        if raw_scope.get("compartment") and not raw_scope.get("room"):
            _raise_contract_error(
                code="MEM_SCOPE_HIERARCHY_VIOLATION",
                message="scope.compartment requires scope.room to be present",
            )
        if raw_scope.get("compartment") and not raw_scope.get("wing"):
            _raise_contract_error(
                code="MEM_SCOPE_HIERARCHY_VIOLATION",
                message="scope.compartment requires scope.wing to be present",
            )
        if raw_scope.get("room") and not raw_scope.get("wing"):
            _raise_contract_error(
                code="MEM_SCOPE_HIERARCHY_VIOLATION",
                message="scope.room requires scope.wing to be present",
            )

        return data

    @model_validator(mode="after")
    def validate_contract_envelope(self) -> MemoryRequest:
        required_section = MEMORY_SECTION_REQUIRED_BY_OPERATION.get(self.operation)
        if required_section is not None and getattr(self, required_section) is None:
            _raise_contract_error(
                code="MEM_MISSING_REQUIRED_FIELD",
                message=(
                    f"'{required_section}' payload is required for operation='{self.operation}'"
                ),
            )

        if self.operation == "ingest" and self.record is not None:
            if self.record.memory_tier != "direct":
                _raise_contract_error(
                    code="MEM_BOUNDARY_VIOLATION",
                    message="record.memory_tier must be 'direct' for ingest",
                )

        return self


class OrgUserMergeTransparency(BaseModel):
    """B-4: Org/user merge transparency fields for two-layer reads."""

    model_config = ConfigDict(extra="forbid")

    effective_source: Literal["org", "user"] = Field(
        description="Which layer produced the effective record (newer updated_at; org wins tie)"
    )
    org_record: dict[str, Any] | None = Field(
        default=None,
        description="Org-layer record when present",
    )
    user_record: dict[str, Any] | None = Field(
        default=None,
        description="User-layer record when present",
    )
    conflict: bool = Field(
        description="True when both org and user records exist and differ"
    )


def _build_merge_transparency(
    *,
    org_record: dict[str, Any] | None,
    user_record: dict[str, Any] | None,
) -> OrgUserMergeTransparency | None:
    """Compute merge transparency when both org and user records are present.

    Merge rule (spec §5.2):
      effective = record with newer updated_at; org wins on tie.
    """
    if org_record is None and user_record is None:
        return None
    if org_record is None:
        return OrgUserMergeTransparency(
            effective_source="user",
            org_record=None,
            user_record=user_record,
            conflict=False,
        )
    if user_record is None:
        return OrgUserMergeTransparency(
            effective_source="org",
            org_record=org_record,
            user_record=None,
            conflict=False,
        )

    # Both present — apply merge rule
    org_updated = org_record.get("updated_at")
    user_updated = user_record.get("updated_at")

    _epoch = datetime.min.replace(tzinfo=UTC)

    def _parse_ts(value: Any) -> datetime:
        """Parse a timestamp string to an aware UTC datetime.

        Handles naive timestamps (no timezone info) by treating them as UTC,
        which matches PostgreSQL's default behavior for timestamp columns.
        Falls back to _epoch on any parse failure.
        """
        if not value:
            return _epoch
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (ValueError, TypeError):
            return _epoch

    org_dt = _parse_ts(org_updated)
    user_dt = _parse_ts(user_updated)

    # Org wins tie (user_dt must be strictly newer to win)
    effective_source: Literal["org", "user"] = "user" if user_dt > org_dt else "org"

    return OrgUserMergeTransparency(
        effective_source=effective_source,
        org_record=org_record,
        user_record=user_record,
        conflict=True,
    )


class MemoryResult(BaseModel):
    """Canonical result envelope."""

    operation: MemoryOperation
    resolved_scope: MemoryScope | None = Field(default=None)
    scope_source: dict[str, Literal["request", "token", "context"]] = Field(default_factory=dict)
    query: QueryMemoryResult | None = Field(default=None)
    manage: ManageMemoryResult | None = Field(default=None)
    # B-4: org/user merge transparency — populated when both layers are queried
    merge: OrgUserMergeTransparency | None = Field(default=None)


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
        exec_context = self._context.execution_context

        locality = resolve_memory_locality(
            LocalityRequest(
                operation=op,
                request_scope=request.scope.model_dump(exclude_none=True),
                scope_token=request.scope_token,
                context_id=request.context_id,
                token_scopes=getattr(exec_context, "memory_scope_tokens", None)
                if exec_context
                else None,
                context_scopes=getattr(exec_context, "memory_context_scopes", None)
                if exec_context
                else None,
                graph_kind=request.graph.kind if request.graph is not None else None,
                graph_from=request.graph.from_ref if request.graph is not None else None,
                graph_to=request.graph.to_ref if request.graph is not None else None,
                graph_ids=request.graph.ids if request.graph is not None else None,
                record_ids=request.record.ids if request.record is not None else None,
                record_superseded_by=request.record.superseded_by
                if request.record is not None
                else None,
            )
        )
        resolved_scope = MemoryScope.model_validate(locality.resolved_scope)
        scope_source = locality.scope_source

        if op == "query":
            if request.query is None:
                raise ValueError("'query' payload is required for operation='query'")

            query_mode = request.query.mode
            strategy: Literal["auto", "communities", "graph", "palace", "context"]
            if query_mode == "context":
                strategy = "context"
            elif query_mode == "graph":
                strategy = "graph"
            elif query_mode == "communities":
                strategy = "communities"
            elif query_mode == "hybrid":
                strategy = "auto"
            elif request.query.radius == 0 and (
                resolved_scope.wing or resolved_scope.room or resolved_scope.compartment
            ):
                strategy = "palace"
            else:
                strategy = "auto"

            query_result = await self.query(
                QueryMemoryRequest.model_validate(
                    {
                        "query": request.query.text,
                        "strategy": strategy,
                        "as_of": request.query.as_of,
                        "from": request.query.from_,
                        "to": request.query.to,
                        "max_tokens": request.query.limits.tokens,
                        "max_items": request.query.limits.items,
                        "palace": resolved_scope.palace,
                        "namespace": resolved_scope.wing,
                        "room": resolved_scope.room,
                        "source": request.query.source,
                        "categories": request.query.categories,
                        "min_confidence": request.query.precision,
                        "start_entity": request.query.graph.start,
                        "end_entity": request.query.graph.end,
                        "graph_op": request.query.graph.op,
                        "relation_types": request.query.graph.relation_types,
                        "max_hops": request.query.limits.hops,
                        "max_nodes": request.query.limits.nodes,
                        "s2_enabled": request.query.s2_enabled,
                        "scope": {"corridor": resolved_scope.compartment},
                    }
                )
            )

            normalized_diagnostics = dict(query_result.diagnostics)
            normalized_diagnostics["effective_strategy"] = strategy
            query_result = query_result.model_copy(update={"diagnostics": normalized_diagnostics})

            # B-4: Wire merge transparency when both org and user layers are present.
            # Org layer = memories (non-USER_VALIDATED authority).
            # User layer = facts (USER_VALIDATED authority).
            merge_transparency = _build_merge_transparency(
                org_record=query_result.memories[0] if query_result.memories else None,
                user_record=query_result.facts[0] if query_result.facts else None,
            )

            return MemoryResult(
                operation=op,
                query=query_result,
                resolved_scope=resolved_scope,
                scope_source=scope_source,
                merge=merge_transparency,
            )

        if op == "ingest":
            if request.record is None:
                raise ValueError("'record' payload is required for operation='ingest'")
            if request.record.format == "structured":
                manage_request = ManageMemoryRequest(
                    operation="ingest_structured",
                    source=request.record.source,
                    path=request.record.path,
                    palace=resolved_scope.palace,
                    namespace=resolved_scope.wing,
                    room=resolved_scope.room,
                    corridor=resolved_scope.compartment,
                    memories=cast(list[StructuredMemoryRecord] | None, request.record.memories),
                    entities=cast(list[StructuredEntityRecord] | None, request.record.entities),
                    relations=cast(list[StructuredRelationRecord] | None, request.record.relations),
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
                    palace=resolved_scope.palace,
                    namespace=resolved_scope.wing,
                    room=resolved_scope.room,
                    corridor=resolved_scope.compartment,
                    confidence=request.record.confidence,
                    authority=request.record.authority,
                    lifecycle_state=request.record.lifecycle_state,
                    categories=request.record.categories,
                    allow_create_categories=request.record.allow_create_categories,
                )
            manage_result = await self.manage(manage_request)
            return MemoryResult(
                operation=op,
                manage=manage_result,
                resolved_scope=resolved_scope,
                scope_source=scope_source,
            )

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
            return MemoryResult(
                operation=op,
                manage=manage_result,
                resolved_scope=resolved_scope,
                scope_source=scope_source,
            )

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
                    palace=resolved_scope.palace,
                    namespace=resolved_scope.wing,
                    room=resolved_scope.room,
                    corridor=resolved_scope.compartment,
                )
            )
            return MemoryResult(
                operation=op,
                manage=manage_result,
                resolved_scope=resolved_scope,
                scope_source=scope_source,
            )

        if op == "graph_upsert":
            if request.graph is None:
                raise ValueError("'graph' payload is required for operation='graph_upsert'")
            if request.graph.kind == "place":
                manage_request = ManageMemoryRequest(
                    operation="graph_store_entity",
                    entity_name=request.graph.place_name,
                    entity_type=request.graph.place_type,
                    palace=resolved_scope.palace,
                    namespace=resolved_scope.wing,
                    room=resolved_scope.room,
                    corridor=resolved_scope.compartment,
                )
            else:
                effective_evidence_ids = list(request.graph.evidence_memory_ids or [])
                if request.graph.evidence_memory_id:
                    effective_evidence_ids.append(request.graph.evidence_memory_id)
                effective_evidence_ids = list(dict.fromkeys(effective_evidence_ids))

                if (
                    _is_corridor_relation(request.graph.link_type)
                    and not request.graph.curated
                    and not effective_evidence_ids
                ):
                    _raise_contract_error(
                        code="MEM_GRAPH_EVIDENCE_REQUIRED",
                        message=(
                            "graph_upsert CORRIDOR link requires evidence_memory_ids "
                            "when curated=false"
                        ),
                    )

                manage_request = ManageMemoryRequest(
                    operation="graph_store_relation",
                    source_entity=request.graph.from_ref,
                    target_entity=request.graph.to_ref,
                    relation_type=request.graph.link_type,
                    evidence_memory_id=request.graph.evidence_memory_id,
                    evidence_memory_ids=effective_evidence_ids,
                    curated=request.graph.curated,
                    palace=resolved_scope.palace,
                    namespace=resolved_scope.wing,
                    room=resolved_scope.room,
                    corridor=resolved_scope.compartment,
                )
            manage_result = await self.manage(manage_request)
            return MemoryResult(
                operation=op,
                manage=manage_result,
                resolved_scope=resolved_scope,
                scope_source=scope_source,
            )

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
            return MemoryResult(
                operation=op,
                manage=manage_result,
                resolved_scope=resolved_scope,
                scope_source=scope_source,
            )

        if op in {
            "ensure_source",
            "ensure_item",
            "store_entities",
            "store_relations",
            "store_memories",
            "store_entity_embeddings",
            "archive_memories",
            "mark_item_dirty",
        }:
            manage_result = await self._dispatch_new_op(op, request, resolved_scope)
            return MemoryResult(
                operation=op,
                manage=manage_result,
                resolved_scope=resolved_scope,
                scope_source=scope_source,
            )

        raise ValueError(f"Unsupported operation: {op}")

    # ------------------------------------------------------------------
    # New executor ops: ensure_source, ensure_item, store_entities,
    # store_relations, store_memories, store_entity_embeddings,
    # archive_memories, mark_item_dirty
    # ------------------------------------------------------------------

    async def _dispatch_new_op(
        self,
        op: str,
        request: MemoryRequest,
        resolved_scope: MemoryScope,
    ) -> ManageMemoryResult:
        """Route the eight new low-level executor ops to their handlers."""
        palace = resolved_scope.palace

        if op == "ensure_source":
            if request.record is None or not request.record.source:
                raise ValueError("'record.source' is required for operation='ensure_source'")
            return await self._manage_ensure_source(
                ManageMemoryRequest(
                    operation="ensure_source",
                    source=request.record.source,
                    source_type="FILE",
                    palace=palace,
                )
            )

        if op == "ensure_item":
            if request.record is None or not request.record.source or not request.record.path:
                raise ValueError(
                    "'record.source' and 'record.path' are required for operation='ensure_item'"
                )
            item_payload = request.record.item or MemoryItemInput()
            return await self._manage_ensure_item(
                ManageMemoryRequest(
                    operation="ensure_item",
                    source=request.record.source,
                    source_type="FILE",
                    path=request.record.path,
                    palace=palace,
                    content_hash=item_payload.content_hash,
                    size_bytes=item_payload.size_bytes,
                    mtime_ns=item_payload.mtime_ns,
                    language=item_payload.language,
                )
            )

        if op == "store_entities":
            if request.record is None or not request.record.entities:
                raise ValueError("'record.entities' is required for operation='store_entities'")
            if not request.record.source:
                raise ValueError("'record.source' is required for operation='store_entities'")
            item_payload = request.record.item or MemoryItemInput()
            return await self._manage_store_entities(
                ManageMemoryRequest(
                    operation="store_entities",
                    source=request.record.source,
                    palace=palace,
                    namespace=resolved_scope.wing,
                    room=resolved_scope.room,
                    corridor=resolved_scope.compartment,
                    item_id=item_payload.id,
                    raw_entities=request.record.entities,
                    confidence=request.record.confidence,
                    authority=request.record.authority,
                )
            )

        if op == "store_relations":
            if request.record is None or not request.record.relations:
                raise ValueError("'record.relations' is required for operation='store_relations'")
            return await self._manage_store_relations(
                ManageMemoryRequest(
                    operation="store_relations",
                    palace=palace,
                    raw_relations=request.record.relations,
                    confidence=request.record.confidence,
                )
            )

        if op == "store_memories":
            if request.record is None or not request.record.memories:
                raise ValueError("'record.memories' is required for operation='store_memories'")
            return await self._manage_store_memories(
                ManageMemoryRequest(
                    operation="store_memories",
                    palace=palace,
                    namespace=resolved_scope.wing,
                    room=resolved_scope.room,
                    corridor=resolved_scope.compartment,
                    raw_memories=request.record.memories,
                    confidence=request.record.confidence,
                    authority=request.record.authority,
                    lifecycle_state=request.record.lifecycle_state,
                )
            )

        if op == "store_entity_embeddings":
            if request.record is None or not request.record.entity_embeddings:
                raise ValueError(
                    "'record.entity_embeddings' is required for operation='store_entity_embeddings'"
                )
            return await self._manage_store_entity_embeddings(
                ManageMemoryRequest(
                    operation="store_entity_embeddings",
                    palace=palace,
                    entity_embeddings=request.record.entity_embeddings,
                )
            )

        if op == "archive_memories":
            if request.record is None:
                raise ValueError("'record' is required for operation='archive_memories'")
            item_payload = request.record.item or MemoryItemInput()
            if not item_payload.id and not request.record.ids:
                raise ValueError(
                    "operation='archive_memories' requires 'record.item.id' or 'record.ids'"
                )
            return await self._manage_archive_memories(
                ManageMemoryRequest(
                    operation="archive_memories",
                    palace=palace,
                    item_id=item_payload.id,
                    memory_ids=request.record.ids,
                    reason=request.record.reason,
                )
            )

        if op == "mark_item_dirty":
            if request.record is None or request.record.item is None or not request.record.item.id:
                raise ValueError("'record.item.id' is required for operation='mark_item_dirty'")
            return await self._manage_mark_item_dirty(
                ManageMemoryRequest(
                    operation="mark_item_dirty",
                    palace=palace,
                    item_id=request.record.item.id,
                    error_metadata=request.record.item.error_metadata,
                )
            )

        raise ValueError(f"Unsupported new op: {op}")

    async def _manage_ensure_source(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Idempotent upsert of knowledge_sources by (palace, name)."""
        if not request.source:
            return ManageMemoryResult(
                operation="ensure_source",
                success=False,
                error="MEM_FIELD_REQUIRED: 'source' is required for ensure_source",
            )
        palace = _normalize_scope_value(_get_palace(request))
        if palace is None:
            return ManageMemoryResult(
                operation="ensure_source",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required for ensure_source",
            )

        result = await self._backend.query(
            """
            INSERT INTO knowledge_sources (id, palace, name, source_type, category_ids)
            VALUES ($1::uuid, $2, $3, $4, '{}'::uuid[])
            ON CONFLICT (palace, name) DO UPDATE SET updated_at = NOW()
            RETURNING id
            """,
            (str(uuid.uuid4()), palace, request.source, request.source_type or "FILE"),
        )
        if not result.rows:
            return ManageMemoryResult(
                operation="ensure_source",
                success=False,
                error="MEM_PERSIST_FAILED: ensure_source returned no row",
            )
        return ManageMemoryResult(
            operation="ensure_source",
            success=True,
            source_id=str(result.rows[0]["id"]),
        )

    async def _manage_ensure_item(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Idempotent upsert of knowledge_items keyed on (palace, source_id, path)."""
        if not request.source or not request.path:
            return ManageMemoryResult(
                operation="ensure_item",
                success=False,
                error="MEM_FIELD_REQUIRED: 'source' and 'path' are required for ensure_item",
            )

        missing_not_null = [
            f for f in ("content_hash", "size_bytes", "mtime_ns")
            if getattr(request, f, None) is None
        ]
        if missing_not_null:
            return ManageMemoryResult(
                operation="ensure_item",
                success=False,
                error=(
                    "MEM_FIELD_REQUIRED: the following NOT NULL fields are required for "
                    f"ensure_item: {', '.join(missing_not_null)}"
                ),
            )

        ensure_source_result = await self._manage_ensure_source(
            ManageMemoryRequest(
                operation="ensure_source",
                source=request.source,
                source_type=request.source_type or "FILE",
                palace=request.palace,
            )
        )
        if not ensure_source_result.success or not ensure_source_result.source_id:
            return ManageMemoryResult(
                operation="ensure_item",
                success=False,
                error=ensure_source_result.error
                or "MEM_PERSIST_FAILED: ensure_item could not resolve source",
            )

        palace = _normalize_scope_value(_get_palace(request))
        if palace is None:
            return ManageMemoryResult(
                operation="ensure_item",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required for ensure_item",
            )

        item_title = os.path.basename(request.path) or request.path
        result = await self._backend.query(
            """
            INSERT INTO knowledge_items
                (id, palace, source_id, path, title,
                 content_hash, size_bytes, mtime_ns, language)
            VALUES
                ($1::uuid, $2, $3::uuid, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (palace, source_id, path) DO UPDATE SET
                title = EXCLUDED.title,
                content_hash = COALESCE(EXCLUDED.content_hash, knowledge_items.content_hash),
                size_bytes = COALESCE(EXCLUDED.size_bytes, knowledge_items.size_bytes),
                mtime_ns = COALESCE(EXCLUDED.mtime_ns, knowledge_items.mtime_ns),
                language = COALESCE(EXCLUDED.language, knowledge_items.language),
                updated_at = NOW()
            RETURNING id
            """,
            (
                str(uuid.uuid4()),
                palace,
                ensure_source_result.source_id,
                request.path,
                item_title,
                request.content_hash,
                request.size_bytes,
                request.mtime_ns,
                request.language,
            ),
        )
        if not result.rows:
            return ManageMemoryResult(
                operation="ensure_item",
                success=False,
                error="MEM_PERSIST_FAILED: ensure_item returned no row",
            )
        return ManageMemoryResult(
            operation="ensure_item",
            success=True,
            source_id=ensure_source_result.source_id,
            item_id=str(result.rows[0]["id"]),
        )

    async def _manage_store_entities(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Bulk upsert structural entities keyed by (palace, source, stable_id)."""
        if not request.raw_entities:
            return ManageMemoryResult(
                operation="store_entities",
                success=False,
                error="MEM_FIELD_REQUIRED: 'entities' is required for store_entities",
            )
        if not request.source:
            return ManageMemoryResult(
                operation="store_entities",
                success=False,
                error="MEM_FIELD_REQUIRED: 'source' is required for store_entities",
            )
        if request.source not in {"STRUCTURAL", "EXTRACTED", "USER"}:
            return ManageMemoryResult(
                operation="store_entities",
                success=False,
                error="MEM_INVALID_SOURCE: 'source' must be STRUCTURAL, EXTRACTED, or USER",
            )

        palace = _normalize_scope_value(_get_palace(request))
        namespace = _normalize_scope_value(request.namespace)
        room = _normalize_scope_value(request.room)
        corridor = _normalize_scope_value(_get_corridor(request))
        if palace is None:
            return ManageMemoryResult(
                operation="store_entities",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required for store_entities",
            )

        entity_ids: list[str] = []
        await self._backend.begin_transaction()
        try:
            for entity in request.raw_entities:
                entity_type = entity["entity_type"]
                name = entity["name"]
                stable_id = entity.get("stable_id")
                metadata = entity.get("metadata") or {}
                confidence = entity.get("confidence")

                row = await self._backend.query(
                    """
                    INSERT INTO knowledge_entities
                        (id, palace, namespace, room, corridor,
                         entity_type, name, source, authority,
                         stable_id, source_item_id, confidence, metadata)
                    VALUES
                        ($1::uuid, $2, $3, $4, $5,
                         $6, $7, $8, $9,
                         $10, $11::uuid, $12, $13::jsonb)
                    ON CONFLICT (palace, source, stable_id)
                        WHERE stable_id IS NOT NULL
                        DO UPDATE SET
                            name = EXCLUDED.name,
                            entity_type = EXCLUDED.entity_type,
                            source_item_id = EXCLUDED.source_item_id,
                            confidence = EXCLUDED.confidence,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        str(uuid.uuid4()),
                        palace,
                        namespace,
                        room,
                        corridor,
                        entity_type,
                        name,
                        request.source,
                        request.authority or "STRUCTURAL",
                        stable_id,
                        request.item_id,
                        confidence if confidence is not None else request.confidence,
                        json.dumps(metadata),
                    ),
                )
                if row.rows:
                    entity_ids.append(str(row.rows[0]["id"]))
            await self._backend.commit()
        except Exception:
            await self._backend.rollback()
            raise

        return ManageMemoryResult(
            operation="store_entities",
            success=True,
            entity_ids=entity_ids,
            entities_stored_count=len(entity_ids),
        )

    async def _manage_store_relations(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Append-only relation insert. Entities must already exist in the same palace."""
        if not request.raw_relations:
            return ManageMemoryResult(
                operation="store_relations",
                success=False,
                error="MEM_FIELD_REQUIRED: 'relations' is required for store_relations",
            )
        palace = _normalize_scope_value(_get_palace(request))
        if palace is None:
            return ManageMemoryResult(
                operation="store_relations",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required for store_relations",
            )

        relation_ids: list[str] = []
        await self._backend.begin_transaction()
        try:
            for relation in request.raw_relations:
                src_id = relation["source_entity_id"]
                tgt_id = relation["target_entity_id"]
                rel_type = relation["relation_type"]
                confidence = relation.get("confidence")
                evidence_ids = relation.get("evidence_memory_ids") or []

                check = await self._backend.query(
                    "SELECT id, palace FROM knowledge_entities "
                    "WHERE id IN ($1::uuid, $2::uuid)",
                    (src_id, tgt_id),
                )
                endpoint_palaces = {str(row["id"]): row["palace"] for row in check.rows}
                if (
                    len(endpoint_palaces) != 2
                    or endpoint_palaces.get(str(src_id)) != palace
                    or endpoint_palaces.get(str(tgt_id)) != palace
                ):
                    raise MemoryContractError(
                        code="MEM_PALACE_MISMATCH",
                        message=(
                            "MEM_PALACE_MISMATCH: store_relations refuses to link entities "
                            "outside the requesting palace"
                        ),
                        retryable=False,
                    )

                row = await self._backend.query(
                    """
                    INSERT INTO knowledge_relations
                        (id, source_entity_id, target_entity_id, relation_type,
                         confidence, evidence_memory_ids)
                    VALUES
                        ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6::uuid[])
                    RETURNING id
                    """,
                    (
                        str(uuid.uuid4()),
                        src_id,
                        tgt_id,
                        rel_type,
                        confidence if confidence is not None else request.confidence,
                        list(evidence_ids),
                    ),
                )
                if row.rows:
                    relation_ids.append(str(row.rows[0]["id"]))
            await self._backend.commit()
        except Exception:
            await self._backend.rollback()
            raise

        return ManageMemoryResult(
            operation="store_relations",
            success=True,
            relation_ids=relation_ids,
            relations_stored_count=len(relation_ids),
        )

    async def _manage_store_memories(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Bulk insert memories with embeddings; optionally link to anchor entities with spans."""
        if not request.raw_memories:
            return ManageMemoryResult(
                operation="store_memories",
                success=False,
                error="MEM_FIELD_REQUIRED: 'memories' is required for store_memories",
            )
        palace = _normalize_scope_value(_get_palace(request))
        if palace is None:
            return ManageMemoryResult(
                operation="store_memories",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required for store_memories",
            )
        if _normalize_scope_value(request.namespace) == "code":
            return ManageMemoryResult(
                operation="store_memories",
                success=False,
                error="MEM_RESERVED_WING: wing='code' is reserved for System 1 structural output",
            )

        created_by = _get_audit_user_id(self._context)
        auth_method = _get_auth_method(self._context)
        user_string = _get_user_string_id(self._context)

        memory_ids: list[str] = []
        await self._backend.begin_transaction()
        try:
            for memory in request.raw_memories:
                content = memory["content"]
                metadata = memory.get("metadata") or {}
                anchor_entity_id = memory.get("anchor_entity_id")
                anchor_kind = memory.get("anchor_kind") or "symbol"
                start_line = memory.get("start_line")
                end_line = memory.get("end_line")
                start_col = memory.get("start_col")
                end_col = memory.get("end_col")
                confidence = memory.get("confidence")
                authority = memory.get("authority") or request.authority
                lifecycle = memory.get("lifecycle_state") or request.lifecycle_state

                memory_id = str(uuid.uuid4())
                memory_ids.append(memory_id)

                try:
                    embedding, model_name, _, _ = await compute_embedding(
                        text=content,
                        context=self._context,
                        profile=request.embedding_profile,
                    )
                except Exception as embed_exc:
                    raise MemoryContractError(
                        code="MEM_EMBEDDING_FAILED",
                        message=(
                            f"MEM_EMBEDDING_FAILED: store_memories failed embedding: {embed_exc}"
                        ),
                        retryable=True,
                    ) from embed_exc

                await self._backend.execute(
                    """
                    INSERT INTO knowledge_memories
                        (id, content, embedding, search_vector,
                         authority, lifecycle_state, confidence, embedding_model,
                         metadata, created_by, auth_method,
                         palace, namespace, room, corridor,
                         memory_tier, derived_kind, parent_memory_ids)
                    VALUES
                        ($1::uuid, $2, $3::vector, to_tsvector('english', $2),
                         $4, $5, $6, $7,
                         $8::jsonb, $9::uuid, $10,
                         $11, $12, $13, $14,
                         $15, $16, $17::uuid[])
                    """,
                    (
                        memory_id,
                        content,
                        str(embedding),
                        authority,
                        lifecycle,
                        confidence if confidence is not None else request.confidence,
                        model_name,
                        json.dumps(metadata),
                        str(created_by),
                        auth_method,
                        palace,
                        request.namespace,
                        request.room,
                        request.corridor,
                        _MEMORY_TIER_DIRECT,
                        None,
                        [],
                    ),
                )
                await self._log_audit_entry(
                    memory_id=memory_id,
                    action="CREATED",
                    performed_by=created_by,
                    auth_method=auth_method,
                    user_string=user_string,
                    metadata={"op": "store_memories"},
                )

                if anchor_entity_id is not None:
                    anchor_check = await self._backend.query(
                        "SELECT palace FROM knowledge_entities WHERE id = $1::uuid",
                        (anchor_entity_id,),
                    )
                    if not anchor_check.rows or anchor_check.rows[0]["palace"] != palace:
                        raise MemoryContractError(
                            code="MEM_PALACE_MISMATCH",
                            message=(
                                "MEM_PALACE_MISMATCH: store_memories refuses to anchor "
                                "a memory to an entity outside the requesting palace"
                            ),
                            retryable=False,
                        )
                    await self._backend.execute(
                        """
                        INSERT INTO knowledge_entity_memories
                            (memory_id, entity_id, confidence,
                             start_line, end_line, start_col, end_col, anchor_kind)
                        VALUES
                            ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (memory_id, entity_id) DO UPDATE SET
                            confidence = GREATEST(
                                knowledge_entity_memories.confidence, EXCLUDED.confidence),
                            start_line = COALESCE(EXCLUDED.start_line,
                                knowledge_entity_memories.start_line),
                            end_line = COALESCE(EXCLUDED.end_line,
                                knowledge_entity_memories.end_line),
                            start_col = COALESCE(EXCLUDED.start_col,
                                knowledge_entity_memories.start_col),
                            end_col = COALESCE(EXCLUDED.end_col,
                                knowledge_entity_memories.end_col),
                            anchor_kind = EXCLUDED.anchor_kind
                        """,
                        (
                            memory_id,
                            anchor_entity_id,
                            confidence if confidence is not None else request.confidence,
                            start_line,
                            end_line,
                            start_col,
                            end_col,
                            anchor_kind,
                        ),
                    )
            await self._backend.commit()
        except Exception:
            await self._backend.rollback()
            raise

        return ManageMemoryResult(
            operation="store_memories",
            success=True,
            memory_ids=memory_ids,
            stored_count=len(memory_ids),
        )

    async def _manage_store_entity_embeddings(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Upsert entity embeddings keyed on (entity_id, profile)."""
        if not request.entity_embeddings:
            return ManageMemoryResult(
                operation="store_entity_embeddings",
                success=False,
                error="MEM_FIELD_REQUIRED: 'entity_embeddings' is required",
            )
        palace = _normalize_scope_value(_get_palace(request))
        if palace is None:
            return ManageMemoryResult(
                operation="store_entity_embeddings",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required",
            )

        await self._backend.begin_transaction()
        try:
            for emb in request.entity_embeddings:
                if emb.dimension != len(emb.embedding):
                    raise MemoryContractError(
                        code="MEM_EMBEDDING_DIMENSION_MISMATCH",
                        message=(
                            "MEM_EMBEDDING_DIMENSION_MISMATCH: "
                            "dimension must equal len(embedding)"
                        ),
                        retryable=False,
                    )
                check = await self._backend.query(
                    "SELECT 1 FROM knowledge_entities WHERE id = $1::uuid AND palace = $2",
                    (emb.entity_id, palace),
                )
                if not check.rows:
                    raise MemoryContractError(
                        code="MEM_PALACE_MISMATCH",
                        message=(
                            "MEM_PALACE_MISMATCH: store_entity_embeddings refuses to write "
                            "for an entity outside the requesting palace"
                        ),
                        retryable=False,
                    )
                await self._backend.execute(
                    """
                    INSERT INTO knowledge_entity_embeddings
                        (entity_id, profile, model, dimension, embedding)
                    VALUES
                        ($1::uuid, $2, $3, $4, $5::vector)
                    ON CONFLICT (entity_id, profile) DO UPDATE SET
                        model = EXCLUDED.model,
                        dimension = EXCLUDED.dimension,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        emb.entity_id,
                        emb.profile,
                        emb.model,
                        emb.dimension,
                        str(emb.embedding),
                    ),
                )
            await self._backend.commit()
        except Exception:
            await self._backend.rollback()
            raise

        return ManageMemoryResult(
            operation="store_entity_embeddings",
            success=True,
            stored_count=len(request.entity_embeddings),
        )

    async def _manage_archive_memories(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Soft-delete memories by item_id or explicit memory_ids; idempotent."""
        palace = _normalize_scope_value(_get_palace(request))
        if palace is None:
            return ManageMemoryResult(
                operation="archive_memories",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required",
            )

        clauses: list[str] = ["palace = $1", "lifecycle_state <> 'ARCHIVED'"]
        params: list[Any] = [palace]
        if request.item_id:
            item_check = await self._backend.query(
                "SELECT palace FROM knowledge_items WHERE id = $1::uuid",
                (request.item_id,),
            )
            if not item_check.rows or item_check.rows[0]["palace"] != palace:
                raise MemoryContractError(
                    code="MEM_PALACE_MISMATCH",
                    message=(
                        "MEM_PALACE_MISMATCH: archive_memories refuses to archive "
                        "for an item outside the requesting palace"
                    ),
                    retryable=False,
                )
            clauses.append(f"item_id = ${len(params) + 1}::uuid")
            params.append(request.item_id)
        if request.memory_ids:
            clauses.append(f"id = ANY(${len(params) + 1}::uuid[])")
            params.append(list(request.memory_ids))
        if not request.item_id and not request.memory_ids:
            return ManageMemoryResult(
                operation="archive_memories",
                success=False,
                error="MEM_FIELD_REQUIRED: provide item_id or memory_ids",
            )

        metadata_param = len(params) + 1
        result = await self._backend.query(
            f"""
            UPDATE knowledge_memories
               SET lifecycle_state = 'ARCHIVED',
                   metadata = metadata || ${metadata_param}::jsonb,
                   updated_at = NOW()
             WHERE {' AND '.join(clauses)}
            RETURNING id
            """,
            tuple(params + [json.dumps({"archived_reason": request.reason or "item_tombstoned"})]),
        )
        affected = [str(row["id"]) for row in result.rows]
        return ManageMemoryResult(
            operation="archive_memories",
            success=True,
            memory_ids=affected,
            stored_count=len(affected),
        )

    async def _manage_mark_item_dirty(
        self, request: ManageMemoryRequest
    ) -> ManageMemoryResult:
        """Set knowledge_items.lifecycle_state='DIRTY' and write error_metadata."""
        if not request.item_id:
            return ManageMemoryResult(
                operation="mark_item_dirty",
                success=False,
                error="MEM_FIELD_REQUIRED: 'item_id' is required",
            )
        palace = _normalize_scope_value(_get_palace(request))
        if palace is None:
            return ManageMemoryResult(
                operation="mark_item_dirty",
                success=False,
                error="MEM_PALACE_REQUIRED: 'palace' is required",
            )

        result = await self._backend.query(
            """
            UPDATE knowledge_items
               SET lifecycle_state = 'DIRTY',
                   error_metadata = $3::jsonb,
                   updated_at = NOW()
             WHERE id = $1::uuid AND palace = $2
            RETURNING id
            """,
            (request.item_id, palace, json.dumps(request.error_metadata or {})),
        )
        if not result.rows:
            return ManageMemoryResult(
                operation="mark_item_dirty",
                success=False,
                error="MEM_PALACE_MISMATCH: item not found in requesting palace",
            )
        return ManageMemoryResult(
            operation="mark_item_dirty",
            success=True,
            item_id=str(result.rows[0]["id"]),
        )

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
        palace = _get_palace(request)
        has_explicit_scope = _has_explicit_scope(request)

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

        # MEMORY-CONTRACT-v3.1: The global companion lane (s2) runs without scope
        # filters and therefore MUST NOT fire when an explicit scope is present —
        # doing so admits out-of-scope rows into the result set (cross-scope leakage).
        # s2_enabled is a retrieval-optimization hint, not a scope-override.
        # Scope isolation takes precedence: companion lane is only safe when no
        # explicit scope is provided (global search context enrichment only).
        include_companion = not has_explicit_scope
        rows = await room_scoped_search(
            embedding,
            request.query,
            self._backend,
            palace=palace,
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
            include_global_companion=include_companion,
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
            # Build lean dict — always include content; include optional fields only when present.
            # B-2: namespace is an internal DB column; it MUST NOT appear in external responses.
            cleaned: dict[str, Any] = {
                "id": str(row.get("id", "")),
                "content": row.get("content", ""),
                "confidence": row.get("confidence"),
                "authority": row.get("authority"),
                "rrf_score": row.get("rrf_score"),
                # provenance fields — consumers can locate the source file/project
                "path": row.get("item_path") or None,
                "source": row.get("source_name") or None,
            }
            # Lane contract:
            # USER_VALIDATED is promoted to facts; other authorities stay in memories.
            if row.get("authority") == Authority.USER_VALIDATED:
                facts.append(cleaned)
            else:
                memories.append(cleaned)

        scope_applied = has_explicit_scope

        return QueryMemoryResult(
            facts=facts,
            memories=memories,
            diagnostics={
                **_base_query_diagnostics(
                    # scope_mode reflects actual retrieval posture:
                    # companion lane is only active for unscoped (global) queries.
                    scope_mode="strict_scoped"
                    if has_explicit_scope
                    else "dual_lane_with_companion",
                    scope_applied=scope_applied,
                    has_results=bool(facts or memories),
                ),
                "authority_routing": _AUTHORITY_ROUTING_FACTS_USER_VALIDATED,
                "retrieval": _build_retrieval_diagnostics(
                    candidate_generation="deterministic",
                    algorithm="rrf",
                    s2_enabled=include_companion,
                    s2_requested=request.s2_enabled,
                    s2_strategy="companion_lane",
                ),
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
        palace = _get_palace(request)

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
            palace=palace,
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
            clauses.append(
                f"(km.valid_from IS NULL OR km.valid_from <= ${as_of_index}::timestamptz)"
            )
            clauses.append(f"(km.valid_to IS NULL OR km.valid_to >= ${as_of_index}::timestamptz)")
            params.append(as_of)
        else:
            if from_dt is not None:
                from_index = len(params) + 1
                clauses.append(
                    f"(km.valid_to IS NULL OR km.valid_to >= ${from_index}::timestamptz)"
                )
                params.append(from_dt)
            if to_dt is not None:
                to_index = len(params) + 1
                clauses.append(
                    f"(km.valid_from IS NULL OR km.valid_from <= ${to_index}::timestamptz)"
                )
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
                kc.palace,
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
            if row.get("palace") is not None:
                community["palace"] = row["palace"]
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
                palace=palace,
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
                # Lane contract:
                # USER_VALIDATED is promoted to facts; other authorities stay in memories.
                if row.get("authority") == Authority.USER_VALIDATED:
                    facts.append(cleaned)
                else:
                    memories.append(cleaned)

        scope_applied = _has_explicit_scope(request)

        return QueryMemoryResult(
            facts=facts,
            memories=memories,
            communities=communities,
            diagnostics={
                "strategy": "communities",
                **_base_query_diagnostics(
                    scope_mode="community_exists_filter",
                    scope_applied=scope_applied,
                    has_results=bool(facts or memories or communities),
                ),
                "authority_routing": _AUTHORITY_ROUTING_FACTS_USER_VALIDATED,
                "retrieval": _build_retrieval_diagnostics(
                    candidate_generation="deterministic",
                    algorithm="rrf",
                    s2_enabled=False,
                    s2_requested=request.s2_enabled,
                    s2_strategy="not_applicable",
                ),
            },
        )

    async def _query_palace(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Palace-scoped retrieval with explicit namespace/room/palace scoping.

        Palace retrieval requires explicit topology scoping (palace, namespace, and/or room).
        """
        palace = _get_palace(request)
        if not request.namespace and not request.room and not palace:
            return QueryMemoryResult(
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode="strict_scoped",
                        scope_applied=False,
                        has_results=False,
                        missing_scope=True,
                    ),
                    "error": (
                        "palace strategy requires palace, namespace, or room for scoped retrieval"
                    ),
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
            palace=palace,
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
                # B-2: namespace is an internal DB column; omitted from external results
            }
            # Lane contract:
            # USER_VALIDATED is promoted to facts; other authorities stay in memories.
            if row.get("authority") == Authority.USER_VALIDATED:
                facts.append(cleaned)
            else:
                memories.append(cleaned)

        return QueryMemoryResult(
            facts=facts,
            memories=memories,
            diagnostics={
                **_base_query_diagnostics(
                    scope_mode="strict_scoped",
                    scope_applied=True,
                    has_results=bool(facts or memories),
                ),
                "authority_routing": _AUTHORITY_ROUTING_FACTS_USER_VALIDATED,
                "retrieval": _build_retrieval_diagnostics(
                    candidate_generation="deterministic",
                    algorithm="rrf",
                    s2_enabled=False,
                    s2_requested=request.s2_enabled,
                    s2_strategy="not_applicable",
                ),
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
            diagnostics=_base_query_diagnostics(
                scope_mode="context_assembly",
                scope_applied=_has_explicit_scope(request),
                has_results=included_count > 0,
                strategy="context",
                retrieval=_build_retrieval_diagnostics(
                    candidate_generation="deterministic",
                    algorithm="context_assembly",
                    s2_enabled=False,
                    s2_requested=request.s2_enabled,
                    s2_strategy="not_applicable",
                ),
            ),
        )

    async def _query_graph(self, request: QueryMemoryRequest) -> QueryMemoryResult:
        """Graph retrieval dispatcher. Routes to sub-operation via request.graph_op."""
        op = request.graph_op
        start_entity = request.start_entity
        end_entity = request.end_entity
        namespace = request.namespace
        room = request.room
        corridor = _get_corridor(request)
        palace = _get_palace(request)
        if palace is None:
            return QueryMemoryResult(
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode="graph_rejected",
                        scope_applied=False,
                        has_results=False,
                        strategy="graph",
                    ),
                    "error_code": "MEM_PALACE_REQUIRED",
                    "error": "MEM_PALACE_REQUIRED: graph queries require palace scope",
                }
            )
        scope_applied = bool(namespace or room or corridor or palace)
        scope_mode = "graph_scoped" if scope_applied else "graph_global"

        async def _resolve_scoped_graph_entity(
            entity_ref: str,
        ) -> str | None:
            if not scope_applied:
                return entity_ref

            normalized_namespace = _normalize_scope_value(namespace)
            normalized_room = _normalize_scope_value(room)
            normalized_corridor = _normalize_scope_value(corridor)

            try:
                entity_uuid = str(uuid.UUID(entity_ref))
            except (ValueError, AttributeError):
                entity_uuid = None

            if entity_uuid is not None:
                scoped_uuid_result = await self._backend.query(
                    "SELECT id FROM knowledge_entities "
                    "WHERE id = $1::uuid AND palace = $2 "
                    "AND namespace = $3 AND room = $4 AND corridor = $5",
                    (
                        entity_uuid,
                        palace,
                        normalized_namespace,
                        normalized_room,
                        normalized_corridor,
                    ),
                )
                if not scoped_uuid_result.rows:
                    return None
                return str(scoped_uuid_result.rows[0]["id"])

            resolved = await _resolve_entity_id_manage(
                entity_ref,
                self._backend,
                namespace=namespace,
                room=room,
                corridor=corridor,
                palace=palace,
            )
            if resolved is None:
                return None
            return resolved

        async def _filter_graph_result_to_scope(
            raw_result: GraphResult,
        ) -> tuple[GraphResult, dict[str, int]]:
            if not scope_applied:
                return raw_result, {"dropped_nodes": 0, "dropped_edges": 0, "dropped_paths": 0}

            raw_nodes = [dict(node) for node in raw_result.get("nodes", [])]
            if not raw_nodes:
                return raw_result, {"dropped_nodes": 0, "dropped_edges": 0, "dropped_paths": 0}

            raw_node_ids = [str(node["id"]) for node in raw_nodes if node.get("id")]
            normalized_namespace = _normalize_scope_value(namespace)
            normalized_room = _normalize_scope_value(room)
            normalized_corridor = _normalize_scope_value(corridor)
            clauses = [
                "id = ANY($1::uuid[])",
                "namespace = $2",
                "room = $3",
                "corridor = $4",
            ]
            filter_params: list[Any] = [
                raw_node_ids,
                normalized_namespace,
                normalized_room,
                normalized_corridor,
            ]
            if palace is not None:
                clauses.append(f"palace = ${len(filter_params) + 1}")
                filter_params.append(palace)
            scoped_nodes_result = await self._backend.query(
                "SELECT id FROM knowledge_entities WHERE "
                + " AND ".join(clauses),
                tuple(filter_params),
            )
            scoped_node_ids = {str(row["id"]) for row in scoped_nodes_result.rows}

            filtered_nodes = [node for node in raw_nodes if str(node.get("id")) in scoped_node_ids]

            raw_edges = [dict(edge) for edge in raw_result.get("edges", [])]
            filtered_edges = [
                edge
                for edge in raw_edges
                if str(edge.get("source_entity_id")) in scoped_node_ids
                and str(edge.get("target_entity_id")) in scoped_node_ids
            ]

            raw_paths = [dict(path) for path in raw_result.get("paths", [])]
            filtered_paths: list[dict[str, Any]] = []
            for path in raw_paths:
                path_nodes_raw = path.get("nodes")
                path_edges_raw = path.get("edges")
                path_nodes: list[dict[str, Any]] = []
                if isinstance(path_nodes_raw, list):
                    path_nodes = [dict(node) for node in path_nodes_raw if isinstance(node, dict)]

                path_edges: list[dict[str, Any]] = []
                if isinstance(path_edges_raw, list):
                    path_edges = [dict(edge) for edge in path_edges_raw if isinstance(edge, dict)]
                node_ids = {str(node.get("id")) for node in path_nodes if node.get("id")}
                if node_ids and not node_ids.issubset(scoped_node_ids):
                    continue
                if any(
                    str(edge.get("source_entity_id")) not in scoped_node_ids
                    or str(edge.get("target_entity_id")) not in scoped_node_ids
                    for edge in path_edges
                ):
                    continue
                filtered_paths.append(path)

            filtered_result = dict(raw_result)
            filtered_result["nodes"] = filtered_nodes
            filtered_result["edges"] = filtered_edges
            filtered_result["paths"] = filtered_paths

            return cast(GraphResult, filtered_result), {
                "dropped_nodes": max(len(raw_nodes) - len(filtered_nodes), 0),
                "dropped_edges": max(len(raw_edges) - len(filtered_edges), 0),
                "dropped_paths": max(len(raw_paths) - len(filtered_paths), 0),
            }

        async def _scoped_graph_stats(as_of: datetime | None) -> dict[str, int]:
            normalized_namespace = _normalize_scope_value(namespace)
            normalized_room = _normalize_scope_value(room)
            normalized_corridor = _normalize_scope_value(corridor)

            scoped_clauses = ["palace = $1", "namespace = $2", "room = $3", "corridor = $4"]
            stats_params: list[Any] = [
                palace,
                normalized_namespace,
                normalized_room,
                normalized_corridor,
            ]

            temporal_clause = ""
            if as_of is not None:
                as_of_idx = len(stats_params) + 1
                temporal_clause = (
                    f"WHERE (kr.valid_from IS NULL OR kr.valid_from <= ${as_of_idx}::timestamptz)"
                    f" AND (kr.valid_to IS NULL OR kr.valid_to >= ${as_of_idx}::timestamptz)"
                )
                stats_params.append(as_of)

            result = await self._backend.query(
                f"""
                WITH scoped_entities AS (
                    SELECT id
                    FROM knowledge_entities
                    WHERE {' AND '.join(scoped_clauses)}
                )
                SELECT
                    (SELECT COUNT(*)::bigint FROM scoped_entities) AS entity_count,
                    COUNT(*)::bigint AS relation_count,
                    COUNT(DISTINCT kr.relation_type)::bigint AS distinct_relation_types
                FROM knowledge_relations kr
                JOIN scoped_entities src ON src.id = kr.source_entity_id
                JOIN scoped_entities dst ON dst.id = kr.target_entity_id
                {temporal_clause}
                """,
                tuple(stats_params),
            )

            row = result.rows[0] if result.rows else {}
            return {
                "entity_count": int(row.get("entity_count") or 0),
                "relation_count": int(row.get("relation_count") or 0),
                "distinct_relation_types": int(row.get("distinct_relation_types") or 0),
            }

        if op in ("traverse", "neighbors") and not start_entity:
            return QueryMemoryResult(
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode=scope_mode,
                        scope_applied=scope_applied,
                        has_results=False,
                        strategy="graph",
                    ),
                    "error": "start_entity required for graph strategy",
                },
            )
        if op == "path" and (not start_entity or not end_entity):
            return QueryMemoryResult(
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode=scope_mode,
                        scope_applied=scope_applied,
                        has_results=False,
                        strategy="graph",
                    ),
                    "error": ("start_entity and end_entity required for graph_op='path'"),
                },
            )

        as_of, from_dt, to_dt = _validate_query_temporal_inputs(
            request.as_of, request.from_, request.to
        )
        if from_dt is not None or to_dt is not None:
            return QueryMemoryResult(
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode=scope_mode,
                        scope_applied=scope_applied,
                        has_results=False,
                        strategy="graph",
                    ),
                    "error": "graph strategy supports 'as_of' only; 'from'/'to' is not supported",
                },
            )

        if start_entity:
            resolved_start = await _resolve_scoped_graph_entity(
                start_entity,
            )
            if resolved_start is None:
                return QueryMemoryResult(
                    diagnostics={
                        **_base_query_diagnostics(
                            scope_mode=scope_mode,
                            scope_applied=scope_applied,
                            has_results=False,
                            strategy="graph",
                        ),
                        "error": "start_entity not found in scoped graph",
                    }
                )
            start_entity = resolved_start

        if end_entity:
            resolved_end = await _resolve_scoped_graph_entity(
                end_entity,
            )
            if resolved_end is None:
                return QueryMemoryResult(
                    diagnostics={
                        **_base_query_diagnostics(
                            scope_mode=scope_mode,
                            scope_applied=scope_applied,
                            has_results=False,
                            strategy="graph",
                        ),
                        "error": "end_entity not found in scoped graph",
                    }
                )
            end_entity = resolved_end

        if op == "traverse":
            assert start_entity is not None
            result = await graph_traverse(
                start_entity,
                self._backend,
                relation_types=request.relation_types,
                max_hops=request.max_hops,
                max_nodes=request.max_nodes,
                as_of=as_of,
                palace=palace,
            )
            result, scope_filter_stats = await _filter_graph_result_to_scope(result)
            hydrated_edges, hydrated_memories = await self._hydrate_graph_supporting_memories(
                [dict(edge) for edge in result["edges"]]
            )
            paths: list[dict[str, Any]] = [dict(p) for p in result["paths"]]
            return QueryMemoryResult(
                paths=paths,
                memories=hydrated_memories,
                evidence=[{"nodes": result["nodes"], "edges": hydrated_edges}],
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode=scope_mode,
                        scope_applied=scope_applied,
                        has_results=bool(paths or hydrated_memories),
                        strategy="graph",
                    ),
                    **dict(result["diagnostics"]),
                    "scope_filter_stats": scope_filter_stats,
                    "fusion_version": "graph-only.v1",
                    "algorithm_versions": {
                        "graph": "bfs-traverse.v1",
                    },
                    "retrieval": _build_retrieval_diagnostics(
                        candidate_generation="not_applicable",
                        algorithm="graph_traverse",
                        s2_enabled=False,
                        s2_requested=request.s2_enabled,
                        s2_strategy="not_applicable",
                    ),
                },
            )

        if op == "neighbors":
            assert start_entity is not None
            result = await graph_neighbors(
                start_entity,
                self._backend,
                relation_types=request.relation_types,
                max_nodes=request.max_nodes,
                as_of=as_of,
                palace=palace,
            )
            result, scope_filter_stats = await _filter_graph_result_to_scope(result)
            hydrated_edges, hydrated_memories = await self._hydrate_graph_supporting_memories(
                [dict(edge) for edge in result["edges"]]
            )
            return QueryMemoryResult(
                paths=[],
                memories=hydrated_memories,
                evidence=[{"nodes": result["nodes"], "edges": hydrated_edges}],
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode=scope_mode,
                        scope_applied=scope_applied,
                        has_results=bool(hydrated_memories),
                        strategy="graph",
                    ),
                    **dict(result["diagnostics"]),
                    "scope_filter_stats": scope_filter_stats,
                    "fusion_version": "graph-only.v1",
                    "algorithm_versions": {
                        "graph": "neighbors-1hop.v1",
                    },
                    "retrieval": _build_retrieval_diagnostics(
                        candidate_generation="not_applicable",
                        algorithm="graph_neighbors",
                        s2_enabled=False,
                        s2_requested=request.s2_enabled,
                        s2_strategy="not_applicable",
                    ),
                },
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
                palace=palace,
            )
            result, scope_filter_stats = await _filter_graph_result_to_scope(result)
            hydrated_edges, hydrated_memories = await self._hydrate_graph_supporting_memories(
                [dict(edge) for edge in result["edges"]]
            )
            paths = [dict(p) for p in result["paths"]]
            return QueryMemoryResult(
                paths=paths,
                memories=hydrated_memories,
                evidence=[{"nodes": result["nodes"], "edges": hydrated_edges}],
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode=scope_mode,
                        scope_applied=scope_applied,
                        has_results=bool(paths or hydrated_memories),
                        strategy="graph",
                    ),
                    **dict(result["diagnostics"]),
                    "scope_filter_stats": scope_filter_stats,
                    "fusion_version": "graph-only.v1",
                    "algorithm_versions": {
                        "graph": "bfs-shortest-path.v1",
                    },
                    "retrieval": _build_retrieval_diagnostics(
                        candidate_generation="not_applicable",
                        algorithm="graph_path",
                        s2_enabled=False,
                        s2_requested=request.s2_enabled,
                        s2_strategy="not_applicable",
                    ),
                },
            )

        if op == "stats":
            if not start_entity:
                scoped_stats = await _scoped_graph_stats(as_of)
                has_scoped_results = bool(
                    scoped_stats["entity_count"] or scoped_stats["relation_count"]
                )
                return QueryMemoryResult(
                    paths=[],
                    evidence=[{"nodes": [], "edges": []}],
                    diagnostics={
                        **_base_query_diagnostics(
                            scope_mode=scope_mode,
                            scope_applied=scope_applied,
                            has_results=has_scoped_results,
                            strategy="graph",
                        ),
                        **scoped_stats,
                        "fusion_version": "graph-only.v1",
                        "algorithm_versions": {
                            "graph": "degree-stats.v1",
                        },
                        "retrieval": _build_retrieval_diagnostics(
                            candidate_generation="not_applicable",
                            algorithm="graph_stats",
                            s2_enabled=False,
                            s2_requested=request.s2_enabled,
                            s2_strategy="not_applicable",
                        ),
                    },
                )

            result = await graph_stats(
                start_entity,
                self._backend,
                as_of=as_of,
                palace=palace,
            )
            stats_has_results = bool(
                result["paths"]
                or result["diagnostics"].get("entity_count")
                or result["diagnostics"].get("relation_count")
            )
            return QueryMemoryResult(
                paths=[dict(p) for p in result["paths"]],
                evidence=[{"nodes": result["nodes"], "edges": result["edges"]}],
                diagnostics={
                    **_base_query_diagnostics(
                        scope_mode=scope_mode,
                        scope_applied=scope_applied,
                        has_results=stats_has_results,
                        strategy="graph",
                    ),
                    **dict(result["diagnostics"]),
                    "fusion_version": "graph-only.v1",
                    "algorithm_versions": {
                        "graph": "degree-stats.v1",
                    },
                    "retrieval": _build_retrieval_diagnostics(
                        candidate_generation="not_applicable",
                        algorithm="graph_stats",
                        s2_enabled=False,
                        s2_requested=request.s2_enabled,
                        s2_strategy="not_applicable",
                    ),
                },
            )

        return QueryMemoryResult(
            diagnostics={
                **_base_query_diagnostics(
                    scope_mode=scope_mode,
                    scope_applied=scope_applied,
                    has_results=False,
                    strategy="graph",
                ),
                "error": f"Unknown graph_op: {op!r}",
            }
        )

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
            "ensure_source": self._manage_ensure_source,
            "ensure_item": self._manage_ensure_item,
            "store_entities": self._manage_store_entities,
            "store_relations": self._manage_store_relations,
            "store_memories": self._manage_store_memories,
            "store_entity_embeddings": self._manage_store_entity_embeddings,
            "archive_memories": self._manage_archive_memories,
            "mark_item_dirty": self._manage_mark_item_dirty,
        }
        handler = handlers.get(op)
        if handler is None:
            return ManageMemoryResult(operation=op, success=False, error=f"Unknown operation: {op}")
        return await handler(request)

    async def _manage_ingest_structured(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Store structured memories, entities, relations, and links atomically."""
        if not request.memories:
            return ManageMemoryResult(
                operation="ingest_structured",
                success=False,
                error="'memories' is required for ingest_structured operation",
            )

        entities: list[StructuredEntityRecord] = cast(
            list[StructuredEntityRecord], request.entities or []
        )
        relations: list[StructuredRelationRecord] = cast(
            list[StructuredRelationRecord], request.relations or []
        )

        for relation in relations:
            if (
                _is_corridor_relation(relation.relation_type)
                and relation.evidence_memory_index is None
            ):
                return ManageMemoryResult(
                    operation="ingest_structured",
                    success=False,
                    error=(
                        "MEM_GRAPH_EVIDENCE_REQUIRED: ingest_structured CORRIDOR relations "
                        "require evidence_memory_index"
                    ),
                )

        transaction_started = False
        try:
            await self._backend.begin_transaction()
            transaction_started = True

            item_id: str | None = None
            source_name: str | None = None
            if request.source and request.path:
                ingest_palace = _normalize_scope_value(_get_palace(request)) or "default"
                source_result = await self._backend.query(
                    """
                    INSERT INTO knowledge_sources
                        (id, palace, name, source_type, category_ids)
                    VALUES ($1::uuid, $2, $3, $4, '{}'::uuid[])
                    ON CONFLICT (palace, name) DO UPDATE SET updated_at = NOW()
                    RETURNING id
                    """,
                    (str(uuid.uuid4()), ingest_palace, request.source, request.source_type),
                )
                if source_result.rows:
                    actual_source_id = str(source_result.rows[0]["id"])
                    item_title = os.path.basename(request.path) or request.path
                    item_result = await self._backend.query(
                        """
                        INSERT INTO knowledge_items
                            (id, palace, source_id, path, title,
                             content_hash, size_bytes, mtime_ns)
                        VALUES ($1::uuid, $2, $3::uuid, $4, $5, $6, $7, $8)
                        ON CONFLICT (palace, source_id, path) DO UPDATE SET
                            title = EXCLUDED.title, updated_at = NOW()
                        RETURNING id
                        """,
                        (
                            str(uuid.uuid4()),
                            ingest_palace,
                            actual_source_id,
                            request.path,
                            item_title,
                            "unknown",
                            0,
                            0,
                        ),
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
                try:
                    embedding, model_name, _, _ = await compute_embedding(
                        text=memory.content,
                        context=self._context,
                        profile=request.embedding_profile,
                    )
                except Exception as embed_exc:
                    raise MemoryContractError(
                        code="MEM_EMBEDDING_FAILED",
                        message=(
                            f"MEM_EMBEDDING_FAILED: Failed to compute embedding for memory: "
                            f"{embed_exc}"
                        ),
                        retryable=True,
                    ) from embed_exc
                await self._backend.execute(
                    """
                    INSERT INTO knowledge_memories
                        (id, item_id, content, embedding, search_vector,
                         authority, lifecycle_state, confidence, embedding_model,
                         metadata, created_by, auth_method,
                         source_name, source_type,
                         palace, namespace, room, corridor,
                         memory_tier, derived_kind, parent_memory_ids)
                    VALUES
                        ($1::uuid, $2::uuid, $3, $4::vector,
                         to_tsvector('english', $3),
                         $5, $6, $7, $8,
                         $9::jsonb, $10::uuid, $11,
                         $12, $13,
                         $14, $15, $16, $17,
                         $18, $19, $20::uuid[])
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
                        _get_palace(request),
                        request.namespace,
                        request.room,
                        request.corridor,
                        _MEMORY_TIER_DIRECT,
                        None,
                        [],
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
            entity_scope_palace = _normalize_scope_value(_get_palace(request))
            entity_scope_namespace = _normalize_scope_value(request.namespace)
            entity_scope_room = _normalize_scope_value(request.room)
            entity_scope_corridor = _normalize_scope_value(_get_corridor(request))
            for entity in entities:
                entity_id = await self._upsert_structured_entity(
                    entity_name=entity.name,
                    entity_type=entity.entity_type,
                    confidence=entity.confidence
                    if entity.confidence is not None
                    else request.confidence,
                    palace=entity_scope_palace,
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
                        confidence=entity.confidence
                        if entity.confidence is not None
                        else request.confidence,
                    )

            for relation in relations:
                source_entity_id = await self._resolve_or_upsert_structured_entity(
                    entity_ids_by_key,
                    relation.source_type,
                    relation.source_name,
                    request.confidence,
                    palace=entity_scope_palace,
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
                    palace=entity_scope_palace,
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
                        confidence=relation.confidence
                        if relation.confidence is not None
                        else request.confidence,
                    )
                    await self._link_memory_entity(
                        memory_id=evidence_memory_id,
                        entity_id=target_entity_id,
                        confidence=relation.confidence
                        if relation.confidence is not None
                        else request.confidence,
                    )
                relation_result = await self._backend.query(
                    """
                    INSERT INTO knowledge_relations
                        (id, source_entity_id, target_entity_id, relation_type,
                         confidence, evidence_memory_id, evidence_memory_ids)
                    VALUES
                        ($1::uuid, $2::uuid, $3::uuid, $4,
                         $5, $6::uuid, $7::uuid[])
                    RETURNING id
                    """,
                    (
                        str(uuid.uuid4()),
                        source_entity_id,
                        target_entity_id,
                        relation.relation_type,
                        relation.confidence
                        if relation.confidence is not None
                        else request.confidence,
                        evidence_memory_id,
                        [evidence_memory_id] if evidence_memory_id else [],
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
                 palace, namespace, room, corridor,
                 memory_tier, derived_kind, parent_memory_ids)
            VALUES
                ($1::uuid, $2::uuid, $3, $4::vector,
                 to_tsvector('english', $3),
                 $5, $6, $7,
                 $8, $9::jsonb,
                 $10::timestamptz, $11::timestamptz,
                 $12::uuid, $13, $14, $15,
                 $16, $17, $18, $19,
                 $20, $21, $22::uuid[])
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
                _get_palace(request),
                request.namespace,
                request.room,
                request.corridor,
                _MEMORY_TIER_DIRECT,
                None,
                [],
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

        replacement_result = await self._backend.query(
            "SELECT id FROM knowledge_memories WHERE id = $1::uuid",
            (request.superseded_by,),
        )
        if not replacement_result.rows:
            return ManageMemoryResult(
                operation="supersede",
                success=False,
                error=(
                    "MEM_SUPERSEDE_REFERENCE_NOT_FOUND: "
                    "'superseded_by' must reference an existing memory"
                ),
            )

        ids = [str(i) for i in request.memory_ids]
        if request.superseded_by in ids:
            return ManageMemoryResult(
                operation="supersede",
                success=False,
                error="MEM_SUPERSEDE_SELF_REFERENCE: replacement memory cannot supersede itself",
            )

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
                superseded_by_memory_id = ${len(ids) + 2}::uuid,
                valid_to = COALESCE(${len(ids) + 3}::timestamptz, NOW()),
                updated_at = NOW()
            WHERE id IN ({placeholders})
              AND id != ${len(ids) + 2}::uuid
              AND superseded_by_memory_id IS NULL
              AND lifecycle_state NOT IN ('SUPERSEDED', 'ARCHIVED')
            RETURNING id
            """,
            tuple(ids) + (metadata, request.superseded_by, explicit_valid_to),
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

        # Pre-flight: detect SUPERSEDED records before issuing any UPDATE.
        # The DB trigger trg_km_supersede_append_only forbids changing lifecycle_state
        # away from SUPERSEDED. Raise a deterministic contract error here so callers
        # receive MEM_SUPERSEDE_APPEND_ONLY rather than a generic MEM_INTERNAL_ERROR.
        superseded_check = await self._backend.query(
            f"""
            SELECT id FROM knowledge_memories
            WHERE id IN ({placeholders})
              AND lifecycle_state = 'SUPERSEDED'
            """,
            tuple(ids),
        )
        if superseded_check.rows:
            superseded_ids = [str(r["id"]) for r in superseded_check.rows]
            _raise_contract_error(
                code="MEM_SUPERSEDE_APPEND_ONLY",
                message=(
                    "superseded memories cannot be archived — "
                    "supersede is an append-only lifecycle state. "
                    f"Affected id(s): {', '.join(superseded_ids)}"
                ),
            )

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
                community_id = await self._insert_community(
                    component, entity_names, memory_rows, request
                )
                await self._backend.execute(
                    "UPDATE knowledge_entities SET community_id = $1::uuid "
                    "WHERE id = ANY($2::uuid[])",
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
        except MemoryContractError as exc:
            await self._backend.rollback()
            return ManageMemoryResult(
                operation="consolidate",
                success=False,
                error=exc.message,
                diagnostics={
                    "mode": request.mode,
                    "status": "failed",
                    "error_code": exc.code,
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
            if not request.palace:
                return ManageMemoryResult(
                    operation="maintain",
                    success=False,
                    error="MEM_PALACE_REQUIRED: community_refresh requires palace scope",
                    diagnostics={
                        "mode": mode,
                        "status": "rejected",
                        "error_code": "MEM_PALACE_REQUIRED",
                    },
                )
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

    async def _maintain_decay_scan(self, request: ManageMemoryRequest) -> ManageMemoryResult:
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
                    days_since_retrieval = max((now - last_retrieved).total_seconds() / 86400, 0)
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

    async def _maintain_prune_candidates(self, request: ManageMemoryRequest) -> ManageMemoryResult:
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

    async def _maintain_expire_quarantine(self, request: ManageMemoryRequest) -> ManageMemoryResult:
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

    async def _maintain_expire_flags(self, request: ManageMemoryRequest) -> ManageMemoryResult:
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
            id_placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(expired_ids)))
            conflict_rows = await self._backend.query(
                "SELECT id FROM knowledge_conflicts "
                f"WHERE new_memory_id IN ({id_placeholders}) "
                "AND resolved_at IS NULL",
                (*expired_ids,),
            )
            if conflict_rows.rows:
                conflict_ids = [str(row["id"]) for row in conflict_rows.rows]
                c_placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(conflict_ids)))
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
            QueryMemoryRequest.model_validate(
                {
                    "query": request.query,
                    "strategy": "context",
                    "scope": {"corridor": request.corridor} if request.corridor else None,
                    "as_of": request.as_of,
                    "from": request.from_,
                    "to": request.to,
                    "max_items": request.max_items,
                    "max_tokens": request.max_tokens,
                    "namespace": request.namespace,
                    "room": request.room,
                    "source": request.source,
                    "categories": request.categories,
                    "min_confidence": request.min_confidence,
                    "lifecycle_state": request.lifecycle_state,
                    "embedding_profile": request.embedding_profile,
                }
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

    async def _manage_graph_store_entity(self, request: ManageMemoryRequest) -> ManageMemoryResult:
        """Upsert a named entity into knowledge_entities."""
        if not request.entity_name:
            return ManageMemoryResult(
                operation="graph_store_entity", success=False, error="'entity_name' is required"
            )
        if not request.entity_type:
            return ManageMemoryResult(
                operation="graph_store_entity", success=False, error="'entity_type' is required"
            )

        entity_scope_palace = _normalize_scope_value(_get_palace(request))
        entity_scope_namespace = _normalize_scope_value(request.namespace)
        entity_scope_room = _normalize_scope_value(request.room)
        entity_scope_corridor = _normalize_scope_value(_get_corridor(request))

        result = await self._backend.query(
            """
            INSERT INTO knowledge_entities (
                id, entity_type, name, palace, namespace, room, corridor, confidence
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (palace, namespace, room, corridor, entity_type, name) DO UPDATE
                SET confidence = GREATEST(knowledge_entities.confidence, EXCLUDED.confidence)
            RETURNING id
            """,
            (
                str(uuid.uuid4()),
                request.entity_type,
                request.entity_name,
                entity_scope_palace,
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

        evidence_ids = list(request.evidence_memory_ids or [])
        if request.evidence_memory_id:
            evidence_ids.append(request.evidence_memory_id)
        evidence_ids = list(dict.fromkeys(evidence_ids))

        if (
            _is_corridor_relation(request.relation_type)
            and not request.curated
            and not evidence_ids
        ):
            return ManageMemoryResult(
                operation="graph_store_relation",
                success=False,
                error=(
                    "MEM_GRAPH_EVIDENCE_REQUIRED: "
                    "non-curated CORRIDOR graph relations require evidence_memory_ids"
                ),
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
        evidence_id = evidence_ids[0] if evidence_ids else None

        if evidence_ids:
            evidence_count_result = await self._backend.query(
                "SELECT COUNT(*) AS cnt FROM knowledge_memories WHERE id = ANY($1::uuid[])",
                (evidence_ids,),
            )
            evidence_count = (
                int(evidence_count_result.rows[0]["cnt"]) if evidence_count_result.rows else 0
            )
            if evidence_count != len(evidence_ids):
                return ManageMemoryResult(
                    operation="graph_store_relation",
                    success=False,
                    error="MEM_NOT_FOUND: one or more evidence_memory_ids "
                    "do not reference existing memories",
                )

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
                    evidence_memory_ids = CASE
                        WHEN cardinality($4::uuid[]) > 0 THEN $4::uuid[]
                        ELSE evidence_memory_ids
                    END,
                    curated = CASE
                        WHEN curated AND NOT $5 THEN TRUE
                        ELSE $5
                    END,
                    valid_from = COALESCE($6::timestamptz, valid_from),
                    valid_to = COALESCE($7::timestamptz, valid_to)
                WHERE id = $1::uuid
                RETURNING id
                """,
                (
                    str(existing_result.rows[0]["id"]),
                    request.confidence,
                    evidence_id,
                    evidence_ids,
                    request.curated,
                    valid_from,
                    valid_to,
                ),
            )
        else:
            result = await self._backend.query(
                """
                INSERT INTO knowledge_relations
                    (id, source_entity_id, target_entity_id, relation_type,
                     confidence, evidence_memory_id, evidence_memory_ids, curated,
                     valid_from, valid_to)
                VALUES
                    ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6::uuid,
                     $7::uuid[], $8, $9::timestamptz, $10::timestamptz)
                RETURNING id
                """,
                (
                    str(uuid.uuid4()),
                    source_id,
                    target_id,
                    request.relation_type,
                    request.confidence,
                    evidence_id,
                    evidence_ids,
                    request.curated,
                    valid_from,
                    valid_to,
                ),
            )
        rid = str(result.rows[0]["id"]) if result.rows else None
        return ManageMemoryResult(operation="graph_store_relation", success=True, relation_id=rid)

    async def _hydrate_graph_supporting_memories(
        self,
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Hydrate graph edges with supporting memory rows."""
        memory_ids: list[str] = []
        for edge in edges:
            edge_ids: list[str] = []
            single_id = edge.get("evidence_memory_id")
            if isinstance(single_id, str) and single_id:
                edge_ids.append(single_id)
            raw_ids = edge.get("evidence_memory_ids")
            if isinstance(raw_ids, list):
                edge_ids.extend(str(item) for item in raw_ids if item)
            deduped = list(dict.fromkeys(edge_ids))
            edge["evidence_memory_ids"] = deduped
            memory_ids.extend(deduped)

        unique_ids = list(dict.fromkeys(memory_ids))
        if not unique_ids:
            for edge in edges:
                edge["supporting_memories"] = []
            return edges, []

        result = await self._backend.query(
            """
            SELECT
                km.id,
                km.content,
                km.confidence,
                km.authority,
                km.source_name,
                ki.path AS item_path,
                km.namespace,
                km.room,
                km.corridor
            FROM knowledge_memories km
            LEFT JOIN knowledge_items ki ON ki.id = km.item_id
            WHERE km.id = ANY($1::uuid[])
            """,
            (unique_ids,),
        )

        memories_by_id: dict[str, dict[str, Any]] = {}
        for row in result.rows:
            memory_id = str(row["id"])
            # B-2: namespace, room, corridor are internal DB columns; omitted from external results.
            # Scope context is conveyed via resolved_scope in the outer MemoryResult envelope.
            memories_by_id[memory_id] = {
                "id": memory_id,
                "content": row.get("content", ""),
                "confidence": row.get("confidence"),
                "authority": row.get("authority"),
                "path": row.get("item_path") or None,
                "source": row.get("source_name") or None,
            }

        for edge in edges:
            edge["supporting_memories"] = [
                memories_by_id[mid]
                for mid in edge.get("evidence_memory_ids", [])
                if mid in memories_by_id
            ]

        hydrated_memories = [memories_by_id[mid] for mid in unique_ids if mid in memories_by_id]
        return edges, hydrated_memories

    async def _manage_graph_forget_entity(self, request: ManageMemoryRequest) -> ManageMemoryResult:
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
        deleted_relations = (
            int(relation_count_result.rows[0]["cnt"]) if relation_count_result.rows else 0
        )
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
                    logger.warning(
                        "Creating new knowledge category via explicit opt-in: %r", normalized_name
                    )
                resolved.append(str(result.rows[0]["id"]))
            else:
                logger.warning("Category resolution returned no rows for %r", normalized_name)

        if missing_names:
            missing = ", ".join(repr(name) for name in missing_names)
            _raise_contract_error(
                code="MEM_UNKNOWN_CATEGORY",
                message=(
                    "Unknown categories: "
                    f"{missing}. "
                    "Set allow_create_categories=true to explicitly create missing categories."
                ),
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
        palace = request.palace or None
        clauses, params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="km",
            palace=palace,
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
            palace=request.palace or None,
        )
        if scope_clauses:
            clauses.append("kr.evidence_memory_id IS NOT NULL")
            clauses.append(
                "EXISTS ("
                "SELECT 1 FROM knowledge_memories km "
                "WHERE km.id = kr.evidence_memory_id AND " + " AND ".join(scope_clauses) + ")"
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
        palace = request.palace or None
        scope_clauses, scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="knowledge_memories",
            palace=palace,
        )
        if scope_clauses:
            await self._backend.execute(
                "UPDATE knowledge_memories SET community_id = NULL WHERE "
                + " AND ".join(scope_clauses),
                tuple(scope_params),
            )
        else:
            await self._backend.execute("UPDATE knowledge_memories SET community_id = NULL", ())

        community_scope_clauses, community_scope_params, _ = _build_memory_scope_filters(
            request.namespace,
            request.room,
            corridor,
            alias="knowledge_communities",
            palace=palace,
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

        ordered_names = sorted(
            entity_names[entity_id] for entity_id in entity_ids if entity_names.get(entity_id)
        )
        summary_names = ordered_names[:5]
        content = "Community: " + ", ".join(summary_names) if summary_names else "Community"

        result = await self._backend.query(
            """
            INSERT INTO knowledge_communities
                (id, content, embedding, member_count, memory_count,
                 palace, namespace, room, corridor)
            VALUES
                ($1::uuid, $2, $3::vector, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            (
                str(uuid.uuid4()),
                content,
                str(centroid) if centroid is not None else None,
                len(entity_ids),
                len({str(row["id"]) for row in memory_rows}),
                _get_palace(request),
                request.namespace,
                request.room,
                _get_corridor(request),
            ),
        )
        if not result.rows:
            raise RuntimeError("community insert returned no id")
        community_id = str(result.rows[0]["id"])

        parent_memory_ids = sorted(
            {str(row["id"]) for row in memory_rows if row.get("id") is not None}
        )
        await self._insert_derived_memory(
            content=content,
            community_id=community_id,
            parent_memory_ids=parent_memory_ids,
            request=request,
            derived_kind=_DERIVED_KIND_COMMUNITY,
        )

        return community_id

    async def _insert_derived_memory(
        self,
        *,
        content: str,
        community_id: str,
        parent_memory_ids: list[str],
        request: ManageMemoryRequest,
        derived_kind: str,
    ) -> str:
        """Persist a derived memory with explicit lineage constraints."""
        await self._assert_valid_parent_lineage(parent_memory_ids)

        derived_memory_id = str(uuid.uuid4())
        created_by = _get_audit_user_id(self._context)
        auth_method = _get_auth_method(self._context)
        embedding, model_name, _, _ = await compute_embedding(
            text=content,
            context=self._context,
            profile=request.embedding_profile,
        )

        await self._backend.execute(
            """
            INSERT INTO knowledge_memories
                (id, community_id, content, embedding, search_vector,
                 authority, lifecycle_state, confidence, embedding_model,
                 metadata, created_by, auth_method,
                 palace, namespace, room, corridor,
                 memory_tier, derived_kind, parent_memory_ids)
            VALUES
                ($1::uuid, $2::uuid, $3, $4::vector,
                 to_tsvector('english', $3),
                 $5, $6, $7, $8,
                 $9::jsonb, $10::uuid, $11,
                 $12, $13, $14, $15,
                 $16, $17, $18::uuid[])
            """,
            (
                derived_memory_id,
                community_id,
                content,
                str(embedding),
                Authority.COMMUNITY_SUMMARY,
                LifecycleState.ACTIVE,
                request.confidence,
                model_name,
                json.dumps(
                    {
                        "derived_from": "community_refresh",
                        "lineage_parent_count": len(parent_memory_ids),
                    }
                ),
                str(created_by),
                auth_method,
                _get_palace(request),
                request.namespace,
                request.room,
                _get_corridor(request),
                _MEMORY_TIER_DERIVED,
                derived_kind,
                parent_memory_ids,
            ),
        )
        return derived_memory_id

    async def _assert_valid_parent_lineage(self, parent_memory_ids: list[str]) -> None:
        """Validate derived memory lineage points to existing direct memories."""
        normalized = sorted({memory_id for memory_id in parent_memory_ids if memory_id})
        if not normalized:
            _raise_contract_error(
                code="MEM_LINEAGE_PARENT_REQUIRED",
                message="derived memories require at least one parent memory reference",
            )

        result = await self._backend.query(
            "SELECT id, memory_tier FROM knowledge_memories WHERE id = ANY($1::uuid[])",
            (normalized,),
        )
        found_ids = {str(row["id"]) for row in result.rows}
        missing_ids = sorted(set(normalized) - found_ids)
        if missing_ids:
            _raise_contract_error(
                code="MEM_LINEAGE_PARENT_NOT_FOUND",
                message=f"unknown parent memories: {', '.join(missing_ids)}",
            )

        non_direct_ids = sorted(
            str(row["id"])
            for row in result.rows
            if str(row.get("memory_tier") or _MEMORY_TIER_DIRECT) != _MEMORY_TIER_DIRECT
        )
        if non_direct_ids:
            _raise_contract_error(
                code="MEM_LINEAGE_PARENT_NOT_DIRECT",
                message=f"derived parents must be direct memories: {', '.join(non_direct_ids)}",
            )

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
        palace: str,
        namespace: str,
        room: str,
        corridor: str,
    ) -> str:
        """Upsert a structured entity and return its UUID string."""
        result = await self._backend.query(
            """
            INSERT INTO knowledge_entities (
                id, entity_type, name, palace, namespace, room, corridor, confidence
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (palace, namespace, room, corridor, entity_type, name) DO UPDATE
                SET confidence = GREATEST(knowledge_entities.confidence, EXCLUDED.confidence)
            RETURNING id
            """,
            (
                str(uuid.uuid4()),
                entity_type,
                entity_name,
                palace,
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
        palace: str,
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
            palace=palace,
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
