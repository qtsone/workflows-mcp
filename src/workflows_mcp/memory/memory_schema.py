"""Memory request/response schema — the unified contract models.

The 29 Pydantic models here define the public memory contract shared by the
memory tool (query_memory, manage_memory), the Memory workflow block, and the
onboard/sync orchestrator. They are deliberately separated from the
orchestration logic in memory_service so the contract surface stays navigable
and importers depend only on the schema, not the engine.

The temporal-window validators live here too: they enforce field-level
invariants on the request models and are reused by the service body.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .knowledge.constants import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_CONFIDENCE,
    Authority,
    LifecycleState,
)
from .memory_errors import _raise_contract_error
from .memory_locality import CONTRACT_SCOPE_FIELDS

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
    "store_relations_by_qname",
    "store_memories",
    "store_entity_embeddings",
    "archive_memories",
    "mark_item_dirty",
    # ADR-013: System 1 / System 2 operations
    "store_system1_structural_evidence",
    "store_system1_structural_graph",
    "record_system1_verification_cycle",
    "derive_system1_project_topology",
    "derive_system1_topology",
    "derive_system2_semantic_claims",
    "apply_semantic_override",
    "reconcile_semantic_lifecycle",
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
    "store_relations_by_qname",
    "store_memories",
    "store_entity_embeddings",
    "archive_memories",
    "mark_item_dirty",
    # ADR-013: System 1 / System 2 operations
    "store_system1_structural_evidence",
    "store_system1_structural_graph",
    "record_system1_verification_cycle",
    "derive_system1_project_topology",
    "derive_system1_topology",
    "derive_system2_semantic_claims",
    "apply_semantic_override",
    "reconcile_semantic_lifecycle",
)

MEMORY_SECTION_REQUIRED_BY_OPERATION: dict[MemoryOperation, str] = {
    "query": "query",
    "ingest": "record",
    "validate": "record",
    "supersede": "record",
    "archive": "record",
    "graph_upsert": "graph",
    "graph_delete": "graph",
    "derive_system1_topology": "derivation",
}


# ---------------------------------------------------------------------------
# Temporal validation helpers
# ---------------------------------------------------------------------------


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
        "store_relations_by_qname",
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
    raw_qname_relations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Qname-keyed relation dicts for store_relations_by_qname (ADR-012)",
    )
    external_fallback: str = Field(
        default="module",
        description=(
            "Fallback for unresolved targets in store_relations_by_qname: "
            "'module' creates an EXTERNAL Module entity; 'reject' marks as unresolved."
        ),
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
    relation_ids: list[str | None] = Field(
        default_factory=list,
        description=(
            "IDs of stored or affected relations. Entries are null for "
            "unresolved/rejected items (store_relations_by_qname); non-null "
            "for all entries from store_relations."
        ),
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

    # store_relations_by_qname outputs (ADR-012)
    created_count: int = Field(
        default=0,
        description="New relation rows inserted by store_relations_by_qname.",
    )
    existing_count: int = Field(
        default=0,
        description="Idempotent matches (relation already existed) for store_relations_by_qname.",
    )
    external_entities_created: list[dict[str, Any]] = Field(
        default_factory=list,
        description="EXTERNAL Module entities created during this call (ADR-012).",
    )
    unresolved_or_ambiguous: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-relation unresolved/ambiguous entries (ADR-012).",
    )

    # ADR-013: System 1 / System 2 result envelope fields
    stored_evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Stable IDs for each persisted structural evidence row"
            " (store_system1_structural_evidence)."
        ),
    )
    cycle_id: str | None = Field(
        default=None,
        description="ID of the persisted verification cycle (record_system1_verification_cycle).",
    )
    claim_ids: list[str] = Field(
        default_factory=list,
        description="IDs of derived semantic claims (derive_system2_semantic_claims).",
    )
    override_id: str | None = Field(
        default=None,
        description="ID of the applied override record (apply_semantic_override).",
    )
    reconciled_count: int = Field(
        default=0,
        description="Number of lifecycle state transitions applied (reconcile_semantic_lifecycle).",
    )

    # ADR-013 Task 2a: topology derivation typed result fields
    derived_wing: str | None = Field(
        default=None,
        description="Wing level resolved by derive_system1_topology.",
    )
    derived_room: str | None = Field(
        default=None,
        description="Room level resolved by derive_system1_topology.",
    )
    derived_compartment: str | None = Field(
        default=None,
        description="Compartment level resolved by derive_system1_topology.",
    )
    derivation_source: str | None = Field(
        default=None,
        description=("How topology was determined: 'explicit_override' or 'system1_derived'."),
    )
    provenance_id: str | None = Field(
        default=None,
        description="ID of the persisted knowledge_topology_provenance row.",
    )
    claim_id: str | None = Field(
        default=None,
        description=(
            "ID of the semantic claim this topology derivation produced or updated "
            "(singular; distinct from claim_ids list used by derive_system2_semantic_claims)."
        ),
    )


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


# ---------------------------------------------------------------------------
# ADR-013 payload models (Task 3 interface seams — no full persistence yet)
# ---------------------------------------------------------------------------


class StructuralEvidenceItem(BaseModel):
    """One structural evidence record for store_system1_structural_evidence."""

    model_config = ConfigDict(extra="forbid")

    entity_stable_id: str = Field(
        description="Stable source anchor ID (e.g. 'src/mod.py::MyClass')"
    )
    entity_type: str = Field(
        default="unknown",
        description="Entity type (e.g. 'class', 'function', 'module')",
    )
    evidence_category: Literal[
        "structural_class",
        "structural_module",
        "structural_function",
        "structural_call_graph",
        "structural_import",
        "structural_test",
        "structural_config",
        "structural_doc",
    ] = Field(description="Evidence category matching schema CHECK constraint")
    evidence_data: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary structural metadata"
    )


class VerificationCycleInput(BaseModel):
    """Verification-cycle metadata for record_system1_verification_cycle."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        description="True when the cycle confirms evidence absence; False when evidence was found"
    )
    covered_scope: dict[str, str] = Field(
        default_factory=dict, description="Scope identity covered by this cycle"
    )


class System1StructuralGraphInput(BaseModel):
    """Raw System1 structural graph contract payload (validation-only in Task 3)."""

    model_config = ConfigDict(extra="forbid")

    palace: str | None = Field(default=None)
    source_name: str | None = Field(default=None)
    repo_relative_path: str | None = Field(default=None)
    parser_metadata: dict[str, Any]
    entities: list[Any]
    relations: list[Any]
    unresolved_imports: list[str] = Field(default_factory=list)
    structural_evidence: list[StructuralEvidenceItem] = Field(default_factory=list)


class ProofBundleInput(BaseModel):
    """Proof bundle for new-wing creation gate."""

    model_config = ConfigDict(extra="forbid")

    evidence_categories: list[str] = Field(default_factory=list)


class TopologyOverrideInput(BaseModel):
    """Explicit caller override for topology derivation (ADR-013 Task 2a).

    All five fields are required when an override is supplied. Whitespace-only
    override_reason or applied_by is rejected to prevent accountability gaps.
    """

    model_config = ConfigDict(extra="forbid")

    wing: str = Field(description="Explicit wing placement for this override.")
    room: str = Field(description="Explicit room placement for this override.")
    compartment: str = Field(description="Explicit compartment placement for this override.")
    override_reason: str = Field(
        description="Non-empty human-readable reason for the explicit override."
    )
    applied_by: str = Field(
        description="Non-empty identifier of the person or system applying this override."
    )

    @model_validator(mode="after")
    def _reject_whitespace_fields(self) -> TopologyOverrideInput:
        if not self.wing.strip():
            raise ValueError("MEM_WHITESPACE_WING: wing must not be empty or whitespace-only")
        if not self.room.strip():
            raise ValueError("MEM_WHITESPACE_ROOM: room must not be empty or whitespace-only")
        if not self.compartment.strip():
            raise ValueError(
                "MEM_WHITESPACE_COMPARTMENT: compartment must not be empty or whitespace-only"
            )
        if not self.override_reason.strip():
            raise ValueError(
                "MEM_WHITESPACE_OVERRIDE_REASON: override_reason must not be whitespace-only"
            )
        if not self.applied_by.strip():
            raise ValueError("MEM_WHITESPACE_APPLIED_BY: applied_by must not be whitespace-only")
        return self


class DeriveSystem1TopologyInput(BaseModel):
    """Derivation payload for derive_system1_topology (ADR-013 Task 2a, Task 5b).

    Caller supplies the palace and either a non-empty list of evidence IDs (previously
    persisted rows) or a non-empty list of inline_candidates (parsed structural evidence
    items to be stored atomically during derivation). When inline_candidates are provided,
    evidence_ids becomes optional.

    An optional topology_override provides a complete explicit placement when the caller
    has authoritative knowledge that supersedes structural derivation.

    parser_metadata carries parser diagnostic context (e.g. language, file path) as
    evidence metadata only. It must never be used as direct topology truth: wing must not
    default to language and room must not default to folder/package path.
    """

    model_config = ConfigDict(extra="forbid")

    palace: str = Field(description="Palace identifier for which topology is being derived.")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "List of structural evidence row IDs to use for derivation. "
            "Optional when inline_candidates are supplied; required otherwise."
        ),
    )
    inline_candidates: list[StructuralEvidenceItem] = Field(
        default_factory=list,
        description=(
            "Inline structural evidence candidates emitted by the parser (e.g. TreeSitter). "
            "When non-empty, candidates are stored as knowledge_structural_evidence rows "
            "atomically within the same derive transaction. Optional when evidence_ids are given."
        ),
    )
    parser_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parser diagnostic context (e.g. language, file_path, package). "
            "Stored as evidence metadata only. Must never be used as topology truth: "
            "wing must not default to language and room must not default to path."
        ),
    )
    topology_override: TopologyOverrideInput | None = Field(
        default=None,
        description=(
            "Optional complete explicit topology override. When supplied, all five fields "
            "(wing, room, compartment, override_reason, applied_by) are required. "
            "Absent means pure structural derivation from evidence."
        ),
    )
    is_new_wing: bool = Field(
        default=False,
        description=(
            "When True, the proof-bundle gate is enforced: evidence_ids must reference "
            ">=2 distinct persisted rows across >=2 distinct evidence categories."
        ),
    )
    proof_bundle: ProofBundleInput | None = Field(
        default=None,
        description=(
            "Required when is_new_wing=True. Declares the expected evidence categories; "
            "each must be backed by at least one distinct persisted evidence row."
        ),
    )

    @model_validator(mode="after")
    def _require_at_least_one_evidence_source(self) -> DeriveSystem1TopologyInput:
        if not self.evidence_ids and not self.inline_candidates:
            raise ValueError(
                "at least one evidence source is required: "
                "supply non-empty 'evidence_ids' or 'inline_candidates'"
            )
        return self

    @model_validator(mode="after")
    def _require_proof_bundle_for_new_wing(self) -> DeriveSystem1TopologyInput:
        if self.is_new_wing and self.topology_override is None and self.proof_bundle is None:
            raise ValueError("proof_bundle is required when is_new_wing=True")
        return self


class DerivationInput(BaseModel):
    """Derivation payload for derive_system2_semantic_claims."""

    model_config = ConfigDict(extra="forbid")

    wing_intent_label: str | None = Field(default=None)
    room_intent_label: str | None = Field(default=None)
    compartment_reasoning_unit: str | None = Field(default=None)
    memory_claim_text: str | None = Field(default=None)
    evidence_entity_stable_ids: list[str] = Field(default_factory=list)
    is_new_wing: bool = Field(default=False)
    proof_bundle: ProofBundleInput | None = Field(default=None)

    # Corridor fields — all three must be supplied together or not at all.
    corridor_from_claim_id: str | None = Field(default=None)
    corridor_to_claim_id: str | None = Field(default=None)
    corridor_type: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_blank_optional_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for key in (
            "wing_intent_label",
            "room_intent_label",
            "compartment_reasoning_unit",
            "memory_claim_text",
            "corridor_from_claim_id",
            "corridor_to_claim_id",
            "corridor_type",
        ):
            value = normalized.get(key)
            if isinstance(value, str) and not value.strip():
                normalized[key] = None
        return normalized

    @model_validator(mode="after")
    def _validate_corridor_fields(self) -> DerivationInput:
        corridor_fields = (
            self.corridor_from_claim_id,
            self.corridor_to_claim_id,
            self.corridor_type,
        )
        present = sum(1 for f in corridor_fields if f is not None)
        if 0 < present < 3:
            raise ValueError(
                "corridor_from_claim_id, corridor_to_claim_id, and corridor_type"
                " must all be supplied together or not at all"
            )
        if present == 3:
            if self.corridor_from_claim_id == self.corridor_to_claim_id:
                raise ValueError(
                    "corridor_from_claim_id and corridor_to_claim_id must differ"
                    " (self-loop corridors are not permitted)"
                )
        return self


class OverrideInput(BaseModel):
    """Override payload for apply_semantic_override."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(description="ID of the claim being overridden")
    override_reason: str = Field(
        min_length=1,
        description="Mandatory provenance reason for the override; must be non-empty.",
    )
    overridden_by: str = Field(
        min_length=1,
        description="Identity of the actor applying the override; must be non-empty.",
    )
    new_lifecycle_state: str = Field(description="Target lifecycle state to force")


class LifecycleReconciliationInput(BaseModel):
    """Lifecycle reconciliation payload for reconcile_semantic_lifecycle."""

    model_config = ConfigDict(extra="forbid")

    scope_key: str = Field(description="Normalized scope identity key")
    degrade_claim_ids: list[str] = Field(
        default_factory=list,
        description="Claim IDs to transition from active_evidenced -> degraded.",
    )
    force_archive_claim_ids: list[str] = Field(default_factory=list)
    absent_verification_cycle_ids: list[str] = Field(default_factory=list)


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

    # ADR-013: System 1 / System 2 typed payload fields
    structural_evidence: list[StructuralEvidenceItem] | None = Field(
        default=None,
        description="Structural evidence items for store_system1_structural_evidence.",
    )
    verification_cycle: VerificationCycleInput | None = Field(
        default=None,
        description="Verification cycle metadata for record_system1_verification_cycle.",
    )
    system1_graph: System1StructuralGraphInput | None = Field(
        default=None,
        description="Raw System1 structural graph payload for store_system1_structural_graph.",
    )
    derivation: DerivationInput | None = Field(
        default=None,
        description="Derivation payload for derive_system2_semantic_claims.",
    )
    override: OverrideInput | None = Field(
        default=None,
        description="Override payload for apply_semantic_override.",
    )
    lifecycle_reconciliation: LifecycleReconciliationInput | None = Field(
        default=None,
        description="Lifecycle reconciliation payload for reconcile_semantic_lifecycle.",
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
    # store_relations_by_qname (ADR-012): qname-keyed relation list and fallback mode
    relations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Qname-keyed relation list for store_relations_by_qname.",
    )
    external_fallback: Literal["module", "reject"] = Field(
        default="module",
        description=(
            "Fallback for unresolved targets: 'module' creates an EXTERNAL Module entity; "
            "'reject' marks the relation as unresolved without creating any entity."
        ),
    )

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
    derivation: DeriveSystem1TopologyInput | None = Field(
        default=None,
        description="Topology derivation payload for derive_system1_topology (ADR-013).",
    )

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
    conflict: bool = Field(description="True when both org and user records exist and differ")


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
