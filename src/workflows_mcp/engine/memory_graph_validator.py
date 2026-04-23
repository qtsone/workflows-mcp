"""Graph payload validator for the memory graph system.

Phase 3 — deterministic invariant checks with machine-readable error codes.

All errors are collected into a :class:`GraphValidationResult` so callers
receive the full list of violations in a single pass (fail-all semantics)
rather than short-circuiting on the first failure.

Error codes (all prefixed ``GRAPH_``):
- GRAPH_MISSING_NODE_TYPE     — required node type absent from payload
- GRAPH_MISSING_CORRIDOR_FIELD — required field absent from a corridor
- GRAPH_ILLEGAL_LINK          — forbidden direct semantic edge between levels
- GRAPH_ORPHAN_NODE           — semantic node with no corridor connectivity
- GRAPH_COMPLETENESS_FAILED   — summary code when ≥1 hard errors are present

The ``GRAPH_COMPLETENESS_FAILED`` code is surfaced at the envelope level;
individual violation details are embedded in ``violations``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory_graph_builder import (
    REQUIRED_CORRIDOR_FIELDS,
    REQUIRED_NODE_TYPES,
    GraphPayload,
    NodeType,
)

# ---------------------------------------------------------------------------
# Illegal direct semantic link rules
# ---------------------------------------------------------------------------

# Same-level primary semantic links are forbidden (intra-level edges).
# The rule is: if source and target share the same NodeType, the link is
# illegal regardless of semantic_type.
#
# Additionally, PALACE-level semantic edges beyond "contains" are forbidden
# at the structural level (they are governance policy violations).
#
# These rules are encoded as a frozenset of (source_type, target_type) pairs
# that are unconditionally forbidden for primary semantic edges.

_FORBIDDEN_SAME_LEVEL_PAIRS: frozenset[tuple[NodeType, NodeType]] = frozenset(
    {
        (NodeType.COMPARTMENT, NodeType.COMPARTMENT),
        (NodeType.ROOM, NodeType.ROOM),
        (NodeType.WING, NodeType.WING),
        (NodeType.PALACE, NodeType.PALACE),
    }
)

# Cross-level forbidden: PALACE may only link downward via "contains".
# Any Palace→* semantic edge that is NOT "contains" and NOT to WING is illegal.
# We represent this as a set of (source_type, allowed_target_type) for "contains" only.
_PALACE_ALLOWED_CONTAINS_TARGETS: frozenset[NodeType] = frozenset({NodeType.WING})


# ---------------------------------------------------------------------------
# Error and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphViolation:
    """Single invariant violation detected by the validator."""

    code: str
    """Machine-readable error code (e.g. GRAPH_MISSING_NODE_TYPE)."""

    message: str
    """Human-readable description of the violation."""

    context: dict[str, Any] = field(default_factory=dict)
    """Structured context for machine consumers (node ids, field names, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


@dataclass
class GraphValidationResult:
    """Outcome of :func:`validate_graph_payload`.

    ``valid`` is ``True`` iff ``violations`` is empty.
    When invalid, ``error_envelope`` is populated with a deterministic
    ``GRAPH_COMPLETENESS_FAILED`` envelope suitable for surfacing to callers.
    """

    valid: bool
    violations: list[GraphViolation] = field(default_factory=list)
    error_envelope: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [v.to_dict() for v in self.violations],
            "error_envelope": self.error_envelope,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_graph_payload(
    payload: GraphPayload | dict[str, Any],
    *,
    # Allow callers to pass raw dicts; we coerce via GraphPayload.from_dict.
    strict_orphan_check: bool = True,
) -> GraphValidationResult:
    """Validate a graph payload against the MemPalace ontology invariants.

    Performs four independent checks (all run before returning so violations
    are fully collected):

    1. **Required node types** — every type in REQUIRED_NODE_TYPES must have
       at least one node present.
    2. **Required corridor fields** — every corridor must contain all fields
       in REQUIRED_CORRIDOR_FIELDS with non-empty/non-None values.
    3. **Illegal direct semantic links** — same-level primary semantic edges
       and forbidden cross-level links.
    4. **Orphan semantic nodes** — nodes not referenced by any corridor in
       either direction.

    When *strict_orphan_check* is ``False``, the orphan check is skipped
    (useful when building incremental payloads).

    Args:
        payload: A :class:`GraphPayload` or raw ``dict`` (auto-coerced).
        strict_orphan_check: Whether to enforce the orphan node check.

    Returns:
        :class:`GraphValidationResult` — ``valid=True`` iff no violations.
    """
    if isinstance(payload, dict):
        payload = GraphPayload.from_dict(payload)

    violations: list[GraphViolation] = []

    # -----------------------------------------------------------------------
    # 1. Required node types
    # -----------------------------------------------------------------------
    present_types: set[NodeType] = {n.node_type for n in payload.nodes}
    for required_type in REQUIRED_NODE_TYPES:
        if required_type not in present_types:
            violations.append(
                GraphViolation(
                    code="GRAPH_MISSING_NODE_TYPE",
                    message=(
                        f"Required node type '{required_type.value}' is absent from the graph "
                        f"payload. At least one node of each type "
                        f"({', '.join(t.value for t in REQUIRED_NODE_TYPES)}) must be present."
                    ),
                    context={"missing_node_type": required_type.value},
                )
            )

    # -----------------------------------------------------------------------
    # 2. Required corridor fields
    # -----------------------------------------------------------------------
    for idx, corridor in enumerate(payload.corridors):
        raw_dict = corridor.to_dict()
        is_structural = corridor.semantic_type.value == "contains"
        for required_field in REQUIRED_CORRIDOR_FIELDS:
            # Structural (contains) corridors are system-generated topology links;
            # they do not require external evidence references.
            if required_field == "evidence" and is_structural:
                continue
            value = raw_dict.get(required_field)
            missing = value is None or value == "" or value == []
            if missing:
                violations.append(
                    GraphViolation(
                        code="GRAPH_MISSING_CORRIDOR_FIELD",
                        message=(
                            f"Corridor at index {idx} is missing required field "
                            f"'{required_field}'. "
                            f"All non-structural corridors must specify: "
                            f"{', '.join(REQUIRED_CORRIDOR_FIELDS)}."
                        ),
                        context={
                            "corridor_index": idx,
                            "missing_field": required_field,
                            "source_id": corridor.source_id,
                            "target_id": corridor.target_id,
                        },
                    )
                )

    # -----------------------------------------------------------------------
    # 3. Illegal direct semantic links
    # -----------------------------------------------------------------------
    node_type_by_id: dict[str, NodeType] = {n.node_id: n.node_type for n in payload.nodes}

    for idx, corridor in enumerate(payload.corridors):
        src_type = node_type_by_id.get(corridor.source_id)
        tgt_type = node_type_by_id.get(corridor.target_id)

        if src_type is None or tgt_type is None:
            # Unknown node references — skip illegal-link check for this corridor;
            # orphan check will catch dangling references.
            continue

        # 3a. Same-level forbidden pairs
        if (src_type, tgt_type) in _FORBIDDEN_SAME_LEVEL_PAIRS:
            violations.append(
                GraphViolation(
                    code="GRAPH_ILLEGAL_LINK",
                    message=(
                        f"Corridor at index {idx} creates an illegal same-level semantic link "
                        f"({src_type.value} → {tgt_type.value}). "
                        f"Direct same-level primary semantic edges are forbidden for all node "
                        f"types. Use cross-level or memory-mediated relations instead."
                    ),
                    context={
                        "corridor_index": idx,
                        "source_id": corridor.source_id,
                        "target_id": corridor.target_id,
                        "source_type": src_type.value,
                        "target_type": tgt_type.value,
                    },
                )
            )
            continue  # Don't double-report for same corridor

        # 3b. Palace semantic edges beyond "contains"
        if src_type == NodeType.PALACE:
            semantic_type_val = corridor.semantic_type.value
            if semantic_type_val != "contains":
                violations.append(
                    GraphViolation(
                        code="GRAPH_ILLEGAL_LINK",
                        message=(
                            f"Corridor at index {idx} creates an illegal Palace-level semantic "
                            f"edge with type '{semantic_type_val}'. "
                            f"Palace nodes may only use 'contains' edges for structural links. "
                            f"Use Wing or lower nodes for other semantic relations."
                        ),
                        context={
                            "corridor_index": idx,
                            "source_id": corridor.source_id,
                            "target_id": corridor.target_id,
                            "source_type": src_type.value,
                            "target_type": tgt_type.value,
                            "semantic_type": semantic_type_val,
                        },
                    )
                )

    # -----------------------------------------------------------------------
    # 4. Orphan semantic nodes (nodes with no corridor connectivity)
    # -----------------------------------------------------------------------
    if strict_orphan_check and payload.nodes:
        connected_ids: set[str] = set()
        for corridor in payload.corridors:
            connected_ids.add(corridor.source_id)
            connected_ids.add(corridor.target_id)

        for node in payload.nodes:
            if node.node_id not in connected_ids:
                violations.append(
                    GraphViolation(
                        code="GRAPH_ORPHAN_NODE",
                        message=(
                            f"Node '{node.node_id}' (type={node.node_type.value}, "
                            f"label={node.label!r}) is an orphan — it is not referenced "
                            f"by any corridor as source or target."
                        ),
                        context={
                            "node_id": node.node_id,
                            "node_type": node.node_type.value,
                            "label": node.label,
                        },
                    )
                )

    # -----------------------------------------------------------------------
    # Build result
    # -----------------------------------------------------------------------
    if not violations:
        return GraphValidationResult(valid=True)

    error_envelope = _build_completeness_failed_envelope(violations)
    return GraphValidationResult(
        valid=False,
        violations=violations,
        error_envelope=error_envelope,
    )


# ---------------------------------------------------------------------------
# Error envelope builders
# ---------------------------------------------------------------------------


def _build_completeness_failed_envelope(violations: list[GraphViolation]) -> dict[str, Any]:
    """Build a deterministic GRAPH_COMPLETENESS_FAILED error envelope."""
    return {
        "error": {
            "code": "GRAPH_COMPLETENESS_FAILED",
            "message": (
                f"Graph payload failed completeness validation with "
                f"{len(violations)} violation(s). "
                f"Inspect 'violations' for details and apply the actionable_fix for each."
            ),
            "retryable": False,
            "stage": "graph_validation",
            "actionable_fix": (
                "Ensure the graph payload contains at least one node of each required type "
                f"({', '.join(t.value for t in REQUIRED_NODE_TYPES)}), "
                "all corridors include the required fields "
                f"({', '.join(REQUIRED_CORRIDOR_FIELDS)}), "
                "no same-level primary semantic edges exist, "
                "and no orphan nodes are present."
            ),
            "violations": [v.to_dict() for v in violations],
        }
    }


def build_graph_error_envelope(
    *,
    code: str,
    message: str,
    stage: str,
    actionable_fix: str,
    retryable: bool = False,
    violations: list[GraphViolation] | None = None,
) -> dict[str, Any]:
    """Build a typed error envelope for graph operation failures.

    Suitable for surfacing from onboard/sync when a graph validation step fails.
    """
    envelope: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
        "stage": stage,
        "actionable_fix": actionable_fix,
    }
    if violations:
        envelope["violations"] = [v.to_dict() for v in violations]
    return {"error": envelope}
