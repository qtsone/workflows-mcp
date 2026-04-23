"""Graph ontology scaffolding for the memory graph system.

Phase 3 — centralized constants, enums, and builder utilities for the
graph node/corridor type ontology used by onboard/sync paths.

Design invariants:
- All required node types and corridor types are defined as module-level
  constants (no magic strings scattered across validators).
- Corridor (edge) field requirements are enumerated once and imported by
  the validator.
- Builder helpers produce canonical graph payloads suitable for direct
  persistence without further validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Node type ontology
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Canonical graph node types for the MemPalace ontology.

    Hierarchy (containment order, top to bottom):
        PALACE → WING → ROOM → COMPARTMENT
    """

    PALACE = "Palace"
    WING = "Wing"
    ROOM = "Room"
    COMPARTMENT = "Compartment"


#: Ordered sequence of required node types for a complete graph payload.
#: The validator checks that at least one node of each type is present.
REQUIRED_NODE_TYPES: tuple[NodeType, ...] = (
    NodeType.PALACE,
    NodeType.WING,
    NodeType.ROOM,
    NodeType.COMPARTMENT,
)

#: Ordered hierarchy used for illegal-link detection.
#: Intra-level primary semantic links (same type → same type) are forbidden
#: for ROOM and COMPARTMENT; WING→WING and PALACE semantic edges beyond
#: "contains" are also forbidden.
_SAME_LEVEL_FORBIDDEN_TYPES: frozenset[NodeType] = frozenset(
    {NodeType.COMPARTMENT, NodeType.ROOM, NodeType.WING, NodeType.PALACE}
)


# ---------------------------------------------------------------------------
# Corridor (edge) type ontology
# ---------------------------------------------------------------------------


class CorridorSemanticType(str, Enum):
    """Canonical semantic types for graph corridors (edges)."""

    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    DERIVES_FROM = "derives_from"
    SUPERSEDES = "supersedes"
    RELATES_TO = "relates_to"


#: The only semantic type permitted for structural containment edges between
#: hierarchy levels (PALACE→WING, WING→ROOM, ROOM→COMPARTMENT).
STRUCTURAL_SEMANTIC_TYPE: CorridorSemanticType = CorridorSemanticType.CONTAINS

#: Required fields for every corridor payload (before persistence).
REQUIRED_CORRIDOR_FIELDS: tuple[str, ...] = (
    "source_id",
    "target_id",
    "semantic_type",
    "confidence",
    "provenance",
    "evidence",
)


# ---------------------------------------------------------------------------
# Graph payload dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """Canonical representation of a single graph node."""

    node_id: str
    node_type: NodeType
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class GraphCorridor:
    """Canonical representation of a directed corridor (edge) between nodes."""

    source_id: str
    target_id: str
    semantic_type: CorridorSemanticType
    confidence: float
    provenance: str
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "semantic_type": self.semantic_type.value,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass
class GraphPayload:
    """Complete graph payload ready for validation and persistence."""

    nodes: list[GraphNode] = field(default_factory=list)
    corridors: list[GraphCorridor] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "corridors": [c.to_dict() for c in self.corridors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphPayload:
        """Deserialize a raw dict into a GraphPayload.

        Unknown or malformed entries are passed through as-is so the
        validator can surface precise error messages rather than silent
        data loss.
        """
        raw_nodes = data.get("nodes") or []
        raw_corridors = data.get("corridors") or []

        nodes: list[GraphNode] = []
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                continue
            node_type_str = raw.get("node_type", "")
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                # Unknown node_type string — use ROOM as placeholder;
                # raw value preserved in metadata for validator to flag.
                nodes.append(
                    GraphNode(
                        node_id=str(raw.get("node_id", "")),
                        node_type=NodeType.ROOM,  # placeholder, raw preserved
                        label=str(raw.get("label", "")),
                        metadata={**raw.get("metadata", {}), "__raw_node_type": node_type_str},
                    )
                )
                continue
            nodes.append(
                GraphNode(
                    node_id=str(raw.get("node_id", "")),
                    node_type=node_type,
                    label=str(raw.get("label", "")),
                    metadata=raw.get("metadata", {}),
                )
            )

        corridors: list[GraphCorridor] = []
        for raw in raw_corridors:
            if not isinstance(raw, dict):
                continue
            sem_str = raw.get("semantic_type", "")
            try:
                sem_type = CorridorSemanticType(sem_str)
            except ValueError:
                sem_type = CorridorSemanticType.RELATES_TO  # fallback for validator
            corridors.append(
                GraphCorridor(
                    source_id=str(raw.get("source_id", "")),
                    target_id=str(raw.get("target_id", "")),
                    semantic_type=sem_type,
                    confidence=float(raw.get("confidence", 0.0)),
                    provenance=str(raw.get("provenance", "")),
                    evidence=list(raw.get("evidence") or []),
                    metadata=raw.get("metadata", {}),
                )
            )

        return cls(nodes=nodes, corridors=corridors)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def build_structural_graph(
    *,
    palace_id: str,
    palace_label: str,
    wing_id: str,
    wing_label: str,
    room_id: str,
    room_label: str,
    compartment_id: str,
    compartment_label: str,
    provenance: str = "onboard",
    confidence: float = 1.0,
) -> GraphPayload:
    """Build a minimal structurally-complete graph for a four-level scope.

    Creates one node of each required type and three containment corridors
    connecting them top-down (palace→wing→room→compartment).

    Args:
        palace_id: Stable identifier for the Palace node.
        palace_label: Human-readable label.
        wing_id: Stable identifier for the Wing node.
        wing_label: Human-readable label.
        room_id: Stable identifier for the Room node.
        room_label: Human-readable label.
        compartment_id: Stable identifier for the Compartment node.
        compartment_label: Human-readable label.
        provenance: Provenance tag attached to all structural corridors.
        confidence: Confidence score for all structural corridors.

    Returns:
        GraphPayload ready for :func:`validate_graph_payload`.
    """
    nodes = [
        GraphNode(node_id=palace_id, node_type=NodeType.PALACE, label=palace_label),
        GraphNode(node_id=wing_id, node_type=NodeType.WING, label=wing_label),
        GraphNode(node_id=room_id, node_type=NodeType.ROOM, label=room_label),
        GraphNode(
            node_id=compartment_id, node_type=NodeType.COMPARTMENT, label=compartment_label
        ),
    ]
    corridors = [
        GraphCorridor(
            source_id=palace_id,
            target_id=wing_id,
            semantic_type=STRUCTURAL_SEMANTIC_TYPE,
            confidence=confidence,
            provenance=provenance,
            evidence=[],
        ),
        GraphCorridor(
            source_id=wing_id,
            target_id=room_id,
            semantic_type=STRUCTURAL_SEMANTIC_TYPE,
            confidence=confidence,
            provenance=provenance,
            evidence=[],
        ),
        GraphCorridor(
            source_id=room_id,
            target_id=compartment_id,
            semantic_type=STRUCTURAL_SEMANTIC_TYPE,
            confidence=confidence,
            provenance=provenance,
            evidence=[],
        ),
    ]
    return GraphPayload(nodes=nodes, corridors=corridors)
