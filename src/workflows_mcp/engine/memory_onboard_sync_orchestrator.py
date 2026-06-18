"""Programmatic and LLM onboard pipeline orchestrators.

Phase 4 — end-to-end onboard pipeline in programmatic mode with:
- Complete-graph gate: hard failure when graph completeness validator reports errors.
- Metadata-only compartments: 1:1 mapping for binary/unsupported files.
- Concise/debug response shaping: default omits debug fields, debug=True expands diagnostics.

Phase 5 — LLM ingestion mode (``mode="llm"``) with:
- Profile-based LLM configuration via the existing LLMConfigLoader registry.
- Strict reproducibility by default (temperature=0.0): deterministic error on violations.
- Explicit relaxed opt-in (``strict=False``): allows non-zero temperature profiles.
- INVALID_LLM_PROFILE error envelope for missing / invalid profiles.
- LLM provenance block (model/profile/version fingerprint) only when debug=True.

The resumable checkpoint state machine and its sync-delta/deletion-policy logic
live in ``project_flow_service`` (ADR-014); this module is the graph builder.

Design invariants:
- ``run_programmatic_onboard`` is the programmatic-mode entry point.
- ``run_llm_onboard`` is the LLM-mode entry point; callers pass an
  ``LLMOnboardRequest`` and receive an ``LLMOnboardResult``.
- Both share the same graph-building and graph-validation infrastructure.
- LLMConfigLoader is injected (not instantiated here) to reuse the existing registry.
- Graph validation is a hard gate for both modes.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .llm_config import LLMConfigLoader, ResolvedLLMConfig
from .memory_graph_builder import (
    CorridorSemanticType,
    GraphCorridor,
    GraphNode,
    GraphPayload,
    NodeType,
    build_structural_backbone,
)
from .memory_graph_validator import (
    GraphValidationResult,
    GraphViolation,
    validate_graph_payload,
)
from .memory_scope_resolver import (
    DEFAULT_PALACE_LABEL,
    normalize_scope,
    normalize_topology_label,
    scope_key,
)

if TYPE_CHECKING:
    from .memory_service import MemoryService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScannedFileEntry:
    """One file from a scan pass, before graph construction."""

    path: str
    """Relative path from scan root."""

    content: str
    """Decoded text content, or empty string for binary/skipped files."""

    size_bytes: int
    is_binary: bool = False
    is_unsupported: bool = False
    content_hash: str = ""

    @property
    def has_readable_content(self) -> bool:
        """True when the file can contribute text content to a compartment."""
        return not self.is_binary and not self.is_unsupported and bool(self.content.strip())

    @property
    def compartment_label(self) -> str:
        """Human-readable compartment label derived from the file path."""
        return Path(self.path).name or self.path


@dataclass
class CompartmentNode:
    """One graph compartment derived from a scanned file."""

    node_id: str
    path: str
    label: str
    is_metadata_only: bool
    """True when the file was binary/unsupported — content not stored."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgrammaticOnboardRequest:
    """Input to the programmatic onboard pipeline."""

    scope: dict[str, Any]
    """Scope dict (palace/wing/room/compartment hierarchy)."""

    files: list[ScannedFileEntry]
    """Scanned file entries to onboard (may include binary/unsupported)."""

    mode: Literal["programmatic"] = "programmatic"
    """Execution mode — only 'programmatic' is supported in Phase 4."""

    debug: bool = False
    """When True, include diagnostics and per-file details in the result."""

    provenance: str = "onboard"
    """Provenance tag for generated graph corridors."""

    confidence: float = 1.0
    """Confidence score for structural corridors."""


@dataclass
class ProgrammaticOnboardResult:
    """Output from the programmatic onboard pipeline."""

    status: Literal["completed", "failed"]
    scope: dict[str, str | None]
    """Normalised scope dict."""

    scope_key_value: str
    """Stable scope_key derived from normalised scope."""

    graph: GraphPayload | None
    """Validated graph payload, or None on failure."""

    compartments: list[CompartmentNode] = field(default_factory=list)
    """All compartment nodes including metadata-only ones."""

    metadata_only_count: int = 0
    """Number of compartments created from binary/unsupported files."""

    error: dict[str, Any] | None = None
    """GRAPH_COMPLETENESS_FAILED envelope when validation fails."""

    violations: list[GraphViolation] = field(default_factory=list)
    """Individual validation violations (populated on failure)."""

    diagnostics: dict[str, Any] | None = None
    """Debug-only diagnostics; None in concise mode."""

    def to_response_dict(self, *, debug: bool = False) -> dict[str, Any]:
        """Serialize to a response dict with verbosity controlled by *debug*."""
        if self.status == "failed":
            # Return the error envelope directly ({"error": {...}}).
            # self.error is already in envelope form {"error": {code, message, ...}}.
            return self.error if self.error is not None else {"error": {"code": "UNKNOWN_FAILURE"}}

        base: dict[str, Any] = {
            "status": "completed",
            "scope": self.scope,
            "scope_key": self.scope_key_value,
            "compartments_total": len(self.compartments),
            "metadata_only_count": self.metadata_only_count,
            "graph": {
                "nodes": len(self.graph.nodes) if self.graph else 0,
                "corridors": len(self.graph.corridors) if self.graph else 0,
            },
        }
        if debug and self.diagnostics:
            base["diagnostics"] = self.diagnostics
        return base


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stable_node_id(prefix: str, label: str) -> str:
    """Derive a stable deterministic node_id from prefix + label."""
    raw = f"{prefix}:{label}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_file_binary_or_unsupported(entry: ScannedFileEntry) -> bool:
    """Return True when the file cannot contribute readable content."""
    return entry.is_binary or entry.is_unsupported or not entry.has_readable_content


def _build_metadata_only_compartment(
    entry: ScannedFileEntry,
    *,
    room_id: str,
    provenance: str,
    confidence: float,
) -> tuple[GraphNode, GraphCorridor]:
    """Build a metadata-only compartment node + containment corridor for a binary file."""
    node_id = _stable_node_id("compartment", entry.path)
    node = GraphNode(
        node_id=node_id,
        node_type=NodeType.COMPARTMENT,
        label=entry.compartment_label,
        metadata={
            "path": entry.path,
            "size_bytes": entry.size_bytes,
            "content_hash": entry.content_hash,
            "metadata_only": True,
            "reason": "binary" if entry.is_binary else "unsupported",
        },
    )
    corridor = GraphCorridor(
        source_id=room_id,
        target_id=node_id,
        semantic_type=CorridorSemanticType.CONTAINS,
        confidence=confidence,
        provenance=provenance,
        evidence=[],
    )
    return node, corridor


def _build_content_compartment(
    entry: ScannedFileEntry,
    *,
    room_id: str,
    provenance: str,
    confidence: float,
) -> tuple[GraphNode, GraphCorridor]:
    """Build a standard content compartment node + containment corridor."""
    node_id = _stable_node_id("compartment", entry.path)
    node = GraphNode(
        node_id=node_id,
        node_type=NodeType.COMPARTMENT,
        label=entry.compartment_label,
        metadata={
            "path": entry.path,
            "size_bytes": entry.size_bytes,
            "content_hash": entry.content_hash,
            "metadata_only": False,
        },
    )
    corridor = GraphCorridor(
        source_id=room_id,
        target_id=node_id,
        semantic_type=CorridorSemanticType.CONTAINS,
        confidence=confidence,
        provenance=provenance,
        evidence=[],
    )
    return node, corridor


def _build_graph_from_files(
    files: list[ScannedFileEntry],
    *,
    scope: dict[str, Any],
    provenance: str = "onboard",
    confidence: float = 1.0,
) -> tuple[GraphPayload, list[CompartmentNode], int]:
    """Build a complete graph payload from a list of scanned files.

    When no files are provided, a single placeholder compartment is created so
    the graph satisfies the REQUIRED_NODE_TYPES invariant.

    Returns:
        (payload, compartment_nodes, metadata_only_count)
    """
    normalized = normalize_scope(scope)

    palace_label = normalized.get("palace") or DEFAULT_PALACE_LABEL
    wing_label = normalize_topology_label(normalized.get("wing"))
    room_label = normalize_topology_label(normalized.get("room"))

    palace_id = _stable_node_id("palace", palace_label)
    wing_id = _stable_node_id("wing", wing_label)
    room_id = _stable_node_id("room", room_label)

    backbone = build_structural_backbone(
        palace_id=palace_id,
        palace_label=palace_label,
        wing_id=wing_id,
        wing_label=wing_label,
        room_id=room_id,
        room_label=room_label,
        provenance=provenance,
        confidence=confidence,
    )

    compartment_nodes_raw: list[GraphNode] = []
    compartment_corridors: list[GraphCorridor] = []
    compartment_infos: list[CompartmentNode] = []
    metadata_only_count = 0

    if not files:
        # Placeholder compartment to satisfy required node type invariant
        placeholder_id = _stable_node_id("compartment", f"{room_label}:placeholder")
        placeholder_node = GraphNode(
            node_id=placeholder_id,
            node_type=NodeType.COMPARTMENT,
            label="placeholder",
            metadata={"metadata_only": True, "reason": "no_files"},
        )
        compartment_nodes_raw.append(placeholder_node)
        compartment_corridors.append(
            GraphCorridor(
                source_id=room_id,
                target_id=placeholder_id,
                semantic_type=CorridorSemanticType.CONTAINS,
                confidence=confidence,
                provenance=provenance,
                evidence=[],
            )
        )
        compartment_infos.append(
            CompartmentNode(
                node_id=placeholder_id,
                path="",
                label="placeholder",
                is_metadata_only=True,
                metadata={"reason": "no_files"},
            )
        )
        metadata_only_count = 1
    else:
        for entry in files:
            if _is_file_binary_or_unsupported(entry):
                node, corridor = _build_metadata_only_compartment(
                    entry, room_id=room_id, provenance=provenance, confidence=confidence
                )
                metadata_only_count += 1
                compartment_infos.append(
                    CompartmentNode(
                        node_id=node.node_id,
                        path=entry.path,
                        label=node.label,
                        is_metadata_only=True,
                        metadata=node.metadata,
                    )
                )
            else:
                node, corridor = _build_content_compartment(
                    entry, room_id=room_id, provenance=provenance, confidence=confidence
                )
                compartment_infos.append(
                    CompartmentNode(
                        node_id=node.node_id,
                        path=entry.path,
                        label=node.label,
                        is_metadata_only=False,
                        metadata=node.metadata,
                    )
                )
            compartment_nodes_raw.append(node)
            compartment_corridors.append(corridor)

    all_nodes = [backbone.palace, backbone.wing, backbone.room] + compartment_nodes_raw
    all_corridors = backbone.corridors + compartment_corridors

    return (
        GraphPayload(nodes=all_nodes, corridors=all_corridors),
        compartment_infos,
        metadata_only_count,
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_programmatic_onboard(
    request: ProgrammaticOnboardRequest,
) -> ProgrammaticOnboardResult:
    """Execute the programmatic onboard pipeline synchronously.

    Pipeline stages:
    1. Normalize and derive scope_key.
    2. Build graph payload from scanned files (binary files → metadata-only compartments).
    3. Validate graph payload — hard failure on GRAPH_COMPLETENESS_FAILED.
    4. Return ``ProgrammaticOnboardResult`` shaped by ``request.debug``.

    Args:
        request: Fully-populated onboard request.

    Returns:
        ProgrammaticOnboardResult with status='completed' or status='failed'.
    """
    if request.mode != "programmatic":
        return ProgrammaticOnboardResult(
            status="failed",
            scope=normalize_scope(request.scope),
            scope_key_value=scope_key(request.scope),
            graph=None,
            error={
                "error": {
                    "code": "UNSUPPORTED_MODE",
                    "message": (
                        f"Mode {request.mode!r} is not supported in Phase 4. "
                        "Only 'programmatic' mode is available."
                    ),
                    "retryable": False,
                    "stage": "mode_validation",
                    "actionable_fix": "Set mode='programmatic'.",
                }
            },
        )

    normalized_scope = normalize_scope(request.scope)
    key = scope_key(request.scope)

    # Stage 1: Build graph
    graph_payload, compartment_infos, metadata_only_count = _build_graph_from_files(
        request.files,
        scope=request.scope,
        provenance=request.provenance,
        confidence=request.confidence,
    )

    # Stage 2: Validate graph — hard gate
    validation_result: GraphValidationResult = validate_graph_payload(graph_payload)
    if not validation_result.valid:
        diagnostics: dict[str, Any] | None = None
        if request.debug:
            diagnostics = {
                "violations": [v.to_dict() for v in validation_result.violations],
                "graph": graph_payload.to_dict(),
                "compartments": [
                    {
                        "node_id": c.node_id,
                        "path": c.path,
                        "is_metadata_only": c.is_metadata_only,
                    }
                    for c in compartment_infos
                ],
            }
        return ProgrammaticOnboardResult(
            status="failed",
            scope=normalized_scope,
            scope_key_value=key,
            graph=None,
            compartments=compartment_infos,
            metadata_only_count=metadata_only_count,
            error=validation_result.error_envelope,
            violations=validation_result.violations,
            diagnostics=diagnostics,
        )

    # Stage 3: Build result
    diagnostics = None
    if request.debug:
        diagnostics = {
            "graph": graph_payload.to_dict(),
            "compartments": [
                {
                    "node_id": c.node_id,
                    "path": c.path,
                    "label": c.label,
                    "is_metadata_only": c.is_metadata_only,
                    "metadata": c.metadata,
                }
                for c in compartment_infos
            ],
        }

    return ProgrammaticOnboardResult(
        status="completed",
        scope=normalized_scope,
        scope_key_value=key,
        graph=graph_payload,
        compartments=compartment_infos,
        metadata_only_count=metadata_only_count,
        error=None,
        violations=[],
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Integration helpers for tools_memory.py
# ---------------------------------------------------------------------------


def build_programmatic_onboard_response(
    result: ProgrammaticOnboardResult,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Build the top-level response dict for the onboard tool.

    Concise (debug=False):
    - status, scope, scope_key, compartments_total, metadata_only_count, graph summary
    - No per-file content, no internal diagnostics

    Debug (debug=True):
    - All concise fields
    - diagnostics block with full graph payload and per-compartment detail
    """
    return result.to_response_dict(debug=debug)


def classify_scan_files_for_programmatic_mode(
    scanned_files: list[dict[str, Any]],
    *,
    base_path: str = ".",
) -> list[ScannedFileEntry]:
    """Convert raw scan output dicts to typed ScannedFileEntry objects.

    Binary detection:
    - Files with empty content after strip are treated as binary/unsupported.
    - Files with non-empty content are treated as readable.

    This is a heuristic for files that run_readfiles_scan already skipped
    (binary files are not returned by run_readfiles_scan). For programmatic
    mode, metadata-only compartments are created for any file entry with no
    readable content (regardless of underlying reason).

    Args:
        scanned_files: Raw file dicts from run_readfiles_scan or test fixtures.
        base_path: Base directory (used for hash computation if content present).

    Returns:
        List of ScannedFileEntry, one per input file.
    """
    entries: list[ScannedFileEntry] = []
    for f in scanned_files:
        path = str(f.get("path") or "")
        content = str(f.get("content") or "")
        size_bytes = int(f.get("size_bytes") or 0)
        content_hash = str(f.get("content_hash") or "")

        # Compute hash if not provided
        if not content_hash and content:
            content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

        is_binary = bool(f.get("is_binary", False))
        is_unsupported = bool(f.get("is_unsupported", False))

        # Files with no readable content after strip are metadata-only
        if not is_binary and not is_unsupported and not content.strip():
            is_unsupported = True

        entries.append(
            ScannedFileEntry(
                path=path,
                content=content,
                size_bytes=size_bytes,
                is_binary=is_binary,
                is_unsupported=is_unsupported,
                content_hash=content_hash,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Phase 5 — LLM ingestion mode
# ---------------------------------------------------------------------------

# Strict mode: temperature must be exactly 0.0 for full determinism.
_STRICT_REQUIRED_TEMPERATURE: float = 0.0


@dataclass
class LLMOnboardRequest:
    """Input to the LLM onboard pipeline (Phase 5).

    The LLM pipeline reuses the same graph-building and graph-validation
    infrastructure as the programmatic pipeline.  The profile is resolved via
    the injected ``LLMConfigLoader`` — no parallel registry is created.

    Args:
        scope: Scope dict (palace/wing/room/compartment hierarchy).
        files: Scanned file entries (may include binary/unsupported).
        profile: LLM profile name to resolve via LLMConfigLoader.  Required.
        strict: When True (default) the resolved profile must have
            ``temperature=0.0``.  Pass ``strict=False`` to opt in to relaxed
            (non-zero temperature) behaviour.
        debug: When True, include LLM provenance and diagnostics in result.
        provenance: Provenance tag for generated graph corridors.
        confidence: Confidence score for structural corridors.
    """

    scope: dict[str, Any]
    files: list[ScannedFileEntry]
    profile: str
    """LLM profile name — must exist in the LLMConfigLoader registry."""

    mode: Literal["llm"] = "llm"
    strict: bool = True
    """Strict reproducibility (default). Require temperature=0.0 on the resolved profile."""

    debug: bool = False
    provenance: str = "onboard"
    confidence: float = 1.0


@dataclass
class LLMProvenanceInfo:
    """Fingerprint of the LLM configuration used for a pipeline run."""

    profile: str
    model: str
    provider: str
    temperature: float | None
    strict: bool
    reproducibility: Literal["strict", "relaxed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "strict": self.strict,
            "reproducibility": self.reproducibility,
        }


@dataclass
class LLMOnboardResult:
    """Output from the LLM onboard pipeline (Phase 5).

    Mirrors ``ProgrammaticOnboardResult`` shape with the addition of the
    ``llm_provenance`` field which is only populated when ``debug=True``.
    """

    status: Literal["completed", "failed"]
    scope: dict[str, str | None]
    scope_key_value: str
    graph: GraphPayload | None

    compartments: list[CompartmentNode] = field(default_factory=list)
    metadata_only_count: int = 0
    error: dict[str, Any] | None = None
    violations: list[GraphViolation] = field(default_factory=list)
    diagnostics: dict[str, Any] | None = None
    llm_provenance: LLMProvenanceInfo | None = None
    """LLM fingerprint (model/profile/temperature/reproducibility). None in concise mode."""

    def to_response_dict(self, *, debug: bool = False) -> dict[str, Any]:
        """Serialize to response dict.  LLM provenance emitted only when debug=True."""
        if self.status == "failed":
            # Return the error envelope directly ({"error": {...}}).
            return self.error if self.error is not None else {"error": {"code": "UNKNOWN_FAILURE"}}

        base: dict[str, Any] = {
            "status": "completed",
            "scope": self.scope,
            "scope_key": self.scope_key_value,
            "compartments_total": len(self.compartments),
            "metadata_only_count": self.metadata_only_count,
            "graph": {
                "nodes": len(self.graph.nodes) if self.graph else 0,
                "corridors": len(self.graph.corridors) if self.graph else 0,
            },
        }
        if debug:
            if self.diagnostics:
                base["diagnostics"] = self.diagnostics
            if self.llm_provenance is not None:
                base["llm_provenance"] = self.llm_provenance.to_dict()
        return base


# ---------------------------------------------------------------------------
# Profile validation helpers
# ---------------------------------------------------------------------------


def _resolve_llm_profile(
    profile: str,
    loader: LLMConfigLoader,
) -> ResolvedLLMConfig:
    """Resolve *profile* via *loader*, raising with INVALID_LLM_PROFILE semantics.

    Wraps ``LLMConfigLoader.resolve_profile`` so that any ``ValueError`` from
    the loader (unknown profile, missing provider, etc.) is surfaced as a
    descriptive Python ``ValueError`` whose message can be embedded in the
    ``INVALID_LLM_PROFILE`` error envelope.

    Raises:
        ValueError: If the profile name is empty.
        ValueError: If the profile or its provider cannot be resolved.
    """
    if not profile or not profile.strip():
        raise ValueError("LLM profile name must be a non-empty string.")
    try:
        resolved = loader.resolve_profile(profile=profile)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if resolved is None:
        # resolve_profile returns None only when profile is None — should not
        # happen here since we guard against empty string above, but be safe.
        raise ValueError(
            f"Profile '{profile}' could not be resolved (returned None). "
            "Ensure the profile exists in the admin /llm configuration."
        )
    return resolved


def _build_invalid_profile_envelope(
    profile: str,
    reason: str,
    *,
    stage: str = "profile_resolution",
) -> dict[str, Any]:
    """Build a deterministic INVALID_LLM_PROFILE error envelope."""
    return {
        "error": {
            "code": "INVALID_LLM_PROFILE",
            "message": (f"LLM profile '{profile}' is invalid or missing: {reason}"),
            "retryable": False,
            "stage": stage,
            "actionable_fix": (
                "Ensure the profile is defined in the admin /llm configuration "
                "and its provider is also configured."
            ),
        }
    }


def _build_strict_violation_envelope(
    profile: str,
    temperature: float | None,
) -> dict[str, Any]:
    """Build a deterministic STRICT_REPRODUCIBILITY_VIOLATION error envelope."""
    return {
        "error": {
            "code": "STRICT_REPRODUCIBILITY_VIOLATION",
            "message": (
                f"Profile '{profile}' has temperature={temperature!r} which violates "
                f"strict reproducibility mode (required: temperature=0.0). "
                "Pass strict=False to opt in to relaxed mode."
            ),
            "retryable": False,
            "stage": "reproducibility_check",
            "actionable_fix": (
                "Either set temperature=0.0 in the profile definition, "
                "or pass strict=False to allow non-zero temperature."
            ),
        }
    }


# ---------------------------------------------------------------------------
# LLM pipeline entry point
# ---------------------------------------------------------------------------


def run_llm_onboard(
    request: LLMOnboardRequest,
    *,
    loader: LLMConfigLoader,
) -> LLMOnboardResult:
    """Execute the LLM onboard pipeline synchronously.

    Pipeline stages:
    1. Validate mode guard (mode must be "llm").
    2. Resolve LLM profile via injected loader — INVALID_LLM_PROFILE on failure.
    3. Enforce strict reproducibility (temperature=0.0) unless strict=False.
    4. Build graph payload from scanned files.
    5. Validate graph payload — hard gate (GRAPH_COMPLETENESS_FAILED).
    6. Return LLMOnboardResult with LLM provenance only when debug=True.

    Args:
        request: Fully-populated LLM onboard request.
        loader: Injected LLMConfigLoader (reuses existing registry; no parallel instance).

    Returns:
        LLMOnboardResult with status='completed' or status='failed'.
    """
    normalized_scope = normalize_scope(request.scope)
    key = scope_key(request.scope)

    # Stage 1: Mode guard
    if request.mode != "llm":
        return LLMOnboardResult(
            status="failed",
            scope=normalized_scope,
            scope_key_value=key,
            graph=None,
            error={
                "error": {
                    "code": "UNSUPPORTED_MODE",
                    "message": (
                        f"Mode {request.mode!r} is not supported by run_llm_onboard. "
                        "Use mode='llm'."
                    ),
                    "retryable": False,
                    "stage": "mode_validation",
                    "actionable_fix": "Set mode='llm'.",
                }
            },
        )

    # Stage 2: Profile resolution
    try:
        resolved = _resolve_llm_profile(request.profile, loader)
    except ValueError as exc:
        return LLMOnboardResult(
            status="failed",
            scope=normalized_scope,
            scope_key_value=key,
            graph=None,
            error=_build_invalid_profile_envelope(request.profile, reason=str(exc)),
        )

    # Stage 3: Strict reproducibility check
    temperature = resolved.temperature
    if request.strict and temperature != _STRICT_REQUIRED_TEMPERATURE:
        return LLMOnboardResult(
            status="failed",
            scope=normalized_scope,
            scope_key_value=key,
            graph=None,
            error=_build_strict_violation_envelope(request.profile, temperature),
        )

    reproducibility: Literal["strict", "relaxed"] = "strict" if request.strict else "relaxed"
    provenance_info = LLMProvenanceInfo(
        profile=request.profile,
        model=resolved.model,
        provider=resolved.provider,
        temperature=temperature,
        strict=request.strict,
        reproducibility=reproducibility,
    )
    logger.info(
        "llm_onboard: profile=%r model=%r provider=%r temperature=%r reproducibility=%s",
        request.profile,
        resolved.model,
        resolved.provider,
        temperature,
        reproducibility,
    )

    # Stage 4: Build graph
    graph_payload, compartment_infos, metadata_only_count = _build_graph_from_files(
        request.files,
        scope=request.scope,
        provenance=request.provenance,
        confidence=request.confidence,
    )

    # Stage 5: Validate graph — hard gate
    validation_result: GraphValidationResult = validate_graph_payload(graph_payload)
    if not validation_result.valid:
        diagnostics: dict[str, Any] | None = None
        if request.debug:
            diagnostics = {
                "violations": [v.to_dict() for v in validation_result.violations],
                "graph": graph_payload.to_dict(),
                "compartments": [
                    {
                        "node_id": c.node_id,
                        "path": c.path,
                        "is_metadata_only": c.is_metadata_only,
                    }
                    for c in compartment_infos
                ],
            }
        return LLMOnboardResult(
            status="failed",
            scope=normalized_scope,
            scope_key_value=key,
            graph=None,
            compartments=compartment_infos,
            metadata_only_count=metadata_only_count,
            error=validation_result.error_envelope,
            violations=validation_result.violations,
            diagnostics=diagnostics,
            llm_provenance=provenance_info if request.debug else None,
        )

    # Stage 6: Build result
    diagnostics = None
    if request.debug:
        diagnostics = {
            "graph": graph_payload.to_dict(),
            "compartments": [
                {
                    "node_id": c.node_id,
                    "path": c.path,
                    "label": c.label,
                    "is_metadata_only": c.is_metadata_only,
                    "metadata": c.metadata,
                }
                for c in compartment_infos
            ],
        }

    return LLMOnboardResult(
        status="completed",
        scope=normalized_scope,
        scope_key_value=key,
        graph=graph_payload,
        compartments=compartment_infos,
        metadata_only_count=metadata_only_count,
        error=None,
        violations=[],
        diagnostics=diagnostics,
        llm_provenance=provenance_info if request.debug else None,
    )


def build_llm_onboard_response(
    result: LLMOnboardResult,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Build the top-level response dict for LLM onboard results.

    Concise (debug=False):
    - status, scope, scope_key, compartments_total, metadata_only_count, graph summary
    - No LLM provenance, no diagnostics

    Debug (debug=True):
    - All concise fields
    - llm_provenance block with model/profile/temperature/reproducibility fingerprint
    - diagnostics block with full graph payload and per-compartment detail
    """
    return result.to_response_dict(debug=debug)


# ---------------------------------------------------------------------------
# ADR-013 Task 12 — System 1 structural evidence extraction from file topology
# ---------------------------------------------------------------------------


def _extract_structural_evidence_from_files(
    files: list[ScannedFileEntry],
) -> list[dict[str, Any]]:
    """Extract module-level structural evidence from scanned file topology.

    Produces one ``structural_module`` evidence item per readable file.  Each
    item uses the file path as the stable anchor ID so evidence rows can be
    correlated with scan-derived compartment nodes.

    This is the seam available to the watcher/sync path without invoking the
    TreeSitter executor.  Class- and function-level evidence is produced by the
    TreeSitter executor in the ``system1-scan`` workflow and falls outside the
    scope of this orchestrator.

    Args:
        files: Scanned file entries from the onboard request.

    Returns:
        List of evidence item dicts compatible with ``StructuralEvidenceItem``.
        Empty when all files are binary or have no readable content.
    """
    items: list[dict[str, Any]] = []
    for entry in files:
        if not entry.has_readable_content:
            continue
        path = entry.path
        items.append(
            {
                "entity_stable_id": path,
                "entity_type": "module",
                "evidence_category": "structural_module",
                "evidence_data": {
                    "path": path,
                    "size_bytes": entry.size_bytes,
                    "content_hash": entry.content_hash,
                },
            }
        )
    return items


# ---------------------------------------------------------------------------
# ADR-013 Task 6 — async wrappers with System 1 cycle recording
# ---------------------------------------------------------------------------


async def run_programmatic_onboard_with_cycle_recording(
    request: ProgrammaticOnboardRequest,
    *,
    memory_service: MemoryService,
) -> ProgrammaticOnboardResult:
    """Execute programmatic onboard and record a System 1 verification cycle on success.

    Wraps ``run_programmatic_onboard`` (synchronous) and, when the result has
    ``status='completed'``, records a successful verification cycle in
    ``knowledge_verification_cycles`` via ``MemoryService``.

    Fail-closed: if the cycle recording operation itself fails, the exception
    propagates to the caller rather than silently proceeding.  A failed onboard
    (``status='failed'``) does not record any cycle.

    Args:
        request: Fully-populated programmatic onboard request.
        memory_service: Injected ``MemoryService`` instance for cycle persistence.

    Returns:
        ``ProgrammaticOnboardResult`` — unchanged from ``run_programmatic_onboard``.

    Raises:
        Exception: Re-raised from ``MemoryService.execute`` if cycle recording fails.
    """
    from .memory_schema import MemoryRequest

    result = run_programmatic_onboard(request)
    if result.status != "completed":
        return result

    covered_scope = dict(result.scope)

    # --- ADR-013 Task 12: write System 1 structural evidence before cycle record ---
    # Extract module-level structural evidence from scanned file topology.
    # This is the seam available to the sync/watcher path without TreeSitter:
    # each readable file contributes one structural_module evidence item keyed
    # by its stable path ID.  TreeSitter-level class/function evidence is
    # produced by the TreeSitter executor block in the system1-scan workflow
    # and is outside the scope of this orchestrator.
    evidence_items = _extract_structural_evidence_from_files(request.files)
    if evidence_items:
        evidence_request = MemoryRequest.model_validate(
            {
                "operation": "store_system1_structural_evidence",
                "scope": covered_scope,
                "record": {
                    "format": "structured",
                    "structural_evidence": evidence_items,
                },
            }
        )
        evidence_result = await memory_service.execute(evidence_request)
        if evidence_result.manage is None or not evidence_result.manage.success:
            error_detail = (
                evidence_result.manage.error
                if evidence_result.manage is not None
                else "no manage result"
            )
            raise RuntimeError(
                f"store_system1_structural_evidence failed during watcher sync: {error_detail}"
            )
        logger.info(
            "programmatic_onboard: stored %d structural evidence items scope_key=%r",
            evidence_result.manage.stored_count,
            result.scope_key_value,
        )

    record_request = MemoryRequest.model_validate(
        {
            "operation": "record_system1_verification_cycle",
            "scope": covered_scope,
            "record": {
                "format": "structured",
                "verification_cycle": {
                    "success": True,
                    "covered_scope": covered_scope,
                },
            },
        }
    )
    cycle_result = await memory_service.execute(record_request)
    if cycle_result.manage is None or not cycle_result.manage.success:
        error_detail = (
            cycle_result.manage.error if cycle_result.manage is not None else "no manage result"
        )
        raise RuntimeError(
            f"record_system1_verification_cycle failed after successful onboard: {error_detail}"
        )
    logger.info(
        "programmatic_onboard: recorded verification cycle cycle_id=%r scope_key=%r",
        cycle_result.manage.cycle_id,
        result.scope_key_value,
    )
    return result
