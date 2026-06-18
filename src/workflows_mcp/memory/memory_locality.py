"""Operation-specific memory locality contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from .memory_errors import _raise_contract_error

CONTRACT_SCOPE_FIELDS: tuple[str, str, str, str] = ("palace", "wing", "room", "compartment")

INSUFFICIENT_LOCALITY = "INSUFFICIENT_LOCALITY"

ScopeSource = Literal["request", "token", "context"]


@dataclass(frozen=True)
class LocalityDecision:
    """Result of locality resolution: resolved scope fields and their sources."""

    resolved_scope: dict[str, str] = field(default_factory=dict)
    scope_source: dict[str, ScopeSource] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalityRequest:
    """Input for locality resolution — all operation and scope context."""

    operation: str
    request_scope: Mapping[str, Any]
    scope_token: str | None = None
    context_id: str | None = None
    token_scopes: Any = None
    context_scopes: Any = None
    graph_kind: str | None = None
    graph_from: str | None = None
    graph_to: str | None = None
    graph_ids: list[str] | None = None
    record_ids: list[str] | None = None
    record_superseded_by: str | None = None


def _is_uuid_ref(value: str | None) -> bool:
    """Return True when value is a valid UUID string."""
    if not value:
        return False
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def _operation_label(request: LocalityRequest) -> str:
    """Build human-readable operation label for error messages."""
    if request.operation == "graph_upsert" and request.graph_kind:
        return f"graph_upsert {request.graph_kind}"
    return request.operation


def _required_scope_fields(request: LocalityRequest) -> tuple[str, ...]:
    """Compute which scope fields are required for the given operation."""
    op = request.operation

    if op == "query":
        return ("palace",)

    if op == "ingest":
        return CONTRACT_SCOPE_FIELDS

    if op == "graph_upsert":
        if request.graph_kind == "place":
            return CONTRACT_SCOPE_FIELDS
        # link kind: UUID refs need no topology; name refs need complete topology
        if _is_uuid_ref(request.graph_from) and _is_uuid_ref(request.graph_to):
            return ()
        return CONTRACT_SCOPE_FIELDS

    return ()


def _lookup_scope_values(
    *,
    scopes: Any,
    scope_key: str | None,
    source_name: str,
) -> dict[str, Any]:
    """Safely resolve keyed scope payloads from execution context."""
    if scope_key is None:
        return {}
    if scopes is None:
        return {}
    if not isinstance(scopes, Mapping):
        _raise_contract_error(
            code="MEM_INVALID_CONTEXT_SCOPE",
            message=(
                f"execution context '{source_name}' must be a mapping; "
                f"received {type(scopes).__name__}"
            ),
        )

    scoped_value = scopes.get(scope_key)
    if scoped_value is None:
        return {}
    if not isinstance(scoped_value, Mapping):
        _raise_contract_error(
            code="MEM_INVALID_CONTEXT_SCOPE",
            message=(
                f"execution context '{source_name}[{scope_key}]' must be a mapping; "
                f"received {type(scoped_value).__name__}"
            ),
        )

    return dict(scoped_value)


def _fail_locality(
    *,
    operation: str,
    requirement: str,
    missing: list[str],
    accepted_sources: str,
    provided: str,
) -> None:
    """Raise INSUFFICIENT_LOCALITY with a structured message."""
    missing_text = ", ".join(missing)
    _raise_contract_error(
        code=INSUFFICIENT_LOCALITY,
        message=(
            f"{operation} requires {requirement}. "
            f"Missing: {missing_text}. "
            f"Accepted sources: {accepted_sources}. "
            f"Provided: {provided}."
        ),
        actionable_fix=(
            f"Retry {operation} with {accepted_sources}; include missing locality: {missing_text}."
        ),
    )


def resolve_memory_locality(request: LocalityRequest) -> LocalityDecision:
    """Compute resolved scope and enforce operation-specific locality requirements."""
    token_scope = _lookup_scope_values(
        scopes=request.token_scopes,
        scope_key=request.scope_token,
        source_name="memory_scope_tokens",
    )
    context_scope = _lookup_scope_values(
        scopes=request.context_scopes,
        scope_key=request.context_id,
        source_name="memory_context_scopes",
    )
    request_scope = dict(request.request_scope)

    # Resolve field-by-field with precedence: request > token > context
    resolved: dict[str, str] = {}
    sources: dict[str, ScopeSource] = {}
    for f in CONTRACT_SCOPE_FIELDS:
        req_val = request_scope.get(f)
        token_val = token_scope.get(f)
        context_val = context_scope.get(f)
        if isinstance(req_val, str) and req_val:
            resolved[f] = req_val
            sources[f] = "request"
        elif isinstance(token_val, str) and token_val:
            resolved[f] = token_val
            sources[f] = "token"
        elif isinstance(context_val, str) and context_val:
            resolved[f] = context_val
            sources[f] = "context"

    # Enforce ID requirements
    op = request.operation
    label = _operation_label(request)

    if op == "graph_delete":
        if not request.graph_ids:
            _fail_locality(
                operation=label,
                requirement="graph.ids for targeted deletion",
                missing=["graph.ids"],
                accepted_sources="graph.ids (list of entity or relation UUIDs)",
                provided="graph.ids=[]" if request.graph_ids is not None else "graph.ids=None",
            )
        return LocalityDecision(resolved_scope=resolved, scope_source=sources)

    if op in ("validate", "archive"):
        if not request.record_ids:
            _fail_locality(
                operation=label,
                requirement="record.ids for targeted lifecycle operations",
                missing=["record.ids"],
                accepted_sources="record.ids (list of memory UUIDs)",
                provided="record.ids=[]" if request.record_ids is not None else "record.ids=None",
            )
        return LocalityDecision(resolved_scope=resolved, scope_source=sources)

    if op == "supersede":
        missing_id_fields: list[str] = []
        if not request.record_ids:
            missing_id_fields.append("record.ids")
        if not request.record_superseded_by:
            missing_id_fields.append("record.superseded_by")
        if missing_id_fields:
            _fail_locality(
                operation=label,
                requirement="record.ids and record.superseded_by for supersede",
                missing=missing_id_fields,
                accepted_sources="record.ids and record.superseded_by (memory UUIDs)",
                provided=str(
                    {
                        "record.ids": request.record_ids,
                        "record.superseded_by": request.record_superseded_by,
                    }
                ),
            )
        return LocalityDecision(resolved_scope=resolved, scope_source=sources)

    # Enforce scope topology requirements
    required = _required_scope_fields(request)
    missing_scope = [f for f in required if f not in resolved]
    if missing_scope:
        # Build requirement and accepted sources labels per operation
        if op == "query":
            requirement = "palace locality for broad reads"
            accepted_sources = (
                "scope.palace, scope_token, context_id, "
                "active project defaults, or session active context"
            )
        elif op == "ingest":
            requirement = "complete topology locality for placement writes"
            accepted_sources = (
                "scope.palace/wing/room/compartment, scope_token, context_id, "
                "or operation-local checkpoint scope"
            )
        elif op == "graph_upsert" and request.graph_kind == "place":
            requirement = "complete topology locality for graph place placement"
            accepted_sources = (
                "scope.palace/wing/room/compartment, scope_token, context_id, "
                "or graph/onboard operation-local scope"
            )
        elif op == "graph_upsert":
            requirement = "complete topology locality for graph link name disambiguation"
            accepted_sources = (
                "UUID refs in graph.from/graph.to, or complete topology scope for name refs"
            )
        else:
            requirement = f"locality for {label}"
            accepted_sources = "scope, scope_token, or context_id"

        provided = str({f: resolved.get(f) for f in required})
        _fail_locality(
            operation=label,
            requirement=requirement,
            missing=missing_scope,
            accepted_sources=accepted_sources,
            provided=provided,
        )

    return LocalityDecision(resolved_scope=resolved, scope_source=sources)
