"""Graph traversal engine for Memory and graph retrieval handlers.

Provides BFS/DFS subgraph traversal, shortest-path search, and per-entity
statistics against the ``knowledge_relations`` and ``knowledge_entities``
tables.

All public helpers accept a connected ``DatabaseBackendBase`` instance and
return typed ``GraphResult`` payloads.  The memory service and executor
handlers delegate here and remain thin.

Temporal semantics
------------------
When ``as_of`` is provided, relations are filtered with:

    (valid_from IS NULL OR valid_from <= as_of)
    AND (valid_to IS NULL OR valid_to >= as_of)

This matches the same open-bound semantics used for proposition filtering
(see ``search.py`` and migration v11/v12).

Hard safety limits
------------------
``max_hops`` and ``max_nodes`` are enforced unconditionally.  A traversal
that would exceed either limit is pruned, with the counts tracked in
``GraphDiagnostics`` for the caller to inspect.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, NotRequired, TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed payloads
# ---------------------------------------------------------------------------


class GraphNode(TypedDict):
    """An entity node returned by graph operations."""

    id: str
    entity_type: str
    name: str
    confidence: float


class GraphEdge(TypedDict):
    """A relation edge returned by graph operations."""

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float
    evidence_memory_id: str | None
    evidence_memory_ids: list[str]
    curated: bool
    valid_from: str | None
    valid_to: str | None


class GraphPath(TypedDict):
    """A single path from source to target (graph_path op)."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    hop_count: int


class GraphDiagnostics(TypedDict):
    """Execution diagnostics returned with every graph result."""

    expanded_nodes: int
    pruned_edges: int
    latency_ms: float
    # Optional operation-specific diagnostics.
    entity_count: NotRequired[int]
    relation_count: NotRequired[int]
    distinct_relation_types: NotRequired[int]
    out_degree: NotRequired[int]
    in_degree: NotRequired[int]
    total_degree: NotRequired[int]


class GraphResult(TypedDict):
    """Envelope returned by all graph operations."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    paths: list[GraphPath]
    traversal_count: int
    diagnostics: GraphDiagnostics


# ---------------------------------------------------------------------------
# Internal SQL helpers
# ---------------------------------------------------------------------------

_ENTITY_SELECT = """
    SELECT id, entity_type, name, confidence
    FROM knowledge_entities
    WHERE id = $1::uuid
"""

_ENTITY_BY_NAME_SELECT = """
    SELECT id, entity_type, name, confidence
    FROM knowledge_entities
    WHERE name = $1
    LIMIT 1
"""

_NEIGHBORS_SQL_TEMPLATE = """
    SELECT
        kr.id,
        kr.source_entity_id,
        kr.target_entity_id,
        kr.relation_type,
        kr.confidence,
        kr.evidence_memory_id,
        kr.evidence_memory_ids,
        kr.curated,
        kr.valid_from,
        kr.valid_to
    FROM knowledge_relations kr
    WHERE ({direction_filter})
      {relation_type_clause}
      {confidence_clause}
      {temporal_clause}
"""


def _build_neighbors_query(
    entity_id: str,
    *,
    relation_types: list[str] | None,
    min_edge_confidence: float | None,
    as_of: datetime | None,
) -> tuple[str, list[Any]]:
    """Return (sql, params) for fetching all edges touching ``entity_id``.

    Fetches both outgoing (source = entity_id) and incoming (target = entity_id)
    edges in a single query so callers can derive undirected neighbourhoods.
    """
    params: list[Any] = []
    idx = 0

    def _p(v: Any) -> str:
        nonlocal idx
        idx += 1
        params.append(v)
        return f"${idx}"

    # Direction filter: outgoing OR incoming
    eid_param = _p(entity_id)
    direction_filter = (
        f"kr.source_entity_id = {eid_param}::uuid OR kr.target_entity_id = {eid_param}::uuid"
    )

    # Relation type filter
    relation_type_clause = ""
    if relation_types:
        rt_param = _p(relation_types)
        relation_type_clause = f"AND kr.relation_type = ANY({rt_param}::text[])"

    # Confidence filter
    confidence_clause = ""
    if min_edge_confidence is not None:
        conf_param = _p(min_edge_confidence)
        confidence_clause = f"AND kr.confidence >= {conf_param}"

    # Temporal filter
    temporal_clause = ""
    if as_of is not None:
        as_of_param = _p(as_of)
        temporal_clause = (
            f"AND (kr.valid_from IS NULL OR kr.valid_from <= {as_of_param}::timestamptz)\n"
            f"      AND (kr.valid_to IS NULL OR kr.valid_to >= {as_of_param}::timestamptz)"
        )

    sql = _NEIGHBORS_SQL_TEMPLATE.format(
        direction_filter=direction_filter,
        relation_type_clause=relation_type_clause,
        confidence_clause=confidence_clause,
        temporal_clause=temporal_clause,
    )
    return sql, params


async def _fetch_entity(entity_id: str, backend: Any) -> GraphNode | None:
    """Fetch a single entity by UUID.  Returns None if not found."""
    result = await backend.query(_ENTITY_SELECT, (entity_id,))
    if not result.rows:
        return None
    row = result.rows[0]
    return GraphNode(
        id=str(row["id"]),
        entity_type=row["entity_type"],
        name=row["name"],
        confidence=float(row["confidence"] or 1.0),
    )


async def _resolve_entity_id(entity_ref: str, backend: Any) -> str | None:
    """Resolve entity_ref to a UUID string.

    Tries UUID parse first; falls back to name lookup.
    Returns None when not found.
    """
    import uuid as _uuid

    # Attempt UUID parse
    try:
        _uuid.UUID(entity_ref)
        # Verify existence
        result = await backend.query(_ENTITY_SELECT, (entity_ref,))
        return str(result.rows[0]["id"]) if result.rows else None
    except (ValueError, TypeError, AttributeError):
        pass

    # Name-based lookup
    result = await backend.query(_ENTITY_BY_NAME_SELECT, (entity_ref,))
    return str(result.rows[0]["id"]) if result.rows else None


def _row_to_edge(row: dict[str, Any]) -> GraphEdge:
    vf = row.get("valid_from")
    vt = row.get("valid_to")
    return GraphEdge(
        id=str(row["id"]),
        source_entity_id=str(row["source_entity_id"]),
        target_entity_id=str(row["target_entity_id"]),
        relation_type=row["relation_type"],
        confidence=float(row["confidence"] or 1.0),
        evidence_memory_id=(
            str(row["evidence_memory_id"]) if row.get("evidence_memory_id") else None
        ),
        evidence_memory_ids=[
            str(item) for item in (row.get("evidence_memory_ids") or []) if item is not None
        ],
        curated=bool(row.get("curated") or False),
        valid_from=(
            vf.isoformat()
            if vf is not None and hasattr(vf, "isoformat")
            else (str(vf) if vf else None)
        ),
        valid_to=(
            vt.isoformat()
            if vt is not None and hasattr(vt, "isoformat")
            else (str(vt) if vt else None)
        ),
    )


def _empty_result(start_ms: float) -> GraphResult:
    return GraphResult(
        nodes=[],
        edges=[],
        paths=[],
        traversal_count=0,
        diagnostics=GraphDiagnostics(
            expanded_nodes=0,
            pruned_edges=0,
            latency_ms=(time.monotonic() - start_ms) * 1000,
        ),
    )


# ---------------------------------------------------------------------------
# Public graph operations
# ---------------------------------------------------------------------------


async def graph_neighbors(
    entity_ref: str,
    backend: Any,
    *,
    relation_types: list[str] | None = None,
    max_nodes: int = 100,
    min_edge_confidence: float | None = None,
    as_of: datetime | None = None,
) -> GraphResult:
    """Return all direct (1-hop) neighbors of ``entity_ref``.

    Returns every entity connected by a single edge, along with the
    connecting edges.  ``entity_ref`` may be a UUID or an entity name.

    Unknown entities produce an empty result (not an error).
    """
    start_ms = time.monotonic()

    entity_id = await _resolve_entity_id(entity_ref, backend)
    if entity_id is None:
        return _empty_result(start_ms)

    root = await _fetch_entity(entity_id, backend)
    if root is None:
        return _empty_result(start_ms)

    sql, params = _build_neighbors_query(
        entity_id,
        relation_types=relation_types,
        min_edge_confidence=min_edge_confidence,
        as_of=as_of,
    )
    result = await backend.query(sql, tuple(params))
    all_edges: list[GraphEdge] = [_row_to_edge(dict(r)) for r in result.rows]

    # Collect neighbor entity IDs
    neighbor_ids: set[str] = set()
    pruned = 0
    included_edges: list[GraphEdge] = []
    for edge in all_edges:
        neighbor_id = (
            edge["target_entity_id"]
            if edge["source_entity_id"] == entity_id
            else edge["source_entity_id"]
        )
        if len(neighbor_ids) >= max_nodes:
            pruned += 1
            continue
        neighbor_ids.add(neighbor_id)
        included_edges.append(edge)

    # Fetch neighbor entity rows
    nodes: list[GraphNode] = [root]
    seen_ids = {entity_id}
    for nid in neighbor_ids:
        if nid not in seen_ids:
            node = await _fetch_entity(nid, backend)
            if node:
                nodes.append(node)
            seen_ids.add(nid)

    return GraphResult(
        nodes=nodes,
        edges=included_edges,
        paths=[],
        traversal_count=len(neighbor_ids),
        diagnostics=GraphDiagnostics(
            expanded_nodes=1,
            pruned_edges=pruned,
            latency_ms=(time.monotonic() - start_ms) * 1000,
        ),
    )


async def graph_traverse(
    start_entity_ref: str,
    backend: Any,
    *,
    relation_types: list[str] | None = None,
    max_hops: int = 3,
    max_nodes: int = 100,
    min_edge_confidence: float | None = None,
    as_of: datetime | None = None,
) -> GraphResult:
    """BFS subgraph traversal from ``start_entity_ref``.

    Expands the reachable subgraph up to ``max_hops`` hops from the start
    entity, stopping early when ``max_nodes`` is reached.

    Edge direction is treated as undirected for traversal (both in-edges
    and out-edges are followed).  Cycles are handled by tracking visited
    entity IDs.

    Unknown start entities produce an empty result (not an error).
    """
    start_ms = time.monotonic()

    entity_id = await _resolve_entity_id(start_entity_ref, backend)
    if entity_id is None:
        return _empty_result(start_ms)

    root = await _fetch_entity(entity_id, backend)
    if root is None:
        return _empty_result(start_ms)

    visited: set[str] = {entity_id}
    nodes: list[GraphNode] = [root]
    edges: list[GraphEdge] = []
    seen_edge_ids: set[str] = set()

    queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
    expanded = 0
    pruned = 0

    while queue:
        current_id, hop = queue.popleft()
        if hop >= max_hops:
            continue

        expanded += 1
        sql, params = _build_neighbors_query(
            current_id,
            relation_types=relation_types,
            min_edge_confidence=min_edge_confidence,
            as_of=as_of,
        )
        result = await backend.query(sql, tuple(params))

        for row in result.rows:
            edge = _row_to_edge(dict(row))
            neighbor_id = (
                edge["target_entity_id"]
                if edge["source_entity_id"] == current_id
                else edge["source_entity_id"]
            )

            # Add edge (dedup by edge id)
            if edge["id"] not in seen_edge_ids:
                seen_edge_ids.add(edge["id"])
                edges.append(edge)

            if neighbor_id in visited:
                continue

            # Hard node cap
            if len(nodes) >= max_nodes:
                pruned += 1
                continue

            node = await _fetch_entity(neighbor_id, backend)
            if node:
                nodes.append(node)
            visited.add(neighbor_id)
            queue.append((neighbor_id, hop + 1))

    return GraphResult(
        nodes=nodes,
        edges=edges,
        paths=[],
        traversal_count=len(visited),
        diagnostics=GraphDiagnostics(
            expanded_nodes=expanded,
            pruned_edges=pruned,
            latency_ms=(time.monotonic() - start_ms) * 1000,
        ),
    )


async def graph_path(
    start_entity_ref: str,
    end_entity_ref: str,
    backend: Any,
    *,
    relation_types: list[str] | None = None,
    max_hops: int = 6,
    max_nodes: int = 200,
    min_edge_confidence: float | None = None,
    as_of: datetime | None = None,
) -> GraphResult:
    """Find the shortest path(s) between two entities using BFS.

    Returns the first shortest path found (unweighted hop count).  If the
    entities are the same, a single degenerate path with zero edges is
    returned.

    Unknown start or end entities produce an empty result (not an error).
    """
    start_ms = time.monotonic()

    start_id = await _resolve_entity_id(start_entity_ref, backend)
    if start_id is None:
        return _empty_result(start_ms)

    end_id = await _resolve_entity_id(end_entity_ref, backend)
    if end_id is None:
        return _empty_result(start_ms)

    # Trivial: same entity
    if start_id == end_id:
        node = await _fetch_entity(start_id, backend)
        nodes = [node] if node else []
        return GraphResult(
            nodes=nodes,
            edges=[],
            paths=[GraphPath(nodes=nodes, edges=[], hop_count=0)] if nodes else [],
            traversal_count=1 if nodes else 0,
            diagnostics=GraphDiagnostics(
                expanded_nodes=1,
                pruned_edges=0,
                latency_ms=(time.monotonic() - start_ms) * 1000,
            ),
        )

    # BFS: track predecessor for path reconstruction
    # State: (entity_id, incoming_edge | None, parent_entity_id | None)
    visited: set[str] = {start_id}
    # predecessor[id] = (parent_id, edge_that_led_here)
    predecessor: dict[str, tuple[str, GraphEdge] | None] = {start_id: None}
    queue: deque[tuple[str, int]] = deque([(start_id, 0)])
    expanded = 0
    pruned = 0
    found = False

    while queue and not found:
        current_id, hop = queue.popleft()
        if hop >= max_hops:
            continue

        expanded += 1
        sql, params = _build_neighbors_query(
            current_id,
            relation_types=relation_types,
            min_edge_confidence=min_edge_confidence,
            as_of=as_of,
        )
        result = await backend.query(sql, tuple(params))

        for row in result.rows:
            edge = _row_to_edge(dict(row))
            neighbor_id = (
                edge["target_entity_id"]
                if edge["source_entity_id"] == current_id
                else edge["source_entity_id"]
            )

            if neighbor_id in visited:
                continue

            if len(visited) >= max_nodes:
                pruned += 1
                continue

            visited.add(neighbor_id)
            predecessor[neighbor_id] = (current_id, edge)

            if neighbor_id == end_id:
                found = True
                break

            queue.append((neighbor_id, hop + 1))

    if not found:
        return GraphResult(
            nodes=[],
            edges=[],
            paths=[],
            traversal_count=len(visited),
            diagnostics=GraphDiagnostics(
                expanded_nodes=expanded,
                pruned_edges=pruned,
                latency_ms=(time.monotonic() - start_ms) * 1000,
            ),
        )

    # Reconstruct path from end to start
    path_edges: list[GraphEdge] = []
    path_entity_ids: list[str] = [end_id]
    current = end_id
    while predecessor[current] is not None:
        parent_id, edge = predecessor[current]  # type: ignore[misc]
        path_edges.insert(0, edge)
        path_entity_ids.insert(0, parent_id)
        current = parent_id

    # Fetch path nodes
    path_nodes: list[GraphNode] = []
    all_nodes: list[GraphNode] = []
    all_edges = path_edges[:]
    for eid in path_entity_ids:
        node = await _fetch_entity(eid, backend)
        if node:
            path_nodes.append(node)
            all_nodes.append(node)

    path = GraphPath(nodes=path_nodes, edges=path_edges, hop_count=len(path_edges))

    return GraphResult(
        nodes=all_nodes,
        edges=all_edges,
        paths=[path],
        traversal_count=len(visited),
        diagnostics=GraphDiagnostics(
            expanded_nodes=expanded,
            pruned_edges=pruned,
            latency_ms=(time.monotonic() - start_ms) * 1000,
        ),
    )


async def graph_stats(
    entity_ref: str | None,
    backend: Any,
    *,
    as_of: datetime | None = None,
) -> GraphResult:
    """Return degree and connectivity statistics for an entity.

    When ``entity_ref`` is provided, computes:
    - ``out_degree``: number of outgoing edges
    - ``in_degree``: number of incoming edges
    - ``total_degree``: sum of in + out
    - ``distinct_relation_types``: count of unique relation types on all edges

    The ``nodes`` list contains the queried entity; ``edges`` is empty.
    All counts are returned in ``diagnostics`` as additional keys.

    When ``entity_ref`` is omitted, returns global graph statistics:
    - ``entity_count``: total entities
    - ``relation_count``: total relations (time-filtered when ``as_of`` is set)
    - ``distinct_relation_types``: unique relation types (time-filtered)

    Unknown entities produce an empty result (not an error).
    """
    start_ms = time.monotonic()

    if not entity_ref:
        temporal_clause = ""
        params: list[Any] = []
        if as_of is not None:
            temporal_clause = (
                "WHERE (kr.valid_from IS NULL OR kr.valid_from <= $1::timestamptz)\n"
                "  AND (kr.valid_to IS NULL OR kr.valid_to >= $1::timestamptz)"
            )
            params.append(as_of)

        global_sql = f"""
            SELECT
                (SELECT COUNT(*)::bigint FROM knowledge_entities) AS entity_count,
                COUNT(*)::bigint AS relation_count,
                COUNT(DISTINCT kr.relation_type)::bigint AS distinct_relation_types
            FROM knowledge_relations kr
            {temporal_clause}
        """
        result = await backend.query(global_sql, tuple(params))
        row = result.rows[0] if result.rows else {}

        diag: GraphDiagnostics = {
            "expanded_nodes": 0,
            "pruned_edges": 0,
            "latency_ms": (time.monotonic() - start_ms) * 1000,
            "entity_count": int(row.get("entity_count") or 0),
            "relation_count": int(row.get("relation_count") or 0),
            "distinct_relation_types": int(row.get("distinct_relation_types") or 0),
        }

        return GraphResult(
            nodes=[],
            edges=[],
            paths=[],
            traversal_count=0,
            diagnostics=diag,
        )

    entity_id = await _resolve_entity_id(entity_ref, backend)
    if entity_id is None:
        return _empty_result(start_ms)

    node = await _fetch_entity(entity_id, backend)
    if node is None:
        return _empty_result(start_ms)

    # Temporal filter
    temporal_clause = ""
    temporal_params: list[Any] = [entity_id, entity_id]
    if as_of is not None:
        temporal_clause = (
            "AND (kr.valid_from IS NULL OR kr.valid_from <= $3::timestamptz)\n"
            "      AND (kr.valid_to IS NULL OR kr.valid_to >= $3::timestamptz)"
        )
        temporal_params.append(as_of)

    stats_sql = f"""
        SELECT
            COUNT(*) FILTER (WHERE kr.source_entity_id = $1::uuid) AS out_degree,
            COUNT(*) FILTER (WHERE kr.target_entity_id = $2::uuid) AS in_degree,
            COUNT(DISTINCT kr.relation_type) AS distinct_relation_types
        FROM knowledge_relations kr
        WHERE (kr.source_entity_id = $1::uuid OR kr.target_entity_id = $2::uuid)
          {temporal_clause}
    """
    result = await backend.query(stats_sql, tuple(temporal_params))
    row = result.rows[0] if result.rows else {}

    out_degree = int(row.get("out_degree") or 0)
    in_degree = int(row.get("in_degree") or 0)
    total_degree = out_degree + in_degree
    distinct_relation_types = int(row.get("distinct_relation_types") or 0)

    # Encode stats as extra diagnostics fields beyond the standard three.
    # We use a plain dict here (TypedDict only enforces at type-check time).
    stats_diag: GraphDiagnostics = {
        "expanded_nodes": 1,
        "pruned_edges": 0,
        "latency_ms": (time.monotonic() - start_ms) * 1000,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "total_degree": total_degree,
        "distinct_relation_types": distinct_relation_types,
    }

    return GraphResult(
        nodes=[node],
        edges=[],
        paths=[],
        traversal_count=1,
        diagnostics=stats_diag,
    )
