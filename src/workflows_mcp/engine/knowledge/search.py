"""Hybrid search for Knowledge executor.

Builds parameterized SQL queries for pgvector cosine similarity and
PostgreSQL full-text search, with RRF rank fusion. All queries use
asyncpg positional params ($1, $2, ...).

Room-scoped retrieval
---------------------
When ``namespace`` and/or ``room`` are provided, the search pipeline runs two
lanes in parallel:

1. **Room-scoped lane** — restricts candidates to propositions matching the
   supplied namespace/room before running the full vector + FTS + RRF
   pipeline. Improves precision for topic-scoped queries.

2. **Global companion lane** — runs a fixed-size global retrieval (no room
   filter) in parallel to preserve recall for cross-room knowledge.

Results from both lanes are fused via a second RRF pass so the final ranked
list benefits from both precision and global recall.  The companion lane is
always active; it is never a fallback.  If benchmarks show it degrades
quality, it is removed and room-only retrieval becomes the single
implementation (no runtime toggle).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from .constants import (
    DEFAULT_FTS_WEIGHT,
    DEFAULT_LIMIT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_RRF_K,
    DEFAULT_VECTOR_WEIGHT,
    LifecycleState,
)

logger = logging.getLogger(__name__)

# Fixed candidate cap for the global companion lane so it never dominates
# the result set at the expense of the room-scoped lane.
_COMPANION_LANE_CANDIDATES = 20


def _append_temporal_filters(
    where_clauses: list[str],
    *,
    next_param: Any,
    as_of: datetime | None,
    from_dt: datetime | None,
    to_dt: datetime | None,
) -> None:
    """Append temporal filters with explicit point-in-time vs interval semantics."""
    if as_of is not None and (from_dt is not None or to_dt is not None):
        raise ValueError("as_of cannot be combined with from/to interval")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise ValueError("from must be less than or equal to to")

    if as_of is not None:
        as_of_param = next_param(as_of)
        where_clauses.append(f"(kp.valid_from IS NULL OR kp.valid_from <= {as_of_param}::timestamptz)")
        where_clauses.append(f"(kp.valid_to IS NULL OR kp.valid_to >= {as_of_param}::timestamptz)")
        return

    if from_dt is not None:
        from_param = next_param(from_dt)
        where_clauses.append(f"(kp.valid_to IS NULL OR kp.valid_to >= {from_param}::timestamptz)")
    if to_dt is not None:
        to_param = next_param(to_dt)
        where_clauses.append(f"(kp.valid_from IS NULL OR kp.valid_from <= {to_param}::timestamptz)")


def build_vector_search_query(
    query_embedding: list[float],
    *,
    source: str | None = None,
    categories: list[str] | None = None,
    as_of: datetime | None = None,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    lifecycle_state: str = LifecycleState.ACTIVE,
    limit: int = DEFAULT_LIMIT,
    include_embeddings: bool = False,
    namespace: str | None = None,
    room: str | None = None,
    corridor: str | None = None,
    community_ids: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build a pgvector cosine similarity search query.

    Returns (sql, params) with asyncpg $N positional params.
    Includes JOINs through knowledge_items → knowledge_sources for
    source/category filtering.

    Args:
        include_embeddings: If True, include the raw embedding vector in SELECT.
            Used internally by _op_context for MMR reranking. Never expose
            embedding values in public API responses.
        namespace: When provided, restrict candidates to propositions in this namespace.
        room: When provided, restrict candidates to propositions in this room.
        corridor: When provided, restrict candidates to propositions in this corridor.
        community_ids: When provided, restrict candidates to these community ids.
    """
    params: list[Any] = []
    param_idx = 0

    def next_param(value: Any) -> str:
        nonlocal param_idx
        param_idx += 1
        params.append(value)
        return f"${param_idx}"

    # Core query: cosine distance via <=> operator
    embedding_param = next_param(str(query_embedding))
    state_param = next_param(lifecycle_state)
    confidence_param = next_param(min_confidence)

    # Candidate limit for pre-filtering (wider net for re-ranking)
    candidate_limit = min(limit * 10, 200)
    candidate_param = next_param(candidate_limit)

    post_where_clauses = [f"kp.confidence >= {confidence_param}"]

    # Source filter (exact match or prefix with *)
    # Uses the denormalized source_name column for consistent, performant queries
    if source:
        if source.endswith("*"):
            prefix_param = next_param(source[:-1] + "%")
            post_where_clauses.append(f"kp.source_name LIKE {prefix_param}")
        else:
            source_param = next_param(source)
            post_where_clauses.append(f"kp.source_name = {source_param}")

    # Category filter — EXISTS subquery on junction table.
    # Works for all proposition types including agent observations (item_id IS NULL).
    # No JOIN needed; no duplicate rows when a proposition matches multiple categories.
    if categories:
        cat_param = next_param(categories)
        post_where_clauses.append(
            f"EXISTS ("
            f"  SELECT 1 FROM knowledge_memory_categories kpc"
            f"  WHERE kpc.memory_id = kp.id"
            f"    AND kpc.category_id = ANY({cat_param}::uuid[])"
            f")"
        )

    _append_temporal_filters(
        post_where_clauses,
        next_param=next_param,
        as_of=as_of,
        from_dt=from_dt,
        to_dt=to_dt,
    )

    # Room-scope filters
    if namespace is not None:
        post_where_clauses.append(f"kp.namespace = {next_param(namespace)}")
    if room is not None:
        post_where_clauses.append(f"kp.room = {next_param(room)}")
    if corridor is not None:
        post_where_clauses.append(f"kp.corridor = {next_param(corridor)}")
    if community_ids:
        post_where_clauses.append(f"kp.community_id = ANY({next_param(community_ids)}::uuid[])")

    lifecycle_clause = f"kp.lifecycle_state = {state_param}"
    post_where_clause = " AND ".join(post_where_clauses) if post_where_clauses else "TRUE"

    embedding_col = ", kp.embedding" if include_embeddings else ""

    sql = f"""
        WITH lifecycle_filtered AS (
            SELECT kp.*
            FROM knowledge_memories kp
            WHERE {lifecycle_clause}
        )
        SELECT kp.id, kp.content, kp.confidence, kp.authority,
               kp.retrieval_count, kp.source_name, kp.namespace,
               1 - (kp.embedding <=> {embedding_param}::vector) AS similarity,
               ki_path.path AS item_path{embedding_col}
        FROM lifecycle_filtered kp
        LEFT JOIN knowledge_items ki_path ON kp.item_id = ki_path.id
        WHERE {post_where_clause}
          AND kp.embedding IS NOT NULL
        ORDER BY kp.embedding <=> {embedding_param}::vector
        LIMIT {candidate_param}
    """

    return sql, params


def build_fts_search_query(
    query_text: str,
    *,
    source: str | None = None,
    categories: list[str] | None = None,
    as_of: datetime | None = None,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    lifecycle_state: str = LifecycleState.ACTIVE,
    limit: int = DEFAULT_LIMIT,
    namespace: str | None = None,
    room: str | None = None,
    corridor: str | None = None,
    community_ids: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build a PostgreSQL full-text search query.

    Returns (sql, params) with asyncpg $N positional params.
    Uses plainto_tsquery for natural-language query parsing.

    Args:
        namespace: When provided, restrict candidates to propositions in this namespace.
        room: When provided, restrict candidates to propositions in this room.
        corridor: When provided, restrict candidates to propositions in this corridor.
        community_ids: When provided, restrict candidates to these community ids.
    """
    params: list[Any] = []
    param_idx = 0

    def next_param(value: Any) -> str:
        nonlocal param_idx
        param_idx += 1
        params.append(value)
        return f"${param_idx}"

    query_param = next_param(query_text)
    state_param = next_param(lifecycle_state)
    confidence_param = next_param(min_confidence)
    candidate_limit = min(limit * 10, 200)
    candidate_param = next_param(candidate_limit)

    post_where_clauses = [
        f"kp.search_vector @@ plainto_tsquery('english', {query_param})",
        f"kp.confidence >= {confidence_param}",
    ]

    # Source filter (exact match or prefix with *)
    # Uses the denormalized source_name column for consistent, performant queries
    if source:
        if source.endswith("*"):
            prefix_param = next_param(source[:-1] + "%")
            post_where_clauses.append(f"kp.source_name LIKE {prefix_param}")
        else:
            source_param = next_param(source)
            post_where_clauses.append(f"kp.source_name = {source_param}")

    # Category filter — EXISTS subquery on junction table.
    # Works for all proposition types including agent observations (item_id IS NULL).
    # No JOIN needed; no duplicate rows when a proposition matches multiple categories.
    if categories:
        cat_param = next_param(categories)
        post_where_clauses.append(
            f"EXISTS ("
            f"  SELECT 1 FROM knowledge_memory_categories kpc"
            f"  WHERE kpc.memory_id = kp.id"
            f"    AND kpc.category_id = ANY({cat_param}::uuid[])"
            f")"
        )

    _append_temporal_filters(
        post_where_clauses,
        next_param=next_param,
        as_of=as_of,
        from_dt=from_dt,
        to_dt=to_dt,
    )

    # Room-scope filters
    if namespace is not None:
        post_where_clauses.append(f"kp.namespace = {next_param(namespace)}")
    if room is not None:
        post_where_clauses.append(f"kp.room = {next_param(room)}")
    if corridor is not None:
        post_where_clauses.append(f"kp.corridor = {next_param(corridor)}")
    if community_ids:
        post_where_clauses.append(f"kp.community_id = ANY({next_param(community_ids)}::uuid[])")

    lifecycle_clause = f"kp.lifecycle_state = {state_param}"
    post_where_clause = " AND ".join(post_where_clauses)

    sql = f"""
        WITH lifecycle_filtered AS (
            SELECT kp.*
            FROM knowledge_memories kp
            WHERE {lifecycle_clause}
        )
        SELECT kp.id, kp.content, kp.confidence, kp.authority,
               kp.retrieval_count, kp.source_name, kp.namespace,
               ts_rank(kp.search_vector, plainto_tsquery('english', {query_param})) AS fts_rank,
               ki_path.path AS item_path
        FROM lifecycle_filtered kp
        LEFT JOIN knowledge_items ki_path ON kp.item_id = ki_path.id
        WHERE {post_where_clause}
        ORDER BY fts_rank DESC
        LIMIT {candidate_param}
    """

    return sql, params


def rrf_fusion(
    vector_results: list[dict[str, Any]],
    fts_results: list[dict[str, Any]],
    *,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    fts_weight: float = DEFAULT_FTS_WEIGHT,
    k: int = DEFAULT_RRF_K,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of vector and FTS search results.

    score = Σ(weight_i / (k + rank_i))
    """
    scores: dict[str, float] = {}
    results_by_id: dict[str, dict[str, Any]] = {}

    # Score vector results
    for rank, row in enumerate(vector_results, start=1):
        row_id = str(row["id"])
        scores[row_id] = scores.get(row_id, 0.0) + vector_weight / (k + rank)
        results_by_id[row_id] = row

    # Score FTS results
    for rank, row in enumerate(fts_results, start=1):
        row_id = str(row["id"])
        scores[row_id] = scores.get(row_id, 0.0) + fts_weight / (k + rank)
        if row_id not in results_by_id:
            results_by_id[row_id] = row

    # Sort by fused score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for row_id in sorted_ids[:limit]:
        result = dict(results_by_id[row_id])
        result["rrf_score"] = scores[row_id]
        fused.append(result)

    return fused


async def room_scoped_search(
    query_embedding: list[float],
    query_text: str,
    backend: Any,
    *,
    namespace: str | None,
    room: str | None,
    corridor: str | None = None,
    source: str | None = None,
    categories: list[str] | None = None,
    as_of: datetime | None = None,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    lifecycle_state: str = LifecycleState.ACTIVE,
    limit: int = DEFAULT_LIMIT,
    include_embeddings: bool = False,
    community_ids: list[str] | None = None,
    include_global_companion: bool = True,
) -> list[dict[str, Any]]:
    """Run room-scoped and global companion lanes in parallel, then fuse.

    Both lanes run concurrently via asyncio.gather.  The room-scoped lane
    restricts candidates to the supplied namespace/room; the global companion
    lane runs a fixed-size global retrieval to preserve cross-room recall.

    Results are merged via a single RRF pass over all four candidate lists
    (room-vector, room-fts, global-vector, global-fts).

    When namespace, room, and corridor are all None this degrades to a
    standard global search (no scoped lane is issued; only the global lane
    runs). Set include_global_companion=False to disable the companion lane
    when strict scoped retrieval is required.
    """
    has_room_scope = namespace is not None or room is not None or corridor is not None
    run_global_lane = include_global_companion or not has_room_scope

    # ---- Room-scoped lane ------------------------------------------------
    async def _run_room_lane() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not has_room_scope:
            return [], []
        vec_sql, vec_params = build_vector_search_query(
            query_embedding,
            source=source,
            categories=categories,
            as_of=as_of,
            from_dt=from_dt,
            to_dt=to_dt,
            min_confidence=min_confidence,
            lifecycle_state=lifecycle_state,
            limit=limit,
            include_embeddings=include_embeddings,
            namespace=namespace,
            room=room,
            corridor=corridor,
            community_ids=community_ids,
        )
        fts_sql, fts_params = build_fts_search_query(
            query_text,
            source=source,
            categories=categories,
            as_of=as_of,
            from_dt=from_dt,
            to_dt=to_dt,
            min_confidence=min_confidence,
            lifecycle_state=lifecycle_state,
            limit=limit,
            namespace=namespace,
            room=room,
            corridor=corridor,
            community_ids=community_ids,
        )
        vec_res, fts_res = await asyncio.gather(
            backend.query(vec_sql, tuple(vec_params)),
            backend.query(fts_sql, tuple(fts_params)),
        )
        return [dict(r) for r in vec_res.rows], [dict(r) for r in fts_res.rows]

    # ---- Global companion lane -------------------------------------------
    async def _run_global_lane() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not run_global_lane:
            return [], []
        companion_limit = _COMPANION_LANE_CANDIDATES if has_room_scope else limit
        vec_sql, vec_params = build_vector_search_query(
            query_embedding,
            source=source,
            categories=categories,
            as_of=as_of,
            from_dt=from_dt,
            to_dt=to_dt,
            min_confidence=min_confidence,
            lifecycle_state=lifecycle_state,
            limit=companion_limit,
            include_embeddings=include_embeddings,
            community_ids=community_ids,
        )
        fts_sql, fts_params = build_fts_search_query(
            query_text,
            source=source,
            categories=categories,
            as_of=as_of,
            from_dt=from_dt,
            to_dt=to_dt,
            min_confidence=min_confidence,
            lifecycle_state=lifecycle_state,
            limit=companion_limit,
            community_ids=community_ids,
        )
        vec_res, fts_res = await asyncio.gather(
            backend.query(vec_sql, tuple(vec_params)),
            backend.query(fts_sql, tuple(fts_params)),
        )
        return [dict(r) for r in vec_res.rows], [dict(r) for r in fts_res.rows]

    # Run both lanes concurrently
    (room_vec, room_fts), (global_vec, global_fts) = await asyncio.gather(
        _run_room_lane(),
        _run_global_lane(),
    )

    if has_room_scope:
        # Two-pass RRF: first fuse within each lane, then fuse the two lane
        # winners together to produce the final ranked list.
        room_fused = rrf_fusion(room_vec, room_fts, limit=limit)
        if not run_global_lane:
            return room_fused
        global_fused = rrf_fusion(global_vec, global_fts, limit=_COMPANION_LANE_CANDIDATES)
        # Final fusion treats room_fused as "vector" lane and global_fused as "fts" lane
        # with equal weights so neither dominates.
        return rrf_fusion(room_fused, global_fused, limit=limit)
    else:
        return rrf_fusion(global_vec, global_fts, limit=limit)
