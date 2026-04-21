#!/usr/bin/env python3
"""Practical benchmark for large multi-service memory workloads.

This benchmark targets scoped memory workloads with high-cardinality topology:

- wing (service/project)
- room (component)
- hall (topic lane)

Workload mix:

1) ingest (writes)
2) scoped search query (room_scoped_search)
3) scoped context query (search + assemble_context)

The benchmark is non-destructive by default:

- writes are isolated under a unique benchmark source name
- source rows are cleaned up after the run unless --keep-data is provided
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from workflows_bench_common import (
    FastEmbedEncoder,
    add_db_args,
    connect_knowledge_backend,
    db_config_from_args,
    purge_benchmark_source,
)

from workflows_mcp.engine.knowledge.constants import Authority, LifecycleState
from workflows_mcp.engine.knowledge.context import assemble_context
from workflows_mcp.engine.knowledge.search import room_scoped_search

SYSTEM_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")


@dataclass(frozen=True)
class ScopeKey:
    wing: str
    room: str
    hall: str


@dataclass(frozen=True)
class OperationMix:
    ingest: float
    scoped_query: float
    context_query: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _build_scopes(args: argparse.Namespace) -> list[ScopeKey]:
    scopes: list[ScopeKey] = []
    for wing_idx in range(args.wings):
        wing = f"wing-{wing_idx:02d}"
        for room_idx in range(args.rooms_per_wing):
            room = f"room-{room_idx:02d}"
            for hall_idx in range(args.halls_per_room):
                hall = f"hall-{hall_idx:02d}"
                scopes.append(ScopeKey(wing=wing, room=room, hall=hall))
    return scopes


def _build_initial_records(
    scopes: list[ScopeKey],
    *,
    records_per_hall: int,
) -> list[tuple[ScopeKey, str, str]]:
    records: list[tuple[ScopeKey, str, str]] = []
    for scope in scopes:
        for idx in range(records_per_hall):
            content = (
                f"Service memo {idx} for {scope.wing}/{scope.room}/{scope.hall}. "
                f"Deployment notes, incident learnings, and ownership details for "
                f"{scope.room} in {scope.wing}."
            )
            path = f"{scope.wing}/{scope.room}/{scope.hall}/seed-{idx:03d}.md"
            records.append((scope, content, path))
    return records


def _build_operation_sequence(
    total_ops: int,
    *,
    mix: OperationMix,
    rng: random.Random,
) -> list[str]:
    sequence: list[str] = []
    thresholds = [mix.ingest, mix.ingest + mix.scoped_query]
    for _ in range(total_ops):
        pick = rng.random()
        if pick < thresholds[0]:
            sequence.append("ingest")
        elif pick < thresholds[1]:
            sequence.append("scoped_query")
        else:
            sequence.append("context_query")
    return sequence


def _validate_mix(mix: OperationMix) -> None:
    total = mix.ingest + mix.scoped_query + mix.context_query
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Operation mix must sum to 1.0")
    if mix.ingest < 0 or mix.scoped_query < 0 or mix.context_query < 0:
        raise ValueError("Operation mix percentages must be non-negative")


async def _upsert_source(backend: Any, source_name: str, source_type: str) -> str:
    result = await backend.query(
        """
        INSERT INTO knowledge_sources (id, name, source_type, category_ids)
        VALUES ($1::uuid, $2, $3, '{}'::uuid[])
        ON CONFLICT (name) DO UPDATE
            SET source_type = EXCLUDED.source_type,
                updated_at = NOW()
        RETURNING id
        """,
        (str(uuid.uuid4()), source_name, source_type),
    )
    if not result.rows:
        raise RuntimeError(f"Failed to upsert source: {source_name}")
    return str(result.rows[0]["id"])


async def _ingest_records(
    backend: Any,
    *,
    source_id: str,
    source_name: str,
    source_type: str,
    embedding_model: str,
    records: list[tuple[ScopeKey, str, str]],
    embeddings: list[list[float]],
    confidence: float,
) -> int:
    if not records:
        return 0
    if len(records) != len(embeddings):
        raise ValueError("records and embeddings must have equal length")

    item_rows: list[tuple[str, str, str, str]] = []
    memory_rows: list[tuple[Any, ...]] = []

    for idx, record_with_embedding in enumerate(zip(records, embeddings, strict=True)):
        (scope, content, path), embedding = record_with_embedding
        item_id = str(uuid.uuid4())
        memory_id = str(uuid.uuid4())

        item_rows.append((item_id, source_id, path, f"{scope.hall}-{idx:06d}"))
        memory_rows.append(
            (
                memory_id,
                item_id,
                content,
                str(embedding),
                Authority.EXTRACTED,
                LifecycleState.ACTIVE,
                confidence,
                embedding_model,
                "{}",
                str(SYSTEM_USER_UUID),
                "BENCHMARK",
                source_name,
                source_type,
                scope.wing,
                scope.room,
                scope.hall,
            )
        )

    await backend.execute_many(
        """
        INSERT INTO knowledge_items (id, source_id, path, title)
        VALUES ($1::uuid, $2::uuid, $3, $4)
        """,
        item_rows,
    )

    await backend.execute_many(
        """
        INSERT INTO knowledge_memories
            (id, item_id, content, embedding, search_vector,
             authority, lifecycle_state, confidence,
             embedding_model, metadata,
             created_by, auth_method, source_name, source_type,
             namespace, room, corridor)
        VALUES
            ($1::uuid, $2::uuid, $3, $4::vector,
             to_tsvector('english', $3),
             $5, $6, $7,
             $8, $9::jsonb,
             $10::uuid, $11, $12, $13,
             $14, $15, $16)
        """,
        memory_rows,
    )
    return len(records)


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": float(len(latencies_ms)),
        "avg_ms": sum(latencies_ms) / len(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "max_ms": max(latencies_ms),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_db_args(parser)

    parser.add_argument("--wings", type=int, default=8)
    parser.add_argument("--rooms-per-wing", type=int, default=8)
    parser.add_argument("--halls-per-room", type=int, default=4)
    parser.add_argument("--records-per-hall", type=int, default=3)

    parser.add_argument("--operations", type=int, default=300)
    parser.add_argument("--ingest-ratio", type=float, default=0.20)
    parser.add_argument("--scoped-query-ratio", type=float, default=0.55)
    parser.add_argument("--context-query-ratio", type=float, default=0.25)

    parser.add_argument("--search-limit", type=int, default=12)
    parser.add_argument("--context-token-budget", type=int, default=420)
    parser.add_argument("--embedding-confidence", type=float, default=0.85)

    parser.add_argument("--embed-model", default="default")
    parser.add_argument("--source-prefix", default="benchmark:multiservice")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    mix = OperationMix(
        ingest=args.ingest_ratio,
        scoped_query=args.scoped_query_ratio,
        context_query=args.context_query_ratio,
    )
    _validate_mix(mix)

    scopes = _build_scopes(args)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = f"{args.source_prefix}:{run_id}"

    initial_records = _build_initial_records(
        scopes,
        records_per_hall=args.records_per_hall,
    )
    workload_size = len(initial_records)

    rng = random.Random(args.seed)
    sequence = _build_operation_sequence(args.operations, mix=mix, rng=rng)

    dry_payload: dict[str, Any] = {
        "dry_run": True,
        "source_name": source_name,
        "scopes": len(scopes),
        "initial_records": workload_size,
        "operations": len(sequence),
        "mix": {
            "ingest": args.ingest_ratio,
            "scoped_query": args.scoped_query_ratio,
            "context_query": args.context_query_ratio,
        },
    }
    if args.dry_run:
        return dry_payload

    encoder = FastEmbedEncoder(model_name=args.embed_model)
    db_config = db_config_from_args(args)
    backend = await connect_knowledge_backend(db_config)

    latencies: dict[str, list[float]] = {
        "ingest": [],
        "scoped_query": [],
        "context_query": [],
    }
    query_result_counts: list[int] = []
    context_tokens: list[int] = []
    context_memory_counts: list[int] = []
    inserted_records = 0

    started = time.perf_counter()
    try:
        source_id = await _upsert_source(backend, source_name, "BENCHMARK_MULTISERVICE")

        initial_texts = [content for _, content, _ in initial_records]
        initial_embeddings = encoder.encode(initial_texts)

        t0 = time.perf_counter()
        inserted_records += await _ingest_records(
            backend,
            source_id=source_id,
            source_name=source_name,
            source_type="BENCHMARK_MULTISERVICE",
            embedding_model=encoder.model_name,
            records=initial_records,
            embeddings=initial_embeddings,
            confidence=args.embedding_confidence,
        )
        latencies["ingest"].append((time.perf_counter() - t0) * 1000)

        query_text_by_scope: dict[ScopeKey, str] = {}
        query_emb_by_scope: dict[ScopeKey, list[float]] = {}
        scope_queries = [
            (
                scope,
                (
                    f"What are the deployment and incident notes for "
                    f"{scope.wing} {scope.room} {scope.hall}?"
                ),
            )
            for scope in scopes
        ]
        scope_query_embeddings = encoder.encode([query for _, query in scope_queries])
        for scope_query_pair in zip(scope_queries, scope_query_embeddings, strict=True):
            (scope, query_text), embedding = scope_query_pair
            query_text_by_scope[scope] = query_text
            query_emb_by_scope[scope] = embedding

        dynamic_ingest_index = 0
        for op in sequence:
            scope = rng.choice(scopes)

            if op == "ingest":
                dynamic_ingest_index += 1
                content = (
                    "Rolling update "
                    f"{dynamic_ingest_index} for {scope.wing}/{scope.room}/{scope.hall}. "
                    "Observed latency spike mitigation and rollback checklist."
                )
                path = (
                    f"{scope.wing}/{scope.room}/{scope.hall}/"
                    f"rolling-{dynamic_ingest_index:06d}.md"
                )
                record = [(scope, content, path)]
                emb = encoder.encode([content])

                t_op = time.perf_counter()
                inserted_records += await _ingest_records(
                    backend,
                    source_id=source_id,
                    source_name=source_name,
                    source_type="BENCHMARK_MULTISERVICE",
                    embedding_model=encoder.model_name,
                    records=record,
                    embeddings=emb,
                    confidence=args.embedding_confidence,
                )
                latencies["ingest"].append((time.perf_counter() - t_op) * 1000)
                continue

            query_text = query_text_by_scope[scope]
            query_embedding = query_emb_by_scope[scope]

            if op == "scoped_query":
                t_op = time.perf_counter()
                rows = await room_scoped_search(
                    query_embedding,
                    query_text,
                    backend,
                    namespace=scope.wing,
                    room=scope.room,
                    corridor=scope.hall,
                    source=source_name,
                    min_confidence=0.0,
                    lifecycle_state=LifecycleState.ACTIVE,
                    limit=args.search_limit,
                )
                latencies["scoped_query"].append((time.perf_counter() - t_op) * 1000)
                query_result_counts.append(len(rows))
                continue

            # context_query
            t_op = time.perf_counter()
            rows = await room_scoped_search(
                query_embedding,
                query_text,
                backend,
                namespace=scope.wing,
                room=scope.room,
                corridor=scope.hall,
                source=source_name,
                min_confidence=0.0,
                lifecycle_state=LifecycleState.ACTIVE,
                limit=max(args.search_limit * 3, args.search_limit),
            )
            _, included_count, tokens_used = assemble_context(
                rows,
                max_tokens=args.context_token_budget,
                diversity=True,
                query_embedding=query_embedding,
            )
            latencies["context_query"].append((time.perf_counter() - t_op) * 1000)
            context_tokens.append(tokens_used)
            context_memory_counts.append(included_count)

    finally:
        if not args.keep_data:
            await purge_benchmark_source(backend, source_name)
        await backend.disconnect()

    total_elapsed = time.perf_counter() - started
    total_ops = sum(len(values) for values in latencies.values())
    total_ops_per_second = (total_ops / total_elapsed) if total_elapsed > 0 else 0.0

    operation_summary: dict[str, Any] = {}
    for op_name, values in latencies.items():
        op_count = len(values)
        op_elapsed_seconds = sum(values) / 1000.0
        operation_summary[op_name] = {
            **_latency_summary(values),
            "ops_per_second": (op_count / op_elapsed_seconds) if op_elapsed_seconds > 0 else 0.0,
        }

    summary: dict[str, Any] = {
        "dry_run": False,
        "source_name": source_name,
        "embedding_model": encoder.model_name,
        "scopes": len(scopes),
        "initial_records": workload_size,
        "records_inserted_total": inserted_records,
        "operations_executed": total_ops,
        "elapsed_seconds": total_elapsed,
        "throughput_ops_per_second": total_ops_per_second,
        "operation_mix": operation_summary,
        "quality_signals": {
            "scoped_query_avg_results": (
                sum(query_result_counts) / len(query_result_counts) if query_result_counts else 0.0
            ),
            "scoped_query_empty_ratio": (
                (sum(1 for count in query_result_counts if count == 0) / len(query_result_counts))
                if query_result_counts
                else 0.0
            ),
            "context_avg_tokens": (
                sum(context_tokens) / len(context_tokens) if context_tokens else 0.0
            ),
            "context_p95_tokens": _percentile(context_tokens, 95),
            "context_avg_memories": (
                sum(context_memory_counts) / len(context_memory_counts)
                if context_memory_counts
                else 0.0
            ),
        },
    }
    return summary


async def _async_main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    summary = await run_benchmark(args)

    print(json.dumps(summary, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote benchmark report: {args.output}")

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
