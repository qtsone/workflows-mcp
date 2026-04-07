#!/usr/bin/env python3
"""Workflows-MCP × LongMemEval baseline benchmark.

This runner uses the workflows-mcp knowledge retrieval stack:
- pgvector cosine search
- PostgreSQL full-text search
- Reciprocal Rank Fusion (RRF)

The dataset format follows the public LongMemEval release.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from workflows_bench_common import (
    DEFAULT_METRIC_KS,
    FastEmbedEncoder,
    add_db_args,
    connect_knowledge_backend,
    db_config_from_args,
    evaluate_retrieval,
    fused_rows_to_rankings,
    ingest_corpus,
    purge_benchmark_source,
    purge_benchmark_sources,
    run_hybrid_search,
    session_id_from_corpus_id,
    write_jsonl,
)


def build_corpus(
    entry: dict[str, Any],
    granularity: str,
) -> tuple[list[str], list[str], list[str]]:
    corpus: list[str] = []
    corpus_ids: list[str] = []
    corpus_timestamps: list[str] = []

    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    for session, session_id, date in zip(sessions, session_ids, dates, strict=True):
        if granularity == "session":
            user_turns = [turn["content"] for turn in session if turn.get("role") == "user"]
            if user_turns:
                corpus.append("\n".join(user_turns))
                corpus_ids.append(session_id)
                corpus_timestamps.append(date)
            continue

        turn_number = 0
        for turn in session:
            if turn.get("role") != "user":
                continue
            corpus.append(turn["content"])
            corpus_ids.append(f"{session_id}_turn_{turn_number}")
            corpus_timestamps.append(date)
            turn_number += 1

    return corpus, corpus_ids, corpus_timestamps


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


async def run_benchmark(args: argparse.Namespace) -> None:
    with open(args.data_file, encoding="utf-8") as handle:
        data = json.load(handle)

    if args.limit > 0:
        data = data[: args.limit]

    encoder = FastEmbedEncoder(model_name=args.embed_model)
    db_config = db_config_from_args(args)
    backend = await connect_knowledge_backend(db_config)

    if args.purge_prefix:
        await purge_benchmark_sources(backend, args.source_prefix)

    metrics_session: dict[str, list[float]] = {f"recall_any@{k}": [] for k in DEFAULT_METRIC_KS}
    metrics_session.update({f"recall_all@{k}": [] for k in DEFAULT_METRIC_KS})
    metrics_session.update({f"ndcg_any@{k}": [] for k in DEFAULT_METRIC_KS})

    metrics_turn: dict[str, list[float]] = {f"recall_any@{k}": [] for k in DEFAULT_METRIC_KS}
    metrics_turn.update({f"recall_all@{k}": [] for k in DEFAULT_METRIC_KS})
    metrics_turn.update({f"ndcg_any@{k}": [] for k in DEFAULT_METRIC_KS})

    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results_log: list[dict[str, Any]] = []

    started_at = time.perf_counter()

    print("=" * 60)
    print("Workflows-MCP x LongMemEval Benchmark")
    print("=" * 60)
    print(f"Data file:     {Path(args.data_file).name}")
    print(f"Questions:     {len(data)}")
    print(f"Granularity:   {args.granularity}")
    if encoder.requested_model_name == encoder.model_name:
        print(f"Embed model:   {encoder.model_name}")
    else:
        print(f"Embed model:   {encoder.requested_model_name} -> {encoder.model_name}")
    print(f"Search limit:  {args.search_limit}")
    print("-" * 60)

    try:
        for index, entry in enumerate(data, start=1):
            question_id = str(entry["question_id"])
            question_type = str(entry["question_type"])
            question_text = str(entry["question"])

            corpus, corpus_ids, corpus_timestamps = build_corpus(entry, args.granularity)
            if not corpus:
                print(f"[{index:4}/{len(data)}] {question_id:<32} SKIP (empty corpus)")
                continue

            source_name = f"{args.source_prefix}:{question_id}"

            rankings: list[int]
            try:
                corpus_embeddings = encoder.encode(corpus)
                await ingest_corpus(
                    backend,
                    source_name=source_name,
                    source_type="BENCHMARK_LONGMEMEVAL",
                    corpus_texts=corpus,
                    corpus_ids=corpus_ids,
                    embeddings=corpus_embeddings,
                    embedding_model=encoder.model_name,
                )

                query_embedding = encoder.encode([question_text])[0]
                fused_rows = await run_hybrid_search(
                    backend,
                    query_text=question_text,
                    query_embedding=query_embedding,
                    source_name=source_name,
                    limit=max(args.search_limit, max(DEFAULT_METRIC_KS)),
                )
                rankings = fused_rows_to_rankings(fused_rows, corpus_ids)
            finally:
                await purge_benchmark_source(backend, source_name)

            answer_session_ids = {str(session_id) for session_id in entry["answer_session_ids"]}

            session_level_ids = [session_id_from_corpus_id(corpus_id) for corpus_id in corpus_ids]
            turn_level_ids = {
                corpus_id
                for corpus_id in corpus_ids
                if session_id_from_corpus_id(corpus_id) in answer_session_ids
            }

            entry_metrics: dict[str, dict[str, float]] = {"session": {}, "turn": {}}

            for k in DEFAULT_METRIC_KS:
                session_any, session_all, session_ndcg = evaluate_retrieval(
                    rankings,
                    answer_session_ids,
                    session_level_ids,
                    k,
                )
                metrics_session[f"recall_any@{k}"].append(session_any)
                metrics_session[f"recall_all@{k}"].append(session_all)
                metrics_session[f"ndcg_any@{k}"].append(session_ndcg)

                turn_any, turn_all, turn_ndcg = evaluate_retrieval(
                    rankings,
                    turn_level_ids,
                    corpus_ids,
                    k,
                )
                metrics_turn[f"recall_any@{k}"].append(turn_any)
                metrics_turn[f"recall_all@{k}"].append(turn_all)
                metrics_turn[f"ndcg_any@{k}"].append(turn_ndcg)

                entry_metrics["session"][f"recall_any@{k}"] = session_any
                entry_metrics["session"][f"ndcg_any@{k}"] = session_ndcg
                entry_metrics["turn"][f"recall_any@{k}"] = turn_any
                entry_metrics["turn"][f"ndcg_any@{k}"] = turn_ndcg

            per_type[question_type]["recall_any@5"].append(entry_metrics["session"]["recall_any@5"])
            per_type[question_type]["recall_any@10"].append(
                entry_metrics["session"]["recall_any@10"]
            )
            per_type[question_type]["ndcg_any@10"].append(entry_metrics["session"]["ndcg_any@10"])

            ranked_items = []
            for ranked_index in rankings[: max(DEFAULT_METRIC_KS)]:
                ranked_items.append(
                    {
                        "corpus_id": corpus_ids[ranked_index],
                        "text": corpus[ranked_index][:500],
                        "timestamp": corpus_timestamps[ranked_index],
                    }
                )

            results_log.append(
                {
                    "question_id": question_id,
                    "question_type": question_type,
                    "question": question_text,
                    "answer": entry.get("answer", ""),
                    "retrieval_results": {
                        "query": question_text,
                        "ranked_items": ranked_items,
                        "metrics": entry_metrics,
                    },
                }
            )

            r5 = entry_metrics["session"]["recall_any@5"]
            r10 = entry_metrics["session"]["recall_any@10"]
            status = "HIT" if r5 > 0 else "miss"
            print(f"[{index:4}/{len(data)}] {question_id:<32} R@5={r5:.0f} R@10={r10:.0f} {status}")

    finally:
        await backend.disconnect()

    elapsed = time.perf_counter() - started_at
    per_question = elapsed / max(1, len(results_log))

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Elapsed: {elapsed:.1f}s ({per_question:.2f}s per question)")
    print()
    print("SESSION-LEVEL")
    for k in DEFAULT_METRIC_KS:
        recall = _safe_avg(metrics_session[f"recall_any@{k}"])
        ndcg = _safe_avg(metrics_session[f"ndcg_any@{k}"])
        print(f"Recall@{k:>2}: {recall:.3f}   NDCG@{k:>2}: {ndcg:.3f}")

    print()
    print("TURN-LEVEL")
    for k in DEFAULT_METRIC_KS:
        recall = _safe_avg(metrics_turn[f"recall_any@{k}"])
        ndcg = _safe_avg(metrics_turn[f"ndcg_any@{k}"])
        print(f"Recall@{k:>2}: {recall:.3f}   NDCG@{k:>2}: {ndcg:.3f}")

    print()
    print("PER-TYPE BREAKDOWN (session recall_any@10)")
    for question_type in sorted(per_type.keys()):
        values = per_type[question_type]["recall_any@10"]
        print(f"{question_type:35} R@10={_safe_avg(values):.3f} (n={len(values)})")

    if args.out:
        if args.out == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_tag = encoder.model_name.replace("/", "-")
            args.out = (
                f"benchmarks/results_workflows_lme_{args.granularity}_{model_tag}_{timestamp}.jsonl"
            )
        write_jsonl(args.out, results_log)
        print()
        print(f"Results saved to: {args.out}")


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="Workflows-MCP x LongMemEval benchmark")
    parser.add_argument("data_file", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument(
        "--granularity",
        choices=["session", "turn"],
        default="session",
        help="Benchmark at session level (default) or user-turn level.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0 means all).")
    parser.add_argument(
        "--embed-model",
        default="default",
        help=(
            "FastEmbed model alias or HF model name. "
            "Aliases: default(all-MiniLM-L6-v2), bge-base, bge-large, nomic, mxbai."
        ),
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=50,
        help="Top-N fused retrieval depth used for scoring pool.",
    )
    parser.add_argument(
        "--source-prefix",
        default="benchmark:lme",
        help="Source name prefix for temporary benchmark rows.",
    )
    parser.add_argument(
        "--purge-prefix",
        action="store_true",
        help="Delete stale benchmark rows for --source-prefix before running.",
    )
    parser.add_argument(
        "--out",
        default="auto",
        help="Output JSONL path. Use 'auto' for timestamped path under benchmarks/.",
    )
    add_db_args(parser)
    args = parser.parse_args()

    if args.search_limit < max(DEFAULT_METRIC_KS):
        parser.error(f"--search-limit must be >= {max(DEFAULT_METRIC_KS)}")

    await run_benchmark(args)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
