#!/usr/bin/env python3
"""Workflows-MCP × LoCoMo baseline benchmark.

This runner uses the public LoCoMo dataset format and evaluates
retrieval recall with workflows-mcp's vector + FTS + RRF pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from workflows_bench_common import (
    FastEmbedEncoder,
    add_db_args,
    connect_knowledge_backend,
    db_config_from_args,
    fused_rows_to_rankings,
    ingest_corpus,
    purge_benchmark_source,
    purge_benchmark_sources,
    run_hybrid_search,
)

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}


def load_conversation_sessions(
    conversation: dict[str, Any],
    session_summaries: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    session_number = 1

    while True:
        key = f"session_{session_number}"
        date_key = f"session_{session_number}_date_time"
        if key not in conversation:
            break

        sessions.append(
            {
                "session_num": session_number,
                "date": conversation.get(date_key, ""),
                "dialogs": conversation[key],
                "summary": (session_summaries or {}).get(f"session_{session_number}_summary", ""),
            }
        )
        session_number += 1

    return sessions


def build_corpus_from_sessions(
    sessions: list[dict[str, Any]],
    granularity: str,
) -> tuple[list[str], list[str], list[str]]:
    corpus: list[str] = []
    corpus_ids: list[str] = []
    corpus_timestamps: list[str] = []

    for session in sessions:
        if granularity == "session":
            texts = []
            for dialog in session["dialogs"]:
                speaker = dialog.get("speaker", "?")
                text = dialog.get("text", "")
                texts.append(f'{speaker} said, "{text}"')
            corpus.append("\n".join(texts))
            corpus_ids.append(f"session_{session['session_num']}")
            corpus_timestamps.append(str(session["date"]))
            continue

        for dialog in session["dialogs"]:
            dialog_id = str(dialog.get("dia_id", f"D{session['session_num']}:?"))
            speaker = dialog.get("speaker", "?")
            text = dialog.get("text", "")
            corpus.append(f'{speaker} said, "{text}"')
            corpus_ids.append(dialog_id)
            corpus_timestamps.append(str(session["date"]))

    return corpus, corpus_ids, corpus_timestamps


def evidence_to_dialog_ids(evidence: list[str]) -> set[str]:
    return {str(item) for item in evidence}


def evidence_to_session_ids(evidence: list[str]) -> set[str]:
    sessions: set[str] = set()
    for evidence_id in evidence:
        match = re.match(r"D(\d+):", str(evidence_id))
        if match:
            sessions.add(f"session_{match.group(1)}")
    return sessions


def compute_retrieval_recall(retrieved_ids: list[str], evidence_ids: set[str]) -> float:
    if not evidence_ids:
        return 1.0
    found = sum(1 for evidence_id in evidence_ids if evidence_id in retrieved_ids)
    return found / len(evidence_ids)


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

    all_recall: list[float] = []
    per_category: dict[int, list[float]] = defaultdict(list)
    results_log: list[dict[str, Any]] = []

    total_questions = 0
    started_at = time.perf_counter()

    print("=" * 60)
    print("Workflows-MCP x LoCoMo Benchmark")
    print("=" * 60)
    print(f"Data file:     {Path(args.data_file).name}")
    print(f"Conversations: {len(data)}")
    print(f"Granularity:   {args.granularity}")
    print(f"Top-k:         {args.top_k}")
    if encoder.requested_model_name == encoder.model_name:
        print(f"Embed model:   {encoder.model_name}")
    else:
        print(f"Embed model:   {encoder.requested_model_name} -> {encoder.model_name}")
    print("-" * 60)

    try:
        for conversation_index, sample in enumerate(data, start=1):
            sample_id = str(sample.get("sample_id", f"conv-{conversation_index}"))
            conversation = sample["conversation"]
            qa_pairs = sample["qa"]
            summaries = sample.get("session_summary", {})

            sessions = load_conversation_sessions(conversation, summaries)
            corpus, corpus_ids, corpus_timestamps = build_corpus_from_sessions(
                sessions,
                granularity=args.granularity,
            )

            if not corpus:
                print(f"[{conversation_index:2}/{len(data)}] {sample_id:<24} SKIP (empty corpus)")
                continue

            source_name = f"{args.source_prefix}:{sample_id}"

            try:
                corpus_embeddings = encoder.encode(corpus)
                await ingest_corpus(
                    backend,
                    source_name=source_name,
                    source_type="BENCHMARK_LOCOMO",
                    corpus_texts=corpus,
                    corpus_ids=corpus_ids,
                    embeddings=corpus_embeddings,
                    embedding_model=encoder.model_name,
                )

                print(
                    f"[{conversation_index:2}/{len(data)}] {sample_id:<24} "
                    f"{len(corpus):3} docs, {len(qa_pairs):3} questions"
                )

                for qa in qa_pairs:
                    question = str(qa["question"])
                    category = int(qa["category"])
                    evidence = qa.get("evidence", [])

                    query_embedding = encoder.encode([question])[0]
                    fused_rows = await run_hybrid_search(
                        backend,
                        query_text=question,
                        query_embedding=query_embedding,
                        source_name=source_name,
                        limit=max(args.top_k, 50),
                    )
                    rankings = fused_rows_to_rankings(fused_rows, corpus_ids)
                    retrieved_ids = [corpus_ids[idx] for idx in rankings[: args.top_k]]

                    if args.granularity == "dialog":
                        evidence_ids = evidence_to_dialog_ids(evidence)
                    else:
                        evidence_ids = evidence_to_session_ids(evidence)

                    recall = compute_retrieval_recall(retrieved_ids, evidence_ids)
                    all_recall.append(recall)
                    per_category[category].append(recall)
                    total_questions += 1

                    results_log.append(
                        {
                            "sample_id": sample_id,
                            "question": question,
                            "answer": qa.get("answer", qa.get("adversarial_answer", "")),
                            "category": category,
                            "evidence": evidence,
                            "retrieved_ids": retrieved_ids,
                            "retrieved_timestamps": [
                                corpus_timestamps[idx] for idx in rankings[: args.top_k]
                            ],
                            "recall": recall,
                        }
                    )

            finally:
                await purge_benchmark_source(backend, source_name)

    finally:
        await backend.disconnect()

    elapsed = time.perf_counter() - started_at
    average_recall = _safe_avg(all_recall)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Elapsed:      {elapsed:.1f}s ({elapsed / max(total_questions, 1):.2f}s per question)")
    print(f"Questions:    {total_questions}")
    print(f"Avg Recall:   {average_recall:.3f}")

    print()
    print("PER-CATEGORY RECALL")
    for category in sorted(per_category.keys()):
        values = per_category[category]
        label = CATEGORIES.get(category, f"Category {category}")
        print(f"{label:25} R={_safe_avg(values):.3f} (n={len(values)})")

    if all_recall:
        perfect = sum(1 for value in all_recall if value >= 1.0)
        partial = sum(1 for value in all_recall if 0.0 < value < 1.0)
        zero = sum(1 for value in all_recall if value == 0.0)

        print()
        print("RECALL DISTRIBUTION")
        print(f"Perfect (1.0): {perfect:4} ({perfect / len(all_recall) * 100:.1f}%)")
        print(f"Partial (0-1): {partial:4} ({partial / len(all_recall) * 100:.1f}%)")
        print(f"Zero (0.0):    {zero:4} ({zero / len(all_recall) * 100:.1f}%)")

    if args.out:
        out_path = args.out
        if out_path == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_tag = encoder.model_name.replace("/", "-")
            out_path = (
                f"benchmarks/results_workflows_locomo_{args.granularity}_{model_tag}_"
                f"top{args.top_k}_{timestamp}.json"
            )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(results_log, handle, indent=2)
        print()
        print(f"Results saved to: {out_path}")


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="Workflows-MCP x LoCoMo benchmark")
    parser.add_argument("data_file", help="Path to locomo10.json")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k retrieval depth.")
    parser.add_argument(
        "--granularity",
        choices=["dialog", "session"],
        default="session",
        help="Evaluate by dialog (turn) or session evidence mapping.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N conversations (0 means all).",
    )
    parser.add_argument(
        "--embed-model",
        default="default",
        help=(
            "FastEmbed model alias or HF model name. "
            "Aliases: default(all-MiniLM-L6-v2), bge-base, bge-large, nomic, mxbai."
        ),
    )
    parser.add_argument(
        "--source-prefix",
        default="benchmark:locomo",
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
        help="Output JSON path. Use 'auto' for timestamped path under benchmarks/.",
    )
    add_db_args(parser)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    await run_benchmark(args)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
