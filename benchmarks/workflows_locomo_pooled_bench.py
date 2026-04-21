#!/usr/bin/env python3
"""Workflows-MCP × LoCoMo POOLED benchmark.

Apples-to-apples comparison of room-scoped vs global search on the same dataset.

Three modes run sequentially on identical data:

  GLOBAL   — all 10 conversations pooled into one shared corpus (no room tags).
              Queries run with global hybrid search.  Cross-conversation noise
              degrades scores relative to the isolated baseline.

  SCOPED   — same pooled corpus, but each session is tagged with
              room=<sample_id> (e.g. "conv-26").  Queries run via
              room_scoped_search with the oracle room key.  Should recover
              most of the quality lost by pooling.

  ISOLATED — baseline: each conversation's corpus is isolated (separate source).
              Equivalent to the existing workflows_locomo_bench.py run.  This is
              the oracle upper bound for room routing.

Scoring: average recall@top_k over all QA pairs, same as the existing benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from workflows_bench_common import (
    SYSTEM_USER_UUID,
    FastEmbedEncoder,
    _upsert_source,
    add_db_args,
    connect_knowledge_backend,
    db_config_from_args,
    fused_rows_to_rankings,
    ingest_corpus,
    purge_benchmark_source,
    purge_benchmark_sources,
    run_hybrid_search,
)

from workflows_mcp.engine.knowledge.constants import Authority, LifecycleState
from workflows_mcp.engine.knowledge.search import room_scoped_search

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}

# Source name used for the pooled corpus (no room tags).
POOLED_GLOBAL_SOURCE = "benchmark:locomo-pooled-global"

# Source name used for the pooled+room-tagged corpus.
POOLED_SCOPED_SOURCE = "benchmark:locomo-pooled-scoped"

# Namespace used when tagging pooled sessions for room-scoped search.
POOLED_NAMESPACE = "locomo-pooled"


# ---------------------------------------------------------------------------
# LoCoMo helpers (same as workflows_locomo_bench.py)
# ---------------------------------------------------------------------------


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


def evidence_to_session_ids(evidence: list[str]) -> set[str]:
    sessions: set[str] = set()
    for evidence_id in evidence:
        match = re.match(r"D(\d+):", str(evidence_id))
        if match:
            sessions.add(f"session_{match.group(1)}")
    return sessions


def evidence_to_dialog_ids(evidence: list[str]) -> set[str]:
    return {str(item) for item in evidence}


def compute_retrieval_recall(retrieved_ids: list[str], evidence_ids: set[str]) -> float:
    if not evidence_ids:
        return 1.0
    found = sum(1 for evidence_id in evidence_ids if evidence_id in retrieved_ids)
    return found / len(evidence_ids)


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Pooled ingest with per-row namespace/room (append-safe, no mid-loop purge)
# ---------------------------------------------------------------------------


async def _ingest_conversation_with_room(
    backend: Any,
    *,
    source_id: str,
    source_name: str,
    corpus_texts: list[str],
    prefixed_ids: list[str],
    embeddings: list[list[float]],
    embedding_model: str,
    namespace: str,
    room: str,
) -> None:
    """Insert one conversation's rows into an already-created source, with room tags."""
    if not corpus_texts:
        return

    item_rows: list[tuple[str, str, str, str]] = []
    memory_rows: list[tuple[Any, ...]] = []

    for position, (text, pid, vector) in enumerate(
        zip(corpus_texts, prefixed_ids, embeddings, strict=True)
    ):
        item_id = str(uuid.uuid4())
        memory_id = str(uuid.uuid4())
        item_path = f"{position}:{pid}"
        item_rows.append((item_id, source_id, item_path, pid))
        memory_rows.append(
            (
                memory_id,
                item_id,
                text,
                str(vector),
                Authority.EXTRACTED,
                LifecycleState.ACTIVE,
                1.0,
                embedding_model,
                "{}",
                str(SYSTEM_USER_UUID),
                "SYSTEM",
                source_name,
                "BENCHMARK_LOCOMO_SCOPED",
                namespace,
                room,
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
             namespace, room)
        VALUES
            ($1::uuid, $2::uuid, $3, $4::vector,
             to_tsvector('english', $3),
             $5, $6, $7, $8, $9::jsonb,
             $10::uuid, $11, $12, $13, $14, $15)
        """,
        memory_rows,
    )


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


async def run_global_mode(
    backend: Any,
    encoder: FastEmbedEncoder,
    data: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[float], dict[int, list[float]]]:
    """Pool all conversations into a single source, query with global search."""
    print("\n--- MODE: GLOBAL (pooled, no room tags) ---")

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    qa_index: list[tuple[str, list[dict[str, Any]], list[str]]] = []

    for sample in data:
        sample_id = str(sample.get("sample_id", "unknown"))
        conversation = sample["conversation"]
        summaries = sample.get("session_summary", {})
        sessions = load_conversation_sessions(conversation, summaries)
        corpus, corpus_ids, _ = build_corpus_from_sessions(sessions, args.granularity)
        if not corpus:
            continue

        # Prefix IDs to avoid collisions across conversations
        prefixed_ids = [f"{sample_id}:{cid}" for cid in corpus_ids]
        embeddings = encoder.encode(corpus)
        all_texts.extend(corpus)
        all_ids.extend(prefixed_ids)
        all_embeddings.extend(embeddings)
        qa_index.append((sample_id, sample["qa"], prefixed_ids))

    print(f"  Ingesting pooled corpus: {len(all_texts)} docs from {len(data)} conversations")
    await ingest_corpus(
        backend,
        source_name=POOLED_GLOBAL_SOURCE,
        source_type="BENCHMARK_LOCOMO_GLOBAL",
        corpus_texts=all_texts,
        corpus_ids=all_ids,
        embeddings=all_embeddings,
        embedding_model=encoder.model_name,
    )

    all_recall: list[float] = []
    per_category: dict[int, list[float]] = defaultdict(list)
    q_count = 0

    try:
        for sample_id, qa_pairs, prefixed_ids in qa_index:
            for qa in qa_pairs:
                question = str(qa["question"])
                category = int(qa["category"])
                evidence = qa.get("evidence", [])

                query_embedding = encoder.encode([question])[0]
                fused_rows = await run_hybrid_search(
                    backend,
                    query_text=question,
                    query_embedding=query_embedding,
                    source_name=POOLED_GLOBAL_SOURCE,
                    limit=max(args.top_k, 50),
                )

                # Retrieve top-k among the *full* pooled corpus, then check
                # whether the correct sessions for this conversation were found.
                # We reconstruct a per-conversation view: only consider prefixed_ids
                # from this conversation in the rankings.
                rankings = fused_rows_to_rankings(fused_rows, prefixed_ids)
                retrieved_ids = [prefixed_ids[i].split(":", 1)[1] for i in rankings[: args.top_k]]

                if args.granularity == "dialog":
                    evidence_ids = evidence_to_dialog_ids(evidence)
                else:
                    evidence_ids = evidence_to_session_ids(evidence)

                recall = compute_retrieval_recall(retrieved_ids, evidence_ids)
                all_recall.append(recall)
                per_category[category].append(recall)
                q_count += 1
    finally:
        await purge_benchmark_source(backend, POOLED_GLOBAL_SOURCE)

    print(f"  Questions evaluated: {q_count}")
    print(f"  Avg Recall@{args.top_k}: {_safe_avg(all_recall):.3f}")
    return all_recall, per_category


async def run_scoped_mode(
    backend: Any,
    encoder: FastEmbedEncoder,
    data: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[float], dict[int, list[float]]]:
    """Pool all conversations with room tags; query with room_scoped_search."""
    print("\n--- MODE: SCOPED (pooled + room routing, oracle room=sample_id) ---")

    # Purge stale data, create the shared source once, then append per conversation.
    await purge_benchmark_source(backend, POOLED_SCOPED_SOURCE)
    source_id = await _upsert_source(backend, POOLED_SCOPED_SOURCE, "BENCHMARK_LOCOMO_SCOPED")

    total_docs = 0
    qa_index: list[tuple[str, list[dict[str, Any]], list[str]]] = []

    for sample in data:
        sample_id = str(sample.get("sample_id", "unknown"))
        conversation = sample["conversation"]
        summaries = sample.get("session_summary", {})
        sessions = load_conversation_sessions(conversation, summaries)
        corpus, corpus_ids, _ = build_corpus_from_sessions(sessions, args.granularity)
        if not corpus:
            continue

        prefixed_ids = [f"{sample_id}:{cid}" for cid in corpus_ids]
        embeddings = encoder.encode(corpus)
        await _ingest_conversation_with_room(
            backend,
            source_id=source_id,
            source_name=POOLED_SCOPED_SOURCE,
            corpus_texts=corpus,
            prefixed_ids=prefixed_ids,
            embeddings=embeddings,
            embedding_model=encoder.model_name,
            namespace=POOLED_NAMESPACE,
            room=sample_id,
        )
        total_docs += len(corpus)
        qa_index.append((sample_id, sample["qa"], prefixed_ids))

    print(f"  Ingested pooled+room corpus: {total_docs} docs from {len(qa_index)} conversations")

    all_recall: list[float] = []
    per_category: dict[int, list[float]] = defaultdict(list)
    q_count = 0

    try:
        for sample_id, qa_pairs, prefixed_ids in qa_index:
            for qa in qa_pairs:
                question = str(qa["question"])
                category = int(qa["category"])
                evidence = qa.get("evidence", [])

                query_embedding = encoder.encode([question])[0]
                fused_rows = await room_scoped_search(
                    query_embedding=query_embedding,
                    query_text=question,
                    backend=backend,
                    namespace=POOLED_NAMESPACE,
                    room=sample_id,
                    source=POOLED_SCOPED_SOURCE,
                    min_confidence=0.0,
                    lifecycle_state=LifecycleState.ACTIVE,
                    limit=max(args.top_k, 50),
                )

                rankings = fused_rows_to_rankings(fused_rows, prefixed_ids)
                retrieved_ids = [prefixed_ids[i].split(":", 1)[1] for i in rankings[: args.top_k]]

                if args.granularity == "dialog":
                    evidence_ids = evidence_to_dialog_ids(evidence)
                else:
                    evidence_ids = evidence_to_session_ids(evidence)

                recall = compute_retrieval_recall(retrieved_ids, evidence_ids)
                all_recall.append(recall)
                per_category[category].append(recall)
                q_count += 1
    finally:
        await purge_benchmark_source(backend, POOLED_SCOPED_SOURCE)

    print(f"  Questions evaluated: {q_count}")
    print(f"  Avg Recall@{args.top_k}: {_safe_avg(all_recall):.3f}")
    return all_recall, per_category


async def run_isolated_mode(
    backend: Any,
    encoder: FastEmbedEncoder,
    data: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[float], dict[int, list[float]]]:
    """Isolated baseline: one source per conversation (same as existing benchmark)."""
    print("\n--- MODE: ISOLATED (per-conversation source, no cross-conversation noise) ---")

    all_recall: list[float] = []
    per_category: dict[int, list[float]] = defaultdict(list)
    q_count = 0

    for conversation_index, sample in enumerate(data, start=1):
        sample_id = str(sample.get("sample_id", f"conv-{conversation_index}"))
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]
        summaries = sample.get("session_summary", {})

        sessions = load_conversation_sessions(conversation, summaries)
        corpus, corpus_ids, _ = build_corpus_from_sessions(sessions, args.granularity)
        if not corpus:
            continue

        source_name = f"benchmark:locomo-isolated:{sample_id}"

        try:
            embeddings = encoder.encode(corpus)
            await ingest_corpus(
                backend,
                source_name=source_name,
                source_type="BENCHMARK_LOCOMO_ISOLATED",
                corpus_texts=corpus,
                corpus_ids=corpus_ids,
                embeddings=embeddings,
                embedding_model=encoder.model_name,
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
                retrieved_ids = [corpus_ids[i] for i in rankings[: args.top_k]]

                if args.granularity == "dialog":
                    evidence_ids = evidence_to_dialog_ids(evidence)
                else:
                    evidence_ids = evidence_to_session_ids(evidence)

                recall = compute_retrieval_recall(retrieved_ids, evidence_ids)
                all_recall.append(recall)
                per_category[category].append(recall)
                q_count += 1

        finally:
            await purge_benchmark_source(backend, source_name)

    print(f"  Questions evaluated: {q_count}")
    print(f"  Avg Recall@{args.top_k}: {_safe_avg(all_recall):.3f}")
    return all_recall, per_category


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_mode_summary(
    label: str,
    recall: list[float],
    per_category: dict[int, list[float]],
    top_k: int,
) -> None:
    avg = _safe_avg(recall)
    print(f"\n  {label}")
    print(f"    Avg Recall@{top_k}: {avg:.3f}  (n={len(recall)})")
    for cat in sorted(per_category.keys()):
        values = per_category[cat]
        cat_label = CATEGORIES.get(cat, f"Cat{cat}")
        print(f"    {cat_label:25} {_safe_avg(values):.3f}  (n={len(values)})")
    if recall:
        perfect = sum(1 for v in recall if v >= 1.0)
        zero = sum(1 for v in recall if v == 0.0)
        print(
            f"    Perfect/zero: {perfect}/{zero}  "
            f"({perfect / len(recall) * 100:.1f}% / {zero / len(recall) * 100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark(args: argparse.Namespace) -> None:
    with open(args.data_file, encoding="utf-8") as handle:
        data = json.load(handle)

    if args.limit > 0:
        data = data[: args.limit]

    encoder = FastEmbedEncoder(model_name=args.embed_model)
    db_config = db_config_from_args(args)
    backend = await connect_knowledge_backend(db_config)

    if args.purge_prefix:
        await purge_benchmark_sources(backend, "benchmark:locomo-pooled")
        await purge_benchmark_sources(backend, "benchmark:locomo-isolated")

    print("=" * 70)
    print("Workflows-MCP x LoCoMo POOLED Benchmark")
    print("=" * 70)
    print(f"Data file:     {Path(args.data_file).name}")
    print(f"Conversations: {len(data)}")
    print(f"Granularity:   {args.granularity}")
    print(f"Top-k:         {args.top_k}")
    if encoder.requested_model_name == encoder.model_name:
        print(f"Embed model:   {encoder.model_name}")
    else:
        print(f"Embed model:   {encoder.requested_model_name} -> {encoder.model_name}")
    print(f"Modes:         {', '.join(args.modes)}")
    print("-" * 70)

    started_at = time.perf_counter()
    results: dict[str, tuple[list[float], dict[int, list[float]]]] = {}

    try:
        if "global" in args.modes:
            results["global"] = await run_global_mode(backend, encoder, data, args)
        if "scoped" in args.modes:
            results["scoped"] = await run_scoped_mode(backend, encoder, data, args)
        if "isolated" in args.modes:
            results["isolated"] = await run_isolated_mode(backend, encoder, data, args)
    finally:
        await backend.disconnect()

    elapsed = time.perf_counter() - started_at

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Elapsed: {elapsed:.1f}s")

    for mode in ["isolated", "scoped", "global"]:
        if mode in results:
            recall, per_cat = results[mode]
            label = {
                "global": "GLOBAL   (pooled, no room tags)         <- degraded baseline",
                "scoped": "SCOPED   (pooled + oracle room routing)  <- room routing value",
                "isolated": "ISOLATED (per-conversation, oracle)     <- upper bound",
            }[mode]
            print_mode_summary(label, recall, per_cat, args.top_k)

    # Delta summary
    if "global" in results and "scoped" in results:
        g = _safe_avg(results["global"][0])
        s = _safe_avg(results["scoped"][0])
        delta_pp = (s - g) * 100
        print(f"\n  Scoped vs Global delta: {delta_pp:+.1f}pp  ({s:.3f} vs {g:.3f})")

    if "isolated" in results and "scoped" in results and "global" in results:
        iso = _safe_avg(results["isolated"][0])
        s = _safe_avg(results["scoped"][0])
        g = _safe_avg(results["global"][0])
        gap = iso - g
        recovery_pct = ((s - g) / gap * 100) if gap > 1e-9 else 0.0
        print(f"  Room routing recovery:  {recovery_pct:.1f}% of gap vs isolated ceiling")

    if args.out:
        out_path = args.out
        if out_path == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_tag = encoder.model_name.replace("/", "-")
            modes_tag = "-".join(sorted(args.modes))
            out_path = (
                f"benchmarks/results_workflows_locomo_pooled_{args.granularity}_{model_tag}_"
                f"top{args.top_k}_{modes_tag}_{timestamp}.json"
            )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "benchmark": "locomo-pooled",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "data_file": str(args.data_file),
                "conversations": len(data),
                "granularity": args.granularity,
                "top_k": args.top_k,
                "embed_model": encoder.model_name,
                "modes": args.modes,
            },
            "results": {
                mode: {
                    "avg_recall": _safe_avg(recall),
                    "n_questions": len(recall),
                    "per_category": {
                        CATEGORIES.get(cat, str(cat)): {
                            "avg_recall": _safe_avg(vals),
                            "n": len(vals),
                        }
                        for cat, vals in per_cat.items()
                    },
                }
                for mode, (recall, per_cat) in results.items()
            },
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2)
        print(f"\nResults saved to: {out_path}")


async def _main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Workflows-MCP x LoCoMo POOLED benchmark (global vs scoped vs isolated)"
    )
    parser.add_argument("data_file", help="Path to locomo10.json")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval depth.")
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
        help="FastEmbed model alias or HF model name.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["global", "scoped", "isolated"],
        default=["global", "scoped", "isolated"],
        help="Which modes to run (default: all three).",
    )
    parser.add_argument(
        "--purge-prefix",
        action="store_true",
        help="Delete stale benchmark rows before running.",
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
