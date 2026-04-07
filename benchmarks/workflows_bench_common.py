from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflows_mcp.engine.knowledge.constants import Authority, LifecycleState
from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.knowledge.search import (
    build_fts_search_query,
    build_vector_search_query,
    rrf_fusion,
)
from workflows_mcp.engine.sql import ConnectionConfig, DatabaseEngine, PostgresBackend

SYSTEM_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_METRIC_KS = [1, 3, 5, 10, 30, 50]
DEFAULT_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _env_default(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return default


@dataclass(frozen=True)
class BenchmarkDbConfig:
    host: str
    port: int
    database: str
    username: str | None
    password: str | None
    ssl: bool = False


def add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db-host",
        default=_env_default("KNOWLEDGE_DB_HOST", "PGHOST", default="localhost"),
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(_env_default("KNOWLEDGE_DB_PORT", "PGPORT", default="5432") or "5432"),
    )
    parser.add_argument(
        "--db-name",
        default=_env_default(
            "KNOWLEDGE_DB_NAME",
            "PGDATABASE",
            "PGUSER",
            default="knowledge_db",
        ),
        help="PostgreSQL database used by workflows-mcp knowledge.",
    )
    parser.add_argument(
        "--db-user",
        default=_env_default("KNOWLEDGE_DB_USER", "PGUSER"),
    )
    parser.add_argument(
        "--db-password",
        default=_env_default("KNOWLEDGE_DB_PASSWORD", "PGPASSWORD"),
    )
    parser.add_argument(
        "--db-ssl",
        action="store_true",
        help="Enable TLS when connecting to PostgreSQL.",
    )


def db_config_from_args(args: argparse.Namespace) -> BenchmarkDbConfig:
    return BenchmarkDbConfig(
        host=str(args.db_host),
        port=int(args.db_port),
        database=str(args.db_name),
        username=(str(args.db_user) if args.db_user else None),
        password=(str(args.db_password) if args.db_password else None),
        ssl=bool(args.db_ssl),
    )


class FastEmbedEncoder:
    """Thin wrapper around fastembed for local, reproducible embeddings."""

    MODEL_ALIASES: dict[str, str] = {
        "default": DEFAULT_FASTEMBED_MODEL,
        "all-minilm-l6-v2": DEFAULT_FASTEMBED_MODEL,
        "minilm": DEFAULT_FASTEMBED_MODEL,
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "nomic": "nomic-ai/nomic-embed-text-v1.5",
        "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    }

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - installation error path
            raise RuntimeError(
                "fastembed is required for benchmarks. Install with: uv pip install fastembed"
            ) from exc

        self.requested_model_name = model_name
        self.model_name = self.MODEL_ALIASES.get(model_name, model_name)
        self._model = TextEmbedding(model_name=self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.embed(texts)
        return [[float(value) for value in vector] for vector in vectors]


async def connect_knowledge_backend(config: BenchmarkDbConfig) -> Any:
    if PostgresBackend is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "PostgreSQL backend unavailable. Install postgres deps with: uv sync --extra postgresql"
        )

    backend = PostgresBackend()
    await backend.connect(
        ConnectionConfig(
            dialect=DatabaseEngine.POSTGRESQL,
            host=config.host,
            port=config.port,
            database=config.database,
            username=config.username,
            password=config.password,
            ssl=config.ssl,
        )
    )
    await ensure_schema(backend)
    return backend


async def purge_benchmark_sources(backend: Any, source_prefix: str) -> None:
    await backend.execute(
        "DELETE FROM knowledge_sources WHERE name LIKE $1",
        (f"{source_prefix}:%",),
    )


async def purge_benchmark_source(backend: Any, source_name: str) -> None:
    await backend.execute(
        "DELETE FROM knowledge_sources WHERE name = $1",
        (source_name,),
    )


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
        raise RuntimeError(f"Failed to upsert knowledge source: {source_name}")
    return str(result.rows[0]["id"])


async def ingest_corpus(
    backend: Any,
    *,
    source_name: str,
    source_type: str,
    corpus_texts: list[str],
    corpus_ids: list[str],
    embeddings: list[list[float]],
    embedding_model: str,
    confidence: float = 1.0,
) -> None:
    if not corpus_texts:
        return

    if len(corpus_texts) != len(corpus_ids):
        raise ValueError("corpus_texts and corpus_ids must have identical length")
    if len(corpus_texts) != len(embeddings):
        raise ValueError("corpus_texts and embeddings must have identical length")

    await purge_benchmark_source(backend, source_name)
    source_id = await _upsert_source(backend, source_name, source_type)

    embedding_dimensions = len(embeddings[0]) if embeddings else 0

    item_rows: list[tuple[str, str, str, str]] = []
    proposition_rows: list[tuple[Any, ...]] = []

    for position, (text, corpus_id, vector) in enumerate(
        zip(corpus_texts, corpus_ids, embeddings, strict=True)
    ):
        item_id = str(uuid.uuid4())
        proposition_id = str(uuid.uuid4())
        item_path = f"{position}:{corpus_id}"

        item_rows.append((item_id, source_id, item_path, corpus_id))
        proposition_rows.append(
            (
                proposition_id,
                item_id,
                text,
                str(vector),
                Authority.EXTRACTED,
                LifecycleState.ACTIVE,
                confidence,
                embedding_model,
                embedding_dimensions,
                "{}",
                str(SYSTEM_USER_UUID),
                "SYSTEM",
                source_name,
                source_type,
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
        INSERT INTO knowledge_propositions
            (id, item_id, content, embedding, search_vector,
             authority, lifecycle_state, confidence,
             embedding_model, embedding_dimensions, metadata,
             created_by, auth_method, source_name, source_type)
        VALUES
            ($1::uuid, $2::uuid, $3, $4::vector,
             to_tsvector('english', $3),
             $5, $6, $7,
             $8, $9, $10::jsonb,
             $11::uuid, $12, $13, $14)
        """,
        proposition_rows,
    )


async def run_hybrid_search(
    backend: Any,
    *,
    query_text: str,
    query_embedding: list[float],
    source_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    vector_sql, vector_params = build_vector_search_query(
        query_embedding=query_embedding,
        source=source_name,
        min_confidence=0.0,
        lifecycle_state=LifecycleState.ACTIVE,
        limit=limit,
    )
    vector_result = await backend.query(vector_sql, tuple(vector_params))
    vector_rows = [dict(row) for row in vector_result.rows]

    fts_sql, fts_params = build_fts_search_query(
        query_text=query_text,
        source=source_name,
        min_confidence=0.0,
        lifecycle_state=LifecycleState.ACTIVE,
        limit=limit,
    )
    fts_result = await backend.query(fts_sql, tuple(fts_params))
    fts_rows = [dict(row) for row in fts_result.rows]

    return rrf_fusion(vector_rows, fts_rows, limit=limit)


def fused_rows_to_rankings(fused_rows: list[dict[str, Any]], corpus_ids: list[str]) -> list[int]:
    index_by_id: dict[str, list[int]] = {}
    for idx, corpus_id in enumerate(corpus_ids):
        index_by_id.setdefault(corpus_id, []).append(idx)

    rankings: list[int] = []
    seen: set[int] = set()

    for row in fused_rows:
        item_path = row.get("item_path")
        if not isinstance(item_path, str):
            continue

        corpus_id = item_path.split(":", 1)[1] if ":" in item_path else item_path
        candidates = index_by_id.get(corpus_id)
        if not candidates:
            continue

        candidate_idx = next((candidate for candidate in candidates if candidate not in seen), None)
        if candidate_idx is None:
            continue

        seen.add(candidate_idx)
        rankings.append(candidate_idx)

    for idx in range(len(corpus_ids)):
        if idx not in seen:
            rankings.append(idx)

    return rankings


def dcg(relevances: list[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(rankings: list[int], correct_ids: set[str], corpus_ids: list[str], k: int) -> float:
    relevances = [1.0 if corpus_ids[idx] in correct_ids else 0.0 for idx in rankings[:k]]
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def evaluate_retrieval(
    rankings: list[int],
    correct_ids: set[str],
    corpus_ids: list[str],
    k: int,
) -> tuple[float, float, float]:
    top_k_ids = {corpus_ids[idx] for idx in rankings[:k]}
    recall_any = float(any(doc_id in top_k_ids for doc_id in correct_ids))
    recall_all = float(all(doc_id in top_k_ids for doc_id in correct_ids))
    ndcg_score = ndcg(rankings, correct_ids, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


def session_id_from_corpus_id(corpus_id: str) -> str:
    if "_turn_" in corpus_id:
        return corpus_id.rsplit("_turn_", 1)[0]
    return corpus_id


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, rows: list[dict[str, Any]] | list[str]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(f"{row}\n")
            else:
                handle.write(f"{json.dumps(row, ensure_ascii=True)}\n")
