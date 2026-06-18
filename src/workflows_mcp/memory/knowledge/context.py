"""Context assembly for Knowledge executor.

Assembles token-budgeted, clean-content-only context from search results.
Enforces metadata/content separation: output contains ONLY proposition content,
never extraction timestamps, file paths, or processing metadata.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .constants import CHARS_PER_TOKEN, DEFAULT_MAX_TOKENS, DEFAULT_MMR_LAMBDA

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses ~4 chars per token heuristic for English text.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _mmr_rerank(
    propositions: list[dict[str, Any]],
    query_embedding: list[float] | None,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> list[dict[str, Any]]:
    """Rerank propositions using Maximal Marginal Relevance.

    MMR score = λ * relevance - (1-λ) * max_similarity_to_selected

    Requires propositions to have an 'embedding' key. Falls back to
    original order if embeddings are unavailable.
    """
    if not propositions or query_embedding is None:
        return propositions

    # Check if embeddings are available
    has_embeddings = all("embedding" in p and p["embedding"] for p in propositions)
    if not has_embeddings:
        return propositions

    selected: list[dict[str, Any]] = []
    remaining = list(propositions)

    while remaining:
        best_score = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            # Relevance to query
            relevance = _cosine_similarity(candidate["embedding"], query_embedding)

            # Max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                sim = _cosine_similarity(candidate["embedding"], sel["embedding"])
                max_sim = max(max_sim, sim)

            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def assemble_context(
    propositions: list[dict[str, Any]],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    diversity: bool = False,
    query_embedding: list[float] | None = None,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> tuple[str, int, int]:
    """Assemble token-budgeted context from propositions.

    Enforces strict metadata/content separation: output contains ONLY
    clean text content from proposition `content` fields. Never includes
    extraction timestamps, file paths, source metadata, or other non-content.

    Args:
        propositions: List of dicts with at minimum a 'content' key.
        max_tokens: Maximum token budget for the assembled context.
        diversity: If True, rerank with MMR to spread across entities/sources.
        query_embedding: Query embedding for MMR (required if diversity=True).
        mmr_lambda: MMR diversity parameter (0=max diversity, 1=pure relevance).

    Returns:
        Tuple of (context_text, proposition_count, tokens_used).
    """
    if not propositions:
        return "", 0, 0

    # Optionally rerank for diversity
    if diversity:
        propositions = _mmr_rerank(propositions, query_embedding, mmr_lambda)

    # Assemble as markdown bullet points within token budget
    lines: list[str] = []
    total_tokens = 0
    count = 0

    for prop in propositions:
        content = prop.get("content", "").strip()
        if not content:
            continue

        line = f"- {content}"
        line_tokens = estimate_tokens(line)

        if total_tokens + line_tokens > max_tokens:
            break

        lines.append(line)
        total_tokens += line_tokens
        count += 1

    context_text = "\n".join(lines)
    return context_text, count, total_tokens
