"""Knowledge subpackage for the workflows-mcp engine.

Provides constants, schema DDL, hybrid search, embedding helpers,
and context assembly for the Knowledge executor.
"""

from .constants import Authority, LifecycleState
from .context import assemble_context
from .schema import ensure_schema
from .search import build_fts_search_query, build_vector_search_query, rrf_fusion

__all__ = [
    "Authority",
    "LifecycleState",
    "assemble_context",
    "build_fts_search_query",
    "build_vector_search_query",
    "ensure_schema",
    "rrf_fusion",
]
