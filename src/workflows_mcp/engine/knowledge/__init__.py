"""Knowledge subpackage for the workflows-mcp engine.

Provides constants, schema DDL, hybrid search, embedding helpers,
and context assembly for the Knowledge executor.
"""

from .constants import Authority, LifecycleState
from .context import assemble_context
from .schema import get_init_schema_sql
from .search import build_fts_search_query, build_vector_search_query, rrf_fusion

__all__ = [
    "Authority",
    "LifecycleState",
    "assemble_context",
    "build_fts_search_query",
    "build_vector_search_query",
    "get_init_schema_sql",
    "rrf_fusion",
]
