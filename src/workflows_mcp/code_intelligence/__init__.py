"""Code-intelligence subsystem: treesitter parsing and file outlines.

Houses the static-analysis blocks and helpers that the workflow engine treats as
an optional capability rather than core DAG machinery. Registration is inverted so
the DAG core never imports this package: the host wires it in at startup via
:func:`register_code_intelligence_executors`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workflows_mcp.engine.executor_base import ExecutorRegistry


def register_code_intelligence_executors(registry: ExecutorRegistry) -> None:
    """Register code-intelligence block executors on a registry, idempotently.

    Generalizes the conditional-registration seam used by the memory subsystem so
    the DAG-core ``create_default_registry`` no longer hard-wires treesitter.
    """
    from .executors_treesitter import TreeSitterExecutor

    if not registry.has("TreeSitter"):
        registry.register(TreeSitterExecutor())


__all__ = ["register_code_intelligence_executors"]
