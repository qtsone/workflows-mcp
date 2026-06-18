"""Memory subsystem: persistent knowledge platform layered on the workflow engine.

Owns the PostgreSQL/pgvector-backed memory service, its graph/scope/locality
helpers, the contract schema, the project-flow service, the memory and System2
block executors, and the ``knowledge/`` data layer. Depends inward on the engine
(executor base, ``sql``, ``llm_config``); the engine never imports this package at
runtime. Memory-gated executors register through the seam in
:mod:`workflows_mcp.memory_runtime`.
"""
