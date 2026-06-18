"""Live-Postgres fixtures for the memory executor-op test package.

Auto-discovered by pytest for the ``test_*`` modules in this directory. They
connect to the pgvector Postgres via the ``MEMORY_DB_*`` environment variables
and reset the test palace around each test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest_asyncio
from _ops_helpers import PALACE, _make_config

from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend


@pytest_asyncio.fixture
async def knowledge_backend() -> AsyncIterator[PostgresBackend]:
    """Provide a connected PostgresBackend with the knowledge schema applied."""
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


@pytest_asyncio.fixture
async def memory_service(knowledge_backend: PostgresBackend) -> Any:
    """Provide a MemoryService wired to the live PostgresBackend."""
    from unittest.mock import MagicMock

    from workflows_mcp.engine.executor_base import Execution
    from workflows_mcp.engine.memory_service import MemoryService

    context = MagicMock(spec=Execution)
    context.execution_context = MagicMock()
    context.execution_context.get = MagicMock(return_value=None)
    context.execution_context.user_id = None
    context.execution_context.user_string_id = None
    context.execution_context.auth_method = None
    return MemoryService(backend=knowledge_backend, context=context)


@pytest_asyncio.fixture
async def clean_palace(knowledge_backend: PostgresBackend) -> AsyncIterator[None]:
    """Wipe rows belonging to the test palace before and after each test."""

    async def _wipe() -> None:
        # Wipe all test palaces (primary and any cross-palace variants created by tests).
        palace_pattern = f"{PALACE}%"
        await knowledge_backend.execute(
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_memories "
            "WHERE memory_id IN (SELECT id FROM knowledge_memories WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_embeddings "
            "WHERE entity_id IN (SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_memories WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entities WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_items WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_sources WHERE palace LIKE $1", (palace_pattern,)
        )
        # Delete topology provenance evidence links before structural evidence (RESTRICT FK).
        await knowledge_backend.execute(
            """
            DELETE FROM knowledge_topology_provenance_evidence
             WHERE provenance_id IN (
                 SELECT id FROM knowledge_topology_provenance WHERE palace LIKE $1
             )
            """,
            (palace_pattern,),
        )
        # Delete topology provenance rows before semantic claims and overrides.
        await knowledge_backend.execute(
            "DELETE FROM knowledge_topology_provenance WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_structural_evidence WHERE palace LIKE $1", (palace_pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_verification_cycles WHERE palace LIKE $1", (palace_pattern,)
        )
        # Delete corridor edge rows before claims — from_claim_id/to_claim_id FKs use CASCADE
        # but we delete explicitly to keep teardown order safe across schema versions.
        await knowledge_backend.execute(
            """
            DELETE FROM knowledge_semantic_corridors
             WHERE from_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
               OR to_claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
            """,
            (palace_pattern,),
        )
        # Delete override rows referencing claims before deleting claims (FK RESTRICT).
        await knowledge_backend.execute(
            """
            DELETE FROM knowledge_semantic_overrides
             WHERE claim_id IN (
                 SELECT id FROM knowledge_semantic_claims WHERE palace LIKE $1
             )
            """,
            (palace_pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_semantic_claims WHERE palace LIKE $1", (palace_pattern,)
        )

    await _wipe()
    yield
    await _wipe()
