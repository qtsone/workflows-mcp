"""Behavior tests for the eight low-level memory executor ops.

Each op is exercised through MemoryService.execute() so the public
envelope (MemoryRequest) is what's contracted, not the internal
ManageMemoryRequest.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio

PALACE = "palace_ops_test"
WING = "default"
CODE_WING = "code"
ROOM = "default"
COMPARTMENT = "ops"


def _scope() -> dict[str, str]:
    return {"palace": PALACE, "wing": WING, "room": ROOM, "compartment": COMPARTMENT}


def _make_config() -> ConnectionConfig:
    return ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=os.environ.get("MEMORY_DB_HOST", "localhost"),
        port=int(os.environ.get("MEMORY_DB_PORT", "5432")),
        database=os.environ.get("MEMORY_DB_NAME", "workflows"),
        username=os.environ.get("MEMORY_DB_USER", "workflows"),
        password=os.environ.get("MEMORY_DB_PASSWORD", "supersecret"),
    )


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
    return MemoryService(backend=knowledge_backend, context=context)


@pytest_asyncio.fixture
async def clean_palace(knowledge_backend: PostgresBackend) -> AsyncIterator[None]:
    """Wipe rows belonging to the test palace before and after each test."""

    async def _wipe() -> None:
        await knowledge_backend.execute(
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN (SELECT id FROM knowledge_entities WHERE palace = $1)",
            (PALACE,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_memories "
            "WHERE memory_id IN (SELECT id FROM knowledge_memories WHERE palace = $1)",
            (PALACE,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_embeddings "
            "WHERE entity_id IN (SELECT id FROM knowledge_entities WHERE palace = $1)",
            (PALACE,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_memories WHERE palace = $1", (PALACE,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entities WHERE palace = $1", (PALACE,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_items WHERE palace = $1", (PALACE,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_sources WHERE palace = $1", (PALACE,)
        )

    await _wipe()
    yield
    await _wipe()


async def test_clean_palace_starts_empty(knowledge_backend: PostgresBackend, clean_palace: None) -> None:
    rows = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities WHERE palace = $1",
        (PALACE,),
    )
    assert rows.rows[0]["n"] == 0


def test_memory_item_input_schema_exists() -> None:
    """MemoryItemInput must accept item identity + file metadata fields."""
    from workflows_mcp.engine.memory_service import MemoryItemInput

    item = MemoryItemInput.model_validate(
        {
            "id": str(uuid.uuid4()),
            "content_hash": "abc123",
            "size_bytes": 4096,
            "mtime_ns": 1714780800_000_000_000,
            "language": "python",
            "error_metadata": {"reason": "test"},
        }
    )
    assert item.content_hash == "abc123"
    assert item.size_bytes == 4096


def test_memory_record_input_accepts_item_and_embeddings() -> None:
    from workflows_mcp.engine.memory_service import MemoryRecordInput

    rec = MemoryRecordInput.model_validate(
        {
            "format": "structured",
            "item": {"content_hash": "h", "size_bytes": 1, "mtime_ns": 1, "language": "py"},
            "entity_embeddings": [
                 {"entity_id": str(uuid.uuid4()), "profile": "embedding",
                  "model": "text-embedding-3-small",
                  "dimension": 1536,
                  "embedding": [0.1] * 1536}
            ],
        }
    )
    assert rec.item is not None
    assert rec.entity_embeddings is not None
    assert len(rec.entity_embeddings[0].embedding) == 1536
    assert rec.entity_embeddings[0].dimension == 1536


@pytest.mark.parametrize(
    "op",
    [
        "ensure_source",
        "ensure_item",
        "store_entities",
        "store_relations",
        "store_memories",
        "store_entity_embeddings",
        "archive_memories",
        "mark_item_dirty",
    ],
)
def test_memory_request_accepts_new_operations(op: str) -> None:
    """MemoryRequest must accept all eight new operation names without raising."""
    from workflows_mcp.engine.memory_service import MemoryRequest

    payload: dict[str, Any] = {
        "operation": op,
        "scope": _scope(),
        "record": {"format": "raw"},
    }
    req = MemoryRequest.model_validate(payload)
    assert req.operation == op
