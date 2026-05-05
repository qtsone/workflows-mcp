"""E2E integration test for the system1-scan built-in workflow.

Phase 5 TDD — RED written before workflow implementation.

Contract tested:
- system1-scan.yaml is a valid built-in workflow (loaded from
  src/workflows_mcp/builtin_workflows/).
- It wires TreeSitter parse output to Memory entity + relation storage.
- A first run stores File/Module/symbol entities and structural relations.
- A second unchanged run inserts zero new rows in all three tables:
  knowledge_items, knowledge_entities, knowledge_relations.

Execution model: the workflow is executed via execute_workflow() MCP tool
with a mock AppContext pointing at the real PostgreSQL test database
(env: MEMORY_DB_HOST / MEMORY_DB_PORT / MEMORY_DB_NAME / MEMORY_DB_USER /
MEMORY_DB_PASSWORD).  The same connection is used for DB assertions so
test setup/teardown are consistent.

Database isolation: all rows belonging to PALACE = "palace_system1_scan_test"
are wiped before and after each test function.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest_asyncio

from workflows_mcp.context import AppContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.executors_memory import MemoryExecutor
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.job_queue import JobQueue
from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend
from workflows_mcp.tools import execute_workflow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALACE = "palace_system1_scan_test"
# Scope used for all Memory block calls inside the workflow
SCOPE = {
    "palace": PALACE,
    "wing": "code",
    "room": "default",
    "compartment": "system1scan",
}

# Simple Python fixture — must produce File + Module + at least one Function entity
# and at least one relation (CONTAINS).
FIXTURE_PY_SRC = """\
import os


class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}"


def main() -> None:
    g = Greeter()
    g.greet("world")
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


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
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


@pytest_asyncio.fixture
async def clean_palace(knowledge_backend: PostgresBackend) -> AsyncIterator[None]:
    """Wipe all rows for PALACE before and after each test."""

    async def _wipe() -> None:
        pattern = f"{PALACE}%"
        await knowledge_backend.execute(
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN "
            "(SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_memories "
            "WHERE memory_id IN "
            "(SELECT id FROM knowledge_memories WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_embeddings "
            "WHERE entity_id IN "
            "(SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_memories WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entities WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_items WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_sources WHERE palace LIKE $1", (pattern,)
        )

    await _wipe()
    yield
    await _wipe()


# ---------------------------------------------------------------------------
# Workflow execution context
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def workflow_context() -> AsyncIterator[MagicMock]:
    """AppContext wired to the built-in workflow registry."""
    builtin_dir = (
        Path(__file__).parent.parent
        / "src"
        / "workflows_mcp"
        / "builtin_workflows"
    )

    # Ensure MemoryExecutor picks up the same DB credentials as the test fixture.
    # These env vars are read at MemoryInput construction time (via default_factory),
    # so they must be set before the workflow executes.
    db_env_defaults = {
        "MEMORY_DB_HOST": os.environ.get("MEMORY_DB_HOST", "localhost"),
        "MEMORY_DB_PORT": os.environ.get("MEMORY_DB_PORT", "5432"),
        "MEMORY_DB_NAME": os.environ.get("MEMORY_DB_NAME", "workflows"),
        "MEMORY_DB_USER": os.environ.get("MEMORY_DB_USER", "workflows"),
        "MEMORY_DB_PASSWORD": os.environ.get("MEMORY_DB_PASSWORD", "supersecret"),
    }
    prev_env = {k: os.environ.get(k) for k in db_env_defaults}
    os.environ.update(db_env_defaults)

    registry = WorkflowRegistry()
    registry.load_from_directory(builtin_dir)

    executor_registry = create_default_registry()
    executor_registry.register(MemoryExecutor())
    llm_config_loader = LLMConfigLoader()
    io_queue = IOQueue()

    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,
    )

    job_queue = JobQueue(app_context, num_workers=2)
    await job_queue._store.init()

    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=job_queue,
    )

    await io_queue.start()
    try:
        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = app_context
        yield mock_ctx
    finally:
        await io_queue.stop()
        # Restore previous env state
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Helper: count rows
# ---------------------------------------------------------------------------


async def _count(backend: PostgresBackend, table: str) -> int:
    rows = await backend.query(
        f"SELECT COUNT(*)::int AS n FROM {table} WHERE palace = $1",  # noqa: S608
        (PALACE,),
    )
    return int(rows.rows[0]["n"])


async def _count_relations(backend: PostgresBackend) -> int:
    rows = await backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations "
        "WHERE source_entity_id IN "
        "(SELECT id FROM knowledge_entities WHERE palace = $1)",
        (PALACE,),
    )
    return int(rows.rows[0]["n"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_system1_scan_workflow_is_registered(
    workflow_context: MagicMock,
) -> None:
    """system1-scan must be discoverable in the built-in workflow registry."""
    app_ctx: AppContext = workflow_context.request_context.lifespan_context
    assert app_ctx.registry.get("system1-scan") is not None, (
        "system1-scan workflow not found in built-in registry"
    )


async def test_system1_scan_first_run_stores_entities_and_relations(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """First scan of a Python file stores File/Module/symbol entities and relations."""
    # Arrange: write fixture Python file
    fixture_file = tmp_path / "greeter.py"
    fixture_file.write_text(FIXTURE_PY_SRC, encoding="utf-8")

    result = await execute_workflow(
        workflow="system1-scan",
        inputs={
            "file_path": str(fixture_file),
            "repo_relative_path": "greeter.py",
            "palace": PALACE,
            "source_name": "test-repo",
        },
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )

    response: dict[str, Any] = result.structuredContent
    assert response.get("status") == "success", (
        f"Workflow did not complete successfully: {response}"
    )

    items = await _count(knowledge_backend, "knowledge_items")
    entities = await _count(knowledge_backend, "knowledge_entities")
    relations = await _count_relations(knowledge_backend)

    assert items >= 1, "Expected at least one knowledge_items row for the scanned file"
    assert entities >= 3, (
        "Expected at least File + Module + one symbol entity "
        f"(got {entities})"
    )
    assert relations >= 1, "Expected at least one CONTAINS relation"


async def test_system1_scan_second_run_is_idempotent(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """Second run on an unchanged file inserts zero new rows in all three tables."""
    fixture_file = tmp_path / "greeter.py"
    fixture_file.write_text(FIXTURE_PY_SRC, encoding="utf-8")

    common_inputs: dict[str, Any] = {
        "file_path": str(fixture_file),
        "repo_relative_path": "greeter.py",
        "palace": PALACE,
        "source_name": "test-repo",
    }

    # First run
    r1 = await execute_workflow(
        workflow="system1-scan",
        inputs=common_inputs,
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )
    assert r1.structuredContent.get("status") == "success", (
        f"First run failed: {r1.structuredContent}"
    )

    items_after_1 = await _count(knowledge_backend, "knowledge_items")
    entities_after_1 = await _count(knowledge_backend, "knowledge_entities")
    relations_after_1 = await _count_relations(knowledge_backend)

    # Second run (same file, same content)
    r2 = await execute_workflow(
        workflow="system1-scan",
        inputs=common_inputs,
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )
    assert r2.structuredContent.get("status") == "success", (
        f"Second run failed: {r2.structuredContent}"
    )

    items_after_2 = await _count(knowledge_backend, "knowledge_items")
    entities_after_2 = await _count(knowledge_backend, "knowledge_entities")
    relations_after_2 = await _count_relations(knowledge_backend)

    assert items_after_2 == items_after_1, (
        f"knowledge_items grew on second run: {items_after_1} -> {items_after_2}"
    )
    assert entities_after_2 == entities_after_1, (
        f"knowledge_entities grew on second run: {entities_after_1} -> {entities_after_2}"
    )
    assert relations_after_2 == relations_after_1, (
        f"knowledge_relations grew on second run: {relations_after_1} -> {relations_after_2}"
    )
