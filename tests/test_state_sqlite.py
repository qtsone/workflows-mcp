#!/usr/bin/env python3
"""Tests for SQLite-based state management workflows.

Tests cover:
1. Task operations (create root, create subtask, update status)
2. Memory operations (get, set, append, merge, delete, keys)
3. Iteration operations (init, advance, checkpoint, complete, reset)
4. Concurrency (parallel writes with WAL mode)
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from workflows_mcp.context import AppContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.tools import execute_workflow

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_context() -> MagicMock:
    """Create mock MCP context with AppContext for testing workflows."""
    registry = WorkflowRegistry()

    # Load built-in templates including agent-memory and agent-iteration
    templates_dir = Path(__file__).parent.parent / "src" / "workflows_mcp" / "templates"
    registry.load_from_directory(templates_dir)

    executor_registry = create_default_registry()
    llm_config_loader = LLMConfigLoader()

    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=IOQueue(),
    )

    mock_ctx = MagicMock()
    mock_ctx.request_context.lifespan_context = app_context
    return mock_ctx


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Create temporary database path."""
    return str(tmp_path / "test_state.db")


@pytest.fixture
def initialized_db(db_path: str) -> str:
    """Create and initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    # Create schema
    conn.execute(
        """CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        parent_id TEXT,
        task TEXT,
        task_type TEXT,
        status TEXT DEFAULT 'pending',
        data JSON,
        created_at TEXT,
        updated_at TEXT
    )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS memory (
        key TEXT PRIMARY KEY,
        value JSON,
        updated_at TEXT
    )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        task_id TEXT,
        caller TEXT,
        action TEXT,
        description TEXT,
        changes JSON,
        parent_id TEXT
    )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS iterations (
        task_id TEXT PRIMARY KEY,
        current INTEGER DEFAULT 0,
        total INTEGER DEFAULT 0,
        cap INTEGER DEFAULT 100,
        started_at TEXT,
        completed_at TEXT,
        checkpoints JSON DEFAULT '[]'
    )"""
    )
    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# Memory Operation Tests
# =============================================================================


class TestMemoryOperations:
    """Test agent-memory workflow operations."""

    @pytest.mark.asyncio
    async def test_memory_set_and_get(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test setting and getting a value."""
        # Set a value
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "set",
                "key": "test.key",
                "value": json.dumps({"foo": "bar"}),
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["value"] == {"foo": "bar"}
        assert result["outputs"]["exists"] is False  # Didn't exist before

        # Get the value back
        result = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "get",
                "key": "test.key",
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["value"] == {"foo": "bar"}
        assert result["outputs"]["exists"] is True

    @pytest.mark.asyncio
    async def test_memory_get_with_default(
        self, mock_context: MagicMock, initialized_db: str
    ) -> None:
        """Test getting a non-existent key with default."""
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "get",
                "key": "nonexistent",
                "default": json.dumps("default_value"),
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["value"] == "default_value"
        assert result["outputs"]["exists"] is False

    @pytest.mark.asyncio
    async def test_memory_append(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test appending to a list."""
        # First append creates list
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "append",
                "key": "findings",
                "value": json.dumps({"issue": "bug1"}),
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["value"] == [{"issue": "bug1"}]

        # Second append adds to list
        result = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "append",
                "key": "findings",
                "value": json.dumps({"issue": "bug2"}),
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["value"] == [{"issue": "bug1"}, {"issue": "bug2"}]

    @pytest.mark.asyncio
    async def test_memory_merge(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test deep merging dicts."""
        # Set initial dict
        await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "set",
                "key": "config",
                "value": json.dumps({"a": 1, "nested": {"x": 10}}),
            },
            ctx=mock_context,
        )

        # Merge additional data
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "merge",
                "key": "config",
                "value": json.dumps({"b": 2, "nested": {"y": 20}}),
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["value"] == {
            "a": 1,
            "b": 2,
            "nested": {"x": 10, "y": 20},
        }

    @pytest.mark.asyncio
    async def test_memory_delete(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test deleting a key."""
        # Set a value
        await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "set",
                "key": "to_delete",
                "value": json.dumps("delete me"),
            },
            ctx=mock_context,
        )

        # Delete it
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "delete",
                "key": "to_delete",
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["value"] == "delete me"
        assert result["outputs"]["exists"] is True

        # Verify deletion
        result = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "get",
                "key": "to_delete",
            },
            ctx=mock_context,
        )
        assert result["outputs"]["exists"] is False

    @pytest.mark.asyncio
    async def test_memory_keys(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test listing keys with pattern matching."""
        # Set multiple keys
        for key in ["cells.root.findings", "cells.root.evidence", "cells.child.data"]:
            await execute_workflow(
                workflow="agent-memory",
                inputs={
                    "path": initialized_db,
                    "op": "set",
                    "key": key,
                    "value": json.dumps(f"value for {key}"),
                },
                ctx=mock_context,
            )

        # List keys with glob pattern
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "keys",
                "key": "cells.root.*",
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert set(result["outputs"]["keys"]) == {
            "cells.root.findings",
            "cells.root.evidence",
        }


# =============================================================================
# Iteration Operation Tests
# =============================================================================


class TestIterationOperations:
    """Test agent-iteration workflow operations."""

    @pytest.mark.asyncio
    async def test_iteration_init_and_advance(
        self, mock_context: MagicMock, initialized_db: str
    ) -> None:
        """Test initializing and advancing iteration counter."""
        # Initialize
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-001",
                "op": "init",
                "total": 10,
                "cap": 20,
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["current"] == 0
        assert result["outputs"]["total"] == 10
        assert result["outputs"]["cap"] == 20

        # Advance
        result = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-001",
                "op": "advance",
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["current"] == 1
        assert result["outputs"]["progress_pct"] == 10.0
        assert "1/10" in result["outputs"]["progress_str"]

    @pytest.mark.asyncio
    async def test_iteration_unbounded(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test unbounded iteration (no total)."""
        # Initialize without total
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-unbounded",
                "op": "init",
                "cap": 50,
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["total"] == 0

        # Advance
        result = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-unbounded",
                "op": "advance",
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["current"] == 1
        assert "cap: 50" in result["outputs"]["progress_str"]

    @pytest.mark.asyncio
    async def test_iteration_checkpoint(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test saving checkpoint data."""
        # Initialize and advance
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-checkpoint",
                "op": "init",
                "total": 5,
            },
            ctx=mock_context,
        )
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-checkpoint",
                "op": "advance",
            },
            ctx=mock_context,
        )

        # Save checkpoint
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-checkpoint",
                "op": "checkpoint",
                "checkpoint_data": json.dumps({"step": 1, "data": "important"}),
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["checkpoint"] == {"step": 1, "data": "important"}

    @pytest.mark.asyncio
    async def test_iteration_complete(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test completing iteration."""
        # Initialize
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-complete",
                "op": "init",
                "total": 2,
            },
            ctx=mock_context,
        )

        # Advance twice
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-complete",
                "op": "advance",
            },
            ctx=mock_context,
        )
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-complete",
                "op": "advance",
            },
            ctx=mock_context,
        )

        # Complete
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-complete",
                "op": "complete",
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["current"] == 2
        assert result["outputs"]["progress_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_iteration_reset(self, mock_context: MagicMock, initialized_db: str) -> None:
        """Test resetting iteration counter."""
        # Initialize and advance
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-reset",
                "op": "init",
                "total": 10,
            },
            ctx=mock_context,
        )
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-reset",
                "op": "advance",
            },
            ctx=mock_context,
        )

        # Reset
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-reset",
                "op": "reset",
                "total": 5,
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["current"] == 0

    @pytest.mark.asyncio
    async def test_iteration_cap_reached(
        self, mock_context: MagicMock, initialized_db: str
    ) -> None:
        """Test is_capped flag when cap is reached."""
        # Initialize with low cap
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-cap",
                "op": "init",
                "total": 100,
                "cap": 2,
            },
            ctx=mock_context,
        )

        # Advance to cap
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-cap",
                "op": "advance",
            },
            ctx=mock_context,
        )
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "task-cap",
                "op": "advance",
            },
            ctx=mock_context,
        )
        assert result["status"] == "success"
        assert result["outputs"]["current"] == 2
        assert result["outputs"]["is_capped"] is True


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Test concurrent access with WAL mode."""

    @pytest.mark.asyncio
    async def test_parallel_memory_appends(
        self, mock_context: MagicMock, initialized_db: str
    ) -> None:
        """Test that parallel append operations don't lose data.

        Uses BEGIN IMMEDIATE transaction to ensure atomic read-modify-write.
        """
        # Launch 10 parallel append operations
        tasks = [
            execute_workflow(
                workflow="agent-memory",
                inputs={
                    "path": initialized_db,
                    "op": "append",
                    "key": "parallel_findings",
                    "value": json.dumps({"item": i}),
                },
                ctx=mock_context,
            )
            for i in range(10)
        ]

        results: list[dict[str, Any]] = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r["status"] == "success" for r in results)

        # Get final value
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "get",
                "key": "parallel_findings",
            },
            ctx=mock_context,
        )

        # All 10 items should be present (no lost updates with IMMEDIATE lock)
        items = result["outputs"]["value"]
        assert len(items) == 10, f"Expected 10 items, got {len(items)}"
        item_ids = {item["item"] for item in items}
        assert item_ids == set(range(10))

    @pytest.mark.asyncio
    async def test_parallel_iteration_advances(
        self, mock_context: MagicMock, initialized_db: str
    ) -> None:
        """Test that parallel advance operations all count correctly."""
        # Initialize
        await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "parallel-task",
                "op": "init",
                "total": 100,
            },
            ctx=mock_context,
        )

        # Launch 10 parallel advances
        tasks = [
            execute_workflow(
                workflow="agent-iteration",
                inputs={
                    "path": initialized_db,
                    "task_id": "parallel-task",
                    "op": "advance",
                },
                ctx=mock_context,
            )
            for _ in range(10)
        ]

        results: list[dict[str, Any]] = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r["status"] == "success" for r in results)

        # Current should be exactly 10 (no lost updates)
        current_values = [r["outputs"]["current"] for r in results]
        assert max(current_values) == 10

        # Verify in database
        conn = sqlite3.connect(initialized_db)
        row = conn.execute(
            "SELECT current FROM iterations WHERE task_id=?", ("parallel-task",)
        ).fetchone()
        conn.close()
        assert row[0] == 10


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in state workflows."""

    @pytest.mark.asyncio
    async def test_iteration_advance_without_init(
        self, mock_context: MagicMock, initialized_db: str
    ) -> None:
        """Test advancing iteration without initializing first.

        When task_id doesn't exist in iterations table, the Python script
        raises a ValueError. This results in either:
        1. The workflow failing (status == "failure")
        2. Or output parsing failing (outputs incomplete)
        """
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-iteration",
            inputs={
                "path": initialized_db,
                "task_id": "not-initialized",
                "op": "advance",
            },
            ctx=mock_context,
        )
        # Script raises ValueError for missing task, which should cause
        # either workflow failure or output parsing failure
        assert result["status"] == "failure" or "current" not in result.get("outputs", {})

    @pytest.mark.asyncio
    async def test_memory_merge_into_non_dict(
        self, mock_context: MagicMock, initialized_db: str
    ) -> None:
        """Test merging into a non-dict value.

        When attempting to merge a dict into a non-dict value, the Python
        script raises a ValueError. This results in either:
        1. The workflow failing (status == "failure")
        2. Or output parsing failing (outputs incomplete)
        """
        # Set a string value
        await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "set",
                "key": "string_key",
                "value": json.dumps("just a string"),
            },
            ctx=mock_context,
        )

        # Try to merge (should fail)
        result: dict[str, Any] = await execute_workflow(
            workflow="agent-memory",
            inputs={
                "path": initialized_db,
                "op": "merge",
                "key": "string_key",
                "value": json.dumps({"new": "data"}),
            },
            ctx=mock_context,
        )
        # Script raises ValueError for non-dict merge, which should cause
        # either workflow failure or output parsing failure
        assert result["status"] == "failure" or result.get("outputs", {}).get("value") != {
            "new": "data"
        }
