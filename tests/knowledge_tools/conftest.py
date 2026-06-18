"""Shared fixtures for the knowledge/memory tool test package."""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _patch_memory_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch memory config resolution so ephemeral-backend tests bypass the
    production config guard (MEMORY_BACKEND_UNAVAILABLE) without touching any
    production code path."""
    sentinel = object()

    async def _noop_ensure_schema(_backend: Any) -> None:
        return

    monkeypatch.setattr(
        "workflows_mcp.tools_memory.memory_connection_config_from_metadata",
        lambda _app_ctx: None,
    )
    monkeypatch.setattr(
        "workflows_mcp.tools_memory.memory_connection_config_from_env",
        lambda: sentinel,
    )
    monkeypatch.setattr(
        "workflows_mcp.memory.knowledge.schema.ensure_schema",
        _noop_ensure_schema,
    )


@pytest.fixture
def mock_ctx() -> MagicMock:
    ctx = MagicMock()
    app_ctx = MagicMock()
    app_ctx.memory_backend = None
    app_ctx.memory_backend_lock = None
    ctx.request_context.lifespan_context = app_ctx
    exec_context = MagicMock()
    exec_context.user_string_id = None
    app_ctx.create_execution_context.return_value = exec_context
    app_ctx.get_user_context.return_value = (uuid.UUID(int=0), "test-user", "OS_USER")
    app_ctx.get_active_context.return_value = None
    return ctx


@pytest.fixture
def workspace_tmp() -> Iterator[Path]:
    """Create a temporary directory inside the workspace root (os.getcwd()).

    Required for scan tests because path safety validation rejects root
    values outside the workspace root.
    """
    workspace = Path(os.getcwd()).resolve()
    with tempfile.TemporaryDirectory(dir=workspace, prefix=".scan_test_") as td:
        yield Path(td)
