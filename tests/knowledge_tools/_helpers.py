"""Shared tool bindings and mock helpers for the knowledge/memory tool tests.

Registers the memory tools on the shared FastMCP server once and exposes the
``memory`` / ``onboard`` / ``sync`` tool functions plus a backend-mock factory,
so the per-concern ``test_*`` modules drive the same registered tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from workflows_mcp.server import mcp as _mcp_server
from workflows_mcp.tools_memory import register_memory_tools

register_memory_tools(_mcp_server)


def _get_tool_fn(name: str) -> Any:
    tool_manager = _mcp_server._tool_manager
    tool = tool_manager._tools.get(name)
    if tool is None:
        raise ValueError(f"Tool {name!r} not registered")
    return tool.fn


memory = _get_tool_fn("memory")
onboard = _get_tool_fn("onboard")
sync = _get_tool_fn("sync")


def _make_backend_mock() -> MagicMock:
    backend = MagicMock()
    backend.connect = AsyncMock()
    backend.disconnect = AsyncMock()
    return backend
