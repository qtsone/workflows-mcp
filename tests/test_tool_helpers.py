"""Unit tests for the shared MCP tool helpers.

Covers the response/scope/auth-scope contract owned by
``workflows_mcp.tool_helpers`` and consumed by both tool modules and the
MCP auth middleware.
"""

import json
from types import SimpleNamespace
from typing import Any, cast

from mcp.types import CallToolResult, TextContent

from workflows_mcp.context import AppContextType
from workflows_mcp.tool_helpers import (
    AUTH_SCOPE_KEY,
    get_request_scope,
    json_response,
)


def _ctx_with_scope(scope: Any) -> AppContextType:
    """Build a minimal stand-in for the MCP tool context.

    ``get_request_scope`` only walks ``ctx.request_context.request.scope`` via
    ``getattr``, so a nested namespace is enough — no live MCP session needed.
    """
    request = SimpleNamespace(scope=scope) if scope is not None else None
    return cast(AppContextType, SimpleNamespace(request_context=SimpleNamespace(request=request)))


def test_json_response_emits_compact_text_and_structured_content() -> None:
    data = {"status": "success", "count": 2, "items": ["a", "b"]}

    result = json_response(data)

    assert isinstance(result, CallToolResult)
    assert result.structuredContent == data
    assert len(result.content) == 1
    text_block = result.content[0]
    assert isinstance(text_block, TextContent)
    # Compact separators, no indentation — preserves the TASK-057 fix.
    assert text_block.text == json.dumps(data, separators=(",", ":"))
    assert " " not in text_block.text


def test_json_response_preserves_nested_structured_content() -> None:
    data: dict[str, Any] = {"nested": {"k": 1}, "list": [1, 2, 3]}

    result = json_response(data)

    assert result.structuredContent == data


def test_get_request_scope_returns_dict_scope() -> None:
    scope = {"type": "http", AUTH_SCOPE_KEY: object()}

    assert get_request_scope(_ctx_with_scope(scope)) is scope


def test_get_request_scope_none_when_request_missing() -> None:
    assert get_request_scope(_ctx_with_scope(None)) is None


def test_get_request_scope_none_when_scope_not_a_dict() -> None:
    assert get_request_scope(_ctx_with_scope("not-a-dict")) is None


def test_auth_scope_key_is_single_shared_constant() -> None:
    """The producer (auth middleware) and both consumers share one constant."""
    from workflows_mcp import tools, tools_memory
    from workflows_mcp.http import auth_mcp

    assert AUTH_SCOPE_KEY == "workflows_mcp.auth_context"
    assert tools.AUTH_SCOPE_KEY is AUTH_SCOPE_KEY
    assert tools_memory.AUTH_SCOPE_KEY is AUTH_SCOPE_KEY
    assert auth_mcp.AUTH_SCOPE_KEY is AUTH_SCOPE_KEY
