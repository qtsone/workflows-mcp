"""Shared helpers for the MCP tool surface.

Owns the response/scope/auth-scope contract used by both tool modules
(`tools`, `tools_memory`) and produced by the MCP auth middleware
(`http.auth_mcp`), so the contract has a single home.
"""

import json
from typing import Any

from mcp.types import CallToolResult, TextContent

from .context import AppContextType

# Key under which the MCP auth middleware stashes the per-request auth context
# on the ASGI scope. The middleware is the producer; tool modules are consumers.
AUTH_SCOPE_KEY = "workflows_mcp.auth_context"


def json_response(data: dict[str, Any]) -> CallToolResult:
    """Build a CallToolResult with both compact text and structured content.

    Returns a CallToolResult that the SDK passes through unchanged:
    - content: compact JSON string in TextContent (preserves TASK-057 fix)
    - structuredContent: raw dict for clients that support it (TASK-062)

    This avoids the SDK's default indent=2 serialization while also providing
    parsed JSON objects via structuredContent for modern MCP clients.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, separators=(",", ":")))],
        structuredContent=data,
    )


def get_request_scope(ctx: AppContextType) -> dict[str, Any] | None:
    """Best-effort access to HTTP request scope from MCP tool context."""
    request = getattr(ctx.request_context, "request", None)
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        return scope
    return None
