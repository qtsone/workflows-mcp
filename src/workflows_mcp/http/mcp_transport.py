from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from workflows_mcp.context import AppContext
from workflows_mcp.tools import register_workflow_tools
from workflows_mcp.tools_memory import register_memory_tools


def _is_enabled_env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _register_tools(mounted_mcp: FastMCP) -> None:
    register_workflow_tools(mounted_mcp)

    oss_mode_enabled = _is_enabled_env_flag("WORKFLOWS_OSS_MODE", default=True)
    project_tools_enabled = _is_enabled_env_flag("WORKFLOWS_ENABLE_PROJECT_TOOLS", default=True)
    if "WORKFLOWS_ENABLE_PROJECT_TOOLS" not in os.environ and (
        "WORKFLOWS_ENABLE_TEMP_PROJECT_TOOLS" in os.environ
    ):
        project_tools_enabled = _is_enabled_env_flag(
            "WORKFLOWS_ENABLE_TEMP_PROJECT_TOOLS",
            default=True,
        )

    register_memory_tools(
        mounted_mcp,
        enable_project_tools=oss_mode_enabled and project_tools_enabled,
    )


class MCPStreamableHTTPMount:
    """Lifecycle-managed mounted Streamable HTTP transport."""

    def __init__(self, server: FastMCP, app: ASGIApp) -> None:
        self.asgi_app = self
        self._app = app
        self._server = server
        self._run_cm: Any | None = None
        self._lock = anyio.Lock()

    async def startup(self) -> None:
        await self._ensure_started()

    async def _ensure_started(self) -> None:
        if self._run_cm is not None:
            return
        async with self._lock:
            if self._run_cm is not None:
                return
            run_cm = self._server.session_manager.run()
            await run_cm.__aenter__()
            self._run_cm = run_cm

    async def shutdown(self) -> None:
        if self._run_cm is None:
            return
        run_cm = self._run_cm
        self._run_cm = None
        await run_cm.__aexit__(None, None, None)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            return
        # Fallback for hosts that do not drive ASGI lifespan (e.g., in-process
        # test transports). In normal server runtime startup() runs this first.
        await self._ensure_started()
        await self._app(scope, receive, send)


def build_mcp_streamable_http_mount(
    *,
    app_context_factory: Callable[[], AppContext],
) -> MCPStreamableHTTPMount:
    @asynccontextmanager
    async def _shared_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
        yield app_context_factory()

    mounted_mcp = FastMCP(
        "workflows_mcp_http",
        streamable_http_path="/",
        stateless_http=False,
        json_response=True,
        lifespan=_shared_lifespan,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1", "localhost", "testserver"],
        ),
    )
    _register_tools(mounted_mcp)
    streamable_app = mounted_mcp.streamable_http_app()
    return MCPStreamableHTTPMount(mounted_mcp, streamable_app)


def resolve_app_context_from_fastapi_state(fastapi_app: object) -> AppContext:
    resources = getattr(getattr(fastapi_app, "state", None), "resources", None)
    app_context = getattr(resources, "app_context", None)
    if not isinstance(app_context, AppContext):
        raise RuntimeError("Shared AppResources missing from FastAPI app state.")
    return app_context
