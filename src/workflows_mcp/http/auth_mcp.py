from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from workflows_mcp.auth import generate_request_id
from workflows_mcp.context import SessionProjectContext
from workflows_mcp.http_models import ErrorEnvelope, ReadinessState
from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository
from workflows_mcp.metadata.repos.tokens_repo import (
    RevokedTokenError,
    SQLiteTokensRepository,
    UnknownTokenError,
)

_AUTH_SCOPE_KEY = "workflows_mcp.auth_context"

# Readiness blockers that do not compromise MCP token authentication,
# project scoping, or transport-session ownership guarantees.
_SOFT_MCP_READINESS_BLOCKERS = {
    "llm_config",
    "knowledge_schema_incompatible",
    "postgresql_dsn_missing",
    "postgresql_connectivity",
    "postgresql_version_unsupported",
    "pgvector_missing",
    "postgresql_readwrite_failed",
}


def _is_hard_mcp_readiness_blocker(blocker: str) -> bool:
    return blocker not in _SOFT_MCP_READINESS_BLOCKERS


@dataclass(frozen=True)
class MCPAuthContext:
    token_id: str
    projects: tuple[SessionProjectContext, ...]


def _to_session_project_context(project: Any, *, source: str) -> SessionProjectContext:
    return SessionProjectContext(
        project_id=project.id,
        slug=project.slug,
        palace=project.palace,
        default_wing=project.default_wing,
        default_room=project.default_room,
        source=source,
        fs_root=project.fs_root,
        fs_allowlist=tuple(project.fs_allowlist),
    )


def _extract_header(scope: Scope, header_name: bytes) -> str | None:
    headers = dict(scope.get("headers", []))
    raw = headers.get(header_name)
    if raw is None:
        return None
    value = raw.decode("latin-1", errors="replace").strip()
    return value or None


def _resolve_sqlite_auth_context(scope: Scope, token: str) -> MCPAuthContext | None:
    app = scope.get("app")
    resources = getattr(getattr(app, "state", None), "resources", None)
    conn = getattr(resources, "metadata_db_conn", None)
    if conn is None:
        return None

    token_repo = SQLiteTokensRepository(conn)
    project_repo = SQLiteProjectsRepository(conn)

    try:
        token_record = token_repo.resolve(token)
    except (UnknownTokenError, RevokedTokenError):
        return None
    except sqlite3.Error:
        return None

    try:
        resolved_projects: list[SessionProjectContext] = []
        if token_record.project_ids:
            for project_id in token_record.project_ids:
                project = project_repo.get_by_id(project_id)
                if project is None:
                    continue
                resolved_projects.append(_to_session_project_context(project, source="token_bound"))
        else:
            resolved_projects = [
                _to_session_project_context(project, source="token_unbound")
                for project in project_repo.list_all()
            ]

        token_repo.mark_last_used(token_record.id)
        return MCPAuthContext(token_id=token_record.id, projects=tuple(resolved_projects))
    except (AttributeError, TypeError, ValueError, sqlite3.Error):
        return None


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload = ErrorEnvelope.for_code(
        code=code,
        message=message,
        details=details,
        request_id=generate_request_id(),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _extract_bearer_token(scope: Scope) -> str | None:
    headers = dict(scope.get("headers", []))
    raw = headers.get(b"authorization")
    if raw is None:
        return None
    value = raw.decode("latin-1", errors="replace")
    parts = value.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token_value = parts
    if scheme.lower() != "bearer":
        return None
    token = token_value.strip()
    return token if token else None


class MCPAuthMiddleware:
    """ASGI middleware for MCP bearer-auth and readiness gating."""

    def __init__(self, app: ASGIApp, *, readiness_service: Any) -> None:
        self.app = app
        self.readiness_service = readiness_service

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        token = _extract_bearer_token(scope)
        authenticated = False
        if token is not None:
            sqlite_ctx = _resolve_sqlite_auth_context(scope, token)
            if sqlite_ctx is not None:
                if sqlite_ctx.token_id:
                    scope[_AUTH_SCOPE_KEY] = sqlite_ctx
                    authenticated = True
                else:
                    authenticated = False

        if not authenticated:
            await _error_response(
                status_code=401,
                code="UNAUTHORIZED",
                message="Missing or invalid bearer token.",
            )(scope, receive, send)
            return

        report = await self.readiness_service.evaluate()
        hard_blockers = [
            blocker for blocker in report.blockers if _is_hard_mcp_readiness_blocker(blocker)
        ]
        should_block_for_readiness = report.state != ReadinessState.READY and (
            not report.blockers or bool(hard_blockers)
        )
        if should_block_for_readiness:
            await _error_response(
                status_code=409,
                code="CONFIG_REQUIRED",
                message=(
                    "Service is not ready. Complete admin setup in "
                    "/api/admin/v1 (for example: database settings, secrets, and LLM config)."
                ),
                details={
                    "readiness_state": str(report.state),
                    "missing": hard_blockers if hard_blockers else list(report.blockers),
                },
            )(scope, receive, send)
            return

        await self.app(scope, receive, send)

        if scope.get("method") == "DELETE":
            session_header = _extract_header(scope, b"mcp-session-id")
            if session_header:
                auth_ctx = scope.get(_AUTH_SCOPE_KEY)
                requester_token_id = (
                    auth_ctx.token_id if isinstance(auth_ctx, MCPAuthContext) else None
                )
                app = scope.get("app")
                resources = getattr(getattr(app, "state", None), "resources", None)
                app_context = getattr(resources, "app_context", None)
                if app_context is not None:
                    clearer = getattr(app_context, "clear_session_state_by_transport_id", None)
                    if callable(clearer):
                        clearer(session_header, requester_token_id=requester_token_id)
