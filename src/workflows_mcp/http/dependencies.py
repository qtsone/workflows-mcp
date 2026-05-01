"""FastAPI dependency helpers for HTTP resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fastapi import Depends, Header, HTTPException, Request, status

from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.security.csrf import CSRF_HEADER_NAME, verify_session_csrf
from workflows_mcp.security.sessions import (
    SESSION_COOKIE_NAME,
    AdminSession,
    get_admin_session,
)

from .lifespan import AppResources


def get_resources(request: Request) -> AppResources:
    """Return shared app resources attached by server.build_app()."""
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return cast(AppResources, resources)


@dataclass(frozen=True)
class CurrentAdminSession:
    session_id: str
    session: AdminSession


def require_current_admin_session(
    request: Request,
    resources: AppResources = Depends(get_resources),
) -> CurrentAdminSession:
    authorization = request.headers.get("Authorization", "")
    if authorization.strip().lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bearer auth not allowed")

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        admin_session = get_admin_session(conn, session_id, refresh_idle=True)
    finally:
        conn.close()

    if admin_session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    return CurrentAdminSession(session_id=session_id, session=admin_session)


def require_admin_csrf(
    request: Request,
    current: CurrentAdminSession = Depends(require_current_admin_session),
    csrf_token: str | None = Header(default=None, alias=CSRF_HEADER_NAME),
    resources: AppResources = Depends(get_resources),
) -> CurrentAdminSession:
    if not csrf_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token required")

    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        valid_csrf = verify_session_csrf(conn, current.session_id, csrf_token)
    finally:
        conn.close()

    if not valid_csrf:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")

    return current
