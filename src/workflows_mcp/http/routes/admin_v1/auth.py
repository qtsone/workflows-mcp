from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.security import passwords
from workflows_mcp.security.csrf import CSRF_HEADER_NAME, rotate_csrf_token
from workflows_mcp.security.sessions import (
    SESSION_COOKIE_NAME,
    SESSION_IDLE_TIMEOUT_SECONDS,
    create_admin_session,
    revoke_admin_session,
)

verify_password = passwords.verify_password

router = APIRouter(prefix="/auth")


class LoginRequest(BaseModel):
    password: str


def _session_payload(current: CurrentAdminSession) -> dict[str, object]:
    return {
        "authenticated": True,
        "session": current.session.to_public_dict(),
    }


def _is_secure_request(request: Request) -> bool:
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    forwarded_https = forwarded_proto.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https" or forwarded_https


def _set_session_cookies(response: Response, *, request: Request, session_id: str) -> None:
    secure = _is_secure_request(request)
    expires_at = datetime.now(tz=UTC) + timedelta(seconds=SESSION_IDLE_TIMEOUT_SECONDS)
    expires = format_datetime(expires_at, usegmt=True)
    secure_attr = "; Secure" if secure else ""
    for path in ("/api/admin/v1", "/api/events/v1"):
        response.headers.append(
            "Set-Cookie",
            f"{SESSION_COOKIE_NAME}={session_id}; Path={path}; HttpOnly; SameSite=Lax; "
            f"Max-Age={SESSION_IDLE_TIMEOUT_SECONDS}; Expires={expires}{secure_attr}",
        )


def _clear_session_cookies(response: Response, *, request: Request) -> None:
    secure = _is_secure_request(request)
    for path in ("/api/admin/v1", "/api/events/v1"):
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            path=path,
            httponly=True,
            secure=secure,
            samesite="lax",
        )


@router.post("/login")
async def login(
    body: LoginRequest,
    response: Response,
    request: Request,
    resources: AppResources = Depends(get_resources),
) -> dict[str, object]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        row = conn.execute("SELECT password_hash FROM admin_credentials WHERE id = 1").fetchone()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        password_hash = str(row["password_hash"])
        if not verify_password(body.password, password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        created = create_admin_session(conn)
    finally:
        conn.close()

    _set_session_cookies(response, request=request, session_id=created.session_id)
    response.headers[CSRF_HEADER_NAME] = created.csrf_token
    return {
        "authenticated": True,
        "session": created.session.to_public_dict(),
        "csrf_token": created.csrf_token,
    }


@router.get("/session", openapi_extra={"security": [{"AdminSessionCookie": []}]})
async def session(
    current: CurrentAdminSession = Depends(require_current_admin_session),
) -> dict[str, object]:
    return _session_payload(current)


@router.get("/csrf", openapi_extra={"security": [{"AdminSessionCookie": []}]})
async def csrf(
    response: Response,
    current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> dict[str, object]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        csrf_token = rotate_csrf_token(conn, current.session_id)
    finally:
        conn.close()

    if csrf_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    response.headers[CSRF_HEADER_NAME] = csrf_token
    return {"csrf_token": csrf_token}


@router.post(
    "/logout",
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def logout(
    response: Response,
    request: Request,
    current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> dict[str, bool]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        revoke_admin_session(conn, current.session_id)
    finally:
        conn.close()

    _clear_session_cookies(response, request=request)
    return {"authenticated": False}
