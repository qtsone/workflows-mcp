from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import TokenStore, generate_request_id
from .http.auth_mcp import MCPAuthMiddleware
from .http.mcp_transport import (
    MCPStreamableHTTPMount,
    build_mcp_streamable_http_mount,
    resolve_app_context_from_fastapi_state,
)
from .http.routes.admin_v1 import router as admin_v1_router
from .http.routes.events_v1 import router as events_v1_router
from .http.routes.public_v1 import router as public_v1_router
from .http.static import install_admin_static
from .http_models import ErrorEnvelope, ReadinessState
from .security.csrf import CSRF_HEADER_NAME
from .security.sessions import SESSION_COOKIE_NAME


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a ``JSONResponse`` using the stable ``ErrorEnvelope`` shape.

    Every non-2xx response on a protected route MUST go through this helper
    so the contract ``{"error": {"code": ..., "message": ..., "request_id": ...}}``
    is enforced uniformly.
    """
    request_id = generate_request_id()
    payload = ErrorEnvelope.for_code(
        code=code,
        message=message,
        details=details,
        request_id=request_id,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _install_exception_handlers(app: FastAPI) -> None:
    """Register application-level exception handlers that emit the stable envelope.

    Overrides FastAPI's default ``RequestValidationError`` and ``HTTPException``
    handlers so all non-2xx responses on protected routes use the stable
    ``ErrorEnvelope`` shape instead of raw ``{"detail": ...}`` payloads.
    """

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(
        _: Request, exc: HTTPException
    ) -> JSONResponse:
        """Convert any HTTPException into the stable ErrorEnvelope shape.

        The ``detail`` field may already contain a structured dict (from the
        auth guard) or a plain string; both cases are normalised here.
        """
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            # Auth guard already built a structured error dict — use it directly.
            code: str = detail["code"]
            message: str = detail.get("message", str(exc.status_code))
            existing_request_id: str | None = detail.get("request_id")
        else:
            code = _http_status_to_code(exc.status_code)
            message = str(detail) if detail else _http_status_to_code(exc.status_code)
            existing_request_id = None

        request_id = existing_request_id or generate_request_id()
        payload = ErrorEnvelope.for_code(
            code=code,
            message=message,
            details=None,
            request_id=request_id,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def _handle_starlette_http_exception(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Catch routing-layer errors (405 Method Not Allowed, 404 Not Found, etc.).

        Starlette raises its own ``HTTPException`` subclass for these — it is
        processed before FastAPI's middleware chain and therefore bypasses the
        ``HTTPException`` handler registered above.  Registering a separate
        handler for the Starlette base class intercepts those responses and
        normalises them into the stable ``ErrorEnvelope``.
        """
        code = _http_status_to_code(exc.status_code)
        message = str(exc.detail) if exc.detail else code
        payload = ErrorEnvelope.for_code(
            code=code,
            message=message,
            details=None,
            request_id=generate_request_id(),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


def _http_status_to_code(status_code: int) -> str:
    """Map common HTTP status codes to SCREAMING_SNAKE error codes."""
    _map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_FAILED",
        500: "INTERNAL_SERVER_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    return _map.get(status_code, f"HTTP_{status_code}")


# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

#: Default maximum request body size (1 MiB).  Override via
#: ``WORKFLOWS_MAX_BODY_BYTES`` environment variable.
_DEFAULT_MAX_BODY_BYTES: int = 1 * 1024 * 1024

#: General rate-limit: maximum requests per ``_RATE_WINDOW_SECONDS`` window.
_GENERAL_RATE_LIMIT: int = 100

#: Stricter rate-limit applied to MCP protected routes.
_MCP_RATE_LIMIT: int = 20

#: Stricter rate-limit applied to admin login endpoint.
_LOGIN_RATE_LIMIT: int = _MCP_RATE_LIMIT

#: Sliding-window duration in seconds.
_RATE_WINDOW_SECONDS: float = 60.0


class BodyLimitMiddleware:
    """Reject requests whose body exceeds *max_bytes*.

    Two enforcement paths:
    1. **Fast path**: if ``Content-Length`` is present and exceeds the limit the
       request is rejected immediately without reading any body bytes.
    2. **Streaming path**: if ``Content-Length`` is absent (chunked / streaming
       transfer), the ``receive`` callable is wrapped so bytes are counted as
       they arrive; once the running total exceeds *max_bytes* a 413 is returned
       and the body stream is drained (not forwarded to the application).

    Both paths respond with the stable ``ErrorEnvelope`` shape and the rejection
    always happens before any route logic sees the request body.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers: dict[bytes, bytes] = dict(scope.get("headers", []))

        # Fast path: Content-Length header present.
        raw_cl = headers.get(b"content-length")
        if raw_cl is not None:
            try:
                content_length = int(raw_cl)
            except ValueError:
                content_length = 0
            if content_length > self.max_bytes:
                response = _error_response(
                    status_code=413,
                    code="REQUEST_TOO_LARGE",
                    message="Request body exceeds configured limit.",
                    details={"max_bytes": self.max_bytes},
                )
                await response(scope, receive, send)
                return

        # Streaming path: pre-buffer the entire body from receive() before
        # forwarding to the app.  This ensures oversized streaming bodies
        # (sent without Content-Length) are rejected deterministically before
        # any route logic sees the request body.
        max_bytes = self.max_bytes
        body = b""
        more_body = True
        while more_body:
            raw_message = await receive()
            message: dict[str, Any] = dict(raw_message)
            if message.get("type") == "http.request":
                chunk: bytes = message.get("body", b"")
                body += chunk
                more_body = message.get("more_body", False)
                if len(body) > max_bytes:
                    response = _error_response(
                        status_code=413,
                        code="REQUEST_TOO_LARGE",
                        message="Request body exceeds configured limit.",
                        details={"max_bytes": max_bytes},
                    )
                    await response(scope, receive, send)
                    return
            else:
                # Disconnect or other non-request message — pass through.
                more_body = False

        # Body is within limits; replay it as a single receive() call.
        body_message: dict[str, Any] = {"type": "http.request", "body": body, "more_body": False}
        consumed = False

        async def _replay_receive() -> dict[str, Any]:
            nonlocal consumed
            if not consumed:
                consumed = True
                return body_message
            # Subsequent receive() calls (e.g. for disconnect) go to the real receive.
            return dict(await receive())

        await self.app(scope, _replay_receive, send)


class SlidingWindowRateLimiter:
    """In-process sliding-window rate limiter keyed by client IP.

    Uses ``collections.deque`` per key to store request timestamps within the
    current window, evicting expired entries on every access.  The ``deque``
    is bounded to ``limit`` entries so memory is capped regardless of traffic.

    Thread-safety: designed for single-process async use (asyncio).  No locks
    are required because the GIL serialises the deque operations.
    """

    def __init__(self, *, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._buckets: dict[str, deque[float]] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.setdefault(key, deque(maxlen=self.limit + 1))
        # Evict timestamps outside the current window.
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


class RateLimitMiddleware:
    """Enforce per-IP sliding-window rate limits on protected routes.

    Rate limiting is applied only to routes that require authentication
    (``/mcp`` and ``/api/admin/v1/auth/login``). Public health/readiness endpoints
    (``/health``, ``/ready``) and documentation endpoints are deliberately
    exempt so liveness probes and tooling are never blocked.

    Two limits are supported:
    - ``login_limiter``: applied to ``/api/admin/v1/auth/login``.
    - ``mcp_limiter``: applied to ``/mcp``.

    The client key is derived from the ``X-Forwarded-For`` header when
    present (first address), falling back to the ASGI ``client`` tuple.
    """

    #: Only these path prefixes are subject to rate limiting.
    _PROTECTED_PREFIXES: tuple[str, ...] = (
        "/mcp",
        "/api/admin/v1/auth/login",
    )

    def __init__(
        self,
        app: ASGIApp,
        *,
        mcp_limiter: SlidingWindowRateLimiter,
        login_limiter: SlidingWindowRateLimiter,
    ) -> None:
        self.app = app
        self.mcp_limiter = mcp_limiter
        self.login_limiter = login_limiter

    def _client_key(self, scope: Scope) -> str:
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        xff = headers.get(b"x-forwarded-for")
        if xff:
            return xff.decode("latin-1", errors="replace").split(",")[0].strip()
        client = scope.get("client")
        if client:
            return str(client[0])
        return "unknown"

    def _is_protected(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self._PROTECTED_PREFIXES)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path: str = scope.get("path", "")
            if self._is_protected(path):
                key = self._client_key(scope)
                limiter = (
                    self.login_limiter
                    if path == "/api/admin/v1/auth/login"
                    else self.mcp_limiter
                )
                if not limiter.is_allowed(key):
                    response = _error_response(
                        status_code=429,
                        code="RATE_LIMITED",
                        message="Too many requests. Please slow down.",
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


class _MCPPathCanonicalizationMiddleware:
    """Normalize exact ``/mcp`` requests to ``/mcp/`` for mounted transport."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            # Mounted ASGI apps are resolved on a path-prefix basis and exact
            # mount roots can otherwise produce 405 for non-GET methods.
            scope = dict(scope)
            scope["path"] = "/mcp/"
        await self.app(scope, receive, send)


def _install_security_middleware(app: FastAPI) -> None:
    """Add CORS, body-limit, and rate-limit middleware to *app*.

    Middleware is added in reverse application order — the last ``add_middleware``
    call wraps outermost.  Desired order (outermost → innermost):

    1. CORS (responds to preflight before any logic)
    2. Rate limiting (reject early, avoids unnecessary processing)
    3. Body limit (cheap header-only check before body is consumed)
    4. Route handlers
    """
    # CORS — deny-by-default; populate only when WORKFLOWS_CORS_ORIGINS is set.
    raw_origins = os.getenv("WORKFLOWS_CORS_ORIGINS", "").strip()
    allow_origins: list[str] = (
        [o.strip() for o in raw_origins.split(",") if o.strip()] if raw_origins else []
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", CSRF_HEADER_NAME],
    )

    # Body limit — checked before route logic on every HTTP request.
    try:
        max_body = int(os.getenv("WORKFLOWS_MAX_BODY_BYTES", str(_DEFAULT_MAX_BODY_BYTES)))
    except ValueError:
        max_body = _DEFAULT_MAX_BODY_BYTES
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body)

    # Rate limiting — MCP routes and login have dedicated caps.
    try:
        general_limit = int(os.getenv("WORKFLOWS_RATE_LIMIT", str(_GENERAL_RATE_LIMIT)))
    except ValueError:
        general_limit = _GENERAL_RATE_LIMIT
    try:
        mcp_limit = int(os.getenv("WORKFLOWS_MCP_RATE_LIMIT", str(_MCP_RATE_LIMIT)))
    except ValueError:
        mcp_limit = _MCP_RATE_LIMIT
    try:
        login_limit = int(os.getenv("WORKFLOWS_LOGIN_RATE_LIMIT", str(_LOGIN_RATE_LIMIT)))
    except ValueError:
        login_limit = _LOGIN_RATE_LIMIT

    # Keep parsed to preserve env validation consistency for shared rate-limit knobs.
    _ = general_limit
    mcp_limiter = SlidingWindowRateLimiter(limit=mcp_limit, window=_RATE_WINDOW_SECONDS)
    login_limiter = SlidingWindowRateLimiter(limit=login_limit, window=_RATE_WINDOW_SECONDS)
    app.add_middleware(
        RateLimitMiddleware,
        mcp_limiter=mcp_limiter,
        login_limiter=login_limiter,
    )

    app.add_middleware(_MCPPathCanonicalizationMiddleware)


def _make_auth_guard(token_store: TokenStore):  # type: ignore[no-untyped-def]
    """Return a FastAPI dependency that validates bearer tokens against *token_store*.

    Raises ``HTTPException(401)`` on failure; the application-level
    ``HTTPException`` handler normalises this into the stable ``ErrorEnvelope``.
    """

    def _auth_guard(authorization: str | None = Header(default=None)) -> None:
        token: str | None = None
        if authorization is not None and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):]
        if token is None or not token_store.validate(token):
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "UNAUTHORIZED",
                    "message": "Missing or invalid bearer token.",
                    "request_id": generate_request_id(),
                },
            )

    return _auth_guard


def _build_openapi_with_bearer_auth(app: FastAPI) -> dict[str, Any]:
    """Generate OpenAPI schema with authentication security schemes injected.

    Overrides the default FastAPI openapi() method so the /docs UI shows
    the lock icon and all protected routes carry a security declaration.
    """
    from fastapi.openapi.utils import get_openapi

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    security_schemes = schema.setdefault("components", {}).setdefault("securitySchemes", {})

    # Keep bearer auth for MCP protected endpoints.
    security_schemes["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
    }

    # Session cookie required for UI session-protected routes.
    security_schemes["AdminSessionCookie"] = {
        "type": "apiKey",
        "in": "cookie",
        "name": SESSION_COOKIE_NAME,
    }

    # CSRF token header required for mutating admin operations.
    security_schemes["CsrfToken"] = {
        "type": "apiKey",
        "in": "header",
        "name": CSRF_HEADER_NAME,
    }

    app.openapi_schema = schema
    return schema


def create_app(
    *,
    readiness_service: Any,
    token_store: TokenStore,
    frontend_static_dir: Path | None = None,
    require_frontend_assets: bool = False,
    lifespan: Any = None,
) -> FastAPI:
    app = FastAPI(
        title="workflows-mcp",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Override OpenAPI schema generator to inject BearerAuth security scheme.
    app.openapi = lambda: _build_openapi_with_bearer_auth(app)  # type: ignore[method-assign]

    _install_exception_handlers(app)
    _install_security_middleware(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        report = await readiness_service.evaluate()
        knowledge_ready = report.state == ReadinessState.READY
        status_code = 200 if knowledge_ready else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "state": str(report.state),
                "blockers": report.blockers,
                # Control-plane status: route reached and app is serving.
                "server_ready": True,
                # Knowledge status: backing knowledge dependencies are ready.
                "knowledge_ready": knowledge_ready,
            },
        )

    app.include_router(public_v1_router)
    app.include_router(admin_v1_router)
    app.include_router(events_v1_router)

    _mount_mcp_transport(
        app,
        readiness_service=readiness_service,
    )

    install_admin_static(
        app,
        static_dir=frontend_static_dir,
        require_assets=require_frontend_assets,
    )

    return app


# ---------------------------------------------------------------------------
# MCP-over-HTTP entry point
# ---------------------------------------------------------------------------

def _mount_mcp_transport(
    app: FastAPI,
    *,
    readiness_service: Any,
) -> None:
    transport_mount: MCPStreamableHTTPMount = build_mcp_streamable_http_mount(
        app_context_factory=lambda: resolve_app_context_from_fastapi_state(app)
    )
    secured_transport: ASGIApp = MCPAuthMiddleware(
        transport_mount.asgi_app,
        readiness_service=readiness_service,
    )

    app.state.mcp_transport_mount = transport_mount
    app.mount("/mcp", secured_transport)
