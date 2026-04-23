from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import TokenStore, generate_request_id
from .config_router import build_config_router
from .config_service import ConfigService
from .http_models import ErrorEnvelope, ReadinessState


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

#: Stricter rate-limit applied to ``/config`` routes.
_CONFIG_RATE_LIMIT: int = 20

#: Sliding-window duration in seconds.
_RATE_WINDOW_SECONDS: float = 60.0


class BodyLimitMiddleware:
    """Reject requests whose ``Content-Length`` exceeds *max_bytes*.

    Checking the ``Content-Length`` header is sufficient for deterministic
    protection: the header is required on POST/PUT/PATCH requests by HTTP
    spec and is always present when FastAPI's ``TestClient`` (httpx) sends
    a body.  Streaming requests without a ``Content-Length`` header are not
    rejected at this layer (they are handled by request-body read timeouts
    at the server level).

    The rejection happens before any route logic so the body is never read.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers: dict[bytes, bytes] = dict(scope.get("headers", []))
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
        await self.app(scope, receive, send)


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
    """Enforce per-IP sliding-window rate limits.

    Two limits are supported:
    - ``config_limiter``: applied to paths starting with ``/config``.
    - ``general_limiter``: applied to all other paths.

    ``/health`` is deliberately exempted to avoid interfering with liveness
    probes that fire at high frequency from orchestrators.

    The client key is derived from the ``X-Forwarded-For`` header when
    present (first address), falling back to the ASGI ``client`` tuple.
    """

    _EXEMPT_PATHS: frozenset[str] = frozenset({"/health"})

    def __init__(
        self,
        app: ASGIApp,
        *,
        general_limiter: SlidingWindowRateLimiter,
        config_limiter: SlidingWindowRateLimiter,
    ) -> None:
        self.app = app
        self.general_limiter = general_limiter
        self.config_limiter = config_limiter

    def _client_key(self, scope: Scope) -> str:
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        xff = headers.get(b"x-forwarded-for")
        if xff:
            return xff.decode("latin-1", errors="replace").split(",")[0].strip()
        client = scope.get("client")
        if client:
            return str(client[0])
        return "unknown"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path: str = scope.get("path", "")
            if path not in self._EXEMPT_PATHS:
                key = self._client_key(scope)
                limiter = (
                    self.config_limiter if path.startswith("/config") else self.general_limiter
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
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Body limit — checked before route logic on every HTTP request.
    try:
        max_body = int(os.getenv("WORKFLOWS_MAX_BODY_BYTES", str(_DEFAULT_MAX_BODY_BYTES)))
    except ValueError:
        max_body = _DEFAULT_MAX_BODY_BYTES
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body)

    # Rate limiting — config routes get a stricter cap.
    try:
        general_limit = int(os.getenv("WORKFLOWS_RATE_LIMIT", str(_GENERAL_RATE_LIMIT)))
    except ValueError:
        general_limit = _GENERAL_RATE_LIMIT
    try:
        config_limit = int(os.getenv("WORKFLOWS_CONFIG_RATE_LIMIT", str(_CONFIG_RATE_LIMIT)))
    except ValueError:
        config_limit = _CONFIG_RATE_LIMIT

    general_limiter = SlidingWindowRateLimiter(limit=general_limit, window=_RATE_WINDOW_SECONDS)
    config_limiter = SlidingWindowRateLimiter(limit=config_limit, window=_RATE_WINDOW_SECONDS)
    app.add_middleware(
        RateLimitMiddleware,
        general_limiter=general_limiter,
        config_limiter=config_limiter,
    )


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


def create_app(
    *,
    readiness_service: Any,
    token_store: TokenStore,
    config_service: ConfigService | None = None,
) -> FastAPI:
    app = FastAPI(title="workflows-mcp", docs_url="/docs", openapi_url="/openapi.json")

    _install_exception_handlers(app)
    _install_security_middleware(app)

    auth_guard = _make_auth_guard(token_store)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        report = await readiness_service.evaluate()
        status_code = 200 if report.state == ReadinessState.READY else 503
        return JSONResponse(
            status_code=status_code,
            content={"state": str(report.state), "blockers": report.blockers},
        )

    if config_service is not None:
        config_router = build_config_router(
            config_service,
            readiness_service=readiness_service,
            auth_guard=auth_guard,
        )
        app.include_router(config_router)

    _register_mcp_endpoint(app, readiness_service=readiness_service, auth_guard=auth_guard)

    return app


# ---------------------------------------------------------------------------
# MCP-over-HTTP entry point
# ---------------------------------------------------------------------------

_MCP_METHODS: frozenset[str] = frozenset({"schema"})


def _register_mcp_endpoint(app: FastAPI, *, readiness_service: Any, auth_guard: Any) -> None:
    """Register the protected ``POST /mcp`` endpoint.

    The endpoint enforces two gates in order:
    1. Bearer authentication (401 on failure).
    2. Service readiness (409 when not ``READY``).

    When both gates pass, the request is dispatched to a method router backed
    by real production adapters:

    - ``schema``: delegates to ``tools_memory.memory_schema_payload()``, the
      same pure function used by ``memory(operation="schema")`` in the MCP tool.
      Requires no DB connectivity.

    All other method names return 400 with code ``METHOD_NOT_FOUND``.
    Error responses always use the stable ``ErrorEnvelope`` shape.
    """

    @app.post("/mcp")
    async def mcp_http_endpoint(
        request: Request,
        _: None = Depends(auth_guard),
    ) -> JSONResponse:
        # Readiness gate — evaluated after auth so we never leak readiness
        # state to unauthenticated callers.
        report = await readiness_service.evaluate()
        if report.state != ReadinessState.READY:
            return _error_response(
                status_code=409,
                code="CONFIG_REQUIRED",
                message="Service is not ready. Complete /config setup.",
                details={
                    "readiness_state": str(report.state),
                    "missing": report.blockers,
                },
            )

        # Method dispatch — backed by real adapter functions.
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        method: str | None = payload.get("method") if isinstance(payload, dict) else None

        if method == "schema":
            from .tools_memory import memory_schema_payload

            return JSONResponse(status_code=200, content={"result": memory_schema_payload()})

        return _error_response(
            status_code=400,
            code="METHOD_NOT_FOUND",
            message=f"Unknown method: {method!r}. Supported methods: {sorted(_MCP_METHODS)}.",
            details={"method": method},
        )
