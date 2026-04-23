from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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
