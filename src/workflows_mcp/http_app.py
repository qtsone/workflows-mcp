from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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


def _http_status_to_code(status_code: int) -> str:
    """Map common HTTP status codes to SCREAMING_SNAKE error codes."""
    _map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
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

    from fastapi import Depends

    @app.get("/config", dependencies=[Depends(auth_guard)])
    async def config() -> dict[str, str]:
        return {"status": "protected"}

    if config_service is not None:
        config_router = build_config_router(config_service, auth_guard=auth_guard)
        app.include_router(config_router)

    return app
