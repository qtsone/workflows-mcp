from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .auth import TokenStore
from .config_service import ConfigService
from .http_models import (
    ConfigApplyResponse,
    ConfigStatusResponse,
    LLMConfigPayload,
)

_MIN_TOKEN_BYTES: int = 32


class _RotatePayload(BaseModel):
    token: str


def build_config_router(
    config_service: ConfigService,
    readiness_service: Any,
    auth_guard: Callable[..., Any] | None = None,
    token_store: TokenStore | None = None,
) -> APIRouter:
    """Return an ``APIRouter`` with config management endpoints.

    Endpoints:
    - ``GET /config``: Alias for status — returns current readiness state and blockers.
    - ``GET /config/status``: Returns current readiness state and blockers.
    - ``POST /config/validate``: Validates a config payload without writing.
    - ``POST /config/apply``: Atomically writes config and returns updated state.
    - ``POST /config/credentials/rotate``: Rotates the bearer token.
    - ``POST /config/credentials/revoke``: Revokes the bearer token.

    All routes are mounted behind *auth_guard* when provided.  Validation
    failures return the stable ``ErrorEnvelope`` shape via ``_error_response``
    rather than raw FastAPI ``{"detail": ...}`` payloads.
    """
    # Import here to avoid a circular dependency at module load time.
    from .http_app import _error_response

    dependencies = [Depends(auth_guard)] if auth_guard is not None else []
    router = APIRouter(prefix="/config", tags=["config"], dependencies=dependencies)

    @router.get("/", response_model=ConfigStatusResponse)
    async def get_config_root() -> ConfigStatusResponse:
        report = await readiness_service.evaluate()
        return ConfigStatusResponse(
            state=report.state,
            blockers=report.blockers,
            config_present=(config_service.base_dir / "llm-config.yml").exists(),
        )

    @router.get("/status", response_model=ConfigStatusResponse)
    async def get_config_status() -> ConfigStatusResponse:
        report = await readiness_service.evaluate()
        return ConfigStatusResponse(
            state=report.state,
            blockers=report.blockers,
            config_present=(config_service.base_dir / "llm-config.yml").exists(),
        )

    @router.post("/validate", response_model=None)
    async def validate_config(payload: LLMConfigPayload) -> Any:
        errors = config_service.validate_payload(payload.model_dump())
        if errors:
            return _error_response(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Configuration payload is invalid.",
                details={"field_errors": errors},
            )
        return {"valid": True}

    @router.post("/apply", response_model=ConfigApplyResponse)
    async def apply_config(payload: LLMConfigPayload) -> ConfigApplyResponse | JSONResponse:
        errors = config_service.validate_payload(payload.model_dump())
        if errors:
            return _error_response(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Configuration payload is invalid.",
                details={"field_errors": errors},
            )
        try:
            await config_service.apply_payload(payload.model_dump())
        except RuntimeError as exc:
            if str(exc) == "CONFIG_WRITE_IN_PROGRESS":
                return _error_response(
                    status_code=409,
                    code="CONFIG_WRITE_IN_PROGRESS",
                    message="A config write is already in progress. Retry after it completes.",
                )
            raise
        report = await readiness_service.evaluate()
        return ConfigApplyResponse(
            applied=True,
            state=report.state,
            blockers=report.blockers,
        )

    if token_store is not None:

        @router.post("/credentials/rotate", response_model=None)
        async def rotate_credentials(payload: _RotatePayload) -> Any:
            if len(payload.token.encode("utf-8")) < _MIN_TOKEN_BYTES:
                return _error_response(
                    status_code=400,
                    code="TOKEN_TOO_SHORT",
                    message=f"Token must be at least {_MIN_TOKEN_BYTES} characters.",
                )
            token_store.rotate_token(payload.token)
            return {"rotated": True}

        @router.post("/credentials/revoke", response_model=None)
        async def revoke_credentials() -> Any:
            token_store.revoke()
            return {"revoked": True}

    return router
