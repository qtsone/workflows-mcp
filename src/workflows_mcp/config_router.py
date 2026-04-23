from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from .config_service import ConfigService


def build_config_router(
    config_service: ConfigService,
    auth_guard: Callable[..., Any] | None = None,
) -> APIRouter:
    """Return an ``APIRouter`` with ``/config/validate`` and ``/config/apply`` endpoints.

    Routes are mounted behind *auth_guard* when provided.  All validation
    failures return the stable ``ErrorEnvelope`` shape via ``_error_response``
    rather than raw FastAPI ``{"detail": ...}`` payloads.
    """
    # Import here to avoid a circular dependency at module load time.
    from .http_app import _error_response

    dependencies = [Depends(auth_guard)] if auth_guard is not None else []
    router = APIRouter(prefix="/config", tags=["config"], dependencies=dependencies)

    @router.post("/validate", response_model=None)
    async def validate_config(payload: dict[str, Any]) -> Any:
        errors = config_service.validate_payload(payload)
        if errors:
            return _error_response(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Configuration payload is invalid.",
                details={"field_errors": errors},
            )
        return {"valid": True}

    @router.post("/apply", response_model=None)
    async def apply_config(payload: dict[str, Any]) -> Any:
        errors = config_service.validate_payload(payload)
        if errors:
            return _error_response(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Configuration payload is invalid.",
                details={"field_errors": errors},
            )
        config_service.apply_payload(payload)
        return {"applied": True}

    return router
