from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base class for all strict user-facing request/response models.

    Unknown fields are rejected (extra="forbid") to enforce the API contract
    and catch payload drift early at the boundary.
    """

    model_config = ConfigDict(extra="forbid")


class ScopeModel(StrictModel):
    """Hierarchical scope selector: palace / wing / room / compartment."""

    palace: str | None = None
    wing: str | None = None
    room: str | None = None
    compartment: str | None = None


class ResponseOptions(StrictModel):
    """Response shaping options.

    Note: ``response.mode`` is unsupported and rejected by schema.
    Extra fields (including ``mode``) are rejected by the strict model config.
    Use ``ingestion.mode`` for ingestion-mode selection.
    """

    debug: bool = False


class IngestionConfig(StrictModel):
    """Ingestion strategy configuration."""

    mode: Literal["programmatic", "llm"]
    llm_profile: str | None = None
    reproducibility: Literal["strict", "relaxed"] = "strict"


class PlanStep(StrictModel):
    """A single step in an onboard or sync execution plan."""

    operation: Literal["ingest", "supersede", "archive", "maintain"]
    payload: dict[str, Any]


class CompletedStep(StrictModel):
    """A step that has already been executed in a checkpointed run."""

    operation: Literal["ingest", "supersede", "archive", "maintain"]
    result: dict[str, Any] | None = None


class CheckpointModel(StrictModel):
    """Checkpoint state for resuming an interrupted onboard or sync run."""

    version: str
    scope: ScopeModel
    plan: list[PlanStep]
    next_index: int
    completed: list[CompletedStep]
    scan: dict[str, Any] | None = None
    scan_snapshot: dict[str, Any] | None = None


class OnboardRequest(StrictModel):
    """Strict request model for the ``onboard`` operation."""

    scope: ScopeModel | None = None
    ingestion: IngestionConfig | None = None
    ingest: dict[str, Any] | None = None
    supersede: dict[str, Any] | None = None
    archive: dict[str, Any] | None = None
    maintain: dict[str, Any] | None = None
    checkpoint: CheckpointModel | None = None
    scan: dict[str, Any] | None = None
    response: ResponseOptions | None = None
    max_operations: int = Field(default=1, ge=1, le=20)
    debug: bool = False


class SyncRequest(StrictModel):
    """Strict request model for the ``sync`` operation."""

    scope: ScopeModel | None = None
    ingest: dict[str, Any] | None = None
    supersede: dict[str, Any] | None = None
    archive: dict[str, Any] | None = None
    maintain: dict[str, Any] | None = None
    checkpoint: CheckpointModel | None = None
    scan: dict[str, Any] | None = None
    response: ResponseOptions | None = None
    max_operations: int = Field(default=1, ge=1, le=20)
    debug: bool = False


class ReadinessState(StrEnum):
    """Readiness states reported by ``/ready`` and used internally by readiness service."""

    UNCONFIGURED = "unconfigured"
    PARTIALLY_CONFIGURED = "partially_configured"
    READY = "ready"


class ErrorDetail(StrictModel):
    """Machine-readable error payload embedded in every non-2xx response."""

    code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str


class ErrorEnvelope(StrictModel):
    """Top-level error envelope for all non-2xx HTTP responses.

    Example::

        {
            "error": {
                "code": "CONFIG_REQUIRED",
                "message": "Service is not ready. Complete /api/admin/v1 setup.",
                "details": {"readiness_state": "partially_configured"},
                "request_id": "a1b2c3d4..."
            }
        }
    """

    error: ErrorDetail

    @classmethod
    def for_code(
        cls,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None,
        request_id: str,
    ) -> ErrorEnvelope:
        """Construct an ``ErrorEnvelope`` from named fields."""
        return cls(
            error=ErrorDetail(
                code=code,
                message=message,
                details=details,
                request_id=request_id,
            )
        )
