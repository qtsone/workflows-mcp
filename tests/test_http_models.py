import pytest
from pydantic import ValidationError

from workflows_mcp.http_models import ErrorEnvelope, OnboardRequest, SyncRequest


def test_onboard_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        OnboardRequest.model_validate({
            "scope": {"palace": "acme"},
            "unknown": True,
        })


def test_onboard_request_rejects_legacy_response_mode() -> None:
    with pytest.raises(ValidationError):
        OnboardRequest.model_validate({
            "scope": {"palace": "acme"},
            "response": {"mode": "programmatic"},
        })


def test_sync_request_accepts_minimal_checkpoint_payload() -> None:
    payload = SyncRequest.model_validate({
        "checkpoint": {
            "version": "oss-r3",
            "scope": {"palace": "acme"},
            "plan": [{"operation": "ingest", "payload": {}}],
            "next_index": 0,
            "completed": [],
        }
    })
    assert payload.checkpoint.version == "oss-r3"


def test_error_envelope_has_stable_shape() -> None:
    payload = ErrorEnvelope.for_code(
        code="CONFIG_REQUIRED",
        message="Service is not ready.",
        details={"readiness_state": "partially_configured"},
        request_id="req-123",
    )
    assert payload.error.code == "CONFIG_REQUIRED"
    assert payload.error.request_id == "req-123"
