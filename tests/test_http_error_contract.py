"""Tests that all protected-route non-2xx responses use the stable ErrorEnvelope shape.

These tests drive Task 1 of the HTTP MCP cutover remediation: normalize error
contracts on protected routes so no raw FastAPI ``detail`` payloads leak out.

Required shape for every non-2xx response on a protected route::

    {
        "error": {
            "code": "<SCREAMING_SNAKE>",
            "message": "<human-readable>",
            "request_id": "<uuid4>",
            "details": <dict | null>
        }
    }
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.server import build_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
    app = build_app(base_dir=tmp_path / ".workflows")
    return TestClient(app)


def test_config_validate_422_uses_error_envelope(client: TestClient) -> None:
    """A 422 on /config/validate must return the stable ErrorEnvelope, not FastAPI detail."""
    response = client.post(
        "/config/validate",
        headers={"Authorization": "Bearer 0123456789abcdef0123456789abcdef"},
        json={"profiles": "bad"},
    )

    assert response.status_code == 422
    payload = response.json()
    # Must have top-level "error" key — not "detail"
    assert "error" in payload, f"Expected 'error' key, got: {list(payload.keys())}"
    assert "detail" not in payload, "Raw FastAPI 'detail' must not leak on protected routes"
    assert set(payload["error"]) >= {"code", "message", "request_id"}


def test_missing_auth_401_uses_error_envelope(client: TestClient) -> None:
    """A 401 on /config/validate (no token) must return the stable ErrorEnvelope."""
    response = client.post("/config/validate", json={"profiles": []})

    assert response.status_code == 401
    payload = response.json()
    assert "error" in payload, f"Expected 'error' key, got: {list(payload.keys())}"
    assert "detail" not in payload, "Raw FastAPI 'detail' must not leak on protected routes"
    assert payload["error"]["code"] == "UNAUTHORIZED"
    assert "request_id" in payload["error"]


def test_config_apply_422_uses_error_envelope(client: TestClient) -> None:
    """A 422 on /config/apply (invalid payload) must return the stable ErrorEnvelope."""
    response = client.post(
        "/config/apply",
        headers={"Authorization": "Bearer 0123456789abcdef0123456789abcdef"},
        json={"profiles": "bad"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload, f"Expected 'error' key, got: {list(payload.keys())}"
    assert "detail" not in payload
    assert payload["error"]["code"] == "VALIDATION_FAILED"


def test_config_apply_missing_auth_uses_error_envelope(client: TestClient) -> None:
    """A 401 on /config/apply (no token) must return the stable ErrorEnvelope."""
    response = client.post("/config/apply", json={"profiles": []})

    assert response.status_code == 401
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "UNAUTHORIZED"
