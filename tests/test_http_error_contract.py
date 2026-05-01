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


def test_login_422_uses_error_envelope(client: TestClient) -> None:
    """A 422 on admin login must return ErrorEnvelope, not raw FastAPI detail."""
    response = client.post("/api/admin/v1/auth/login", json={"password": 123})

    assert response.status_code == 422
    payload = response.json()
    # Must have top-level "error" key — not "detail"
    assert "error" in payload, f"Expected 'error' key, got: {list(payload.keys())}"
    assert "detail" not in payload, "Raw FastAPI 'detail' must not leak on protected routes"
    assert set(payload["error"]) >= {"code", "message", "request_id"}


def test_missing_auth_401_uses_error_envelope(client: TestClient) -> None:
    """A 401 on session route (no cookie) must return the stable ErrorEnvelope."""
    response = client.get("/api/admin/v1/auth/session")

    assert response.status_code == 401
    payload = response.json()
    assert "error" in payload, f"Expected 'error' key, got: {list(payload.keys())}"
    assert "detail" not in payload, "Raw FastAPI 'detail' must not leak on protected routes"
    assert payload["error"]["code"] == "UNAUTHORIZED"
    assert "request_id" in payload["error"]


def test_login_missing_password_422_uses_error_envelope(client: TestClient) -> None:
    """A 422 on admin login (missing required field) must return ErrorEnvelope."""
    response = client.post("/api/admin/v1/auth/login", json={})

    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload, f"Expected 'error' key, got: {list(payload.keys())}"
    assert "detail" not in payload
    assert payload["error"]["code"] == "VALIDATION_FAILED"


def test_mcp_missing_auth_uses_error_envelope(client: TestClient) -> None:
    """A 401 on /mcp (no bearer token) must return the stable ErrorEnvelope."""
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
    )

    assert response.status_code == 401
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "UNAUTHORIZED"


def test_method_not_allowed_uses_error_envelope(client: TestClient) -> None:
    """A 405 on a protected route (wrong HTTP method) must use the stable ErrorEnvelope.

    Starlette emits MethodNotAllowed as its own HTTPException subclass via the
    routing layer.  This test ensures our handler intercepts it before the raw
    ``{"detail": "Method Not Allowed"}`` response escapes to callers.
    """
    # /api/admin/v1/auth/logout is POST-only; calling GET triggers a 405.
    response = client.get("/api/admin/v1/auth/logout")

    assert response.status_code == 405
    payload = response.json()
    assert "error" in payload, f"Expected 'error' key, got: {list(payload.keys())}"
    assert "detail" not in payload, "Raw Starlette 'detail' must not leak on protected routes"
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "METHOD_NOT_ALLOWED"
