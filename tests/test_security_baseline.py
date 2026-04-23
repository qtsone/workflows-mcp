"""Security baseline middleware tests.

Covers:
- CORS deny-by-default: unlisted origins are denied.
- CORS allowlist: listed origins are allowed when configured.
- Oversized body rejected with 413 before route logic runs.
- Rate limiting returns 429 on protected routes.
- /config routes have a stricter rate limit than general protected routes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

TOKEN = "0123456789abcdef0123456789abcdef"
AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", TOKEN)
    monkeypatch.delenv("WORKFLOWS_CORS_ORIGINS", raising=False)
    from workflows_mcp.server import build_app

    return build_app(base_dir=tmp_path / ".workflows")


@pytest.fixture()
def app_with_cors(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", TOKEN)
    monkeypatch.setenv("WORKFLOWS_CORS_ORIGINS", "https://allowed.example.com")
    from workflows_mcp.server import build_app

    return build_app(base_dir=tmp_path / ".workflows")


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_unlisted_origin_denied_by_default(app):
    """An unlisted origin must not appear in CORS response headers."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Either the header is absent or empty — the origin must NOT be echoed back.
    allow_origin = response.headers.get("access-control-allow-origin", "")
    assert allow_origin != "https://evil.example.com", (
        f"Expected unlisted origin to be denied, got: {allow_origin!r}"
    )


def test_cors_listed_origin_allowed_when_configured(app_with_cors):
    """An explicitly listed origin must be reflected in the CORS response."""
    client = TestClient(app_with_cors, raise_server_exceptions=False)
    response = client.options(
        "/health",
        headers={
            "Origin": "https://allowed.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    allow_origin = response.headers.get("access-control-allow-origin", "")
    assert allow_origin == "https://allowed.example.com", (
        f"Expected listed origin to be allowed, got: {allow_origin!r}"
    )


# ---------------------------------------------------------------------------
# Body size limit
# ---------------------------------------------------------------------------


def test_oversized_body_returns_413(app):
    """A body larger than the configured limit must be rejected with 413."""
    client = TestClient(app, raise_server_exceptions=False)
    # Build a payload whose raw JSON serialization exceeds the limit.
    large_payload = {"profiles": [{"x": "a" * 2_000_000}]}
    response = client.post(
        "/config/validate",
        headers=AUTH_HEADER,
        json=large_payload,
    )
    assert response.status_code == 413


def test_oversized_body_returns_error_envelope(app):
    """The 413 response must use the stable error envelope shape."""
    client = TestClient(app, raise_server_exceptions=False)
    large_payload = {"profiles": [{"x": "a" * 2_000_000}]}
    response = client.post(
        "/config/validate",
        headers=AUTH_HEADER,
        json=large_payload,
    )
    assert response.status_code == 413
    payload = response.json()
    assert "error" in payload
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "REQUEST_TOO_LARGE"


# ---------------------------------------------------------------------------
# Rate limiting — general protected routes
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_on_protected_routes(app):
    """Repeated requests beyond the rate limit must return 429."""
    client = TestClient(app, raise_server_exceptions=False)
    # Fire enough requests to exhaust the rate limit.
    # The limit is intentionally small in tests; we fire 200 to guarantee a hit.
    responses = [
        client.get("/ready")
        for _ in range(200)
    ]
    status_codes = {r.status_code for r in responses}
    assert 429 in status_codes, (
        f"Expected at least one 429 after 200 rapid requests; got statuses: {status_codes}"
    )


def test_rate_limit_429_uses_error_envelope(app):
    """The 429 response must use the stable error envelope shape."""
    client = TestClient(app, raise_server_exceptions=False)
    responses = [client.get("/ready") for _ in range(200)]
    rate_limited = [r for r in responses if r.status_code == 429]
    assert rate_limited, "Expected at least one 429 response"
    payload = rate_limited[0].json()
    assert "error" in payload
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "RATE_LIMITED"


# ---------------------------------------------------------------------------
# Rate limiting — /config stricter
# ---------------------------------------------------------------------------


def test_config_routes_have_stricter_rate_limit_than_general(app):
    """/config routes must hit 429 sooner than general routes.

    We fire the same number of requests at /config/validate and /ready and
    confirm that /config/validate hits the rate limit at a lower request count
    (i.e., it has a tighter cap).
    """
    client = TestClient(app, raise_server_exceptions=False)

    # Exhaust /config/validate rate limit.
    config_429_at: int | None = None
    for i in range(100):
        r = client.post("/config/validate", headers=AUTH_HEADER, json={"profiles": []})
        if r.status_code == 429:
            config_429_at = i + 1
            break

    # Exhaust /ready rate limit.
    ready_429_at: int | None = None
    for i in range(200):
        r = client.get("/ready")
        if r.status_code == 429:
            ready_429_at = i + 1
            break

    assert config_429_at is not None, "/config/validate never returned 429 within 100 requests"
    assert ready_429_at is not None, "/ready never returned 429 within 200 requests"
    assert config_429_at <= ready_429_at, (
        f"/config rate limit ({config_429_at}) should be stricter (lower) "
        f"than general rate limit ({ready_429_at})"
    )
