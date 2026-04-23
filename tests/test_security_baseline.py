"""Security baseline middleware tests.

Covers:
- CORS deny-by-default: unlisted origins are denied.
- CORS allowlist: listed origins are allowed when configured.
- Oversized body rejected with 413 before route logic runs — including when
  Content-Length header is absent (streaming / chunked bodies).
- Rate limiting returns 429 on protected routes (/config, /mcp).
- /config routes have a stricter rate limit than other protected routes.
- Public routes (/health, /ready) are NOT rate-limited.
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
# Body size limit — with Content-Length header
# ---------------------------------------------------------------------------


def test_oversized_body_returns_413(app):
    """A body larger than the configured limit must be rejected with 413."""
    client = TestClient(app, raise_server_exceptions=False)
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
# Body size limit — without Content-Length (streaming / chunked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_body_without_content_length_returns_413(app):
    """Bodies exceeding the limit must be rejected even when Content-Length is absent.

    Tests the BodyLimitMiddleware via httpx ASGITransport with a chunked body
    whose Content-Length header is stripped so the middleware cannot fast-path
    on the header and must instead count accumulated bytes from receive().
    """
    import httpx

    large_bytes = b"x" * 2_000_000

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://testserver",
    ) as client:
        # Send as a streaming generator so httpx does not set Content-Length.
        async def _body_gen():
            # Yield in chunks to simulate streaming without Content-Length.
            chunk = 65536
            for offset in range(0, len(large_bytes), chunk):
                yield large_bytes[offset : offset + chunk]

        response = await client.post(
            "/config/validate",
            content=_body_gen(),
            headers={**AUTH_HEADER, "Content-Type": "application/json"},
        )

    assert response.status_code == 413, (
        f"Expected 413 for oversized streaming body without Content-Length, "
        f"got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Rate limiting — protected routes (/config, /mcp)
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_on_protected_routes(app):
    """Repeated requests to protected routes beyond the limit must return 429."""
    client = TestClient(app, raise_server_exceptions=False)
    # /config/validate is a protected route; fire enough to exhaust the cap.
    responses = [
        client.post("/config/validate", headers=AUTH_HEADER, json={"profiles": []})
        for _ in range(100)
    ]
    status_codes = {r.status_code for r in responses}
    assert 429 in status_codes, (
        f"Expected at least one 429 after 100 rapid requests to /config/validate; "
        f"got statuses: {status_codes}"
    )


def test_rate_limit_429_uses_error_envelope(app):
    """The 429 response must use the stable error envelope shape."""
    client = TestClient(app, raise_server_exceptions=False)
    responses = [
        client.post("/config/validate", headers=AUTH_HEADER, json={"profiles": []})
        for _ in range(100)
    ]
    rate_limited = [r for r in responses if r.status_code == 429]
    assert rate_limited, "Expected at least one 429 response"
    payload = rate_limited[0].json()
    assert "error" in payload
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "RATE_LIMITED"


def test_public_routes_are_not_rate_limited(app):
    """Public health/readiness probes must not be rate-limited.

    Liveness probes fire at high frequency; rate-limiting them would cause
    false unavailability signals from orchestrators.
    """
    client = TestClient(app, raise_server_exceptions=False)
    # Fire well above any configured general limit.
    responses = [client.get("/health") for _ in range(200)]
    assert all(r.status_code == 200 for r in responses), (
        "Expected all /health responses to be 200 (not rate-limited)"
    )


# ---------------------------------------------------------------------------
# Rate limiting — /config stricter than other protected routes
# ---------------------------------------------------------------------------


def test_config_routes_have_stricter_rate_limit_than_mcp(app):
    """/config routes must hit 429 sooner than /mcp.

    Both are protected; /config carries a tighter cap.
    """
    client = TestClient(app, raise_server_exceptions=False)

    # Exhaust /config/validate rate limit.
    config_429_at: int | None = None
    for i in range(100):
        r = client.post("/config/validate", headers=AUTH_HEADER, json={"profiles": []})
        if r.status_code == 429:
            config_429_at = i + 1
            break

    # Exhaust /mcp rate limit (will 409 while not ready, but rate limit fires first).
    mcp_429_at: int | None = None
    for i in range(200):
        r = client.post("/mcp", headers=AUTH_HEADER, json={})
        if r.status_code == 429:
            mcp_429_at = i + 1
            break

    assert config_429_at is not None, "/config/validate never returned 429 within 100 requests"
    assert mcp_429_at is not None, "/mcp never returned 429 within 200 requests"
    assert config_429_at <= mcp_429_at, (
        f"/config rate limit ({config_429_at}) should be stricter (lower) "
        f"than /mcp rate limit ({mcp_429_at})"
    )
