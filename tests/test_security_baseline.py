"""Security baseline middleware tests.

Covers:
- CORS deny-by-default: unlisted origins are denied.
- CORS allowlist: listed origins are allowed when configured.
- Oversized body rejected with 413 before route logic runs — including when
  Content-Length header is absent (streaming / chunked bodies).
- Rate limiting returns 429 on protected routes (/mcp, /api/admin/v1/auth/login).
- MCP and login routes use independent limiter buckets.
- Public routes (/health, /ready) are NOT rate-limited.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.security.csrf import CSRF_HEADER_NAME

TOKEN = "0123456789abcdef0123456789abcdef"
AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}"}


def _assert_no_sensitive_literals(serialized: str, literals: list[str]) -> None:
    for literal in literals:
        assert literal not in serialized, f"Sensitive literal leaked in response: {literal!r}"


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


def test_cors_preflight_admin_logout_allows_csrf_header_and_credentials(app_with_cors):
    """Configured CORS must support cookie+CSRF browser admin mutations."""
    client = TestClient(app_with_cors, raise_server_exceptions=False)
    response = client.options(
        "/api/admin/v1/auth/logout",
        headers={
            "Origin": "https://allowed.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"{CSRF_HEADER_NAME}, Content-Type",
        },
    )

    assert response.status_code in {200, 204}
    assert response.headers.get("access-control-allow-origin") == "https://allowed.example.com"
    assert response.headers.get("access-control-allow-credentials") == "true"
    allow_headers = response.headers.get("access-control-allow-headers", "")
    assert CSRF_HEADER_NAME.lower() in allow_headers.lower()


# ---------------------------------------------------------------------------
# Body size limit — with Content-Length header
# ---------------------------------------------------------------------------


def test_oversized_body_returns_413(app):
    """A body larger than the configured limit must be rejected with 413."""
    client = TestClient(app, raise_server_exceptions=False)
    large_payload = {"profiles": [{"x": "a" * 2_000_000}]}
    response = client.post("/mcp", headers=AUTH_HEADER, json=large_payload)
    assert response.status_code == 413


def test_oversized_body_returns_error_envelope(app):
    """The 413 response must use the stable error envelope shape."""
    client = TestClient(app, raise_server_exceptions=False)
    large_payload = {"profiles": [{"x": "a" * 2_000_000}]}
    response = client.post("/mcp", headers=AUTH_HEADER, json=large_payload)
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
            "/mcp",
            content=_body_gen(),
            headers={**AUTH_HEADER, "Content-Type": "application/json"},
        )

    assert response.status_code == 413, (
        f"Expected 413 for oversized streaming body without Content-Length, "
        f"got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Rate limiting — protected routes (/mcp, /api/admin/v1/auth/login)
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_on_protected_routes(app):
    """Repeated requests to protected routes beyond the limit must return 429."""
    client = TestClient(app, raise_server_exceptions=False)
    responses = [client.post("/mcp", headers=AUTH_HEADER, json={}) for _ in range(100)]
    status_codes = {r.status_code for r in responses}
    assert 429 in status_codes, (
        f"Expected at least one 429 after 100 rapid requests to /mcp; got statuses: {status_codes}"
    )


def test_rate_limit_429_uses_error_envelope(app):
    """The 429 response must use the stable error envelope shape."""
    client = TestClient(app, raise_server_exceptions=False)
    responses = [client.post("/mcp", headers=AUTH_HEADER, json={}) for _ in range(100)]
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
# Rate limiting — MCP and login are independently limited
# ---------------------------------------------------------------------------


def test_login_routes_have_independent_rate_limit_from_mcp(tmp_path, monkeypatch):
    """Login and MCP routes must use separate limiter buckets."""
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", TOKEN)
    monkeypatch.setenv("WORKFLOWS_LOGIN_RATE_LIMIT", "3")
    monkeypatch.setenv("WORKFLOWS_MCP_RATE_LIMIT", "50")
    monkeypatch.setenv("WORKFLOWS_RATE_LIMIT", "100")
    monkeypatch.delenv("WORKFLOWS_CORS_ORIGINS", raising=False)
    base_dir = tmp_path / ".workflows"
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password="phase2-admin-password",
    )

    from workflows_mcp.server import build_app

    client = TestClient(build_app(base_dir=base_dir), raise_server_exceptions=False)

    login_statuses = [
        client.post("/api/admin/v1/auth/login", json={"password": "wrong-password"}).status_code
        for _ in range(6)
    ]
    assert 429 in login_statuses

    mcp_response = client.post("/mcp", headers=AUTH_HEADER, json={})
    assert mcp_response.status_code != 429


def test_mcp_rate_limit_applies_to_mcp_route(app):
    """MCP limiter must eventually throttle repeated /mcp requests."""
    client = TestClient(app, raise_server_exceptions=False)
    mcp_429_at: int | None = None
    for i in range(200):
        r = client.post("/mcp", headers=AUTH_HEADER, json={})
        if r.status_code == 429:
            mcp_429_at = i + 1
            break

    assert mcp_429_at is not None, "/mcp never returned 429 within 200 requests"


def test_exhausted_login_limiter_does_not_throttle_mcp_route(tmp_path, monkeypatch):
    """Login and MCP must use separate limiter buckets."""
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", TOKEN)
    monkeypatch.setenv("WORKFLOWS_LOGIN_RATE_LIMIT", "3")
    monkeypatch.setenv("WORKFLOWS_MCP_RATE_LIMIT", "50")
    monkeypatch.setenv("WORKFLOWS_RATE_LIMIT", "100")
    monkeypatch.delenv("WORKFLOWS_CORS_ORIGINS", raising=False)
    base_dir = tmp_path / ".workflows"
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password="phase2-admin-password",
    )

    from workflows_mcp.server import build_app

    client = TestClient(build_app(base_dir=base_dir), raise_server_exceptions=False)

    login_statuses = [
        client.post("/api/admin/v1/auth/login", json={"password": "wrong-password"}).status_code
        for _ in range(6)
    ]
    assert 429 in login_statuses

    mcp_response = client.post("/mcp", headers=AUTH_HEADER, json={})
    assert mcp_response.status_code != 429


# ---------------------------------------------------------------------------
# OpenAPI / API response non-leak baseline
# ---------------------------------------------------------------------------


def test_openapi_json_does_not_leak_sensitive_literals(app):
    """OpenAPI schema/examples must not contain concrete secret-like local values."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    serialized = response.text
    disallowed_literals = [
        "sk-live-local-openapi-secret-123456",
        "/private/tmp/workflows-mcp/secret.key",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.local.payload.signature",
        "mcp_tok_local_secret_0123456789abcdef",
        "-----BEGIN PRIVATE KEY-----",
        TOKEN,
    ]
    _assert_no_sensitive_literals(serialized, disallowed_literals)

    # Guard against leaking concrete bearer token values while allowing
    # harmless security scheme metadata such as "BearerAuth".
    assert "BearerAuth" in serialized
    assert "Bearer eyJ" not in serialized


def test_error_responses_do_not_echo_sensitive_request_literals(tmp_path, monkeypatch):
    """Auth error responses must not reflect attacker-controlled secret-like literals."""
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", TOKEN)
    from workflows_mcp.server import build_app

    client = TestClient(build_app(base_dir=tmp_path / ".workflows"), raise_server_exceptions=False)

    injected_secret = "sk-live-local-response-secret-987654321"
    injected_path = "/private/tmp/workflows-mcp/client-secrets.json"
    injected_bearer = "Bearer eyJhbGciOiJIUzI1NiJ9.reflected.payload"
    injected_key_marker = "-----BEGIN PRIVATE KEY-----"

    # Invalid token format path: keep route protected and trigger 401 error flow.
    auth_response = client.post(
        "/mcp",
        headers={"Authorization": injected_bearer},
        json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
    )
    assert auth_response.status_code == 401

    # Auth-required API path: inject sensitive literals in body and force 401.
    unauth_response = client.post(
        "/api/admin/v1/secrets",
        json={
            "name": "OPENAI_API_KEY",
            "value": injected_secret,
            "path": injected_path,
            "key": injected_key_marker,
        },
    )
    assert unauth_response.status_code == 401

    for response in (auth_response, unauth_response):
        _assert_no_sensitive_literals(
            response.text,
            [
                injected_secret,
                injected_path,
                injected_bearer,
                injected_key_marker,
            ],
        )
