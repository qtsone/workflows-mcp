"""Snapshot test: OpenAPI schema must match the committed snapshot.

Run the generator when the schema changes:

    uv run python tests/generate_openapi_snapshot.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from workflows_mcp.security.csrf import CSRF_HEADER_NAME
from workflows_mcp.security.sessions import SESSION_COOKIE_NAME

_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "openapi.json"


def _operation(schema: dict, path: str, method: str) -> dict:
    path_item = schema.get("paths", {}).get(path)
    assert isinstance(path_item, dict), f"Path missing from schema: {path}"
    op = path_item.get(method)
    assert isinstance(op, dict), f"Operation missing from schema: {method.upper()} {path}"
    return op


def _build_schema() -> dict:
    """Build the OpenAPI schema from the live app."""
    import os

    from workflows_mcp.auth import TokenStore
    from workflows_mcp.server import build_app

    token = "0123456789abcdef0123456789abcdef"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Pre-seed token store so build_app() doesn't fail.
        store = TokenStore(tmp_path / "auth.json")
        store.write_token(token)
        # WORKFLOWS_BOOTSTRAP_TOKEN must be set for build_app() bootstrap check.
        old = os.environ.get("WORKFLOWS_BOOTSTRAP_TOKEN")
        os.environ["WORKFLOWS_BOOTSTRAP_TOKEN"] = token
        try:
            app = build_app(base_dir=tmp_path)
        finally:
            if old is None:
                os.environ.pop("WORKFLOWS_BOOTSTRAP_TOKEN", None)
            else:
                os.environ["WORKFLOWS_BOOTSTRAP_TOKEN"] = old
    return app.openapi()


class TestOpenAPISchema:
    def test_snapshot_exists(self) -> None:
        """The committed snapshot file must exist."""
        assert _SNAPSHOT_PATH.exists(), (
            f"OpenAPI snapshot not found at {_SNAPSHOT_PATH}. "
            "Run: uv run python tests/generate_openapi_snapshot.py"
        )

    def test_bearer_auth_scheme_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The OpenAPI spec must declare a BearerAuth security scheme."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()
        components = schema.get("components", {})
        security_schemes = components.get("securitySchemes", {})
        assert "BearerAuth" in security_schemes, (
            f"BearerAuth not found in securitySchemes. Got: {list(security_schemes)}"
        )
        bearer = security_schemes["BearerAuth"]
        assert bearer.get("type") == "http"
        assert bearer.get("scheme") == "bearer"

    def test_admin_session_and_csrf_security_schemes_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAPI must declare cookie/header apiKey schemes for UI auth boundaries."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()
        security_schemes = schema.get("components", {}).get("securitySchemes", {})

        assert "AdminSessionCookie" in security_schemes
        session_cookie = security_schemes["AdminSessionCookie"]
        assert session_cookie.get("type") == "apiKey"
        assert session_cookie.get("in") == "cookie"
        assert session_cookie.get("name") == SESSION_COOKIE_NAME

        assert "CsrfToken" in security_schemes
        csrf_token = security_schemes["CsrfToken"]
        assert csrf_token.get("type") == "apiKey"
        assert csrf_token.get("in") == "header"
        assert csrf_token.get("name") == CSRF_HEADER_NAME

    def test_legacy_config_routes_absent_from_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy /config surface must not be present in the default OpenAPI schema."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()
        paths = schema.get("paths", {})

        legacy_config_paths = [p for p in paths if p.startswith("/config")]
        assert legacy_config_paths == [], (
            f"Legacy /config paths should be absent from OpenAPI; found: {legacy_config_paths}"
        )

    def test_ui_session_routes_declare_cookie_auth_security(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session-protected UI routes must declare AdminSessionCookie security."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()

        for path, method in (
            ("/api/admin/v1/auth/session", "get"),
            ("/api/admin/v1/auth/csrf", "get"),
            ("/api/events/v1/system", "get"),
        ):
            op = _operation(schema, path, method)
            security = op.get("security")
            assert security is not None, f"{method.upper()} {path} missing security"
            assert {"AdminSessionCookie": []} in security, (
                f"{method.upper()} {path} must include AdminSessionCookie security"
            )

    def test_system_events_openapi_uses_sse_media_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """System events endpoint must advertise SSE media type in OpenAPI."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()
        op = _operation(schema, "/api/events/v1/system", "get")

        responses = op.get("responses", {})
        ok_response = responses.get("200", {})
        content = ok_response.get("content", {})

        assert "text/event-stream" in content, (
            "GET /api/events/v1/system must document text/event-stream for 200 response"
        )
        assert "application/json" not in content, (
            "GET /api/events/v1/system must not document application/json for 200 response"
        )

    def test_logout_declares_cookie_plus_csrf_in_same_security_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Logout must require both session cookie and CSRF header."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()
        logout = _operation(schema, "/api/admin/v1/auth/logout", "post")
        security = logout.get("security")
        assert security is not None, "POST /api/admin/v1/auth/logout missing security"
        assert {"AdminSessionCookie": [], "CsrfToken": []} in security, (
            "POST /api/admin/v1/auth/logout must include both AdminSessionCookie and "
            "CsrfToken in same security requirement"
        )

    def test_public_routes_remain_without_auth_security(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Public endpoints must not declare auth security requirements."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()

        for path, method in (
            ("/api/admin/v1/auth/login", "post"),
            ("/api/public/v1/system/status", "get"),
            ("/health", "get"),
            ("/ready", "get"),
        ):
            op = _operation(schema, path, method)
            assert "security" not in op, f"{method.upper()} {path} must be public (no security)"

    def test_schema_matches_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Live schema must match the committed snapshot.

        If this test fails, regenerate with:
            uv run python tests/generate_openapi_snapshot.py
        """
        assert _SNAPSHOT_PATH.exists(), (
            f"OpenAPI snapshot not found at {_SNAPSHOT_PATH}. "
            "Run: uv run python tests/generate_openapi_snapshot.py"
        )
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        current = _build_schema()
        snapshot = json.loads(_SNAPSHOT_PATH.read_text())

        assert current == snapshot, (
            "OpenAPI schema has drifted. Run: uv run python tests/generate_openapi_snapshot.py"
        )
