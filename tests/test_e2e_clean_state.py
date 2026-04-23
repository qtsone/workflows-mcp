"""End-to-end clean-state release gate.

Exercises the full HTTP service startup sequence from a clean tmpdir with no
auth.json or config files.  No PostgreSQL is required — the readiness state
"not_ready" (or any non-READY state) is acceptable.
"""

import pytest
from fastapi.testclient import TestClient

# 64-char hex token — satisfies the 32-byte minimum.
_TOKEN = "a" * 64


class TestCleanStateE2E:
    """Full HTTP service startup sequence from a clean state."""

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        """Build a TestClient from a fresh base_dir with a valid bootstrap token."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _TOKEN)
        from workflows_mcp.server import build_app

        app = build_app(base_dir=tmp_path / ".workflows")
        return TestClient(app, raise_server_exceptions=False)

    def test_step1_build_app_succeeds_with_clean_tmpdir(self, tmp_path, monkeypatch):
        """build_app() must succeed and return a FastAPI app from a clean tmpdir."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _TOKEN)
        from fastapi import FastAPI

        from workflows_mcp.server import build_app

        app = build_app(base_dir=tmp_path / ".workflows")
        assert isinstance(app, FastAPI)

    def test_step2_build_app_fails_without_token(self, tmp_path, monkeypatch):
        """build_app() must raise BootstrapTokenError when token env var is absent."""
        monkeypatch.delenv("WORKFLOWS_BOOTSTRAP_TOKEN", raising=False)
        from workflows_mcp.auth import BootstrapTokenError
        from workflows_mcp.server import build_app

        with pytest.raises(BootstrapTokenError):
            build_app(base_dir=tmp_path / ".workflows")

    def test_step3_health_returns_200_ok(self, client):
        """GET /health must return 200 {"status": "ok"}."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_step4_readiness_returns_200_not_500(self, client):
        """GET /ready must not return 500; state may be not_ready when postgres absent."""
        response = client.get("/ready")
        # 200 (ready) or 503 (not ready) are both acceptable;
        # 500 indicates an implementation error.
        assert response.status_code != 500
        body = response.json()
        assert "state" in body

    def test_step5_config_without_token_returns_401(self, client):
        """GET /config without bearer token must return 401."""
        response = client.get("/config")
        assert response.status_code == 401

    def test_step6_config_with_valid_token_returns_200_with_state_and_blockers(self, client):
        """GET /config with valid bearer token must return 200 with state and blockers."""
        response = client.get("/config", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert response.status_code == 200
        body = response.json()
        assert "state" in body
        assert "blockers" in body

    def test_step7_mcp_without_token_returns_401(self, client):
        """POST /mcp without bearer token must return 401."""
        response = client.post("/mcp", json={})
        assert response.status_code == 401

    def test_step8_mcp_with_token_returns_json_not_500_not_404(self, client):
        """POST /mcp with valid bearer token must return a JSON body, not 500, not 404."""
        response = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {_TOKEN}"},
            json={"method": "schema"},
        )
        # 404 means the route is missing; 500 is an implementation error.
        # 409 (CONFIG_REQUIRED), 400, or 200 are all acceptable for this gate.
        assert response.status_code not in (404, 500)
        # Must be valid JSON.
        body = response.json()
        assert body is not None

    def test_step9_openapi_json_returns_200_with_bearer_auth(self, client):
        """GET /openapi.json must return 200 and contain BearerAuth."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "BearerAuth" in response.text

    def test_step10_docs_returns_200(self, client):
        """GET /docs must return 200."""
        response = client.get("/docs")
        assert response.status_code == 200
