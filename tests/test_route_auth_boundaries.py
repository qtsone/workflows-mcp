from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase2-admin-password"


class _FakeReadinessReport:
    def __init__(self, state: ReadinessState) -> None:
        self.state = state
        self.blockers: list[str] = []


class _FakeReadiness:
    def __init__(self, state: ReadinessState) -> None:
        self._state = state

    async def evaluate(self) -> _FakeReadinessReport:
        return _FakeReadinessReport(self._state)


@pytest.fixture()
def auth_app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )
    return TestClient(build_app(base_dir=base_dir), raise_server_exceptions=False)


def test_public_routes_remain_public_with_frontend_assets(tmp_path: Path) -> None:
    token_store = TokenStore(tmp_path / "auth.json")
    token_store.write_token(_MCP_BOOTSTRAP_TOKEN)

    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text("<html><body>admin-ui</body></html>", encoding="utf-8")

    app = create_app(
        readiness_service=_FakeReadiness(ReadinessState.READY),
        token_store=token_store,
        frontend_static_dir=static_dir,
        require_frontend_assets=True,
    )
    client = TestClient(app, raise_server_exceptions=False)

    for path in (
        "/docs",
        "/openapi.json",
        "/health",
        "/ready",
        "/",
        "/login",
        "/api/public/v1/system/status",
    ):
        response = client.get(path)
        assert response.status_code == 200, f"Expected public route {path} to remain public"


def test_unknown_public_v1_path_is_not_reclassified_as_admin_auth_failure(
    auth_app_client: TestClient,
) -> None:
    response = auth_app_client.get("/api/public/v1/does-not-exist")
    assert response.status_code == 404
    assert response.status_code not in {401, 403}


def test_legacy_config_routes_are_not_exposed_on_default_http_surface(
    auth_app_client: TestClient,
) -> None:
    """Legacy /config control-plane routes must not be mounted by default."""
    legacy_paths = (
        ("GET", "/config"),
        ("GET", "/config/status"),
        ("POST", "/config/apply"),
        ("POST", "/config/credentials/rotate"),
        ("POST", "/config/credentials/revoke"),
    )

    for method, path in legacy_paths:
        response = auth_app_client.request(method, path)
        assert response.status_code == 404, (
            f"Expected legacy route {method} {path} to be unmounted, "
            f"got status {response.status_code}"
        )


def test_events_endpoint_requires_ui_session_and_rejects_mcp_bearer_token(
    auth_app_client: TestClient,
) -> None:
    unauthenticated = auth_app_client.get("/api/events/v1/system")
    assert unauthenticated.status_code == 401

    mcp_bearer = auth_app_client.get(
        "/api/events/v1/system",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert mcp_bearer.status_code in {401, 403}

    login = auth_app_client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert login.status_code == 200

    with_ui_session = auth_app_client.get("/api/events/v1/system")
    assert with_ui_session.status_code == 200
    assert with_ui_session.headers["content-type"].startswith("text/event-stream")


def test_mcp_rejects_ui_session_only_requests(auth_app_client: TestClient) -> None:
    login = auth_app_client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert login.status_code == 200

    response = auth_app_client.post("/mcp", json={"method": "schema"})
    assert response.status_code in {401, 403}


def test_public_status_exposes_non_loopback_bind_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    monkeypatch.setenv("WORKFLOWS_BIND_HOST", "0.0.0.0")
    bootstrap_if_needed(
        config_dir=base_dir,
        host="0.0.0.0",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )

    client = TestClient(build_app(base_dir=base_dir), raise_server_exceptions=False)
    response = client.get("/api/public/v1/system/status")
    assert response.status_code == 200
    payload = response.json()

    warning_candidates = [
        str(payload.get("warning", "")),
        " ".join(payload.get("warnings", [])) if isinstance(payload.get("warnings"), list) else "",
    ]
    assert any(
        "0.0.0.0" in warning and "loopback" in warning.lower()
        for warning in warning_candidates
    )
