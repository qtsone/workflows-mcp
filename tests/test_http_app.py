from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
from workflows_mcp.http.static import FrontendAssetsMissingError, is_reserved_path
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState

_VALID_TOKEN = "a" * 40


class FakeReadinessReport:
    def __init__(self, state: ReadinessState) -> None:
        self.state = state
        self.blockers: list[str] = []


class FakeReadiness:
    def __init__(self, state: ReadinessState) -> None:
        self._state = state

    async def evaluate(self) -> FakeReadinessReport:
        return FakeReadinessReport(self._state)


@pytest.fixture()
def token_store(tmp_path: Path) -> TokenStore:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token(_VALID_TOKEN)
    return store


def test_docs_is_public(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.UNCONFIGURED),
        token_store=token_store,
    )
    client = TestClient(app)
    assert client.get("/docs").status_code == 200


def test_ready_is_503_when_not_ready(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.UNCONFIGURED),
        token_store=token_store,
    )
    client = TestClient(app)
    assert client.get("/ready").status_code == 503


def test_protected_route_requires_token(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)
    response = client.get("/api/admin/v1/auth/session")
    assert response.status_code == 401


def test_protected_route_rejects_wrong_token(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)
    response = client.get(
        "/api/admin/v1/auth/session", headers={"Authorization": "Bearer wrongtoken"}
    )
    # Admin v1 session endpoints only accept UI session cookies; MCP bearer
    # tokens are rejected as unauthorized/forbidden.
    assert response.status_code in {401, 403}


def test_protected_route_rejects_even_valid_mcp_token_for_ui_session_endpoint(
    token_store: TokenStore,
) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)
    response = client.get(
        "/api/admin/v1/auth/session", headers={"Authorization": f"Bearer {_VALID_TOKEN}"}
    )
    # Admin v1 session endpoints are cookie-authenticated and must not accept
    # MCP bearer tokens, even if those tokens are valid elsewhere.
    assert response.status_code in {401, 403}


def test_public_v1_status_namespace_exists(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)

    response = client.get("/api/public/v1/system/status")
    assert response.status_code in {200, 503}


def test_admin_v1_session_namespace_exists(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)

    response = client.get("/api/admin/v1/auth/session")
    assert response.status_code in {200, 401}


def test_events_v1_system_namespace_exists(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)

    response = client.get("/api/events/v1/system")
    assert response.status_code in {200, 401, 405}
    if response.status_code == 200:
        assert response.headers["content-type"].startswith("text/event-stream")


def test_spa_fallback_serves_ui_route_but_never_swallows_reserved_paths(
    tmp_path: Path, token_store: TokenStore
) -> None:
    static_dir = tmp_path / "admin-static"
    static_dir.mkdir()
    index_html = static_dir / "index.html"
    index_html.write_text("<html><body>phase-0-ui</body></html>", encoding="utf-8")

    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
        frontend_static_dir=static_dir,
    )
    client = TestClient(app)

    ui_response = client.get("/admin/projects/abc")
    assert ui_response.status_code == 200
    assert "phase-0-ui" in ui_response.text

    reserved_paths = [
        "/api",
        "/api/",
        "/api/not-a-ui-route",
        "/mcp",
        "/health",
        "/ready",
        "/docs",
        "/openapi.json",
    ]
    for path in reserved_paths:
        response = client.get(path)
        assert not (response.status_code == 200 and "phase-0-ui" in response.text), (
            f"Reserved path {path!r} was swallowed by SPA fallback"
        )


def test_is_reserved_path_treats_api_root_variants_as_reserved() -> None:
    assert is_reserved_path("/api") is True
    assert is_reserved_path("/api/") is True


def test_require_frontend_assets_raises_clear_error_when_index_missing(
    tmp_path: Path, token_store: TokenStore
) -> None:
    static_dir = tmp_path / "missing-index"
    static_dir.mkdir()

    with pytest.raises(FrontendAssetsMissingError) as exc_info:
        create_app(
            readiness_service=FakeReadiness(ReadinessState.READY),
            token_store=token_store,
            frontend_static_dir=static_dir,
            require_frontend_assets=True,
        )

    message = str(exc_info.value)
    assert "index.html" in message
    assert str(static_dir / "index.html") in message
    assert "build" in message or "package" in message


def test_optional_frontend_assets_do_not_raise_when_index_missing_and_api_still_works(
    tmp_path: Path, token_store: TokenStore
) -> None:
    static_dir = tmp_path / "incomplete-static"
    static_dir.mkdir()

    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
        frontend_static_dir=static_dir,
        require_frontend_assets=False,
    )
    client = TestClient(app)

    api_response = client.get("/health")
    assert api_response.status_code == 200

    ui_like_response = client.get("/admin/projects/abc")
    assert ui_like_response.status_code == 404
