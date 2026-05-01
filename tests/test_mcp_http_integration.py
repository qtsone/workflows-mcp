"""Integration tests for protected MCP Streamable HTTP transport at ``/mcp``."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from workflows_mcp.auth import TokenStore
from workflows_mcp.http.lifespan import build_resources
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState
from workflows_mcp.metadata.repos.projects_repo import ProjectCreate, SQLiteProjectsRepository
from workflows_mcp.metadata.repos.tokens_repo import SQLiteTokensRepository

_VALID_TOKEN = "a" * 40


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _NotReadyProbe:
    async def check(self) -> tuple[bool, list[str]]:
        return False, ["postgresql_dsn_missing"]


class _ReadyProbe:
    async def check(self) -> tuple[bool, list[str]]:
        return True, []


class _FakeReadinessService:
    def __init__(
        self,
        *,
        ready: bool,
        blockers: list[str] | None = None,
        state: ReadinessState | None = None,
    ) -> None:
        self._ready = ready
        self._blockers = blockers if blockers is not None else ["llm_config"]
        self._state = state

    async def evaluate(self):  # noqa: ANN201
        class _Report:
            pass

        report = _Report()
        if self._ready:
            report.state = ReadinessState.READY  # type: ignore[attr-defined]
            report.blockers = []  # type: ignore[attr-defined]
        else:
            report.state = self._state or ReadinessState.UNCONFIGURED  # type: ignore[attr-defined]
            report.blockers = list(self._blockers)  # type: ignore[attr-defined]
        return report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def token_store(tmp_path: Path) -> TokenStore:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token(_VALID_TOKEN)
    return store


@pytest.fixture()
def not_ready_client(tmp_path: Path, token_store: TokenStore) -> TestClient:
    readiness = _FakeReadinessService(
        ready=False,
        blockers=["workflows_dir"],
        state=ReadinessState.UNCONFIGURED,
    )
    app = create_app(readiness_service=readiness, token_store=token_store)
    sqlite_token, resources = _provision_sqlite_mcp_token(tmp_path)
    app.state.resources = resources
    app.state.sqlite_test_token = sqlite_token
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def ready_client(tmp_path: Path, token_store: TokenStore) -> TestClient:
    readiness = _FakeReadinessService(ready=True)
    app = create_app(readiness_service=readiness, token_store=token_store)
    return TestClient(app, raise_server_exceptions=False)


def _provision_sqlite_mcp_token(tmp_path: Path) -> tuple[str, object]:
    resources = build_resources(base_dir=tmp_path / ".workflows")
    projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
    project = projects_repo.create(
        ProjectCreate(
            name="MCP Integration",
            slug="mcp-integration",
            palace="palace-mcp-integration",
            default_wing="core",
            default_room="main",
            fs_root=str(tmp_path),
            fs_allowlist=[],
        )
    )
    token = SQLiteTokensRepository(resources.metadata_db_conn).create(
        label="mcp-integration-token",
        project_ids=[project.id],
    )
    return token.token_secret, resources


# ---------------------------------------------------------------------------
# AUTH tests (unauthenticated → 401)
# ---------------------------------------------------------------------------


def test_mcp_requires_auth_no_header(not_ready_client: TestClient) -> None:
    """POST /mcp without Authorization header must return 401."""
    response = not_ready_client.post("/mcp", json={})
    assert response.status_code == 401


def test_mcp_requires_auth_wrong_token(not_ready_client: TestClient) -> None:
    """POST /mcp with an invalid token must return 401."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": "Bearer wrong-token"},
        json={},
    )
    assert response.status_code == 401


def test_mcp_401_uses_error_envelope(not_ready_client: TestClient) -> None:
    """401 from /mcp must use the stable ErrorEnvelope shape."""
    response = not_ready_client.post("/mcp", json={})
    assert response.status_code == 401
    payload = response.json()
    assert "error" in payload
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# READINESS gate tests (authenticated but not ready → 409)
# ---------------------------------------------------------------------------


def test_mcp_returns_409_when_not_ready(not_ready_client: TestClient) -> None:
    """Hard readiness blockers still gate /mcp with CONFIG_REQUIRED."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {not_ready_client.app.state.sqlite_test_token}"},
        json={},
    )
    assert response.status_code == 409


def test_mcp_allows_valid_token_when_only_llm_config_blocked(tmp_path: Path) -> None:
    """Soft blockers like llm_config must not block token-auth MCP transport."""
    token_store = TokenStore(tmp_path / "auth.json")
    token_store.write_token(_VALID_TOKEN)
    app = create_app(
        readiness_service=_FakeReadinessService(
            ready=False,
            blockers=["llm_config"],
            state=ReadinessState.UNCONFIGURED,
        ),
        token_store=token_store,
    )
    sqlite_token, resources = _provision_sqlite_mcp_token(tmp_path)
    app.state.resources = resources

    response = TestClient(app, raise_server_exceptions=False).post(
        "/mcp",
        headers={"Authorization": f"Bearer {sqlite_token}"},
        json={"method": "schema"},
    )
    assert response.status_code != 409


def test_mcp_accepts_case_insensitive_bearer_scheme(not_ready_client: TestClient) -> None:
    """Lowercase bearer scheme should still authenticate and hit readiness gate."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": f"bearer {not_ready_client.app.state.sqlite_test_token}"},
        json={},
    )
    assert response.status_code == 409


def test_mcp_rejects_legacy_token_store_token_when_sqlite_has_no_tokens(
    tmp_path: Path, token_store: TokenStore
) -> None:
    """Legacy bootstrap token must never authorize /mcp without SQLite MCP token."""
    app = create_app(
        readiness_service=_FakeReadinessService(ready=True),
        token_store=token_store,
    )
    app.state.resources = build_resources(base_dir=tmp_path / ".workflows")
    response = TestClient(app, raise_server_exceptions=False).post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={},
    )
    assert response.status_code == 401


def test_mcp_fails_closed_when_resources_are_unavailable(
    token_store: TokenStore,
) -> None:
    """/mcp must fail closed when AppResources are missing from app state."""
    app = create_app(
        readiness_service=_FakeReadinessService(ready=True),
        token_store=token_store,
    )
    response = TestClient(app, raise_server_exceptions=False).post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={},
    )
    assert response.status_code == 401


def test_mcp_fails_closed_when_metadata_db_connection_is_closed(
    tmp_path: Path,
    token_store: TokenStore,
) -> None:
    """Closed metadata DB must be treated as invalid auth (401), not 500."""
    app = create_app(
        readiness_service=_FakeReadinessService(ready=True),
        token_store=token_store,
    )
    sqlite_token, resources = _provision_sqlite_mcp_token(tmp_path)
    resources.metadata_db_conn.close()
    app.state.resources = resources

    response = TestClient(app, raise_server_exceptions=False).post(
        "/mcp",
        headers={"Authorization": f"Bearer {sqlite_token}"},
        json={},
    )
    assert response.status_code == 401


def test_mcp_409_uses_error_envelope(not_ready_client: TestClient) -> None:
    """409 from /mcp must use the stable ErrorEnvelope with CONFIG_REQUIRED code."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {not_ready_client.app.state.sqlite_test_token}"},
        json={},
    )
    assert response.status_code == 409
    payload = response.json()
    assert "error" in payload
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "CONFIG_REQUIRED"


def test_mcp_409_message_has_admin_guidance_not_legacy_config(tmp_path: Path) -> None:
    """CONFIG_REQUIRED guidance must not reference legacy /config route."""
    token_store = TokenStore(tmp_path / "auth.json")
    token_store.write_token(_VALID_TOKEN)
    app = create_app(
        readiness_service=_FakeReadinessService(
            ready=False,
            blockers=["workflows_dir"],
            state=ReadinessState.UNCONFIGURED,
        ),
        token_store=token_store,
    )
    sqlite_token, resources = _provision_sqlite_mcp_token(tmp_path)
    app.state.resources = resources

    response = TestClient(app, raise_server_exceptions=False).post(
        "/mcp",
        headers={"Authorization": f"Bearer {sqlite_token}"},
        json={},
    )
    assert response.status_code == 409
    message = response.json()["error"]["message"]
    assert "/config" not in message


def test_mcp_409_includes_readiness_details(not_ready_client: TestClient) -> None:
    """409 from /mcp must include readiness_state in error details."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {not_ready_client.app.state.sqlite_test_token}"},
        json={},
    )
    payload = response.json()
    assert payload["error"]["details"] is not None
    assert "readiness_state" in payload["error"]["details"]


@pytest.mark.anyio
async def test_mcp_streamable_client_can_initialize_and_list_tools_when_ready(
    tmp_path: Path,
) -> None:
    """A real MCP client can initialize + list tools via /mcp with bearer auth."""
    token_store = TokenStore(tmp_path / "auth.json")
    token_store.write_token(_VALID_TOKEN)
    sqlite_token, resources = _provision_sqlite_mcp_token(tmp_path)
    app = create_app(
        readiness_service=_FakeReadinessService(ready=True),
        token_store=token_store,
    )
    app.state.resources = resources

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {sqlite_token}"},
    ) as http_client:
        async with streamable_http_client(
            "http://127.0.0.1/mcp",
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                init_result = await session.initialize()
                assert init_result is not None

                tools = await session.list_tools()
                assert tools.tools
                names = {tool.name for tool in tools.tools}
                assert "list_workflows" in names


def test_mcp_build_app_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_app() wires /mcp correctly — auth guard works end-to-end."""
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
    from workflows_mcp.server import build_app

    app = build_app(base_dir=tmp_path / ".workflows")
    client = TestClient(app, raise_server_exceptions=False)

    # No auth → 401
    response = client.post("/mcp", json={})
    assert response.status_code == 401

    # Bootstrap token is not a valid /mcp auth source without SQLite MCP token.
    response = client.post(
        "/mcp",
        headers={"Authorization": "Bearer 0123456789abcdef0123456789abcdef"},
        json={},
    )
    assert response.status_code == 401
