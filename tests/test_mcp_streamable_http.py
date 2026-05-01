from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
from workflows_mcp.config_service import ConfigService
from workflows_mcp.http.lifespan import build_resources
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState
from workflows_mcp.metadata.repos.projects_repo import ProjectCreate, SQLiteProjectsRepository
from workflows_mcp.metadata.repos.tokens_repo import SQLiteTokensRepository

_VALID_TOKEN = "a" * 40


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
def ready_client(tmp_path: Path) -> TestClient:
    token_store = TokenStore(tmp_path / "auth.json")
    token_store.write_token(_VALID_TOKEN)
    app = create_app(
        readiness_service=_FakeReadiness(ReadinessState.READY),
        token_store=token_store,
        config_service=ConfigService(base_dir=tmp_path / ".workflows"),
    )
    resources = build_resources(base_dir=tmp_path / ".workflows")
    project = SQLiteProjectsRepository(resources.metadata_db_conn).create(
        ProjectCreate(
            name="MCP Streamable",
            slug="mcp-streamable",
            palace="palace-mcp-streamable",
            default_wing="core",
            default_room="main",
            fs_root=str(tmp_path),
            fs_allowlist=[],
        )
    )
    sqlite_token = SQLiteTokensRepository(resources.metadata_db_conn).create(
        label="mcp-streamable-token",
        project_ids=[project.id],
    )
    app.state.resources = resources
    app.state.sqlite_test_token = sqlite_token.token_secret
    return TestClient(app, raise_server_exceptions=False)


def test_mcp_no_longer_behaves_like_schema_only_json_shim(ready_client: TestClient) -> None:
    """Legacy JSON-RPC-ish schema payload should not return shim success shape."""
    response = ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {ready_client.app.state.sqlite_test_token}"},
        json={"method": "schema"},
    )
    assert response.status_code >= 400


def test_mcp_rejects_ui_cookie_without_bearer_even_when_ready(ready_client: TestClient) -> None:
    """Session cookie alone must not authorize MCP transport."""
    ready_client.cookies.set("workflows_admin_session", "fake")
    response = ready_client.post("/mcp", json={})
    assert response.status_code == 401


def test_mcp_transport_does_not_use_fastmcp_private_tool_internals() -> None:
    """Mounted MCP transport must avoid SDK-private tool registry cloning."""
    source = (
        Path(__file__).resolve().parents[1] / "src/workflows_mcp/http/mcp_transport.py"
    ).read_text(encoding="utf-8")

    assert "_tool_manager" not in source
    assert re.search(r"[\"']_tools[\"']|\._tools\b", source) is None
    assert "canonical_mcp" not in source
