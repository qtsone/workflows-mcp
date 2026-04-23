from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
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
    response = client.get("/config")
    assert response.status_code == 401


def test_protected_route_rejects_wrong_token(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)
    response = client.get("/config", headers={"Authorization": "Bearer wrongtoken"})
    assert response.status_code == 401


def test_protected_route_accepts_valid_token(token_store: TokenStore) -> None:
    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
    )
    client = TestClient(app)
    response = client.get("/config", headers={"Authorization": f"Bearer {_VALID_TOKEN}"})
    assert response.status_code == 200
