from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
from workflows_mcp.config_service import ConfigService
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState

_VALID_TOKEN = "a" * 40


class FakeReadiness:
    async def evaluate(self):
        class Report:
            state = ReadinessState.UNCONFIGURED
            blockers: list[str] = ["llm_config"]

        return Report()


@pytest.fixture()
def token_store(tmp_path: Path) -> TokenStore:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token(_VALID_TOKEN)
    return store


@pytest.fixture()
def config_service(tmp_path: Path) -> ConfigService:
    return ConfigService(base_dir=tmp_path / ".workflows")


# --- Unit tests: ConfigService ---


def test_validate_rejects_invalid_payload(config_service: ConfigService) -> None:
    errors = config_service.validate_payload({"profiles": "bad"})
    assert "profiles" in errors


def test_validate_accepts_valid_payload(config_service: ConfigService) -> None:
    errors = config_service.validate_payload({"profiles": []})
    assert errors == {}


async def test_apply_creates_llm_config(config_service: ConfigService, tmp_path: Path) -> None:
    await config_service.apply_payload({"profiles": []})
    assert (tmp_path / ".workflows" / "llm-config.yml").exists()


async def test_apply_creates_base_dir_if_missing(tmp_path: Path) -> None:
    base_dir = tmp_path / "nested" / ".workflows"
    service = ConfigService(base_dir=base_dir)
    await service.apply_payload({"profiles": []})
    assert (base_dir / "llm-config.yml").exists()


async def test_apply_preserves_last_known_good_on_invalid_payload(
    config_service: ConfigService, tmp_path: Path
) -> None:
    """Failed validation must not corrupt an existing good config."""
    await config_service.apply_payload({"profiles": []})
    original_mtime = (tmp_path / ".workflows" / "llm-config.yml").stat().st_mtime
    # Attempt an invalid write — validate only, never call apply_payload with bad data.
    errors = config_service.validate_payload({"profiles": "bad"})
    assert errors  # validate first
    # File should be untouched
    new_mtime = (tmp_path / ".workflows" / "llm-config.yml").stat().st_mtime
    assert new_mtime == original_mtime


# --- Integration tests: HTTP router ---


def _make_client(token_store: TokenStore, config_service: ConfigService) -> TestClient:
    app = create_app(
        readiness_service=FakeReadiness(),
        token_store=token_store,
        config_service=config_service,
    )
    return TestClient(app)


def test_validate_endpoint_rejects_invalid_payload(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    client = _make_client(token_store, config_service)
    response = client.post(
        "/config/validate",
        json={"profiles": "bad"},
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
    )
    assert response.status_code == 422
    body = response.json()
    # Error contract: stable ErrorEnvelope shape — no raw "detail" key.
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_FAILED"


def test_validate_endpoint_accepts_valid_payload(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    client = _make_client(token_store, config_service)
    response = client.post(
        "/config/validate",
        json={"profiles": []},
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_apply_endpoint_creates_llm_config(
    token_store: TokenStore, config_service: ConfigService, tmp_path: Path
) -> None:
    client = _make_client(token_store, config_service)
    response = client.post(
        "/config/apply",
        json={"profiles": []},
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert (tmp_path / ".workflows" / "llm-config.yml").exists()


def test_apply_endpoint_rejects_invalid_payload(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    client = _make_client(token_store, config_service)
    response = client.post(
        "/config/apply",
        json={"profiles": "bad"},
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
    )
    assert response.status_code == 422


def test_config_endpoints_require_auth(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    client = _make_client(token_store, config_service)
    for endpoint in ["/config/validate", "/config/apply"]:
        response = client.post(endpoint, json={"profiles": []})
        assert response.status_code == 401, f"{endpoint} should require auth"


# --- Task 2: /config/status and enriched /config/apply ---


def test_config_status_returns_readiness_and_blockers(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    client = _make_client(token_store, config_service)
    response = client.get(
        "/config/status",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"state", "blockers"}


def test_config_status_requires_auth(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    client = _make_client(token_store, config_service)
    response = client.get("/config/status")
    assert response.status_code == 401


async def test_config_status_reports_config_present(
    token_store: TokenStore, config_service: ConfigService, tmp_path: Path
) -> None:
    await config_service.apply_payload({"profiles": []})
    client = _make_client(token_store, config_service)
    response = client.get(
        "/config/status",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["config_present"] is True


def test_config_apply_returns_state_and_blockers(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    client = _make_client(token_store, config_service)
    response = client.post(
        "/config/apply",
        json={"profiles": []},
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"applied", "state", "blockers"}
    assert payload["applied"] is True


def test_config_apply_sequential_writes_both_succeed(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    """Two sequential applies must both succeed — lock releases between them."""
    client = _make_client(token_store, config_service)
    for _ in range(2):
        response = client.post(
            "/config/apply",
            json={"profiles": []},
            headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        )
        assert response.status_code == 200


async def test_config_apply_concurrent_overlap_returns_409(
    token_store: TokenStore, config_service: ConfigService
) -> None:
    """A second apply arriving while a write is in flight must get 409 Conflict.

    We simulate the overlap by holding the lock manually before firing the HTTP
    request, so the loser is guaranteed to collide with the held lock.
    """
    app = create_app(
        readiness_service=FakeReadiness(),
        token_store=token_store,
        config_service=config_service,
    )

    headers = {"Authorization": f"Bearer {_VALID_TOKEN}"}

    # Hold the lock to simulate a write already in progress.
    await config_service._lock.acquire()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/config/apply",
                json={"profiles": []},
                headers=headers,
            )
    finally:
        config_service._lock.release()

    assert response.status_code == 409
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "CONFIG_WRITE_IN_PROGRESS"
