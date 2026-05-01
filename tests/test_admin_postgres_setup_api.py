from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.http_models import ReadinessState
from workflows_mcp.readiness import ReadinessService
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase5-admin-password"


def _bootstrap_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )
    return base_dir


def _client_from_base_dir(base_dir: Path) -> TestClient:
    return TestClient(
        build_app(base_dir=base_dir),
        base_url="http://testserver",
        raise_server_exceptions=False,
    )


def _login_and_csrf(client: TestClient) -> str:
    login_response = client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert login_response.status_code == 200
    csrf_token = login_response.headers.get("X-CSRF-Token") or login_response.json().get(
        "csrf_token"
    )
    assert csrf_token
    return str(csrf_token)


def test_admin_database_setup_guidance_includes_pinned_docker_and_podman_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    _login_and_csrf(client)

    response = client.get("/api/admin/v1/database/setup")
    assert response.status_code == 200

    payload = response.json()
    assert "docker" in payload
    assert "podman" in payload
    assert "pgvector/pgvector:" in payload["docker"]
    assert "pgvector/pgvector:" in payload["podman"]
    assert "latest" not in payload["docker"]
    assert "latest" not in payload["podman"]
    assert "postgres://" not in str(payload).lower()
    assert "super-secret" not in str(payload).lower()
    assert "sk-live" not in str(payload).lower()


def test_admin_database_routes_reject_unauthenticated_access_consistently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    settings_body = {
        "enabled": True,
        "dsn": "postgresql://wf_admin:super-secret@127.0.0.1:5432/workflows",
    }

    assert client.get("/api/admin/v1/database/setup").status_code == 401
    assert client.get("/api/admin/v1/database/settings").status_code == 401
    assert client.put("/api/admin/v1/database/settings", json=settings_body).status_code == 401
    assert client.post("/api/admin/v1/database/connection-test").status_code == 401


def test_admin_database_routes_reject_bearer_token_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    headers = {"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"}
    settings_body = {
        "enabled": True,
        "dsn": "postgresql://wf_admin:super-secret@127.0.0.1:5432/workflows",
    }

    assert client.get("/api/admin/v1/database/setup", headers=headers).status_code == 403
    assert client.get("/api/admin/v1/database/settings", headers=headers).status_code == 403
    assert client.put(
        "/api/admin/v1/database/settings", json=settings_body, headers=headers
    ).status_code == 403
    assert client.post("/api/admin/v1/database/connection-test", headers=headers).status_code == 403


def test_admin_can_save_postgres_settings_with_csrf_and_response_is_safe_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    body = {
        "enabled": True,
        "dsn": "postgresql://wf_admin:super-secret@127.0.0.1:5432/workflows",
    }

    missing_csrf = client.put("/api/admin/v1/database/settings", json=body)
    assert missing_csrf.status_code == 403

    saved = client.put(
        "/api/admin/v1/database/settings",
        json=body,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert saved.status_code == 200
    saved_payload = saved.json()

    assert saved_payload["enabled"] is True
    assert saved_payload["configured"] is True
    assert "updated_at" in saved_payload
    assert "dsn" not in saved_payload
    assert "password" not in str(saved_payload).lower()

    fetched = client.get("/api/admin/v1/database/settings")
    assert fetched.status_code == 200
    fetched_payload = fetched.json()
    assert fetched_payload["enabled"] is True
    assert fetched_payload["configured"] is True
    assert "dsn" not in fetched_payload
    assert "password" not in str(fetched_payload).lower()


def test_connection_test_returns_actionable_degraded_when_unconfigured_or_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    unconfigured = client.post(
        "/api/admin/v1/database/connection-test",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert unconfigured.status_code == 200
    unconfigured_payload = unconfigured.json()
    assert unconfigured_payload["status"] == "degraded"
    assert unconfigured_payload["configured"] is False
    assert unconfigured_payload["ok"] is False
    assert unconfigured_payload["actionable"]

    client.put(
        "/api/admin/v1/database/settings",
        json={"enabled": True, "dsn": "postgresql://wf:bad@127.0.0.1:1/not_real"},
        headers={"X-CSRF-Token": csrf_token},
    )
    unreachable = client.post(
        "/api/admin/v1/database/connection-test",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert unreachable.status_code == 200
    unreachable_payload = unreachable.json()
    assert unreachable_payload["status"] == "degraded"
    assert unreachable_payload["configured"] is True
    assert unreachable_payload["ok"] is False
    assert unreachable_payload["actionable"]


def test_enabled_database_settings_reject_blank_dsn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    response = client.put(
        "/api/admin/v1/database/settings",
        json={"enabled": True, "dsn": "   "},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422


def test_connection_test_surfaces_unexpected_internal_probe_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    save_response = client.put(
        "/api/admin/v1/database/settings",
        json={"enabled": True, "dsn": "postgresql://wf:pw@127.0.0.1:5432/workflows"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert save_response.status_code == 200

    class _BrokenProbe:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def check(self) -> tuple[bool, list[str]]:
            raise RuntimeError("unexpected probe bug")

    import workflows_mcp.http.routes.admin_v1.database as database_module

    monkeypatch.setattr(database_module, "PostgresProbe", _BrokenProbe)

    response = client.post(
        "/api/admin/v1/database/connection-test",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 500


def test_admin_login_and_database_routes_remain_usable_when_ready_is_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Knowledge readiness degradation must not block control-plane admin routes."""
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))

    class _Report:
        state = ReadinessState.PARTIALLY_CONFIGURED
        blockers = ["knowledge_schema_incompatible"]

    async def _degraded_evaluate(self: ReadinessService) -> _Report:
        return _Report()

    monkeypatch.setattr(ReadinessService, "evaluate", _degraded_evaluate)

    ready_response = client.get("/ready")
    assert ready_response.status_code == 503
    ready_payload = ready_response.json()
    assert ready_payload["server_ready"] is True
    assert ready_payload["knowledge_ready"] is False
    assert "knowledge_schema_incompatible" in ready_payload["blockers"]

    csrf_token = _login_and_csrf(client)

    setup = client.get("/api/admin/v1/database/setup")
    assert setup.status_code == 200

    settings = client.get("/api/admin/v1/database/settings")
    assert settings.status_code == 200

    connection_test = client.post(
        "/api/admin/v1/database/connection-test",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert connection_test.status_code == 200
