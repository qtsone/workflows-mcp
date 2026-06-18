from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
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


def _structured_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "host": "127.0.0.1",
        "port": 5432,
        "database": "workflows",
        "username": "workflows",
        "password": "safe-secret",
        "password_clear": False,
        "ssl_mode": "disable",
        "extra_params": "application_name=workflows",
        "container_name": "workflows-postgres",
        "container_image": "pgvector/pgvector:pg17",
        "container_host_port": 5432,
        "volume_name": "workflows-postgres-data",
        "dsn_import": None,
    }
    body.update(overrides)
    return body


def test_admin_can_save_structured_postgres_settings_and_response_is_safe_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    saved = client.put(
        "/api/admin/v1/database/settings",
        json=_structured_body(),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert saved.status_code == 200
    payload = saved.json()
    assert "enabled" not in payload
    assert payload["configured"] is True
    assert payload["password_configured"] is True
    assert payload["legacy_profile_reentry_required"] is False
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 5432
    assert payload["database"] == "workflows"
    assert payload["username"] == "workflows"
    assert payload["ssl_mode"] == "disable"
    assert payload["container_name"] == "workflows-postgres"
    assert "updated_at" in payload
    rendered = str(payload).lower()
    assert "safe-secret" not in rendered
    assert "postgresql://" not in rendered


def test_admin_database_routes_reject_unauthenticated_access_consistently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    settings_body = _structured_body()

    assert client.get("/api/admin/v1/database/setup").status_code == 401
    assert client.get("/api/admin/v1/database/settings").status_code == 401
    assert client.put("/api/admin/v1/database/settings", json=settings_body).status_code == 401
    assert client.post("/api/admin/v1/database/connection-test").status_code == 401


def test_admin_database_routes_reject_bearer_token_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    settings_body = _structured_body()
    headers = {"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"}

    assert client.get("/api/admin/v1/database/setup", headers=headers).status_code == 403
    assert client.get("/api/admin/v1/database/settings", headers=headers).status_code == 403
    settings_response = client.put(
        "/api/admin/v1/database/settings",
        json=settings_body,
        headers=headers,
    )
    assert settings_response.status_code == 403
    assert client.post("/api/admin/v1/database/connection-test", headers=headers).status_code == 403


def test_admin_settings_put_requires_csrf_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    _login_and_csrf(client)
    response = client.put("/api/admin/v1/database/settings", json=_structured_body())
    assert response.status_code == 403


def test_password_semantics_omitted_null_empty_and_clear_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    assert (
        client.put(
            "/api/admin/v1/database/settings",
            json=_structured_body(password="first-secret"),
            headers={"X-CSRF-Token": csrf_token},
        ).status_code
        == 200
    )
    omitted = client.put(
        "/api/admin/v1/database/settings",
        json=_structured_body(),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert omitted.status_code == 200
    assert omitted.json()["password_configured"] is True

    null_password = client.put(
        "/api/admin/v1/database/settings",
        json=_structured_body(password=None),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert null_password.status_code == 200
    assert null_password.json()["password_configured"] is True

    empty_without_clear = client.put(
        "/api/admin/v1/database/settings",
        json=_structured_body(password="", password_clear=False),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert empty_without_clear.status_code == 422

    clear_password = client.put(
        "/api/admin/v1/database/settings",
        json=_structured_body(password="", password_clear=True),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert clear_password.status_code == 422


def test_dsn_import_parses_to_structured_fields_without_persisting_raw_dsn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    body = _structured_body(password=None)
    body["dsn_import"] = (
        "postgresql://wf_user:super-secret@db.example:5544/wfdb"
        "?sslmode=require"
        "&application_name=a%26b"
        "&search_path=public%3Dprod"
        "&tag=a%2Bb"
    )
    body.pop("password")

    imported = client.put(
        "/api/admin/v1/database/settings",
        json=body,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert imported.status_code == 200
    payload = imported.json()
    assert payload["host"] == "db.example"
    assert payload["port"] == 5544
    assert payload["database"] == "wfdb"
    assert payload["username"] == "wf_user"
    assert payload["ssl_mode"] == "require"
    assert payload["extra_params"] == "application_name=a%26b&search_path=public%3Dprod&tag=a%2Bb"
    assert "application_name=a&b" not in payload["extra_params"]
    assert "postgresql://" not in str(payload)


def test_setup_guidance_uses_saved_profile_and_shell_quotes_metacharacters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    tricky_password = "p@ss ' \" ; $(whoami)"
    container_name = "wf.name-1"
    volume_name = "wf_data-1"
    database = "wf$db 'name'"
    username = "wf user;echo"
    saved = client.put(
        "/api/admin/v1/database/settings",
        json=_structured_body(
            container_name=container_name,
            container_image="pgvector/pgvector:pg17",
            volume_name=volume_name,
            database=database,
            username=username,
            password=tricky_password,
        ),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert saved.status_code == 200
    saved_payload = saved.json()
    assert saved_payload["password_configured"] is True
    assert tricky_password not in str(saved_payload)

    setup = client.get("/api/admin/v1/database/setup")
    assert setup.status_code == 200
    payload = setup.json()
    docker = payload["docker"]
    podman = payload["podman"]
    expected_tokens = [
        shlex.quote(container_name),
        shlex.quote(f"POSTGRES_DB={database}"),
        shlex.quote(f"POSTGRES_USER={username}"),
        shlex.quote("POSTGRES_PASSWORD=<configured-password>"),
        shlex.quote("5432:5432"),
        shlex.quote(f"{volume_name}:/var/lib/postgresql/data"),
        shlex.quote("pgvector/pgvector:pg17"),
    ]
    assert "<configured-password>" in docker
    assert "<configured-password>" in podman
    assert "POSTGRES_PASSWORD=p@ss" not in docker
    assert "POSTGRES_PASSWORD=p@ss" not in podman
    for token in expected_tokens:
        assert token in docker
        assert token in podman
    assert "postgresql://" not in str(payload)


def test_connection_test_uses_structured_runtime_dsn_without_returning_secret(
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
    assert unconfigured_payload["configured"] is False
    assert "postgresql_dsn_missing" in unconfigured_payload["blockers"]
    assert (
        "Save PostgreSQL settings in /api/admin/v1/database/settings."
        in unconfigured_payload["actionable"]
    )

    assert (
        client.put(
            "/api/admin/v1/database/settings",
            json=_structured_body(password="super-secret"),
            headers={"X-CSRF-Token": csrf_token},
        ).status_code
        == 200
    )

    tested = client.post(
        "/api/admin/v1/database/connection-test",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert tested.status_code == 200
    rendered = str(tested.json()).lower()
    assert "super-secret" not in rendered
    assert "postgresql://" not in rendered
