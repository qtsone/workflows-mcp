from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.projects_repo import ProjectCreate, SQLiteProjectsRepository
from workflows_mcp.metadata.repos.tokens_repo import (
    InvalidTokenProjectBindingError,
    RevokedTokenError,
    SQLiteTokensRepository,
    UnknownTokenError,
)
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase2-admin-password"


def _create_project(repo: SQLiteProjectsRepository, *, slug: str, palace: str) -> str:
    created = repo.create(
        ProjectCreate(
            name=f"Project {slug}",
            slug=slug,
            palace=palace,
            default_wing="platform",
            default_room="runtime",
            fs_root="/tmp/workflows",
            fs_allowlist=["/tmp/workflows"],
        )
    )
    return created.id


def _repos(tmp_path: Path) -> tuple[object, SQLiteProjectsRepository, SQLiteTokensRepository]:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    migrate_metadata_db(conn)
    return conn, SQLiteProjectsRepository(conn), SQLiteTokensRepository(conn)


def test_single_project_token_roundtrip_and_no_plaintext_persistence(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="single-token", palace="single-palace")

        created = tokens_repo.create(
            label="ci-agent",
            project_ids=[project_id],
            capabilities={"scopes": ["read:workflows"]},
        )
        assert created.token_secret

        listed = tokens_repo.list_tokens()
        assert len(listed) == 1
        assert listed[0].label == "ci-agent"
        assert listed[0].project_ids == [project_id]
        assert not hasattr(listed[0], "token_secret")

        persisted = conn.execute(
            "SELECT token_hash FROM mcp_tokens WHERE id = ?",
            (created.id,),
        ).fetchone()
        assert persisted is not None
        assert str(persisted[0])
        assert str(persisted[0]) != created.token_secret

        resolved = tokens_repo.resolve(created.token_secret)
        assert resolved.id == created.id
        assert resolved.project_ids == [project_id]
        assert resolved.label == "ci-agent"
    finally:
        conn.close()


def test_multi_project_token_resolves_all_bound_projects(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_a = _create_project(projects_repo, slug="multi-a", palace="multi-palace-a")
        project_b = _create_project(projects_repo, slug="multi-b", palace="multi-palace-b")

        created = tokens_repo.create(
            label="multi-project-token",
            project_ids=[project_a, project_b],
            capabilities={"scopes": ["read:workflows", "execute:workflow"]},
        )

        resolved = tokens_repo.resolve(created.token_secret)
        assert resolved.id == created.id
        assert sorted(resolved.project_ids) == sorted([project_a, project_b])
    finally:
        conn.close()


def test_unknown_and_revoked_tokens_are_rejected(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="reject-token", palace="reject-palace")
        created = tokens_repo.create(
            label="rejectable",
            project_ids=[project_id],
            capabilities={"scopes": ["read:workflows"]},
        )

        with pytest.raises(UnknownTokenError):
            tokens_repo.resolve("not-a-real-token")

        tokens_repo.revoke(created.id)

        with pytest.raises(RevokedTokenError):
            tokens_repo.resolve(created.token_secret)
    finally:
        conn.close()


def test_delete_removes_token_bindings_and_frees_label(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="delete-token", palace="delete-palace")
        created = tokens_repo.create(
            label="deletable-client",
            project_ids=[project_id],
            capabilities={"scopes": ["read:workflows"]},
        )
        tokens_repo.revoke(created.id)

        tokens_repo.delete(created.id)

        assert tokens_repo.list_tokens() == []
        bindings = conn.execute(
            "SELECT COUNT(*) FROM project_token_bindings WHERE token_id = ?",
            (created.id,),
        ).fetchone()
        assert bindings is not None
        assert bindings[0] == 0
        with pytest.raises(UnknownTokenError):
            tokens_repo.resolve(created.token_secret)

        recreated = tokens_repo.create(
            label="deletable-client",
            project_ids=[project_id],
            capabilities={"scopes": ["read:workflows"]},
        )
        assert recreated.id != created.id
    finally:
        conn.close()


def test_mark_last_used_updates_timestamp(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="last-used", palace="last-used-palace")
        created = tokens_repo.create(
            label="used-token",
            project_ids=[project_id],
            capabilities={"scopes": ["read:workflows"]},
        )

        before = conn.execute(
            "SELECT last_used_at FROM mcp_tokens WHERE id = ?",
            (created.id,),
        ).fetchone()
        assert before is not None
        assert before[0] is None

        tokens_repo.mark_last_used(created.id)

        after = conn.execute(
            "SELECT last_used_at FROM mcp_tokens WHERE id = ?",
            (created.id,),
        ).fetchone()
        assert after is not None
        assert after[0] is not None
    finally:
        conn.close()


def test_create_allows_unbound_token_and_rejects_unknown_project_bindings(
    tmp_path: Path,
) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        _create_project(projects_repo, slug="valid-binding", palace="valid-binding-palace")

        created = tokens_repo.create(label="empty", project_ids=[], capabilities={"scopes": []})
        assert created.project_ids == []
        assert tokens_repo.resolve(created.token_secret).project_ids == []
        listed = tokens_repo.list_tokens()
        assert len(listed) == 1
        assert listed[0].project_ids == []

        with pytest.raises(InvalidTokenProjectBindingError):
            tokens_repo.create(
                label="unknown-project",
                project_ids=["does-not-exist"],
                capabilities={"scopes": ["read:workflows"]},
            )
    finally:
        conn.close()


def test_update_project_bindings_allows_bound_and_unbound_tokens(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_a = _create_project(projects_repo, slug="update-a", palace="update-palace-a")
        project_b = _create_project(projects_repo, slug="update-b", palace="update-palace-b")
        created = tokens_repo.create(
            label="editable-token",
            project_ids=[],
            capabilities={"scopes": ["read:workflows"]},
        )

        updated = tokens_repo.update_project_bindings(created.id, [project_b, project_a])
        assert sorted(updated.project_ids) == sorted([project_a, project_b])
        assert sorted(tokens_repo.resolve(created.token_secret).project_ids) == sorted(
            [project_a, project_b]
        )

        unbound = tokens_repo.update_project_bindings(created.id, [])
        assert unbound.project_ids == []
        assert tokens_repo.resolve(created.token_secret).project_ids == []

        with pytest.raises(InvalidTokenProjectBindingError):
            tokens_repo.update_project_bindings(created.id, ["does-not-exist"])
    finally:
        conn.close()


def test_token_hash_is_uniqueness_enforced(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="hash-unique", palace="hash-unique-palace")
        created = tokens_repo.create(
            label="unique-hash-source",
            project_ids=[project_id],
            capabilities={"scopes": ["read:workflows"]},
        )
        source = conn.execute(
            "SELECT token_hash FROM mcp_tokens WHERE id = ?",
            (created.id,),
        ).fetchone()
        assert source is not None
        token_hash = str(source[0])

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO mcp_tokens (id, label, token_hash, capabilities_json)
                VALUES (?, ?, ?, ?)
                """,
                ("duplicate-hash-id", "duplicate-hash-label", token_hash, "{}"),
            )
            conn.commit()
    finally:
        conn.close()


def test_resolve_rejects_malformed_or_oversized_token_input(tmp_path: Path) -> None:
    conn, projects_repo, tokens_repo = _repos(tmp_path)
    try:
        project_id = _create_project(
            projects_repo,
            slug="token-format",
            palace="token-format-palace",
        )
        created = tokens_repo.create(
            label="format-token",
            project_ids=[project_id],
            capabilities={"scopes": ["read:workflows"]},
        )

        assert tokens_repo.resolve(created.token_secret).id == created.id

        with pytest.raises(UnknownTokenError, match="invalid token format"):
            tokens_repo.resolve(created.token_secret + "=")

        with pytest.raises(UnknownTokenError, match="invalid token format"):
            tokens_repo.resolve("a" * 1024)
    finally:
        conn.close()


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )
    return TestClient(build_app(base_dir=base_dir), raise_server_exceptions=False)


def _login_and_csrf(client: TestClient) -> str:
    response = client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert response.status_code == 200
    csrf_token = response.headers.get("X-CSRF-Token") or response.json().get("csrf_token")
    assert csrf_token
    return str(csrf_token)


def _project_payload(*, slug: str, palace: str) -> dict[str, object]:
    return {
        "name": f"Project {slug}",
        "slug": slug,
        "palace": palace,
        "default_wing": "platform",
        "default_room": "runtime",
        "fs_root": "/workspace/workflows",
        "fs_allowlist": ["/workspace/workflows"],
    }


def _error_code(payload: dict[str, object]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            return code
    detail = payload.get("detail")
    if isinstance(detail, dict):
        code = detail.get("code")
        if isinstance(code, str):
            return code
    return None


def _error_message(payload: dict[str, object]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    detail = payload.get("detail")
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str):
            return message
    return None


def _mcp_server_config(snippet: str, server_name: str) -> dict[str, object]:
    parsed = json.loads(snippet)
    assert set(parsed) == {"mcpServers"}
    servers = parsed["mcpServers"]
    assert isinstance(servers, dict)
    assert set(servers) == {server_name}
    server_config = servers[server_name]
    assert isinstance(server_config, dict)
    return server_config


def test_admin_mcp_clients_create_list_revoke_and_secret_one_time(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKFLOWS_MCP_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("WORKFLOWS_PORT", raising=False)
    csrf_token = _login_and_csrf(app_client)

    project_a = app_client.post(
        "/api/admin/v1/projects",
        json=_project_payload(slug="mcp-clients-a", palace="mcp-clients-palace-a"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert project_a.status_code == 201
    project_a_id = project_a.json()["id"]

    project_b = app_client.post(
        "/api/admin/v1/projects",
        json=_project_payload(slug="mcp-clients-b", palace="mcp-clients-palace-b"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert project_b.status_code == 201
    project_b_id = project_b.json()["id"]

    create_response = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={
            "label": "CI Token / Prod!",
            "project_ids": [project_b_id, project_a_id],
            "capabilities": {"scopes": ["read:workflows"]},
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert isinstance(created.get("token"), str)
    assert created["token"]
    assert isinstance(created.get("config_snippet"), str)
    assert created["config_snippet"]
    assert created["token"] in created["config_snippet"]
    server_config = _mcp_server_config(created["config_snippet"], "ci-token-prod")
    assert server_config == {
        "url": "http://127.0.0.1:8000/mcp",
        "headers": {"Authorization": f"Bearer {created['token']}"},
    }
    assert "stdio" not in created["config_snippet"].lower()
    assert "transport" not in server_config
    assert sorted(created["project_ids"]) == sorted([project_a_id, project_b_id])
    token_id = created["id"]

    list_response = app_client.get("/api/admin/v1/mcp-clients")
    assert list_response.status_code == 200
    listed = list_response.json()["mcp_clients"]
    assert len(listed) == 1
    assert listed[0]["id"] == token_id
    assert listed[0]["label"] == "CI Token / Prod!"
    assert sorted(listed[0]["project_ids"]) == sorted([project_a_id, project_b_id])
    assert "token" not in listed[0]
    assert "token_secret" not in listed[0]
    assert "token_hash" not in listed[0]

    revoke_response = app_client.delete(
        f"/api/admin/v1/mcp-clients/{token_id}",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert revoke_response.status_code == 200
    assert revoke_response.json() == {"revoked": True}

    after_revoke = app_client.get("/api/admin/v1/mcp-clients")
    assert after_revoke.status_code == 200
    revoked = after_revoke.json()["mcp_clients"][0]
    assert revoked["id"] == token_id
    assert revoked["revoked_at"] is not None
    assert "token" not in revoked
    assert "token_hash" not in revoked

    delete_response = app_client.delete(
        f"/api/admin/v1/mcp-clients/{token_id}/registration",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True}

    after_delete = app_client.get("/api/admin/v1/mcp-clients")
    assert after_delete.status_code == 200
    assert after_delete.json()["mcp_clients"] == []

    recreate_response = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={
            "label": "CI Token / Prod!",
            "project_ids": [project_a_id],
            "capabilities": {"scopes": ["read:workflows"]},
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert recreate_response.status_code == 201
    assert recreate_response.json()["id"] != token_id


def test_admin_mcp_clients_regenerate_rotates_secret_one_time(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project = app_client.post(
        "/api/admin/v1/projects",
        json=_project_payload(slug="mcp-regenerate", palace="mcp-regenerate-palace"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert project.status_code == 201
    project_id = project.json()["id"]

    create_response = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={
            "label": "regen-token",
            "project_ids": [project_id],
            "capabilities": {"scopes": ["read:workflows"]},
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    token_id = created["id"]
    initial_token = created["token"]

    monkeypatch.setenv(
        "WORKFLOWS_MCP_PUBLIC_BASE_URL",
        "https://workflows.example.com/workflows/mcp/",
    )
    regenerate = app_client.post(
        f"/api/admin/v1/mcp-clients/{token_id}/regenerate",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert regenerate.status_code == 200
    regenerated = regenerate.json()
    assert regenerated["id"] == token_id
    assert regenerated["token"]
    assert regenerated["token"] != initial_token
    server_config = _mcp_server_config(regenerated["config_snippet"], "regen-token")
    assert server_config == {
        "url": "https://workflows.example.com/workflows/mcp",
        "headers": {"Authorization": f"Bearer {regenerated['token']}"},
    }
    assert "token_hash" not in regenerated


def test_admin_mcp_clients_create_unbound_and_update_project_access(
    app_client: TestClient,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_a = app_client.post(
        "/api/admin/v1/projects",
        json=_project_payload(slug="mcp-update-a", palace="mcp-update-palace-a"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert project_a.status_code == 201
    project_a_id = project_a.json()["id"]
    project_b = app_client.post(
        "/api/admin/v1/projects",
        json=_project_payload(slug="mcp-update-b", palace="mcp-update-palace-b"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert project_b.status_code == 201
    project_b_id = project_b.json()["id"]

    create_response = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={"label": "client-selects-project"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["project_ids"] == []
    token_id = created["id"]

    update_response = app_client.patch(
        f"/api/admin/v1/mcp-clients/{token_id}",
        json={"project_ids": [project_b_id, project_a_id]},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert update_response.status_code == 200
    assert sorted(update_response.json()["project_ids"]) == sorted([project_a_id, project_b_id])

    unbind_response = app_client.patch(
        f"/api/admin/v1/mcp-clients/{token_id}",
        json={"project_ids": []},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert unbind_response.status_code == 200
    assert unbind_response.json()["project_ids"] == []

    invalid_project = app_client.patch(
        f"/api/admin/v1/mcp-clients/{token_id}",
        json={"project_ids": ["does-not-exist"]},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert invalid_project.status_code == 400
    assert _error_code(invalid_project.json()) == "invalid_project_binding"


def test_admin_mcp_clients_error_and_auth_semantics(app_client: TestClient) -> None:
    unauth_list = app_client.get("/api/admin/v1/mcp-clients")
    assert unauth_list.status_code == 401

    unauth_create = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={"label": "x", "project_ids": ["missing"], "capabilities": {}},
    )
    assert unauth_create.status_code == 401

    bearer_list = app_client.get(
        "/api/admin/v1/mcp-clients",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert bearer_list.status_code == 403

    csrf_token = _login_and_csrf(app_client)
    project = app_client.post(
        "/api/admin/v1/projects",
        json=_project_payload(slug="mcp-errors", palace="mcp-errors-palace"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert project.status_code == 201
    project_id = project.json()["id"]

    no_csrf_create = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={"label": "x", "project_ids": [project_id], "capabilities": {}},
    )
    assert no_csrf_create.status_code == 403

    valid_create = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={
            "label": "deterministic-label",
            "project_ids": [project_id],
            "capabilities": {"scopes": ["read:workflows"]},
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert valid_create.status_code == 201
    token_id = valid_create.json()["id"]

    duplicate_label = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={
            "label": "deterministic-label",
            "project_ids": [project_id],
            "capabilities": {"scopes": ["read:workflows"]},
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert duplicate_label.status_code == 409
    assert _error_code(duplicate_label.json()) == "token_label_conflict"

    invalid_project = app_client.post(
        "/api/admin/v1/mcp-clients",
        json={
            "label": "invalid-project-binding",
            "project_ids": ["does-not-exist"],
            "capabilities": {"scopes": ["read:workflows"]},
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert invalid_project.status_code == 400
    assert _error_code(invalid_project.json()) == "invalid_project_binding"

    revoke_missing = app_client.delete(
        "/api/admin/v1/mcp-clients/does-not-exist",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert revoke_missing.status_code == 404
    revoke_missing_payload = revoke_missing.json()
    assert _error_code(revoke_missing_payload) == "token_not_found"
    revoke_missing_message = _error_message(revoke_missing_payload)
    assert revoke_missing_message == "Token not found"
    assert "does-not-exist" not in revoke_missing_message

    delete_missing = app_client.delete(
        "/api/admin/v1/mcp-clients/does-not-exist/registration",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert delete_missing.status_code == 404
    delete_missing_payload = delete_missing.json()
    assert _error_code(delete_missing_payload) == "token_not_found"
    delete_missing_message = _error_message(delete_missing_payload)
    assert delete_missing_message == "Token not found"
    assert "does-not-exist" not in delete_missing_message

    regenerate_missing = app_client.post(
        "/api/admin/v1/mcp-clients/does-not-exist/regenerate",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert regenerate_missing.status_code == 404
    regenerate_missing_payload = regenerate_missing.json()
    assert _error_code(regenerate_missing_payload) == "token_not_found"
    regenerate_missing_message = _error_message(regenerate_missing_payload)
    assert regenerate_missing_message == "Token not found"
    assert "does-not-exist" not in regenerate_missing_message

    revoke_without_csrf = app_client.delete(f"/api/admin/v1/mcp-clients/{token_id}")
    assert revoke_without_csrf.status_code == 403

    delete_without_csrf = app_client.delete(f"/api/admin/v1/mcp-clients/{token_id}/registration")
    assert delete_without_csrf.status_code == 403

    regenerate_without_csrf = app_client.post(f"/api/admin/v1/mcp-clients/{token_id}/regenerate")
    assert regenerate_without_csrf.status_code == 403
