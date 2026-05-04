"""Phase 11 clean-state end-to-end HTTP flow.

This gate validates a real control-plane bootstrap path from a fresh base dir:
admin login/session + CSRF, DB setup/settings, LLM profile save, encrypted
secret metadata behavior, project onboarding metadata, MCP client token minting,
authenticated /mcp call, dirty/sync behavior after file mutation, and run
history list/detail contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.http.routes.admin_v1 import sync as sync_routes
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.run_history_repo import SQLiteRunHistoryRepository
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase11-clean-state-password"


@pytest.fixture()
def clean_state_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
    login = client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert login.status_code == 200
    csrf_token = login.headers.get("X-CSRF-Token") or login.json().get("csrf_token")
    assert csrf_token
    return str(csrf_token)


def _seed_run(client: TestClient, *, run_id: str, project_id: str, token_id: str) -> None:
    resources = client.app.state.resources
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteRunHistoryRepository(conn)
        repo.create_run(
            run_id=run_id,
            workflow_name="phase11-e2e-flow",
            status="completed",
            timeout_seconds=300,
            created_at="2026-04-29T12:00:00Z",
            started_at="2026-04-29T12:00:01Z",
            updated_at="2026-04-29T12:00:03Z",
            project_id=project_id,
            token_id=token_id,
            cancellable=False,
            result_summary="ok",
            error_summary=None,
        )
        repo.update_run(
            run_id=run_id,
            status="completed",
            updated_at="2026-04-29T12:00:03Z",
            finished_at="2026-04-29T12:00:03Z",
            cancellable=False,
            result_summary="ok",
            error_summary=None,
        )
    finally:
        conn.close()


def test_phase11_clean_state_http_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_state_client: TestClient,
) -> None:
    csrf_token = _login_and_csrf(clean_state_client)
    persisted_projects: list[str] = []

    async def fake_persist_project_graph_from_scan(resources: Any, project: Any) -> dict[str, int]:
        persisted_projects.append(str(project.id))
        return {"nodes": 1, "corridors": 0}

    monkeypatch.setattr(
        sync_routes,
        "_persist_project_graph_from_scan",
        fake_persist_project_graph_from_scan,
    )

    setup_response = clean_state_client.get("/api/admin/v1/database/setup")
    assert setup_response.status_code == 200
    setup_payload = setup_response.json()
    assert "docker" in setup_payload
    assert "podman" in setup_payload

    db_settings = clean_state_client.put(
        "/api/admin/v1/database/settings",
        json={
            "enabled": True,
            "dsn_import": "postgresql://wf_admin:super-secret@127.0.0.1:1/workflows",
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert db_settings.status_code == 200
    db_payload = db_settings.json()
    assert db_payload["enabled"] is True
    assert db_payload["configured"] is True
    assert db_payload["password_configured"] is True
    assert "dsn" not in db_payload
    assert "postgresql://" not in str(db_payload).lower()
    assert "super-secret" not in str(db_payload).lower()

    llm_put = clean_state_client.put(
        "/api/admin/v1/llm/config",
        json={
            "version": "1.0",
            "providers": {
                "openai-cloud": {
                    "type": "openai",
                    "api_url": "https://api.openai.com/v1/chat/completions",
                    "api_key_secret": "OPENAI_API_KEY",
                }
            },
            "profiles": {
                "quick": {
                    "provider": "openai-cloud",
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "max_tokens": 800,
                }
            },
            "default_profile": "quick",
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert llm_put.status_code == 200
    llm_payload = llm_put.json()
    assert llm_payload["default_profile"] == "quick"
    assert llm_payload["providers"]["openai-cloud"]["api_key_secret"] == "OPENAI_API_KEY"
    assert "api_key_value" not in str(llm_payload)
    assert "sk-" not in str(llm_payload)

    synthetic_secret_value = "phase11-synthetic-secret-value"

    secret_create = clean_state_client.post(
        "/api/admin/v1/secrets",
        json={"name": "OPENAI_API_KEY", "value": synthetic_secret_value, "key_id": "phase11"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert secret_create.status_code == 200
    secret_payload = secret_create.json()
    assert set(secret_payload.keys()) == {"name", "key_id", "created_at", "updated_at"}
    assert "value" not in secret_payload
    assert "ciphertext" not in secret_payload

    secret_list = clean_state_client.get("/api/admin/v1/secrets")
    assert secret_list.status_code == 200
    listed_secrets = secret_list.json()["secrets"]
    openai_secret = next(item for item in listed_secrets if item["name"] == "OPENAI_API_KEY")
    assert "value" not in openai_secret
    assert "ciphertext" not in openai_secret
    assert synthetic_secret_value not in str(secret_list.json())

    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True, exist_ok=True)
    created_project = clean_state_client.post(
        "/api/admin/v1/projects",
        json={
            "name": "Phase 11 Project",
            "slug": "phase11-project",
            "palace": "phase11-palace",
            "default_wing": "platform",
            "default_room": "runtime",
            "fs_root": str(project_root),
            "fs_allowlist": [str(project_root)],
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created_project.status_code == 201
    project_payload = created_project.json()
    project_id = project_payload["id"]
    assert project_payload["slug"] == "phase11-project"
    assert project_payload["palace"] == "phase11-palace"
    assert project_payload["default_wing"] == "platform"
    assert project_payload["default_room"] == "runtime"

    mcp_client = clean_state_client.post(
        "/api/admin/v1/mcp-clients",
        json={
            "label": "phase11-mcp-client",
            "project_ids": [project_id],
            "capabilities": {"scopes": ["read:workflows"]},
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert mcp_client.status_code == 201
    mcp_client_payload = mcp_client.json()
    mcp_token = mcp_client_payload["token"]
    token_id = mcp_client_payload["id"]
    assert mcp_token
    assert "token_hash" not in mcp_client_payload

    mcp_call = clean_state_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {mcp_token}"},
        json={"method": "schema"},
    )
    assert mcp_call.status_code != 401
    assert mcp_call.status_code != 409

    (project_root / "workflow.yaml").write_text("name: phase11\n", encoding="utf-8")

    reconcile = clean_state_client.post(
        f"/api/admin/v1/sync/{project_id}/reconcile",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reconcile.status_code == 200
    assert reconcile.json()["requires_reconciliation"] is True

    sync_now = clean_state_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert sync_now.status_code == 200
    sync_payload = sync_now.json()
    assert sync_payload["project_id"] == project_id
    assert sync_payload["dirty_count"] == 0
    assert persisted_projects == [project_id, project_id]

    _seed_run(
        clean_state_client,
        run_id="phase11-run-001",
        project_id=project_id,
        token_id=token_id,
    )

    runs_list = clean_state_client.get("/api/admin/v1/runs", params={"limit": 10, "offset": 0})
    assert runs_list.status_code == 200
    run_rows = runs_list.json()["runs"]
    assert any(item["run_id"] == "phase11-run-001" for item in run_rows)

    runs_detail = clean_state_client.get("/api/admin/v1/runs/phase11-run-001")
    assert runs_detail.status_code == 200
    run_detail = runs_detail.json()
    assert run_detail["run_id"] == "phase11-run-001"
    assert run_detail["project_id"] == project_id
    assert run_detail["token_id"] == token_id
    assert run_detail["result_summary"] == "ok"
