from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.engine.workflow_source_loader import WorkflowSourceReloadError
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase2-admin-password"


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


def _project_create_payload(*, slug: str, palace: str) -> dict[str, object]:
    return {
        "name": "Workflow Service",
        "slug": slug,
        "palace": palace,
        "default_wing": "platform",
        "default_room": "runtime",
        "fs_root": "/workspace/workflows",
        "fs_allowlist": ["/workspace/workflows"],
    }


def _create_project(client: TestClient, csrf_token: str) -> str:
    response = client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="wf-service", palace="wf-palace"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _write_workflow_yaml(directory: Path, *, filename: str, name: str) -> Path:
    target = directory / filename
    target.write_text(
        "\n".join(
            [
                f"name: {name}",
                f"description: workflow {name}",
                "version: 1.0.0",
                "tags:",
                "  - admin",
                "blocks:",
                "  - id: run",
                "    type: Shell",
                "    inputs:",
                "      command: \"printf 'ok'\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def _write_invalid_workflow_yaml(directory: Path, *, filename: str) -> Path:
    target = directory / filename
    target.write_text("name: invalid\nblocks: [", encoding="utf-8")
    return target


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


def test_admin_workflow_sources_and_registry_happy_path(
    app_client: TestClient, tmp_path: Path
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_id = _create_project(app_client, csrf_token)

    workflow_dir = tmp_path / "wf-source"
    workflow_dir.mkdir(parents=True)
    _write_workflow_yaml(workflow_dir, filename="good.yaml", name="wf-admin-good")

    create_source = app_client.post(
        "/api/admin/v1/workflows/sources",
        json={
            "project_id": project_id,
            "source_path": str(workflow_dir),
            "checksum": None,
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_source.status_code == 201
    source = create_source.json()
    source_id = source["source_id"]
    assert source["project_id"] == project_id

    list_sources = app_client.get("/api/admin/v1/workflows/sources")
    assert list_sources.status_code == 200
    listed_sources = list_sources.json()["sources"]
    assert [item["source_id"] for item in listed_sources] == [source_id]

    validate_source = app_client.post(
        f"/api/admin/v1/workflows/sources/{source_id}/validate",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert validate_source.status_code == 200
    validate_payload = validate_source.json()
    assert validate_payload["valid"] is True
    assert validate_payload["total"] == 1
    assert validate_payload["workflow_names"] == ["wf-admin-good"]

    reload_response = app_client.post(
        "/api/admin/v1/workflows/reload",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reload_response.status_code == 200
    reload_payload = reload_response.json()
    assert reload_payload["status"] == "ok"
    assert reload_payload["total"] == 1
    assert reload_payload["source_count"] == 1
    assert reload_payload["workflow_names"] == ["wf-admin-good"]

    list_workflows = app_client.get("/api/admin/v1/workflows")
    assert list_workflows.status_code == 200
    workflows = list_workflows.json()["workflows"]
    assert len(workflows) == 1
    assert workflows[0]["name"] == "wf-admin-good"
    assert workflows[0]["source_path"]

    detail_response = app_client.get("/api/admin/v1/workflows/wf-admin-good")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["name"] == "wf-admin-good"
    assert detail_payload["description"] == "workflow wf-admin-good"
    assert detail_payload["source_path"]
    assert detail_payload["raw_yaml"] is not None
    assert "name: wf-admin-good" in detail_payload["raw_yaml"]
    assert detail_payload["yaml_path"] == str(workflow_dir / "good.yaml")
    assert detail_payload["load_logs"]

    schema_response = app_client.get("/api/admin/v1/workflows/schema")
    assert schema_response.status_code == 200
    schema_payload = schema_response.json()
    assert isinstance(schema_payload, dict)
    required = schema_payload.get("required")
    assert isinstance(required, list)
    assert "blocks" in required

    delete_source = app_client.delete(
        f"/api/admin/v1/workflows/sources/{source_id}",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert delete_source.status_code == 200
    assert delete_source.json() == {"deleted": True}

    reload_empty = app_client.post(
        "/api/admin/v1/workflows/reload",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reload_empty.status_code == 200
    assert reload_empty.json()["total"] == 0

    list_empty = app_client.get("/api/admin/v1/workflows")
    assert list_empty.status_code == 200
    assert list_empty.json()["workflows"] == []


def test_workflow_sources_auth_and_csrf_boundaries(app_client: TestClient, tmp_path: Path) -> None:
    unauth_list = app_client.get("/api/admin/v1/workflows/sources")
    assert unauth_list.status_code == 401

    csrf_token = _login_and_csrf(app_client)
    project_id = _create_project(app_client, csrf_token)
    workflow_dir = tmp_path / "wf-source-auth"
    workflow_dir.mkdir(parents=True)

    missing_csrf = app_client.post(
        "/api/admin/v1/workflows/sources",
        json={"project_id": project_id, "source_path": str(workflow_dir)},
    )
    assert missing_csrf.status_code == 403


def test_workflow_sources_duplicate_conflict_code(app_client: TestClient, tmp_path: Path) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_id = _create_project(app_client, csrf_token)
    workflow_dir = tmp_path / "wf-source-duplicate"
    workflow_dir.mkdir(parents=True)

    first = app_client.post(
        "/api/admin/v1/workflows/sources",
        json={"project_id": project_id, "source_path": str(workflow_dir)},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert first.status_code == 201

    second = app_client.post(
        "/api/admin/v1/workflows/sources",
        json={"project_id": project_id, "source_path": str(workflow_dir)},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert second.status_code == 409
    assert _error_code(second.json()) == "workflow_source_conflict"


def test_reload_invalid_workflow_does_not_mutate_live_registry(
    app_client: TestClient, tmp_path: Path
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_id = _create_project(app_client, csrf_token)

    good_source = tmp_path / "wf-source-good"
    good_source.mkdir(parents=True)
    _write_workflow_yaml(good_source, filename="good.yaml", name="wf-admin-good")

    create_good = app_client.post(
        "/api/admin/v1/workflows/sources",
        json={"project_id": project_id, "source_path": str(good_source)},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_good.status_code == 201

    first_reload = app_client.post(
        "/api/admin/v1/workflows/reload",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert first_reload.status_code == 200
    assert first_reload.json()["workflow_names"] == ["wf-admin-good"]

    bad_source = tmp_path / "wf-source-bad"
    bad_source.mkdir(parents=True)
    _write_invalid_workflow_yaml(bad_source, filename="broken.yaml")
    create_bad = app_client.post(
        "/api/admin/v1/workflows/sources",
        json={"project_id": project_id, "source_path": str(bad_source)},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_bad.status_code == 201

    failed_reload = app_client.post(
        "/api/admin/v1/workflows/reload",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert failed_reload.status_code == 422
    assert _error_code(failed_reload.json()) == "workflow_invalid_definition"

    list_after_failure = app_client.get("/api/admin/v1/workflows")
    assert list_after_failure.status_code == 200
    listed = list_after_failure.json()["workflows"]
    assert [item["name"] for item in listed] == ["wf-admin-good"]


@pytest.mark.parametrize(
    ("reload_error", "expected_status", "expected_code"),
    [
        (
            WorkflowSourceReloadError(
                code="workflow_duplicate_name",
                message="Duplicate workflow name 'dup' found",
            ),
            409,
            "workflow_duplicate_name",
        ),
        (
            WorkflowSourceReloadError(
                code="workflow_invalid_definition",
                message="Invalid workflow definition at '/tmp/broken.yaml'",
            ),
            422,
            "workflow_invalid_definition",
        ),
    ],
)
def test_reload_callback_errors_are_mapped_to_structured_admin_errors(
    app_client: TestClient,
    reload_error: WorkflowSourceReloadError,
    expected_status: int,
    expected_code: str,
) -> None:
    csrf_token = _login_and_csrf(app_client)

    def _raise_reload_error() -> object:
        raise reload_error

    app_client.app.state.resources.app_context.reload_workflows = _raise_reload_error

    response = app_client.post(
        "/api/admin/v1/workflows/reload",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == expected_status
    assert _error_code(response.json()) == expected_code


def test_workflow_detail_reports_missing_raw_yaml_without_failing(
    app_client: TestClient, tmp_path: Path
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_id = _create_project(app_client, csrf_token)

    workflow_dir = tmp_path / "wf-source-missing-yaml"
    workflow_dir.mkdir(parents=True)
    yaml_path = _write_workflow_yaml(workflow_dir, filename="loaded.yaml", name="wf-missing-raw")

    create_source = app_client.post(
        "/api/admin/v1/workflows/sources",
        json={"project_id": project_id, "source_path": str(workflow_dir)},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_source.status_code == 201

    reload_response = app_client.post(
        "/api/admin/v1/workflows/reload",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reload_response.status_code == 200

    yaml_path.unlink()

    detail_response = app_client.get("/api/admin/v1/workflows/wf-missing-raw")

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["name"] == "wf-missing-raw"
    assert detail_payload["raw_yaml"] is None
    assert detail_payload["yaml_path"] is None
    assert detail_payload["load_logs"]
    assert "not found" in detail_payload["load_logs"][0].lower()
