from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import workflows_mcp.http.routes.admin_v1.sync as sync_routes
from workflows_mcp.bootstrap import bootstrap_if_needed
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


def _project_create_payload(
    *,
    slug: str = "wf-service",
    palace: str = "wf-palace",
    fs_root: str = "/workspace/workflows",
    fs_allowlist: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": "Workflow Service",
        "slug": slug,
        "palace": palace,
        "default_wing": "platform",
        "default_room": "runtime",
        "fs_root": fs_root,
        "fs_allowlist": fs_allowlist
        if fs_allowlist is not None
        else ["/workspace/workflows", "/workspace/shared"],
        "system2_enabled": False,
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


def test_admin_project_crud_happy_path(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)

    created_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created_response.status_code == 201
    created = created_response.json()
    project_id = created["id"]
    assert created["slug"] == "wf-service"
    assert created["palace"] == "wf-palace"
    assert created["system1_enabled"] is True
    assert created["system2_enabled"] is False
    assert created["memory_mode"] == "simple"

    list_response = app_client.get("/api/admin/v1/projects")
    assert list_response.status_code == 200
    listed = list_response.json()["projects"]
    assert [item["id"] for item in listed] == [project_id]

    get_response = app_client.get(f"/api/admin/v1/projects/{project_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == project_id

    update_response = app_client.patch(
        f"/api/admin/v1/projects/{project_id}",
        json={
            "name": "Workflow Service V2",
            "slug": "wf-service-v2",
            "default_wing": "ops",
            "default_room": "control-plane",
            "fs_root": "/workspace/workflows-v2",
            "fs_allowlist": ["/workspace/workflows-v2"],
            "system2_enabled": True,
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["id"] == project_id
    assert updated["palace"] == "wf-palace"
    assert updated["name"] == "Workflow Service V2"
    assert updated["slug"] == "wf-service-v2"
    assert updated["default_wing"] == "ops"
    assert updated["default_room"] == "control-plane"
    assert updated["system1_enabled"] is True
    assert updated["system2_enabled"] is True
    assert updated["memory_mode"] == "advanced"

    delete_response = app_client.delete(
        f"/api/admin/v1/projects/{project_id}",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True}

    missing_after_delete = app_client.get(f"/api/admin/v1/projects/{project_id}")
    assert missing_after_delete.status_code == 404
    assert _error_code(missing_after_delete.json()) == "project_not_found"


def test_admin_project_create_initializes_watcher_enabled_by_default(
    app_client: TestClient,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    resources = app_client.app.state.resources
    resources.watcher_manager.start()
    assert resources.watcher_manager.is_started is True

    created_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="watcher-default", palace="watcher-default-palace"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created_response.status_code == 201
    project_id = str(created_response.json()["id"])

    watcher_status_response = app_client.get(f"/api/admin/v1/watchers/{project_id}")
    assert watcher_status_response.status_code == 200
    watcher_status = watcher_status_response.json()
    assert watcher_status["project_id"] == project_id
    assert watcher_status["state"] == "enabled"

    assert project_id in resources.watcher_manager.active_project_ids


def test_admin_project_create_allows_default_wing_and_room_to_be_absent_or_null(
    app_client: TestClient,
    tmp_path: Path,
) -> None:
    csrf_token = _login_and_csrf(app_client)

    omitted_root = tmp_path / "omitted-defaults"
    omitted_root.mkdir()
    omitted_payload = _project_create_payload(
        slug="omitted-defaults",
        palace="omitted-defaults-palace",
        fs_root=str(omitted_root),
        fs_allowlist=[str(omitted_root)],
    )
    omitted_payload.pop("default_wing")
    omitted_payload.pop("default_room")

    omitted_response = app_client.post(
        "/api/admin/v1/projects",
        json=omitted_payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert omitted_response.status_code == 201
    omitted = omitted_response.json()
    assert omitted["default_wing"] is None
    assert omitted["default_room"] is None

    null_root = tmp_path / "null-defaults"
    null_root.mkdir()
    null_response = app_client.post(
        "/api/admin/v1/projects",
        json={
            **_project_create_payload(
                slug="null-defaults",
                palace="null-defaults-palace",
                fs_root=str(null_root),
                fs_allowlist=[str(null_root)],
            ),
            "default_wing": None,
            "default_room": None,
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert null_response.status_code == 201
    explicit_null = null_response.json()
    assert explicit_null["default_wing"] is None
    assert explicit_null["default_room"] is None


def test_admin_project_create_normalizes_blank_default_wing_and_room_to_null(
    app_client: TestClient,
    tmp_path: Path,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "blank-defaults"
    project_root.mkdir()

    response = app_client.post(
        "/api/admin/v1/projects",
        json={
            **_project_create_payload(
                slug="blank-defaults",
                palace="blank-defaults-palace",
                fs_root=str(project_root),
                fs_allowlist=[str(project_root)],
            ),
            "default_wing": "",
            "default_room": "   ",
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 201
    created = response.json()
    assert created["default_wing"] is None
    assert created["default_room"] is None


def test_admin_project_create_enqueues_initial_graph_rebuild_status(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "initial-sync-project"
    project_root.mkdir()
    (project_root / "workflow.yaml").write_text("steps: []\n")

    sync_attempts: list[str] = []

    async def _process_project_sync_now(**kwargs: Any) -> sync_routes.SyncNowResponse:
        sync_attempts.append(str(kwargs["project_id"]))
        return sync_routes.SyncNowResponse(
            project_id=str(kwargs["project_id"]),
            status="queued",
            dirty_count=1,
        )

    monkeypatch.setattr(sync_routes, "process_project_sync_now", _process_project_sync_now)

    created_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(
            slug="initial-sync",
            palace="initial-sync-palace",
            fs_root=str(project_root),
            fs_allowlist=[str(project_root)],
        ),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created_response.status_code == 201
    project_id = str(created_response.json()["id"])

    watcher_status_response = app_client.get(f"/api/admin/v1/watchers/{project_id}")
    assert watcher_status_response.status_code == 200
    watcher_status = watcher_status_response.json()
    assert watcher_status["dirty_count"] == 1
    assert watcher_status["requires_reconciliation"] is True
    assert watcher_status["last_event_at"] is not None
    assert sync_attempts == []

    sync_response = app_client.get("/api/admin/v1/sync")
    assert sync_response.status_code == 200
    summaries = sync_response.json()["projects"]
    by_project = {item["project_id"]: item for item in summaries}
    assert by_project[project_id]["dirty_count"] == 1
    assert by_project[project_id]["requires_reconciliation"] is True

    logs_response = app_client.get(f"/api/admin/v1/sync/{project_id}/logs")
    assert logs_response.status_code == 200
    entries = logs_response.json()["entries"]
    assert entries[0]["path"] == "."
    assert entries[0]["event_type"] == "rebuild"
    assert entries[0]["reason"] == "reconciliation_required:project_created"
    assert entries[0]["status"] == "queued"


def test_admin_sync_details_report_system_states_and_collected_counts(
    app_client: TestClient,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    created_response = app_client.post(
        "/api/admin/v1/projects",
        json={
            **_project_create_payload(slug="semantic-sync", palace="semantic-palace"),
            "system2_enabled": True,
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created_response.status_code == 201
    project_id = str(created_response.json()["id"])

    class FakeBackend:
        async def query(self, sql: str, params: tuple[object, ...]) -> object:
            assert params == ("semantic-palace",)
            if "knowledge_items" in sql:
                return type("Result", (), {"rows": [{"n": 7}]})()
            if "knowledge_structural_evidence" in sql and "COUNT(DISTINCT wing)" in sql:
                return type("Result", (), {"rows": [{"wings": 2, "rooms": 3, "compartments": 5}]})()
            if "knowledge_structural_evidence" in sql:
                return type("Result", (), {"rows": [{"n": 12}]})()
            if "knowledge_verification_cycles" in sql:
                return type("Result", (), {"rows": [{"n": 1}]})()
            if "knowledge_semantic_claims" in sql:
                return type("Result", (), {"rows": [{"n": 4}]})()
            if "knowledge_memories" in sql:
                return type("Result", (), {"rows": [{"n": 2}]})()
            raise AssertionError(f"unexpected query: {sql}")

    app_client.app.state.resources.app_context.memory_backend = FakeBackend()

    response = app_client.get(f"/api/admin/v1/sync/{project_id}/details")

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == project_id
    assert payload["system1_enabled"] is True
    assert payload["system1_state"] == "queued"
    assert payload["system2_enabled"] is True
    assert payload["system2_state"] == "completed"
    assert payload["embedding_profile_required"] is True
    assert payload["embedding_profile"] == "embedding"
    assert payload["memory_backend_ready"] is True
    assert payload["counts"] == {
        "source_items": 7,
        "structural_evidence": 12,
        "verification_cycles": 1,
        "wings": 2,
        "rooms": 3,
        "compartments": 5,
        "semantic_claims": 4,
        "semantic_memories": 2,
    }


def test_admin_project_create_watcher_init_failure_rolls_back_project_persistence(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    resources = app_client.app.state.resources

    def _fail_enable(_project_id: str) -> str:
        raise RuntimeError("forced watcher init failure")

    monkeypatch.setattr(resources.watcher_manager, "enable_project_by_default", _fail_enable)

    create_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="watcher-fail", palace="watcher-fail-palace"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_response.status_code == 500

    list_response = app_client.get("/api/admin/v1/projects")
    assert list_response.status_code == 200
    slugs = [item["slug"] for item in list_response.json()["projects"]]
    assert "watcher-fail" not in slugs


def test_admin_projects_require_session_and_reject_mcp_bearer(app_client: TestClient) -> None:
    unauthenticated_list = app_client.get("/api/admin/v1/projects")
    assert unauthenticated_list.status_code == 401

    unauthenticated_create = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(),
    )
    assert unauthenticated_create.status_code == 401

    bearer_list = app_client.get(
        "/api/admin/v1/projects",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert bearer_list.status_code == 403


def test_admin_project_item_routes_require_session(app_client: TestClient) -> None:
    unauth_get = app_client.get("/api/admin/v1/projects/missing-id")
    assert unauth_get.status_code == 401

    unauth_patch = app_client.patch("/api/admin/v1/projects/missing-id", json={"name": "x"})
    assert unauth_patch.status_code == 401

    unauth_delete = app_client.delete("/api/admin/v1/projects/missing-id")
    assert unauth_delete.status_code == 401


def test_mcp_bearer_does_not_authorize_project_routes(app_client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"}

    list_response = app_client.get("/api/admin/v1/projects", headers=headers)
    assert list_response.status_code == 403

    get_response = app_client.get("/api/admin/v1/projects/missing-id", headers=headers)
    assert get_response.status_code == 403

    patch_response = app_client.patch(
        "/api/admin/v1/projects/missing-id",
        json={"name": "x"},
        headers=headers,
    )
    assert patch_response.status_code == 403

    delete_response = app_client.delete("/api/admin/v1/projects/missing-id", headers=headers)
    assert delete_response.status_code == 403


def test_admin_projects_mutations_require_csrf_header(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)

    create_without_csrf = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(),
    )
    assert create_without_csrf.status_code == 403

    created_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="csrf-project", palace="csrf-palace"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created_response.status_code == 201
    project_id = created_response.json()["id"]

    patch_without_csrf = app_client.patch(
        f"/api/admin/v1/projects/{project_id}",
        json={"name": "patched"},
    )
    assert patch_without_csrf.status_code == 403

    delete_without_csrf = app_client.delete(f"/api/admin/v1/projects/{project_id}")
    assert delete_without_csrf.status_code == 403


def test_admin_project_patch_rejects_palace_mutation_with_conflict_code(
    app_client: TestClient,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    created_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="immutable", palace="immutable-palace"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created_response.status_code == 201
    project_id = created_response.json()["id"]

    patch_response = app_client.patch(
        f"/api/admin/v1/projects/{project_id}",
        json={"palace": "new-palace"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert patch_response.status_code == 409
    payload = patch_response.json()
    assert _error_code(payload) == "project_palace_immutable"


def test_admin_project_patch_palace_change_is_immutable_even_if_target_exists(
    app_client: TestClient,
) -> None:
    csrf_token = _login_and_csrf(app_client)

    first_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="first-for-palace", palace="palace-a"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert first_response.status_code == 201

    second_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="second-for-palace", palace="palace-b"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert second_response.status_code == 201
    second_id = second_response.json()["id"]

    mutate_palace_response = app_client.patch(
        f"/api/admin/v1/projects/{second_id}",
        json={"palace": "palace-a"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert mutate_palace_response.status_code == 409
    assert _error_code(mutate_palace_response.json()) == "project_palace_immutable"


def test_admin_project_duplicate_conflicts_have_deterministic_codes(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)
    first_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="first", palace="palace-a"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert first_response.status_code == 201
    first_id = first_response.json()["id"]

    duplicate_slug = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="first", palace="palace-b"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert duplicate_slug.status_code == 409
    assert _error_code(duplicate_slug.json()) == "project_slug_conflict"

    duplicate_palace = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="second", palace="palace-a"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert duplicate_palace.status_code == 409
    assert _error_code(duplicate_palace.json()) == "project_palace_conflict"

    second_response = app_client.post(
        "/api/admin/v1/projects",
        json=_project_create_payload(slug="third", palace="palace-c"),
        headers={"X-CSRF-Token": csrf_token},
    )
    assert second_response.status_code == 201
    second_id = second_response.json()["id"]

    duplicate_slug_update = app_client.patch(
        f"/api/admin/v1/projects/{second_id}",
        json={"slug": "first"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert duplicate_slug_update.status_code == 409
    assert _error_code(duplicate_slug_update.json()) == "project_slug_conflict"

    immutable_palace_update = app_client.patch(
        f"/api/admin/v1/projects/{second_id}",
        json={"palace": "other-palace"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert immutable_palace_update.status_code == 409
    assert _error_code(immutable_palace_update.json()) == "project_palace_immutable"

    # Keep first_id used to avoid dead-code drift in test intent.
    assert first_id


def test_admin_project_missing_returns_404(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)
    missing_id = "missing-project-id"

    get_response = app_client.get(f"/api/admin/v1/projects/{missing_id}")
    assert get_response.status_code == 404
    assert _error_code(get_response.json()) == "project_not_found"

    patch_response = app_client.patch(
        f"/api/admin/v1/projects/{missing_id}",
        json={"name": "updated"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert patch_response.status_code == 404
    assert _error_code(patch_response.json()) == "project_not_found"

    delete_response = app_client.delete(
        f"/api/admin/v1/projects/{missing_id}",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert delete_response.status_code == 404
    assert _error_code(delete_response.json()) == "project_not_found"
