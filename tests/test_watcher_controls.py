from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import workflows_mcp.server as server_module
from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.watcher_repo import SQLiteWatcherRepository
from workflows_mcp.server import build_app
from workflows_mcp.watcher.ignore import WatcherIgnorePolicy
from workflows_mcp.watcher.scanner import scan_project_files

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase2-admin-password"


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_gitignore_patterns_are_applied(tmp_path: Path) -> None:
    _write(tmp_path / ".gitignore", "*.tmp\nignored-dir/\n")
    _write(tmp_path / "keep.txt", "ok")
    _write(tmp_path / "drop.tmp", "ignore")
    _write(tmp_path / "ignored-dir" / "nested.txt", "ignore")

    policy = WatcherIgnorePolicy.from_project_root(tmp_path)
    scanned = scan_project_files(tmp_path, policy=policy)

    assert scanned == [Path("keep.txt")]


def test_workflowsignore_patterns_extend_gitignore_policy(tmp_path: Path) -> None:
    _write(tmp_path / ".gitignore", "")
    _write(tmp_path / ".workflowsignore", "wf-temp/\n*.wfskip\n")
    _write(tmp_path / "keep.yaml", "ok")
    _write(tmp_path / "wf-temp" / "pipeline.yaml", "ignore")
    _write(tmp_path / "notes.wfskip", "ignore")

    policy = WatcherIgnorePolicy.from_project_root(tmp_path)
    scanned = scan_project_files(tmp_path, policy=policy)

    assert scanned == [Path("keep.yaml")]


def test_default_heavy_generated_directories_are_ignored_without_ignore_files(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "tracked" / "keep.py", "print('ok')")
    _write(tmp_path / ".git" / "config", "ignore")
    _write(tmp_path / "node_modules" / "pkg" / "index.js", "ignore")
    _write(tmp_path / ".venv" / "bin" / "python", "ignore")
    _write(tmp_path / "dist" / "bundle.js", "ignore")
    _write(tmp_path / "build" / "build.txt", "ignore")
    _write(tmp_path / "__pycache__" / "module.cpython-312.pyc", "ignore")
    _write(tmp_path / ".pytest_cache" / "v" / "cache", "ignore")
    _write(tmp_path / ".mypy_cache" / "meta.json", "ignore")
    _write(tmp_path / ".ruff_cache" / "lint", "ignore")

    scanned = scan_project_files(tmp_path)

    assert scanned == [Path("tracked/keep.py")]


def test_scanner_returns_only_files_with_sorted_relative_paths(tmp_path: Path) -> None:
    _write(tmp_path / "zeta.txt", "z")
    _write(tmp_path / "alpha.txt", "a")
    (tmp_path / "empty-dir").mkdir()

    scanned = scan_project_files(tmp_path)

    assert scanned == [Path("alpha.txt"), Path("zeta.txt")]
    assert all(not path.is_absolute() for path in scanned)


def test_scanner_excludes_symlink_targets_outside_project_root(tmp_path: Path) -> None:
    outside_root = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_root.mkdir(parents=True, exist_ok=True)
    outside_file = outside_root / "outside.txt"
    outside_file.write_text("external", encoding="utf-8")
    outside_dir = outside_root / "outside-dir"
    outside_dir.mkdir(parents=True, exist_ok=True)
    nested_external_file = outside_dir / "nested.txt"
    nested_external_file.write_text("external", encoding="utf-8")

    try:
        (tmp_path / "external-file-link.txt").symlink_to(outside_file)
        (tmp_path / "external-dir-link").symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        msg = f"symlink setup unavailable on this platform/filesystem: {exc}"
        raise AssertionError(msg) from exc

    _write(tmp_path / "internal.txt", "internal")

    scanned = scan_project_files(tmp_path)

    assert scanned == [Path("internal.txt")]


def test_scanner_prunes_ignored_directories_before_descending(tmp_path: Path) -> None:
    _write(tmp_path / "ignored-dir" / "nested.txt", "x")
    _write(tmp_path / "kept.txt", "y")

    class PrunePolicy:
        def is_ignored(self, path: Path) -> bool:
            relative = path.resolve().relative_to(tmp_path.resolve()).as_posix()
            if relative.startswith("ignored-dir/"):
                msg = "scanner descended into ignored directory"
                raise AssertionError(msg)
            return relative == "ignored-dir"

    scanned = scan_project_files(tmp_path, policy=PrunePolicy())

    assert scanned == [Path("kept.txt")]


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


def _create_project(
    client: TestClient,
    csrf_token: str,
    *,
    slug: str,
    palace: str,
    fs_root: str = "/workspace/workflows",
    fs_allowlist: list[str] | None = None,
) -> str:
    response = client.post(
        "/api/admin/v1/projects",
        json={
            "name": "Watcher Control Project",
            "slug": slug,
            "palace": palace,
            "default_wing": "platform",
            "default_room": "runtime",
            "fs_root": fs_root,
            "fs_allowlist": fs_allowlist or [fs_root],
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _seed_project_record(conn: sqlite3.Connection, *, slug: str, palace: str) -> str:
    project_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO projects (
            id,
            name,
            slug,
            palace,
            default_wing,
            default_room,
            fs_root,
            fs_allowlist_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            "Lifecycle Restore Project",
            slug,
            palace,
            "wing-a",
            "room-a",
            "/tmp",
            "[]",
        ),
    )
    conn.commit()
    return project_id


def _parse_single_sse_event(payload: str) -> tuple[str, dict[str, object]]:
    lines = [line for line in payload.strip().splitlines() if line]
    event_name = ""
    data_json = ""
    for line in lines:
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        if line.startswith("data:"):
            data_json = line.split(":", 1)[1].strip()
    assert event_name
    assert data_json
    parsed = json.loads(data_json)
    assert isinstance(parsed, dict)
    return event_name, parsed


def test_admin_watcher_routes_require_session_and_csrf(app_client: TestClient) -> None:
    project_id = "missing-project"

    unauth_list = app_client.get("/api/admin/v1/watchers")
    assert unauth_list.status_code == 401

    unauth_detail = app_client.get(f"/api/admin/v1/watchers/{project_id}")
    assert unauth_detail.status_code == 401

    unauth_pause = app_client.post(f"/api/admin/v1/watchers/{project_id}/pause")
    assert unauth_pause.status_code == 401

    csrf_token = _login_and_csrf(app_client)

    pause_without_csrf = app_client.post(f"/api/admin/v1/watchers/{project_id}/pause")
    assert pause_without_csrf.status_code == 403

    resume_without_csrf = app_client.post(f"/api/admin/v1/watchers/{project_id}/resume")
    assert resume_without_csrf.status_code == 403

    disable_without_csrf = app_client.post(f"/api/admin/v1/watchers/{project_id}/disable")
    assert disable_without_csrf.status_code == 403

    _ = csrf_token


def test_admin_sync_routes_require_session_and_csrf_for_mutations(app_client: TestClient) -> None:
    project_id = "missing-project"

    unauth_list = app_client.get("/api/admin/v1/sync")
    assert unauth_list.status_code == 401

    unauth_now = app_client.post(f"/api/admin/v1/sync/{project_id}/now")
    assert unauth_now.status_code == 401

    unauth_reconcile = app_client.post(f"/api/admin/v1/sync/{project_id}/reconcile")
    assert unauth_reconcile.status_code == 401

    unauth_rebuild = app_client.post(f"/api/admin/v1/sync/{project_id}/rebuild")
    assert unauth_rebuild.status_code == 401

    _login_and_csrf(app_client)

    now_without_csrf = app_client.post(f"/api/admin/v1/sync/{project_id}/now")
    assert now_without_csrf.status_code == 403

    reconcile_without_csrf = app_client.post(f"/api/admin/v1/sync/{project_id}/reconcile")
    assert reconcile_without_csrf.status_code == 403

    rebuild_without_csrf = app_client.post(f"/api/admin/v1/sync/{project_id}/rebuild")
    assert rebuild_without_csrf.status_code == 403


def test_watcher_state_controls_persist_and_status_exposes_metadata(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)
    resources = app_client.app.state.resources
    watcher_manager = resources.watcher_manager

    project_id = _create_project(
        app_client,
        csrf_token,
        slug="watcher-state",
        palace="watcher-state-palace",
    )

    # Runtime watcher transitions apply only when manager is started.
    watcher_manager.start()
    assert watcher_manager.is_started is True
    watcher_manager.start_project_watcher(project_id)
    assert project_id in watcher_manager.active_project_ids

    paused = app_client.post(
        f"/api/admin/v1/watchers/{project_id}/pause",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert paused.status_code == 200
    paused_payload = paused.json()
    assert paused_payload["project_id"] == project_id
    assert paused_payload["state"] == "paused"
    assert paused_payload["dirty_count"] == 0
    assert paused_payload["requires_reconciliation"] is False
    assert paused_payload["last_event_at"] is None
    assert paused_payload["updated_at"]
    assert project_id not in watcher_manager.active_project_ids

    resumed = app_client.post(
        f"/api/admin/v1/watchers/{project_id}/resume",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resumed.status_code == 200
    assert resumed.json()["state"] == "enabled"
    assert project_id in watcher_manager.active_project_ids

    disabled = app_client.post(
        f"/api/admin/v1/watchers/{project_id}/disable",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert disabled.status_code == 200
    assert disabled.json()["state"] == "disabled"
    assert project_id not in watcher_manager.active_project_ids

    watcher_manager.stop()
    assert watcher_manager.is_started is False
    assert project_id not in watcher_manager.active_project_ids

    resumed_when_stopped = app_client.post(
        f"/api/admin/v1/watchers/{project_id}/resume",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resumed_when_stopped.status_code == 200
    assert resumed_when_stopped.json()["state"] == "enabled"
    assert watcher_manager.is_started is False
    assert project_id not in watcher_manager.active_project_ids

    detail = app_client.get(f"/api/admin/v1/watchers/{project_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["project_id"] == project_id
    assert detail_payload["state"] == "enabled"
    assert detail_payload["dirty_count"] == 0
    assert detail_payload["requires_reconciliation"] is False
    assert detail_payload["last_event_at"] is None
    assert detail_payload["updated_at"]

    listed = app_client.get("/api/admin/v1/watchers")
    assert listed.status_code == 200
    entries = listed.json().get("watchers", [])
    ids = [entry["project_id"] for entry in entries]
    assert project_id in ids


def test_sync_now_scans_project_root_and_clears_processed_queue(
    app_client: TestClient,
    tmp_path: Path,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-now-project"
    _write(project_root / "workflow-a.yaml", "steps: []\n")
    _write(project_root / "nested" / "workflow-b.yaml", "steps: []\n")
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-now-scans",
        palace="sync-now-scans-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )

    idle_now = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert idle_now.status_code == 200
    idle_payload = idle_now.json()
    assert idle_payload["project_id"] == project_id
    assert idle_payload["status"] == "idle"
    assert idle_payload["dirty_count"] == 0

    reconcile = app_client.post(
        f"/api/admin/v1/sync/{project_id}/reconcile",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reconcile.status_code == 200
    reconcile_payload = reconcile.json()
    assert reconcile_payload["project_id"] == project_id
    assert reconcile_payload["requires_reconciliation"] is True

    queued_now = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert queued_now.status_code == 200
    queued_payload = queued_now.json()
    assert queued_payload["project_id"] == project_id
    assert queued_payload["status"] == "idle"
    assert queued_payload["dirty_count"] == 0

    rebuild = app_client.post(
        f"/api/admin/v1/sync/{project_id}/rebuild",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert rebuild.status_code == 200
    rebuild_payload = rebuild.json()
    assert rebuild_payload["project_id"] == project_id
    assert rebuild_payload["requires_reconciliation"] is True

    sync_list = app_client.get("/api/admin/v1/sync")
    assert sync_list.status_code == 200
    summaries = sync_list.json().get("projects", [])
    by_project = {entry["project_id"]: entry for entry in summaries}
    assert project_id in by_project
    assert by_project[project_id]["dirty_count"] >= 1
    assert by_project[project_id]["requires_reconciliation"] is True


def test_sync_now_empty_project_root_is_deterministic_noop(
    app_client: TestClient,
    tmp_path: Path,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-now-empty"
    project_root.mkdir(parents=True, exist_ok=True)

    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-now-empty",
        palace="sync-now-empty-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )

    response = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == project_id
    assert payload["status"] == "idle"
    assert payload["dirty_count"] == 0


def test_sync_now_clears_existing_dirty_queue_after_successful_scan(
    app_client: TestClient,
    tmp_path: Path,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-now-clears"
    _write(project_root / "workflow.yaml", "steps: []\n")
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-now-clears",
        palace="sync-now-clears-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )

    reconcile = app_client.post(
        f"/api/admin/v1/sync/{project_id}/reconcile",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reconcile.status_code == 200
    assert reconcile.json()["dirty_count"] == 1

    synced = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert synced.status_code == 200
    synced_payload = synced.json()
    assert synced_payload["project_id"] == project_id
    assert synced_payload["status"] == "idle"
    assert synced_payload["dirty_count"] == 0

    listed = app_client.get("/api/admin/v1/sync")
    assert listed.status_code == 200
    summaries = listed.json().get("projects", [])
    by_project = {entry["project_id"]: entry for entry in summaries}
    assert project_id not in by_project


def test_events_watchers_sse_and_polling_state_are_equivalent(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="events-watchers-state",
        palace="events-watchers-state-palace",
    )

    paused = app_client.post(
        f"/api/admin/v1/watchers/{project_id}/pause",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert paused.status_code == 200

    state_response = app_client.get("/api/events/v1/watchers/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["version"] == 1
    assert isinstance(state_payload["items"], list)

    sse_response = app_client.get("/api/events/v1/watchers")
    assert sse_response.status_code == 200
    assert sse_response.headers["content-type"].startswith("text/event-stream")
    event_name, event_data = _parse_single_sse_event(sse_response.text)
    assert event_name == "watcher.status"
    assert event_data == state_payload


def test_events_sync_sse_and_polling_state_are_equivalent(app_client: TestClient) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="events-sync-state",
        palace="events-sync-state-palace",
    )

    reconcile = app_client.post(
        f"/api/admin/v1/sync/{project_id}/reconcile",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reconcile.status_code == 200

    state_response = app_client.get("/api/events/v1/sync/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["version"] == 1
    assert isinstance(state_payload["items"], list)

    sse_response = app_client.get("/api/events/v1/sync")
    assert sse_response.status_code == 200
    assert sse_response.headers["content-type"].startswith("text/event-stream")
    event_name, event_data = _parse_single_sse_event(sse_response.text)
    assert event_name == "sync.status"
    assert event_data == state_payload


def test_events_watchers_and_sync_require_ui_session_not_bearer(app_client: TestClient) -> None:
    watcher_unauth = app_client.get("/api/events/v1/watchers")
    assert watcher_unauth.status_code == 401

    watcher_state_unauth = app_client.get("/api/events/v1/watchers/state")
    assert watcher_state_unauth.status_code == 401

    sync_unauth = app_client.get("/api/events/v1/sync")
    assert sync_unauth.status_code == 401

    sync_state_unauth = app_client.get("/api/events/v1/sync/state")
    assert sync_state_unauth.status_code == 401

    _login_and_csrf(app_client)

    watcher_bearer = app_client.get(
        "/api/events/v1/watchers",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert watcher_bearer.status_code == 403

    sync_bearer = app_client.get(
        "/api/events/v1/sync",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert sync_bearer.status_code == 403


def test_http_startup_restores_enabled_watchers_into_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / ".workflows"
    monkeypatch_env = {
        "WORKFLOWS_BOOTSTRAP_TOKEN": _MCP_BOOTSTRAP_TOKEN,
        "WORKFLOWS_IO_QUEUE_ENABLED": "false",
        "WORKFLOWS_JOB_QUEUE_ENABLED": "false",
    }
    for key, value in monkeypatch_env.items():
        monkeypatch.setenv(key, value)

    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )

    conn = connect_metadata_db(base_dir / "server.db")
    try:
        migrate_metadata_db(conn)
        project_id = _seed_project_record(
            conn,
            slug="startup-restore",
            palace="startup-restore-palace",
        )
        SQLiteWatcherRepository(conn).set_status(project_id=project_id, state="enabled")
    finally:
        conn.close()

    app = build_app(base_dir=base_dir)
    resources = app.state.resources
    assert resources.watcher_manager.active_project_ids == ()

    with TestClient(app, raise_server_exceptions=False):
        assert resources.watcher_manager.active_project_ids == (project_id,)

    assert resources.watcher_manager.active_project_ids == ()


def test_build_app_uses_lifespan_not_on_event_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)

    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )

    app = build_app(base_dir=base_dir)
    assert app.router.on_startup == []
    assert app.router.on_shutdown == []


@pytest.mark.asyncio
async def test_mcp_lifespan_startup_failure_still_cleans_up_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_CONFIG_DIR", str(base_dir))
    monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")

    stop_called = 0
    original_stop_resources = server_module.stop_resources

    async def _counting_stop(resources: object) -> None:
        nonlocal stop_called
        stop_called += 1
        await original_stop_resources(resources)  # type: ignore[arg-type]

    def _failing_load_workflows(_: object) -> None:
        raise RuntimeError("forced startup failure")

    monkeypatch.setattr(server_module, "stop_resources", _counting_stop)
    monkeypatch.setattr(server_module, "load_workflows", _failing_load_workflows)

    with pytest.raises(RuntimeError, match="forced startup failure"):
        async with server_module.app_lifespan(object()):
            raise AssertionError("unreachable")

    assert stop_called == 1
