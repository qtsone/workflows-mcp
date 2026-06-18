from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import workflows_mcp.http.routes.admin_v1.projects as projects_routes
import workflows_mcp.http.routes.admin_v1.sync as sync_routes
import workflows_mcp.server as server_module
from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.memory.memory_schema import MemoryRequest
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.projects_repo import ProjectRecord
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
    system2_enabled: bool = False,
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
            "system2_enabled": system2_enabled,
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _capture_sync_submissions(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    resources = app_client.app.state.resources
    resources.app_context.memory_backend = object()
    submitted: list[dict[str, object]] = []

    async def _submit_job(
        workflow: str,
        inputs: dict[str, Any] | None = None,
        timeout: int | None = None,
        *,
        project_id: str | None = None,
        token_id: str | None = None,
    ) -> str:
        job_id = f"job_sync_{len(submitted) + 1}"
        submitted.append(
            {
                "job_id": job_id,
                "workflow": workflow,
                "inputs": inputs or {},
                "timeout": timeout,
                "project_id": project_id,
                "token_id": token_id,
            }
        )
        return job_id

    monkeypatch.setattr(resources.job_queue, "submit_job", _submit_job)
    return submitted


def test_sync_now_without_dirty_work_is_idle_and_does_not_enqueue_job(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-idle-project"
    project_root.mkdir()
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-idle",
        palace="sync-idle-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )
    SQLiteWatcherRepository(
        app_client.app.state.resources.metadata_db_conn
    ).mark_active_dirty_processed(project_id=project_id)
    submitted: list[dict[str, object]] = []

    async def _submit_job(*args: object, **kwargs: object) -> str:
        submitted.append({"args": args, "kwargs": kwargs})
        return "job_should_not_exist"

    monkeypatch.setattr(app_client.app.state.resources.job_queue, "submit_job", _submit_job)

    response = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.json() == {
        "project_id": project_id,
        "status": "idle",
        "dirty_count": 0,
        "action": "sync",
        "memory_mode": "simple",
        "job_id": None,
        "workflow": None,
        "error": None,
    }
    assert submitted == []


def test_sync_rebuild_enqueues_project_memory_sync_without_running_inline(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    resources = app_client.app.state.resources
    resources.app_context.memory_backend = object()
    project_root = tmp_path / "sync-async-project"
    _write(project_root / "src" / "app.py", "def main():\n    return 1\n")
    submitted: list[dict[str, object]] = []

    async def _submit_job(
        workflow: str,
        inputs: dict[str, Any] | None = None,
        timeout: int | None = None,
        *,
        project_id: str | None = None,
        token_id: str | None = None,
    ) -> str:
        submitted.append(
            {
                "workflow": workflow,
                "inputs": inputs or {},
                "timeout": timeout,
                "project_id": project_id,
                "token_id": token_id,
            }
        )
        return "job_async_rebuild"

    monkeypatch.setattr(resources.job_queue, "submit_job", _submit_job)

    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-async",
        palace="sync-async-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
        system2_enabled=True,
    )
    assert submitted[-1]["inputs"]["sync_scope"] == "rebuild"
    assert submitted[-1]["inputs"]["memory_mode"] == "advanced"
    submitted.clear()

    response = app_client.post(
        f"/api/admin/v1/sync/{project_id}/rebuild",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == project_id
    assert payload["status"] == "queued"
    assert payload["action"] == "rebuild"
    assert payload["memory_mode"] == "advanced"
    assert payload["job_id"] == "job_async_rebuild"
    assert payload["workflow"] == "project-memory-sync"
    assert payload["dirty_count"] == 0
    assert submitted == [
        {
            "workflow": "project-memory-sync",
            "inputs": {
                "project_root": str(project_root),
                "fs_allowlist": [str(project_root)],
                "candidate_paths": [],
                "palace": "sync-async-palace",
                "source_name": "sync-async",
                "default_wing": "platform",
                "default_room": "runtime",
                "default_compartment": "sync-async",
                "sync_scope": "rebuild",
                "memory_mode": "advanced",
            },
            "timeout": None,
            "project_id": project_id,
            "token_id": None,
        }
    ]


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


async def _collect_single_live_sse_event(
    event_name: str,
    payload_factory: Callable[[], dict[str, object]],
) -> tuple[str, dict[str, object]]:
    from workflows_mcp.http.routes.events_v1 import _live_status_event_stream

    async for chunk in _live_status_event_stream(
        event_name,
        payload_factory,
        interval_seconds=0,
        max_events=1,
    ):
        return _parse_single_sse_event(chunk.decode())
    raise AssertionError("live SSE stream did not emit an event")


def _single_live_sse_event(
    event_name: str,
    payload_factory: Callable[[], dict[str, object]],
) -> tuple[str, dict[str, object]]:
    return asyncio.run(_collect_single_live_sse_event(event_name, payload_factory))


def test_sync_scope_maps_project_slug_to_required_memory_compartment() -> None:
    project = ProjectRecord(
        id="project-id",
        name="Project",
        slug="project-slug",
        palace="project-palace",
        default_wing="platform",
        default_room="runtime",
        fs_root="/tmp/project",
        fs_allowlist=["/tmp/project"],
        system2_enabled=False,
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
    )

    assert sync_routes._scope_for_project(project) == {
        "palace": "project-palace",
        "wing": "platform",
        "room": "runtime",
        "compartment": "project-slug",
    }


def test_sync_scope_suppresses_blank_and_placeholder_wing_room() -> None:
    project = ProjectRecord(
        id="project-id",
        name="Project",
        slug="project-slug",
        palace="project-palace",
        default_wing=" default-wing ",
        default_room="code",
        fs_root="/tmp/project",
        fs_allowlist=["/tmp/project"],
        system2_enabled=False,
        created_at="2026-05-03T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
    )

    assert sync_routes._scope_for_project(project) == {
        "palace": "project-palace",
        "wing": "",
        "room": "",
        "compartment": "project-slug",
    }


@pytest.mark.asyncio
async def test_live_status_event_stream_can_emit_repeated_snapshots() -> None:
    from workflows_mcp.http.routes.events_v1 import _live_status_event_stream

    snapshots: list[dict[str, object]] = [
        {"version": 1, "items": []},
        {"version": 2, "items": [{"project_id": "p1"}]},
    ]
    emitted: list[str] = []

    async for chunk in _live_status_event_stream(
        "watcher.status",
        lambda: snapshots[len(emitted)],
        interval_seconds=0,
        max_events=2,
    ):
        emitted.append(chunk.decode())

    assert [_parse_single_sse_event(payload) for payload in emitted] == [
        ("watcher.status", snapshots[0]),
        ("watcher.status", snapshots[1]),
    ]


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


def test_watcher_state_controls_persist_and_status_exposes_metadata(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _queue_project_rebuild_noop(**kwargs: Any) -> sync_routes.SyncNowResponse:
        return sync_routes.SyncNowResponse(
            project_id=str(kwargs["project_id"]),
            status="queued",
            dirty_count=1,
        )

    monkeypatch.setattr(projects_routes, "queue_project_rebuild", _queue_project_rebuild_noop)

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
    assert paused_payload["dirty_count"] == 1
    assert paused_payload["requires_reconciliation"] is True
    assert paused_payload["last_event_at"] is not None
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
    assert detail_payload["dirty_count"] == 1
    assert detail_payload["requires_reconciliation"] is True
    assert detail_payload["last_event_at"] is not None
    assert detail_payload["updated_at"]

    listed = app_client.get("/api/admin/v1/watchers")
    assert listed.status_code == 200
    entries = listed.json().get("watchers", [])
    ids = [entry["project_id"] for entry in entries]
    assert project_id in ids


def test_sync_now_scans_project_root_and_clears_processed_queue(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-now-project"
    _write(project_root / "workflow-a.yaml", "steps: []\n")
    _write(project_root / "nested" / "workflow-b.yaml", "steps: []\n")
    submitted = _capture_sync_submissions(app_client, monkeypatch)
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-now-scans",
        palace="sync-now-scans-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )
    assert submitted[-1]["inputs"]["sync_scope"] == "rebuild"
    submitted.clear()

    idle_now = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert idle_now.status_code == 200
    idle_payload = idle_now.json()
    assert idle_payload["project_id"] == project_id
    assert idle_payload["status"] == "idle"
    assert idle_payload["dirty_count"] == 0
    assert idle_payload["action"] == "sync"
    assert submitted == []

    reconcile = app_client.post(
        f"/api/admin/v1/sync/{project_id}/reconcile",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reconcile.status_code == 200
    reconcile_payload = reconcile.json()
    assert reconcile_payload["project_id"] == project_id
    assert reconcile_payload["status"] == "queued"
    assert reconcile_payload["dirty_count"] == 0
    assert reconcile_payload["job_id"] == "job_sync_1"
    assert submitted[-1]["workflow"] == "project-memory-sync"
    assert submitted[-1]["inputs"]["sync_scope"] == "reconcile"
    assert submitted[-1]["inputs"]["candidate_paths"] == []

    queued_now = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert queued_now.status_code == 200
    queued_payload = queued_now.json()
    assert queued_payload["project_id"] == project_id
    assert queued_payload["status"] == "idle"
    assert queued_payload["dirty_count"] == 0
    assert len(submitted) == 1

    rebuild = app_client.post(
        f"/api/admin/v1/sync/{project_id}/rebuild",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert rebuild.status_code == 200
    rebuild_payload = rebuild.json()
    assert rebuild_payload["project_id"] == project_id
    assert rebuild_payload["status"] == "queued"
    assert rebuild_payload["dirty_count"] == 0
    assert rebuild_payload["job_id"] == "job_sync_2"
    assert submitted[-1]["workflow"] == "project-memory-sync"
    assert submitted[-1]["inputs"]["sync_scope"] == "rebuild"

    sync_list = app_client.get("/api/admin/v1/sync")
    assert sync_list.status_code == 200
    summaries = sync_list.json().get("projects", [])
    by_project = {entry["project_id"]: entry for entry in summaries}
    assert project_id not in by_project


def test_sync_reconcile_runs_project_sync_and_clears_processed_queue(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-reconcile-project"
    _write(project_root / "workflow.yaml", "steps: []\n")
    submitted = _capture_sync_submissions(app_client, monkeypatch)
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-reconcile-runs",
        palace="sync-reconcile-runs-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )
    submitted.clear()

    reconcile = app_client.post(
        f"/api/admin/v1/sync/{project_id}/reconcile",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reconcile.status_code == 200
    reconcile_payload = reconcile.json()
    assert reconcile_payload["project_id"] == project_id
    assert reconcile_payload["status"] == "queued"
    assert reconcile_payload["dirty_count"] == 0
    assert reconcile_payload["job_id"] == "job_sync_1"
    assert submitted[-1]["workflow"] == "project-memory-sync"
    assert submitted[-1]["inputs"]["sync_scope"] == "reconcile"

    listed = app_client.get("/api/admin/v1/sync")
    assert listed.status_code == 200
    summaries = listed.json().get("projects", [])
    by_project = {entry["project_id"]: entry for entry in summaries}
    assert project_id not in by_project


def test_sync_rebuild_runs_project_sync_and_clears_processed_queue(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-rebuild-project"
    _write(project_root / "workflow.yaml", "steps: []\n")
    submitted = _capture_sync_submissions(app_client, monkeypatch)
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-rebuild-runs",
        palace="sync-rebuild-runs-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )
    submitted.clear()

    rebuild = app_client.post(
        f"/api/admin/v1/sync/{project_id}/rebuild",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert rebuild.status_code == 200
    rebuild_payload = rebuild.json()
    assert rebuild_payload["project_id"] == project_id
    assert rebuild_payload["status"] == "queued"
    assert rebuild_payload["dirty_count"] == 0
    assert rebuild_payload["job_id"] == "job_sync_1"
    assert submitted[-1]["workflow"] == "project-memory-sync"
    assert submitted[-1]["inputs"]["sync_scope"] == "rebuild"

    listed = app_client.get("/api/admin/v1/sync")
    assert listed.status_code == 200
    summaries = listed.json().get("projects", [])
    by_project = {entry["project_id"]: entry for entry in summaries}
    assert project_id not in by_project


def test_sync_rebuild_derives_default_topology_when_project_defaults_are_blank(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-derived-defaults-project"
    _write(project_root / "workflow.yaml", "steps: []\n")
    submitted = _capture_sync_submissions(app_client, monkeypatch)

    project_response = app_client.post(
        "/api/admin/v1/projects",
        json={
            "name": "Derived Defaults Project",
            "slug": "sync-derived-defaults",
            "palace": "sync-derived-defaults-palace",
            "default_wing": "",
            "default_room": "",
            "fs_root": str(project_root),
            "fs_allowlist": [str(project_root)],
        },
        headers={"X-CSRF-Token": csrf_token},
    )
    assert project_response.status_code == 201
    project_id = str(project_response.json()["id"])

    rebuild = app_client.post(
        f"/api/admin/v1/sync/{project_id}/rebuild",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert rebuild.status_code == 200
    rebuild_payload = rebuild.json()
    assert rebuild_payload["project_id"] == project_id
    assert rebuild_payload["status"] == "queued"
    assert rebuild_payload["dirty_count"] == 0
    assert submitted[-1]["inputs"]["default_wing"] is None
    assert submitted[-1]["inputs"]["default_room"] is None
    assert submitted[-1]["inputs"]["default_compartment"] == "sync-derived-defaults"


def test_sync_error_detail_surfaces_wrapped_memory_contract_validation_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        MemoryRequest.model_validate(
            {
                "operation": "graph_upsert",
                "scope": {
                    "palace": "forge",
                    "compartment": "forge",
                },
                "graph": {
                    "kind": "place",
                    "place_name": "forge",
                    "place_type": "Palace",
                },
            }
        )

    detail = sync_routes._sync_error_detail(exc_info.value)

    assert detail.code == "project_system1_sync_failed"
    assert "MEM_SCOPE_HIERARCHY_VIOLATION" in detail.message


def test_sync_now_reports_system1_workflow_failure_details(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "MEMORY_DB_HOST",
        "MEMORY_DB_PORT",
        "MEMORY_DB_NAME",
        "MEMORY_DB_USER",
        "MEMORY_DB_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)

    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-failure-project"
    _write(project_root / "workflow.yaml", "steps: []\n")

    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-failure-details",
        palace="sync-failure-details-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )
    SQLiteWatcherRepository(app_client.app.state.resources.metadata_db_conn).enqueue_dirty(
        project_id=project_id,
        path="workflow.yaml",
        event_type="modified",
        reason="file_event",
    )

    response = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == project_id
    assert payload["status"] == "failed"
    assert payload["dirty_count"] > 0
    assert payload["error"]["code"] == "project_system1_sync_failed"
    assert "PostgreSQL memory" in payload["error"]["message"]


def test_system1_project_sync_records_workflow_run_history(
    app_client: TestClient,
    tmp_path: Path,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    resources = app_client.app.state.resources
    resources.app_context.memory_backend = object()
    server_module.load_workflows(resources)

    project_root = tmp_path / "sync-run-history-project"
    _write(project_root / "src" / "app.py", "def main():\n    return 1\n")
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-run-history",
        palace="sync-run-history-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )

    response = app_client.post(
        f"/api/admin/v1/sync/{project_id}/rebuild",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"

    rows = resources.metadata_db_conn.execute(
        """
        SELECT workflow_name, status, execution_mode, inputs_json, project_id
          FROM job_runs
         WHERE project_id = ?
         ORDER BY created_at ASC
        """,
        (project_id,),
    ).fetchall()
    assert rows
    latest = rows[-1]
    assert latest["workflow_name"] == "project-memory-sync"
    assert latest["status"] in {"queued", "running", "completed", "failed"}
    assert latest["execution_mode"] == "async"
    assert latest["project_id"] == project_id
    assert '"sync_scope":"rebuild"' in str(latest["inputs_json"])


def test_sync_now_empty_project_root_is_deterministic_noop(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-now-empty"
    project_root.mkdir(parents=True, exist_ok=True)
    submitted = _capture_sync_submissions(app_client, monkeypatch)

    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-now-empty",
        palace="sync-now-empty-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )
    submitted.clear()

    response = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == project_id
    assert payload["status"] == "idle"
    assert payload["dirty_count"] == 0
    assert submitted == []


def test_sync_now_clears_existing_dirty_queue_after_successful_scan(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    project_root = tmp_path / "sync-now-clears"
    _write(project_root / "workflow.yaml", "steps: []\n")
    submitted = _capture_sync_submissions(app_client, monkeypatch)
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-now-clears",
        palace="sync-now-clears-palace",
        fs_root=str(project_root),
        fs_allowlist=[str(project_root)],
    )
    submitted.clear()

    SQLiteWatcherRepository(app_client.app.state.resources.metadata_db_conn).enqueue_dirty(
        project_id=project_id,
        path="workflow.yaml",
        event_type="modified",
        reason="file_event",
    )

    synced = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert synced.status_code == 200
    synced_payload = synced.json()
    assert synced_payload["project_id"] == project_id
    assert synced_payload["status"] == "queued"
    assert synced_payload["dirty_count"] == 0
    assert synced_payload["job_id"] == "job_sync_1"
    assert submitted[-1]["inputs"]["sync_scope"] == "dirty"
    assert submitted[-1]["inputs"]["candidate_paths"] == ["workflow.yaml"]

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

    from workflows_mcp.http.routes.events_v1 import _watcher_status_payload

    event_name, event_data = _single_live_sse_event(
        "watcher.status",
        lambda: _watcher_status_payload(app_client.app.state.resources).model_dump(mode="json"),
    )
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

    from workflows_mcp.http.routes.events_v1 import _sync_status_payload

    event_name, event_data = _single_live_sse_event(
        "sync.status",
        lambda: _sync_status_payload(app_client.app.state.resources).model_dump(mode="json"),
    )
    assert event_name == "sync.status"
    assert event_data == state_payload


def test_sync_project_logs_include_queued_and_processed_activity(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csrf_token = _login_and_csrf(app_client)
    submitted = _capture_sync_submissions(app_client, monkeypatch)
    project_id = _create_project(
        app_client,
        csrf_token,
        slug="sync-project-logs",
        palace="sync-project-logs-palace",
        fs_root=str(tmp_path),
        fs_allowlist=[str(tmp_path)],
    )

    reconcile = app_client.post(
        f"/api/admin/v1/sync/{project_id}/reconcile",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert reconcile.status_code == 200

    processed = app_client.post(
        f"/api/admin/v1/sync/{project_id}/now",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert processed.status_code == 200

    rebuild = app_client.post(
        f"/api/admin/v1/sync/{project_id}/rebuild",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert rebuild.status_code == 200

    logs = app_client.get(f"/api/admin/v1/sync/{project_id}/logs")

    assert logs.status_code == 200
    payload = logs.json()
    assert payload["project_id"] == project_id
    entries = payload["entries"]
    assert [entry["event_type"] for entry in entries] == ["rebuild", "reconcile", "rebuild"]
    assert [entry["reason"] for entry in entries] == [
        "reconciliation_required:manual_rebuild",
        "reconciliation_required:manual_reconcile",
        "reconciliation_required:project_created",
    ]
    assert entries[0]["status"] == "processed"
    assert entries[0]["processed_at"] is not None
    assert entries[1]["status"] == "processed"
    assert entries[1]["processed_at"] is not None
    assert entries[2]["status"] == "processed"
    assert entries[2]["processed_at"] is not None
    assert [entry["inputs"]["sync_scope"] for entry in submitted] == [
        "rebuild",
        "reconcile",
        "rebuild",
    ]


def test_events_watchers_and_sync_require_ui_session_not_bearer(app_client: TestClient) -> None:
    watcher_unauth = app_client.get("/api/events/v1/watchers")
    assert watcher_unauth.status_code == 401

    watcher_state_unauth = app_client.get("/api/events/v1/watchers/state")
    assert watcher_state_unauth.status_code == 401

    sync_unauth = app_client.get("/api/events/v1/sync")
    assert sync_unauth.status_code == 401

    sync_state_unauth = app_client.get("/api/events/v1/sync/state")
    assert sync_state_unauth.status_code == 401

    sync_logs_unauth = app_client.get("/api/admin/v1/sync/p1/logs")
    assert sync_logs_unauth.status_code == 401

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

    sync_logs_bearer = app_client.get(
        "/api/admin/v1/sync/p1/logs",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert sync_logs_bearer.status_code == 403


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


# ---------------------------------------------------------------------------
# Task 12 — watcher-driven System 1 structural evidence + verification cycles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_sync_writes_structural_evidence_before_verification_cycle() -> None:
    """Watcher-triggered sync must write System 1 structural evidence rows
    before recording a verification cycle.

    Verifies ADR-013 Task 12: run_programmatic_onboard_with_cycle_recording
    issues store_system1_structural_evidence BEFORE record_system1_verification_cycle
    so that the cycle record is causally downstream of evidence writes.
    """
    from workflows_mcp.memory.memory_onboard_sync_orchestrator import (
        ProgrammaticOnboardRequest,
        ScannedFileEntry,
        run_programmatic_onboard_with_cycle_recording,
    )
    from workflows_mcp.memory.memory_schema import ManageMemoryResult, MemoryResult

    operations_called: list[str] = []

    class _FakeMemoryService:
        async def execute(self, request: MemoryRequest) -> MemoryResult:
            op = request.operation
            operations_called.append(op)
            if op == "store_system1_structural_evidence":
                return MemoryResult(
                    operation=op,
                    manage=ManageMemoryResult(
                        operation=op,
                        success=True,
                        stored_count=1,
                        stored_evidence_ids=["fake-evidence-id"],
                    ),
                )
            if op == "record_system1_verification_cycle":
                return MemoryResult(
                    operation=op,
                    manage=ManageMemoryResult(
                        operation=op,
                        success=True,
                        cycle_id="fake-cycle-id",
                    ),
                )
            raise AssertionError(f"Unexpected operation: {op!r}")

    scope = {
        "palace": "watcher-test-palace",
        "wing": "platform",
        "room": "runtime",
        "compartment": "watcher-project",
    }
    request = ProgrammaticOnboardRequest(
        scope=scope,
        files=[
            ScannedFileEntry(
                path="src/main.py",
                content="def main(): pass\n",
                size_bytes=20,
            )
        ],
        mode="programmatic",
    )

    result = await run_programmatic_onboard_with_cycle_recording(
        request,
        memory_service=_FakeMemoryService(),  # type: ignore[arg-type]
    )

    assert result.status == "completed", f"Expected completed, got: {result.error}"
    assert "store_system1_structural_evidence" in operations_called, (
        "store_system1_structural_evidence must be called during watcher-triggered sync"
    )
    assert "record_system1_verification_cycle" in operations_called, (
        "record_system1_verification_cycle must be called during watcher-triggered sync"
    )
    evidence_index = operations_called.index("store_system1_structural_evidence")
    cycle_index = operations_called.index("record_system1_verification_cycle")
    assert evidence_index < cycle_index, (
        "store_system1_structural_evidence must be called BEFORE record_system1_verification_cycle"
    )


@pytest.mark.asyncio
async def test_watcher_sync_structural_evidence_write_failure_blocks_cycle_recording() -> None:
    """When structural evidence write fails, the verification cycle must NOT be recorded.

    Verifies ADR-013 Task 12 fail-closed behavior: a RuntimeError must propagate
    from run_programmatic_onboard_with_cycle_recording so that the caller sees
    failure and cannot incorrectly record a cycle without prior evidence.
    """
    from workflows_mcp.memory.memory_onboard_sync_orchestrator import (
        ProgrammaticOnboardRequest,
        ScannedFileEntry,
        run_programmatic_onboard_with_cycle_recording,
    )
    from workflows_mcp.memory.memory_schema import ManageMemoryResult, MemoryResult

    cycle_recording_attempted = False

    class _FailingEvidenceMemoryService:
        async def execute(self, request: MemoryRequest) -> MemoryResult:
            op = request.operation
            if op == "store_system1_structural_evidence":
                return MemoryResult(
                    operation=op,
                    manage=ManageMemoryResult(
                        operation=op,
                        success=False,
                        error="simulated structural evidence write failure",
                    ),
                )
            if op == "record_system1_verification_cycle":
                nonlocal cycle_recording_attempted
                cycle_recording_attempted = True
                return MemoryResult(
                    operation=op,
                    manage=ManageMemoryResult(
                        operation=op,
                        success=True,
                        cycle_id="should-not-be-recorded",
                    ),
                )
            raise AssertionError(f"Unexpected operation: {op!r}")

    scope = {
        "palace": "watcher-fail-palace",
        "wing": "platform",
        "room": "runtime",
        "compartment": "watcher-fail-project",
    }
    request = ProgrammaticOnboardRequest(
        scope=scope,
        files=[
            ScannedFileEntry(
                path="src/service.py",
                content="class Service: pass\n",
                size_bytes=22,
            )
        ],
        mode="programmatic",
    )

    with pytest.raises(RuntimeError, match="store_system1_structural_evidence"):
        await run_programmatic_onboard_with_cycle_recording(
            request,
            memory_service=_FailingEvidenceMemoryService(),  # type: ignore[arg-type]
        )

    assert not cycle_recording_attempted, (
        "record_system1_verification_cycle must NOT be attempted"
        " when structural evidence write fails"
    )


@pytest.mark.asyncio
async def test_watcher_sync_derives_structural_module_evidence_from_file_paths() -> None:
    """Watcher-triggered sync must extract at least one structural_module evidence item
    per readable scanned file and submit them to store_system1_structural_evidence.

    Verifies ADR-013 Task 12: the sync path produces System 1-owned structural
    evidence from file topology (not TreeSitter), preserving System 1 ownership
    without deriving System 2 semantics.
    """
    from workflows_mcp.memory.memory_onboard_sync_orchestrator import (
        ProgrammaticOnboardRequest,
        ScannedFileEntry,
        run_programmatic_onboard_with_cycle_recording,
    )
    from workflows_mcp.memory.memory_schema import ManageMemoryResult, MemoryResult

    captured_evidence: list[dict[str, Any]] = []

    class _CapturingMemoryService:
        async def execute(self, request: MemoryRequest) -> MemoryResult:
            op = request.operation
            if op == "store_system1_structural_evidence":
                if request.record and request.record.structural_evidence:
                    for item in request.record.structural_evidence:
                        captured_evidence.append(item.model_dump())
                return MemoryResult(
                    operation=op,
                    manage=ManageMemoryResult(
                        operation=op,
                        success=True,
                        stored_count=len(captured_evidence),
                        stored_evidence_ids=["eid-1"],
                    ),
                )
            if op == "record_system1_verification_cycle":
                return MemoryResult(
                    operation=op,
                    manage=ManageMemoryResult(
                        operation=op,
                        success=True,
                        cycle_id="cap-cycle-id",
                    ),
                )
            raise AssertionError(f"Unexpected operation: {op!r}")

    scope = {
        "palace": "watcher-evidence-palace",
        "wing": "backend",
        "room": "api",
        "compartment": "evidence-project",
    }
    request = ProgrammaticOnboardRequest(
        scope=scope,
        files=[
            ScannedFileEntry(
                path="src/api.py",
                content="class Api: pass\n",
                size_bytes=18,
            ),
            ScannedFileEntry(
                path="src/models.py",
                content="class Model: pass\n",
                size_bytes=20,
            ),
        ],
        mode="programmatic",
    )

    result = await run_programmatic_onboard_with_cycle_recording(
        request,
        memory_service=_CapturingMemoryService(),  # type: ignore[arg-type]
    )

    assert result.status == "completed", f"Expected completed, got: {result.error}"
    assert len(captured_evidence) >= 2, (
        f"Expected at least 2 structural evidence items (one per readable file), "
        f"got {len(captured_evidence)}: {captured_evidence}"
    )
    categories = {item["evidence_category"] for item in captured_evidence}
    assert "structural_module" in categories, (
        f"Expected structural_module evidence category; got categories: {categories}"
    )
    stable_ids = {item["entity_stable_id"] for item in captured_evidence}
    assert "src/api.py" in stable_ids or any("api" in sid for sid in stable_ids), (
        f"Expected evidence item with stable_id referencing src/api.py; got: {stable_ids}"
    )


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
