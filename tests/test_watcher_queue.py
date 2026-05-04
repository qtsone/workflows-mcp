from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from workflows_mcp.http.lifespan import build_resources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.watcher_repo import (
    DirtyQueueEntry,
    InvalidWatcherStateError,
    SQLiteWatcherRepository,
    WatcherStatusRecord,
)
from workflows_mcp.watcher.events import WatcherEvent
from workflows_mcp.watcher.manager import WatcherManager


def _seed_project(conn: sqlite3.Connection) -> str:
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
            "Watcher Test Project",
            f"watcher-test-{project_id[:8]}",
            f"palace.watcher.test.{project_id[:8]}",
            "wing-a",
            "room-a",
            "/tmp",
            "[]",
        ),
    )
    conn.commit()
    return project_id


def test_watcher_status_persists_across_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    enabled = repo.set_status(project_id=project_id, state="enabled")
    assert isinstance(enabled, WatcherStatusRecord)
    assert enabled.state == "enabled"

    paused = repo.set_status(project_id=project_id, state="paused")
    assert paused.state == "paused"
    assert paused.updated_at >= enabled.updated_at

    disabled = repo.set_status(project_id=project_id, state="disabled")
    assert disabled.state == "disabled"
    conn.close()

    reopened = connect_metadata_db(db_path)
    try:
        reloaded = SQLiteWatcherRepository(reopened).get_status(project_id)
        assert isinstance(reloaded, WatcherStatusRecord)
        assert reloaded is not None
        assert reloaded.project_id == project_id
        assert reloaded.state == "disabled"
        assert reloaded.updated_at >= disabled.updated_at
    finally:
        reopened.close()


def test_dirty_queue_survives_connection_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.enqueue_dirty(
        project_id=project_id,
        path="workflows/build.yaml",
        event_type="modified",
        reason="file_event",
    )
    repo.enqueue_dirty(
        project_id=project_id,
        path="workflows/release.yaml",
        event_type="created",
        reason="reconciliation",
    )
    conn.close()

    reopened = connect_metadata_db(db_path)
    try:
        queue = SQLiteWatcherRepository(reopened).list_active_dirty(project_id=project_id)
        assert len(queue) == 2

        first = queue[0]
        assert isinstance(first, DirtyQueueEntry)
        assert first.project_id == project_id
        assert first.path in {"workflows/build.yaml", "workflows/release.yaml"}
        assert first.event_type in {"modified", "created"}
        assert first.reason in {"file_event", "reconciliation"}
        assert first.enqueued_at
        assert first.updated_at
    finally:
        reopened.close()


def test_dirty_queue_coalesces_active_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    first = repo.enqueue_dirty(
        project_id=project_id,
        path="workflows/sync.yaml",
        event_type="modified",
        reason="file_event",
    )
    second = repo.enqueue_dirty(
        project_id=project_id,
        path="workflows/sync.yaml",
        event_type="modified",
        reason="reconciliation",
    )

    active = repo.list_active_dirty(project_id=project_id)
    assert len(active) == 1
    entry = active[0]
    assert entry.path == "workflows/sync.yaml"
    assert entry.event_type == "modified"
    assert entry.reason == "reconciliation"
    assert entry.id == first.id
    assert second.id == first.id
    assert entry.updated_at >= first.updated_at
    conn.close()


def test_dirty_history_includes_active_and_processed_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.enqueue_dirty(
        project_id=project_id,
        path="workflows/build.yaml",
        event_type="modified",
        reason="file_event",
    )
    repo.enqueue_dirty(
        project_id=project_id,
        path="workflows/release.yaml",
        event_type="created",
        reason="file_event",
    )
    repo.mark_active_dirty_processed(project_id=project_id)
    repo.enqueue_dirty(
        project_id=project_id,
        path="workflows/rebuild.yaml",
        event_type="rebuild",
        reason="reconciliation_required:manual_rebuild",
    )

    history = repo.list_dirty_history(project_id=project_id, limit=10)

    assert [entry.path for entry in history] == [
        "workflows/rebuild.yaml",
        "workflows/release.yaml",
        "workflows/build.yaml",
    ]
    assert history[0].processed_at is None
    assert history[1].processed_at is not None
    assert history[2].processed_at is not None
    conn.close()


def test_set_status_rejects_invalid_state(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    with pytest.raises(InvalidWatcherStateError):
        repo.set_status(project_id=project_id, state="unknown")
    conn.close()


def test_manager_restore_starts_only_enabled_watchers(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)

    enabled_project = _seed_project(conn)
    paused_project = _seed_project(conn)
    disabled_project = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.set_status(project_id=enabled_project, state="enabled")
    repo.set_status(project_id=paused_project, state="paused")
    repo.set_status(project_id=disabled_project, state="disabled")

    manager = WatcherManager(repo)
    restored = manager.restore_enabled_watchers()

    assert restored == [enabled_project]
    assert manager.active_project_ids == (enabled_project,)
    conn.close()


def test_manager_records_reliable_events_to_dirty_queue(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.set_status(project_id=project_id, state="enabled")
    manager = WatcherManager(repo)

    created = manager.record_event(
        WatcherEvent(project_id=project_id, path="wf/a.yaml", event_type="created")
    )
    renamed = manager.record_event(
        WatcherEvent(
            project_id=project_id,
            path="wf/b.yaml",
            event_type="renamed",
            previous_path="wf/a.yaml",
        )
    )

    assert created.project_id == project_id
    assert created.path == "wf/a.yaml"
    assert created.event_type == "created"
    assert created.reason == "file_event"

    assert renamed.project_id == project_id
    assert renamed.path == "wf/b.yaml"
    assert renamed.event_type == "renamed"
    assert renamed.reason == "file_event"

    status = repo.get_status(project_id)
    assert status is not None
    assert status.last_event_at is not None
    conn.close()


def test_manager_marks_unreliable_events_for_reconciliation(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.set_status(project_id=project_id, state="enabled")
    manager = WatcherManager(repo)

    dropped = manager.record_event(
        WatcherEvent(
            project_id=project_id,
            path="wf/c.yaml",
            event_type="dropped",
            message="queue overrun",
        )
    )
    unknown = manager.record_event(
        WatcherEvent(project_id=project_id, path="wf/d.yaml", event_type="unknown")
    )

    assert dropped.event_type == "dropped"
    assert dropped.reason == "reconciliation_required:dropped"
    assert unknown.event_type == "unknown"
    assert unknown.reason == "reconciliation_required:unknown"

    queue = repo.list_active_dirty(project_id=project_id)
    reasons_by_path = {entry.path: entry.reason for entry in queue}
    assert reasons_by_path["wf/c.yaml"] == "reconciliation_required:dropped"
    assert reasons_by_path["wf/d.yaml"] == "reconciliation_required:unknown"
    conn.close()


def test_manager_event_matrix_reliable_event_types(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.set_status(project_id=project_id, state="enabled")
    manager = WatcherManager(repo)

    modified = manager.record_event(
        WatcherEvent(project_id=project_id, path="wf/m.yaml", event_type="modified")
    )
    deleted = manager.record_event(
        WatcherEvent(project_id=project_id, path="wf/del.yaml", event_type="deleted")
    )

    assert modified.event_type == "modified"
    assert modified.reason == "file_event"
    assert deleted.event_type == "deleted"
    assert deleted.reason == "file_event"

    queue = repo.list_active_dirty(project_id=project_id)
    by_path = {entry.path: entry for entry in queue}
    assert by_path["wf/m.yaml"].event_type == "modified"
    assert by_path["wf/m.yaml"].reason == "file_event"
    assert by_path["wf/del.yaml"].event_type == "deleted"
    assert by_path["wf/del.yaml"].reason == "file_event"
    conn.close()


def test_manager_event_matrix_unreliable_event_types(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    project_id = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.set_status(project_id=project_id, state="enabled")
    manager = WatcherManager(repo)

    overflow = manager.record_event(
        WatcherEvent(project_id=project_id, path="wf/of.yaml", event_type="overflow")
    )
    permission_error = manager.record_event(
        WatcherEvent(
            project_id=project_id,
            path="wf/perm.yaml",
            event_type="permission_error",
            message="permission denied",
        )
    )

    assert overflow.event_type == "overflow"
    assert overflow.reason == "reconciliation_required:overflow"
    assert permission_error.event_type == "permission_error"
    assert permission_error.reason == "reconciliation_required:permission_error"

    queue = repo.list_active_dirty(project_id=project_id)
    by_path = {entry.path: entry for entry in queue}
    assert by_path["wf/of.yaml"].event_type == "overflow"
    assert by_path["wf/of.yaml"].reason == "reconciliation_required:overflow"
    assert by_path["wf/perm.yaml"].event_type == "permission_error"
    assert by_path["wf/perm.yaml"].reason == "reconciliation_required:permission_error"
    conn.close()


def test_manager_start_stop_is_idempotent_and_restores_enabled_watchers(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)

    enabled_project = _seed_project(conn)
    paused_project = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.set_status(project_id=enabled_project, state="enabled")
    repo.set_status(project_id=paused_project, state="paused")

    manager = WatcherManager(repo)

    first_restore = manager.start()
    second_restore = manager.start()

    assert first_restore == [enabled_project]
    assert second_restore == []
    assert manager.active_project_ids == (enabled_project,)

    manager.stop()
    assert manager.active_project_ids == ()

    # Stop is idempotent and remains empty.
    manager.stop()
    assert manager.active_project_ids == ()
    conn.close()


def test_manager_reports_polling_fallback_backend_mode_by_default(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)

    manager = WatcherManager(SQLiteWatcherRepository(conn))

    assert manager.backend_mode == "polling_fallback"
    assert manager.backend_native_available is False
    assert manager.is_started is False

    manager.start()
    assert manager.is_started is True
    assert manager.backend_mode == "polling_fallback"
    assert manager.backend_native_available is False

    manager.stop()
    conn.close()


def test_enable_project_by_default_is_idempotent_and_preserves_explicit_states(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)

    missing_project = _seed_project(conn)
    enabled_project = _seed_project(conn)
    paused_project = _seed_project(conn)
    disabled_project = _seed_project(conn)

    repo = SQLiteWatcherRepository(conn)
    repo.set_status(project_id=enabled_project, state="enabled")
    repo.set_status(project_id=paused_project, state="paused")
    repo.set_status(project_id=disabled_project, state="disabled")

    manager = WatcherManager(repo)

    manager.enable_project_by_default(missing_project)
    manager.enable_project_by_default(missing_project)
    manager.enable_project_by_default(enabled_project)
    manager.enable_project_by_default(enabled_project)
    manager.enable_project_by_default(paused_project)
    manager.enable_project_by_default(disabled_project)

    assert repo.get_status(missing_project) is not None
    assert repo.get_status(missing_project).state == "enabled"
    assert repo.get_status(enabled_project) is not None
    assert repo.get_status(enabled_project).state == "enabled"
    assert repo.get_status(paused_project) is not None
    assert repo.get_status(paused_project).state == "paused"
    assert repo.get_status(disabled_project) is not None
    assert repo.get_status(disabled_project).state == "disabled"
    conn.close()


def test_enable_project_by_default_starts_runtime_watcher_when_manager_started(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)

    project_id = _seed_project(conn)
    repo = SQLiteWatcherRepository(conn)
    manager = WatcherManager(repo)

    manager.start()
    assert manager.active_project_ids == ()

    manager.enable_project_by_default(project_id)
    assert manager.active_project_ids == (project_id,)
    conn.close()


def test_build_resources_wires_same_watcher_manager_into_app_context(tmp_path: Path) -> None:
    resources = build_resources(base_dir=tmp_path)
    try:
        assert resources.app_context.watcher_manager is resources.watcher_manager
    finally:
        resources.metadata_db_conn.close()
