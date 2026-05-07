from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import (
    IncompatibleMetadataSchema,
    migrate_metadata_db,
)


def test_concurrent_metadata_migrations_apply_once(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    # Pre-initialize the DB once so concurrent test threads only race on migration,
    # not first-open PRAGMA negotiation.
    init_conn = connect_metadata_db(db_path)
    init_conn.close()

    def _run_migration() -> None:
        conn = connect_metadata_db(db_path)
        try:
            barrier.wait(timeout=5)
            migrate_metadata_db(conn)
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)
        finally:
            conn.close()

    t1 = threading.Thread(target=_run_migration)
    t2 = threading.Thread(target=_run_migration)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive(), "migration thread t1 did not terminate within timeout"
    assert not t2.is_alive(), "migration thread t2 did not terminate within timeout"

    assert not errors

    conn = connect_metadata_db(db_path)
    try:
        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    finally:
        conn.close()


def test_incompatible_metadata_schema_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    seed_conn = sqlite3.connect(db_path)
    try:
        seed_conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        seed_conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, CURRENT_TIMESTAMP)",
            (999,),
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    conn = connect_metadata_db(db_path)
    try:
        with pytest.raises(IncompatibleMetadataSchema):
            migrate_metadata_db(conn)

        # Fail-closed means migration DDL/DML is not applied once incompatible version is detected.
        server_settings = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='server_settings'"
        ).fetchall()
        assert server_settings == []

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [999]
    finally:
        conn.close()


def test_project_defaults_are_nullable_after_v7_shape_repair(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    seed_conn = sqlite3.connect(db_path)
    try:
        seed_conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        seed_conn.execute(
            """
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                palace TEXT NOT NULL UNIQUE,
                default_wing TEXT NOT NULL,
                default_room TEXT NOT NULL,
                fs_root TEXT NOT NULL,
                fs_allowlist_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        seed_conn.execute("CREATE TABLE mcp_tokens (id TEXT PRIMARY KEY)")
        seed_conn.execute(
            """
            CREATE TABLE project_token_bindings (
                token_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (token_id, project_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE CASCADE
            )
            """
        )
        seed_conn.execute(
            """
            INSERT INTO projects (
                id,
                name,
                slug,
                palace,
                default_wing,
                default_room,
                fs_root,
                fs_allowlist_json,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "project-existing",
                "Existing Project",
                "existing-project",
                "existing-palace",
                "existing-wing",
                "existing-room",
                "/tmp/existing",
                '["/tmp/existing"]',
                "2026-05-01T10:00:00Z",
                "2026-05-01T10:01:00Z",
            ),
        )
        seed_conn.execute("INSERT INTO mcp_tokens(id) VALUES (?)", ("token-existing",))
        seed_conn.execute(
            """
            INSERT INTO project_token_bindings(token_id, project_id)
            VALUES (?, ?)
            """,
            ("token-existing", "project-existing"),
        )
        seed_conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (7, CURRENT_TIMESTAMP)"
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        columns = conn.execute("PRAGMA table_info('projects')").fetchall()
        column_by_name = {str(column[1]): column for column in columns}
        assert int(column_by_name["default_wing"][3]) == 0
        assert int(column_by_name["default_room"][3]) == 0

        existing = conn.execute(
            """
            SELECT
                default_wing,
                default_room,
                fs_allowlist_json,
                created_at,
                updated_at
            FROM projects
            WHERE id = ?
            """,
            ("project-existing",),
        ).fetchone()
        assert existing is not None
        assert str(existing[0]) == "existing-wing"
        assert str(existing[1]) == "existing-room"
        assert str(existing[2]) == '["/tmp/existing"]'
        assert str(existing[3]) == "2026-05-01T10:00:00Z"
        assert str(existing[4]) == "2026-05-01T10:01:00Z"

        binding = conn.execute(
            """
            SELECT project_id
            FROM project_token_bindings
            WHERE token_id = ?
            """,
            ("token-existing",),
        ).fetchone()
        assert binding is not None
        assert str(binding[0]) == "project-existing"

        binding_foreign_keys = conn.execute(
            "PRAGMA foreign_key_list('project_token_bindings')"
        ).fetchall()
        assert any(
            str(row[2]) == "projects"
            and str(row[3]) == "project_id"
            and str(row[4]) == "id"
            for row in binding_foreign_keys
        )

        conn.execute(
            """
            INSERT INTO projects (
                id,
                name,
                slug,
                palace,
                default_wing,
                default_room,
                fs_root
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "project-null-defaults",
                "Null Defaults",
                "null-defaults",
                "null-defaults-palace",
                None,
                None,
                "/tmp/null-defaults",
            ),
        )
        inserted = conn.execute(
            """
            SELECT default_wing, default_room
            FROM projects
            WHERE id = ?
            """,
            ("project-null-defaults",),
        ).fetchone()
        assert inserted is not None
        assert inserted[0] is None
        assert inserted[1] is None

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 7, 8, 9, 10]
    finally:
        conn.close()


def test_watcher_queue_v1_shape_is_upgraded_to_v2(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    seed_conn = sqlite3.connect(db_path)
    try:
        seed_conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        seed_conn.execute(
            """
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                palace TEXT NOT NULL UNIQUE,
                default_wing TEXT NOT NULL,
                default_room TEXT NOT NULL,
                fs_root TEXT NOT NULL,
                fs_allowlist_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        seed_conn.execute(
            """
            CREATE TABLE watcher_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                enqueued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
            """
        )
        seed_conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, CURRENT_TIMESTAMP)"
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        columns = conn.execute("PRAGMA table_info('watcher_queue')").fetchall()
        column_names = {str(column[1]) for column in columns}
        assert {"id", "project_id", "reason", "enqueued_at"}.issubset(column_names)
        assert {"path", "event_type", "updated_at", "processed_at"}.issubset(column_names)

        column_by_name = {str(column[1]): column for column in columns}
        updated_at = column_by_name["updated_at"]
        assert int(updated_at[3]) == 1
        assert str(updated_at[4]) == "CURRENT_TIMESTAMP"

        index_names = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list('watcher_queue')").fetchall()
        }
        assert "idx_watcher_queue_active_dedupe" in index_names

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # After v10, workflow_sources has a global unique constraint on source_path (autoindex).
        index_rows = conn.execute("PRAGMA index_list('workflow_sources')").fetchall()
        assert any(int(row[2]) == 1 for row in index_rows), (
            f"No unique index on workflow_sources after migration; "
            f"indexes: {[r[1] for r in index_rows]}"
        )
    finally:
        conn.close()


def test_workflow_sources_unique_index_is_applied_idempotently_in_v3(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        migrate_metadata_db(conn)

        # After v10, workflow_sources has a global unique constraint on source_path (autoindex).
        index_rows = conn.execute("PRAGMA index_list('workflow_sources')").fetchall()
        assert any(int(row[2]) == 1 for row in index_rows), (
            f"No unique index on workflow_sources after idempotent migration; "
            f"indexes: {[r[1] for r in index_rows]}"
        )

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    finally:
        conn.close()


def test_postgresql_settings_structured_columns_are_added_without_touching_legacy_dsn(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.db"
    seed_conn = sqlite3.connect(db_path)
    try:
        seed_conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        seed_conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (6, CURRENT_TIMESTAMP)"
        )
        seed_conn.execute(
            """
            CREATE TABLE postgresql_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                dsn_ref TEXT,
                enabled INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        seed_conn.execute(
            "INSERT INTO postgresql_settings(id, dsn_ref, enabled) VALUES (1, 'postgresql.dsn', 1)"
        )
        seed_conn.execute(
            """
            CREATE TABLE encrypted_secret_metadata (
                id TEXT PRIMARY KEY,
                secret_name TEXT NOT NULL UNIQUE,
                key_id TEXT,
                encrypted_payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        seed_conn.execute(
            "INSERT INTO encrypted_secret_metadata"
            "(id, secret_name, encrypted_payload) "
            "VALUES ('s1', 'postgresql.dsn', 'ciphertext')"
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info('postgresql_settings')")
        }
        assert {
            "host",
            "port",
            "database",
            "username",
            "ssl_mode",
            "extra_params",
            "container_name",
            "container_image",
            "container_host_port",
            "volume_name",
            "legacy_dsn_upgrade_status",
        }.issubset(columns)
        settings = conn.execute(
            "SELECT dsn_ref, enabled, legacy_dsn_upgrade_status "
            "FROM postgresql_settings WHERE id = 1"
        ).fetchone()
        assert settings is not None
        assert str(settings[0]) == "postgresql.dsn"
        assert int(settings[1]) == 1
        assert str(settings[2]) == "not_started"
        secret = conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata "
            "WHERE secret_name = 'postgresql.dsn'"
        ).fetchone()
        assert secret is not None
        assert str(secret[0]) == "ciphertext"
    finally:
        conn.close()


def test_job_runs_v3_shape_is_upgraded_to_v4(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    seed_conn = sqlite3.connect(db_path)
    try:
        legacy_project_id = "project-legacy"
        legacy_run_id = "run-legacy"
        legacy_workflow_name = "legacy-workflow"
        legacy_status = "completed"
        legacy_started_at = "2026-04-28T11:22:33Z"
        legacy_finished_at = "2026-04-28T11:25:00Z"

        seed_conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        seed_conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY)")
        seed_conn.execute("CREATE TABLE mcp_tokens (id TEXT PRIMARY KEY)")
        seed_conn.execute(
            "INSERT INTO projects(id) VALUES (?)",
            (legacy_project_id,),
        )
        seed_conn.execute(
            """
            CREATE TABLE job_runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT
            )
            """
        )
        seed_conn.execute(
            """
            INSERT INTO job_runs (
                run_id,
                project_id,
                workflow_name,
                status,
                started_at,
                finished_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                legacy_run_id,
                legacy_project_id,
                legacy_workflow_name,
                legacy_status,
                legacy_started_at,
                legacy_finished_at,
            ),
        )
        seed_conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (3, CURRENT_TIMESTAMP)"
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        columns = conn.execute("PRAGMA table_info('job_runs')").fetchall()
        column_names = {str(column[1]) for column in columns}
        assert {
            "run_id",
            "project_id",
            "token_id",
            "workflow_name",
            "status",
            "cancellable",
            "result_summary",
            "error_summary",
            "started_at",
            "finished_at",
        }.issubset(column_names)

        row = conn.execute(
            """
            SELECT
                run_id,
                project_id,
                token_id,
                workflow_name,
                status,
                cancellable,
                result_summary,
                error_summary,
                started_at,
                finished_at
            FROM job_runs
            WHERE run_id = ?
            """,
            (legacy_run_id,),
        ).fetchone()
        assert row is not None
        assert str(row[0]) == legacy_run_id
        assert str(row[1]) == legacy_project_id
        assert row[2] is None
        assert str(row[3]) == legacy_workflow_name
        assert str(row[4]) == legacy_status
        assert int(row[5]) == 0
        assert row[6] is None
        assert row[7] is None
        assert str(row[8]) == legacy_started_at
        assert str(row[9]) == legacy_finished_at

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 3, 4, 5, 6, 7, 8, 9, 10]
    finally:
        conn.close()


def test_job_runs_v4_shape_is_upgraded_to_v5_with_safe_defaults(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    seed_conn = sqlite3.connect(db_path)
    try:
        legacy_run_id = "run-v4"
        legacy_project_id = "project-v4"
        legacy_workflow_name = "workflow-v4"
        legacy_status = "running"
        legacy_started_at = "2026-04-29T09:00:00Z"

        seed_conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        seed_conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY)")
        seed_conn.execute("CREATE TABLE mcp_tokens (id TEXT PRIMARY KEY)")
        seed_conn.execute("INSERT INTO projects(id) VALUES (?)", (legacy_project_id,))
        seed_conn.execute(
            """
            CREATE TABLE job_runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT,
                token_id TEXT,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL,
                cancellable INTEGER NOT NULL DEFAULT 0,
                result_summary TEXT,
                error_summary TEXT,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL,
                FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE SET NULL
            )
            """
        )
        seed_conn.execute(
            """
            INSERT INTO job_runs (
                run_id,
                project_id,
                token_id,
                workflow_name,
                status,
                cancellable,
                result_summary,
                error_summary,
                started_at,
                finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                legacy_run_id,
                legacy_project_id,
                None,
                legacy_workflow_name,
                legacy_status,
                0,
                None,
                None,
                legacy_started_at,
                None,
            ),
        )
        seed_conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (4, CURRENT_TIMESTAMP)"
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        columns = conn.execute("PRAGMA table_info('job_runs')").fetchall()
        column_names = {str(column[1]) for column in columns}
        assert {
            "run_id",
            "project_id",
            "token_id",
            "workflow_name",
            "status",
            "cancellable",
            "result_summary",
            "error_summary",
            "started_at",
            "finished_at",
            "timeout_seconds",
            "updated_at",
            "inputs_json",
        }.issubset(column_names)

        row = conn.execute(
            """
            SELECT
                run_id,
                workflow_name,
                status,
                started_at,
                finished_at,
                timeout_seconds,
                updated_at,
                inputs_json
            FROM job_runs
            WHERE run_id = ?
            """,
            (legacy_run_id,),
        ).fetchone()
        assert row is not None
        assert str(row[0]) == legacy_run_id
        assert str(row[1]) == legacy_workflow_name
        assert str(row[2]) == legacy_status
        assert str(row[3]) == legacy_started_at
        assert row[4] is None
        assert int(row[5]) == 3600
        assert str(row[6]) == legacy_started_at
        assert row[7] is None

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 4, 5, 6, 7, 8, 9, 10]
    finally:
        conn.close()


def test_job_runs_v5_shape_is_upgraded_to_v6_with_created_started_contract(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"

    seed_conn = sqlite3.connect(db_path)
    try:
        seed_conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        seed_conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY)")
        seed_conn.execute("CREATE TABLE mcp_tokens (id TEXT PRIMARY KEY)")
        seed_conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (5, CURRENT_TIMESTAMP)"
        )
        seed_conn.execute(
            """
            CREATE TABLE job_runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT,
                token_id TEXT,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL,
                cancellable INTEGER NOT NULL DEFAULT 0,
                result_summary TEXT,
                error_summary TEXT,
                timeout_seconds INTEGER NOT NULL DEFAULT 3600,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                inputs_json TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL,
                FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE SET NULL
            )
            """
        )

        seed_conn.executemany(
            """
            INSERT INTO job_runs(
                run_id, project_id, token_id, workflow_name, status, cancellable,
                result_summary, error_summary, timeout_seconds, started_at, updated_at,
                finished_at, inputs_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "run-queued-v5",
                    None,
                    None,
                    "wf",
                    "queued",
                    0,
                    None,
                    None,
                    30,
                    "2026-04-29T12:00:00Z",
                    "2026-04-29T12:00:00Z",
                    None,
                    None,
                ),
                (
                    "run-running-v5",
                    None,
                    None,
                    "wf",
                    "running",
                    1,
                    None,
                    None,
                    30,
                    "2026-04-29T12:00:01Z",
                    "2026-04-29T12:00:01Z",
                    None,
                    None,
                ),
                (
                    "run-completed-v5",
                    None,
                    None,
                    "wf",
                    "completed",
                    0,
                    "ok",
                    None,
                    30,
                    "2026-04-29T12:00:02Z",
                    "2026-04-29T12:00:03Z",
                    "2026-04-29T12:00:04Z",
                    None,
                ),
            ],
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        columns = conn.execute("PRAGMA table_info('job_runs')").fetchall()
        column_names = {str(column[1]) for column in columns}
        assert "created_at" in column_names
        assert "started_at" in column_names

        rows = conn.execute(
            """
            SELECT run_id, status, created_at, started_at
            FROM job_runs
            ORDER BY run_id ASC
            """
        ).fetchall()
        by_id = {str(row[0]): row for row in rows}

        queued = by_id["run-queued-v5"]
        assert str(queued[1]) == "queued"
        assert str(queued[2]) == "2026-04-29T12:00:00Z"
        assert queued[3] is None

        running = by_id["run-running-v5"]
        assert str(running[1]) == "running"
        assert str(running[2]) == "2026-04-29T12:00:01Z"
        assert str(running[3]) == "2026-04-29T12:00:01Z"

        completed = by_id["run-completed-v5"]
        assert str(completed[1]) == "completed"
        assert str(completed[2]) == "2026-04-29T12:00:02Z"
        assert str(completed[3]) == "2026-04-29T12:00:02Z"

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 5, 6, 7, 8, 9, 10]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# v9: workflow_sources.is_system column
# ---------------------------------------------------------------------------


def _sqlite_column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {str(row[1]) for row in rows}


def _make_metadata_db(db_path: Path) -> sqlite3.Connection:
    """Open a new SQLite metadata DB with all migrations applied."""
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    return conn


def test_fresh_schema_workflow_sources_columns(tmp_path: Path) -> None:
    """A freshly initialised metadata DB must have the v10 workflow_sources shape
    (no is_system, no project_id)."""
    conn = _make_metadata_db(tmp_path / "fresh.db")
    try:
        cols = _sqlite_column_names(conn, "workflow_sources")
        assert "is_system" not in cols, f"is_system must not exist in v10 schema; columns: {cols}"
        assert "project_id" not in cols, f"project_id must not exist in v10 schema; columns: {cols}"
        assert "source_path" in cols, f"source_path missing from workflow_sources; columns: {cols}"
    finally:
        conn.close()


def test_fresh_schema_workflow_source_create(tmp_path: Path) -> None:
    """New workflow_sources rows use global (project-free) create API."""
    from workflows_mcp.metadata.repos.workflow_sources_repo import (
        SQLiteWorkflowSourcesRepository,
        WorkflowSourceCreate,
    )

    conn = _make_metadata_db(tmp_path / "default.db")
    try:
        source_dir = tmp_path / "wf-dir"
        source_dir.mkdir()
        sources_repo = SQLiteWorkflowSourcesRepository(conn)
        record = sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
        assert record.source_path == str(source_dir)

        raw = conn.execute(
            "SELECT source_path FROM workflow_sources WHERE source_id = ?",
            (record.source_id,),
        ).fetchone()
        assert raw is not None
        assert str(raw[0]) == str(source_dir)
    finally:
        conn.close()


def test_migration_schema_version_advances_to_ten(tmp_path: Path) -> None:
    """After migration the schema_migrations table must contain version 10."""
    from workflows_mcp.metadata.migrations import CURRENT_SCHEMA_VERSION

    assert CURRENT_SCHEMA_VERSION == 10, (
        f"Expected CURRENT_SCHEMA_VERSION == 10, got {CURRENT_SCHEMA_VERSION}"
    )

    conn = connect_metadata_db(tmp_path / "version.db")
    try:
        migrate_metadata_db(conn)
        row = conn.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()
        assert row is not None
        assert int(row[0]) >= 10
    finally:
        conn.close()


def test_v8_to_v9_and_v10_migration_from_pre_v9_db(tmp_path: Path) -> None:
    """v8->v9->v10 regression: source row is preserved; final schema has no is_system column.

    This test manually constructs a minimal pre-v9 SQLite metadata schema that
    mirrors the state a real v8 DB would be in: workflow_sources exists without
    the is_system column and schema_migrations contains version 8 as the max.
    After calling migrate_metadata_db the column chain runs through v9 (adds is_system)
    and v10 (removes project_id/is_system), leaving the source row intact without those columns.
    """
    db_path = tmp_path / "pre_v9.db"

    # Build a minimal v8 schema by hand — no is_system column on workflow_sources.
    seed_conn = sqlite3.connect(db_path)
    try:
        seed_conn.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                palace TEXT NOT NULL UNIQUE,
                default_wing TEXT,
                default_room TEXT,
                fs_root TEXT NOT NULL,
                fs_allowlist_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE mcp_tokens (id TEXT PRIMARY KEY);
            CREATE TABLE project_token_bindings (
                token_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (token_id, project_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE CASCADE
            );
            CREATE TABLE workflow_sources (
                source_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                checksum TEXT,
                discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX idx_workflow_sources_project_path_unique
                ON workflow_sources(project_id, source_path);
            CREATE TABLE watcher_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                enqueued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                processed_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX idx_watcher_queue_active_dedupe
                ON watcher_queue(project_id, path, event_type)
                WHERE processed_at IS NULL;
            CREATE TABLE encrypted_secret_metadata (
                id TEXT PRIMARY KEY,
                secret_name TEXT NOT NULL UNIQUE,
                key_id TEXT,
                encrypted_payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE server_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                setup_complete INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE postgresql_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                dsn_ref TEXT,
                host TEXT NOT NULL DEFAULT '127.0.0.1',
                port INTEGER NOT NULL DEFAULT 5432,
                database TEXT NOT NULL DEFAULT 'workflows',
                username TEXT NOT NULL DEFAULT 'workflows',
                ssl_mode TEXT NOT NULL DEFAULT 'disable',
                extra_params TEXT NOT NULL DEFAULT '',
                container_name TEXT NOT NULL DEFAULT 'workflows-postgres',
                container_image TEXT NOT NULL DEFAULT 'pgvector/pgvector:pg17',
                container_host_port INTEGER NOT NULL DEFAULT 5432,
                volume_name TEXT NOT NULL DEFAULT 'workflows-postgres-data',
                legacy_dsn_upgrade_status TEXT NOT NULL DEFAULT 'not_started',
                enabled INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE job_runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT,
                token_id TEXT,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL,
                cancellable INTEGER NOT NULL DEFAULT 0,
                result_summary TEXT,
                error_summary TEXT,
                timeout_seconds INTEGER NOT NULL DEFAULT 3600,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                inputs_json TEXT,
                execution_mode TEXT NOT NULL DEFAULT 'async',
                execution_json TEXT,
                execution_state_json TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL,
                FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE SET NULL
            );
            CREATE INDEX idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC);
            CREATE INDEX idx_job_runs_status ON job_runs(status ASC, started_at ASC, run_id ASC);
            CREATE INDEX idx_job_runs_created_at ON job_runs(created_at ASC, run_id ASC);
            CREATE INDEX idx_job_runs_workflow
                ON job_runs(workflow_name ASC, created_at DESC, run_id ASC);
            CREATE INDEX idx_job_runs_mode
                ON job_runs(execution_mode ASC, created_at DESC, run_id ASC);
            CREATE TABLE workflow_reload_state (
                source_id TEXT PRIMARY KEY,
                last_loaded_at TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                FOREIGN KEY (source_id) REFERENCES workflow_sources(source_id) ON DELETE CASCADE
            );
            INSERT INTO schema_migrations(version) VALUES (1),(2),(3),(4),(5),(6),(7),(8);
            """
        )
        # Insert a pre-existing workflow_sources row (no is_system column yet).
        seed_conn.execute(
            "INSERT INTO projects(id, name, slug, palace, fs_root) VALUES (?,?,?,?,?)",
            ("proj-v8", "V8 Project", "v8-project", "palace.v8", "/tmp/v8"),
        )
        seed_conn.execute(
            "INSERT INTO workflow_sources(source_id, project_id, source_path) VALUES (?,?,?)",
            ("src-v8", "proj-v8", "/tmp/v8/workflows"),
        )
        seed_conn.commit()
    finally:
        seed_conn.close()

    # Confirm is_system is absent before migration.
    pre_cols = _sqlite_column_names(sqlite3.connect(db_path), "workflow_sources")
    assert "is_system" not in pre_cols, "Pre-condition failed: is_system should not exist in v8 DB"

    # Run migration via the official path (v8 -> v9 -> v10).
    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        # After v10, is_system and project_id must be gone.
        cols = _sqlite_column_names(conn, "workflow_sources")
        assert "is_system" not in cols, (
            f"is_system must not exist after v10 migration; columns: {cols}"
        )
        assert "project_id" not in cols, (
            f"project_id must not exist after v10 migration; columns: {cols}"
        )

        # Pre-existing user source row must be preserved.
        row = conn.execute(
            "SELECT source_path FROM workflow_sources WHERE source_id = ?",
            ("src-v8",),
        ).fetchone()
        assert row is not None, "Pre-existing source row must survive v10 migration"
        assert str(row[0]) == "/tmp/v8/workflows"

        # Versions 9 and 10 both recorded.
        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        version_ints = [int(r[0]) for r in versions]
        assert 9 in version_ints
        assert 10 in version_ints
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# v10: Remove project_id and is_system from workflow_sources; drop System project row
# ---------------------------------------------------------------------------


def _build_pre_v10_db(db_path: Path, *, duplicate_paths: bool = False) -> None:
    """Construct a realistic v9 DB with workflow_sources rows and a legacy System project."""
    seed_conn = sqlite3.connect(db_path)
    try:
        seed_conn.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                palace TEXT NOT NULL UNIQUE,
                default_wing TEXT,
                default_room TEXT,
                fs_root TEXT NOT NULL,
                fs_allowlist_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE mcp_tokens (id TEXT PRIMARY KEY);
            CREATE TABLE project_token_bindings (
                token_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (token_id, project_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE CASCADE
            );
            CREATE TABLE workflow_sources (
                source_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                checksum TEXT,
                is_system INTEGER NOT NULL DEFAULT 0,
                discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX idx_workflow_sources_project_path_unique
                ON workflow_sources(project_id, source_path);
            CREATE TABLE workflow_reload_state (
                source_id TEXT PRIMARY KEY,
                last_loaded_at TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                FOREIGN KEY (source_id) REFERENCES workflow_sources(source_id) ON DELETE CASCADE
            );
            CREATE TABLE server_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE admin_credentials (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                password_hash TEXT NOT NULL,
                password_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE admin_sessions (
                session_hash TEXT PRIMARY KEY,
                csrf_token_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                idle_expires_at TEXT NOT NULL,
                absolute_expires_at TEXT NOT NULL,
                revoked_at TEXT
            );
            CREATE TABLE session_secrets (
                id TEXT PRIMARY KEY,
                secret_ref TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT
            );
            CREATE TABLE encrypted_secret_metadata (
                id TEXT PRIMARY KEY,
                secret_name TEXT NOT NULL UNIQUE,
                key_id TEXT,
                encrypted_payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE llm_providers (
                provider_name TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE llm_profiles (
                profile_name TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                model TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (provider_name)
                    REFERENCES llm_providers(provider_name) ON DELETE CASCADE
            );
            CREATE TABLE postgresql_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                dsn_ref TEXT,
                enabled INTEGER NOT NULL DEFAULT 0,
                host TEXT NOT NULL DEFAULT '127.0.0.1',
                port INTEGER NOT NULL DEFAULT 5432,
                database TEXT NOT NULL DEFAULT 'workflows',
                username TEXT NOT NULL DEFAULT 'workflows',
                ssl_mode TEXT NOT NULL DEFAULT 'disable',
                extra_params TEXT NOT NULL DEFAULT '',
                container_name TEXT NOT NULL DEFAULT 'workflows-postgres',
                container_image TEXT NOT NULL DEFAULT 'pgvector/pgvector:pg17',
                container_host_port INTEGER NOT NULL DEFAULT 5432,
                volume_name TEXT NOT NULL DEFAULT 'workflows-postgres-data',
                legacy_dsn_upgrade_status TEXT NOT NULL DEFAULT 'not_started',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE watcher_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                enqueued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                processed_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX idx_watcher_queue_active_dedupe
                ON watcher_queue(project_id, path, event_type) WHERE processed_at IS NULL;
            CREATE TABLE watcher_status (
                project_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                last_event_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE TABLE job_runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT,
                token_id TEXT,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL,
                execution_mode TEXT NOT NULL DEFAULT 'async',
                cancellable INTEGER NOT NULL DEFAULT 0,
                result_summary TEXT,
                error_summary TEXT,
                execution_state_json TEXT,
                execution_json TEXT,
                timeout_seconds INTEGER NOT NULL DEFAULT 3600,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                inputs_json TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL,
                FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE SET NULL
            );
            CREATE INDEX idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC);
            CREATE INDEX idx_job_runs_status ON job_runs(status ASC, started_at ASC, run_id ASC);
            CREATE INDEX idx_job_runs_created_at ON job_runs(created_at ASC, run_id ASC);
            CREATE INDEX idx_job_runs_workflow
                ON job_runs(workflow_name ASC, created_at DESC, run_id ASC);
            CREATE INDEX idx_job_runs_mode
                ON job_runs(execution_mode ASC, created_at DESC, run_id ASC);
            """
        )
        # Insert the legacy System project (seeded by old code).
        seed_conn.execute(
            "INSERT INTO projects(id, name, slug, palace, fs_root) VALUES (?,?,?,?,?)",
            ("system-project-id", "System", "system", "__system__", "/builtin/path"),
        )
        # Insert a legacy is_system=1 source row.
        seed_conn.execute(
            """
            INSERT INTO workflow_sources(source_id, project_id, source_path, is_system)
            VALUES (?, ?, ?, ?)
            """,
            ("system-source-id", "system-project-id", "/deleted/builtin/path", 1),
        )
        seed_conn.execute(
            "INSERT INTO workflow_reload_state(source_id, status) VALUES (?, ?)",
            ("system-source-id", "loaded"),
        )
        # Insert a user project and a real user source.
        seed_conn.execute(
            "INSERT INTO projects(id, name, slug, palace, fs_root) VALUES (?,?,?,?,?)",
            ("user-project-id", "User Project", "user-project", "user-palace", "/user/wf"),
        )
        seed_conn.execute(
            """
            INSERT INTO workflow_sources(source_id, project_id, source_path, is_system)
            VALUES (?, ?, ?, ?)
            """,
            ("user-source-id", "user-project-id", "/user/wf/dir", 0),
        )
        seed_conn.execute(
            "INSERT INTO workflow_reload_state(source_id, status) VALUES (?, ?)",
            ("user-source-id", "loaded"),
        )

        if duplicate_paths:
            # Add a second user project with the same source_path (triggers fail-closed).
            seed_conn.execute(
                "INSERT INTO projects(id, name, slug, palace, fs_root) VALUES (?,?,?,?,?)",
                ("user2-project-id", "User2", "user2-slug", "palace2", "/user2/wf"),
            )
            seed_conn.execute(
                """
                INSERT INTO workflow_sources(source_id, project_id, source_path, is_system)
                VALUES (?, ?, ?, ?)
                """,
                ("user2-source-id", "user2-project-id", "/user/wf/dir", 0),
            )

        seed_conn.execute(
            "INSERT INTO schema_migrations(version) VALUES (1),(2),(3),(4),(5),(6),(7),(8),(9)"
        )
        seed_conn.commit()
    finally:
        seed_conn.close()


def test_v10_migration_schema_version_advances_to_ten(tmp_path: Path) -> None:
    """After migration schema_migrations must contain version 10."""
    from workflows_mcp.metadata.migrations import CURRENT_SCHEMA_VERSION

    assert CURRENT_SCHEMA_VERSION == 10, (
        f"Expected CURRENT_SCHEMA_VERSION == 10, got {CURRENT_SCHEMA_VERSION}"
    )

    conn = connect_metadata_db(tmp_path / "v10.db")
    try:
        migrate_metadata_db(conn)
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        assert row is not None
        assert int(row[0]) >= 10
    finally:
        conn.close()


def test_v10_migration_removes_project_id_and_is_system_from_workflow_sources(
    tmp_path: Path,
) -> None:
    """v9->v10: workflow_sources must not have project_id or is_system after migration."""
    db_path = tmp_path / "v10_cols.db"
    _build_pre_v10_db(db_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        cols = _sqlite_column_names(conn, "workflow_sources")
        assert "project_id" not in cols, f"project_id must be removed; columns: {cols}"
        assert "is_system" not in cols, f"is_system must be removed; columns: {cols}"
        assert "source_id" in cols
        assert "source_path" in cols
        assert "discovered_at" in cols
    finally:
        conn.close()


def test_v10_migration_preserves_user_source_row_and_reload_state(tmp_path: Path) -> None:
    """v9->v10: non-system user source row is preserved with reload state intact."""
    db_path = tmp_path / "v10_preserve.db"
    _build_pre_v10_db(db_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        row = conn.execute(
            "SELECT source_id, source_path FROM workflow_sources WHERE source_id = ?",
            ("user-source-id",),
        ).fetchone()
        assert row is not None, "User source row must be preserved after migration"
        assert str(row[1]) == "/user/wf/dir"

        reload_row = conn.execute(
            "SELECT status FROM workflow_reload_state WHERE source_id = ?",
            ("user-source-id",),
        ).fetchone()
        assert reload_row is not None, "Reload state for user source must be preserved"
        assert str(reload_row[0]) == "loaded"
    finally:
        conn.close()


def test_v10_migration_drops_is_system_source_rows(tmp_path: Path) -> None:
    """v9->v10: legacy is_system=1 source rows must be deleted."""
    db_path = tmp_path / "v10_system.db"
    _build_pre_v10_db(db_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        row = conn.execute(
            "SELECT 1 FROM workflow_sources WHERE source_id = ?",
            ("system-source-id",),
        ).fetchone()
        assert row is None, "Legacy is_system=1 source row must be deleted"
    finally:
        conn.close()


def test_v10_migration_drops_exact_system_project_row(tmp_path: Path) -> None:
    """v9->v10: exact legacy System project (slug=system, palace=__system__) is deleted."""
    db_path = tmp_path / "v10_proj.db"
    _build_pre_v10_db(db_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        row = conn.execute(
            "SELECT 1 FROM projects WHERE slug = ? AND palace = ?",
            ("system", "__system__"),
        ).fetchone()
        assert row is None, "Exact legacy System project row must be deleted"

        # Other projects must remain.
        user_project = conn.execute(
            "SELECT 1 FROM projects WHERE id = ?",
            ("user-project-id",),
        ).fetchone()
        assert user_project is not None, "Non-system user project must not be deleted"
    finally:
        conn.close()


def test_v10_migration_fails_closed_on_duplicate_non_system_source_paths(
    tmp_path: Path,
) -> None:
    """v9->v10: duplicate non-system source paths cause a RuntimeError with actionable message."""
    db_path = tmp_path / "v10_dup.db"
    _build_pre_v10_db(db_path, duplicate_paths=True)

    conn = connect_metadata_db(db_path)
    try:
        dup_pattern = "[Dd]uplicate.*source_path|source_path.*[Dd]uplicate"
        with pytest.raises(RuntimeError, match=dup_pattern):
            migrate_metadata_db(conn)
    finally:
        conn.close()


def test_v10_migration_source_path_globally_unique_index(tmp_path: Path) -> None:
    """After v10 migration, workflow_sources has a unique index on source_path."""
    db_path = tmp_path / "v10_idx.db"
    _build_pre_v10_db(db_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        # PRAGMA index_list columns: (seq, name, unique, origin, partial)
        index_rows = conn.execute("PRAGMA index_list('workflow_sources')").fetchall()
        assert any(int(row[2]) == 1 for row in index_rows), (
            f"No unique index found on workflow_sources; indexes: {[r[1] for r in index_rows]}"
        )

        # Verify the unique constraint actually works.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO workflow_sources(source_id, source_path, discovered_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                ("dup1", "/user/wf/dir"),
            )
            conn.execute(
                "INSERT INTO workflow_sources(source_id, source_path, discovered_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                ("dup2", "/user/wf/dir"),
            )
            conn.commit()
    finally:
        conn.close()
