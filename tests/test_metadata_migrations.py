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
        assert [int(row[0]) for row in versions] == [1, 2, 3, 4, 5, 6, 7]
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
        assert [int(row[0]) for row in versions] == [1, 2, 3, 4, 5, 6, 7]

        workflow_sources_indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list('workflow_sources')").fetchall()
        }
        assert "idx_workflow_sources_project_path_unique" in workflow_sources_indexes
    finally:
        conn.close()


def test_workflow_sources_unique_index_is_applied_idempotently_in_v3(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        migrate_metadata_db(conn)

        workflow_sources_indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list('workflow_sources')").fetchall()
        }
        assert "idx_workflow_sources_project_path_unique" in workflow_sources_indexes

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == [1, 2, 3, 4, 5, 6, 7]
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
        assert [int(row[0]) for row in versions] == [1, 3, 4, 5, 6, 7]
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
        assert [int(row[0]) for row in versions] == [1, 4, 5, 6, 7]
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
        assert [int(row[0]) for row in versions] == [1, 5, 6, 7]
    finally:
        conn.close()
