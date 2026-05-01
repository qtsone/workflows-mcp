from __future__ import annotations

import sqlite3
from pathlib import Path

from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import CURRENT_SCHEMA_VERSION, migrate_metadata_db


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _expected_schema_versions() -> list[int]:
    return list(range(1, CURRENT_SCHEMA_VERSION + 1))


def test_connect_metadata_db_sets_pragmas_and_creates_parent_dirs(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "metadata" / "metadata.db"

    conn = connect_metadata_db(db_path)
    try:
        assert db_path.parent.exists()
        assert db_path.exists()
        assert conn.row_factory is sqlite3.Row

        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]

        assert foreign_keys == 1
        assert str(journal_mode).lower() == "wal"
        assert int(busy_timeout) >= 5000
    finally:
        conn.close()


def test_migrate_metadata_db_creates_required_and_reserved_tables_and_version(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        tables = _table_names(conn)
        required_tables = {
            "schema_migrations",
            "server_settings",
            "admin_credentials",
            "projects",
        }
        reserved_future_tables = {
            "admin_sessions",
            "session_secrets",
            "encrypted_secret_metadata",
            "llm_providers",
            "llm_profiles",
            "postgresql_settings",
            "mcp_tokens",
            "project_token_bindings",
            "watcher_queue",
            "watcher_status",
            "workflow_sources",
            "workflow_reload_state",
            "job_runs",
        }
        assert required_tables.issubset(tables)
        assert reserved_future_tables.issubset(tables)

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == _expected_schema_versions()
    finally:
        conn.close()


def test_migrate_metadata_db_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        migrate_metadata_db(conn)

        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(row[0]) for row in versions] == _expected_schema_versions()
    finally:
        conn.close()
