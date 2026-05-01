from __future__ import annotations

import sqlite3
from pathlib import Path

CURRENT_SCHEMA_VERSION = 6
DEFAULT_JOB_TIMEOUT_SECONDS = 3600


class IncompatibleMetadataSchema(RuntimeError):  # noqa: N818 - exception naming kept explicit for domain clarity
    """Raised when on-disk metadata schema is newer than this binary supports."""


def _schema_sql_path() -> Path:
    return Path(__file__).with_name("schema_v1.sql")


def _split_sql_statements(sql: str) -> list[str]:
    """Split simple migration SQL by ';' line endings.

    This intentionally handles our controlled schema files only and does not parse
    SQL strings, triggers, or procedure bodies that embed semicolons.
    """
    statements: list[str] = []
    buffer: list[str] = []

    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip()
            statements.append(statement)
            buffer.clear()

    trailing = "\n".join(buffer).strip()
    if trailing:
        statements.append(trailing)

    return statements


def migrate_metadata_db(conn: sqlite3.Connection) -> None:
    sql = _schema_sql_path().read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)

    conn.execute("BEGIN IMMEDIATE")
    try:
        max_version = 0
        schema_table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        ).fetchone()
        if schema_table_exists is not None:
            max_version_row = conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()
            max_version = (
                int(max_version_row[0])
                if max_version_row and max_version_row[0] is not None
                else 0
            )
            if max_version > CURRENT_SCHEMA_VERSION:
                raise IncompatibleMetadataSchema(
                    "Metadata schema version is newer than supported by this server binary"
                )

        for statement in statements:
            conn.execute(statement)

        if max_version < 2:
            _migrate_v1_to_v2(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)"
            )
        if max_version < 3:
            _migrate_v2_to_v3(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (3)"
            )
        if max_version < 4:
            _migrate_v3_to_v4(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (4)"
            )
        if max_version < 5:
            _migrate_v4_to_v5(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (5)"
            )
        if max_version < 6:
            _migrate_v5_to_v6(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (6)"
            )

        # Phase 10+ internal-only resumable state for paused runs.
        # Keep schema version stable while ensuring additive column exists.
        _ensure_job_runs_execution_state_column(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return {str(row[1]) for row in rows}


def _index_exists(conn: sqlite3.Connection, index_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (index_name,),
    ).fetchone()
    return row is not None


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    watcher_columns = conn.execute("PRAGMA table_info('watcher_queue')").fetchall()
    if _watcher_queue_needs_rebuild(watcher_columns):
        _rebuild_watcher_queue_to_v2_shape(conn, watcher_columns)

    if not _index_exists(conn, "idx_watcher_queue_active_dedupe"):
        conn.execute(
            """
            CREATE UNIQUE INDEX idx_watcher_queue_active_dedupe
            ON watcher_queue(project_id, path, event_type)
            WHERE processed_at IS NULL
            """
        )


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    if not _index_exists(conn, "idx_workflow_sources_project_path_unique"):
        conn.execute(
            """
            CREATE UNIQUE INDEX idx_workflow_sources_project_path_unique
            ON workflow_sources(project_id, source_path)
            """
        )


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    job_runs_columns = _table_columns(conn, "job_runs")
    required = {
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
    }
    if required.issubset(job_runs_columns):
        if not _index_exists(conn, "idx_job_runs_started_at"):
            conn.execute(
                "CREATE INDEX idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC)"
            )
        if not _index_exists(conn, "idx_job_runs_status"):
            conn.execute(
                "CREATE INDEX idx_job_runs_status ON job_runs("
                "status ASC, started_at ASC, run_id ASC)"
            )
        return

    conn.execute("ALTER TABLE job_runs RENAME TO job_runs_legacy_v3")
    conn.execute(
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

    legacy_columns = _table_columns(conn, "job_runs_legacy_v3")
    token_id_expr = "token_id" if "token_id" in legacy_columns else "NULL"
    cancellable_expr = "cancellable" if "cancellable" in legacy_columns else "0"
    result_summary_expr = "result_summary" if "result_summary" in legacy_columns else "NULL"
    error_summary_expr = "error_summary" if "error_summary" in legacy_columns else "NULL"

    conn.execute(
        f"""
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
        )
        SELECT
            run_id,
            project_id,
            {token_id_expr},
            workflow_name,
            status,
            {cancellable_expr},
            {result_summary_expr},
            {error_summary_expr},
            started_at,
            finished_at
        FROM job_runs_legacy_v3
        """
    )
    conn.execute("DROP TABLE job_runs_legacy_v3")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_runs_status ON job_runs("
        "status ASC, started_at ASC, run_id ASC)"
    )


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    job_runs_columns = _table_columns(conn, "job_runs")
    required = {
        "run_id",
        "project_id",
        "token_id",
        "workflow_name",
        "status",
        "cancellable",
        "result_summary",
        "error_summary",
        "timeout_seconds",
        "started_at",
        "updated_at",
        "finished_at",
        "inputs_json",
    }
    if required.issubset(job_runs_columns):
        if not _index_exists(conn, "idx_job_runs_started_at"):
            conn.execute(
                "CREATE INDEX idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC)"
            )
        if not _index_exists(conn, "idx_job_runs_status"):
            conn.execute(
                "CREATE INDEX idx_job_runs_status ON job_runs("
                "status ASC, started_at ASC, run_id ASC)"
            )
        return

    conn.execute("ALTER TABLE job_runs RENAME TO job_runs_legacy_v4")
    conn.execute(
        f"""
        CREATE TABLE job_runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT,
            token_id TEXT,
            workflow_name TEXT NOT NULL,
            status TEXT NOT NULL,
            cancellable INTEGER NOT NULL DEFAULT 0,
            result_summary TEXT,
            error_summary TEXT,
            timeout_seconds INTEGER NOT NULL DEFAULT {DEFAULT_JOB_TIMEOUT_SECONDS},
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT,
            inputs_json TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL,
            FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE SET NULL
        )
        """
    )

    legacy_columns = _table_columns(conn, "job_runs_legacy_v4")
    timeout_default_sql = str(DEFAULT_JOB_TIMEOUT_SECONDS)
    timeout_expr = (
        f"COALESCE(timeout_seconds, {timeout_default_sql})"
        if "timeout_seconds" in legacy_columns
        else timeout_default_sql
    )
    updated_at_expr = (
        "COALESCE(updated_at, started_at, CURRENT_TIMESTAMP)"
        if "updated_at" in legacy_columns
        else "COALESCE(started_at, CURRENT_TIMESTAMP)"
    )
    inputs_json_expr = "inputs_json" if "inputs_json" in legacy_columns else "NULL"

    conn.execute(
        f"""
        INSERT INTO job_runs (
            run_id,
            project_id,
            token_id,
            workflow_name,
            status,
            cancellable,
            result_summary,
            error_summary,
            timeout_seconds,
            started_at,
            updated_at,
            finished_at,
            inputs_json
        )
        SELECT
            run_id,
            project_id,
            token_id,
            workflow_name,
            status,
            cancellable,
            result_summary,
            error_summary,
            {timeout_expr},
            started_at,
            {updated_at_expr},
            finished_at,
            {inputs_json_expr}
        FROM job_runs_legacy_v4
        """
    )
    conn.execute("DROP TABLE job_runs_legacy_v4")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_runs_status ON job_runs("
        "status ASC, started_at ASC, run_id ASC)"
    )


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    job_runs_columns = _table_columns(conn, "job_runs")
    required = {
        "run_id",
        "project_id",
        "token_id",
        "workflow_name",
        "status",
        "cancellable",
        "result_summary",
        "error_summary",
        "timeout_seconds",
        "created_at",
        "started_at",
        "updated_at",
        "finished_at",
        "inputs_json",
    }
    if required.issubset(job_runs_columns):
        if not _index_exists(conn, "idx_job_runs_created_at"):
            conn.execute(
                "CREATE INDEX idx_job_runs_created_at ON job_runs(created_at ASC, run_id ASC)"
            )
        if not _index_exists(conn, "idx_job_runs_started_at"):
            conn.execute(
                "CREATE INDEX idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC)"
            )
        if not _index_exists(conn, "idx_job_runs_status"):
            conn.execute(
                "CREATE INDEX idx_job_runs_status ON job_runs("
                "status ASC, started_at ASC, run_id ASC)"
            )
        return

    conn.execute("ALTER TABLE job_runs RENAME TO job_runs_legacy_v5")
    conn.execute(
        f"""
        CREATE TABLE job_runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT,
            token_id TEXT,
            workflow_name TEXT NOT NULL,
            status TEXT NOT NULL,
            cancellable INTEGER NOT NULL DEFAULT 0,
            result_summary TEXT,
            error_summary TEXT,
            timeout_seconds INTEGER NOT NULL DEFAULT {DEFAULT_JOB_TIMEOUT_SECONDS},
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT,
            inputs_json TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL,
            FOREIGN KEY (token_id) REFERENCES mcp_tokens(id) ON DELETE SET NULL
        )
        """
    )

    conn.execute(
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
            timeout_seconds,
            created_at,
            started_at,
            updated_at,
            finished_at,
            inputs_json
        )
        SELECT
            run_id,
            project_id,
            token_id,
            workflow_name,
            status,
            cancellable,
            result_summary,
            error_summary,
            timeout_seconds,
            started_at,
            CASE
                WHEN status IN ('queued', 'pending') THEN NULL
                ELSE started_at
            END,
            updated_at,
            finished_at,
            inputs_json
        FROM job_runs_legacy_v5
        """
    )
    conn.execute("DROP TABLE job_runs_legacy_v5")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_runs_created_at ON job_runs(created_at ASC, run_id ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_runs_started_at ON job_runs(started_at ASC, run_id ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_runs_status ON job_runs("
        "status ASC, started_at ASC, run_id ASC)"
    )


def _ensure_job_runs_execution_state_column(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "job_runs")
    if "execution_state_json" in columns:
        return
    conn.execute("ALTER TABLE job_runs ADD COLUMN execution_state_json TEXT")


def _watcher_queue_needs_rebuild(columns: list[sqlite3.Row | tuple[object, ...]]) -> bool:
    by_name: dict[str, sqlite3.Row | tuple[object, ...]] = {
        str(column[1]): column for column in columns
    }
    required = {
        "id",
        "project_id",
        "path",
        "event_type",
        "reason",
        "enqueued_at",
        "updated_at",
        "processed_at",
    }
    if not required.issubset(set(by_name.keys())):
        return True

    updated_at = by_name["updated_at"]
    updated_not_null_raw = updated_at[3]
    if not isinstance(updated_not_null_raw, int):
        raise RuntimeError("invalid watcher_queue.updated_at notnull metadata type")
    updated_not_null = updated_not_null_raw
    updated_default = str(updated_at[4]) if updated_at[4] is not None else ""
    return updated_not_null != 1 or updated_default.upper() != "CURRENT_TIMESTAMP"


def _rebuild_watcher_queue_to_v2_shape(
    conn: sqlite3.Connection,
    columns: list[sqlite3.Row | tuple[object, ...]],
) -> None:
    column_names = {str(column[1]) for column in columns}

    path_expr = "path" if "path" in column_names else "''"
    event_type_expr = "event_type" if "event_type" in column_names else "'unknown'"
    updated_at_expr = (
        "COALESCE(updated_at, enqueued_at, CURRENT_TIMESTAMP)"
        if "updated_at" in column_names
        else "COALESCE(enqueued_at, CURRENT_TIMESTAMP)"
    )
    processed_at_expr = "processed_at" if "processed_at" in column_names else "NULL"

    conn.execute("ALTER TABLE watcher_queue RENAME TO watcher_queue_legacy_v1")
    conn.execute(
        """
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
        )
        """
    )
    conn.execute(
        f"""
        INSERT INTO watcher_queue (
            id,
            project_id,
            path,
            event_type,
            reason,
            enqueued_at,
            updated_at,
            processed_at
        )
        SELECT
            id,
            project_id,
            {path_expr},
            {event_type_expr},
            reason,
            enqueued_at,
            {updated_at_expr},
            {processed_at_expr}
        FROM watcher_queue_legacy_v1
        """
    )
    conn.execute("DROP TABLE watcher_queue_legacy_v1")
