from __future__ import annotations

import sqlite3
from pathlib import Path

CURRENT_SCHEMA_VERSION = 10
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

    project_defaults_rebuild_may_be_needed = (
        _project_default_locations_rebuild_may_be_needed(conn)
    )
    v9_to_v10_rebuild_may_be_needed = _v9_to_v10_rebuild_may_be_needed(conn)
    foreign_keys_row = conn.execute("PRAGMA foreign_keys").fetchone()
    foreign_keys_were_enabled = int(foreign_keys_row[0]) if foreign_keys_row else 0
    foreign_keys_disabled = False
    needs_rebuild = project_defaults_rebuild_may_be_needed or v9_to_v10_rebuild_may_be_needed
    if needs_rebuild and foreign_keys_were_enabled:
        conn.execute("PRAGMA foreign_keys = OFF")
        foreign_keys_disabled = True

    transaction_started = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        transaction_started = True

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
        if max_version < 7:
            _migrate_v6_to_v7(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (7)"
            )
        if max_version < 8:
            _migrate_v7_to_v8(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (8)"
            )
        if max_version < 9:
            _migrate_v8_to_v9(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (9)"
            )
        if max_version < 10:
            _migrate_v9_to_v10(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (10)"
            )

        # Phase 10+ internal-only resumable state for paused runs.
        # Keep schema version stable while ensuring additive column exists.
        _ensure_job_runs_execution_state_column(conn)
        _ensure_job_runs_execution_columns(conn)
        _ensure_project_default_locations_nullable(conn)
        _ensure_project_extraction_settings(conn)
        conn.commit()
    except Exception:
        if transaction_started:
            conn.rollback()
        raise
    finally:
        if foreign_keys_disabled:
            conn.execute("PRAGMA foreign_keys = ON")


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
    ws_cols = _table_columns(conn, "workflow_sources")
    if "project_id" not in ws_cols:
        # Fresh v10 schema already has global UNIQUE on source_path; skip legacy index creation.
        return
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


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    _ensure_postgresql_settings_structured_columns(conn)


def _migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    _ensure_job_runs_execution_columns(conn)


def _migrate_v8_to_v9(conn: sqlite3.Connection) -> None:
    _ensure_workflow_sources_is_system_column(conn)


def _migrate_v9_to_v10(conn: sqlite3.Connection) -> None:
    """Decouple workflow_sources from projects.

    Steps:
    1. Fail closed if duplicate non-system source_path values exist (cannot safely pick one).
    2. Delete legacy is_system=1 source rows (and their reload state via CASCADE).
    3. Delete the exact legacy System project row (slug='system', palace='__system__').
    4. Rebuild workflow_sources without project_id/is_system columns; add global UNIQUE on
       source_path.
    5. Drop the old per-project unique index.
    """
    # Check whether workflow_sources has the old columns at all.
    ws_cols = _table_columns(conn, "workflow_sources")
    has_project_id = "project_id" in ws_cols
    has_is_system = "is_system" in ws_cols

    if not has_project_id and not has_is_system:
        # Already migrated (fresh schema or idempotent re-run).
        return

    # Step 1: Fail closed on duplicate non-system source paths.
    if has_is_system:
        dup_check = conn.execute(
            """
            SELECT source_path, COUNT(*) AS cnt
            FROM workflow_sources
            WHERE is_system = 0
            GROUP BY source_path
            HAVING cnt > 1
            """
        ).fetchall()
    else:
        dup_check = conn.execute(
            """
            SELECT source_path, COUNT(*) AS cnt
            FROM workflow_sources
            GROUP BY source_path
            HAVING cnt > 1
            """
        ).fetchall()

    if dup_check:
        dup_paths = [str(row[0]) for row in dup_check]
        raise RuntimeError(
            "Cannot migrate workflow_sources to v10: duplicate source_path values found "
            "among non-system sources. Resolve duplicates manually before upgrading. "
            f"Duplicate source_path(s): {', '.join(dup_paths)}"
        )

    # Step 2: Delete legacy is_system=1 source rows.
    # workflow_reload_state rows cascade-delete via FK.
    if has_is_system:
        conn.execute("DELETE FROM workflow_sources WHERE is_system = 1")

    # Step 3: Delete exact legacy System project row.
    if has_project_id:
        conn.execute(
            "DELETE FROM projects WHERE slug = 'system' AND palace = '__system__'"
        )

    # Step 4: Rebuild workflow_sources without project_id and is_system.
    # SQLite does not support DROP COLUMN for old schema; use rename-create-copy-drop pattern.
    # FK enforcement is disabled at the top of migrate_metadata_db before the transaction starts,
    # so dropping workflow_sources_legacy_v9 will not cascade-delete workflow_reload_state rows.
    #
    # IMPORTANT: Use PRAGMA legacy_alter_table = ON so that SQLite does NOT update FK references
    # in other tables (e.g. workflow_reload_state) when we rename workflow_sources. Without this,
    # SQLite would rewrite the FK in workflow_reload_state to reference workflow_sources_legacy_v9,
    # which is then dropped, leaving a dangling FK reference.
    legacy_alter = conn.execute("PRAGMA legacy_alter_table").fetchone()
    legacy_alter_was = int(legacy_alter[0]) if legacy_alter else 0
    conn.execute("PRAGMA legacy_alter_table = ON")
    try:
        conn.execute("ALTER TABLE workflow_sources RENAME TO workflow_sources_legacy_v9")
    finally:
        conn.execute(f"PRAGMA legacy_alter_table = {legacy_alter_was}")
    conn.execute(
        """
        CREATE TABLE workflow_sources (
            source_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL UNIQUE,
            checksum TEXT,
            discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO workflow_sources (source_id, source_path, checksum, discovered_at)
        SELECT source_id, source_path, checksum, discovered_at
        FROM workflow_sources_legacy_v9
        """
    )
    conn.execute("DROP TABLE workflow_sources_legacy_v9")


def _ensure_workflow_sources_is_system_column(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "workflow_sources")
    if "is_system" not in columns:
        conn.execute(
            "ALTER TABLE workflow_sources ADD COLUMN is_system INTEGER NOT NULL DEFAULT 0"
        )


def _ensure_postgresql_settings_structured_columns(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "postgresql_settings")
    if "host" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN host TEXT NOT NULL DEFAULT '127.0.0.1'"
        )
    if "port" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN port INTEGER NOT NULL DEFAULT 5432"
        )
    if "database" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN database TEXT NOT NULL DEFAULT 'workflows'"
        )
    if "username" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN username TEXT NOT NULL DEFAULT 'workflows'"
        )
    if "ssl_mode" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN ssl_mode TEXT NOT NULL DEFAULT 'disable'"
        )
    if "extra_params" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN extra_params TEXT NOT NULL DEFAULT ''"
        )
    if "container_name" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN "
            "container_name TEXT NOT NULL DEFAULT 'workflows-postgres'"
        )
    if "container_image" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN "
            "container_image TEXT NOT NULL DEFAULT 'pgvector/pgvector:pg17'"
        )
    if "container_host_port" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN "
            "container_host_port INTEGER NOT NULL DEFAULT 5432"
        )
    if "volume_name" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN "
            "volume_name TEXT NOT NULL DEFAULT 'workflows-postgres-data'"
        )
    if "legacy_dsn_upgrade_status" not in columns:
        conn.execute(
            "ALTER TABLE postgresql_settings ADD COLUMN "
            "legacy_dsn_upgrade_status TEXT NOT NULL DEFAULT 'not_started'"
        )


def _ensure_job_runs_execution_state_column(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "job_runs")
    if "execution_state_json" in columns:
        return
    conn.execute("ALTER TABLE job_runs ADD COLUMN execution_state_json TEXT")


def _ensure_job_runs_execution_columns(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "job_runs")
    if "execution_mode" not in columns:
        conn.execute(
            "ALTER TABLE job_runs ADD COLUMN execution_mode TEXT NOT NULL DEFAULT 'async'"
        )
    if "execution_json" not in columns:
        conn.execute("ALTER TABLE job_runs ADD COLUMN execution_json TEXT")
    if not _index_exists(conn, "idx_job_runs_workflow"):
        conn.execute(
            "CREATE INDEX idx_job_runs_workflow "
            "ON job_runs(workflow_name ASC, created_at DESC, run_id ASC)"
        )
    if not _index_exists(conn, "idx_job_runs_mode"):
        conn.execute(
            "CREATE INDEX idx_job_runs_mode "
            "ON job_runs(execution_mode ASC, created_at DESC, run_id ASC)"
        )


def _ensure_project_extraction_settings(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "projects")
    if "system2_enabled" not in columns:
        conn.execute(
            "ALTER TABLE projects ADD COLUMN system2_enabled INTEGER NOT NULL DEFAULT 0"
        )


def _project_default_locations_rebuild_may_be_needed(conn: sqlite3.Connection) -> bool:
    projects_table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'projects'"
    ).fetchone()
    if projects_table_exists is None:
        return True

    columns = conn.execute("PRAGMA table_info('projects')").fetchall()
    return _projects_default_locations_need_rebuild(columns)


def _v9_to_v10_rebuild_may_be_needed(conn: sqlite3.Connection) -> bool:
    """Return True if workflow_sources still has project_id or is_system (v9 or earlier shape)."""
    ws_table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'workflow_sources'"
    ).fetchone()
    if ws_table_exists is None:
        return False
    cols = _table_columns(conn, "workflow_sources")
    return "project_id" in cols or "is_system" in cols


def _ensure_project_default_locations_nullable(conn: sqlite3.Connection) -> None:
    columns = conn.execute("PRAGMA table_info('projects')").fetchall()
    if not _projects_default_locations_need_rebuild(columns):
        return

    legacy_alter_table_row = conn.execute("PRAGMA legacy_alter_table").fetchone()
    legacy_alter_table = (
        int(legacy_alter_table_row[0]) if legacy_alter_table_row else 0
    )
    conn.execute("PRAGMA legacy_alter_table = ON")
    try:
        _rebuild_projects_with_nullable_default_locations(conn)
    finally:
        conn.execute(f"PRAGMA legacy_alter_table = {legacy_alter_table}")


def _projects_default_locations_need_rebuild(
    columns: list[sqlite3.Row | tuple[object, ...]],
) -> bool:
    by_name: dict[str, sqlite3.Row | tuple[object, ...]] = {
        str(column[1]): column for column in columns
    }
    required = {
        "id",
        "name",
        "slug",
        "palace",
        "default_wing",
        "default_room",
        "fs_root",
        "fs_allowlist_json",
        "created_at",
        "updated_at",
    }
    if not required.issubset(set(by_name.keys())):
        return False

    default_wing_not_null_raw = by_name["default_wing"][3]
    default_room_not_null_raw = by_name["default_room"][3]
    if not isinstance(default_wing_not_null_raw, int) or not isinstance(
        default_room_not_null_raw, int
    ):
        raise RuntimeError("invalid projects default location notnull metadata type")

    return default_wing_not_null_raw != 0 or default_room_not_null_raw != 0


def _rebuild_projects_with_nullable_default_locations(conn: sqlite3.Connection) -> None:
    legacy_columns = _table_columns(conn, "projects")
    system2_enabled_expr = "system2_enabled" if "system2_enabled" in legacy_columns else "0"
    conn.execute("ALTER TABLE projects RENAME TO projects_legacy_default_locations")
    conn.execute(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            palace TEXT NOT NULL UNIQUE,
            default_wing TEXT,
            default_room TEXT,
            fs_root TEXT NOT NULL,
            fs_allowlist_json TEXT,
            system2_enabled INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        f"""
        INSERT INTO projects (
            id,
            name,
            slug,
            palace,
            default_wing,
            default_room,
            fs_root,
            fs_allowlist_json,
            system2_enabled,
            created_at,
            updated_at
        )
        SELECT
            id,
            name,
            slug,
            palace,
            default_wing,
            default_room,
            fs_root,
            fs_allowlist_json,
            {system2_enabled_expr},
            created_at,
            updated_at
        FROM projects_legacy_default_locations
        """
    )
    conn.execute("DROP TABLE projects_legacy_default_locations")


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
