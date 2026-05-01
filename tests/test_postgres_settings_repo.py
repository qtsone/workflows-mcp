from __future__ import annotations

import os
from pathlib import Path

import pytest

from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.postgres_repo import (
    PostgresProfileInput,
    SQLitePostgresSettingsRepository,
)
from workflows_mcp.security.crypto import (
    decrypt_secret_value,
    encrypt_secret_value,
    load_secret_key,
)


def _repo(tmp_path: Path) -> tuple[SQLitePostgresSettingsRepository, Path]:
    base_dir = tmp_path / ".workflows"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "server.db"
    key_path = base_dir / "secrets.key"
    key_path.write_bytes(os.urandom(32))
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    return SQLitePostgresSettingsRepository(conn=conn, key_path=key_path), key_path


def test_save_and_load_structured_settings_encrypts_password_without_returning_plaintext(
    tmp_path: Path,
) -> None:
    repo, key_path = _repo(tmp_path)
    metadata = repo.save_settings(
        PostgresProfileInput(
            enabled=True,
            host="db.internal",
            port=5433,
            database="wfdb",
            username="wf_user",
            password="super-secret",
            password_was_provided=True,
            password_clear=False,
            ssl_mode="require",
            extra_params="application_name=wf",
            container_name="wf-postgres",
            container_image="pgvector/pgvector:pg17",
            container_host_port=5544,
            volume_name="wf-pg-data",
        )
    )
    assert metadata.enabled is True
    assert metadata.configured is True
    assert metadata.password_configured is True
    assert metadata.legacy_profile_reentry_required is False
    assert "super-secret" not in repr(metadata).lower()

    loaded = repo.load_settings()
    assert loaded.host == "db.internal"
    assert loaded.port == 5433
    assert loaded.database == "wfdb"
    assert loaded.username == "wf_user"
    assert loaded.ssl_mode == "require"
    assert loaded.extra_params == "application_name=wf"

    secret_row = repo._conn.execute(
        "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
        ("postgresql.password",),
    ).fetchone()
    assert secret_row is not None
    ciphertext = str(secret_row[0])
    assert "super-secret" not in ciphertext
    assert decrypt_secret_value(ciphertext, load_secret_key(key_path)) == "super-secret"


def test_password_omitted_or_null_keeps_existing_secret_and_password_clear_removes_it(
    tmp_path: Path,
) -> None:
    repo, key_path = _repo(tmp_path)
    repo.save_settings(
        PostgresProfileInput(
            enabled=True,
            host="127.0.0.1",
            port=5432,
            database="workflows",
            username="workflows",
            password="first-secret",
            password_was_provided=True,
            password_clear=False,
            ssl_mode="disable",
            extra_params="",
            container_name="workflows-postgres",
            container_image="pgvector/pgvector:pg17",
            container_host_port=5432,
            volume_name="workflows-postgres-data",
        )
    )
    original_cipher = str(
        repo._conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            ("postgresql.password",),
        ).fetchone()[0]
    )

    repo.save_settings(
        PostgresProfileInput(
            enabled=True,
            host="127.0.0.1",
            port=5432,
            database="workflows",
            username="workflows",
            password=None,
            password_was_provided=False,
            password_clear=False,
            ssl_mode="disable",
            extra_params="",
            container_name="workflows-postgres",
            container_image="pgvector/pgvector:pg17",
            container_host_port=5432,
            volume_name="workflows-postgres-data",
        )
    )
    after_omit_cipher = str(
        repo._conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            ("postgresql.password",),
        ).fetchone()[0]
    )
    assert after_omit_cipher == original_cipher

    repo.save_settings(
        PostgresProfileInput(
            enabled=False,
            host="127.0.0.1",
            port=5432,
            database="workflows",
            username="workflows",
            password="",
            password_was_provided=True,
            password_clear=True,
            ssl_mode="disable",
            extra_params="",
            container_name="workflows-postgres",
            container_image="pgvector/pgvector:pg17",
            container_host_port=5432,
            volume_name="workflows-postgres-data",
        )
    )
    cleared = repo._conn.execute(
        "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
        ("postgresql.password",),
    ).fetchone()
    assert cleared is None

    with pytest.raises(ValueError):
        repo.save_settings(
            PostgresProfileInput(
                enabled=True,
                host="127.0.0.1",
                port=5432,
                database="workflows",
                username="workflows",
                password="",
                password_was_provided=True,
                password_clear=False,
                ssl_mode="disable",
                extra_params="",
                container_name="workflows-postgres",
                container_image="pgvector/pgvector:pg17",
                container_host_port=5432,
                volume_name="workflows-postgres-data",
            )
        )

    _ = key_path


def test_load_dsn_builds_runtime_dsn_from_structured_fields_and_secret(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    repo.save_settings(
        PostgresProfileInput(
            enabled=True,
            host="127.0.0.1",
            port=5432,
            database="db/name",
            username="user@name",
            password="pw:word",
            password_was_provided=True,
            password_clear=False,
            ssl_mode="verify-full",
            extra_params="application_name=wf&connect_timeout=10",
            container_name="workflows-postgres",
            container_image="pgvector/pgvector:pg17",
            container_host_port=5432,
            volume_name="workflows-postgres-data",
        )
    )

    dsn = repo.load_dsn()
    assert dsn is not None
    assert dsn.startswith("postgresql://user%40name:pw%3Aword@127.0.0.1:5432/db%2Fname?")
    assert "sslmode=verify-full" in dsn
    assert "application_name=wf" in dsn
    assert "connect_timeout=10" in dsn


def test_lazy_legacy_dsn_upgrade_success_moves_fields_and_preserves_query_params(
    tmp_path: Path,
) -> None:
    repo, key_path = _repo(tmp_path)
    legacy = "postgresql://legacy_user:legacy_pw@db.example:6432/legacy_db?sslmode=require&application_name=wf&connect_timeout=7"
    repo._conn.execute(
        "INSERT INTO encrypted_secret_metadata"
        "(id, secret_name, key_id, encrypted_payload) VALUES (?, ?, ?, ?)",
        (
            "legacy-secret",
            "postgresql.dsn",
            None,
            encrypt_secret_value(legacy, load_secret_key(key_path)),
        ),
    )
    repo._conn.execute(
        "INSERT INTO postgresql_settings"
        "(id, dsn_ref, enabled, legacy_dsn_upgrade_status) "
        "VALUES (1, 'postgresql.dsn', 1, 'not_started')"
    )
    repo._conn.commit()

    settings = repo.load_settings()
    assert settings.legacy_profile_reentry_required is False
    assert settings.username == "legacy_user"
    assert settings.host == "db.example"
    assert settings.port == 6432
    assert settings.database == "legacy_db"
    assert settings.ssl_mode == "require"
    assert "application_name=wf" in settings.extra_params
    assert "connect_timeout=7" in settings.extra_params
    assert settings.password_configured is True

    row = repo._conn.execute(
        "SELECT dsn_ref, legacy_dsn_upgrade_status FROM postgresql_settings WHERE id = 1"
    ).fetchone()
    assert row is not None
    assert row[0] is None
    assert str(row[1]) == "completed"
    assert (
        repo._conn.execute(
            "SELECT 1 FROM encrypted_secret_metadata WHERE secret_name = 'postgresql.dsn'"
        ).fetchone()
        is None
    )


def test_lazy_legacy_dsn_upgrade_failure_preserves_legacy_secret_and_sets_reentry_required(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    repo._conn.execute(
        "INSERT INTO encrypted_secret_metadata"
        "(id, secret_name, key_id, encrypted_payload) VALUES (?, ?, ?, ?)",
        ("legacy-secret", "postgresql.dsn", None, "not-json"),
    )
    repo._conn.execute(
        "INSERT INTO postgresql_settings"
        "(id, dsn_ref, enabled, legacy_dsn_upgrade_status) "
        "VALUES (1, 'postgresql.dsn', 1, 'not_started')"
    )
    repo._conn.commit()

    settings = repo.load_settings()
    assert settings.legacy_profile_reentry_required is True

    row = repo._conn.execute(
        "SELECT dsn_ref, legacy_dsn_upgrade_status FROM postgresql_settings WHERE id = 1"
    ).fetchone()
    assert row is not None
    assert str(row[0]) == "postgresql.dsn"
    assert str(row[1]) == "failed"
    secret = repo._conn.execute(
        "SELECT encrypted_payload FROM encrypted_secret_metadata "
        "WHERE secret_name = 'postgresql.dsn'"
    ).fetchone()
    assert secret is not None

    assert repo.load_dsn() is None
