from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection

from workflows_mcp.security.crypto import (
    decrypt_secret_value,
    encrypt_secret_value,
    load_secret_key,
)

_POSTGRES_DSN_SECRET_NAME = "postgresql.dsn"


@dataclass(frozen=True)
class PostgresSettingsMetadata:
    enabled: bool
    configured: bool
    updated_at: str


class SQLitePostgresSettingsRepository:
    def __init__(self, conn: Connection, key_path: Path) -> None:
        self._conn = conn
        self._key_path = key_path

    def load_settings(self) -> PostgresSettingsMetadata:
        row = self._conn.execute(
            """
            SELECT enabled, dsn_ref, updated_at
            FROM postgresql_settings
            WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return PostgresSettingsMetadata(enabled=False, configured=False, updated_at="")
        dsn_ref = str(row[1]) if row[1] is not None else ""
        return PostgresSettingsMetadata(
            enabled=bool(int(row[0])),
            configured=bool(dsn_ref),
            updated_at=str(row[2]),
        )

    def save_settings(self, *, enabled: bool, dsn: str | None = None) -> PostgresSettingsMetadata:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            current = self._conn.execute(
                "SELECT dsn_ref FROM postgresql_settings WHERE id = 1"
            ).fetchone()
            dsn_ref = str(current[0]) if current and current[0] is not None else None

            if dsn is not None:
                key = load_secret_key(self._key_path)
                encrypted_payload = encrypt_secret_value(dsn, key)
                self._conn.execute(
                    """
                    INSERT INTO encrypted_secret_metadata (
                        id,
                        secret_name,
                        key_id,
                        encrypted_payload
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(secret_name) DO UPDATE SET
                        encrypted_payload = excluded.encrypted_payload,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (str(uuid.uuid4()), _POSTGRES_DSN_SECRET_NAME, None, encrypted_payload),
                )
                dsn_ref = _POSTGRES_DSN_SECRET_NAME

            self._conn.execute(
                """
                INSERT INTO postgresql_settings (id, dsn_ref, enabled)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    dsn_ref = excluded.dsn_ref,
                    enabled = excluded.enabled,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (dsn_ref, 1 if enabled else 0),
            )

            row = self._conn.execute(
                "SELECT enabled, dsn_ref, updated_at FROM postgresql_settings WHERE id = 1"
            ).fetchone()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        assert row is not None
        configured = row[1] is not None and str(row[1]) != ""
        return PostgresSettingsMetadata(
            enabled=bool(int(row[0])),
            configured=configured,
            updated_at=str(row[2]),
        )

    def load_dsn(self) -> str | None:
        settings = self._conn.execute(
            "SELECT dsn_ref FROM postgresql_settings WHERE id = 1"
        ).fetchone()
        if settings is None or settings[0] is None:
            return None

        dsn_ref = str(settings[0])
        encrypted = self._conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            (dsn_ref,),
        ).fetchone()
        if encrypted is None:
            return None

        key = load_secret_key(self._key_path)
        return decrypt_secret_value(str(encrypted[0]), key)
