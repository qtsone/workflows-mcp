from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection
from urllib.parse import parse_qsl, quote, urlencode, urlparse

from workflows_mcp.security.crypto import (
    SecretCryptoError,
    decrypt_secret_value,
    encrypt_secret_value,
    load_secret_key,
)

_POSTGRES_DSN_SECRET_NAME = "postgresql.dsn"
_POSTGRES_PASSWORD_SECRET_NAME = "postgresql.password"
_UPGRADE_NOT_STARTED = "not_started"
_UPGRADE_COMPLETED = "completed"
_UPGRADE_FAILED = "failed"

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5432
_DEFAULT_DATABASE = "workflows"
_DEFAULT_USERNAME = "workflows"
_DEFAULT_SSL_MODE = "disable"
_DEFAULT_EXTRA_PARAMS = ""
_DEFAULT_CONTAINER_NAME = "workflows-postgres"
_DEFAULT_CONTAINER_IMAGE = "pgvector/pgvector:pg17"
_DEFAULT_CONTAINER_HOST_PORT = 5432
_DEFAULT_VOLUME_NAME = "workflows-postgres-data"


@dataclass(frozen=True)
class PostgresProfileInput:
    enabled: bool
    host: str
    port: int
    database: str
    username: str
    password: str | None
    password_was_provided: bool
    password_clear: bool
    ssl_mode: str
    extra_params: str
    container_name: str
    container_image: str
    container_host_port: int
    volume_name: str
    dsn_import: str | None = None


@dataclass(frozen=True)
class PostgresSettingsMetadata:
    enabled: bool
    configured: bool
    password_configured: bool
    legacy_profile_reentry_required: bool
    host: str
    port: int
    database: str
    username: str
    ssl_mode: str
    extra_params: str
    container_name: str
    container_image: str
    container_host_port: int
    volume_name: str
    updated_at: str


class SQLitePostgresSettingsRepository:
    def __init__(self, conn: Connection, key_path: Path) -> None:
        self._conn = conn
        self._key_path = key_path

    def load_settings(self) -> PostgresSettingsMetadata:
        self._maybe_upgrade_legacy_dsn()
        row = self._conn.execute(
            """
            SELECT
                enabled,
                dsn_ref,
                host,
                port,
                database,
                username,
                ssl_mode,
                extra_params,
                container_name,
                container_image,
                container_host_port,
                volume_name,
                legacy_dsn_upgrade_status,
                updated_at
            FROM postgresql_settings
            WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return PostgresSettingsMetadata(
                enabled=False,
                configured=False,
                password_configured=False,
                legacy_profile_reentry_required=False,
                host=_DEFAULT_HOST,
                port=_DEFAULT_PORT,
                database=_DEFAULT_DATABASE,
                username=_DEFAULT_USERNAME,
                ssl_mode=_DEFAULT_SSL_MODE,
                extra_params=_DEFAULT_EXTRA_PARAMS,
                container_name=_DEFAULT_CONTAINER_NAME,
                container_image=_DEFAULT_CONTAINER_IMAGE,
                container_host_port=_DEFAULT_CONTAINER_HOST_PORT,
                volume_name=_DEFAULT_VOLUME_NAME,
                updated_at="",
            )

        password_configured = self._secret_exists(_POSTGRES_PASSWORD_SECRET_NAME)
        has_structured_required = self._has_structured_required_values(
            host=str(row[2]),
            port=int(str(row[3])),
            database=str(row[4]),
            username=str(row[5]),
        )
        configured = has_structured_required and password_configured
        dsn_ref = str(row[1]) if row[1] is not None else ""
        legacy_upgrade_status = str(row[12]) if row[12] is not None else _UPGRADE_NOT_STARTED
        legacy_reentry = (
            legacy_upgrade_status == _UPGRADE_FAILED and dsn_ref == _POSTGRES_DSN_SECRET_NAME
        )
        return PostgresSettingsMetadata(
            enabled=bool(int(row[0])),
            configured=configured,
            password_configured=password_configured,
            legacy_profile_reentry_required=legacy_reentry,
            host=str(row[2]),
            port=int(str(row[3])),
            database=str(row[4]),
            username=str(row[5]),
            ssl_mode=str(row[6]),
            extra_params=str(row[7]),
            container_name=str(row[8]),
            container_image=str(row[9]),
            container_host_port=int(row[10]),
            volume_name=str(row[11]),
            updated_at=str(row[13]),
        )

    def save_settings(self, profile: PostgresProfileInput) -> PostgresSettingsMetadata:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            has_existing_password = self._secret_exists(_POSTGRES_PASSWORD_SECRET_NAME)
            password_configured = self._resolve_password_update(
                profile=profile,
                has_existing_password=has_existing_password,
            )
            if profile.enabled and not password_configured:
                raise ValueError("enabled PostgreSQL settings require a configured password")

            self._conn.execute(
                """
                INSERT INTO postgresql_settings (
                    id,
                    dsn_ref,
                    enabled,
                    host,
                    port,
                    database,
                    username,
                    ssl_mode,
                    extra_params,
                    container_name,
                    container_image,
                    container_host_port,
                    volume_name,
                    legacy_dsn_upgrade_status
                )
                VALUES (1, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    dsn_ref = excluded.dsn_ref,
                    enabled = excluded.enabled,
                    host = excluded.host,
                    port = excluded.port,
                    database = excluded.database,
                    username = excluded.username,
                    ssl_mode = excluded.ssl_mode,
                    extra_params = excluded.extra_params,
                    container_name = excluded.container_name,
                    container_image = excluded.container_image,
                    container_host_port = excluded.container_host_port,
                    volume_name = excluded.volume_name,
                    legacy_dsn_upgrade_status = excluded.legacy_dsn_upgrade_status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    1 if profile.enabled else 0,
                    profile.host,
                    profile.port,
                    profile.database,
                    profile.username,
                    profile.ssl_mode,
                    profile.extra_params,
                    profile.container_name,
                    profile.container_image,
                    profile.container_host_port,
                    profile.volume_name,
                    _UPGRADE_COMPLETED,
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return self.load_settings()

    def load_dsn(self) -> str | None:
        self._maybe_upgrade_legacy_dsn()
        settings = self._conn.execute(
            """
            SELECT enabled, host, port, database, username, ssl_mode, extra_params
            FROM postgresql_settings
            WHERE id = 1
            """
        ).fetchone()
        if settings is None:
            return None
        enabled = bool(int(settings[0]))
        host = str(settings[1])
        port = int(settings[2])
        database = str(settings[3])
        username = str(settings[4])
        ssl_mode = str(settings[5])
        extra_params = str(settings[6])
        if not enabled or not self._has_structured_required_values(
            host=host,
            port=port,
            database=database,
            username=username,
        ):
            return None
        password = self._load_secret(_POSTGRES_PASSWORD_SECRET_NAME)
        if password is None:
            return None
        query_pairs = parse_qsl(extra_params, keep_blank_values=True)
        query_map: dict[str, str] = {k: v for k, v in query_pairs if k != "sslmode"}
        query_map["sslmode"] = ssl_mode
        return (
            "postgresql://"
            f"{quote(username, safe='')}:{quote(password, safe='')}@"
            f"{host}:{port}/{quote(database, safe='')}?{urlencode(query_map)}"
        )

    def _resolve_password_update(
        self,
        *,
        profile: PostgresProfileInput,
        has_existing_password: bool,
    ) -> bool:
        if profile.password_clear:
            self._delete_secret(_POSTGRES_PASSWORD_SECRET_NAME)
            return False
        if not profile.password_was_provided or profile.password is None:
            return has_existing_password
        if profile.password == "":
            raise ValueError("password_clear must be true when password is empty")
        self._upsert_secret(_POSTGRES_PASSWORD_SECRET_NAME, profile.password)
        return True

    def _maybe_upgrade_legacy_dsn(self) -> None:
        row = self._conn.execute(
            """
            SELECT
                dsn_ref,
                legacy_dsn_upgrade_status,
                host,
                port,
                database,
                username,
                ssl_mode,
                extra_params,
                enabled
            FROM postgresql_settings
            WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return
        dsn_ref = str(row[0]) if row[0] is not None else ""
        status = str(row[1]) if row[1] is not None else _UPGRADE_NOT_STARTED
        if dsn_ref != _POSTGRES_DSN_SECRET_NAME or status != _UPGRADE_NOT_STARTED:
            return
        if not self._is_safe_for_legacy_upgrade(
            host=str(row[2]),
            port=int(row[3]),
            database=str(row[4]),
            username=str(row[5]),
            ssl_mode=str(row[6]),
            extra_params=str(row[7]),
            enabled=bool(int(row[8])),
        ):
            return

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            legacy_dsn = self._load_secret(_POSTGRES_DSN_SECRET_NAME)
            if legacy_dsn is None:
                self._set_legacy_upgrade_status(_UPGRADE_FAILED)
                self._conn.commit()
                return
            parsed = self._parse_legacy_dsn(legacy_dsn)
            if parsed is None:
                self._set_legacy_upgrade_status(_UPGRADE_FAILED)
                self._conn.commit()
                return

            self._upsert_secret(_POSTGRES_PASSWORD_SECRET_NAME, str(parsed["password"]))
            self._conn.execute(
                """
                UPDATE postgresql_settings
                SET
                    host = ?,
                    port = ?,
                    database = ?,
                    username = ?,
                    ssl_mode = ?,
                    extra_params = ?,
                    dsn_ref = NULL,
                    legacy_dsn_upgrade_status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                (
                    str(parsed["host"]),
                    int(parsed["port"]),
                    str(parsed["database"]),
                    str(parsed["username"]),
                    str(parsed["ssl_mode"]),
                    str(parsed["extra_params"]),
                    _UPGRADE_COMPLETED,
                ),
            )
            self._delete_secret(_POSTGRES_DSN_SECRET_NAME)
            self._conn.commit()
        except (SecretCryptoError, ValueError):
            self._conn.rollback()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._set_legacy_upgrade_status(_UPGRADE_FAILED)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        except Exception:
            self._conn.rollback()
            raise

    def _parse_legacy_dsn(self, dsn: str) -> Mapping[str, str | int] | None:
        parsed = urlparse(dsn)
        if parsed.scheme not in {"postgres", "postgresql"}:
            return None
        if parsed.hostname is None or parsed.port is None:
            return None
        if parsed.username is None or parsed.password is None:
            return None
        database = parsed.path.lstrip("/")
        if not database:
            return None
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        ssl_mode = "disable"
        extras: list[tuple[str, str]] = []
        for key, value in query_pairs:
            if key == "sslmode":
                ssl_mode = value
            else:
                extras.append((key, value))
        return {
            "host": parsed.hostname,
            "port": parsed.port,
            "database": database,
            "username": parsed.username,
            "password": parsed.password,
            "ssl_mode": ssl_mode,
            "extra_params": urlencode(extras),
        }

    def _is_safe_for_legacy_upgrade(
        self,
        *,
        host: str,
        port: int,
        database: str,
        username: str,
        ssl_mode: str,
        extra_params: str,
        enabled: bool,
    ) -> bool:
        return (
            host == _DEFAULT_HOST
            and port == _DEFAULT_PORT
            and database == _DEFAULT_DATABASE
            and username == _DEFAULT_USERNAME
            and ssl_mode == _DEFAULT_SSL_MODE
            and extra_params == _DEFAULT_EXTRA_PARAMS
            and enabled
            and not self._secret_exists(_POSTGRES_PASSWORD_SECRET_NAME)
        )

    def _set_legacy_upgrade_status(self, status: str) -> None:
        self._conn.execute(
            """
            UPDATE postgresql_settings
            SET legacy_dsn_upgrade_status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (status,),
        )

    def _has_structured_required_values(
        self,
        *,
        host: str,
        port: int,
        database: str,
        username: str,
    ) -> bool:
        return bool(host.strip()) and port > 0 and bool(database.strip()) and bool(username.strip())

    def _upsert_secret(self, secret_name: str, plaintext: str) -> None:
        encrypted_payload = encrypt_secret_value(plaintext, load_secret_key(self._key_path))
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
            (str(uuid.uuid4()), secret_name, None, encrypted_payload),
        )

    def _delete_secret(self, secret_name: str) -> None:
        self._conn.execute(
            "DELETE FROM encrypted_secret_metadata WHERE secret_name = ?",
            (secret_name,),
        )

    def _secret_exists(self, secret_name: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM encrypted_secret_metadata WHERE secret_name = ?",
                (secret_name,),
            ).fetchone()
            is not None
        )

    def _load_secret(self, secret_name: str) -> str | None:
        encrypted = self._conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            (secret_name,),
        ).fetchone()
        if encrypted is None:
            return None
        return decrypt_secret_value(str(encrypted[0]), load_secret_key(self._key_path))
