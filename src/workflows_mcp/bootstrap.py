from __future__ import annotations

import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast, overload

from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.security.passwords import hash_password

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_PORT = 1
MAX_PORT = 65535


def validate_port(port: int) -> int:
    """Validate TCP port range and return normalized int."""
    if port < MIN_PORT or port > MAX_PORT:
        msg = f"port must be between {MIN_PORT} and {MAX_PORT} inclusive"
        raise ValueError(msg)
    return port


@dataclass(frozen=True)
class BootstrapResult:
    created: bool
    reconfigured: bool
    db_path: Path
    secrets_key_path: Path


@dataclass(frozen=True)
class BootstrapDefaults:
    host: str
    port: int


@overload
def _read_json_setting(conn: sqlite3.Connection, key: str, fallback: str) -> str: ...


@overload
def _read_json_setting(conn: sqlite3.Connection, key: str, fallback: int) -> int: ...


def _read_json_setting(conn: sqlite3.Connection, key: str, fallback: str | int) -> str | int:
    row = conn.execute(
        "SELECT value FROM server_settings WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return fallback

    return cast("str | int", json.loads(str(row[0])))


def _ensure_secrets_key(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    key_data = secrets.token_bytes(32)

    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return

    try:
        with os.fdopen(fd, "wb") as file_obj:
            file_obj.write(key_data)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def read_bootstrap_defaults(config_dir: Path) -> BootstrapDefaults:
    """Read persisted host/port settings with factory fallbacks."""
    db_path = config_dir / "server.db"
    if not db_path.exists():
        return BootstrapDefaults(host=DEFAULT_HOST, port=DEFAULT_PORT)

    conn = connect_metadata_db(db_path)
    try:
        host = _read_json_setting(conn, "host", DEFAULT_HOST)
        port = _read_json_setting(conn, "port", DEFAULT_PORT)
        resolved_port = int(port)
        if resolved_port < MIN_PORT or resolved_port > MAX_PORT:
            resolved_port = DEFAULT_PORT
        return BootstrapDefaults(host=str(host), port=resolved_port)
    finally:
        conn.close()


def bootstrap_if_needed(
    config_dir: Path,
    host: str | None,
    port: int | None,
    admin_password: str | None,
    *,
    reconfigure: bool = False,
) -> BootstrapResult:
    if admin_password is not None and not admin_password.strip():
        msg = "admin_password must not be empty or whitespace"
        raise ValueError(msg)

    if port is not None:
        validate_port(port)

    db_path = config_dir / "server.db"
    secrets_key_path = config_dir / "secrets.key"
    existed_before = db_path.exists()

    if existed_before and not reconfigure:
        return BootstrapResult(
            created=False,
            reconfigured=False,
            db_path=db_path,
            secrets_key_path=secrets_key_path,
        )

    _ensure_secrets_key(secrets_key_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)

        existing_host = _read_json_setting(conn, "host", DEFAULT_HOST)
        existing_port = _read_json_setting(conn, "port", DEFAULT_PORT)

        resolved_host = host if host is not None else existing_host
        resolved_port = port if port is not None else existing_port
        resolved_port = validate_port(int(resolved_port))

        if admin_password:
            conn.execute(
                "INSERT OR REPLACE INTO admin_credentials (id, password_hash) VALUES (1, ?)",
                (hash_password(admin_password),),
            )

        conn.execute(
            "INSERT OR REPLACE INTO server_settings (key, value) VALUES (?, ?)",
            ("host", json.dumps(resolved_host)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO server_settings (key, value) VALUES (?, ?)",
            ("port", json.dumps(resolved_port)),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return BootstrapResult(
        created=not existed_before,
        reconfigured=existed_before,
        db_path=db_path,
        secrets_key_path=secrets_key_path,
    )
