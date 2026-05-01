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


@dataclass(frozen=True)
class SecretMetadata:
    name: str
    key_id: str | None
    created_at: str
    updated_at: str


class SQLiteSecretsRepository:
    def __init__(self, conn: Connection, key_path: Path) -> None:
        self._conn = conn
        self._key_path = key_path

    def upsert_secret(self, name: str, value: str, key_id: str | None = None) -> SecretMetadata:
        key = load_secret_key(self._key_path)
        encrypted_payload = encrypt_secret_value(value, key)

        self._conn.execute(
            """
            INSERT INTO encrypted_secret_metadata (id, secret_name, key_id, encrypted_payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(secret_name) DO UPDATE SET
                key_id = excluded.key_id,
                encrypted_payload = excluded.encrypted_payload,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(uuid.uuid4()), name, key_id, encrypted_payload),
        )
        row = self._conn.execute(
            """
            SELECT secret_name, key_id, created_at, updated_at
            FROM encrypted_secret_metadata
            WHERE secret_name = ?
            """,
            (name,),
        ).fetchone()
        self._conn.commit()
        assert row is not None
        return SecretMetadata(
            name=str(row[0]),
            key_id=str(row[1]) if row[1] is not None else None,
            created_at=str(row[2]),
            updated_at=str(row[3]),
        )

    def delete_secret(self, name: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM encrypted_secret_metadata WHERE secret_name = ?", (name,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_metadata(self) -> list[SecretMetadata]:
        rows = self._conn.execute(
            """
            SELECT secret_name, key_id, created_at, updated_at
            FROM encrypted_secret_metadata
            ORDER BY secret_name ASC
            """
        ).fetchall()
        return [
            SecretMetadata(
                name=str(row[0]),
                key_id=str(row[1]) if row[1] is not None else None,
                created_at=str(row[2]),
                updated_at=str(row[3]),
            )
            for row in rows
        ]

    def get_secret_value(self, name: str) -> str | None:
        row = self._conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None

        key = load_secret_key(self._key_path)
        return decrypt_secret_value(str(row[0]), key)
