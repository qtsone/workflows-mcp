from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any


class TokenRepositoryError(RuntimeError):
    """Base class for deterministic token repository errors."""


class InvalidTokenProjectBindingError(TokenRepositoryError):
    """Raised when token project bindings are empty or reference unknown projects."""


class UnknownTokenError(TokenRepositoryError):
    """Raised when a presented token or id cannot be resolved."""


class RevokedTokenError(TokenRepositoryError):
    """Raised when attempting to use a revoked token."""


class DuplicateTokenLabelError(TokenRepositoryError):
    """Raised when creating a token with a duplicate label."""


class TokenIntegrityError(TokenRepositoryError):
    """Raised when token persistence invariants are violated."""


@dataclass(frozen=True)
class TokenRecord:
    id: str
    label: str
    capabilities: dict[str, Any]
    project_ids: list[str]
    created_at: str
    last_used_at: str | None
    revoked_at: str | None


@dataclass(frozen=True)
class CreatedToken:
    id: str
    label: str
    token_secret: str
    capabilities: dict[str, Any]
    project_ids: list[str]
    created_at: str
    last_used_at: str | None
    revoked_at: str | None


def _hash_token(token_secret: str) -> str:
    return hashlib.sha256(token_secret.encode("utf-8")).hexdigest()


def _encode_capabilities(capabilities: dict[str, Any] | None) -> str | None:
    if capabilities is None:
        return None
    return json.dumps(capabilities, sort_keys=True, separators=(",", ":"))


def _decode_capabilities(capabilities_json: str | None) -> dict[str, Any]:
    if capabilities_json is None or capabilities_json == "":
        return {}
    parsed = json.loads(capabilities_json)
    if isinstance(parsed, dict):
        return parsed
    return {}


_TOKEN_SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{20,128}$")


def _is_valid_token_secret(token_secret: str) -> bool:
    if not token_secret:
        return False
    return _TOKEN_SECRET_PATTERN.fullmatch(token_secret) is not None


class SQLiteTokensRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def create(
        self,
        *,
        label: str,
        project_ids: list[str],
        capabilities: dict[str, Any] | None = None,
    ) -> CreatedToken:
        if not project_ids:
            raise InvalidTokenProjectBindingError("token must bind to at least one project")

        unique_project_ids = sorted(set(project_ids))
        existing = self._conn.execute(
            f"SELECT id FROM projects WHERE id IN ({','.join(['?'] * len(unique_project_ids))})",
            tuple(unique_project_ids),
        ).fetchall()
        existing_ids = {str(row[0]) for row in existing}
        if len(existing_ids) != len(unique_project_ids):
            missing = sorted(set(unique_project_ids) - existing_ids)
            raise InvalidTokenProjectBindingError(
                f"token binding contains unknown project ids: {', '.join(missing)}"
            )

        token_id = str(uuid.uuid4())
        token_secret = secrets.token_urlsafe(32)
        token_hash = _hash_token(token_secret)
        capabilities_json = _encode_capabilities(capabilities)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                INSERT INTO mcp_tokens (
                    id,
                    label,
                    token_hash,
                    capabilities_json
                ) VALUES (?, ?, ?, ?)
                """,
                (token_id, label, token_hash, capabilities_json),
            )
            self._conn.executemany(
                """
                INSERT INTO project_token_bindings (
                    token_id,
                    project_id
                ) VALUES (?, ?)
                """,
                [(token_id, project_id) for project_id in unique_project_ids],
            )
            row = self._conn.execute(
                """
                SELECT id, label, capabilities_json, created_at, last_used_at, revoked_at
                FROM mcp_tokens
                WHERE id = ?
                """,
                (token_id,),
            ).fetchone()
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            if "mcp_tokens.label" in str(exc):
                raise DuplicateTokenLabelError("token label already exists") from exc
            raise
        except Exception:
            self._conn.rollback()
            raise

        if row is None:
            raise TokenRepositoryError(
                f"token create persisted but row reload failed for id={token_id}"
            )

        return CreatedToken(
            id=str(row[0]),
            label=str(row[1]),
            token_secret=token_secret,
            capabilities=_decode_capabilities(str(row[2]) if row[2] is not None else None),
            project_ids=unique_project_ids,
            created_at=str(row[3]),
            last_used_at=str(row[4]) if row[4] is not None else None,
            revoked_at=str(row[5]) if row[5] is not None else None,
        )

    def list_tokens(self) -> list[TokenRecord]:
        rows = self._conn.execute(
            """
            SELECT id, label, capabilities_json, created_at, last_used_at, revoked_at
            FROM mcp_tokens
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def resolve(self, token_secret: str) -> TokenRecord:
        if not _is_valid_token_secret(token_secret):
            raise UnknownTokenError("invalid token format")

        token_hash = _hash_token(token_secret)
        rows = self._conn.execute(
            """
            SELECT id, label, capabilities_json, created_at, last_used_at, revoked_at
            FROM mcp_tokens
            WHERE token_hash = ?
            LIMIT 2
            """,
            (token_hash,),
        ).fetchall()
        if not rows:
            raise UnknownTokenError("token not found")
        if len(rows) > 1:
            raise TokenIntegrityError("token integrity violation")

        row = rows[0]
        record = self._row_to_record(row)
        if record.revoked_at is not None:
            raise RevokedTokenError("token is revoked")
        return record

    def mark_last_used(self, token_id: str) -> None:
        cursor = self._conn.execute(
            "UPDATE mcp_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (token_id,),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise UnknownTokenError(f"token not found: {token_id}")

    def revoke(self, token_id: str) -> None:
        cursor = self._conn.execute(
            "UPDATE mcp_tokens SET revoked_at = CURRENT_TIMESTAMP WHERE id = ?",
            (token_id,),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise UnknownTokenError(f"token not found: {token_id}")

    def regenerate(self, token_id: str) -> CreatedToken:
        token_secret = secrets.token_urlsafe(32)
        token_hash = _hash_token(token_secret)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                "UPDATE mcp_tokens SET token_hash = ?, revoked_at = NULL WHERE id = ?",
                (token_hash, token_id),
            )
            if cursor.rowcount == 0:
                self._conn.rollback()
                raise UnknownTokenError(f"token not found: {token_id}")
            row = self._conn.execute(
                """
                SELECT id, label, capabilities_json, created_at, last_used_at, revoked_at
                FROM mcp_tokens
                WHERE id = ?
                """,
                (token_id,),
            ).fetchone()
            self._conn.commit()
        except UnknownTokenError:
            raise
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            raise TokenIntegrityError("token integrity violation") from exc
        except Exception:
            self._conn.rollback()
            raise

        if row is None:
            raise TokenRepositoryError(
                f"token regenerate persisted but row reload failed for id={token_id}"
            )

        bindings = self._conn.execute(
            """
            SELECT project_id
            FROM project_token_bindings
            WHERE token_id = ?
            ORDER BY project_id ASC
            """,
            (token_id,),
        ).fetchall()

        return CreatedToken(
            id=str(row[0]),
            label=str(row[1]),
            token_secret=token_secret,
            capabilities=_decode_capabilities(str(row[2]) if row[2] is not None else None),
            project_ids=[str(binding[0]) for binding in bindings],
            created_at=str(row[3]),
            last_used_at=str(row[4]) if row[4] is not None else None,
            revoked_at=str(row[5]) if row[5] is not None else None,
        )

    def _row_to_record(self, row: sqlite3.Row | tuple[object, ...]) -> TokenRecord:
        token_id = str(row[0])
        bindings = self._conn.execute(
            """
            SELECT project_id
            FROM project_token_bindings
            WHERE token_id = ?
            ORDER BY project_id ASC
            """,
            (token_id,),
        ).fetchall()
        return TokenRecord(
            id=token_id,
            label=str(row[1]),
            capabilities=_decode_capabilities(str(row[2]) if row[2] is not None else None),
            project_ids=[str(binding[0]) for binding in bindings],
            created_at=str(row[3]),
            last_used_at=str(row[4]) if row[4] is not None else None,
            revoked_at=str(row[5]) if row[5] is not None else None,
        )
