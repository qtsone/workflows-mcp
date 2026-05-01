from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import UTC, datetime

CSRF_HEADER_NAME = "X-CSRF-Token"


def _hash_secret(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _utc_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(tz=UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _serialize_ts(value: datetime) -> str:
    return _utc_now(value).isoformat()


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def rotate_csrf_token(conn: sqlite3.Connection, session_id: str) -> str | None:
    session_hash = _hash_secret(session_id)
    now_utc = _utc_now()

    csrf_token = _new_token()
    cursor = conn.execute(
        """
        UPDATE admin_sessions
        SET csrf_token_hash = ?
        WHERE session_hash = ?
          AND revoked_at IS NULL
          AND idle_expires_at > ?
          AND absolute_expires_at > ?
        """,
        (
            _hash_secret(csrf_token),
            session_hash,
            _serialize_ts(now_utc),
            _serialize_ts(now_utc),
        ),
    )
    if cursor.rowcount != 1:
        conn.rollback()
        return None
    conn.commit()
    return csrf_token


def verify_session_csrf(
    conn: sqlite3.Connection,
    session_id: str,
    csrf_token: str,
    *,
    now: datetime | None = None,
) -> bool:
    if not session_id or not csrf_token:
        return False

    session_hash = _hash_secret(session_id)
    row = conn.execute(
        """
        SELECT csrf_token_hash, revoked_at, idle_expires_at, absolute_expires_at
        FROM admin_sessions
        WHERE session_hash = ?
        """,
        (session_hash,),
    ).fetchone()
    if row is None or row["revoked_at"] is not None:
        return False

    now_utc = _utc_now(now)
    idle_expires_at = _parse_ts(str(row["idle_expires_at"]))
    absolute_expires_at = _parse_ts(str(row["absolute_expires_at"]))
    if now_utc >= idle_expires_at or now_utc >= absolute_expires_at:
        return False

    expected_hash = str(row["csrf_token_hash"])
    actual_hash = _hash_secret(csrf_token)
    return hmac.compare_digest(expected_hash, actual_hash)
