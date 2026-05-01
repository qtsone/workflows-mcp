from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

SESSION_COOKIE_NAME = "workflows_admin_session"
SESSION_IDLE_TIMEOUT_SECONDS = 30 * 60
SESSION_ABSOLUTE_TIMEOUT_SECONDS = 12 * 60 * 60


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


def _hash_secret(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _new_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class AdminSession:
    created_at: datetime
    last_seen_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime

    def to_public_dict(self) -> dict[str, str]:
        return {
            "idle_expires_at": _serialize_ts(self.idle_expires_at),
            "absolute_expires_at": _serialize_ts(self.absolute_expires_at),
        }


@dataclass(frozen=True)
class CreatedAdminSession:
    session_id: str
    csrf_token: str
    session: AdminSession


def create_admin_session(
    conn: sqlite3.Connection, *, now: datetime | None = None
) -> CreatedAdminSession:
    now_utc = _utc_now(now)
    session_id = _new_token()
    csrf_token = _new_token()

    session_hash = _hash_secret(session_id)
    csrf_token_hash = _hash_secret(csrf_token)
    idle_expires_at = now_utc + timedelta(seconds=SESSION_IDLE_TIMEOUT_SECONDS)
    absolute_expires_at = now_utc + timedelta(seconds=SESSION_ABSOLUTE_TIMEOUT_SECONDS)

    conn.execute(
        """
        INSERT INTO admin_sessions (
            session_hash,
            csrf_token_hash,
            created_at,
            last_seen_at,
            idle_expires_at,
            absolute_expires_at,
            revoked_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            session_hash,
            csrf_token_hash,
            _serialize_ts(now_utc),
            _serialize_ts(now_utc),
            _serialize_ts(idle_expires_at),
            _serialize_ts(absolute_expires_at),
        ),
    )
    conn.commit()

    return CreatedAdminSession(
        session_id=session_id,
        csrf_token=csrf_token,
        session=AdminSession(
            created_at=now_utc,
            last_seen_at=now_utc,
            idle_expires_at=idle_expires_at,
            absolute_expires_at=absolute_expires_at,
        ),
    )


def get_admin_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    now: datetime | None = None,
    refresh_idle: bool = True,
) -> AdminSession | None:
    session_hash = _hash_secret(session_id)
    row = conn.execute(
        """
        SELECT created_at, last_seen_at, idle_expires_at, absolute_expires_at, revoked_at
        FROM admin_sessions
        WHERE session_hash = ?
        """,
        (session_hash,),
    ).fetchone()
    if row is None:
        return None

    if row["revoked_at"] is not None:
        return None

    now_utc = _utc_now(now)
    idle_expires_at = _parse_ts(str(row["idle_expires_at"]))
    absolute_expires_at = _parse_ts(str(row["absolute_expires_at"]))
    if now_utc >= idle_expires_at or now_utc >= absolute_expires_at:
        return None

    last_seen_at = _parse_ts(str(row["last_seen_at"]))
    if refresh_idle:
        refreshed_idle_expires_at = now_utc + timedelta(seconds=SESSION_IDLE_TIMEOUT_SECONDS)
        cursor = conn.execute(
            """
            UPDATE admin_sessions
            SET last_seen_at = ?, idle_expires_at = ?
            WHERE session_hash = ?
              AND revoked_at IS NULL
              AND idle_expires_at > ?
              AND absolute_expires_at > ?
            """,
            (
                _serialize_ts(now_utc),
                _serialize_ts(refreshed_idle_expires_at),
                session_hash,
                _serialize_ts(now_utc),
                _serialize_ts(now_utc),
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return None
        conn.commit()
        last_seen_at = now_utc
        idle_expires_at = refreshed_idle_expires_at

    return AdminSession(
        created_at=_parse_ts(str(row["created_at"])),
        last_seen_at=last_seen_at,
        idle_expires_at=idle_expires_at,
        absolute_expires_at=absolute_expires_at,
    )


def revoke_admin_session(conn: sqlite3.Connection, session_id: str) -> None:
    session_hash = _hash_secret(session_id)
    conn.execute(
        "UPDATE admin_sessions SET revoked_at = ? WHERE session_hash = ?",
        (_serialize_ts(_utc_now()), session_hash),
    )
    conn.commit()
