from __future__ import annotations

import importlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.security.passwords import hash_password, verify_password
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase2-admin-password"


def _bootstrap_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )
    return base_dir


def _client_from_base_dir(base_dir: Path, *, base_url: str = "http://testserver") -> TestClient:
    return TestClient(
        build_app(base_dir=base_dir),
        base_url=base_url,
        raise_server_exceptions=False,
    )


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    return _client_from_base_dir(base_dir)


@pytest.fixture()
def https_app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    return _client_from_base_dir(base_dir, base_url="https://testserver")


def _login(client: TestClient, password: str) -> httpx.Response:
    response = client.post("/api/admin/v1/auth/login", json={"password": password})
    assert response.status_code == 200
    return response


def _login_and_csrf(client: TestClient) -> tuple[httpx.Response, str]:
    login_response = _login(client, _ADMIN_PASSWORD)
    csrf_token = login_response.headers.get("X-CSRF-Token") or login_response.json().get(
        "csrf_token"
    )
    assert csrf_token
    return login_response, csrf_token


def test_login_success_sets_scoped_http_only_session_cookies_and_csrf_token(
    https_app_client: TestClient,
) -> None:
    response = _login(https_app_client, _ADMIN_PASSWORD)

    payload = response.json()
    assert payload["authenticated"] is True
    assert payload.get("session") is not None

    set_cookie_values = response.headers.get_list("set-cookie")
    assert len(set_cookie_values) >= 2

    admin_cookie = next(cookie for cookie in set_cookie_values if "Path=/api/admin/v1" in cookie)
    events_cookie = next(cookie for cookie in set_cookie_values if "Path=/api/events/v1" in cookie)

    for cookie in (admin_cookie, events_cookie):
        assert "HttpOnly" in cookie
        assert "Secure" in cookie
        assert "SameSite=Lax" in cookie or "SameSite=Strict" in cookie
        assert "Max-Age=" in cookie
        assert "Expires=" in cookie

    csrf_header = response.headers.get("X-CSRF-Token")
    csrf_body = payload.get("csrf_token")
    csrf_token = csrf_header or csrf_body
    assert csrf_token
    assert not any("csrf" in cookie.lower() for cookie in set_cookie_values)


def test_login_wrong_password_rejected_and_no_session_cookie_created(
    app_client: TestClient,
) -> None:
    response = app_client.post("/api/admin/v1/auth/login", json={"password": "wrong-password"})
    assert response.status_code in {401, 403}

    set_cookie_values = response.headers.get_list("set-cookie")
    assert set_cookie_values == []


def test_login_verifies_persisted_password_hash_not_plaintext_or_bypass(
    app_client: TestClient,
) -> None:
    derived_hash = hash_password(_ADMIN_PASSWORD)
    assert derived_hash
    assert derived_hash != _ADMIN_PASSWORD
    assert verify_password(_ADMIN_PASSWORD, derived_hash) is True
    assert verify_password("wrong-password", derived_hash) is False

    correct_password = app_client.post(
        "/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD}
    )
    assert correct_password.status_code == 200

    wrong_password = app_client.post(
        "/api/admin/v1/auth/login", json={"password": "wrong-password"}
    )
    assert wrong_password.status_code in {401, 403}


def test_login_route_depends_on_password_verifier_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []

    def _deny_and_record(password: str, password_hash: str) -> bool:
        calls.append((password, password_hash))
        return False

    passwords_module = importlib.import_module("workflows_mcp.security.passwords")
    monkeypatch.setattr(passwords_module, "verify_password", _deny_and_record, raising=False)

    for module_name in (
        "workflows_mcp.http.routes.admin_v1",
        "workflows_mcp.http.routes.admin_v1.auth",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        monkeypatch.setattr(module, "verify_password", _deny_and_record, raising=False)

    client = _client_from_base_dir(base_dir)
    response = client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})

    assert response.status_code in {401, 403}
    assert calls, "Expected login to invoke verify_password seam"
    submitted_password, persisted_hash = calls[0]
    assert submitted_password == _ADMIN_PASSWORD
    assert persisted_hash
    assert persisted_hash != _ADMIN_PASSWORD


def test_session_endpoint_requires_ui_session_and_rejects_mcp_bearer_token(
    app_client: TestClient,
) -> None:
    unauthenticated = app_client.get("/api/admin/v1/auth/session")
    assert unauthenticated.status_code == 401

    _login(app_client, _ADMIN_PASSWORD)
    session_response = app_client.get("/api/admin/v1/auth/session")
    assert session_response.status_code == 200
    payload = session_response.json()
    assert payload["authenticated"] is True
    assert payload.get("session") is not None
    assert payload["session"].get("idle_expires_at")
    assert payload["session"].get("absolute_expires_at")

    bearer_response = app_client.get(
        "/api/admin/v1/auth/session",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert bearer_response.status_code in {401, 403}


def test_logout_requires_csrf_token_header_when_session_cookie_present(
    app_client: TestClient,
) -> None:
    _, csrf_token = _login_and_csrf(app_client)

    cookie_only_logout = app_client.post("/api/admin/v1/auth/logout")
    assert cookie_only_logout.status_code == 403

    csrf_logout = app_client.post(
        "/api/admin/v1/auth/logout",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert csrf_logout.status_code == 200


def test_mcp_bearer_token_cannot_access_mutating_admin_endpoint(
    app_client: TestClient,
) -> None:
    response = app_client.post(
        "/api/admin/v1/auth/logout",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert response.status_code in {401, 403}


def test_csrf_not_cookie_bound_and_mutating_admin_endpoint_requires_csrf_header(
    app_client: TestClient,
) -> None:
    login_response, _ = _login_and_csrf(app_client)
    set_cookie_values = login_response.headers.get_list("set-cookie")
    assert not any("csrf" in cookie.lower() for cookie in set_cookie_values)

    missing_csrf = app_client.post("/api/admin/v1/auth/logout")
    assert missing_csrf.status_code == 403


def test_mutating_admin_request_rejects_invalid_or_mismatched_csrf_token(
    app_client: TestClient,
) -> None:
    _, csrf_token = _login_and_csrf(app_client)

    invalid_csrf = app_client.post(
        "/api/admin/v1/auth/logout",
        headers={"X-CSRF-Token": "invalid-csrf-token"},
    )
    assert invalid_csrf.status_code == 403


def test_http_login_does_not_require_secure_cookie_attribute(
    app_client: TestClient,
) -> None:
    response = _login(app_client, _ADMIN_PASSWORD)
    set_cookie_values = response.headers.get_list("set-cookie")
    assert len(set_cookie_values) >= 2
    assert all("HttpOnly" in cookie for cookie in set_cookie_values)
    assert all("Secure" not in cookie for cookie in set_cookie_values)


def test_forwarded_https_login_sets_secure_cookie_attributes(
    app_client: TestClient,
) -> None:
    response = app_client.post(
        "/api/admin/v1/auth/login",
        json={"password": _ADMIN_PASSWORD},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert response.status_code == 200

    set_cookie_values = response.headers.get_list("set-cookie")
    assert len(set_cookie_values) >= 2
    assert all("Secure" in cookie for cookie in set_cookie_values)


def test_forwarded_https_logout_clears_secure_cookies_and_revokes_session(
    app_client: TestClient,
) -> None:
    login_response = app_client.post(
        "/api/admin/v1/auth/login",
        json={"password": _ADMIN_PASSWORD},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert login_response.status_code == 200
    csrf_token = login_response.headers.get("X-CSRF-Token") or login_response.json().get(
        "csrf_token"
    )
    assert csrf_token

    session_cookie = next(
        cookie
        for cookie in login_response.headers.get_list("set-cookie")
        if "Path=/api/admin/v1" in cookie
    )
    session_cookie_pair = session_cookie.split(";", 1)[0]

    logout_response = app_client.post(
        "/api/admin/v1/auth/logout",
        headers={
            "X-CSRF-Token": csrf_token,
            "X-Forwarded-Proto": "https",
            "Cookie": session_cookie_pair,
        },
    )
    assert logout_response.status_code == 200

    set_cookie_values = logout_response.headers.get_list("set-cookie")
    assert len(set_cookie_values) >= 2
    admin_clear_cookie = next(
        cookie for cookie in set_cookie_values if "Path=/api/admin/v1" in cookie
    )
    events_clear_cookie = next(
        cookie for cookie in set_cookie_values if "Path=/api/events/v1" in cookie
    )
    assert "Secure" in admin_clear_cookie
    assert "Secure" in events_clear_cookie

    session_after_logout = app_client.get(
        "/api/admin/v1/auth/session",
        headers={"X-Forwarded-Proto": "https"},
    )
    assert session_after_logout.status_code == 401


def test_login_rate_limit_blocks_repeated_failed_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKFLOWS_RATE_LIMIT", "100")
    monkeypatch.setenv("WORKFLOWS_MCP_RATE_LIMIT", "100")
    monkeypatch.setenv("WORKFLOWS_LOGIN_RATE_LIMIT", "3")
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    client = _client_from_base_dir(base_dir)

    statuses: list[int] = []
    for _ in range(6):
        response = client.post("/api/admin/v1/auth/login", json={"password": "wrong-password"})
        statuses.append(response.status_code)

    assert 429 in statuses


def test_login_and_session_payload_encode_idle_and_absolute_expiry_metadata(
    app_client: TestClient,
) -> None:
    login_response = _login(app_client, _ADMIN_PASSWORD)
    login_payload = login_response.json()
    assert login_payload["authenticated"] is True
    assert login_payload.get("session") is not None
    assert login_payload["session"].get("idle_expires_at")
    assert login_payload["session"].get("absolute_expires_at")

    session_response = app_client.get("/api/admin/v1/auth/session")
    assert session_response.status_code == 200
    session_payload = session_response.json()
    assert session_payload["authenticated"] is True
    assert session_payload.get("session") is not None
    assert session_payload["session"].get("idle_expires_at")
    assert session_payload["session"].get("absolute_expires_at")


def _metadata_conn(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "metadata.db"
    conn = connect_metadata_db(db_path)
    migrate_metadata_db(conn)
    return conn


def test_session_store_hashes_only_and_public_metadata_excludes_sensitive_fields(
    tmp_path: Path,
) -> None:
    from workflows_mcp.security.sessions import create_admin_session

    conn = _metadata_conn(tmp_path)
    try:
        created = create_admin_session(conn)
        row = conn.execute("SELECT * FROM admin_sessions").fetchone()
        assert row is not None
        assert row["session_hash"]
        assert row["session_hash"] != created.session_id
        assert row["csrf_token_hash"]
        assert row["csrf_token_hash"] != created.csrf_token

        public = created.session.to_public_dict()
        assert set(public.keys()) == {"idle_expires_at", "absolute_expires_at"}
        assert "session_id" not in public
        assert "session_hash" not in public
        assert "csrf_token_hash" not in public
    finally:
        conn.close()


def test_session_validation_enforces_idle_and_absolute_and_refreshes_idle_only(
    tmp_path: Path,
) -> None:
    from workflows_mcp.security.sessions import (
        SESSION_IDLE_TIMEOUT_SECONDS,
        create_admin_session,
        get_admin_session,
    )

    conn = _metadata_conn(tmp_path)
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    try:
        created = create_admin_session(conn, now=now)
        original_absolute = created.session.absolute_expires_at
        original_idle = created.session.idle_expires_at

        accessed_at = now + timedelta(seconds=30)
        refreshed = get_admin_session(conn, created.session_id, now=accessed_at, refresh_idle=True)
        assert refreshed is not None
        assert refreshed.absolute_expires_at == original_absolute
        assert refreshed.idle_expires_at > original_idle
        assert refreshed.idle_expires_at == accessed_at + timedelta(
            seconds=SESSION_IDLE_TIMEOUT_SECONDS
        )

        no_refresh = get_admin_session(
            conn,
            created.session_id,
            now=accessed_at + timedelta(seconds=10),
            refresh_idle=False,
        )
        assert no_refresh is not None
        assert no_refresh.idle_expires_at == refreshed.idle_expires_at

        idle_expired = get_admin_session(
            conn,
            created.session_id,
            now=refreshed.idle_expires_at + timedelta(seconds=1),
        )
        assert idle_expired is None
    finally:
        conn.close()


def test_session_revocation_causes_validation_failure(tmp_path: Path) -> None:
    from workflows_mcp.security.sessions import (
        create_admin_session,
        get_admin_session,
        revoke_admin_session,
    )

    conn = _metadata_conn(tmp_path)
    try:
        created = create_admin_session(conn)
        assert get_admin_session(conn, created.session_id) is not None
        revoke_admin_session(conn, created.session_id)
        assert get_admin_session(conn, created.session_id) is None
    finally:
        conn.close()


def test_csrf_verification_rejects_invalid_or_mismatched_tokens_and_rotation_invalidates_old(
    tmp_path: Path,
) -> None:
    from workflows_mcp.security.csrf import rotate_csrf_token, verify_session_csrf
    from workflows_mcp.security.sessions import create_admin_session

    conn = _metadata_conn(tmp_path)
    try:
        created = create_admin_session(conn)
        assert verify_session_csrf(conn, created.session_id, created.csrf_token) is True
        assert verify_session_csrf(conn, created.session_id, "invalid-token") is False

        other = create_admin_session(conn)
        assert verify_session_csrf(conn, created.session_id, other.csrf_token) is False

        rotated = rotate_csrf_token(conn, created.session_id)
        assert rotated is not None
        assert rotated != created.csrf_token
        assert verify_session_csrf(conn, created.session_id, created.csrf_token) is False
        assert verify_session_csrf(conn, created.session_id, rotated) is True
    finally:
        conn.close()


def test_admin_sessions_schema_has_phase2_columns(tmp_path: Path) -> None:
    conn = _metadata_conn(tmp_path)
    try:
        rows = conn.execute("PRAGMA table_info(admin_sessions)").fetchall()
        columns = {str(row[1]) for row in rows}
        assert {
            "session_hash",
            "csrf_token_hash",
            "created_at",
            "last_seen_at",
            "idle_expires_at",
            "absolute_expires_at",
            "revoked_at",
        }.issubset(columns)
    finally:
        conn.close()


def test_verify_session_csrf_rejects_idle_or_absolute_expired_session(tmp_path: Path) -> None:
    from workflows_mcp.security.csrf import verify_session_csrf
    from workflows_mcp.security.sessions import create_admin_session

    conn = _metadata_conn(tmp_path)
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    try:
        created = create_admin_session(conn, now=now)

        assert (
            verify_session_csrf(
                conn,
                created.session_id,
                created.csrf_token,
                now=created.session.idle_expires_at + timedelta(seconds=1),
            )
            is False
        )

        created2 = create_admin_session(conn, now=now)
        assert (
            verify_session_csrf(
                conn,
                created2.session_id,
                created2.csrf_token,
                now=created2.session.absolute_expires_at + timedelta(seconds=1),
            )
            is False
        )
    finally:
        conn.close()


def test_rotate_csrf_token_rejects_revoked_and_expired_sessions(tmp_path: Path) -> None:
    from workflows_mcp.security.csrf import rotate_csrf_token
    from workflows_mcp.security.sessions import (
        create_admin_session,
        revoke_admin_session,
    )

    conn = _metadata_conn(tmp_path)
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    try:
        revoked = create_admin_session(conn, now=now)
        revoke_admin_session(conn, revoked.session_id)
        assert rotate_csrf_token(conn, revoked.session_id) is None

        idle_expired = create_admin_session(conn, now=now)
        conn.execute(
            "UPDATE admin_sessions SET idle_expires_at = ? WHERE session_hash = ?",
            (
                (now - timedelta(seconds=1)).isoformat(),
                __import__("hashlib").sha256(idle_expired.session_id.encode("utf-8")).hexdigest(),
            ),
        )
        conn.commit()
        assert rotate_csrf_token(conn, idle_expired.session_id) is None

        absolute_expired = create_admin_session(conn, now=now)
        conn.execute(
            "UPDATE admin_sessions SET absolute_expires_at = ? WHERE session_hash = ?",
            (
                (now - timedelta(seconds=1)).isoformat(),
                __import__("hashlib").sha256(
                    absolute_expired.session_id.encode("utf-8")
                ).hexdigest(),
            ),
        )
        conn.commit()
        assert rotate_csrf_token(conn, absolute_expired.session_id) is None
    finally:
        conn.close()


def test_naive_db_timestamps_are_treated_as_utc_for_session_and_csrf(tmp_path: Path) -> None:
    from workflows_mcp.security.csrf import verify_session_csrf
    from workflows_mcp.security.sessions import create_admin_session, get_admin_session

    conn = _metadata_conn(tmp_path)
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    try:
        created = create_admin_session(conn, now=now)
        session_hash = __import__("hashlib").sha256(
            created.session_id.encode("utf-8")
        ).hexdigest()
        conn.execute(
            """
            UPDATE admin_sessions
            SET created_at = ?, last_seen_at = ?, idle_expires_at = ?, absolute_expires_at = ?
            WHERE session_hash = ?
            """,
            (
                "2026-01-01T10:00:00",
                "2026-01-01T10:05:00",
                "2026-01-01T10:30:00",
                "2026-01-01T22:00:00",
                session_hash,
            ),
        )
        conn.commit()

        session = get_admin_session(
            conn,
            created.session_id,
            now=datetime(2026, 1, 1, 10, 10, tzinfo=UTC),
            refresh_idle=False,
        )
        assert session is not None
        assert session.created_at.tzinfo is not None
        assert (
            verify_session_csrf(
                conn,
                created.session_id,
                created.csrf_token,
                now=datetime(2026, 1, 1, 10, 15, tzinfo=UTC),
            )
            is True
        )
    finally:
        conn.close()
